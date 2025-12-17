"""
Finetune trainer for report generation
"""
import json

import pandas as pd

from basetrainer import BaseTrainer
import torch
import time
from modules.knowledge_builder import KnowledgeBuilder
from trainer.CECalculator import CECalculator
from trainer.count import EntityCounter, AttCollector
import os

from utils.loss import calculate_multi_label_loss

class KMTrainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader,tokenizer):
        super(KMTrainer, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.best_score = 0.

        print("length of train_dataloader:",len(self.train_dataloader),len(self.train_dataloader.dataset))
        print("length of val_dataloader:", len(self.val_dataloader),len(self.val_dataloader.dataset))
        print("length of test_dataloader:", len(self.test_dataloader),len(self.test_dataloader.dataset))

        self.att_flag = False        #是否记录注意力矩阵
        self.entity_flag = False
        self.gt_flag = False
        self.ce_flag = False

        self.dataset_name = args["dataset_name"]


        self.tokenizer = tokenizer

        self.knowledge = KnowledgeBuilder(tokenizer,args["dataset_name"])
        res = self.knowledge.knowledge_init()
        self.model.gra_encoder.get_kg_dict(res,self.knowledge.topic)


        self.label_dir = args["label_dir"]
        self.ann_path = args["ann_path"]
        self.label = [
            "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
            "Pneumonia",
            "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
            "No Finding"
        ]
        train_df = self.create_gt(os.path.join(self.label_dir, "train_labeled_reports.csv"), label_type="train")
        val_df = self.create_gt(os.path.join(self.label_dir, "val_labeled_reports.csv"), label_type="val")
        test_df = self.create_gt(os.path.join(self.label_dir, "test_labeled_reports.csv"), label_type="test")

        self.combined_df = pd.concat([train_df, val_df, test_df], axis=0)


    def _train_epoch(self, epoch):

        train_loss = 0
        self.model.train()
        start_time = time.time()

        # self.model.text_gra_embed.radgraph_model.eval()

        for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.train_dataloader):
            images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()

            rad_list = None
            sims = self.get_topic_index(images_id,"train")

            output, vg_effect, to_effect,o_output = self.model(images, sims, reports_ids, mode='train',rad_list=rad_list)
            nll_loss,lm_loss,v_cf_loss,g_cf_loss = self.criterion(output, reports_ids, reports_masks,vg_effect, to_effect)
            cl_loss = calculate_multi_label_loss(o_output[0], self.combined_df, images_id)
            loss = nll_loss + 0.1 * cl_loss
            self.optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()

            print(
                f"\repoch: {epoch} {batch_idx}/{len(self.train_dataloader)}\tloss: {loss:.3f}\tmean loss: {train_loss / (batch_idx + 1):.3f}\tlm_loss:{lm_loss:.3f}\to_loss:{cl_loss:.3f}",
                flush=True, end='')

        if self.args["lr_scheduler"] == 'StepLR':
            self.lr_scheduler.step()

        log = {'train_loss': train_loss / len(self.train_dataloader)}
        print("\n")
        print("\tEpoch {}\tmean_loss: {:.4f}\ttime: {:.4f}s".format(epoch, log['train_loss'], time.time() - start_time))


        self.model.eval()
        start_time = time.time()
        with torch.no_grad():
            val_gts, val_res = [], []
            p = torch.zeros([1, self.args["max_seq_length"]]).cuda()
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.val_dataloader):
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
                output, _ = self.model(images, None, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)
                p = torch.cat([p, output])

                print(f"\rVal Processing: [{int((batch_idx + 1) / len(self.val_dataloader) * 100)}%]", end='',
                      flush=True)
            print(f"\ttime:{(time.time() - start_time):.4f}s")

            if self.ce_flag:
                self.cal.calculate(val_res,val_gts,"val")
            tp, lp = count_p(p[1:])
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            # record val metrics
            for k, v in val_met.items():
                self.monitor.logkv(key='val_' + k, val=v)
            val_met['p'] = lp
            log.update(**{'val_' + k: v for k, v in val_met.items()})

        self.model.eval()
        start_time = time.time()
        with torch.no_grad():
            test_gts, test_res, p = [], [], []
            p = torch.zeros([1, self.args["max_seq_length"]]).cuda()
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()          
                output, _ = self.model(images, None,mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)
                p = torch.cat([p, output])

                print(f"\rTest Processing: [{int((batch_idx + 1) / len(self.test_dataloader) * 100)}%]", end='',
                      flush=True)
            print(f"\ttime:{(time.time() - start_time):.4f}s")
            if self.ce_flag:
                self.cal.calculate(test_res, test_gts, "test")
            tp, lp = count_p(p[1:])
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})

            for k, v in test_met.items():
                self.monitor.logkv(key='test_' + k, val=v)
            test_met['p'] = lp
            log.update(**{'test_' + k: v for k, v in test_met.items()})

        if self.args['monitor_metric_curves']:
            self.monitor.plot_current_metrics(epoch, self.monitor.name2val)
        self.monitor.dumpkv(epoch)
        return log

    def get_topic_index(self,img_id,data_type):
        res = []
        for id_ in img_id:
            row = self.combined_df[self.combined_df['id'] == id_]
            row_data = row.iloc[0]
            cols_with_ones = []
            for col_idx in range(1, len(row_data)):
                if row_data.iloc[col_idx] == 1:
                    cols_with_ones.append(col_idx - 1)  
            if len(cols_with_ones) > 4:
                cols_with_ones = cols_with_ones[:4]      
            res.append(cols_with_ones)
        return res

    def create_gt(self,label_path,label_type):
        df = pd.read_csv(label_path)
        ann_all = json.loads(open(self.ann_path, 'r').read())
        ann_t = ann_all[label_type]
        extracted_values = [item["id"] for item in ann_t]
        json_df = pd.DataFrame({
            "id": extracted_values,  # 替换为你想要的列名
        })
        df_first = df[self.label].replace(-1.0, 1.0).rename(columns=lambda x: f"{x}_T")
        df_last = df[self.label].replace(1.0, -1.0).replace(0.0, 1.0).replace(-1.0, 0.0).rename(
            columns=lambda x: f"{x}_F")
        combined_df = pd.concat([json_df, df_first, df_last], axis=1).fillna(0.0)
        return combined_df


class SampleTrainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader,tokenizer):
        super(SampleTrainer, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        print("length of train_dataloader:",len(self.train_dataloader),len(self.train_dataloader.dataset))
        print("length of val_dataloader:", len(self.val_dataloader),len(self.val_dataloader.dataset))
        print("length of test_dataloader:", len(self.test_dataloader),len(self.test_dataloader.dataset))

        self.dataset_name = args["dataset_name"]

        self.tokenizer = tokenizer

        self.cal = CECalculator(save_dir=args["record_dir"])

        self.label_dir = args["label_dir"]
        self.ann_path = args["ann_path"]
        self.label = [
            "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
            "Pneumonia",
            "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
            "No Finding"
        ]
        train_df = self.create_gt(os.path.join(self.label_dir, "train_labeled_reports.csv"), label_type="train")
        val_df = self.create_gt(os.path.join(self.label_dir, "val_labeled_reports.csv"), label_type="val")
        test_df = self.create_gt(os.path.join(self.label_dir, "test_labeled_reports.csv"), label_type="test")

        self.combined_df = pd.concat([train_df, val_df, test_df], axis=0)


    def train(self):
        state = torch.load("/root/autodl-tmp/results/test.pth", map_location='cuda')
        pretrained_dict = state['state_dict']
        self.model.load_state_dict(pretrained_dict, False)
        knowledge = KnowledgeBuilder(self.tokenizer, self.args["dataset_name"])
        res = knowledge.knowledge_init()
        self.model.gra_encoder.get_kg_dict(res, knowledge.topic)

        self._train_epoch(1)


    def _train_epoch(self, epoch):

        self.model.eval()
        start_time = time.time()
        with torch.no_grad():
            val_gts, val_res = [], []
            p = torch.zeros([1, self.args["max_seq_length"]]).cuda()
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.val_dataloader):
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
                output, _ = self.model(images, None, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)
                p = torch.cat([p, output])
                print(f"\rVal Processing: [{int((batch_idx + 1) / len(self.val_dataloader) * 100)}%]", end='',
                      flush=True)
            print(f"\ttime:{(time.time() - start_time):.4f}s")
            self.cal.calculate(val_res, val_gts, "val")
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})

        self.model.eval()
        start_time = time.time()
        with torch.no_grad():
            test_gts, test_res, p = [], [], []
            p = torch.zeros([1, self.args["max_seq_length"]]).cuda()
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
                output, _ = self.model(images, None, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)
                p = torch.cat([p, output])
                print(f"\rTest Processing: [{int((batch_idx + 1) / len(self.test_dataloader) * 100)}%]", end='',
                      flush=True)
            print(f"\ttime:{(time.time() - start_time):.4f}s")
            self.cal.calculate(test_res, test_gts, "test")
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})

    def get_topic_index(self, img_id, data_type):
        res = []
        # 遍历每个图像ID
        for id_ in img_id:
            row = self.combined_df[self.combined_df['id'] == id_]
            row_data = row.iloc[0]
            cols_with_ones = []
            for col_idx in range(1, len(row_data)):
                if row_data.iloc[col_idx] == 1:
                    cols_with_ones.append(col_idx - 1)
            if len(cols_with_ones) > 4:
                cols_with_ones = cols_with_ones[:4]

            res.append(cols_with_ones)
        return res

    def create_gt(self,label_path,label_type):
        df = pd.read_csv(label_path)

        ann_all = json.loads(open(self.ann_path, 'r').read())
        ann_t = ann_all[label_type]

        extracted_values = [item["id"] for item in ann_t]

        json_df = pd.DataFrame({
            "id": extracted_values,  # 替换为你想要的列名
        })

        df_first = df[self.label].replace(-1.0, 1.0).rename(columns=lambda x: f"{x}_T")
        df_last = df[self.label].replace(1.0, -1.0).replace(0.0, 1.0).replace(-1.0, 0.0).rename(
            columns=lambda x: f"{x}_F")

        combined_df = pd.concat([json_df, df_first, df_last], axis=1).fillna(0.0)
        return combined_df


def count_p(p):
    t = torch.unique(p, dim=0)
    l = t.size(0)
    return t, l

