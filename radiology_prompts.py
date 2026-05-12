"""
Prompt templates for radiology knowledge graph construction.

This file contains two prompt templates:
1. Stage 1: Tag Assignment Prompt
2. Stage 2: Domain Organization Prompt

Usage:
    from radiology_prompts import build_tag_assignment_prompt, build_domain_organization_prompt

    prompt_1 = build_tag_assignment_prompt(report_text, triple_list)
    prompt_2 = build_domain_organization_prompt(node_list)
"""


TAG_ASSIGNMENT_PROMPT_TEMPLATE = """Stage 1: Tag Assignment Prompt

You are a clinical radiology expert specializing in chest X-ray interpretation.

Your task is to assign each input medical triple to the most appropriate semantic tag based on clinical radiological reasoning.

Definitions:
- A "triple" consists of (entity, relation, attribute), extracted from a radiology report.
- A "tag" is a standardized semantic category representing a clinically meaningful finding (e.g., Pleural Effusion, Cardiomegaly, Lung Opacity).

Instructions:
1. Carefully read each triple and determine its clinical meaning.
2. Assign each triple to:
   - ONE best matching tag (preferred), OR
   - MULTIPLE tags if necessary, OR
   - NONE if the triple is not clinically meaningful.
3. Use clinically consistent and normalized terminology.
4. Avoid overly specific or redundant labels.

Output format (STRICT JSON):
{{
  "Tag1": ["Triple1", "Triple2"],
  "Tag2": ["Triple3"],
  "None": ["Triple4"]
}}

Input:
Report: {report_text}

Triples:
{triple_list}
"""


DOMAIN_ORGANIZATION_PROMPT_TEMPLATE = """Stage 2: Domain Organization Prompt

You are a clinical radiology expert with knowledge of anatomical structures.

Your task is to organize medical entities into anatomically meaningful domains.

Definitions:
- A "node" represents a clinical entity (e.g., lung opacity, pleural effusion).
- A "domain" represents an anatomical or physiological category (e.g., Lung, Pleura, Cardiac).

Instructions:
1. Group nodes based on anatomical location or clinical relevance.
2. Use standard anatomical categories.
3. Each node must belong to ONE domain only.
4. Ensure consistency with clinical reasoning.

Common domains include (but are not limited to):
- Lung
- Pleura
- Cardiac
- Mediastinum
- Bone
- Diaphragm

Output format (STRICT JSON):
{{
  "Lung": ["Node1", "Node2"],
  "Pleura": ["Node3"],
  "Cardiac": ["Node4"]
}}

Input:
Nodes:
{node_list}
"""


def _format_list(items):
    """
    Convert a list/tuple into a newline-separated string.
    If the input is already a string, return it unchanged.
    """
    if isinstance(items, str):
        return items

    if isinstance(items, (list, tuple)):
        return "\n".join(f"- {item}" for item in items)

    raise TypeError("Input must be a string, list, or tuple.")


def build_tag_assignment_prompt(report_text, triple_list):
    """
    Build the Stage 1 tag assignment prompt.

    Args:
        report_text (str): The original radiology report text.
        triple_list (str | list | tuple): Extracted medical triples.

    Returns:
        str: A formatted prompt for tag assignment.
    """
    return TAG_ASSIGNMENT_PROMPT_TEMPLATE.format(
        report_text=report_text,
        triple_list=_format_list(triple_list),
    )


def build_domain_organization_prompt(node_list):
    """
    Build the Stage 2 domain organization prompt.

    Args:
        node_list (str | list | tuple): Medical entity nodes.

    Returns:
        str: A formatted prompt for anatomical domain organization.
    """
    return DOMAIN_ORGANIZATION_PROMPT_TEMPLATE.format(
        node_list=_format_list(node_list),
    )


if __name__ == "__main__":
    example_report = "There is mild cardiomegaly with small bilateral pleural effusions."
    example_triples = [
        "(heart, has_attribute, mild enlargement)",
        "(pleura, has_finding, bilateral effusion)",
    ]
    example_nodes = [
        "cardiomegaly",
        "pleural effusion",
        "lung opacity",
    ]

    print("===== Stage 1 Prompt =====")
    print(build_tag_assignment_prompt(example_report, example_triples))

    print("\n===== Stage 2 Prompt =====")
    print(build_domain_organization_prompt(example_nodes))
