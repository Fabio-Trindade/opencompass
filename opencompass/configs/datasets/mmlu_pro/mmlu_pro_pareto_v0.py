from mmengine.config import read_base
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import MMLUProDataset
from opencompass.utils.text_postprocessors import match_answer_pattern

with read_base():
    from .mmlu_pro_categories import categories



QUERY_TEMPLATE = """
Question:
{question}

Options:
{options_str}

Choose the option that correctly answers the question, following the instructions:
  1) Return only the final answer
  2) The answer must be in the exact format: 'ANSWER: $LETTER' (without quotes)
  3) Do not include explanations
  4) Do not include reasoning steps
  5) Do not include any additional text
  6) Output must be exactly one line in the required format
"""
mmlu_pro_datasets = []

for category in categories:
    mmlu_pro_reader_cfg = dict(
        input_columns=['question', 'cot_content', 'options_str'],
        output_column='answer',
        train_split='validation',
        test_split='test',
    )
    mmlu_pro_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(role='HUMAN',
                         prompt=QUERY_TEMPLATE),
                ],
            ),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )

    mmlu_pro_eval_cfg = dict(
        evaluator=dict(type=AccEvaluator),
        pred_postprocessor=dict(
            type=match_answer_pattern,
            answer_pattern=r'(?i)ANSWER\s*:\s*([A-P])')
    )

    mmlu_pro_datasets.append(
        dict(
            abbr=f'mmlu_pro_{category.replace(" ", "_")}',
            type=MMLUProDataset,
            path='opencompass/mmlu_pro',
            category=category,
            reader_cfg=mmlu_pro_reader_cfg,
            infer_cfg=mmlu_pro_infer_cfg,
            eval_cfg=mmlu_pro_eval_cfg,
        ))
