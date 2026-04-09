from opencompass.datasets.supergpqa.supergpqa import (
    SuperGPQADataset,
    SuperGPQAEvaluator,
)
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

QUERY="""
{infer prompt}

Choose the option that correctly answers the question, following the instructions:
  1) Return only the final answer
  2) The answer must be exactly in the format: 'Answer: $LETTER' (without quotes)
  3) $LETTER must be one of A, B, C, D, E, F, G, H, I, or J
  4) Do not include explanations
  5) Do not include reasoning steps
  6) Do not include any additional text
  7) Output must be exactly one line

Answer:
"""
# Reader configuration
reader_cfg = dict(
    input_columns=[
        'question',
        'options',
        'discipline',
        'field',
        'subfield',
        'difficulty',
        'infer_prompt',
        'prompt_mode',
    ],
    output_column='answer_letter',
)

# Inference configuration
infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='{infer_prompt}',
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

# Evaluation configuration
eval_cfg = dict(
    evaluator=dict(type=SuperGPQAEvaluator),
    pred_role='BOT',
)
supergpqa_dataset = dict(
    type=SuperGPQADataset,
    abbr='supergpqa',
    path='m-a-p/SuperGPQA',
    prompt_mode='zero-shot',
    reader_cfg=reader_cfg,
    infer_cfg=infer_cfg,
    eval_cfg=eval_cfg,
)

supergpqa_datasets = [supergpqa_dataset]
