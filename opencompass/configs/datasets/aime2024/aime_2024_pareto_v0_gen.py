from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.evaluator import MATHVerifyEvaluator
from opencompass.datasets import Aime2024Dataset


aime2024_reader_cfg = dict(
    input_columns=['question'],
    output_column='answer'
)

prompt="""
Question:
{question}

Answer the question following the rules:
  1) Return only the final answer.
  1) Put your final answer within \\boxed{}.
  2) Do not include explanations
  3) Do not include reasoning steps
  4) Do not include any additional text outside \\boxed{}

Answer:
"""

aime2024_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt=prompt),
            ],
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

aime2024_eval_cfg = dict(
    evaluator=dict(type=MATHVerifyEvaluator)
)

aime2024_datasets = [
    dict(
        abbr='aime2024',
        type=Aime2024Dataset,
        path='opencompass/aime2024',
        reader_cfg=aime2024_reader_cfg,
        infer_cfg=aime2024_infer_cfg,
        eval_cfg=aime2024_eval_cfg,
    )
]
