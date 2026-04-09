from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.evaluator import MATHVerifyEvaluator
from opencompass.datasets import Aime2024Dataset
from opencompass.datasets import CustomDataset


aime_reader_cfg = dict(
    input_columns=['question'],
    output_column='answer'
)

aime2026_reader_cfg = dict(input_columns=['problem'], output_column='answer')

prompt="""
Question:
{question}

Answer the question following the instructions:
  1) Return only the final answer
  2) Put your final answer within \\boxed{}
  3) Do not include explanations
  4) Do not include reasoning steps
  5) Do not include any additional text outside \\boxed{}

Answer:
"""

aime2026_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt="""
                            Problem:
                            {problem}

                            Solve the problem following the instructions:
                              1) Return only the final answer
                              2) Put your final answer within \\boxed{}
                              3) Do not include explanations
                              4) Do not include reasoning steps
                              5) Do not include any additional text outside \\boxed{}

                            Answer:
                            """,
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
aime_infer_cfg = dict(
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

aime_eval_cfg = dict(
    evaluator=dict(type=MATHVerifyEvaluator)
)

aime_datasets = [
    dict(
        abbr='aime2024',
        type=Aime2024Dataset,
        path='opencompass/aime2024',
        reader_cfg=aime_reader_cfg,
        infer_cfg=aime_infer_cfg,
        eval_cfg=aime_eval_cfg,
    ),
    
    dict(
        type=CustomDataset,
        abbr='aime2025',
        path='opencompass/aime2025',
        reader_cfg=aime_reader_cfg,
        infer_cfg=aime_infer_cfg,
        eval_cfg=aime_eval_cfg
    ),
    
     dict(
        type=CustomDataset,
        abbr='aime2026',
        path='opencompass/aime2026',
        reader_cfg=aime2026_reader_cfg,
        infer_cfg=aime2026_infer_cfg,
        eval_cfg=aime_eval_cfg
    )   
]
