from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import CustomDataset
from opencompass.evaluator import MATHVerifyEvaluator

math_reader_cfg = dict(input_columns=['problem'], output_column='solution')
prompt = """
Question:
{problem}

Answer the question following the instructions:
  1) Return only the final answer
  2) Put your final answer within \\boxed{}
  3) Do not include any additional text outside \\boxed{}
  4) Do not include explanations
  5) Do not include reasoning steps

Answer:
""" 
math_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt=prompt,
                ),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)


math_eval_cfg = dict(
    evaluator=dict(type=MATHVerifyEvaluator),
)

math_datasets = [
    dict(
        type=CustomDataset,
        abbr='math-500',
        path='opencompass/math',
        file_name='test_prm800k_500.jsonl',
        reader_cfg=math_reader_cfg,
        infer_cfg=math_infer_cfg,
        eval_cfg=math_eval_cfg,
    )
]
