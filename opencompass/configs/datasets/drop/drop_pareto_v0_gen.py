from mmengine.config import read_base
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import DropOpenAIDataset, DropOpenAIEvaluator

with read_base():
    from .drop_examples import drop_examples  # noqa: F401, F403

drop_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='answers',
    train_split='validation',
    test_split='validation',
)

prompt = """
Passage:
{drop_examples}

Question:
{prompt}

Answer the question following the instructions:
  1) Return only the final answer in the exact format: 'Answer: $ANSWER'
  2) Do not include explanations
  3) Do not include reasoning steps
  4) Do not include any additional text
  5) Output must be exactly one line in the required format

Answer:
"""

drop_infer_cfg = dict(
    prompt_template=dict(type=PromptTemplate, template=dict(round=[dict(role='HUMAN', prompt=prompt)])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

drop_eval_cfg = dict(evaluator=dict(type=DropOpenAIEvaluator))

drop_datasets = [
    dict(
        abbr='drop',
        type=DropOpenAIDataset,
        path='data/drop_simple_eval/dev.jsonl',
        reader_cfg=drop_reader_cfg,
        infer_cfg=drop_infer_cfg,
        eval_cfg=drop_eval_cfg)
]
