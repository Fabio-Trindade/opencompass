# THIS SHALL ALSO BE DEPRECATED
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import HumanevalDataset, HumanEvalPlusEvaluator, humaneval_postprocess_v2

humaneval_plus_reader_cfg = dict(
    input_columns=['prompt'], output_column='task_id', train_split='test')

humaneval_plus_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                 prompt = "You are given an incomplete Python function:"
                         "{prompt}"
                         "Follow the instructions:"
                         " 1) Implement it correctly according to the docstring"
                         " 2) Do not add explanations or extra text"
                         " 3) Return only the final function code into ```python\n<function>\n```\n"),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

humaneval_plus_eval_cfg = dict(
    evaluator=dict(type=HumanEvalPlusEvaluator),
    pred_role='BOT',
    k=[1], 
    pred_postprocessor=dict(type=humaneval_postprocess_v2),
)

humaneval_plus_datasets = [
    dict(
        abbr='humaneval_plus',
        type=HumanevalDataset,
        path='opencompass/humaneval',
        reader_cfg=humaneval_plus_reader_cfg,
        infer_cfg=humaneval_plus_infer_cfg,
        eval_cfg=humaneval_plus_eval_cfg)
]
