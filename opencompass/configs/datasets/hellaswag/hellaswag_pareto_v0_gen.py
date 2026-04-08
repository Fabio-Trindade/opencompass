from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import HellaswagDataset_V2
from opencompass.utils.text_postprocessors import first_option_postprocess

hellaswag_reader_cfg = dict(
    input_columns=['ctx', 'A', 'B', 'C', 'D'],
    output_column='label',
    train_range = 1000,
    test_range = 1000
)

prompt = """
{ctx}

Question:
Which ending makes the most sense?

Options:
A. {A}
B. {B}
C. {C}
D. {D}

Choose the option that correctly answers the question, following the instructions:
  1) Return only the final answer
  2) The answer must be exactly in the format: 'ANSWER: $LETTER'
  3) $LETTER must be one of A, B, C, or D
  4) Do not include explanations
  5) Do not include reasoning steps
  6) Do not include any additional text
  7) Output must be exactly one line
"""

hellaswag_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt=prompt,
            ),
        ]),
    ),
    retriever=dict(type=ZeroRetriever, ),
    inferencer=dict(type=GenInferencer),
)

hellaswag_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'),
)

hellaswag_datasets = [
    dict(
        abbr='hellaswag',
        type=HellaswagDataset_V2,
        path='opencompass/hellaswag',
        reader_cfg=hellaswag_reader_cfg,
        infer_cfg=hellaswag_infer_cfg,
        eval_cfg=hellaswag_eval_cfg)
]
