from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import LongBenchv2Dataset, LongBenchv2Evaluator
from opencompass.utils.text_postprocessors import first_option_postprocess

LongBenchv2_reader_cfg = dict(
    input_columns=['context', 'question', 'choice_A', 'choice_B', 'choice_C', 'choice_D', 'difficulty', 'length'],
    output_column='answer',
)
prompt=prompt = """
Please read the following text and answer the question below.

<text>
{context}
</text>

Question:
{question}

Choices:
(A) {choice_A}
(B) {choice_B}
(C) {choice_C}
(D) {choice_D}

Choose the option that correctly answers the question, following the instructions:
  1) Return only the final answer
  2) The answer must be exactly one of: (A), (B), (C), or (D)
  3) The output must follow the format: 'The correct answer is $OPTION' (without quotes)
  4) Do not include explanations
  5) Do not include reasoning steps
  6) Do not include any additional text
  7) Output must be exactly one line
"""
LongBenchv2_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt=prompt,
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

LongBenchv2_eval_cfg = dict(
    evaluator=dict(type=LongBenchv2Evaluator),  
    pred_role='BOT',
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD')  
)

LongBenchv2_datasets = [
    dict(
        type=LongBenchv2Dataset,
        abbr='LongBenchv2',
        path='opencompass/longbenchv2',
        reader_cfg=LongBenchv2_reader_cfg,
        infer_cfg=LongBenchv2_infer_cfg,
        eval_cfg=LongBenchv2_eval_cfg,
    )
]
