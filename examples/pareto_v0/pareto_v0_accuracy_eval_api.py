__json = __import__("json")
__os = __import__("os")
__Path = __import__("pathlib", fromlist=["Path"]).Path

from mmengine.config import read_base
from opencompass.models import OpenAISDK
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.runners.local_api import LocalAPIRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    from ...opencompass.configs.datasets.math.math_500_gen_pareto_v0 import math_datasets
    from ...opencompass.configs.datasets.gsm8k.gsm_8k_pareto_v0_gen import gsm8k_datasets
    from ...opencompass.configs.datasets.bbh.bbh_pareto_v0_gen import bbh_datasets
    from ...opencompass.configs.datasets.drop.drop_pareto_v0_gen import drop_datasets
    from ...opencompass.configs.datasets.IFEval.IFEval_gen import ifeval_datasets
    from ...opencompass.configs.datasets.commonsenseqa.commomsenseqa_pareto_v0_gen import commonsenseqa_datasets
    from ...opencompass.configs.datasets.PMMEval.xnli_pareto_v0_gen import PMMEval_XNLI_datasets
    from ...opencompass.configs.datasets.piqa.piqa_gen_pareto_v0 import piqa_datasets
    from ...opencompass.configs.datasets.winogrande.winogrande_gen_pareto_v0 import winogrande_datasets
    from ...opencompass.configs.datasets.ruler.ruler_pareto_v0 import ruler_datasets
    from ...opencompass.configs.datasets.mmlu_pro.mmlu_pro_pareto_v0 import mmlu_pro_datasets
    from ...opencompass.configs.datasets.hellaswag.hellaswag_pareto_v0_gen import hellaswag_datasets
    from ...opencompass.configs.datasets.humaneval_plus.humaneval_plus_pareto_v0_gen import humaneval_plus_datasets
    from ...opencompass.configs.datasets.mmmlu.mmmlu_gen import mmmlu_datasets
    from ...opencompass.configs.datasets.supergpqa.supergpqa_pareto_v0 import supergpqa_datasets
    from ...opencompass.configs.datasets.longbenchv2.longbenchv2_pareto_v0_gen import LongBenchv2_datasets
    from ...opencompass.configs.datasets.aime.aime_pareto_v0 import aime_datasets
    from .load_vars import total_cpus, total_gpus

data_per_dataset = 100
max_out = 2048

dataset_groups = [
    math_datasets,
    gsm8k_datasets,
    bbh_datasets,
    drop_datasets,
    ifeval_datasets,
    commonsenseqa_datasets,
    PMMEval_XNLI_datasets,
    piqa_datasets,
    winogrande_datasets,
    mmlu_pro_datasets,
    hellaswag_datasets,
    humaneval_plus_datasets,
    mmmlu_datasets,
    supergpqa_datasets,
    aime_datasets,
    ruler_datasets,
    LongBenchv2_datasets,
]

for dataset_list in dataset_groups:
    for dataset in dataset_list:
        dataset["reader_cfg"]["train_range"] = f"[:{data_per_dataset}]"
        dataset["reader_cfg"]["test_range"] = f"[:{data_per_dataset}]"

datasets = sum(dataset_groups, [])

vllm_state = __json.loads(
    __Path(__os.environ["OC_VLLM_STATE_JSON"]).read_text(encoding="utf-8")
)

models = []

for record in vllm_state["models"]:
    estimated_parallel_seqs = min(
        data_per_dataset,
        int(record.get("estimated_parallel_seqs", 128)),
    )

    models.append(
    dict(
        type=OpenAISDK,
        abbr=record["model_name"],
        path=record["served_name"],

        key=vllm_state.get("api_key", __os.getenv("VLLM_API_KEY", "EMPTY")),
        openai_api_base=record["base_url"],


        tokenizer_path=record.get("tokenizer_path", record.get("hf_path", record["served_name"])),
        mode = "front",
        max_out_len=max_out,
        max_seq_len=int(__os.getenv("OC_MAX_SEQ_LEN", "131072")),
        batch_size=estimated_parallel_seqs,
        query_per_second=int(__os.getenv("OC_QPS_PER_MODEL", "100")),
        retry=int(__os.getenv("OC_API_RETRY", "3")),
        temperature=0.0,
        rpm_verbose=True,
        run_cfg=dict(
            num_gpus=0,
            num_procs=1,
        ),
        extra_body={
            "truncate_prompt_tokens": int(__os.getenv("OC_MAX_SEQ_LEN", "131072")) - max_out,
            "truncation_side": "left",
            "skip_special_tokens": False,
            "return_token_ids": True,
        }
    )
)

infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker=int(__os.getenv("OC_INFER_NUM_WORKERS", str(max(1, total_gpus)))),
    ),
    runner=dict(
        type=LocalAPIRunner,
        max_num_workers=int(__os.getenv("OC_MAX_API_WORKERS", str(max(1, len(models))))),
        concurrent_users=int(__os.getenv("OC_CONCURRENT_USERS", str(max(1, total_gpus)))),
        task=dict(type=OpenICLInferTask),
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=int(__os.getenv("OC_EVAL_WORKERS", str(total_cpus))),
        task=dict(type=OpenICLEvalTask),
    ),
)

work_dir = __os.getenv("OC_WORK_DIR", "experiments/pareto_v0/")