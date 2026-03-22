from mmengine.config import read_base
from opencompass.models import VLLMwithChatTemplate
from opencompass.partitioners import SizePartitioner, NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
    
with read_base():
    from ...opencompass.configs.datasets.humaneval.humaneval_gen_pareto_v0 \
        import humaneval_datasets  
    from ...opencompass.configs.datasets.piqa.piqa_gen_pareto_v0 import piqa_datasets
    from ...opencompass.configs.datasets.math.math_500_gen_pareto_v0 import math_datasets
    from ...opencompass.configs.datasets.winogrande.winogrande_gen_pareto_v0 import winogrande_datasets
    from ...opencompass.configs.datasets.ruler.ruler_pareto_v0 import ruler_datasets
    from ...opencompass.configs.datasets.PMMEval.xnli_gen import PMMEval_XNLI_datasets
    from .load_vars import total_cpus, total_gpus, hf_models

datasets = [
    piqa_datasets,
    math_datasets,
    winogrande_datasets,
    humaneval_datasets,
    ruler_datasets,
    PMMEval_XNLI_datasets
]

max_out =  8192
models = []
for model_config in hf_models:
    required_gpus = model_config["required_gpus"]
    estimated_parallel_seqs =  model_config["estimated_parallel_seqs"]
    model_name = model_config["model_name"]
    hf_path = model_config["hf_path"]
    config = dict(
        type=VLLMwithChatTemplate,
        abbr = model_name,
        path=hf_path,
        batch_size = estimated_parallel_seqs, 
        generation_kwargs=dict(
                            temperature = 0,
                            max_tokens = max_out
                            ),
        auto_truncate_size = max_out,
        model_kwargs=dict(
            gpu_memory_utilization = 0.95,
            max_num_seqs = estimated_parallel_seqs,
            trust_remote_code = True,
            enable_prefix_caching = True
        ),
        run_cfg=dict(
            num_gpus=required_gpus,
            num_procs=1
        )
    )
    models.append(config)

datasets = sum(datasets, [])


infer = dict(
    partitioner=dict(
        type=SizePartitioner,  
        max_task_size = 40000
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=total_gpus,
        task=dict(type=OpenICLInferTask),
        retry=1
    )
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=total_cpus,
        task=dict(type=OpenICLEvalTask)
    )
)

work_dir = "experiments/pareto_v0/performance/"
