from mmengine.config import read_base
from opencompass.models import VLLMwithChatTemplate
from opencompass.partitioners import SizePartitioner, NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
    
with read_base():
    from ...opencompass.configs.datasets.math.math_500_gen_pareto_v0 import math_datasets
    # from ...opencompass.configs.datasets.humaneval.humaneval_gen_pareto_v0 \
    #     import humaneval_datasets  
    # from ...opencompass.configs.datasets.mmlu.mmlu_pareto_v0_gen import mmlu_datasets
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
    from .load_vars import total_cpus, total_gpus, hf_models

data_per_dataset = 100

datasets = [
    math_datasets,
    gsm8k_datasets,
    bbh_datasets,
    drop_datasets,
    ifeval_datasets,
    commonsenseqa_datasets,
    PMMEval_XNLI_datasets,
    piqa_datasets,
    winogrande_datasets,
    ruler_datasets,
    mmlu_pro_datasets,
    hellaswag_datasets,
    humaneval_plus_datasets,
    mmmlu_datasets,
    supergpqa_datasets,
    LongBenchv2_datasets,
    aime_datasets    
]

for i,dataset_list in enumerate(datasets):
    for j,dataset in enumerate(dataset_list):
        dataset["reader_cfg"]["train_range"] = data_per_dataset
        dataset["reader_cfg"]["test_range"] = data_per_dataset
        
max_out =  2048
models = []
for model_config in hf_models:
    required_gpus = model_config["required_gpus"]
    estimated_parallel_seqs =  min(data_per_dataset,model_config["estimated_parallel_seqs"])
    model_name = model_config["model_name"]
    hf_path = model_config["hf_path"]
    sampling_kwargs = model_config["sampling_kwargs"]
 
   
    sampling_kwargs["max_tokens"] = max_out
    config = dict(
        type=VLLMwithChatTemplate,
        abbr = model_name,
        path=hf_path,
        batch_size = estimated_parallel_seqs, 
        generation_kwargs= dict(
            seed = 0,
            temperature = 0,
            max_tokens = max_out
            ),
        auto_truncate_size = max_out,
        model_kwargs=dict(
            gpu_memory_utilization = 0.95,
            max_num_seqs = estimated_parallel_seqs,
            trust_remote_code = True,
            enable_prefix_caching = True,
            pipeline_parallel_size = required_gpus
        ),
        run_cfg=dict(
            num_gpus=required_gpus,
            num_procs=1
        ),
        
    )
    models.append(config)

datasets = sum(datasets, [])


infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker = total_gpus
        # max_task_size = 40000
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
