import importlib

lib_models_list = [
    "experiments.pareto_v0.compression.quantization.sym_asym.quantize_weight_sym_asym_llama_3_1"
]

for lib in lib_models_list:
    module = importlib.import_module(
        lib
    )
    module.register_all()

hf_module = importlib.import_module(
    "src.utils.hf",
)

system_module = importlib.import_module(
    "src.utils.system",
)


opencompass_module = importlib.import_module(
    "src.utils.opencompass",
)

total_gpus = system_module.get_num_gpus_slurm()
total_cpus = system_module.get_num_cpus_slurm()

hf_models = [opencompass_module.create_performance_template_from_hf_variant(model) for model in hf_module.HFModelRegistry.get_models()]