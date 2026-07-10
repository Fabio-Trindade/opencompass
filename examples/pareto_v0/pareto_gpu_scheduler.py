#!/usr/bin/env python3

import argparse
import concurrent.futures as cf
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
import certifi

def load_registry(path: str):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("load_vars", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.hf_models


def safe_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name)
    return name.strip("-")


def visible_gpus(total_gpus: int):
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if raw:
        return [x for x in raw.split(",") if x != ""]
    return [str(i) for i in range(total_gpus)]


def wait_ready(
    host: str,
    port: int,
    api_key: str,
    timeout_s: int,
    proc: subprocess.Popen | None = None,
    log_path: Path | None = None,
) -> tuple[bool, str]:
    url = f"http://{host}:{port}/v1/models"
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        if proc is not None:
            ret = proc.poll()
            if ret is not None:
                tail = ""
                if log_path is not None and log_path.exists():
                    try:
                        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-80:])
                    except Exception:
                        tail = ""

                return (
                    False,
                    f"vLLM process exited before readiness. exit_code={ret}. "
                    f"log={log_path}\n--- log tail ---\n{tail}",
                )

        try:
            req = urllib.request.Request(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return True, "ready"
        except Exception:
            time.sleep(2)

    return False, f"vLLM readiness timeout after {timeout_s}s. log={log_path}"


def kill_process_group(proc: subprocess.Popen):
    if proc is None:
        return

    if proc.poll() is not None:
        return

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        pass

    time.sleep(5)

    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass


def run_one_model(model, gpu_ids, port, args):
    model_name = model["model_name"]
    hf_path = model["hf_path"]
    required_gpus = int(model["required_gpus"])
    estimated_parallel_seqs = int(model.get("estimated_parallel_seqs", 128))
    disable_q_quant = model.get("disable_q_quant", False) and "KV8" in hf_path
    served_name = safe_name(model_name)
    if disable_q_quant:
        model_name = f"{model_name}-FI"
    run_name = f"{safe_name(model_name)}"

    runtime_dir = Path(args.runtime_dir).resolve()
    log_dir = Path(args.log_dir).resolve()
    work_root = Path(args.work_root).resolve()

    runtime_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    state_path = runtime_dir / f"vllm_state_{run_name}.json"
    vllm_log = log_dir / f"vllm_{run_name}.log"
    oc_log = log_dir / f"opencompass_{run_name}.log"
    model_work_dir = work_root / run_name

    host = args.host
    base_url = f"http://{host}:{port}/v1"
    api_key = args.api_key
    
    state = {
        "api_key": api_key,
        "models": [
            {
                "model_name": model_name,
                "hf_path": hf_path,
                "served_name": served_name,
                "base_url": base_url,
                "host": host,
                "port": port,
                "required_gpus": required_gpus,
                "estimated_parallel_seqs": estimated_parallel_seqs,
                "cuda_visible_devices": ",".join(gpu_ids),
            }
        ],
    }

    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    vllm_env = os.environ.copy()
    vllm_env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    vllm_env["VLLM_API_KEY"] = api_key
    vllm_env["SSL_CERT_FILE"] = certifi.where()
    vllm_env["REQUESTS_CA_BUNDLE"] = certifi.where()
    vllm_env["CURL_CA_BUNDLE"] = certifi.where()

    vllm_cmd = [
        args.vllm_bin,
        "serve",
        hf_path,
        "--served-model-name",
        served_name,
        "--host",
        host,
        "--port",
        str(port),
        "--api-key",
        api_key,
        "--trust-remote-code",
        "--pipeline-parallel-size",
        str(required_gpus),
        "--enable-prefix-caching",
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(estimated_parallel_seqs),
        "--kv-cache-dtype",
        "fp8" if "KV8" in hf_path else "auto"
    ]
    
    if disable_q_quant:
        vllm_cmd.append(
            "--attention-config"
        )
        vllm_cmd.append((
                '{"backend":"FLASHINFER",'
                '"disable_flashinfer_q_quantization":true}'
            ))
        
    if args.extra_vllm_args:
        vllm_cmd.extend(args.extra_vllm_args.split())

    vllm_proc = None
    
    is_qwen_model = "qwen" in hf_path.lower()

    if is_qwen_model:
        print("Qwen model identified -> Disabling thinking + enabling 128k context")
        vllm_cmd.append("--default-chat-template-kwargs")
        vllm_cmd.append('{"enable_thinking": false}')

        vllm_cmd.append("--hf-overrides")
        vllm_cmd.append(
            '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}'
        )
        
    try:
        print(
            f"[START] {model_name} | GPUs={gpu_ids} | port={port} | TP={required_gpus}",
            flush=True,
        )

        with open(vllm_log, "w", buffering=1) as f_vllm:
            vllm_proc = subprocess.Popen(
                vllm_cmd,
                stdout=f_vllm,
                stderr=subprocess.STDOUT,
                env=vllm_env,
                start_new_session=True,
            )

            state["models"][0]["pid"] = vllm_proc.pid
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

            ready, ready_msg = wait_ready(
                host=host,
                port=port,
                api_key=api_key,
                timeout_s=args.vllm_timeout_s,
                proc=vllm_proc,
                log_path=vllm_log,
            )

            if not ready:
                raise RuntimeError(
                    f"vLLM server did not become ready for {model_name}.\n{ready_msg}"
                )

            print(f"[READY] {model_name} at {base_url}", flush=True)

            oc_env = os.environ.copy()
            oc_env["OC_VLLM_STATE_JSON"] = str(state_path)
            oc_env["VLLM_API_KEY"] = api_key
            oc_env["OC_WORK_DIR"] = str(model_work_dir)

            oc_env["OC_DATA_PER_DATASET"] = str(args.data_per_dataset)
            oc_env["OC_MAX_OUT"] = str(args.max_out)
            oc_env["OC_MAX_SEQ_LEN"] = str(args.max_model_len)

            oc_env["OC_INFER_NUM_WORKERS"] = str(args.oc_infer_workers)
            oc_env["OC_MAX_API_WORKERS"] = str(args.oc_api_workers)
            oc_env["OC_CONCURRENT_USERS"] = str(args.oc_concurrent_users)
            oc_env["OC_QPS_PER_MODEL"] = str(args.oc_qps)
            oc_env["OC_EVAL_WORKERS"] = str(args.oc_eval_workers)

            oc_cmd = [
                args.opencompass_bin,
                args.opencompass_config,
            ]

            if args.reuse:
                oc_cmd.extend(["--reuse", args.reuse])

            print(f"[EVAL] {model_name} | work_dir={model_work_dir}", flush=True)

            with open(oc_log, "w", buffering=1) as f_oc:
                oc_proc = subprocess.Popen(
                    oc_cmd,
                    stdout=f_oc,
                    stderr=subprocess.STDOUT,
                    cwd=args.opencompass_cwd,
                    env=oc_env,
                )

                ret = oc_proc.wait()

            if ret != 0:
                raise RuntimeError(
                    f"OpenCompass failed for {model_name} with exit code {ret}. "
                    f"See log: {oc_log}"
                )

            print(f"[DONE] {model_name}", flush=True)

            return {
                "model_name": model_name,
                "status": "ok",
                "work_dir": str(model_work_dir),
                "vllm_log": str(vllm_log),
                "opencompass_log": str(oc_log),
            }

    except Exception as e:
        print(f"[FAIL] {model_name}: {e}", flush=True)
        return {
            "model_name": model_name,
            "status": "failed",
            "error": str(e),
            "vllm_log": str(vllm_log),
            "opencompass_log": str(oc_log),
        }

    finally:
        kill_process_group(vllm_proc)
        print(f"[FREE] {model_name} | GPUs={gpu_ids}", flush=True)


def filter_models(hf_models, args):
    selected = []

    name_filter = None
    if args.model_names:
        name_filter = {x.strip() for x in args.model_names.split(",") if x.strip()}

    for model in hf_models:
        if args.only_8b and "8B" not in model["hf_path"]:
            continue

        if name_filter is not None and model["model_name"] not in name_filter:
            continue

        selected.append(model)

    selected = selected[args.start_index:]

    if args.max_models is not None:
        selected = selected[:args.max_models]

    if args.sort_by_gpus == "desc":
        selected = sorted(selected, key=lambda m: int(m["required_gpus"]), reverse=True)
    elif args.sort_by_gpus == "asc":
        selected = sorted(selected, key=lambda m: int(m["required_gpus"]))

    return selected


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--registry", required=True)
    parser.add_argument("--opencompass-config", required=True)
    parser.add_argument("--opencompass-cwd", required=True)

    parser.add_argument("--vllm-bin", required=True)
    parser.add_argument("--opencompass-bin", required=True)

    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--work-root", required=True)

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=8000)
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))

    parser.add_argument("--total-gpus", type=int, default=4)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--model-names", default="")
    parser.add_argument("--only-8b", action="store_true")

    # Política de packing:
    # none: mantém ordem do registry
    # desc: tenta modelos maiores primeiro
    # asc: tenta modelos menores primeiro
    parser.add_argument("--sort-by-gpus", choices=["none", "asc", "desc"], default="none")

    parser.add_argument("--reuse", default="")

    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32000)
    parser.add_argument("--vllm-timeout-s", type=int, default=1600)
    parser.add_argument("--extra-vllm-args", default="")

    parser.add_argument("--data-per-dataset", type=int, default=100)
    parser.add_argument("--max-out", type=int, default=2048)

    # Importante: como pode haver vários OpenCompass simultâneos,
    # estes valores devem ser pequenos.
    parser.add_argument("--oc-infer-workers", type=int, default=1)
    parser.add_argument("--oc-api-workers", type=int, default=1)
    parser.add_argument("--oc-concurrent-users", type=int, default=1)
    parser.add_argument("--oc-qps", type=int, default=20)
    parser.add_argument("--oc-eval-workers", type=int, default=8)

    args = parser.parse_args()

    hf_models = load_registry(args.registry)
    pending = filter_models(hf_models, args)

    if not pending:
        raise RuntimeError("No models selected.")

    all_gpus = visible_gpus(args.total_gpus)
    free_gpus = list(all_gpus)

    print(f"[INFO] Visible GPUs: {all_gpus}", flush=True)
    print(f"[INFO] Pending models: {[m['model_name'] for m in pending]}", flush=True)

    port_counter = args.base_port
    running = {}
    results = []

    max_workers = len(pending)

    with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
        while pending or running:
            launched_any = False

            # First-fit: lança qualquer modelo que couber agora.
            i = 0
            while i < len(pending):
                model = pending[i]
                need = int(model["required_gpus"])

                if need <= len(free_gpus):
                    allocated = free_gpus[:need]
                    del free_gpus[:need]

                    port = port_counter
                    port_counter += 1

                    future = executor.submit(
                        run_one_model,
                        model,
                        allocated,
                        port,
                        args,
                    )

                    running[future] = {
                        "model": model,
                        "gpus": allocated,
                        "port": port,
                    }

                    print(
                        f"[ALLOC] {model['model_name']} -> GPUs={allocated}, port={port}",
                        flush=True,
                    )

                    pending.pop(i)
                    launched_any = True
                else:
                    i += 1

            if not running:
                raise RuntimeError(
                    "Scheduler deadlock: pending models require more GPUs than available."
                )

            if not launched_any:
                done, _ = cf.wait(
                    running.keys(),
                    return_when=cf.FIRST_COMPLETED,
                )

                for future in done:
                    meta = running.pop(future)
                    free_gpus.extend(meta["gpus"])
                    free_gpus.sort(key=lambda x: int(x) if x.isdigit() else x)

                    try:
                        result = future.result()
                    except Exception as e:
                        result = {
                            "model_name": meta["model"]["model_name"],
                            "status": "failed",
                            "error": str(e),
                        }

                    results.append(result)

                    print(
                        f"[RELEASE] {meta['model']['model_name']} -> GPUs={meta['gpus']}",
                        flush=True,
                    )

        if running:
            done, _ = cf.wait(running.keys())
            for future in done:
                meta = running[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {
                        "model_name": meta["model"]["model_name"],
                        "status": "failed",
                        "error": str(e),
                    }
                results.append(result)

    summary_path = Path(args.runtime_dir).resolve() / "scheduler_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"[INFO] Summary written to {summary_path}", flush=True)

    failed = [r for r in results if r.get("status") != "ok"]
    if failed:
        print("[ERROR] Some models failed:", flush=True)
        print(json.dumps(failed, indent=2), flush=True)
        sys.exit(1)

    print("[INFO] All model evaluations completed successfully.", flush=True)


if __name__ == "__main__":
    main()