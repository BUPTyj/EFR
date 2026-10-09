"""Run the optimized MobileAgent V3 method on one free-form desktop task."""

import argparse
import datetime
import io
import json
import logging
import os
import re
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from mobileagent_v3_optimized import run_single
from mobileagent_v3_optimized.agent import MobileAgentV3


def _configure_stdio_for_utf8() -> None:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
            continue
        except Exception:
            pass

        buffer = getattr(stream, "buffer", None)
        if buffer is None:
            continue
        try:
            wrapped = io.TextIOWrapper(
                buffer, encoding="utf-8", errors="replace", line_buffering=True
            )
            setattr(sys, stream_name, wrapped)
        except Exception:
            pass


def configure_logging(result_dir: str) -> logging.Logger:
    os.makedirs(result_dir, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    log_path = os.path.join(result_dir, "run.log")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    stdout_handler = logging.StreamHandler(sys.stdout)
    file_handler.setLevel(logging.DEBUG)
    stdout_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="[%(asctime)s %(levelname)s %(module)s/%(lineno)d] %(message)s"
    )
    file_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    return logging.getLogger("desktopenv.experiment")


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MobileAgent V3 on one natural-language desktop task."
    )

    parser.add_argument(
        "--instruction",
        type=str,
        required=True,
        help="Natural-language task for the desktop agent to complete.",
    )
    parser.add_argument("--task_id", type=str, default="")

    parser.add_argument("--path_to_vm", type=str, default=None)
    parser.add_argument("--provider_name", type=str, default="docker")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--action_space", type=str, default="pyautogui")
    parser.add_argument(
        "--observation_type",
        choices=["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"],
        default="screenshot",
    )
    parser.add_argument("--screen_width", type=int, default=1920)
    parser.add_argument("--screen_height", type=int, default=1080)
    parser.add_argument("--env_ready_wait", type=float, default=60.0)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--max_trajectory_length", type=int, default=50)

    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--engine", type=str, default="openai")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--api_url", type=str, default="")
    parser.add_argument("--manager_model", type=str, default="")
    parser.add_argument("--worker_model", type=str, default="")
    parser.add_argument("--reflector_model", type=str, default="")
    parser.add_argument("--grounding_model", type=str, default="")
    parser.add_argument("--grounding_stage", type=int, default=1)
    parser.add_argument("--grounding_info_level", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=1500)

    parser.add_argument(
        "--result_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "artifacts", "runs"),
    )
    parser.add_argument(
        "--artifact_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "artifacts", "mobileagent_v3"),
    )
    return parser.parse_args()


def get_model_label(args: argparse.Namespace) -> str:
    if args.model:
        return args.model
    role_models = [
        args.manager_model,
        args.worker_model,
        args.reflector_model,
        args.grounding_model,
    ]
    label = "_".join(model for model in role_models if model)
    return label or "unspecified_model"


def make_task_id(instruction: str, explicit_task_id: str = "") -> str:
    if explicit_task_id:
        return explicit_task_id
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", instruction.strip())[:60].strip("_")
    timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
    return f"{timestamp}_{slug or 'instruction_task'}"


def build_agent(args: argparse.Namespace) -> MobileAgentV3:
    manager_engine_params = {
        "engine_type": args.engine,
        "api_key": args.api_key,
        "base_url": args.api_url,
        "model": args.model if args.model else args.manager_model,
    }
    worker_engine_params = {
        "engine_type": args.engine,
        "api_key": args.api_key,
        "base_url": args.api_url,
        "model": args.model if args.model else args.worker_model,
    }
    reflector_engine_params = {
        "engine_type": args.engine,
        "api_key": args.api_key,
        "base_url": args.api_url,
        "model": args.model if args.model else args.reflector_model,
    }
    grounding_engine_params = {
        "engine_type": args.engine,
        "api_key": args.api_key,
        "base_url": args.api_url,
        "model": args.model if args.model else args.grounding_model,
    }

    missing_roles = [
        name
        for name, params in {
            "manager": manager_engine_params,
            "worker": worker_engine_params,
            "reflector": reflector_engine_params,
            "grounding": grounding_engine_params,
        }.items()
        if not params["model"]
    ]
    if missing_roles:
        raise ValueError(
            "Model is required. Pass --model for all roles or provide role-specific "
            f"models. Missing: {', '.join(missing_roles)}"
        )

    return MobileAgentV3(
        manager_engine_params,
        worker_engine_params,
        reflector_engine_params,
        grounding_engine_params,
        artifact_dir=args.artifact_dir,
    )


def build_env(args: argparse.Namespace):
    from desktop_env.desktop_env import DesktopEnv

    return DesktopEnv(
        provider_name=args.provider_name,
        path_to_vm=args.path_to_vm,
        action_space=args.action_space,
        screen_size=(args.screen_width, args.screen_height),
        headless=args.headless,
        os_type="Ubuntu",
        require_a11y_tree=args.observation_type
        in ["a11y_tree", "screenshot_a11y_tree", "som"],
    )


def main() -> None:
    _configure_stdio_for_utf8()
    args = config()
    task_id = make_task_id(args.instruction, args.task_id)
    run_dir = os.path.join(
        args.result_dir,
        args.action_space,
        args.observation_type,
        get_model_label(args),
        task_id,
    )
    logger = configure_logging(run_dir)
    logger.info("Instruction: %s", args.instruction)
    logger.info("Run directory: %s", run_dir)

    with open(os.path.join(run_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    env = None
    try:
        agent = build_agent(args)
        env = build_env(args)
        run_single.run_instruction_task(
            agent=agent,
            env=env,
            instruction=args.instruction,
            args=args,
            example_result_dir=run_dir,
        )
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
