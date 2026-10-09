import datetime
import dataclasses
import json
import logging
import os
import time
from wrapt_timeout_decorator import *

logger = logging.getLogger("desktopenv.experiment")


def _is_json_action_like(action) -> bool:
    return dataclasses.is_dataclass(action) and hasattr(action, "action_code")


def _action_to_code(action):
    if _is_json_action_like(action):
        return getattr(action, "action_code", None)
    return action


def update_agent_image_urls(data, url_1, url_2):
    """
    Replace embedded image URLs in agent messages based on image count.
    - One image is replaced with url_1.
    - Two images are replaced with url_1 and url_2 respectively.
    """
    # Iterate through each agent entry, such as manager, operator, or reflector.
    try:
        for agent_name, agent_info in data.items():
            # Skip entries that do not contain a messages list.
            if "messages" not in agent_info:
                continue

            # Iterate through each message.
            for message in agent_info["messages"]:
                msgs_to_process = message if isinstance(message, list) else [message]

                for msg in msgs_to_process:
                    # Safety check and bug fix for nested message handling.
                    # Ensure msg is a dict with content.
                    if not isinstance(msg, dict) or "content" not in msg:
                        continue

                    # Collect image nodes.
                    image_nodes = [
                        item
                        for item in msg["content"]
                        if isinstance(item, dict) and item.get("type") == "image_url"
                    ]

                    # Simplified replacement logic.
                    # If the list is not empty, the first image maps to url_1.
                    if image_nodes:
                        image_nodes[0]["image_url"]["url"] = url_1

                    # If there are at least two images, the second maps to url_2.
                    if len(image_nodes) >= 2:
                        image_nodes[1]["image_url"]["url"] = url_2
    except Exception:
        pass
    finally:
        return data


def run_instruction_task(agent, env, instruction, args, example_result_dir):
    """Run one free-form instruction without benchmark evaluator metadata."""
    os.makedirs(example_result_dir, exist_ok=True)
    runtime_logger = setup_logger({"id": "instruction_task"}, example_result_dir)
    try:
        agent.reset(runtime_logger)
    except TypeError:
        agent.reset()

    for attempt_idx in range(3):
        try:
            env.reset(task_config=None)
            break
        except Exception:
            if attempt_idx == 2:
                raise
            time.sleep(5)

    env.instruction = instruction
    time.sleep(getattr(args, "env_ready_wait", 60))

    done = False
    step_idx = 0

    recording_started = False
    try:
        env.controller.start_recording()
        recording_started = True
    except Exception as e:
        logger.warning("Failed to start recording: %s", e)

    try:
        while not done and step_idx < args.max_steps:
            logger.info(
                "======================================================================================"
            )
            (
                global_state,
                action,
                step_status,
                reward,
                done,
                screenshot_before,
                screenshot_after,
            ) = agent.step(instruction, env, args, example_result_dir)

            updated_global_state = update_agent_image_urls(
                global_state, screenshot_before, screenshot_after
            )
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

            with open(
                os.path.join(example_result_dir, "global_state.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(updated_global_state, f, ensure_ascii=False, indent=2)

            if step_status is False:
                done = True
                reward = None
            else:
                obs = env._get_obs()
                step_idx += 1
                with open(
                    os.path.join(example_result_dir, f"step_{step_idx}.png"), "wb"
                ) as _f:
                    _f.write(obs["screenshot"])

            with open(
                os.path.join(example_result_dir, "traj.jsonl"),
                "a",
                encoding="utf-8",
            ) as f:
                f.write(
                    json.dumps(
                        {
                            "step_num": step_idx,
                            "step_status": step_status,
                            "action_timestamp": action_timestamp,
                            "action": _action_to_code(action),
                            "reward": reward,
                            "done": done,
                            "screenshot_file": f"step_{step_idx}.png",
                        },
                        ensure_ascii=False,
                    )
                )
                f.write("\n")

            if done:
                logger.info("The instruction run is done.")
                break

    finally:
        if recording_started:
            try:
                env.controller.end_recording(
                    os.path.join(example_result_dir, "recording.mp4")
                )
            except Exception as e:
                logger.warning("Failed to end recording: %s", e)


def setup_logger(example, example_result_dir):
    runtime_logger = logging.getLogger(f"desktopenv.example.{example['id']}")
    runtime_logger.setLevel(logging.DEBUG)
    runtime_logger.addHandler(
        logging.FileHandler(
            os.path.join(example_result_dir, "runtime.log"), encoding="utf-8"
        )
    )
    return runtime_logger
