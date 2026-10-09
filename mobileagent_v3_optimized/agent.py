from typing import Dict, Optional
from datetime import datetime

import dataclasses
import os
import time
import copy
import json

from .modules import (
    InfoPool,
    Manager,
    Executor,
    ATOMIC_ACTION_SIGNITURES_COMPUTER,
    ATOMIC_ACTION_SIGNITURES_COMPUTER_2stage,
    Grounding,
    Reflector,
)

from .vision_utils import (
    visual_action,
    get_clean_diff_boxes,
    extract_affected_areas,
    draw_boxes_on_image,
    save_image_from_cv2,
)


@dataclasses.dataclass()
class JSONAction:
    action_type: Optional[str] = None
    action_code: Optional[str] = None
    x: Optional[int] = None
    y: Optional[int] = None
    text: Optional[str] = None
    clear: Optional[int] = None
    time: Optional[int] = None
    value: Optional[float] = None
    key_list: Optional[list] = None


def convert_xy(x, y):
    x_ = x * 1920 / 999
    y_ = y * 1080 / 999
    return x_, y_


def convert_fc_action_to_json_action_grounding(
    dummy_action, grounding_model, image_list, grounding_info=""
):  # -> json_action.JSONAction:

    action_json = json.loads(dummy_action)
    action_type = action_json["action"]

    x = None
    y = None
    text = None
    clear = None
    value = None
    time = None
    key_list = None
    action_code = ""

    if "element_description" in action_json:
        [x, y], grounding_messages = grounding_model.predict(
            grounding_info + action_json["element_description"], image_list
        )
    elif "element1_description" in action_json:
        [x1, y1], grounding_messages1 = grounding_model.predict(
            grounding_info + action_json["element1_description"], image_list
        )
        [x2, y2], grounding_messages2 = grounding_model.predict(
            grounding_info + action_json["element2_description"], image_list
        )
        grounding_messages = [grounding_messages1, grounding_messages2]
    else:
        grounding_messages = None

    if action_type == "click":
        x, y = convert_xy(x, y)
        action_code = f"import pyautogui; pyautogui.click(x={x}, y={y})"
    elif action_type == "double_click":
        x, y = convert_xy(x, y)
        action_code = f"import pyautogui; pyautogui.doubleClick(x={x}, y={y})"
    elif action_type == "right_click":
        x, y = convert_xy(x, y)
        action_code = f"import pyautogui; pyautogui.rightClick(x={x}, y={y})"
    elif action_type == "type":
        x, y = convert_xy(x, y)
        text = action_json["text"]
        if "\n" in text and "\\n" not in text:
            text = text.replace("\n", "\\n")
        clear = action_json["clear"]
        enter = action_json["enter"]
        action_code = (
            f"import pyautogui; pyautogui.doubleClick(x={x}, y={y}); "
        )
        if clear > 0:
            action_code += "pyautogui.hotkey('ctrl', 'a'); pyautogui.press('delete'); "
        if "'" not in text:
            action_code += f"pyautogui.typewrite('{text}', interval=1.0)"
        else:
            action_code += f'pyautogui.typewrite("{text}", interval=1.0)'
        if enter > 0:
            action_code += "; pyautogui.press('enter')"
    elif action_type == "hotkey":
        key_list = action_json["keys"]
        key_list_str = "'" + "', '".join(key_list) + "'"
        action_code = f"import pyautogui; pyautogui.hotkey({key_list_str})"
    elif action_type == "scroll":
        x, y = convert_xy(x, y)
        value = action_json["value"]
        action_code = (
            f"import pyautogui; pyautogui.moveTo({x}, {y}); pyautogui.scroll({value})"
        )
    elif action_type == "wait":
        time = action_json["time"]
        action_code = f"import time; time.sleep({time})"
    elif action_type == "done":
        action_type = "done"
        action_code = "DONE"

    elif action_type == "drag":
        x1, y1 = convert_xy(x1, y1)
        x2, y2 = convert_xy(x2, y2)
        action_code = "import pyautogui; "
        action_code += f"pyautogui.moveTo({x1}, {y1}); "
        action_code += (
            f"pyautogui.dragTo({x2}, {y2}, duration=1.); pyautogui.mouseUp(); "
        )

    return JSONAction(
        action_type=action_type,
        x=x,
        y=y,
        text=text,
        clear=clear,
        value=value,
        time=time,
        key_list=key_list,
        action_code=action_code,
    ), grounding_messages


def convert_fc_action_to_json_action(dummy_action):  # -> json_action.JSONAction:

    action_json = json.loads(dummy_action)
    action_type = action_json["action"]

    x = None
    y = None
    text = None
    clear = None
    value = None
    time = None
    key_list = None
    action_code = ""

    if action_type == "click":
        x, y = action_json["coordinate"][0], action_json["coordinate"][1]
        x, y = convert_xy(x, y)
        action_code = f"import pyautogui; pyautogui.click(x={x}, y={y})"
    elif action_type == "double_click":
        x, y = action_json["coordinate"][0], action_json["coordinate"][1]
        x, y = convert_xy(x, y)
        action_code = f"import pyautogui; pyautogui.doubleClick(x={x}, y={y})"
    elif action_type == "right_click":
        x, y = action_json["coordinate"][0], action_json["coordinate"][1]
        x, y = convert_xy(x, y)
        action_code = f"import pyautogui; pyautogui.rightClick(x={x}, y={y})"
    elif action_type == "type":
        x, y = action_json["coordinate"][0], action_json["coordinate"][1]
        x, y = convert_xy(x, y)
        text = action_json["text"]
        if "\n" in text and "\\n" not in text:
            text = text.replace("\n", "\\n")
        clear = action_json["clear"]
        enter = action_json["enter"]
        action_code = f"import pyautogui; pyautogui.click(x={x}, y={y}); "
        if clear > 0:
            action_code += "pyautogui.hotkey('ctrl', 'a'); pyautogui.press('delete'); "
        action_code += f"pyautogui.typewrite('{text}', interval=1.0)"
        if enter > 0:
            action_code += "; pyautogui.press('enter')"
    elif action_type == "hotkey":
        key_list = action_json["keys"]
        key_list_str = "'" + "', '".join(key_list) + "'"
        action_code = f"import pyautogui; pyautogui.hotkey({key_list_str})"
    elif action_type == "scroll":
        x, y = action_json["coordinate"][0], action_json["coordinate"][1]
        x, y = convert_xy(x, y)
        value = action_json["value"]
        action_code = (
            f"import pyautogui; pyautogui.moveTo({x}, {y}); pyautogui.scroll({value})"
        )
    elif action_type == "wait":
        time = action_json["time"]
        action_code = f"import time; time.sleep({time})"
    elif action_type == "done":
        action_type = "done"
        action_code = "DONE"

    elif action_type == "drag":
        x1, y1 = action_json["coordinate"][0], action_json["coordinate"][1]
        x1, y1 = convert_xy(x1, y1)
        x2, y2 = action_json["coordinate2"][0], action_json["coordinate2"][1]
        x2, y2 = convert_xy(x2, y2)
        action_code = "import pyautogui; "
        action_code += f"pyautogui.moveTo({x1}, {y1}); "
        action_code += (
            f"pyautogui.dragTo({x2}, {y2}, duration=1.); pyautogui.mouseUp(); "
        )

    return JSONAction(
        action_type=action_type,
        x=x,
        y=y,
        text=text,
        clear=clear,
        value=value,
        time=time,
        key_list=key_list,
        action_code=action_code,
    )


INIT_TIPS = """
General:
- If you see the "can't update chrome" popup, click the nearby X to close the prompt. Be sure not to click the "reinstall chrome" button.
- Make sure the element you need to operate on is on the screen, if not, you can try scroll action. Before performing a scroll action, verify that your cursor is positioned over the correct scrollable area or the target container. If you want to perform scroll action, 5 or -5 is an appropriate choice for `value` parameter.
- My computer's password is 'password', feel free to use it when you need sudo rights.

Chrome:
- If the Chrome browser page is not maximized, you can use the alt+f10 shortcut to maximize the window, thereby displaying more information.
- If you cannot find the element you want to click on the current webpage, you can use the search function provided by the webpage (usually a search box or a magnifying glass icon), or directly search with appropriate keywords in the Google search engine.
"""


class MobileAgentV3:
    def __init__(
        self,
        manager_engine_params: Dict,
        operator_engine_params: Dict,
        reflector_engine_params: Dict,
        grounding_enging_params: Dict,
        wait_after_action_seconds: float = 3.0,
        artifact_dir: str = os.path.join("artifacts", "mobileagent_v3"),
    ):
        self.manager_engine_params = manager_engine_params
        self.operator_engine_params = operator_engine_params
        self.reflector_engine_params = reflector_engine_params
        self.grounding_enging_params = grounding_enging_params

        self.wait_after_action_seconds = wait_after_action_seconds
        self.artifact_dir = artifact_dir

        # init info pool
        self.info_pool = InfoPool(
            additional_knowledge=copy.deepcopy(INIT_TIPS), err_to_manager_thresh=2
        )

        now = datetime.now()
        time_str = now.strftime("%Y%m%d_%H%M%S")

        self.step_idx = 0

    def reset(self):
        self.info_pool = InfoPool(
            additional_knowledge=copy.deepcopy(INIT_TIPS), err_to_manager_thresh=2
        )

        now = datetime.now()
        time_str = now.strftime("%Y%m%d_%H%M%S")

    def step(self, instruction: str, env, args, example_result_dir: str):
        ## init agents ##
        manager = Manager(self.manager_engine_params)
        executor = Executor(self.operator_engine_params)
        reflector = Reflector(self.reflector_engine_params)

        global_state = {}

        message_manager, message_operator, message_reflector = None, None, None

        self.info_pool.instruction = instruction
        step_idx = len(self.info_pool.action_history)

        print("----------step " + str(step_idx + 1))

        observation = env._get_obs()
        before_screenshot = observation["screenshot"]
        with open(os.path.join(example_result_dir, f"step_{step_idx}.png"), "wb") as _f:
            _f.write(before_screenshot)
        norm_path = os.path.normpath(example_result_dir)
        segments = norm_path.split(os.sep)
        relative_part = os.path.join(*segments[-2:])

        final_path1 = os.path.join(
            self.artifact_dir,
            str(relative_part),
            f"{step_idx}_before.png",
        )
        directory = os.path.dirname(final_path1)
        os.makedirs(directory, exist_ok=True)
        with open(final_path1, "wb") as f:  # "wb" means write binary.
            f.write(before_screenshot)

        self.info_pool.width = 1920
        self.info_pool.height = 1080

        ## check error escalation
        self.info_pool.error_flag_plan = False
        err_to_manager_thresh = self.info_pool.err_to_manager_thresh
        if len(self.info_pool.action_outcomes) >= err_to_manager_thresh:
            # check if the last err_to_manager_thresh actions are all errors
            latest_outcomes = self.info_pool.action_outcomes[-err_to_manager_thresh:]
            count = 0
            for outcome in latest_outcomes:
                if outcome in ["B", "C"]:
                    count += 1
            if count == err_to_manager_thresh:
                self.info_pool.error_flag_plan = True

        skip_manager = False
        ## if previous action is invalid, skip the manager and try again first ##
        if (
            not self.info_pool.error_flag_plan
            and len(self.info_pool.action_history) > 0
        ):
            if self.info_pool.action_history[-1]["action"] == "invalid":
                skip_manager = True

        if not skip_manager:
            print("\n### Manager ... ###\n")

            planning_start_time = time.time()
            prompt_planning = manager.get_prompt(self.info_pool)
            output_planning, message_manager = manager.predict(
                prompt_planning, [before_screenshot]
            )

            global_state["manager"] = {
                "name": "manager",
                "messages": message_manager,
                "response": output_planning,
            }

            parsed_result_planning = manager.parse_response(output_planning)
            self.info_pool.plan = parsed_result_planning["plan"]
            self.info_pool.current_subgoal = parsed_result_planning["current_subgoal"]
            planning_end_time = time.time()

            print("\n\nPlan: " + self.info_pool.plan)
            print("Current subgoal: " + self.info_pool.current_subgoal)
            print("Planning thought: " + parsed_result_planning["thought"], "\n")

        ## if stopping by planner ##
        if "Finished" in self.info_pool.current_subgoal.strip():
            self.info_pool.finish_thought = parsed_result_planning["thought"]
            action_thought = "Finished by planner"
            action_object_str = '{"action": "done"}'
            action_description = "Finished by planner"

        else:
            print("\n### Operator ... ###\n")
            action_decision_start_time = time.time()
            prompt_action = executor.get_prompt(self.info_pool, args.grounding_stage)
            output_action, message_operator = executor.predict(
                prompt_action, [before_screenshot]
            )

            parsed_result_action = executor.parse_response(output_action)
            action_thought, action_object_str, action_description = (
                parsed_result_action["thought"],
                parsed_result_action["action"],
                parsed_result_action["description"],
            )
            action_decision_end_time = time.time()

            action_object_str = action_object_str.split("```json")[-1].split("```")[0]
            self.info_pool.last_action_thought = action_thought
            self.info_pool.last_summary = action_description

            # If the output is not in the right format, add it to step summary which
            # will be passed to next step and return.
            if (not action_thought) or (not action_object_str):
                print("Action prompt output is not in the correct format.")
                self.info_pool.last_action = {"action": "invalid"}
                self.info_pool.action_history.append({"action": "invalid"})
                self.info_pool.summary_history.append(action_description)
                self.info_pool.action_outcomes.append("C")  # no change
                self.info_pool.error_descriptions.append(
                    "invalid action format, do nothing."
                )
                return global_state, None, False, None, False, final_path1, None

        print("\n\nThought: " + action_thought)
        print("Action: " + action_object_str)
        print("Action description: " + action_description, "\n\n")

        format_action_object_str = action_object_str

        operator_response = f"""### Thought ###
{action_thought}

### Action ###
{format_action_object_str}

### Description ###
{action_description}"""
        global_state["operator"] = {
            "name": "operator",
            "messages": message_operator,
            "response": operator_response,
        }

        # --- Validate action_type. ---
        try:
            parsed_action_json = json.loads(action_object_str)
            parsed_action_type = parsed_action_json.get("action", "")
        except (json.JSONDecodeError, AttributeError):
            parsed_action_json = {"action": ""}
            parsed_action_type = ""

        valid_actions = (
            ATOMIC_ACTION_SIGNITURES_COMPUTER_2stage
            if args.grounding_stage > 0
            else ATOMIC_ACTION_SIGNITURES_COMPUTER
        )
        # 'done' is also valid because it is the planner's termination action.
        valid_action_types = set(valid_actions.keys()) | {"done"}

        if parsed_action_type not in valid_action_types:
            print(f'Invalid action_type: "{parsed_action_type}". Skipping reflector.')

            # Build output consistent with a normal reflector result.
            invalid_action_outcome = "C"
            invalid_error_desc = f'The action type "{parsed_action_type}" is invalid. Please use the actions in Atomic Actions.'
            invalid_progress = (
                f'{self.info_pool.progress_status} (The action type "{parsed_action_type}" is invalid)'
                if self.info_pool.progress_status
                else f'(The action type "{parsed_action_type}" is invalid)'
            )

            invalid_reflector_response = f"""### Analyse ###
The action type "{parsed_action_type}" is invalid. Please use the actions in Atomic Actions.

### Outcome ###
C

### Error Description ###
The action type "{parsed_action_type}" is invalid. Please use the actions in Atomic Actions.

### Progress Status ###
{invalid_progress}"""

            global_state["reflector"] = {
                "name": "reflector",
                "messages": None,
                "response": invalid_reflector_response,
            }

            # Print logs consistent with a normal reflector result.
            print("\n### Reflector ... ###\n")
            print(f'  [Skip] Invalid action_type: "{parsed_action_type}"')
            print("\n\nAction reflection outcome: " + invalid_action_outcome)
            print("Action reflection error description: " + invalid_error_desc)
            print("Action reflection progress status: " + invalid_progress, "\n")

            # Update info_pool consistently with the normal flow.
            self.info_pool.last_action = parsed_action_json
            self.info_pool.action_history.append(parsed_action_json)
            self.info_pool.summary_history.append(action_description)
            self.info_pool.action_outcomes.append(invalid_action_outcome)
            self.info_pool.error_descriptions.append(invalid_error_desc)
            self.info_pool.progress_status = invalid_progress
            self.info_pool.progress_status_history.append(invalid_progress)

            return (
                global_state,
                action_object_str,
                False,
                None,
                False,
                final_path1,
                None,
            )

        try:
            if args.grounding_stage > 0:
                grouding_model = Grounding(self.grounding_enging_params)
                if args.grounding_info_level == 0:
                    converted_action, grounding_messages = (
                        convert_fc_action_to_json_action_grounding(
                            action_object_str, grouding_model, [before_screenshot]
                        )
                    )
                elif args.grounding_info_level == 1:
                    grounding_info = (
                        "Thought: " + action_thought + "\nElement description: "
                    )
                    converted_action, grounding_messages = (
                        convert_fc_action_to_json_action_grounding(
                            action_object_str,
                            grouding_model,
                            [before_screenshot],
                            grounding_info,
                        )
                    )
                if grounding_messages is not None:
                    global_state["grounding"] = {
                        "name": "grounding",
                        "messages": grounding_messages,
                    }
            else:
                converted_action = convert_fc_action_to_json_action(action_object_str)

        except Exception as e:
            print("Failed to convert the output to a valid action.")
            print(str(e))
            self.info_pool.last_action = {"action": "invalid"}
            self.info_pool.action_history.append({"action": "invalid"})
            self.info_pool.summary_history.append(action_description)
            self.info_pool.action_outcomes.append("C")  # no change
            self.info_pool.error_descriptions.append(
                "invalid action format, do nothing."
            )
            return (
                global_state,
                action_object_str,
                False,
                None,
                False,
                final_path1,
                None,
            )

        if converted_action.action_type == "done":
            outcome = "A"
            error_description = "None"

            self.info_pool.last_action = json.loads(action_object_str)
            self.info_pool.action_history.append(json.loads(action_object_str))
            self.info_pool.summary_history.append(action_description)
            self.info_pool.action_outcomes.append(outcome)  # no change
            self.info_pool.error_descriptions.append(error_description)
            return (
                global_state,
                converted_action.action_code,
                True,
                None,
                True,
                final_path1,
                None,
            )

        try:
            if len(self.info_pool.action_history) >= args.max_trajectory_length - 1:
                converted_action.action_code = "FAIL"
            obs, env_reward, env_done, env_info = env.step(
                converted_action.action_code, self.wait_after_action_seconds
            )

        except Exception as e:
            print("Failed to execute action.")
            print(str(e))
            self.info_pool.last_action = json.loads({"action": "invalid"})
            self.info_pool.action_history.append({"action": "invalid"})
            self.info_pool.summary_history.append(action_description)
            self.info_pool.action_outcomes.append("C")  # no change
            self.info_pool.error_descriptions.append(
                f"Failed to execute the action: {converted_action}"
            )
            return (
                global_state,
                converted_action.action_code,
                False,
                None,
                False,
                final_path1,
                None,
            )

        print("Done action execution.\n")
        self.info_pool.last_action = json.loads(action_object_str)

        after_screenshot = obs["screenshot"]

        final_path2 = os.path.join(
            self.artifact_dir,
            str(relative_part),
            f"{step_idx}_after.png",
        )
        with open(final_path2, "wb") as f:  # "wb" means write binary.
            f.write(after_screenshot)

        print("\n### Reflector ... ###\n")
        if converted_action.action_type != "answer":
            action_reflection_start_time = time.time()

            # --- Build action_execute_dict in the format required by visual_action. ---
            action_execute_dict = {
                "action_type": converted_action.action_type,
                "action_code": converted_action.action_code,
                "x": converted_action.x,
                "y": converted_action.y,
                "text": converted_action.text,
                "clear": converted_action.clear,
                "time": converted_action.time,
                "value": converted_action.value,
                "key_list": converted_action.key_list,
            }
            self.info_pool.last_action_execute = action_execute_dict

            # --- Stage 1: analyze differences between before/after screenshots. ---
            before_img_path = os.path.join(example_result_dir, f"step_{step_idx}.png")
            after_img_path = final_path2  # Saved earlier.

            # 1. Use get_clean_diff_boxes to detect screenshot differences and produce red-box annotations.
            pre_img_mark, cur_img_mark, boxes_coordinate = get_clean_diff_boxes(
                before_img_path, after_img_path
            )
            cur_img_mark_path = save_image_from_cv2(cur_img_mark)

            # 2. Use visual_action to annotate the before screenshot with the action location.
            processed_img, coordinate = visual_action(
                action_execute_dict, before_img_path
            )

            # 3. Draw the same red difference boxes on processed_img.
            import cv2

            for idx, (x1, y1, x2, y2) in enumerate(boxes_coordinate):
                label = str(idx + 1)
                cv2.rectangle(processed_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )

                label_bg_x1 = x1
                label_bg_y1 = (
                    y1 - label_h - baseline if y1 - label_h - baseline > 0 else y1
                )
                label_bg_x2 = x1 + label_w + 4
                label_bg_y2 = label_bg_y1 + label_h + baseline

                cv2.rectangle(
                    processed_img,
                    (label_bg_x1, label_bg_y1),
                    (label_bg_x2, label_bg_y2),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    processed_img,
                    label,
                    (label_bg_x1 + 2, label_bg_y2 - baseline),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

            processed_img_path = save_image_from_cv2(processed_img)

            # 3. Call reflector stage 1 to find differences.
            print("  [Stage 1] Reflector find_diff ...")
            reflector_diff = reflector.find_diff(
                processed_img_path=processed_img_path,
                cur_img_mark_path=cur_img_mark_path,
                boxes_coordinate=boxes_coordinate,
                action_execute=action_execute_dict,
                coordinate=coordinate,
            )

            # --- Stage 2: judge action success from the change description. ---
            if reflector_diff:
                # 4. Extract affected area IDs.
                affected_areas = extract_affected_areas(reflector_diff)
                affected_boxes = [
                    boxes_coordinate[i - 1]
                    for i in affected_areas
                    if 1 <= i <= len(boxes_coordinate)
                ]

                # 5. Re-annotate the before screenshot, keeping only affected regions.
                pre_img_mark_clean = draw_boxes_on_image(
                    before_img_path, affected_boxes, labels=affected_areas
                )
                pre_img_mark_clean_path = save_image_from_cv2(pre_img_mark_clean)

                # 5.5 Re-annotate the after screenshot, keeping only affected regions.
                after_img_mark_clean = draw_boxes_on_image(
                    after_img_path, affected_boxes, labels=affected_areas
                )
                after_img_mark_clean_path = save_image_from_cv2(after_img_mark_clean)

                # 6. Build the stage 2 prompt and call the reflector.
                # Use the operator's original response as the action description.
                judge_prompt = reflector.get_judge_prompt(
                    reflector_diff=reflector_diff,
                    action_description=operator_response,
                )

                print("  [Stage 2] Reflector judge_action ...")
                output_action_reflect, message_reflector = reflector.judge_action(
                    pre_img_mark_clean_path, after_img_mark_clean_path, judge_prompt
                )

                # Save complete two-stage reflector intermediate data into global_state.
                global_state["reflector"] = {
                    "name": "reflector",
                    "mode": "two_stage",
                    # Stage 1: find_diff.
                    "stage1_find_diff": {
                        "input_images": [processed_img_path, cur_img_mark_path],
                        "boxes_coordinate": boxes_coordinate,
                        "coordinate": coordinate,
                        "response": reflector_diff,
                    },
                    # Stage 1 post-processing.
                    "affected_areas": affected_areas,
                    "affected_boxes": affected_boxes,
                    # Stage 2: judge_action.
                    "stage2_judge_action": {
                        "input_images": [
                            pre_img_mark_clean_path,
                            after_img_mark_clean_path,
                        ],
                        "prompt": judge_prompt,
                        "response": output_action_reflect,
                        "messages": message_reflector,
                    },
                    # Compatibility with the legacy format.
                    "response": output_action_reflect,
                    "messages": message_reflector,
                }
            else:
                # find_diff returned None for an unsupported action_type; fall back to single-stage reflection.
                print("  [Fallback] Using original single-stage reflector ...")
                prompt_action_reflect = reflector.get_prompt(self.info_pool)
                output_action_reflect, message_reflector = reflector.predict(
                    prompt_action_reflect, [before_screenshot, after_screenshot]
                )

                global_state["reflector"] = {
                    "name": "reflector",
                    "mode": "single_stage",
                    "input_images": [before_img_path, after_img_path],
                    "messages": message_reflector,
                    "response": output_action_reflect,
                }

            # Parse the result; two-stage and single-stage outputs share the outcome format.
            if reflector_diff:
                parsed_result_action_reflect = reflector.parse_judge_response(
                    output_action_reflect
                )
            else:
                parsed_result_action_reflect = reflector.parse_response(
                    output_action_reflect
                )

            outcome, error_description, progress_status = (
                parsed_result_action_reflect["outcome"],
                parsed_result_action_reflect["error_description"],
                parsed_result_action_reflect["progress_status"],
            )
            action_reflection_end_time = time.time()

            if (
                "A" in outcome
            ):  # Successful. The result of the last action meets the expectation.
                action_outcome = "A"
            elif (
                "B" in outcome
            ):  # Failed. The last action results in a wrong page. I need to return to the previous state.
                action_outcome = "B"
            elif "C" in outcome:  # Failed. The last action produces no changes.
                action_outcome = "C"
            else:
                raise ValueError("Invalid outcome:", outcome)

        print("\n\nAction reflection outcome: " + action_outcome)
        print("Action reflection error description: " + error_description)
        print("Action reflection progress status: " + progress_status, "\n")

        self.info_pool.action_history.append(json.loads(action_object_str))
        self.info_pool.summary_history.append(action_description)
        self.info_pool.action_outcomes.append(action_outcome)
        self.info_pool.error_descriptions.append(error_description)
        self.info_pool.progress_status = progress_status
        self.info_pool.progress_status_history.append(progress_status)

        return (
            global_state,
            converted_action,
            True,
            env_reward,
            env_done,
            final_path1,
            final_path2,
        )
