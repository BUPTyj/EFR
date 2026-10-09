from __future__ import annotations

import json
import re
import base64
import uuid
import cv2
import numpy as np
import requests
import os
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime

load_dotenv()


def read_trajectory_json(file_path):
    # Check whether the file exists.
    if not os.path.exists(file_path):
        print(f"File does not exist: {file_path}")
        return

    try:
        # Open and read the JSON file.
        # Use UTF-8 to avoid decoding issues in saved trajectories.
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)  # Load the JSON file into Python objects.

        print("Read trajectory JSON successfully.")
        return data

    except json.JSONDecodeError:
        print(f"Invalid JSON file: {file_path}")
    except Exception as e:
        print(f"Failed to read trajectory JSON: {e}")


def download_image(source):
    """
    Load an image from a URL or local file path.

    Args:
        source: Image URL or local file path.

    Returns:
        OpenCV image object.
    """
    # Check whether the source is a URL.
    if source.startswith(("http://", "https://")):
        # Download image data from the URL.
        resp = requests.get(source, stream=True).raw
        return cv2.imdecode(
            np.asarray(bytearray(resp.read()), dtype="uint8"), cv2.IMREAD_COLOR
        )
    else:
        # Read the image from a local path.
        return cv2.imread(source, cv2.IMREAD_COLOR)


def save_image_from_cv2(image: np.ndarray, folder: str = "reflector"):
    """
    Save an OpenCV image under folder/date in the current working directory.
    :param image: NumPy array loaded or processed by cv2.
    :param folder: Local target folder.
    :return: Absolute path to the saved image, or None on failure.
    """
    if image is None:
        return None

    # Build a date-based output directory and random file name.
    date_str = datetime.now().strftime("%Y%m%d")
    unique_id = uuid.uuid4().hex[:8]

    base_dir = os.getcwd()
    target_dir = os.path.join(base_dir, folder, date_str)
    os.makedirs(target_dir, exist_ok=True)

    file_name = f"{unique_id}.png"
    file_path = os.path.join(target_dir, file_name)

    # Save the image locally.
    success = cv2.imwrite(file_path, image)
    if not success:
        print("Failed to save image.")
        return None

    # Return the absolute path.
    return os.path.abspath(file_path)


def load_image_as_base64(image_path):
    """
    Load an image as a base64 data URL from a URL or local path.

    Args:
        image_path: Image URL or local file path.

    Returns:
        A data:image/png;base64,... string, or None on failure.
    """
    try:
        if image_path.startswith(("http://", "https://")):
            response = requests.get(image_path, timeout=15)
            response.raise_for_status()
            image_bytes = response.content
        else:
            if not os.path.exists(image_path):
                print(f"Image does not exist: {image_path}")
                return None
            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()

        img_base64 = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        print(f"Failed to convert image to base64: {e}")
        return None


def robust_json_loads(raw_str):
    if not raw_str:
        return None

    # Remove Markdown code fences such as ```json ... ```.
    clean_str = re.sub(r"```json\s*|```", "", raw_str).strip()

    # Fix Python-style booleans and nulls with conservative token replacements.
    # Regex token matching is safer than direct string replacement here.
    clean_str = re.sub(r":\s*True\b", ": true", clean_str)
    clean_str = re.sub(r":\s*False\b", ": false", clean_str)
    clean_str = re.sub(r":\s*None\b", ": null", clean_str)

    try:
        # First parse attempt.
        # strict=False permits control characters inside strings.
        return json.loads(clean_str, strict=False)
    except json.JSONDecodeError:
        # If parsing fails, escape invalid backslashes.
        # A backslash not followed by a standard JSON escape is doubled.
        # Standard JSON escapes include quote, backslash, slash, b, f, n, r, t, and u.
        clean_str = re.sub(r'\\(?![\\"/bfnrtu])', r"\\\\", clean_str)

        try:
            return json.loads(clean_str, strict=False)
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed after cleanup: {e}")
            print(f"Raw response content: {raw_str}")
            return None


def llmCall(model: str, msg: list, tp: str):
    if OpenAI is None:
        raise ImportError("llmCall requires the 'openai' package.")
    if tp == "ali":
        api_key = os.getenv("OPENAI_ALI_API_KEY")
        base_url = os.getenv("OPENAI_ALI_URL")
    elif tp == "local":
        api_key = "EMPTY"
        base_url = "http://127.0.0.1:8000/v1"
    else:
        return ""

    user_client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    response = user_client.chat.completions.create(
        model=model,
        messages=msg,
        max_tokens=1500,
        top_p=0.9,
        temperature=0,
        extra_body={
            "repetition_penalty": 1.05,
            "top_k": 50,
            "vl_high_resolution_images": True,
            "enable_thinking": False,
        },
        # , "vl_high_resolution_images":True
    )
    usage = response.usage

    print(f"Prompt tokens: {usage.prompt_tokens}")
    print(f"Completion tokens: {usage.completion_tokens}")
    print(f"Total tokens: {usage.total_tokens}")
    return response.choices[0].message.content


def process_gui_diff_pair(
    url_before, url_after, blur_ksize=(51, 51), threshold=25, merge_dist=30
):
    """
    Detect image differences and merge nearby changed regions.

    Args:
    - merge_dist: Merge distance. Larger values connect farther regions.
    """

    # Download and decode the images.
    def download_image(url):
        resp = requests.get(url, stream=True).raw
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        return cv2.imdecode(image, cv2.IMREAD_COLOR)

    img1 = download_image(url_before)
    img2 = download_image(url_after)

    # Align image sizes.
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Compute the initial difference mask.
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Method A: erode small noise.
    # Remove isolated pixel noise, such as tiny antialiasing differences.
    noise_kernel = np.ones((2, 2), np.uint8)
    mask = cv2.erode(mask, noise_kernel, iterations=1)

    # Merge nearby regions with a large morphological closing operation.
    # Use a larger kernel to make neighboring areas touch, then erode slightly.
    merge_kernel = np.ones((merge_dist, merge_dist), np.uint8)
    # MORPH_CLOSE connects gaps while preserving the main outline.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, merge_kernel)

    # Dilate once more to ensure connection areas are fully covered.
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)

    # Blur and composite unchanged regions.
    blur1 = cv2.GaussianBlur(img1, blur_ksize, 0)
    blur2 = cv2.GaussianBlur(img2, blur_ksize, 0)

    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
    res1 = (img1 * mask_3ch + blur1 * (1 - mask_3ch)).astype(np.uint8)
    res2 = (img2 * mask_3ch + blur2 * (1 - mask_3ch)).astype(np.uint8)

    return res1, res2


def get_clean_diff_boxes(
    url1, url2, threshold=25, merge_dist=40, small_limit=100, min_density=0.05
):
    img1, img2 = download_image(url1), download_image(url2)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    gray1, gray2 = (
        cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY),
    )
    diff = cv2.absdiff(gray1, gray2)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((1, 1), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 2 and h > 2:  # Filter out tiny regions.
            if y + h > 28:  # Ignore the top system status bar area.
                rx = max(x - 5, 0)
                ry = max(y - 5, 0)
                rx2 = min(x + w + 5, img1.shape[1])
                ry2 = min(y + h + 5, img1.shape[0])
                rects.append([rx, ry, rx2 - rx, ry2 - ry])  # [x, y, w, h] format.

    # --- Merge logic with blank-area density checks ---
    def get_union(r1, r2):
        x, y = min(r1[0], r2[0]), min(r1[1], r2[1])
        w = max(r1[0] + r1[2], r2[0] + r2[2]) - x
        h = max(r1[1] + r1[3], r2[1] + r2[3]) - y
        return [x, y, w, h]

    def calc_mask_density(r1, r2):
        """
        Calculate the changed-pixel density of the merged region on the mask.
        This uses real pixel differences and is more reliable than geometry alone.

        Returns:
            Ratio of changed pixels inside the merged region (0.0 to 1.0).
        """
        union = get_union(r1, r2)
        x, y, w, h = union
        roi = mask[y : y + h, x : x + w]
        if roi.size == 0:
            return 0.0
        return np.count_nonzero(roi) / roi.size

    def should_merge(r1, r2, loose_dist, size_limit, min_density):
        # Configuration parameters.
        TIGHT_DIST = 10  # Pixel threshold for regions considered very close.
        # ---------------

        # Compute physical gaps on the X and Y axes.
        x_gap = max(0, r1[0] - (r2[0] + r2[2]), r2[0] - (r1[0] + r1[2]))
        y_gap = max(0, r1[1] - (r2[1] + r2[3]), r2[1] - (r1[1] + r1[3]))

        # Handle regions that are very close to each other.
        if x_gap <= TIGHT_DIST and y_gap <= TIGHT_DIST:
            # Even very close regions must satisfy the merged density check.
            if calc_mask_density(r1, r2) < min_density:
                return False
            return True

        is_small_block = (
            r1[2] < size_limit
            or r1[3] < size_limit
            or r2[2] < size_limit
            or r2[3] < size_limit
        )

        if is_small_block and (x_gap <= loose_dist and y_gap <= loose_dist):
            # Small-region merges also need to satisfy the density check.
            if calc_mask_density(r1, r2) < min_density:
                return False
            return True

        return False

    changed = True
    while changed:
        changed, final_rects = False, []
        while rects:
            curr = rects.pop(0)
            merged = False
            for i in range(len(final_rects)):
                # print(curr, final_rects[i], should_merge(curr, final_rects[i], merge_dist, small_limit, min_density))
                if should_merge(
                    curr, final_rects[i], merge_dist, small_limit, min_density
                ):
                    final_rects[i] = get_union(curr, final_rects[i])
                    merged = True
                    changed = True
                    break
            # print("----------------------------------------------")
            if not merged:
                final_rects.append(curr)
        rects = final_rects

    # Sort top-to-bottom, then left-to-right.
    rects.sort(key=lambda r: (r[1] // 20, r[0]))

    padding = 0

    def _draw_boxes(img, rects_list):
        """Draw red boxes and index labels on an image; rects_list uses [x, y, w, h]."""
        res = img.copy()
        out_boxes = []
        for idx, (x, y, w, h) in enumerate(rects_list):
            label = str(idx + 1)
            pt1_x = max(0, x - padding)
            pt1_y = max(0, y - padding)
            pt2_x = min(img.shape[1], x + w + padding)
            pt2_y = min(img.shape[0], y + h + padding)
            cv2.rectangle(res, (pt1_x, pt1_y), (pt2_x, pt2_y), (0, 0, 255), 2)

            out_boxes.append([x, y, x + w, y + h])  # Return [x1, y1, x2, y2].

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (label_w, label_h), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )

            label_bg_x1 = x
            label_bg_y1 = y - label_h - baseline if y - label_h - baseline > 0 else y
            label_bg_x2 = x + label_w + 4
            label_bg_y2 = label_bg_y1 + label_h + baseline

            cv2.rectangle(
                res,
                (label_bg_x1, label_bg_y1),
                (label_bg_x2, label_bg_y2),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                res,
                label,
                (label_bg_x1 + 2, label_bg_y2 - baseline),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
        return res, out_boxes

    res_img1, _ = _draw_boxes(img1, rects)
    res_img2, boxes = _draw_boxes(img2, rects)

    return res_img1, res_img2, boxes


def draw_boxes_on_image(img_source, boxes, labels=None):
    """
    Draw red annotation boxes using the same style as get_clean_diff_boxes.

    Args:
        img_source: Image URL or local file path.
        boxes: Boxes in the get_clean_diff_boxes output format, [x1, y1, x2, y2].
        labels: Optional custom labels, such as original indices.
                    Defaults to sequential labels 1, 2, 3, ...

    Returns:
        OpenCV image object as a BGR NumPy array.
    """
    image = download_image(img_source)
    res_img = image.copy()

    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        label = str(labels[idx]) if labels else str(idx + 1)

        # Draw the red rectangle.
        cv2.rectangle(res_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Draw the index label with a black background and white text.
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (label_w, label_h), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        label_bg_x1 = x1
        label_bg_y1 = y1 - label_h - baseline if y1 - label_h - baseline > 0 else y1
        label_bg_x2 = x1 + label_w + 4
        label_bg_y2 = label_bg_y1 + label_h + baseline

        cv2.rectangle(
            res_img,
            (label_bg_x1, label_bg_y1),
            (label_bg_x2, label_bg_y2),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            res_img,
            label,
            (label_bg_x1 + 2, label_bg_y2 - baseline),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return res_img


def visual_action(action: dict, img_url: str):
    action_type = action["action_type"]
    action_code = action["action_code"]
    image = download_image(img_url)

    one_dot = ["click", "double_click", "right_click", "type", "scroll"]
    if action_type in one_dot:
        x = action["x"]
        y = action["y"]
        x = int(round(float(x)))
        y = int(round(float(y)))
        cv2.circle(image, (x, y), 20, (255, 191, 0), 2)
        return image, [[x, y], [0, 0]]
    elif action_type == "drag":
        move_match = re.search(r"moveTo\((.*?), (.*?)\)", action_code)
        drag_match = re.search(
            r"dragTo\((.*?), (.*?)\s*,", action_code
        )  # Include the drag duration argument in the regex.
        if move_match and drag_match:
            x1, y1 = (
                int(round(float(move_match.group(1)))),
                int(round(float(move_match.group(2)))),
            )
            x2, y2 = (
                int(round(float(drag_match.group(1)))),
                int(round(float(drag_match.group(2)))),
            )
            start_pt = (x1, y1)
            end_pt = (x2, y2)

            cv2.arrowedLine(image, start_pt, end_pt, (0, 255, 255), 3, tipLength=0.1)
            # Draw a hollow circle at the start and a filled circle at the end.
            cv2.circle(image, start_pt, 8, (0, 255, 255), 2)
            cv2.circle(image, end_pt, 8, (0, 255, 255), -1)
            return image, [[x1, y1], [x2, y2]]

    return image, [[0, 0], [0, 0]]


def extract_affected_areas(reflector_diff: str) -> list:
    """
    Extract the Affected Areas [IDs] section from reflector_diff.
    Return an empty list when reflector_diff is None or no IDs are found.

    Args:
        reflector_diff: Text returned by reflector_find_diff.

    Returns:
        List of affected area IDs, for example [1, 2].
    """
    match = re.search(r"\*\*Affected Areas \[IDs\]\*\*:\s*(.*)", reflector_diff)
    if not match:
        return []

    value = match.group(1).strip()

    # Return an empty list for None.
    if value.lower() == "none":
        return []

    # Extract numbers from a bracketed list, such as [1, 2].
    numbers = re.findall(r"\d+", value)
    return [int(n) for n in numbers]


