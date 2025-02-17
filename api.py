import os
import sys
import json
import random
import string

import cv2
from PIL import Image
import numpy as np

from roi import (
    select_square,
    select_rectangle,
    select_four_points,
)
from image_processing import (
    remove_background,
    imshow_in_screen_size,
    draw_quadrilateral_with_label,
    project_text_on_image,
    project_text_on_image_two_line,
    overlay_image_with_perspective,
)
from text_processing import (
    tree_type_english_keys,
    compare_string,
    generate_information_box_content,
    text_file_content_to_dict,
    DEFAULT_SIMILARITY_RATIO,
)
from database import (
    clean_posters,
    initiate_relations,
    get_poster_by_id,
    get_all_posters_data,
    insert_all_new_posters,
    upsert_gift_taker_pos,
    get_posters_join_on_positions,
    upsert_handwritten_pos,
    upsert_information_box_pos,
    upsert_qrcode_message_pos,
)

cwd = os.getcwd()

INPUT_DIR_PATH = os.path.join(cwd, "inputs")
EXPORTS_DIR_PATH = os.path.join(cwd, "exports")
POSTERS_DIR_PATH = os.path.join(cwd, "posters")
QRCODES_DIR_PATH = os.path.join(cwd, "qrcodes")


def initialize():
    _init_database()


def initialize_file_structure():
    _init_file_structure()


def _init_database():
    initiate_relations()


def _init_file_structure():
    """
    this function will initiate the file structures, including:
    - inputs  : for text inputs and their relating handwritten png
    - exports : for final exports result.
    - posters : all different types of posters are in different directories here
    - qrcodes : repository of QRcodes
    """

    for path in [INPUT_DIR_PATH, EXPORTS_DIR_PATH, POSTERS_DIR_PATH, QRCODES_DIR_PATH]:
        if not os.path.exists(path):
            os.mkdir(path)
            if path == POSTERS_DIR_PATH:
                for tree_type in tree_type_english_keys:
                    os.mkdir(os.path.join(POSTERS_DIR_PATH, tree_type))


def search_and_insert_posters():
    posters_type_dirs = [
        f
        for f in os.listdir(POSTERS_DIR_PATH)
        if os.path.isdir(os.path.join(POSTERS_DIR_PATH, f))
    ]

    posters = {}

    for dir_name in posters_type_dirs:
        posters_list = [
            f
            for f in os.listdir(os.path.join(POSTERS_DIR_PATH, dir_name))
            if f.endswith(("jpg", "png", "jpeg"))
        ]
        posters[dir_name] = posters_list

    clean_posters(posters)
    insert_all_new_posters(posters)


def gather_inputs():
    grouped_inputs = []
    free_inputs = []
    group = None
    for root, dirs, files in os.walk(INPUT_DIR_PATH):
        if root == INPUT_DIR_PATH:
            continue

        if len(files) > 0 and len(dirs) > 0:
            raise ValueError("wrong input structure!")

        if len(files) == 0:
            grouped_inputs.append([])
            group = root + "\\"

        if len(dirs) == 0:
            if group and group in root + "\\":
                grouped_inputs[-1].append(root)
            else:
                free_inputs.append(root)

    return free_inputs, grouped_inputs


# gather_inputs()


def extract_input_content(input_dir):
    txt_files = [file for file in os.listdir(input_dir) if file.endswith(".txt")]
    json_files = [file for file in os.listdir(input_dir) if file.endswith(".json")]

    if len(json_files) > 0 and len(txt_files) > 0:
        raise Exception(
            "cant have json and txt files both for on input in " + input_dir
        )
    elif len(json_files) == 0 and len(txt_files) == 0:
        raise Exception("no txt or json file provided in " + input_dir)
    elif len(json_files) > 1:
        raise Exception("more than one json file provided in " + input_dir)
    elif len(txt_files) > 1:
        raise Exception("more than one txt files provided in " + input_dir)

    if len(txt_files) > 0:
        with open(os.path.join(input_dir, txt_files[0]), "r", encoding="utf-8") as file:
            return file.read()
    if len(json_files) > 0:
        with open(
            os.path.join(input_dir, json_files[0]), "r", encoding="utf-8"
        ) as file:
            return json.load(file)


def extract_handwritten_image(input_dir):
    image_file_address = [
        file for file in os.listdir(input_dir) if file.endswith(("jpg", "png", "jpeg"))
    ][0]
    return os.path.join(input_dir, image_file_address)


def get_qrcode():
    if os.path.exists(QRCODES_DIR_PATH):
        qrcode_files_list = os.listdir(QRCODES_DIR_PATH)
        try:
            selected_qrcode_index = random.randint(0, len(qrcode_files_list) - 1)
            selected_qrcode = os.path.join(
                QRCODES_DIR_PATH, qrcode_files_list[selected_qrcode_index]
            )
            return selected_qrcode
        except ValueError:
            raise ValueError("no qrcode found")


def delete_qrcode(path):
    os.remove(path)


def export_all():

    def check_poster_status(status_dict):
        for _, status in status_dict.items():
            for value in status.values():
                if not value:
                    raise ValueError(
                        "please first set all element positions on every available posters then come to export!"
                    )

    def get_positions(poster):
        if (
            "gift taker" in content_dict
            and "tree name" in content_dict
            and compare_string(content_dict["gift taker"], content_dict["tree name"])
            > DEFAULT_SIMILARITY_RATIO
        ):
            gift_taker_position = next(
                (t for t in gift_takers if t[0] == poster[0]), None
            )
            if not gift_taker_position:
                raise ValueError(
                    "no gift taker position for poster with id=" + poster[0]
                )

        if "gift_taker_position" not in locals():
            gift_taker_position = None

        information_box_position = next(
            (t for t in information_boxes if t[0] == poster[0])
        )
        if not information_box_position:
            raise ValueError(
                "no information box position for poster with id=" + poster[0]
            )

        handwrittens_position = [t for t in handwrittens if t[0] == poster[0]]
        if not handwrittens_position:
            raise ValueError("no handwritten position for poster with id=" + poster[0])

        qrcodes_and_messages_position = [
            t for t in qrcodes_and_messages if t[0] == poster[0]
        ]
        if not qrcodes_and_messages_position:
            raise ValueError(
                "no qrcode or message position for poster with id=" + poster[0]
            )

        return (
            gift_taker_position,
            information_box_position,
            handwrittens_position,
            qrcodes_and_messages_position,
        )

    def fill_poster(poster, positions, handwritten_image_address):
        poster_address = os.path.join(POSTERS_DIR_PATH, *poster[1:])

        if getattr(sys, "frozen", False):
            # If running in a PyInstaller bundle, use _MEIPASS to get the temporary directory
            bundle_dir = sys._MEIPASS
        else:
            # If running in normal Python mode, use the current directory or relative path
            bundle_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the full path to the fonts directory
        font_path = os.path.join(bundle_dir, "fonts", "YEKAN.ttf")

        generated_info_box = generate_information_box_content(content_dict)
        handwritten_image = remove_background(handwritten_image_address)

        image = project_text_on_image_two_line(
            poster_address,
            generated_info_box,
            points=positions[1][1],
            font_path=font_path,
            line_spacing=60,
        )

        for p in positions[2]:
            image = overlay_image_with_perspective(
                base_image=image, overlay_image=handwritten_image, dst_points=p[2]
            )

        if (
            positions[0]
            and compare_string(content_dict["tree name"], content_dict["gift taker"])
            < 0.85
        ):
            image = project_text_on_image(
                image,
                "$$برای$$ " + content_dict["gift taker"],
                points=positions[0][1],
                font_path=font_path,
                font_size=80,
            )

        has_qrcode = (
            "voice image attachment" in content_dict
            and compare_string(content_dict["voice image attachment"], "خیر") < 0.7
        )
        has_message = "message of tree" in content_dict

        if has_qrcode and has_message:
            _, qrcode_position, message_position = positions[3][2]
            qrcode = get_qrcode()
            image = overlay_image_with_perspective(image, qrcode, qrcode_position)
            image = project_text_on_image(
                image,
                content_dict["message of tree"],
                points=message_position,
                font_path=font_path,
                font_size=70,
                word_spacing=30,
                text_alignment="centered",
                line_spacing=30,
            )
            delete_qrcode(qrcode)
        elif has_qrcode and not has_message:
            _, qrcode_position, message_position = positions[3][1]
            qrcode = get_qrcode()
            image = overlay_image_with_perspective(image, qrcode, qrcode_position)
            delete_qrcode(qrcode)
        elif not has_qrcode and has_message:
            _, qrcode_position, message_position = positions[3][0]
            image = project_text_on_image(
                image,
                content_dict["message of tree"],
                points=message_position,
                font_path=font_path,
                word_spacing=30,
                font_size=70,
                text_alignment="centered",
                line_spacing=100,
            )
        elif not has_qrcode and not has_message:
            pass

        return image

    posters_status_dict = get_posters_join_on_positions()
    check_poster_status(posters_status_dict)

    free_input_dirs, grouped_input_dirs = gather_inputs()
    posters, gift_takers, information_boxes, handwrittens, qrcodes_and_messages = (
        get_all_posters_data()
    )

    for input_dir in free_input_dirs:
        content = extract_input_content(input_dir)
        if isinstance(content, str):
            content_dict = text_file_content_to_dict(content)
        else:
            content_dict = content
        handwritten_image_address = extract_handwritten_image(input_dir)

        choices = [poster for poster in posters if poster[1] == content_dict["type"]]
        choosen_poster = random.choice(choices)

        positions = get_positions(choosen_poster)
        result = fill_poster(choosen_poster, positions, handwritten_image_address)
        result_file_name = (
            (
                content_dict["tree name"]
                if "tree name" in content_dict
                else content_dict["gift taker"]
            )
            + "_"
            + "".join(random.choices(string.ascii_letters + string.digits, k=7))
            + ".pdf"
        )
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        Image.fromarray(result).save(os.path.join(EXPORTS_DIR_PATH, result_file_name))
        print(result_file_name + " done.")

    for group in grouped_input_dirs:
        first_item = group[0]
        first_item_content_dict = text_file_content_to_dict(
            extract_input_content(first_item)
        )

        choices = [
            poster for poster in posters if poster[1] == first_item_content_dict["type"]
        ]

        if len(group) > len(choices):
            raise ValueError(
                "not enough posters of type " + first_item_content_dict["type"]
            )

        for input_dir, choosen_poster in zip(group, choices):
            content = extract_input_content(input_dir)
            content_dict = text_file_content_to_dict(content)
            handwritten_image_address = extract_handwritten_image(input_dir)
            positions = get_positions(choosen_poster)
            result = fill_poster(choosen_poster, positions, handwritten_image_address)
            result_file_name = (
                (
                    content_dict["tree name"]
                    if "tree name" in content_dict
                    else content_dict["gift taker"]
                )
                + "_"
                + "".join(random.choices(string.ascii_letters + string.digits, k=7))
                + ".pdf"
            )
            # cv2.imwrite(os.path.join(EXPORTS_DIR_PATH, result_file_name), result)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            Image.fromarray(result).save(
                os.path.join(EXPORTS_DIR_PATH, result_file_name)
            )
            print(result_file_name + " done.")


def check_for_enough_posters():
    pass  # TODO: check if eanough posters exists for a group


def get_posters_with_status():
    return get_posters_join_on_positions()


def set_gift_taker_position(poster_id: int):
    poster_record = get_poster_by_id(poster_id)
    image_path = os.path.join(POSTERS_DIR_PATH, *poster_record[1:])
    roi = select_rectangle(image_path)
    if np.all(roi):
        upsert_gift_taker_pos(poster_record[0], roi)


def set_information_box_position(poster_id: int):
    poster_record = get_poster_by_id(poster_id)
    image_path = os.path.join(POSTERS_DIR_PATH, *poster_record[1:])
    roi = select_rectangle(image_path)
    if np.all(roi):
        upsert_information_box_pos(poster_record[0], roi)


def set_primary_handwritten_text_position(poster_id: int):
    poster_record = get_poster_by_id(poster_id)
    image_path = os.path.join(POSTERS_DIR_PATH, *poster_record[1:])
    roi = select_four_points(image_path)
    if np.all(roi):
        upsert_handwritten_pos(poster_record[0], True, roi)


def set_secondary_handwritten_text_position(poster_id: int):
    poster_record = get_poster_by_id(poster_id)
    image_path = os.path.join(POSTERS_DIR_PATH, *poster_record[1:])
    roi = select_four_points(image_path)
    if np.all(roi):
        upsert_handwritten_pos(poster_record[0], False, roi)


def set_only_message_position(poster_id: int):
    poster_record = get_poster_by_id(poster_id)
    image_path = os.path.join(POSTERS_DIR_PATH, *poster_record[1:])
    roi = select_rectangle(image_path)
    if np.all(roi):
        upsert_qrcode_message_pos(poster_record[0], None, roi)


def set_only_qrcode_position(poster_id: int):
    poster_record = get_poster_by_id(poster_id)
    image_path = os.path.join(POSTERS_DIR_PATH, *poster_record[1:])
    roi = select_square(image_path)
    if np.all(roi):
        upsert_qrcode_message_pos(poster_record[0], roi, None)


def set_message_and_qrcode_position(poster_id: int):
    poster_record = get_poster_by_id(poster_id)
    image_path = os.path.join(POSTERS_DIR_PATH, *poster_record[1:])
    qrcode_roi = select_square(image_path)

    image_with_qrcode = draw_quadrilateral_with_label(image_path, qrcode_roi, "qrcode")
    message_roi = select_rectangle(image_with_qrcode)

    if np.all(qrcode_roi) and np.all(message_roi):
        upsert_qrcode_message_pos(poster_record[0], qrcode_roi, message_roi)
