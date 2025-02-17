from difflib import SequenceMatcher

from rapidfuzz import fuzz, process

from utils.date_tools import (
    persian_date_to_ymd,
    get_current_shamsi_year,
    get_current_season_persian,
)
from utils.numeric_tools import find_list_of_trees
from utils.string_tools import convert_to_persian_numbers

DEFAULT_SIMILARITY_RATIO = 0.8

english_keys = [
    "type",
    "tree name",
    "gift giver",
    "gift taker",
    "tree code",
    "message of tree",
    "special date explanation",
    "date day",
    "date month",
    "voice image attachment",
    "is ergent",
    "physical version",
    "insurance",
]

persian_keys = [
    "نوع درخت",
    "نام درخت",
    "نام هدیه دهنده",
    "نام هدیه گیرنده",
    "کد درخت",
    "پیام درخت",
    "آیا تاریخ خاصی به صورت نمادین به عنوان تاریخ کاشت در سند هدیه درخت شما ذکر شود؟",
    "روز",
    "ماه",
    "تمایل دارید صدا یا تصویری را به درخت ضمیمه کنید.",
    "آیا سفارش شما فوری است؟",
    "تمایل دارید نسخه فیزیکی سند هدیه درخت برای شما ارسال شود؟",
    "آیا تمایل به بیمه کردن نهال خود دارید؟",
]

tree_type_english_keys = [
    "sarv",
    "aras",
    "kaj",
    "arghavan",
    "daghdaghan",
]

tree_type_persian_keys = [
    "سرو",
    "ارس",
    "کاج",
    "ارغوان",
    "داغداغان",
]


# compare two string and return the similarity ratio
def compare_string(a, b):
    def is_junk(char):
        return char == " " or char == ":" or char == "."

    return SequenceMatcher(is_junk, a, b).ratio()


def find_closest_key(key, choices, threshold=DEFAULT_SIMILARITY_RATIO):
    best_match, score, i = process.extractOne(key, choices, scorer=fuzz.ratio)
    if score > threshold:
        return best_match, i


def json_file_content_to_dict(content):
    print(content)


def text_file_content_to_dict(content):
    splitted_parts = [i.strip() for i in content.split("\n") if i]

    key_value_pair_dict = {}

    key_value_pair_dict["type"] = tree_type_english_keys[
        find_closest_key(splitted_parts.pop(0), tree_type_persian_keys)[1]
    ]

    key = None

    while len(splitted_parts) > 0:
        item = splitted_parts.pop(0)

        if ":" in item:
            key, value = item.split(":")

            if not key:
                key, value = value, key

            key = english_keys[find_closest_key(key, persian_keys, threshold=70)[1]]

            if not key:
                raise ValueError("unknown key!")

            if value:
                key_value_pair_dict[key] = value.strip()
                key = value = None
            else:
                key_value_pair_dict[key] = splitted_parts.pop(0)

        else:
            if not key:
                raise ValueError("the text structure is destroyed!")
            key_value_pair_dict[key] += item

    return key_value_pair_dict


def highlight_text(text: str):
    return " ".join([f"$${word}$$" for word in text.split(" ")])


def generate_information_box_content(
    parameters_dict, highlight_function=highlight_text
):
    number_of_trees = len(find_list_of_trees(parameters_dict["tree code"]))

    content = " ".join(
        [
            (
                "نهال"
                if number_of_trees == 1
                else f"{convert_to_persian_numbers(number_of_trees)} نهال"
            ),
            (
                highlight_function(
                    tree_type_persian_keys[
                        tree_type_english_keys.index(parameters_dict["type"])
                    ]
                )
            ),
            ("به شماره" if parameters_dict["tree code"].isdigit() else "به شماره های"),
            (highlight_function(parameters_dict["tree code"])),
            "به نام",
            (
                highlight_function(
                    parameters_dict["tree name"]
                    if "tree name" in parameters_dict
                    else parameters_dict["gift taker"]
                )
            ),
        ]
    )

    if "gift giver" in parameters_dict:
        content = " ".join(
            [content, f"به سفارش {highlight_function(parameters_dict['gift giver'])}"]
        )

    if "date day" in parameters_dict and "date month" in parameters_dict:
        date_ymd = persian_date_to_ymd(
            month_name=parameters_dict["date month"],
            day=int(parameters_dict["date day"]),
        )
        content = " ".join([content, f"در تاریخ {highlight_function(date_ymd)}"])
    elif "date month" in parameters_dict:
        content = " ".join(
            [
                content,
                f"در ماه {highlight_function(parameters_dict['date month'])} {get_current_shamsi_year()}",
            ]
        )
    elif not "date day" in parameters_dict and not "date month" in parameters_dict:
        content = " ".join(
            [
                content,
                f"در {highlight_function(get_current_season_persian())} {highlight_function(get_current_shamsi_year())}",
            ]
        )

    content = " ".join([content, "به زمین هدیه شد."])

    return content
