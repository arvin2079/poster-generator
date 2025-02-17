import re


def find_list_of_trees(tree_codes):
    # Find all sequences of digits in the string
    integers = re.findall(r"\d+", tree_codes)
    # Convert the found sequences from strings to integers
    integers = sorted([int(i) for i in integers])

    if "ا" in tree_codes or "ل" in tree_codes or "ی" in tree_codes:
        while integers[-1] - 1 not in integers:
            integers.append(integers[-1] - 1)

    return sorted(integers)
