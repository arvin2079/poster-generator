# Function to convert English numbers to Persian numbers
def convert_to_persian_numbers(s: str | int) -> str:
    s = str(s)
    persian_digits = {
        "0": "۰",
        "1": "۱",
        "2": "۲",
        "3": "۳",
        "4": "۴",
        "5": "۵",
        "6": "۶",
        "7": "۷",
        "8": "۸",
        "9": "۹",
    }
    return "".join(persian_digits.get(ch, ch) for ch in s)
