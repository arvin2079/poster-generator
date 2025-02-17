import jdatetime

from utils.string_tools import convert_to_persian_numbers


def persian_date_to_ymd(month_name: str, day: int, year: int = None) -> str:
    # Map Persian month names to their corresponding month numbers
    persian_months = {
        "فروردین": 1,
        "اردیبهشت": 2,
        "خرداد": 3,
        "تیر": 4,
        "مرداد": 5,
        "شهریور": 6,
        "مهر": 7,
        "آبان": 8,
        "آذر": 9,
        "دی": 10,
        "بهمن": 11,
        "اسفند": 12,
    }

    # Get the month number from the dictionary
    month = persian_months.get(month_name)

    if month is None:
        raise ValueError(f"Invalid Persian month name: {month_name}")

    if day > 31:
        raise ValueError(f"Invalid Persian day number: {day}")

    if not year:
        year = jdatetime.date.today().year

    # Format the date as a string in English numbers
    date_str = "%d/%d/%d" % (year, month, day)

    # Convert the English numbers to Persian numbers
    return convert_to_persian_numbers(date_str)


# Example usage
# print(persian_date_to_ymd("شهریور", 31))  # Output: '1400/07/15'


def get_current_shamsi_year():
    current_date = jdatetime.date.today()
    year_str = str(current_date.year)
    return convert_to_persian_numbers(year_str)


def get_current_season_persian():
    current_date = jdatetime.date.today()
    month = current_date.month

    if 1 <= month <= 3:
        return "بهار"  # Spring
    elif 4 <= month <= 6:
        return "تابستان"  # Summer
    elif 7 <= month <= 9:
        return "پاییز"  # Autumn
    elif 10 <= month <= 12:
        return "زمستان"  # Winter
