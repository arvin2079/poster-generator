import sys
import os
import time

import pyfiglet
from rich.text import Text
from rich.table import Table
from rich.panel import Panel
from rich.align import Align
from rich.padding import Padding
from rich.console import Console

from api import (
    initialize,
    initialize_file_structure,
    export_all,
    get_posters_with_status,
    set_gift_taker_position,
    search_and_insert_posters,
    set_only_message_position,
    set_only_qrcode_position,
    set_information_box_position,
    set_message_and_qrcode_position,
    set_primary_handwritten_text_position,
    set_secondary_handwritten_text_position,
)


# if hasattr(sys, "_MEIPASS"):
#     fonts_path = os.path.join(sys._MEIPASS, "pyfiglet", "fonts")
# else:
#     fonts_path = os.path.join(os.path.dirname(pyfiglet.__file__), "fonts")


# Initialize the console
console = Console()


def print_welcome_banner():
    figlet_text = pyfiglet.figlet_format("mini-PSD")
    large_text = Text(figlet_text, justify="center", style="bold blue")
    description_text = Text(
        "photoshop-like app to automate Tree poster generation",
        justify="center",
        style="white",
    )
    combined_text = large_text + description_text

    panel = Panel(Padding(Align.center(combined_text), 3), expand=True, width=900)
    console.print(panel)


def show_main_menu(menu: dict):
    console.print(Padding("Main Menu:", (0, 0, 1, 0)))
    for i, value in enumerate(menu):
        message = Padding(f"[blue][{i}][/blue] : {value[0]}", (0, 0, 1, 2))
        console.print(message)


def int_input(msg, int_choices: list, error_msg: str):
    while True:
        input_value = console.input(
            " ".join([msg.strip() + " ([bold red]b[/bold red] to go back): "])
        )

        selected_no = None
        break_outer = input_value.lower() == "b"
        if break_outer:
            break

        try:
            selected_no = int(input_value)
            if selected_no not in int_choices:
                raise ValueError
            break
        except ValueError:
            console.print(error_msg, style="red")

    return selected_no, break_outer


def reload_posters():
    with console.status("searching for posters...", spinner="dots"):
        time.sleep(1)
        search_and_insert_posters()
        console.print("Posters list Reloaded successfully :white_check_mark:")


def show_posters_menu():

    def print_table():
        posters = get_posters_with_status()
        table = Table(title="posters")
        table.add_column("Index")
        table.add_column("Tree type")
        table.add_column("File Name")
        table.add_column("Gift Taker [green][1][/green]")
        table.add_column("Info Box [green][2][/green]")
        table.add_column("Primary Handwritten [green][3][/green]")
        table.add_column("Secondary Handwritten [green][4][/green]")
        table.add_column("Only Message [green][5][/green]")
        table.add_column("Only Qrcode [green][6][/green]")
        table.add_column("Qrcode And Message [green][7][/green]")

        for index in posters.keys():
            table.add_row(
                f"[blue]{index}[/blue]",
                posters[index]["tree_type"],
                posters[index]["image_name"],
                (
                    ":white_heavy_check_mark:"
                    if posters[index]["has_gift_taker_pos"]
                    else ":x:"
                ),
                (
                    ":white_heavy_check_mark:"
                    if posters[index]["has_info_box_pos"]
                    else ":x:"
                ),
                (
                    ":white_heavy_check_mark:"
                    if posters[index]["has_primary_handwritten_pos"]
                    else ":x:"
                ),
                (
                    ":white_heavy_check_mark:"
                    if posters[index]["has_secondary_handwritten_pos"]
                    else ":x:"
                ),
                (
                    ":white_heavy_check_mark:"
                    if posters[index]["only_message_pos"]
                    else ":x:"
                ),
                (
                    ":white_heavy_check_mark:"
                    if posters[index]["only_qrcode_pos"]
                    else ":x:"
                ),
                (
                    ":white_heavy_check_mark:"
                    if posters[index]["qrcode_and_message_pos"]
                    else ":x:"
                ),
            )

        console.print(table)

    print_table()

    posters = get_posters_with_status()

    while True:
        row_no, break_loop = int_input(
            "enter the [blue]row[/] number", posters.keys(), "choose from index column."
        )
        if break_loop:
            break

        column_no, break_loop = int_input(
            "enter [blue]column[/] number",
            range(1, 8),
            "choose from [blue bold]1[/] to [blue bold]7[/]",
        )
        if break_loop:
            continue

        if row_no and column_no:
            position_setters_list = [
                set_gift_taker_position,
                set_information_box_position,
                set_primary_handwritten_text_position,
                set_secondary_handwritten_text_position,
                set_only_message_position,
                set_only_qrcode_position,
                set_message_and_qrcode_position,
            ]

            position_setters_list[column_no - 1](row_no)

            print_table()


if __name__ == "__main__":
    print_welcome_banner()

    with console.status("initiating database...", spinner="monkey"):
        initialize()
        time.sleep(3)

    main_menu = [
        ["Initiate files structure", initialize_file_structure],
        ["Reload posters", reload_posters],
        ["Show posters (status & position configurations)", show_posters_menu],
        ["Export", export_all],
    ]

    while True:
        show_main_menu(main_menu)

        selected_no, break_loop = int_input(
            "input [bold blue]number[/]",
            list(range(len(main_menu))),
            f"choose from [blue bold]0[/] to [blue bold]{len(main_menu) - 1}[/]",
        )
        if break_loop:
            break

        # try:
        main_menu[selected_no][1]()
        # except Exception as e:
        #     console.print(str(e), style="red")

        console.rule()
