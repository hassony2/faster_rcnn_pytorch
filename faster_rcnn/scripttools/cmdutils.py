try:
    from termcolor import cprint
except ImportError:
    cprint = None


def confirm(message):
    """
    Ask user to enter Y or N (case-insensitive).

    Args:
        message (str): message to display
    Returns:
        True if the answer is Y.
    """
    answer = ""
    while answer not in ["y", "n"]:
        answer = input(message + " [Y/N]? ").lower()
    return answer == "y"


def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)
