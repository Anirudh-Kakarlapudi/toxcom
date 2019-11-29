import pandas as pd
import re


def clean_text(comment, punct_num):
    """ Converts the comment into the lower case and removes punctuations

    Arguments:
        comment(str):
            A string from which the punctuations and special characters are to
            be removed
        punct_num(int):
            Removes all the punctuations and special characters if punct_num is 0
            or removes all other characters and timestamps except comma, fullstop,
            exclamation, dollar, apostrophe, question mark if punct_num is 1

    Returns:
        (str):
            A string with punctuations removed
    """
    comment_lower = comment.lower()
    escape_remov = re.sub('\\n', ' ', comment_lower)

    if punct_num == 0:
        return re.sub(r'[^\w\s]', r'', escape_remov)

    elif punct_num == 1:
        non_ascii_remov = ''.join([st for st in escape_remov if ord(st) < 127])
        brack_remov = re.sub(r'[\(\)\[\]\{\}]', r' ', non_ascii_remov)
        time_remov = re.sub(r'(\d\d\:)+(\d\d)*', r' ', brack_remov)
        colon_remov = re.sub(r'[\;\:]', r' ', time_remov)
        slash_remov = re.sub(r'[\/\\]', r' ', colon_remov)
        oper_remov = re.sub(
            r'[\+\-\_\*\^\%\#\@\|\`\~\=\&\<\>"]', r' ', slash_remov)
        extra_space_remov = re.sub(r' +', r' ', oper_remov)

        # adding a space for dot, comma, exclamation and question occurrence
        space_dot = re.sub(r'\.+', r' .', extra_space_remov)
        comma_dot = re.sub(r'\,+', r' ,', space_dot)
        excla_dot = re.sub(r'\!+', r' !', comma_dot)
        quest_dot = re.sub(r'\?+', r' ?', excla_dot)
        return quest_dot


def clean_dataframe(df, punct_num, target_col):
    """ Converts a column in dataframe into the lower case and removes punctuations

    Arguments:
        df(dataframe):
            A dataframe with raw text.

        punct_num(int):
            Removes all the punctuations and special characters if punct_num is 0
            or removes all other characters and timestamps except comma, fullstop,
            exclamation, dollar, apostrophe, question mark if punct_num is 1
        target_col(str):
            Name of the column in dataframe with raw text

    Returns:
        (dataframe):
            Returns a dataframe with cleaned text added as a column
    """

    df['cleaned_text'] = df[target_col].apply(
        lambda x: clean_text(x, punct_num))
    return df