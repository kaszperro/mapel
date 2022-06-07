import pandas as pd
from pandas import DataFrame


def convert_csv_to_latex(path, columns: dict, bold_columns: dict):
    df = pd.read_csv(path, sep=',')
    df = df.rename(columns=columns)
    df = df[columns.values()]

    for col in df.columns.get_level_values(0).unique():
        if col in bold_columns:
            df[col] = df[[col]].apply(func=lambda data: bold_extreme_values(data, max_=bold_columns[col]), axis=0)

    df['Algorithm name'] = df['Algorithm name'].apply(replace_underscore)
    text = df.to_latex(escape=False)
    output = f"""
\\begin{{table}}[ht]
\\centering
\\resizebox{{\\columnwidth}}{{!}}{{
    {text}
     }}
\\caption{{Caption below table.}}
\\label{{tab:caption}}
\\end{{table}}%
    """
    print(output)


def bold_extreme_values(data, format_string="%.3f", max_=True):
    if max_:
        extrema = data != data.max()
    else:
        extrema = data != data.min()
    bolded = data.apply(lambda x: "\\textbf{%s}" % format_string % x)
    formatted = data.apply(lambda x: format_string % x)
    return formatted.where(extrema, bolded)


def replace_underscore(string):
    return string.replace('_', '\\_')


def print_latex_mallows():
    convert_csv_to_latex(
        'experiments/only-mallows/features/results.csv',
        {
            'name': 'Algorithm name',
            'mean_stability': 'Mean stability',
            'mean_distortion': 'Mean distortion',
            # 'max_distortion': 'Max distortion',
            # 'min_distortion': 'Min distortion',
            'mean_monotonicity': 'Mean monotonicity',
            # 'max_monotonicity': 'Max monotonicity',
            # 'min_monotonicity': 'Min monotonicity'
        },
        {
            'Mean stability': False,
            'Mean distortion': False,
            'Max distortion': False,
            'Min distortion': False,
            'Mean monotonicity': True,
            'Max monotonicity': True,
            'Min monotonicity': True
        }
    )

    convert_csv_to_latex(
        'only_mallows_running_times.csv',
        {
            'algorithm_name': 'Algorithm name',
            'mean_time': 'Mean running time',
        },
        {
            'Mean running time': False,
        }
    )


if __name__ == '__main__':
    print_latex_mallows()
