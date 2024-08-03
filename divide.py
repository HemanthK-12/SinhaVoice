import pandas as pd

def filter_tsv_by_first_two_chars(input_file):
    df = pd.read_csv(input_file, delimiter='\t')
    df=df.drop(df.columns[1], axis=1)
    unique_prefixes = df.iloc[:, 0].str[:2].unique()
    c=0;
    for prefix in unique_prefixes:
        temp_df = df[df.iloc[:, 0].str.startswith(prefix)]
        output_file_name = f'transcripts/{prefix}.tsv'
        temp_df.to_csv(output_file_name, sep='\t', index=False)
input_file = 'utt_spk_text.tsv'

filter_tsv_by_first_two_chars(input_file)