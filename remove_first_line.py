import os

def remove_first_line_from_tsv(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".tsv"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()
            with open(file_path, 'w') as file:
                file.writelines(lines[1:])

# Specify the directory containing the TSV files
directory = './transcripts'
remove_first_line_from_tsv(directory)

#0000f47c22	මහවැලි ගඟට ගොස් ආපසු එන ගමනේදී