import os
import librosa
import csv
import sys

csv.field_size_limit(sys.maxsize)

def get_audio_duration(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return librosa.get_duration(y=y, sr=sr)

def create_tsv(input_file, output_file, output2_file):
       c=0
       with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile,open(output2_file, 'w', newline='') as missing :
        reader = csv.reader(infile, delimiter='\t')
        writer = csv.writer(outfile, delimiter='\t')
        writer2 = csv.writer(missing, delimiter='\t')
        writer.writerow(['audio_path', 'transcript', 'duration'])
        writer2.writerow(['audio_path', 'transcript'])
        for row in reader:
            audio_file = row[0]
            transcript = row[2]
            first_digit = audio_file[0]
            first_two_digits = audio_file[:2]
            audio_path = os.path.join('/media/switchblade/Windows/Users/heman/OneDrive/Desktop/BITS/Coding/Projects/SinhaVoice/audio',f'asr_sinhala_{first_digit}/asr_sinhala/data/{first_two_digits}', f'{audio_file}.flac')
            if os.path.exists(audio_path):
                writer.writerow([audio_path, transcript])
            else:
                writer2.writerow([audio_file, transcript])     
        
# Define the paths
input_file = '/media/switchblade/Windows/Users/heman/OneDrive/Desktop/BITS/Coding/Projects/SinhaVoice/all_transcripts.tsv'  # File containing audio file names and transcripts
output_file = 'lookup.tsv'  
output2_file='transcriptButNoAudio.tsv'          # Output TSV file

# Create the TSV file
create_tsv(input_file, output_file,output2_file)