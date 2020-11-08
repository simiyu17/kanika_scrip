from os.path import expanduser
from text_extraction import *
import pandas as pd
import argparse

source_path = expanduser('~') + '/Downloads/kanika/source2'

parser = argparse.ArgumentParser()
parser.add_argument("--source_dir", help="1 This should be the source directory of files to be processed", type=str, required=False)
args = parser.parse_args()


def create_excel_file(file_source, tess_dir, excel_output_dir):
    count = 1
    f_list = os.listdir(file_source)
    number_files = len(f_list)
    data_df = pd.DataFrame()
    for file_path in os.listdir(file_source):
        if allowed_file(file_path):
            print(f'{str(count)}/{str(number_files)} Processing {file_source}/{file_path} ')
            # Extract Text from the file
            custom_config = r'--oem 3 -l eng+ind --psm 11 --tessdata-dir "' + tess_dir + '"'
            file_full_text = extract_text_in_file(f'{file_source}/{file_path}', custom_config)
            # print(file_full_text)
            n, fn, dob, num = create_text_file(f'{file_source}/{file_path}', excel_output_dir, file_full_text)
            data_df = data_df.append({'Name': n, 'FName': fn, 'DOB': dob, 'PAM': num}, ignore_index=True)
        else:
            print("{} not allowed. Only pdf, jpg, jpeg and  png".format(file_path))
        count += 1
    create_excel_output(data_df, excel_output_dir)


def process_files(source_dir):
    # Check if source folder and output folder both exist.
    if not os.path.exists(source_dir):
        print("Source directory {} does not exist".format(source_dir))
    else:
        # Reset tempfiles
        delete_dir(f'{os.getcwd()}/image_temp_dir')
        delete_dir(f'{os.getcwd()}/tempfiles')
        create_folder(os.getcwd(), 'image_temp_dir')
        create_folder(os.getcwd(), 'tempfiles')
        delete_file(f'{source_dir}/Output.xlsx')

        count = 1
        f_list = os.listdir(source_dir)
        number_files = len(f_list)
        for file_path in os.listdir(source_dir):
            if allowed_file(file_path):
                print(f'Preparing {str(count)}/{str(number_files)} Files')
                # Extract Text from the file
                if file_path.lower().endswith('pdf'):
                    convert_pdf_to_image_and_save(f'{source_dir}/{file_path}', f'{os.getcwd()}/image_temp_dir')
                else:
                    copy_file(f'{source_dir}/{file_path}', f'{os.getcwd()}/image_temp_dir')
            else:
                print("{} not allowed. Only pdf, jpg, jpeg and  png".format(file_path))
            count += 1
        print(f' ************************ DONE PREPARING FILES *************************')
        create_excel_file(f'{os.getcwd()}/image_temp_dir', f'{os.getcwd()}/tessdata', source_dir)
        # Remove tempfiles
        delete_dir(f'{os.getcwd()}/image_temp_dir')
        delete_dir(f'{os.getcwd()}/tempfiles')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    source_directory = source_path
    if args.source_dir:
        source_directory = args.source_dir
    process_files(source_directory)
