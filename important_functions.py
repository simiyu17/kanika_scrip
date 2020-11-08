import datetime
import linecache
import os
import re
from pdf2image import convert_from_path
import xlsxwriter
import shutil

ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}


# Check that file being processed is valid for this script
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Move file to another directory
def copy_file(file_path, dst_dir):
    '''
    :param file_path: Path for Document to be moved
    :param dst_dir: Destination directory
    '''
    try:
        shutil.copy2(file_path, dst_dir)
    except OSError:
        print("Copying of the file %s failed" % file_path)
    else:
        print("{} File Copied to: {}".format(file_path, dst_dir))


# Delete file
def delete_file(file_path):
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except OSError:
            print("Deleting of the file %s failed" % file_path)


# Delete Directory
def delete_dir(dir_path):
    if os.path.exists(dir_path):
        try:
            delete_files = []
            delete_dirs = []
            for root, dirs, files in os.walk(dir_path):
                for f in files:
                    delete_files.append(os.path.join(root, f))
                for d in dirs:
                    delete_dirs.append(os.path.join(root, d))
            for f in delete_files:
                os.remove(f)
            for d in delete_dirs:
                os.rmdir(d)
            os.rmdir(dir_path)
        except OSError:
            pass


# Create a directory inside another directory
def create_folder(parent_dir, dir_to_create):
    '''
    :param parent_dir: Directory inside which we want to create another Directory
    :param dir_to_create: Directory To be created
    '''
    if os.path.exists(parent_dir) and not os.path.exists(f'{parent_dir}/{dir_to_create}'):
        try:
            os.makedirs(f'{parent_dir}/{dir_to_create}')
        except OSError:
            print(f'Creation of the directory {parent_dir}/{dir_to_create} failed')
        else:
            print(f'Successfully created the directory {parent_dir}/{dir_to_create}')


def the_given_date_is_valid(date_string):
    try:
        day, month, year = date_string.split('/')
        datetime.datetime(int(year), int(month), int(day))
        return True
    except ValueError:
        return False


def is_the_given_pan_no_is_valid(pan_card_no):
    regex = "[A-Z]{5}[0-9]{4}[A-Z]{1}"
    # Compile the ReGex
    p = re.compile(regex)
    # If the PAN Card number
    # is empty return false
    if pan_card_no is None:
        return False
    # matched the ReGex
    if re.search(p, pan_card_no) and len(pan_card_no) == 10:
        return True
    else:
        return False


# Create a .txt file for each processed file for verification
def create_text_file(file_path, destination_dir, file_text):
    file_name = os.path.basename(file_path).split('.')[0]
    f = open(f'{destination_dir}/{file_name}.txt', 'w+')
    f.write(file_text)
    f.close()
    return read_the_file(f'{destination_dir}/{file_name}.txt')


def read_the_file(file_path):
    f = open(file_path, 'r')
    date_line = -1
    num = '-'
    for i, line in enumerate(f):
        if the_given_date_is_valid(line):
            date_line = i
        if is_the_given_pan_no_is_valid(line.strip()):
            print(f'PAN********************{line.strip()}')
            num = f'{line.strip()}'
    f.close
    if date_line > -1:
        n = f'{linecache.getline(file_path, date_line - 3).strip()}'
        fn = f'{linecache.getline(file_path, date_line - 1).strip()}'
        dob = f'{linecache.getline(file_path, date_line + 1).strip()}'
        # num = f'{linecache.getline(file_path, date_line + 5).strip()}'
        delete_file(file_path)
        return n, fn, dob, num
    else:
        delete_file(file_path)
        return '-', '-', '-', '-'


def create_excel_output(df, output_dir):
    # Create an new Excel file and add a worksheet.
    workbook = xlsxwriter.Workbook(str(output_dir) + '/Output.xlsx')
    output = workbook.add_worksheet('Output')
    # Create a format to use in the merged range.
    merge_format = workbook.add_format(
        {'bold': 1, 'border': 1, 'align': 'center', 'valign': 'vcenter', 'text_wrap': True})
    # Add a bold format to use to highlight cells.
    bold = workbook.add_format({'bold': 1, 'border': 1})
    output.write('B2', 'HOLDER\'S NAME', merge_format)
    output.write('C2', 'FATHER\'S NAME', merge_format)
    output.write('D2', 'DATE OF BIRTH', merge_format)
    output.write('E2', 'PAN NUMBER', merge_format)
    i = 3
    for index, row in df.iterrows():
        output.write('B' + str(i), row['Name'], bold)
        output.write('C' + str(i), row['FName'], bold)
        output.write('D' + str(i), row['DOB'], bold)
        output.write('E' + str(i), row['PAM'], bold)
        i += 1
    output.set_column(1, 4, 40)
    workbook.close()


# Convert PDF file to Image
def convert_pdf_to_image_and_save(file_path, save_dir):
    file_name = os.path.basename(file_path).split('.')[0]
    pages = convert_from_path(file_path)
    for page in pages:
        page.save(f'{save_dir}/{file_name}.jpg', 'JPEG')


# Search a WHOLE word in a given text
def words_found_in_text(file_text, search_text):
    '''
    :param file_text: Text extracted from a document
    :param search_text: The keyword to be searched in the text
    :return:
    '''
    res_search = re.search(search_text, file_text, flags=re.IGNORECASE)
    if res_search:
        return True
    return False
