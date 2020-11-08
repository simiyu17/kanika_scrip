from deskew import determine_skew
from image_utils import *
from important_functions import *


# Overal text extraction from a file
def extract_text_in_file(file_path, custom_config):
    file_text = text_from_license(file_path, custom_config)
    delete_file(file_path)
    return file_text


# Extract text from an image file
def text_from_image_file_original(file_path, custom_config):
    img = cv2.imread(file_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)
    angles = []
    for x1, y1, x2, y2 in lines[0]:
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)
    median_angle = np.median(angles)
    img_rotated = ndimage.rotate(img, median_angle)
    # cv2.imwrite('img_rotated.png', img_rotated)
    return pytesseract.image_to_string(img_rotated, config=custom_config)


# Extract text from an image file
def text_from_image_file(file_path):
    img = cv2.imread(file_path)
    deskew = correct_skew(img)
    adaptive_thresh = cv2.adaptiveThreshold(deskew, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 31, 2)
    # cv2.imwrite('dddddddd.png', adaptive_thresh)
    custom_config = r'--oem 3 -l eng+ind --psm 6'
    return pytesseract.image_to_string(adaptive_thresh, config=custom_config)


# Extract text from a licence size image file
def text_from_licence_image_file(file_path, custom_config):
    img = cv2.imread(file_path)
    img = resize_image_size(img)

    angle = determine_skew(get_grayscale(img))
    rotated = rotate(img, angle, (0, 0, 0))

    grayed = get_grayscale(rotated)
    # img_rotate_180_clockwise = cv2.rotate(img, cv2.ROTATE_180)
    # cv2.imwrite('dddddddd.png', img_rotate_180_clockwise)
    return pytesseract.image_to_string(grayed, config=custom_config)


# Extract text from a licence size image file
def text_from_licence_image_file_with_cropping1(file_path, custom_config):
    im = Image.open(file_path)
    background_removed = remove_single_color_background(im)
    background_removed.save(f'tempfiles/bg_rmd.png')
    img = cv2.imread(f'tempfiles/bg_rmd.png')
    img = resize_image_size(img)
    delete_file(f'tempfiles/bg_rmd.png')

    angle = determine_skew(get_grayscale(img))
    rotated = rotate(img, angle, (0, 0, 0))
    # cv2.imwrite('rotated.png', rotated)

    grayed = get_grayscale(rotated)
    # cv2.imwrite('grayed.png', grayed)
    # blurred = cv2.GaussianBlur(grayed, (5, 5), 0)
    blurred = cv2.medianBlur(grayed, 3)
    # blurred = cv2.bilateralFilter(grayed, 9, 75, 75)
    # blurred = cv2.threshold(grayed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # cv2.imwrite('blurred.png', blurred)

    return pytesseract.image_to_string(blurred, config=custom_config)


# Extract text from a licence size image file
def text_from_licence_image_file_with_cropping_small(file_path, custom_config):
    im = Image.open(file_path)
    background_removed = remove_single_color_background(im)
    background_removed.save(f'tempfiles/bg_rmd.png')
    img = cv2.imread(f'tempfiles/bg_rmd.png')
    img = resize_image_size(img)
    delete_file(f'tempfiles/bg_rmd.png')

    angle = determine_skew(get_grayscale(img))
    print(f'Angle of rotation ***************** {angle}')
    rotated = rotate(img, angle, (0, 0, 0))
    # cv2.imwrite('rotated.png', rotated)

    grayed = get_grayscale(rotated)
    # cv2.imwrite('grayed.png', grayed)
    # blurred = cv2.GaussianBlur(grayed, (5, 5), 0)
    blurred = cv2.medianBlur(grayed, 3)
    # blurred = cv2.bilateralFilter(grayed, 9, 75, 75)
    # blurred = cv2.threshold(grayed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # cv2.imwrite('blurred.png', blurred)

    return pytesseract.image_to_string(blurred, config=custom_config)


# Extract text from a licence size image file
def text_from_licence_image_file_with_cropping(file_path, custom_config):
    im = Image.open(file_path)
    # removing a background that surrounds the image
    background_removed = remove_single_color_background(im)
    background_removed.save(f'tempfiles/bg_rmd.png')
    img = cv2.imread(f'tempfiles/bg_rmd.png')
    # doubling the size of the image
    img = resize_image_size(img)
    delete_file(f'tempfiles/bg_rmd.png')

    angle = determine_skew(get_grayscale(img))
    rotated = rotate(img, angle, (0, 0, 0))
    # cv2.imwrite('rotated.png', rotated)

    rotated2 = deskew_image(img)
    # cv2.imwrite('rotated2.png', rotated2)

    # Scenario 1
    '''mask_removed = remove_water_mask(rotated)
    cv2.imwrite('mask_removed.png', mask_removed)
    grayed = binarization(mask_removed)
    cv2.imwrite('grayed.png', grayed)
    return pytesseract.image_to_string(grayed, config=custom_config)'''

    # Scenario 1
    alpha = 2.0
    beta = -160

    new = alpha * rotated + beta
    new = np.clip(new, 0, 255).astype(np.uint8)
    # cv2.imwrite("cleaned.png", new)

    mask_removed = remove_water_mask(new)
    # cv2.imwrite('mask_removed.png', mask_removed)
    grayed = binarization(mask_removed, 31, 2)
    # cv2.imwrite('grayed.png', grayed)
    return pytesseract.image_to_string(grayed, config=custom_config)


# Extract text from a licence size image file
def text_from_license(file_path, custom_config):
    im = Image.open(file_path)
    background_removed = remove_single_color_background(im)
    background_removed.save(f'tempfiles/bg_rmd.png')
    img = cv2.imread(f'tempfiles/bg_rmd.png')
    img = resize_image_size(img)
    delete_file(f'tempfiles/bg_rmd.png')

    angle = determine_skew(get_grayscale(img))
    rotated = rotate(img, angle, (0, 0, 0))
    # cv2.imwrite('rotated.png', rotated)

    grayed = get_grayscale(rotated)
    # cv2.imwrite('grayed.png', grayed)
    # blurred = cv2.GaussianBlur(grayed, (5, 5), 0)
    blurred = cv2.medianBlur(grayed, 3)
    # blurred = cv2.bilateralFilter(grayed, 9, 75, 75)
    # blurred = cv2.threshold(grayed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # cv2.imwrite('blurred.png', blurred)

    file_text = pytesseract.image_to_string(blurred, config=custom_config)
    if words_found_in_text(file_text, 'GOVT.'):
        return file_text
    else:
        for t in range(0, 5):
            rotated = rotate(rotated, 90, (0, 0, 0))
            grayed = get_grayscale(rotated)
            blurred = cv2.medianBlur(grayed, 3)
            # cv2.imwrite('blurred' + str(t) + '.png', blurred)
            file_text = pytesseract.image_to_string(blurred, config=custom_config)
            # print(file_text)
            if words_found_in_text(file_text, 'GOVT.'):
                return file_text
    return 'IMAGE TEXT NOT CLEAR !!!!!!!'


# Extract text from a receipt size image file
def text_from_licence_plain_receipt_file(file_path, custom_config):
    img = cv2.imread(file_path)
    img = resize_image_size(img)
    deskew = skew_correction(img)
    grayed = get_grayscale(deskew)
    return pytesseract.image_to_string(grayed, config=custom_config)
