import cv2
from os.path import join
import pandas as pd

def fill_in (img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    """
    This function fill in the annotation with threshold and contours
    """

    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(imgray, 127, 255, 0)
    contours,_ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ellipse = cv2.fitEllipse(contours[0])

    return cv2.ellipse(img, ellipse, (255,255,255), -1)

def masking (df:pd.DataFrame, train_dir:str, mask_dir:str) -> None:
    for index in range(len(df)):
        file_name = df.iloc[index, 0].replace('.png', '_Annotation.png')
        file_path = join(train_dir, file_name)

        image = cv2.imread(file_path)

        mask = fill_in(image)

        name_mask = file_name.replace('_Annotation.png', '_Mask.png')
        output_path = join(mask_dir, name_mask)

        cv2.imwrite(output_path, mask)
