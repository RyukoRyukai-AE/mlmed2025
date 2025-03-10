from tensorflow.keras.optimizers import Adam
from src.data_loader import HC18Dataset
from sklearn.model_selection import train_test_split
import yaml
import pandas as pd
from os.path import join
from src.unet_model import UNet
from src.utils import dice_loss, masking_annotation, predict

with open('src/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

train_dir = config['paths']['train_data']
test_dir = config['paths']['test_data']
mask_dir = config['paths']['train_mask']
data = config['paths']['data_csv']
fig_path = config['paths']['figure']
model_path = config['paths']['model']

pix_train_df = pd.read_csv(join(data, 'training_set_pixel_size_and_HC.csv'))
pix_test_df = pd.read_csv(join(data, 'test_set_pixel_size.csv'))

masking_annotation.masking(pix_train_df, train_dir, mask_dir)

dataset = HC18Dataset(img_dir=train_dir, mask_dir=mask_dir)
X, Y = dataset.load_data()
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

optimizer = Adam(learning_rate=1e-4)

unet = UNet()
unet.compile(optimize=optimizer, loss=dice_loss.dice_loss)
unet.train(X_train, Y_train, X_val, Y_val, epochs=200)

predict.predict_and_show(join(unet, test_dir, '000_HC.png'))