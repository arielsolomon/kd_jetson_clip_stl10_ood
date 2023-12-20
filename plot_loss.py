import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import re
from PIL import Image

root = '/Data/federated_learning/kd_jetson_cifar_ood/data/experiments/cifar10_c/'
exp_list = os.listdir(root)
# if not os.path.exists('/Data/federated_learning/kd_jetson_clip_stl10_ood/data/exp_loss_acc_images/'):
#     os.mkdir('/Data/federated_learning/kd_jetson_clip_stl10_ood/data/exp_loss_acc_images/')
dest = '/Data/federated_learning/kd_jetson_cifar_ood/data/exp_loss_acc_images_cifar10_c/'
if not os.path.exists(dest):
    os.mkdir(dest)
loaded_images = []
def extract_values(log):
    match = re.search(r'TRAIN LOSS ([\d\.e-]+) \| TEST ACC ([\d\.]+) \|', log)
    if match:
        train_loss = float(match.group(1))
        test_acc = float(match.group(2))
        return train_loss, test_acc
    else:
        return None, None

def read_and_extract_values(file_path):
    # Read the text file into a DataFrame
    df = pd.read_csv(file_path, header=None, names=['Log'])
    df['Epochs'] = df.index
    # Extract values using the provided function
    df[['Train Loss', 'Test Accuracy']] = df['Log'].apply(lambda x: pd.Series(extract_values(x)))
    df.drop(columns=['Log'], inplace=True)
    # plot:
    title = file_path.split('/')[-2]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot Train Loss on the first subplot (ax1)
    ax1.plot(df['Epochs'], df['Train Loss'], label='Train Loss',color='green', marker="o")
    ax1.set_title(title+' Train Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')

    # Plot Test Accuracy on the second subplot (ax2)
    ax2.plot(df['Epochs'], df['Test Accuracy'], label='Test Accuracy', color='orange', marker="o")
    ax2.set_title(title+' Test Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')

    # Adjust layout to prevent clipping of titles
    plt.tight_layout()

    # Save the figure to a destination path
    destination_path = dest+'/'+title+'.png'
    plt.savefig(destination_path)

def create_single_image(image_paths,images):
    for path in glob.glob(os.path.join(image_paths, '*.png')):
        img = Image.open(path)
        images.append(img)

    # Get the dimensions of the images
    width, height = images[0].size

    # Create a new image with a size large enough for a 2x3 grid
    result_width = 3 * width
    result_height = 2 * height
    result_image = Image.new('RGB', (result_width, result_height))

    # Paste each image into the result image
    for i in range(4):
        row = i // 2
        col = i % 2
        result_image.paste(images[i], (col * width, row * height))

    # Save the result image
    result_image.save(dest+'composed.jpg')



for exp in exp_list:
    exp_path = root+exp+'/'
    log = exp_path+'log.txt'
    read_and_extract_values(log)
create_single_image(dest, loaded_images)

