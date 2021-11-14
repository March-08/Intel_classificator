import os
import glob
import shutil

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def split_data(path_to_data,path_to_save_train,path_to_save_val,split_size = 0.2):
    
    folders = os.listdir(path_to_data)
    
    for folder in folders:
        path_to_images = os.path.join(path_to_data,folder)
        images_paths = glob.glob(os.path.join(path_to_images,"*.jpg"))

        x_train ,x_val = train_test_split(images_paths,test_size = split_size)
        
        path_to_train_folder = os.path.join(path_to_save_train,folder)
        path_to_val_folder = os.path.join(path_to_save_val,folder)
        
        if not os.path.isdir(path_to_train_folder):
            os.mkdir(path_to_train_folder)

        if not os.path.isdir(path_to_val_folder):
            os.mkdir(path_to_val_folder)

        for x in x_train:
            shutil.copy(x,path_to_train_folder)

        for x in x_val:
            shutil.copy(x,path_to_val_folder)

def create_generators(batch_size , path_to_train, path_to_val, path_to_test):
    train_preprocessor = ImageDataGenerator(
        rescale=1/255.,
        rotation_range=10,
        width_shift_range=0.1
        )
    
    val_preprocessor = ImageDataGenerator(
        rescale=1/255.
    )


    train_generator = train_preprocessor.flow_from_directory(
        path_to_train,
        class_mode="categorical",
        target_size=(150,150),
        color_mode="rgb",
        shuffle=True,
        batch_size= batch_size
    )

    val_generator = val_preprocessor.flow_from_directory(
        path_to_val,
        class_mode="categorical",
        target_size=(150,150),
        color_mode="rgb",
        shuffle=True,
        batch_size= batch_size
    )


    test_generator = val_preprocessor.flow_from_directory(
        path_to_test,
        class_mode="categorical",
        target_size=(150,150),
        color_mode="rgb",
        shuffle=True,
        batch_size= batch_size
    )


    return train_generator,val_generator,test_generator


if __name__ == "__main__":

    #path_to_data = "D:\DATASETS\INTEL_IMAGE\seg_train"
    #path_to_save_train = "D:\DATASETS\INTEL_IMAGE\\training\\train"
    #path_to_save_val = "D:\DATASETS\INTEL_IMAGE\\training\\val"

    #split_data(path_to_data,path_to_save_train,path_to_save_val)

    pass
