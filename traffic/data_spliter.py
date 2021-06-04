
import numpy as np
import os
import shutil
import cv2




def split_dataset_into_3(path_to_dataset, valid_number, test_number):
    """
    split the dataset in the given path into three subsets(test,validation,train)
    :param path_to_dataset:
    :param train_ratio:
    :param valid_ratio:
    :return:
    """
    _, sub_dirs, _ = next(iter(os.walk(path_to_dataset)))  # retrieve name of subdirectories
    sub_dir_item_cnt = [0 for i in range(len(sub_dirs))]  # list for counting items in each sub directory(class)

    # directories where the splitted dataset will lie
    dir_train = os.path.join(os.path.dirname(path_to_dataset), 'train')
    dir_valid = os.path.join(os.path.dirname(path_to_dataset), 'validation')
    dir_test = os.path.join(os.path.dirname(path_to_dataset), 'test')

    for i, sub_dir in enumerate(sub_dirs):

        dir_train_dst = os.path.join(dir_train, sub_dir)  # directory for destination of train dataset
        dir_valid_dst = os.path.join(dir_valid, sub_dir)  # directory for destination of validation dataset
        dir_test_dst = os.path.join(dir_test, sub_dir)  # directory for destination of validation dataset

        # variables to save the sub directory name(class name) and to count the images of each sub directory(class)
        class_name = sub_dir
        sub_dir = os.path.join(path_to_dataset, sub_dir)
        sub_dir_item_cnt[i] = len(os.listdir(sub_dir))

        items = os.listdir(sub_dir)

        if sub_dir_item_cnt[i] > 500:

            # transfer data to trainset
            all_sample = np.array(range(sub_dir_item_cnt[i]))
            np.random.shuffle(all_sample)
            valid_array = all_sample[0:valid_number]
            test_array = all_sample[valid_number:(valid_number + test_number)]
            train_array = all_sample[(valid_number + test_number):]

            for item_idx in train_array:
                if not os.path.exists(dir_train_dst):
                    os.makedirs(dir_train_dst)

                source_file = os.path.join(sub_dir, items[item_idx])
                dst_file = os.path.join(dir_train_dst, items[item_idx])
                shutil.copyfile(source_file, dst_file)

            # transfer data to validation
            for item_idx in valid_array:
                if not os.path.exists(dir_valid_dst):
                    os.makedirs(dir_valid_dst)

                source_file = os.path.join(sub_dir, items[item_idx])
                dst_file = os.path.join(dir_valid_dst, items[item_idx])
                shutil.copyfile(source_file, dst_file)

            # transfer data test
            for item_idx in test_array:
                if not os.path.exists(dir_test_dst):
                    os.makedirs(dir_test_dst)

                source_file = os.path.join(sub_dir, items[item_idx])
                dst_file = os.path.join(dir_test_dst, items[item_idx])
                shutil.copyfile(source_file, dst_file)



    return

split_dataset_into_3('data/Images',40,50)