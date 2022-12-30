import os
import sys

base_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../../"))
sys.path.append(base_dir)  # 设置项目根目录

image_path_train = os.path.abspath(os.path.join(base_dir, "dataset/MNIST/mul-digits/train/"))
image_path_eval = os.path.abspath(os.path.join(base_dir, "dataset/MNIST/mul-digits/test/"))
train_path = os.path.abspath(os.path.join(base_dir, "dataset/MNIST/mul-digits/train.txt"))
eval_path = os.path.abspath(os.path.join(base_dir, "dataset/MNIST/mul-digits/eval.txt"))

image_list_train = os.listdir(image_path_train)     # 文件列表
image_list_eval = os.listdir(image_path_eval)
# x_train, x_eval = train_test_split(image_list, test_size=0.1, random_state=0)

fp_train = open(train_path, "w", encoding="utf-8")
fp_eval = open(eval_path, "w", encoding="utf-8")

for image_single_name in image_list_train:
    image_single_path = os.path.abspath(os.path.join(image_path_train, image_single_name))
    fp_train.write(image_single_path)
    fp_train.write("\n")
fp_train.close()

for image_single_name in image_list_eval:
    image_single_path = os.path.abspath(os.path.join(image_path_eval, image_single_name))
    fp_eval.write(image_single_path)
    fp_eval.write("\n")
fp_eval.close()