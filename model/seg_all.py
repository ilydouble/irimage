from segment_anything import SamPredictor, sam_model_registry
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. 加载模型
model_type = "vit_h"
checkpoint = "sam_vit_h_4b8939.pth"
sam = sam_model_registry[model_type](checkpoint=checkpoint)
predictor = SamPredictor(sam)


# 2. 定义处理单张图片的函数
def process_image(image_path, output_folder, subfolder_name, image_name):
    try:
        # 加载并验证图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图片: {image_path}")
            return
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # SAM模型处理
        predictor.set_image(image)
        input_prompts = {
            "point_coords": np.array([
                [364, 188], [614, 447], [610, 595],
                [477, 455], [470, 422], [600, 727],
                [160, 450], [1100, 450], [600, 30],
                [916, 840], [260, 860]
            ]),
            "point_labels": np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]), #该数列数字数量要求与点的数量保持一致，其中1代表这个点需要保留，0代表此处可覆盖掩码
            "box": np.array([[270, 80, 1000, 850]]) #该数组是一个类似于窗口，模型会优先保留该窗口中的图像
        }

        # 生成掩码
        masks, _, _ = predictor.predict(
            **input_prompts,
            multimask_output=True
        )

        if len(masks) > 0:
            combined_mask = np.logical_or.reduce(masks)
        else:
            raise ValueError("未生成有效掩码")

        refined_mask = optimize_mask(combined_mask)
        save_results(image, refined_mask, output_folder, subfolder_name, image_name)

    except Exception as e:
        print(f"处理 {image_path} 失败: {str(e)}")


# 3. 掩码优化函数
def optimize_mask(mask, kernel_size=5, feather_radius=10): #kernel_size可以使得切割出来的图像更加光滑而不会变得很斑驳，该值越大其边缘越光滑但会丢失细节；feather_radius是使得边缘羽化，越大其边缘显示越明显。
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed_mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    blurred = cv2.GaussianBlur(closed_mask.astype(float), (0, 0), feather_radius)
    refined = (blurred - blurred.min()) / (blurred.max() - blurred.min() + 1e-7)  # 避免除以零
    return refined


# 4. 结果保存函数（强制转换为 uint8 类型）
def save_results(image, refined_mask, output_folder, subfolder_name, image_name):
    sub_output = os.path.join(output_folder, subfolder_name)
    os.makedirs(sub_output, exist_ok=True)

    black_bg = np.zeros_like(image)
    masked_img = image * refined_mask[..., None] + (1 - refined_mask[..., None]) * black_bg

    # 强制转换为 uint8 类型（确保范围在 0-255）
    masked_img = masked_img.astype(np.uint8)

    # 提取并绘制边界
    binary_mask = (refined_mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(masked_img, contours, -1, (255, 255, 255), 2)

    # 保存图像
    output_img = os.path.join(sub_output, f"boundary_{subfolder_name}_{image_name}")
    plt.imsave(output_img, masked_img)  # 此时 masked_img 是 uint8 类型，无报错

    # 保存坐标
    output_txt = os.path.join(sub_output, f"coords_{subfolder_name}_{image_name.split('.')[0]}.txt")
    with open(output_txt, "w") as f:
        for cnt in contours:
            f.write("\n".join([f"{p[0][0]},{p[0][1]}" for p in cnt]))


# 5. 主处理函数（修改为处理所有图片）
def process_main_folder(main_input, main_output):
    for root, dirs, _ in os.walk(main_input):
        for dir_name in dirs:
            subfolder = os.path.join(root, dir_name)
            # 遍历子文件夹中的所有图片
            for file in sorted(os.listdir(subfolder)):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    image_path = os.path.join(subfolder, file)
                    print(f"正在处理: {dir_name} -> {file}")
                    process_image(image_path, main_output, dir_name, file)
            print(f"完成文件夹: {dir_name}")


if __name__ == "__main__":
    main_input_folder = ".\data" #代码读取的文件夹
    main_output_folder = ".\output" #代码产生的图片存放的文件夹
    process_main_folder(main_input_folder, main_output_folder)
    print("所有图片处理完成！")