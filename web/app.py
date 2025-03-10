import streamlit as st
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, models
from PIL import Image
from torch import nn
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# 获取模型
def get_model(model_name: str):
    use_pretrained = True
    num_classes = 7  # 皮肤癌分类数

    if model_name == 'resnet_pret':
        model_ft = models.resnet50(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'densenet_pret':
        model_ft = models.densenet121(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'mobilenet_v3':
        model_ft = models.mobilenet_v3_large(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[3].in_features
        model_ft.classifier[3] = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'efficientnet':
        model_ft = models.efficientnet_b1(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
    else:
        st.error("请选择有效的模型: 'resnet_pret', 'densenet_pret', 'mobilenet_v3', 'efficientnet'.")
        return None

    return model_ft

# 载入模型
@st.cache(allow_output_mutation=True)
def load_model(model_path, model_name):
    model = get_model(model_name)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# 预处理图片
norm_means = [0.77148203, 0.55764165, 0.58345652]
norm_std = [0.12655577, 0.14245141, 0.15189891]
img_h, img_w = 224, 224
val_test_transform = transforms.Compose([
    transforms.Resize((img_h, img_w)),
    transforms.ToTensor(),
    transforms.Normalize(norm_means, norm_std)
])

# 分类图片
def classify_image(model, image):
    with torch.no_grad():
        image_tensor = val_test_transform(image).unsqueeze(0)
        scores = model(image_tensor)
        return torch.softmax(scores, dim=1).squeeze(0).numpy()

# 生成 PDF 报告
def generate_pdf(patient_info, scores, labels):
    pdf_path = "classification_report.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.drawString(100, 750, "皮肤癌分类报告")
    c.drawString(100, 730, f"姓名: {patient_info['name']}")
    c.drawString(100, 710, f"性别: {patient_info['gender']}")
    c.drawString(100, 690, f"年龄: {patient_info['age']}")
    c.drawString(100, 670, f"ID: {patient_info['id']}")
    
    c.drawString(100, 640, "分类得分:")
    for i, label in enumerate(labels):
        c.drawString(120, 620 - i * 20, f"{label}: {scores[i]:.2f}")
    
    c.drawString(100, 500, f"最终分类结果: {labels[np.argmax(scores)]}")
    c.save()
    return pdf_path

# Streamlit 主函数
def main():
    st.title("皮肤癌图像分类")
    labels = ["类别 0", "类别 1", "类别 2", "类别 3", "类别 4", "类别 5", "类别 6"]

    # 输入病人信息
    st.sidebar.header("病人信息")
    name = st.sidebar.text_input("姓名")
    gender = st.sidebar.selectbox("性别", ("男", "女"))
    age = st.sidebar.number_input("年龄", min_value=0, max_value=120, step=1)
    patient_id = st.sidebar.text_input("病人 ID")
    
    patient_info = {"name": name, "gender": gender, "age": age, "id": patient_id}

    model_name = st.sidebar.selectbox("选择模型", ('resnet_pret', 'densenet_pret', 'mobilenet_v3', 'efficientnet'))
    model_path = st.sidebar.file_uploader("上传模型权重文件 (.pth)")
    image_file = st.file_uploader("上传皮肤癌图片", type=["jpg", "jpeg", "png"])

    if model_path is not None and image_file is not None:
        model = load_model(model_path, model_name)
        image = Image.open(image_file)
        st.image(image, caption="上传的图片", use_column_width=True)

        if st.button("开始分类"):
            scores = classify_image(model, image)
            chart_placeholder = st.empty()

            for i in range(10):
                progress_scores = scores * (i + 1) / 10  # 逐步增加分数
                
                fig, ax = plt.subplots()
                ax.bar(labels, progress_scores, color='skyblue')
                ax.set_ylim(0, 1)
                ax.set_title("分类得分变化")
                
                chart_placeholder.pyplot(fig)  # 更新图表
                time.sleep(0.1)  # 控制动画速度

            st.write(f"最终分类结果: {labels[np.argmax(scores)]}")
            
            # 生成 PDF 报告
            pdf_path = generate_pdf(patient_info, scores, labels)
            with open(pdf_path, "rb") as pdf_file:
                st.download_button("📥 下载报告", pdf_file, file_name="classification_report.pdf", mime="application/pdf")

if __name__ == '__main__':
    main()