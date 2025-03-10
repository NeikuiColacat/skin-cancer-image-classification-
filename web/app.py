import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
from torch import nn

# 自定义 CSS 样式
st.markdown("""
    <style>
        .title {
            font-size: 40px;
            font-weight: bold;
            color: #4CAF50;
            text-align: center;
        }
        .prediction {
            font-size: 20px;
            font-weight: bold;
            color: #FF5722;
            text-align: center;
        }
        .bar-chart {
            margin: auto;
            display: block;
            width: 80%;
        }
    </style>
""", unsafe_allow_html=True)

# 定义模型加载函数
def get_model(model_name: str):
    use_pretrained = True
    num_classes = 7
    
    if model_name == 'mobilenet_v3':
        model = models.mobilenet_v3_large(weights='DEFAULT')
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, num_classes)
    else:
        st.warning("无效的模型名称")
        return None

    return model

# 加载模型
model_name = 'mobilenet_v3'
model = get_model(model_name)
model.load_state_dict(torch.load('mobilenetv3_pret.pth', map_location=torch.device('cpu')))
model.eval()

# 定义图像预处理函数
norm_means = [0.77148203, 0.55764165, 0.58345652]
norm_std = [0.12655577, 0.14245141, 0.15189891]
img_h, img_w = 224, 224

val_test_transform = transforms.Compose([
    transforms.Resize((img_h, img_w)),
    transforms.ToTensor(),
    transforms.Normalize(norm_means, norm_std)
])

# Streamlit 应用界面
st.markdown('<div class="title">图像分类服务</div>', unsafe_allow_html=True)
st.write("上传一张图像进行分类")

# 文件上传
uploaded_file = st.file_uploader("选择一张图像", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.convert("RGB")  # 确保图像为 RGB 模式
    st.image(image, caption="上传的图像", use_column_width=True)

    if st.button("预测"):
        # 预处理图像
        image_tensor = val_test_transform(image).unsqueeze(0)

        # 进行预测
        with torch.no_grad():
            scores = model(image_tensor)
            probabilities = torch.nn.functional.softmax(scores, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)

            # 显示预测结果
            st.markdown(f'<div class="prediction">预测类别索引: {predicted_class.item()}</div>', unsafe_allow_html=True)
            
            # 假设类别名称为一个列表
            class_names = ["类别1", "类别2", "类别3", "类别4", "类别5", "类别6", "类别7"]
            st.markdown(f'<div class="prediction">预测类别: {class_names[predicted_class.item()]}</div>', unsafe_allow_html=True)
            
            # 展示分类分数的柱状图
            st.bar_chart(probabilities.squeeze().numpy(), use_container_width=True)