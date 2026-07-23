from ultralytics import YOLO

# 加载一个预训练的 YOLO26n 模型
model = YOLO("yolo26n.pt")

# 在 COCO8 数据集上训练模型 100 个周期
train_results = model.train(
    data="coco8.yaml",  # 数据集配置文件路径
    epochs=100,  # 训练周期数
    imgsz=640,  # 训练图像尺寸
    device="cpu",  # 运行设备 (例如 'cpu', 0, [0,1,2,3])
    project="/workspace"
)

# 评估模型在验证集上的性能
metrics = model.val()

# 对图像执行目标检测
results = model("/workspace/bus.jpg")  # 对图像进行预测
results[0].show()  # 显示结果

# 将模型导出为 ONNX 格式以进行部署
path = model.export(format="onnx")  # 返回导出模型的路径