from ultralytics import YOLO

# 加载一个预训练的 YOLO26n 模型
model = YOLO("yolo26n-pose.pt")

# # 在 COCO8 数据集上训练模型 100 个周期
# train_results = model.train(
#     data="coco8.yaml",  # 数据集配置文件路径
#     epochs=100,  # 训练周期数
#     imgsz=640,  # 训练图像尺寸
#     device="cpu",  # 运行设备 (例如 'cpu', 0, [0,1,2,3])
#     project="/workspace"
# )

# # 评估模型在验证集上的性能
# metrics = model.val()

# 对图像执行目标检测
results = model("/workspace/ultralytics/assets/bus.jpg")  # 对图像进行预测

result = results[0]
result.show()  # 显示结果

# 保存带标注框的推理结果图片
result.save("runs/predict/bus.jpg")
# 保存检测框坐标 (格式: class x_center y_center width height confidence，均为归一化值)
result.save_txt("runs/predict/bus.txt", save_conf=True)

# 将模型导出为 ONNX 格式以进行部署
path = model.export(format="onnx")  # 返回导出模型的路径