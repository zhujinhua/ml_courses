from ultralytics import YOLO
'''
labelImg 标注图片
labelImg [IMAGE_PATH] [PRE-DEFINED CLASS FILE]
labelImg /Users/jhzhu/code_repository/git_project/yolo_materials/label_demo/images /Users/jhzhu/code_repository/git_project/yolo_materials/label_demo/classes.txt
过程：1. 先用labelImg标注图像，将标注的结果参考coco8数据组织结构放好
     2. 复制一份yolov8n.yaml，修改path，classes
     3. 将data路径指定为自己标注的数据集地址
'''
if __name__ == "__main__":
    # Load a model
    model = YOLO("yolov8n.yaml")  # build a new model from YAML

    # Train the model
    results = model.train(data="my_det_data.yaml", 
                          epochs=10, 
                          imgsz=640, 
                          batch=2,
                          workers=1)