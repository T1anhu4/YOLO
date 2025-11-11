from ultralytics import YOLO

model = YOLO('yolov8m.pt')

model.train(
    data='/home/zhy/wth/wth3/YOLODataset/dataset.yaml',
    epochs=200,
    imgsz=640,
    batch=4,
    workers=4,
    device=1,
    optimizer='AdamW',
    lr0=0.0003, 
    lrf=0.1,
    weight_decay=0.005,
    warmup_epochs=5.0,
    patience=30, 
    
    # 余弦退火
   #  scheduler='cosine',

    # 损失函数超参调整
    box=7.5,
    cls=0.5,
    dfl=1.0,

    # 适度增强
    mosaic=0.8,          # 前期开足
    mixup=0.2,
    erasing=0.1,
    copy_paste=0.3,

    # 几何变换
    hsv_h=0.01, hsv_s=0.4, hsv_v=0.3,
    translate=0.2, scale=0.8,
    degrees=15, shear=0.0, perspective=0.0003,
    fliplr=0.5, flipud=0.3,    # 航拍视角可以接受上下翻转
    auto_augment='randaugment',
    close_mosaic=20,     # 最后15个epoch关闭马赛克
    multi_scale=True,    # 开多尺度（v8 支持）
    cache=True,
    rect=False, amp=True,save_period=10
)
