$script = {
    # python3 detect.py --source data/images/Original --weights ./runs/train/exp12/weights/best.pt --conf 0.25
    function detect()
    {
        python3 detect.py `
            --image_folder ..\barcode_detection_dataset\pictures\images\20210317_2 `
            --weights_path runs/train/exp13/weights/last.weights  `
            --class_path .\data\barcode\classes.names `
            --model_def .\config\yolov3-barcode.cfg
    }
    function train()
    {
        python3 ./train.py `
            --n_cpu 12  `
            --model_def config/yolov3-barcode.cfg `
            --data_config config/barcode.data `
            --epochs 300 `
            --batch_size 8 `
            --verbose
        # --pretrained_weights checkpoints/yolov3_ckpt_87.pth

    }
    function train_all()
    {
        python3 ./train.py `
            --n_cpu 12  `
            --model_def config/yolov3-barcode.cfg `
            --data_config config/barcode_All.data `
            --epochs 300 `
            --batch_size 8 `
            --verbose
        # --pretrained_weights checkpoints/yolov3_ckpt_87.pth

    }
    # outputModel
    detect
    #train_all
    #detect
}
#netsh winhttp set proxy 127.0.0.1:1080
Invoke-Command $script
