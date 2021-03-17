param (
    [string]$method = "NoMethod"
)
$script = {
    # python3 detect.py --source data/images/Original --weights ./runs/train/exp12/weights/best.pt --conf 0.25
    function detect()
    {
        python3 detect.py `
            --imglist_file data\barcode_All\train.txt `
            --weights_path runs/train/exp13/weights/last.weights  `
            --class_path .\data\barcode_All\classes.names `
            --model_def .\config\yolov3-barcode.cfg
        # --image_folder ..\barcode_detection_dataset\pictures\images\20210317_2 `
    }
    function detectCorrect()
    {
        python3 detect.py `
            --imglist_file data\CorrectDetect.txt `
            --weights_path runs/train/exp14/weights/last.weights  `
            --class_path .\data\barcode_All\classes.names `
            --model_def .\config\yolov3-barcode.cfg
        # --image_folder ..\barcode_detection_dataset\pictures\images\20210317_2 `
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
    Write-Output("zaima")
    Write-Output($method)
    switch ($method)
    {
        "detect"
        {
            detect
        }
        "train_all"{
            train_all
        }
        "train"{
            train
        }
        "detectCorrect"{
            detectCorrect
        }
    }
    # outputModel
    #detect
    #train_all
    #detect
}
#netsh winhttp set proxy 127.0.0.1:1080
Invoke-Command $script
