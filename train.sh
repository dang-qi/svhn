#python train.py --config_path configs/fcos/fcos_resnet50_fpn_coco.py -b 2 --gpu_num 1 --linear_lr "$@"
python train.py --config_path configs/retinanet/retinanet_resnet50_fpn_svhn.py -b 2 --gpu_num 1 --linear_lr "$@"
