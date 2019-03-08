#! /bin/bash


#if mount | grep  > /dev/null; then
#    echo 'drive mounted'
#else
#    sudo mount /dev/sda6 /media/dc/virt
#fi
filename=$(date +%d-%m-%Y_%H-%M-%S)
path="$HOME/workspace/datasets/video/$filename.avi"

source activate ptz
echo $path
python record_video.py $path -u 0
conda deactivate
