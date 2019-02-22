#! /bin/bash


if mount | grep /media/dc/virt > /dev/null; then
    echo 'drive mounted'
else
    sudo mount /dev/sda6 /media/dc/virt
fi
filename=$(date +%d-%m-%Y_%H-%M-%S)
echo "/media/dc/virt/$filename.avi"

source activate tx2
python record_video.py /media/dc/virt/$filename.avi -u 2
source deactivate
