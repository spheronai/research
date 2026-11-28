#!/bin/bash
# Save every task image to the volume once, so a replacement node can `docker load` them instead of pulling
# through Docker Hub's rate limit. One tar for all images, so shared layers are stored once.
cd /root/s3
mkdir -p /mnt/vol/images
s=$(date +%s)
docker save -o /mnt/vol/images/images.tar.part $(cat raw/env/images_all.txt) && mv /mnt/vol/images/images.tar.part /mnt/vol/images/images.tar
echo "SAVE_DONE $? in $(( $(date +%s)-s )) s, $(du -sh /mnt/vol/images/images.tar 2>/dev/null | cut -f1)"
