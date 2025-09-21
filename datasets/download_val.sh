#!/bin/bash

for i in $(seq -f "%08g" 1 50000); do
  FILENAME="ILSVRC/Data/CLS-LOC/val/ILSVRC2012_val_$i.JPEG"
  echo "Downloading $FILENAME"
  kaggle competitions download -c imagenet-object-localization-challenge -f "$FILENAME" -p "imagenet/val"
done
