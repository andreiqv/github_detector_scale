rm dataset-bboxes-train.list
rm dataset-bboxes-valid.list
DATAPATH=/home/andrei/Data/Datasets/ScalesDetector/output
find $DATAPATH/train -type f -name "*.jpg" > dataset-bboxes-train.list
find $DATAPATH/valid -type f -name "*.jpg" > dataset-bboxes-valid.list
