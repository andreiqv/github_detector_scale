rm dataset-bboxes-train.list
rm dataset-bboxes-valid.list
DATAPATH=/home/andrei/Data/Datasets/ScalesDetector/output_2019-01-20
find $DATAPATH/train -type f -name "*.jpg" > dataset-bboxes-train.list
find $DATAPATH/valid -type f -name "*.jpg" > dataset-bboxes-valid.list
