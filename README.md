# github_detector_scale
github_detector_scale

1) prepare_dataset.py: json -> txt

2) делаем tfrecords:
find $PWD/train -type f -name "*.jpg" > dataset-bboxes-train.list
find $PWD/valid -type f -name "*.jpg" > dataset-bboxes-valid.list
python3 sp_tfrecords_converter.py

3) train model
