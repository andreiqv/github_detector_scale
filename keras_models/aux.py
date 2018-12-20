import tensorflow as tf

EMPTY_PLATFORM_THRESHOLD = 0.7

"""
def bboxes_loss(labels, logits):
    objectness_loss = tf.contrib.losses.mean_squared_error(logits[:, 4], labels[:, 4])
    bbox_loss = tf.reduce_mean(tf.squared_difference(logits[:, :4], labels[:, :4]), axis=1)
    bbox_loss = tf.reduce_mean(bbox_loss * labels[:, 4])
    return objectness_loss + 1000 * bbox_loss
"""

def bboxes_loss(labels, logits):
    objectness_loss = tf.contrib.losses.mean_squared_error(logits[:, 4], labels[:, 4])
    #bbox_loss = tf.reduce_mean(tf.squared_difference(logits[:, :4], labels[:, :4]), axis=1)    
    zeros = tf.zeros([4], tf.float32)
    bbox_loss = tf.reduce_mean(tf.squared_difference(logits[:, :4], zeros), axis=1)
    bbox_loss = tf.reduce_mean(bbox_loss * labels[:, 4])
    return objectness_loss + 1 * bbox_loss


def accuracy(_labels, _logits):
    def overlap(logits, labels, center, dimension):
        logits_w_2 = tf.divide(logits[:, dimension], tf.constant(2.0))
        labels_w_2 = tf.divide(labels[:, dimension], tf.constant(2.0))
        l1 = logits[:, center] - logits_w_2
        l2 = labels[:, center] - labels_w_2
        left = tf.maximum(l1, l2)

        r1 = logits[:, center] + logits_w_2
        r2 = labels[:, center] + labels_w_2

        right = tf.minimum(r1, r2)

        width = tf.subtract(right, left)

        # https://stackoverflow.com/questions/41043894/setting-all-negative-values-of-a-tensor-to-zero-in-tensorflow
        return tf.nn.relu(width)

    correct = tf.Variable(0, dtype=tf.float32, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    wrong = tf.Variable(0, dtype=tf.float32, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    threshold = tf.constant(EMPTY_PLATFORM_THRESHOLD)

    empty_correct = tf.Variable(0, dtype=tf.float32, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    empty_wrong = tf.Variable(0, dtype=tf.float32, collections=[tf.GraphKeys.LOCAL_VARIABLES])

    not_empty_correct = tf.Variable(0, dtype=tf.float32, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    not_empty_wrong = tf.Variable(0, dtype=tf.float32, collections=[tf.GraphKeys.LOCAL_VARIABLES])

    not_empty = tf.equal(_labels[:, 4], tf.constant([1.0]))
    empty = tf.logical_not(not_empty)

    logit_not_empty = tf.greater_equal(_logits[:, 4], threshold)
    logit_empty = tf.logical_not(logit_not_empty)

    correct_mask_e = tf.logical_and(empty, logit_empty)
    correct_mask_n = tf.logical_and(not_empty, logit_not_empty)
    wrong_mask_e = tf.logical_and(empty, logit_not_empty)
    wrong_mask_n = tf.logical_and(not_empty, logit_empty)

    empty_correct = empty_correct.assign_add(tf.cast(tf.shape(tf.where(correct_mask_e))[0], dtype=tf.float32))
    empty_wrong = empty_wrong.assign_add(tf.cast(tf.shape(tf.where(wrong_mask_e))[0], dtype=tf.float32))

    not_empty_correct = not_empty_correct.assign_add(tf.cast(tf.shape(tf.where(correct_mask_n))[0], dtype=tf.float32))
    not_empty_wrong = not_empty_wrong.assign_add(tf.cast(tf.shape(tf.where(wrong_mask_n))[0], dtype=tf.float32))

    correct = correct.assign_add(tf.cast(tf.shape(tf.where(tf.logical_and(not_empty, logit_not_empty)))[0] + \
                                         tf.shape(tf.where(tf.logical_and(empty, logit_empty)))[0], dtype=tf.float32))

    wrong = wrong.assign_add(tf.cast(tf.shape(tf.where(tf.logical_and(not_empty, logit_empty)))[0] + \
                                     tf.shape(tf.where(tf.logical_and(empty, logit_not_empty)))[0], dtype=tf.float32))

    w = overlap(_logits, _labels, 0, 2)
    h = overlap(_logits, _labels, 1, 3)
    intersection = tf.multiply(w, h)
    union = tf.multiply(_logits[:, 2], (_logits[:, 3])) + tf.multiply(_labels[:, 2], (_labels[:, 3])) - intersection
    iou = tf.divide(intersection, union)
    correct_empty_amount = tf.cast(tf.shape(tf.where(correct_mask_e))[0], dtype=tf.float32)
    mean_iou = tf.reduce_sum(tf.boolean_mask(iou, correct_mask_n))
    mean_iou = tf.divide(mean_iou, tf.cast(tf.shape(iou)[0], dtype=tf.float32) - correct_empty_amount)

    return correct / (correct + wrong)


def miou(_labels, _logits):
    def overlap(logits, labels, center, dimension):
        logits_w_2 = tf.divide(logits[:, dimension], tf.constant(2.0))
        labels_w_2 = tf.divide(labels[:, dimension], tf.constant(2.0))
        l1 = logits[:, center] - logits_w_2
        l2 = labels[:, center] - labels_w_2
        left = tf.maximum(l1, l2)

        r1 = logits[:, center] + logits_w_2
        r2 = labels[:, center] + labels_w_2

        right = tf.minimum(r1, r2)

        width = tf.subtract(right, left)

        # https://stackoverflow.com/questions/41043894/setting-all-negative-values-of-a-tensor-to-zero-in-tensorflow
        return tf.nn.relu(width)

    correct = tf.Variable(0, dtype=tf.float32, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    wrong = tf.Variable(0, dtype=tf.float32, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    threshold = tf.constant(EMPTY_PLATFORM_THRESHOLD)

    empty_correct = tf.Variable(0, dtype=tf.float32, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    empty_wrong = tf.Variable(0, dtype=tf.float32, collections=[tf.GraphKeys.LOCAL_VARIABLES])

    not_empty_correct = tf.Variable(0, dtype=tf.float32, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    not_empty_wrong = tf.Variable(0, dtype=tf.float32, collections=[tf.GraphKeys.LOCAL_VARIABLES])

    not_empty = tf.equal(_labels[:, 4], tf.constant([1.0]))
    empty = tf.logical_not(not_empty)

    logit_not_empty = tf.greater_equal(_logits[:, 4], threshold)
    logit_empty = tf.logical_not(logit_not_empty)

    correct_mask_e = tf.logical_and(empty, logit_empty)
    correct_mask_n = tf.logical_and(not_empty, logit_not_empty)
    wrong_mask_e = tf.logical_and(empty, logit_not_empty)
    wrong_mask_n = tf.logical_and(not_empty, logit_empty)

    empty_correct = empty_correct.assign_add(tf.cast(tf.shape(tf.where(correct_mask_e))[0], dtype=tf.float32))
    empty_wrong = empty_wrong.assign_add(tf.cast(tf.shape(tf.where(wrong_mask_e))[0], dtype=tf.float32))

    not_empty_correct = not_empty_correct.assign_add(tf.cast(tf.shape(tf.where(correct_mask_n))[0], dtype=tf.float32))
    not_empty_wrong = not_empty_wrong.assign_add(tf.cast(tf.shape(tf.where(wrong_mask_n))[0], dtype=tf.float32))

    correct = correct.assign_add(tf.cast(tf.shape(tf.where(tf.logical_and(not_empty, logit_not_empty)))[0] + \
                                         tf.shape(tf.where(tf.logical_and(empty, logit_empty)))[0], dtype=tf.float32))

    wrong = wrong.assign_add(tf.cast(tf.shape(tf.where(tf.logical_and(not_empty, logit_empty)))[0] + \
                                     tf.shape(tf.where(tf.logical_and(empty, logit_not_empty)))[0], dtype=tf.float32))

    w = overlap(_logits, _labels, 0, 2)
    h = overlap(_logits, _labels, 1, 3)
    intersection = tf.multiply(w, h)
    union = tf.multiply(_logits[:, 2], (_logits[:, 3])) + tf.multiply(_labels[:, 2], (_labels[:, 3])) - intersection
    iou = tf.divide(intersection, union)
    correct_empty_amount = tf.cast(tf.shape(tf.where(correct_mask_e))[0], dtype=tf.float32)
    mean_iou = tf.reduce_sum(tf.boolean_mask(iou, correct_mask_n))
    mean_iou = tf.divide(mean_iou, tf.cast(tf.shape(iou)[0], dtype=tf.float32) - correct_empty_amount)

    return mean_iou


def lr_scheduler(epoch, lr):
    pass
