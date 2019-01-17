import tensorflow as tf

EMPTY_PLATFORM_THRESHOLD = 0.7

def bboxes_loss(labels, logits):
    """ For 1-nd model: objectness
    """
    objectness_loss = tf.contrib.losses.mean_squared_error(logits[:, 4], labels[:, 4])
    bbox_loss = tf.reduce_mean(tf.squared_difference(logits[:, :4], labels[:, :4]), axis=1)
    bbox_loss = tf.reduce_mean(bbox_loss * labels[:, 4])
    return objectness_loss + 1 * bbox_loss

