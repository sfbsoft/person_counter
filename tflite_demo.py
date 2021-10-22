import numpy as np
import tensorflow as tf
from collections import defaultdict, deque
import cv2
import os
from datetime import datetime, timedelta
from motrackers import CentroidTracker
from motrackers.utils import draw_tracks


H = 0
W = 0


def yxyx2xywh(yxyx):
    if len(yxyx.shape) == 2:
        w, h = yxyx[:, 3] - yxyx[:, 1] + 1, yxyx[:, 2] - yxyx[:, 0] + 1
        xywh = np.concatenate((yxyx[:, 1, None], yxyx[:, 0, None], w[:, None], h[:, None]), axis=1)
        return xywh.astype("int")
    elif len(yxyx.shape) == 1:
        (top, left, bottom, right) = yxyx
        width = right - left + 1
        height = bottom - top + 1
        return np.array([left, top, width, height]).astype('int')
    else:
        raise ValueError("Input shape not compatible.")


def checkLeft(pts, x_c):
    global H, W
    if len(pts) >= 2 and pts[-1]['x_c'] < W // 2 and pts[-2]['x_c'] >= W // 2:
        # direction = x_c - np.mean(pts)
        mean_pts = np.mean([p['x_c'] for p in pts])
        direction = x_c - mean_pts
        if direction < 0:
            return True
        else:
            return False
    else:
        return False


def checkRight(pts, x_c):
    global H, W
    if len(pts) >= 2 and pts[-1]['x_c'] > W // 2 and pts[-2]['x_c'] <= W // 2:
        # direction = x_c - np.mean(pts)
        mean_pts = np.mean([p['x_c'] for p in pts])
        direction = x_c - mean_pts
        if direction > 0:
            return True
        else:
            return False
    else:
        return False


def del_id(pts):
    previous_time = datetime.now() - timedelta(hours=0, minutes=5)
    del_ids = []
    for track_id, value in pts.items():
        # check timestamp of the first track_id
        # delete that track_id's value if timestamp is 5 minutes before
        if value[0]['timestamp'] < previous_time:
            del_ids.append(track_id)

    for del_id in del_ids:
        del pts[del_id]
    return pts


def main():
    global H, W
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="./coco_ssd_mobilenet/ssd_mobilenet_v2_coco_quant_postprocess.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    # get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_height = int(input_details[0]['shape'][1])
    input_width = int(input_details[0]['shape'][2])
    floating_model = (input_details[0]['dtype'] == np.float32)
    input_mean = 127.5
    input_std = 127.5
    labels_path = os.path.join('./coco_ssd_mobilenet/coco_labels.txt')
    with open(labels_path, 'r') as f:
        lines = f.readlines()
    labels = [x.strip() for x in lines]
    print('class names: {}'.format(labels))
    cap = cv2.VideoCapture(0)
    totalLeft = 0
    totalRight = 0
    counted_ids = deque(maxlen=50)
    pts = defaultdict(list)
    tracker = CentroidTracker(max_lost=30, tracker_output_format='mot_challenge')

    while(True):
        ret, image = cap.read()
        if not ret:
            break
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W = image_rgb.shape[:2]
        image_resized = cv2.resize(image_rgb, (input_width, input_height))
        input_data = np.expand_dims(image_resized, axis=0)

        # normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # retrieve detection results
        # # boundingbox
        all_bboxes = interpreter.get_tensor(output_details[0]['index'])[0]
        # (ymin, xmin, ymax, xmax) -> (xmin, ymin, width, height)
        all_bboxes = all_bboxes * np.array([H, W, H, W])
        all_bboxes = yxyx2xywh(all_bboxes)

        # class_ids
        all_class_ids = interpreter.get_tensor(output_details[1]['index'])[0]

        # confidences
        all_confidences = interpreter.get_tensor(output_details[2]['index'])[0]

        condition_class_ids = all_class_ids == 0.
        condition_confidences = all_confidences > 0.4
        condition = np.logical_and(condition_class_ids, condition_confidences)

        bboxes = all_bboxes[condition]
        confidences = all_confidences[condition]
        class_ids = all_class_ids[condition]

        assert len(bboxes) == len(confidences) == len(class_ids), 'something is wrong!'
        # update tracker
        tracks = tracker.update(bboxes, confidences, class_ids)

        # update monitor result
        for track in tracks:
            # track_id
            track_id = track[1]

            # x coordinate
            x_coord = track[2]

            # width
            width = track[4]

            # x_center
            x_c = int(x_coord + 0.5 * width)

            # append x coordinate
            pts[track_id].append({'timestamp': datetime.now(), 'x_c': x_c})

            # make sure mean of x coordinate is stable
            if track_id not in counted_ids and checkRight(pts[track_id], x_c):
                counted_ids.append(track_id)
                totalRight += 1
                del pts[track_id]
            if track_id not in counted_ids and checkLeft(pts[track_id], x_c):
                counted_ids.append(track_id)
                totalLeft += 1
                del pts[track_id]

        # draw centertroid
        image = draw_tracks(image, tracks)

        # draw_bboxes
        for bb, conf, cid in zip(bboxes, confidences, class_ids):
            clr = (255, 0, 255)
            cv2.rectangle(image, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]), clr, 2)
            label = "{}:{:.4f}".format(labels[int(cid)], conf)
            (label_width, label_height), baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            y_label = max(bb[1], label_height)
            cv2.rectangle(image, (bb[0], y_label - label_height), (bb[0] + label_width, y_label + baseLine), (255, 255, 255), cv2.FILLED)
            cv2.putText(image, label, (bb[0], y_label), cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2)

        # draw center line
        # cv2.line(image, (W // 2, 0), (W // 2, H), (0, 0, 255), 2)

        # draw boundingbox
        # TODO

        # draw centertroid
        # image = draw_tracks(image, tracks)

        # construct a tuple of information we will be displaying on the
        # info = [
        #     ("Right2Left", totalLeft),
        #     ("Left2Right", totalRight),
        # ]

        # Display the monitor result
        # for (i, (k, v)) in enumerate(info):
        #         text = "{}: {}".format(k, v)
        #         cv2.putText(image, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # check time to remove not counted id
        pts = del_id(pts)
        cv2.imshow("image", image)
        result = {}
        if result is not None:
            result['totalLeft'] = totalLeft
            result['totalRight'] = totalRight
        print('result: {}'.format(result))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
