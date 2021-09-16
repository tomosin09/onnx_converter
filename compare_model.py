from feature_extractor import IR_50
from sklearn.metrics.pairwise import cosine_distances

import cv2
import time
import torch
import numpy as np
import onnxruntime as ort


def compare(vector1, vector2):
    result = False
    dist = cosine_distances(vector1, vector2)
    print(f'dist between vector 1 and vector 2 is {dist}')
    if dist < 0.01:
        result = True
    return result


def l2_norm(inputs, axis=1):
    norm = torch.norm(inputs, 2, axis, True)
    outputs = torch.div(inputs, norm)
    return outputs


def to_format(image):
    image = image.swapaxes(1, 2).swapaxes(0, 1)
    image = np.reshape(image, [1, 3, 112, 112])
    image = np.array(image, dtype=np.float32)
    image = (image - 127.5) / 128.0
    image = torch.from_numpy(image)
    return image


def preprocess_image(image):
    if image.shape[:2] != (128, 128):
        image = cv2.resize(image, (128, 128))

    # center crop image
    a = int((128 - 112) / 2)  # x start
    b = int((128 - 112) / 2 + 112)  # x end
    c = int((128 - 112) / 2)  # y start
    d = int((128 - 112) / 2 + 112)  # y end
    cropped = image[a:b, c:d]  # center crop the image
    cropped = cropped[..., ::-1]  # BGR to RGB
    # flip image horizontally
    flipped = cv2.flip(cropped, 1)
    # cv2.imwrite('input_image.png', cropped+flipped)
    return to_format(cropped) + to_format(flipped)


if __name__ == "__main__":
    path_to_image = "test_image/track-3.png"
    image = cv2.imread(path_to_image)
    weights_pth = 'weights/IR50.pth'
    path_to_onnx = "face_evolve.onnx"
    input_image = preprocess_image(image)
    print(f'input image have shape {input_image.shape}')

    # TorchScript format
    device = torch.device('cpu')
    model = IR_50((112, 112))
    model.load_state_dict(torch.load(weights_pth, map_location='cpu'))
    model.to(device)
    model.eval()
    with torch.no_grad():
        emb_batch = model(input_image)
        feature_1 = l2_norm(emb_batch).numpy()
    print(f'vector from TorchScript model is {feature_1.shape}')

    # ONNX format
    ort_sess = ort.InferenceSession(path_to_onnx)
    start_onnx_time = time.time()
    feature_2 = ort_sess.run(None, {'input0': input_image.numpy().astype(np.float32)})
    print(f'inference time is {time.time()-start_onnx_time}')
    print(f'vector from onnx model is {torch.tensor(feature_2).shape}')

    if compare(feature_1, feature_2[0]):
        print(f'vector is equal')
    else:
        print(f'vector is not equal, try more')
