import cv2
from deepface import DeepFace
import os
import sys

def main() :
    print('[START]', flush=True)
    # 이미지 경로
    # reference_path = input("Reference Image File: ")
    # images_folder = input("Target Image Folder: ")
    # output_folder = input("Outut Folder: ")
    # model = 'ArcFace'
    # models = [
    #     "ArcFace",
    #     "DeepFace",
    #     "DeepID",
    #     "Dlib",
    #     "Facenet",
    #     "Facenet512",
    #     "GhostFaceNet"
    #     "OpenFace",
    #     "SFace",
    #     "VGG-Face"
    # ]
    args = sys.argv[1:]
    if len(args) != 6:
        raise Exception(f"[ARGERROR] Arguments Error. 6 Arguments(reference_path, images_folder, output_folder, model, width_times, height_times) needed. Current Argument: {str(args)}")
    reference_path, images_folder, output_folder, model, width_times, height_times = args
    width_times = float(width_times)
    height_times = float(height_times)

    os.makedirs(output_folder, exist_ok=True)

    # 레퍼런스 이미지 얼굴 검출 및 비교
    reference_img = cv2.imread(reference_path)
    image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    for img_file in image_files:
        img_path = os.path.join(images_folder, img_file)
        compare_img = cv2.imread(img_path)
        print(f"[TARGET] {img_path}", flush=True)
        # DeepFace 얼굴 비교
        result = DeepFace.verify(img1_path=reference_img,
                                 img2_path=compare_img,
                                 detector_backend='retinaface',
                                 model_name=model)

        if result['verified']:
            print(f"[MATCH] {img_file}", flush=True)

            # 얼굴 좌표 추출 (DeepFace)
            try:
                faces = DeepFace.extract_faces(img_path=compare_img, detector_backend='retinaface')
                if faces:
                    for face_info in faces:
                        # 얼굴 영역 추출
                        facial_area = face_info['facial_area']
                        x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']

                        # 패딩 설정 (얼굴 중심을 기준으로 크롭)
                        padding_x = int(w * width_times)   # x좌표 2.5배
                        padding_y = int(h * height_times)      # y좌표 3배

                        # 얼굴의 중앙 좌표
                        center_x = x + w // 2
                        center_y = y + h // 2

                        # 크롭된 이미지의 중앙에서 약 1/5 지점으로 여백 추가
                        new_x = max(0, center_x - padding_x // 2)
                        new_y = max(0, center_y - padding_y // 2)
                        new_w = min(compare_img.shape[1] - new_x, padding_x)
                        new_h = min(compare_img.shape[0] - new_y, padding_y)

                        # 크롭
                        cropped_img = compare_img[new_y:new_y + new_h, new_x:new_x + new_w]
                        output_path = os.path.join(output_folder, f"cropped_{img_file}")
                        cv2.imwrite(output_path, cropped_img)
                        print(f"[SAVED] {output_path}", flush=True)
                        break
            except Exception as e:
                print(f"[FAIL] {img_file}: {e}", flush=True)
        else:
            print(f"[NOMATCH] {img_file}", flush=True)


if __name__ == "__main__":
    main()