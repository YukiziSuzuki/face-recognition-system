# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 12:09:36 2023

@author: suzuki
"""

import face_recognition
import cv2
import numpy as np
import PIL


video_capture = cv2.VideoCapture(0)

#認証を行える人数を増やしたい場合は下のコードをコピーして増やし、変数名の変更と画像データのファイル名の入力を行う

# 画像を読み込み、顔の特徴値を取得する
#このプログラムが置いてあるフォルダ内に”images”というフォルダを作り、そこに認証を行いたい人の顔画像を置いておく
#　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　↓に画像のファイル名を入力
suzuki_image = face_recognition.load_image_file("images/suzuki.jpg")
suzuki_face_encoding = face_recognition.face_encodings(suzuki_image)[0]

matida_image = face_recognition.load_image_file("images/matida.jpg")
matida_face_encoding = face_recognition.face_encodings(matida_image)[0]

sekine_image = face_recognition.load_image_file("images/sekine.jpg")
sekine_face_encoding = face_recognition.face_encodings(sekine_image)[0]

ideue_image = face_recognition.load_image_file("images/ideue.jpg")
ideue_face_encoding = face_recognition.face_encodings(ideue_image)[0]


#配列に取り込んだ画像データから抽出した特徴量を代入している。人数の変更を行う場合、上のプログラムで増やした○○_face_encodingを新たに登録する
known_face_encodings = [
    suzuki_face_encoding,
    matida_face_encoding,
    sekine_face_encoding,
    ideue_face_encoding,
]

#上の配列で記録した人物の名前の登録をしている。名前を登録する際は、上の配列で登録した顔画像の順序と合致するように名前を登録すること
known_face_names = [
    "suzuki",
    "matida",
    "sekine",
    "ideue",
]

while True:
    # Webカメラの1フレームを取得、顔を検出し顔の特徴値を取得する
    _, frame = video_capture.read()
    #rgb_frame = frame[:, :, ::-1]
    
    #face_locations = face_recognition.face_locations(rgb_frame)
    #face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # 1フレームで検出した顔分ループする
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # 認識したい顔の特徴値と検出した顔の特徴値を比較する
        matches = face_recognition.compare_faces(
            known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(
            known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # 顔の周りに四角を描画する
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # 名前ラベルを描画
        cv2.rectangle(frame, (left, bottom - 35),
                      (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 2)
    
    #ウィンドウの大きさを変更している　512の数字を変えるとウィンドウの大きさが変化する    
    image_hight, image_width, image_color = frame.shape
    cv2.namedWindow("WebCam", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("WebCam", 512, int(512*image_hight/image_width)) 
    #ここより上の3行はコメントアウトしても動作する
    # 結果を表示する
    cv2.imshow('WebCam', frame)
    #プログラムの終わらせ方、表示されたウィンドウをクリックしてからキーボードの”Q”を押すことでプログラムが終了する。これ以外で終わらせるとエラーが出る可能性があるので注意
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()