# importando bibliotecas
import os
import csv
import copy
import argparse
import itertools

import cv2 as cv
import numpy as np
import mediapipe as mp

# caso queira usar em Jupyter Notebook, trocando pasta do Jupyter para o do Notebook para poder importar os módulos próprios
cwd = os.getcwd()
csd = os.path.dirname(os.path.realpath("__file__"))
os.chdir(csd)

# importando modelos de classificação
from model import Mlp_KeyPointClassifier, Svm_KeyPointClassifier

#-------------------------------------------------------------------------------------
# definindo configurações do mediapipe, opencv, e modelo de inferência
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type = int, default = 0) # câmera padrão/default
    parser.add_argument("--width",  help = 'cap width' , type = int, default = 640) # largura do frame 960
    parser.add_argument("--height", help = 'cap height', type = int, default = 480) # altura frame 540

    parser.add_argument('--use_static_image_mode', action='store_true') # mão será detectada em todo frame
    parser.add_argument("--min_detection_confidence", # nível mínimo de confiança para detecção da mão
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)  
    parser.add_argument("--min_tracking_confidence", # nível mínimo de confiança para detecção dos landmarks da mão
                        help='min_tracking_confidence',
                        type=float,
                        default=0.5)

    parser.add_argument("--model", help = 'model type', type = int, default = 1) # modelo selecionado 1: MLP-ANN 2: SVM

    args = parser.parse_args(args=[])

    return args


def main():
    # Recebendo os argumentos com as configurações
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True # usar retângulo limitante

    # Preparação da câmera
    cap = cv.VideoCapture(cap_device)
    
    # Caso o ret no loop esteja retornando falso, ative a linha abaixo
    # e desative a linha acima. Normalmente, reiniciar o PC resolve o problema
    #cap = cv.VideoCapture(cap_device, cv.CAP_DSHOW) 
    
    if not cap.isOpened():
        print("cv2.VideoCapute.isOpened() retornou 'False'. Captura de video não inicializou.")
        
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Instanciando gravador de vídeo
    #video_writer = cv.VideoWriter('demo.avi', cv.VideoWriter_fourcc(*'MJPG'), 20.0, (cap_width, cap_height))

    # Carregando o modelo de deteção das mãos
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode = use_static_image_mode,
        max_num_hands = 1, # somente uma mão detectada por vez
        min_detection_confidence = min_detection_confidence,
        min_tracking_confidence = min_tracking_confidence,
    )

    # Instanciando classificador de imagem escolhido
    model = args.model
    model_name = ""
    if model == 1:
        model_name = "Tensor Flow MLP-ANN"
        keypoint_classifier = Mlp_KeyPointClassifier() # Tensor Flow MLP-ANN
    elif model == 2:
        model_name = "Sklearn SVM"
        keypoint_classifier = Svm_KeyPointClassifier() # Sklearn SVM
    else:
        print("Modelo não existente. Escolha 1: MLP-ANN ou 2: SVM")

    # Lendo os nomes dos sinais da mão
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]

    # Captura, detecção, processamendo, e classificação
    mode = 0

    while True:
        # Processa tecla (ESC: finaliza)
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode) # seleciona modo ou captura número/identificador do sinal

        # Captura da imagem da câmera
        ret, image = cap.read()
        if not ret:
            print("cv2.read() retornou 'False'. Leitura da imagem falhou.")
            break
        image = cv.flip(image, 1)  # projetando imagem no espelho
        debug_image = copy.deepcopy(image)
        
        # Implementando a detecção da mão
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = True
        results = hands.process(image)
        image.flags.writeable = True

        # Processamento das coordenadas dos landmarks
        # Classificação do sinal da mão
        # Adição de textos e formas à imagem
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Cálculo do retângulo limitante
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                
                # Obtenção das coordenadas landmarks e adição destas a uma lista
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Processamento das coordenadas para coordenadas relativas e normalizadas
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                # Armazena no arquivo de dados de coordenadas dos landmarks
                logging_csv(number, mode, pre_processed_landmark_list)

                # Classificação do sinal da mão
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                # Desenhando retângulo dos limites, os landmarks, e informaões de texto
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                )
        debug_image = draw_info(debug_image, mode, number)

        # Gravando o vídeo
        #video_writer.write(debug_image)

        # Mostrando a imagem processada
        cv.imshow('Reconhecimento de gestos de mao - ' + model_name, debug_image)

    cap.release()
    #video_writer.release()
    cv.destroyAllWindows()

# função para selecionar modo
def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n, modo normal de classificação 
        mode = 0
    if key == 107:  # k, mode de captura de dados para o dataset
        mode = 1

    return number, mode

# função para calcular limites do retângulo
def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x - 10, y - 10, x + w + 10, y + h + 10]

# função para calcular coordenadas dos landmarks
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Determinando coordenadas dos landmarks
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

# função para definir pulso como origem do sistema de coordenadas, 
# e normalizar todas as demais entre -1 e 1
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Definindo o landmark do pulso como coordenada (0,0), origem do sistema de coordenadas
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convertendo as coordenadas dos landmarks para uma lista unidimensional
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalização, ou seja, todos as coordenadas entre [-1,1]
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

# função que adiciona dados ao dataset quando em modo de captura de dados
def logging_csv(number, mode, landmark_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint_dataset.csv'
        with open(csv_path, 'a', newline = "") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])

    return

# função que desenha círculos nos landmarks e linhas entre eles
def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        landmarks_dict = {
        'thumb'         : [ 2,  3,  4],
        'index_finger'  : [ 5,  6,  7,  8],
        'middle_finger' : [ 9, 10, 11, 12],
        'ring_finger'   : [13, 14, 16, 16],
        'little_finger' : [17, 18, 19, 20],
        'palm'          : [ 0,  1,  2,  5,  9, 13, 17,  0]
        }
        # Linhas entre os landmarks
        for _, landmarks_index in landmarks_dict.items():
            for ind in range(len(landmarks_index) - 1):
                cv.line(image, tuple(landmark_point[landmarks_index[ind]]), 
                               tuple(landmark_point[landmarks_index[ind + 1]]), (  0,   0,   0), 6)
                cv.line(image, tuple(landmark_point[landmarks_index[ind]]), 
                               tuple(landmark_point[landmarks_index[ind + 1]]), (255, 255, 255), 2)

        # Circulos nos pontos principais dos dedos
        for _, landmark in enumerate(landmark_point):
            if ind in [4, 8, 12, 16, 20]:  # ponta dos dedos
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),-1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (  0,   0,   0), 1)
            else: # demais pontos
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (  0,   0,   0), 1)

    return image

# função que desenha o retângulo externo
def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)

    return image

# função que adiciona texto informando qual mão e sinal são identificados
def draw_info_text(image, brect, handedness, hand_sign_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0),-1)
    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    return image

# função que adiciona texto informando quando está no modo de gravação de dados,
# além de qual número/identificador está sendo teclado
def draw_info(image, mode, number):
    mode_string = ['Captura de dados de landamarks da mao']
    if mode == 1:
        cv.putText(image, "MODO: " + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    
    return image

#----------------------------------------------------------------------------------
# inicializando o programa
if __name__ == '__main__':
    main()
    os.chdir(cwd)