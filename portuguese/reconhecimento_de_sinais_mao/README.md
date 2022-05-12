# Reconhecimento de gestos/sinais das mãos, usando Mediapipe e Opencv
Classifica o sinal da mão usando MediaPipe (Python version).<br> Este é um programa 
de exemplo que reconhece sinais da mão e dos dedos com uma simples rede neural artifical (ANN) usando MLP (Multilayer Perceptron), ou um aprendizado de máquina supervisionado (supervised ML), neste caso SVM (Support Vector Machine), a partir de pontos chaves ("key points") reconhecidos da mão.
<br> ❗ _️**Este programa foi adpatado do [repositório original](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe).**_ ❗
<br> 


Este repositório tem o seguinte conteúdo.
* Programa exemplo
* Modelo de reconhecimento dos sinais (TFLite, Sklearn-Joblib)
* Dataset e notebook para inferência/aprendizado dos modelos. Obs: 1 único dataset e 1 notebook para cada modelo (MLP-ANN e SVM)

# Requisitos
* mediapipe 0.8.1
* OpenCV 3.4.2 ou posterior
* Tensorflow 2.3.0 ou posterior <br>tf-nightly 2.5.0.dev ou posterior (Apenas ao criar o TFLite para um modelo LSTM)
* scikit-learn 0.23.2 ou posterior (Apenas para estimar a matriz de confusão e inferir o modelo SVM) 
* matplotlib 3.3.2 ou posterior (Apenas para estimar a matriz de confusão)

# Demo
Para rodar o demo usando a webcam.
```bash
python app.py
```

As seguintes opções podem ser especificadas ao rodar a demo.
* --device<br>Especifica o número do aparelho da câmera (Default：0)
* --width<br>Largura da imagem no momento da captura da câmera (Default：640)
* --height<br>Altura da imagem no momento da captura da câmera (Default：480)
* --use_static_image_mode<br>Usar ou não a opção static_image_mode para inferência doMediaPipe (Default：Unspecified)
* --min_detection_confidence<br>Limite do nível de confiança da detecção (Default：0.5)
* --min_tracking_confidence<br>Limite do nível de confiança do tracking das mãos (Default：0.5)
* --model<br>Modelo a ser utilizado na classificação do sinal da mão - 1: MLP-ANN | 2: SVM (Default：1)

# Diretório
<pre>
│  app.py
│  keypoint_classification_mlp.ipynb
│  keypoint_classification_svm.ipynb
│  
└─ model
     └─ keypoint_classifier
          │  keypoint_dataset.csv
          │  keypoint_classifier_mlp.hdf5
          │  keypoint_classifier_mlp.py
          │  keypoint_classifier_mlp.tflite
          │  keypoint_classifier_svm.py
          │  keypoint_classifier_svm.joblib
          └─ keypoint_classifier_label.csv


</pre>
### app.py
Este é o programa exemplo para inferência/classificação.<br>
Além disso, também coleta dados de aprendizado para reconhecimento de sinais de mão.<br>

### keypoint_classification_svm.ipynb 
Este é um notebook para treino de reconhecimento dos sinais da mão, usando SVM.

### keypoint_classification_mlp.ipynb
Este é um notebook para treino de reconhecimento dos sinais da mão, usando MLP-ANN.

### model/keypoint_classifier
Este diretório armazena os arquivos relacionados ao reconhecimento de sinais de mão.<br>
Os seguintes arquivos são armazenados:
* Dados de treino (keypoint_dataset.csv)
* Modelos treinados (keypoint_classifier_mlp.tflite e keypoint_classifier_svm.joblib)
* Rótulo dos dados (keypoint_classifier_label.csv)
* Módulos de inferência (keypoint_classifier_svm.py e keypoint_classifier_mlp.py)

# Treino
Reconhecimento de sinal de mão pode ter dados modificados e adicionados para retreino do modelo.

### Treino do Reconhecimento de Sinal da Mão
#### 1. Coleta de Dados de Treino
Pressione "k" para entra no modo para salvar os "key points" (mostrado como'「MODO: Captura de dados de landamarks da mão」）<br><br>
<img src = https://user-images.githubusercontent.com/88405559/168139450-afceebb0-3442-40c7-a41b-0429e577ac92.png width="60%"><br><br>
Para sair do modo de captura basta pressionar "n".
Se o usuário pressionar "0" to "9", os "key points" serão adicionados a "model/keypoint_classifier/keypoint_dataset.csv" como abaixo.<br>
Primeira coluna: número pressionado (usado como ID da classe), segunda coluna e subsequentes: coordenadas do "key point".<br><br>
<img src="https://user-images.githubusercontent.com/37477845/102345725-28d26280-3fe1-11eb-9eeb-8c938e3f625b.png" width="80%"><br><br>
As coordenadas do "key point" são as que passaram pelo seguinte pré-processamento até up to ④.<br><br>
<img src="https://user-images.githubusercontent.com/37477845/102242918-ed328c80-3f3d-11eb-907c-61ba05678d54.png" width="80%">
<img src="https://user-images.githubusercontent.com/37477845/102244114-418a3c00-3f3f-11eb-8eef-f658e5aa2d0d.png" width="80%"><br><br>
No estado inicial, 4 tipos de dados de aprendizado estão incluídos: letra A (ID da classe: 0), letra B (ID da classe: 1), letra C (ID da classe: 2), letra D (ID da classe: 3).<br>
Se necessário, adicione ID 4 ou maior, ou delete os dados existentes do arquivo csv para preparar os dados de treino.<br><br>
<img src=https://user-images.githubusercontent.com/88405559/168140568-191b2140-1ac1-46c3-9415-fd5e87b8dbb7.png width="60%">
<img src=https://user-images.githubusercontent.com/88405559/168140576-ae93ccb0-48d8-4765-8e16-1a0fe361e662.png width="60%">
<img src=https://user-images.githubusercontent.com/88405559/168140580-1febacf9-394d-40dc-80fe-c700aefc0918.png width="60%">


#### 2. Treino do Modelo
Abra "[keypoint_classification_mlp.ipynb](keypoint_classification_mlp.ipynb)" para modelo MLP-ANN, ou "[keypoint_classification_svm.ipynb](keypoint_classification_svm.ipynb)" para modelo SVM, no Jupyter Notebook e execute do topo até a última célula.<br>
Para mudar o número de classes de treino, mude o valor "NUM_CLASSES = 4" <br>e modifique o os rótulos de "model/keypoint_classifier/keypoint_classifier_label.csv" de forma apropriada.<br><br>

# Referência
* [MediaPipe](https://mediapipe.dev/)

# Autor
[Kazuhito Takahashi](https://twitter.com/KzhtTkhs)

# Alterações, Adaptações, Adições e Melhoras
[Pablo Oliveira - LindedIn](https://br.linkedin.com/in/pablo-oliveira-msc-cqf-88365716)
[Pablo Oliveira - GitHub](https://github.com/pablofrioli)
 
# Licença
reconhecimento_de_sinais_mao está sob licença [Apache v2 license](LICENSE).
