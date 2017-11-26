# 再帰的ニューラルネットワークを使用した手書き文字認識

TensorFlow公式サイトのチュートリアルにも、再帰的ニューラルネットワークについてのものがある。  
Recurrent Neural Networks  
https://www.tensorflow.org/tutorials/recurrent

日本語訳はこちらを参照。  
https://media.accel-brain.com/tensorflow-recurrent-neural-networks/

ただ、言語モデリングを扱うため、言語のベクトル表現を先に学習した方がよいらしい。  
※チュートリアルもその順序になっている。  
Vector Representations of Words  
https://www.tensorflow.org/versions/master/tutorials/word2vec

webでは手書き文字認識を題材に再帰的ニューラルネットワークの実装を行った例がある。  
今回はこちらの内容を見ていくことにする。  
RNN：時系列データを扱うRecurrent Neural Networksとは  
https://deepage.net/deep_learning/2017/05/23/recurrent-neural-networks.html

※他にも数列の合算を題材にした例がある。  
TensorFlowのRNNを基本的なモデルで試す  
https://qiita.com/yukiB/items/f6314d2861fc8d9b739f

処理の内容をコメントしたソースコードはこちら  
[mnist_recurrent.py](../source/TF_MNIST/mnist_recurrent.py)  
