# 準備メモ

## はじめに

本当はChainerの方がわかりやすいということだったが、  
ChainerはどうやらWindowsがサポートされていない(2017年10月現在)ようだったので
TensorFowをインストール。  
(Linux仮想環境を作る手もあるが、PCスペック的に心配。Windowsでもできないことはなさそうだけれど、変な癖がついてしまってはいけないと思って。)

## インストール

TensorFlow公式サイトに書いてある通りの手順を実施。  
https://www.tensorflow.org/install/install_windows  
GPUは搭載していないのでCPU onlyで。  

Python環境構築にanacondaを使った方法が書いてあったが、今回はminiconda3 4.4.30(64-bit)を使用。  
(minicondaはanacondaの最小構成版。ストレージに不安があったので。)  
https://conda.io/miniconda.html  
Python3.6/Windows64bit用を取得してインストール。

Python実行環境とTensorFlowをインストールしたら、スタートメニューから"Anaconda Prompt"を起動してコマンド実行する。  
(基本なのだろうけれど、コマンドプロンプトでやってこけたので。。。)  
ここから`activate tensorflow`を実行したらTensorFlow実行環境が起動できる。  

公式にあるHello worldを実行したら以下のエラーが出たが、特に動作に問題はなさそう。

~~~
2017-11-04 01:09:12.091023: W C:\tf_jenkins\home\workspace\rel-win\M\windows\PY\35\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
~~~
