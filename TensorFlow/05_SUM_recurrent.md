# 再帰的ニューラルネットワークを使用した合計値算出

再帰的ニューラルネットワークの理解を深めるため、サンプルコードをもう一つ、  
以下のサイトに掲載されているものを見ていく。  
TensorFlowのRNNを基本的なモデルで試す  
https://qiita.com/yukiB/items/f6314d2861fc8d9b739f  

TensorFlowのバージョンがやや古いので、まずはダウンロードしたソースコードのエラー取りから。。。  
~~~
ValueError: Tensor conversion requested dtype int32 for Tensor with dtype float32: 'Tensor("inference/add:0", shape=(?, 80), dtype=float32)'
TypeError: Input 'split_dim' of 'Split' Op has type float32 that does not match expected type of int32.
~~~
