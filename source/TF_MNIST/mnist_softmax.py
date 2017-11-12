from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# 手書き文字データを入手
# one_hotはある要素のみ1、それ以外は0の表現方法
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 手書き文字は28*28(=784ピクセル)のモノクロ画像
# これを784次元のベクトルとして扱う
# 各要素はピクセルの濃度を0(白)～1(黒)としている(ゆえにfloat型を用いる)
# Noneとなっている個所は次元が任意であることを示す
# つまりここでは、任意の数の画像を扱うことを示す
x = tf.placeholder(tf.float32, [None, 784])

# 重み(Weights)を設定
# それぞれのピクセルの濃度が、数字判定にどれだけの影響を及ぼすかを示す
# 初期値はすべて0で設定する
W = tf.Variable(tf.zeros([784, 10]))

# バイアス(biases)を設定
# 重みを計算した後に行われる調整
# こちらも初期値を0で設定する
b = tf.Variable(tf.zeros([10]))

# 画像がどの数字であるかの確率を算出
# yはその画像が[<0である確率>, <1である確率>, ...]を示す10次元のベクトル
# 確率とするためにsoftmax関数を適用して、「xである確率」が0～1、
# その総和が1となるようにする
y = tf.nn.softmax(tf.matmul(x, W) + b)

# y_には正解が入る
# yと同様に[<0である確率>, <1である確率>, ...]となっているため
# 正解箇所には1、それ以外には0が入っている(前述のone_hot表現)
y_ = tf.placeholder(tf.float32, [None, 10])

# 算出した確率と正解との誤差を交差エントロピーで導出
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# 学習率0.5で勾配降下(GradientDescent)法を用い、交差エントロピーを最小化
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# セッション開始
# 先に設定した変数(重み/バイアス)を初期化
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# 学習を1000回繰り返し実行
# 手書き文字データの中からランダムで100個を抽出し学習を行う
# batch_xsには訓練データ、batch_ysには訓練データに対する正解が入っている
# これをプレースホルダーであるxとy_に設定し、学習(train_step)を実行
# この結果、重み(W)とバイアス(b)が最適と思われる値に調整される
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 確率の算出結果(y)と正解データ(y_)から
# argmaxを使って最も高い値を求める
# つまりyからは予測した数字、y_からは正解の数字が取れる
# それが一致しているかどうかをcorrect_predictionに格納
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# correct_predictionはtrue/falseのリストなので
# 正解率を算出するためtrue=1、false=0の浮動小数点型数値として平均をとる
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 調整した重み(W)とバイアス(b)を使用して、テストデータでの検算を行う
# 検算の結果はコンソールに出力する
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
