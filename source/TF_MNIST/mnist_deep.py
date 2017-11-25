from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# 重みを初期化する関数
# "truncated_normal"関数でTensorを正規分布かつ標準偏差の2倍までの
# ランダムな値で初期化
# ここでは-0.2～0.2の範囲の値をランダムで返却する
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

# バイアスを初期化する関数
# constant関数で定数で初期化
# ここでは0.1で初期化する
# この後ReLUニューロン(値が0以下なら0、0より大きければその値を返す)を使用するため
# 小さい正の数で初期化するのがgood practiceであるらしい
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# 畳み込みを行う関数
# tf.nn.conv2d関数は以下の引数を取る
#   第1引数: input
#     [batch, in_height, in_width, in_channels]の4次元テンソル
#   第2引数: filter
#     畳込みでinputテンソルとの積和に使用する「重み」
#     [filter_height, filter_width, in_channels, channel_multiplier] のテンソル
#     channel_multiplierだけchannel数が拡張される
#   第3引数: strides
#     filterの適用範囲を何画素ずつ移動させて計算するか(=ストライド)を示す
#     指定は[1, stride, stride, 1]とする必要がある
#   第4引数: padding
#     inputの周囲をパディングする方法を指定
#     'SAME'はフィルタ分を補うように0パディングすることを示す
#     (つまりフィルタ適用後も画素数は変化しない)
# つまりここではストライド1、画素数保持のパディングで畳み込みを行う
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# 最大プーリング用の関数
# tf.nn.max_pool関数は以下の引数を取る
#   第1引数: value
#     畳み込み層の出力データ
#   第2引数: ksize
#     プーリングサイズを指定
#   第3引数: strides
#     filterの適用範囲を何画素ずつ移動させて計算するか(=ストライド)を示す
#     指定は[1, stride, stride, 1]とする必要がある
#     プーリングサイズとストライドには同じ値を設定するのが一般的
#   第4引数: padding
#     inputの周囲をパディングする方法を指定
#     'SAME'は0パディングすることを示す
# つまりここでは2*2のプーリングサイズで、ストライド2で適用して最大プーリングを取得する
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# 手書き文字データを入手
# one_hotはある要素のみ1、それ以外は0の表現方法
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 手書き文字は28*28(=784ピクセル)のモノクロ画像
# これを784次元のベクトルとして扱う
# 各要素はピクセルの濃度を0(白)～1(黒)としている(ゆえにfloat型を用いる)
# Noneとなっている個所は次元が任意であることを示す
# つまりここでは、任意の数の画像を扱うことを示す
x = tf.placeholder(tf.float32, shape=[None, 784])

# 入力を4次元テンソルに変換
#   1次元: 画像の枚数(任意の枚数なので-1として次元を削減)
#   2次元: ピクセルの高さ（28px)
#   3次元: ピクセルの幅（28px)
#   4次元: カラーチャネル(モノクロなので1チャネル)
# 先のチュートリアルではピクセルを1列に並べて考えていたが
# こちらではピクセルを平面として、さらに濃さの次元を加えたものとして扱う
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 第1層の重みとバイアスを設定
# 重みとバイアスの初期化には先ほど定義した関数を使用
# 重みには5*5*1のフィルタを使用、アウトプットは32チャンネルにする
# バイアスもアウトプットに合わせて32にする
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# 第1層の畳み込みを実施
# 変換した入力(x_image)を重みで畳み込み、バイアスを加算し
# さらに活性化関数としてReLU関数(値が0以下なら0、0より大きければその値を返す)を適用
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

# 第1層のプーリングを実施
# 畳み込み層の出力に対し、2*2の最大プーリングを実施
h_pool1 = max_pool_2x2(h_conv1)

# 第2層の重みとバイアスを設定
# 第1層でアウトプットを32チャネルとしている
# 第2層ではさらに64チャネルに拡張
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

# 第2層の畳み込みを実施
# 第1層のプーリング層の出力を同様に畳み込む
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

# 第2層のプーリングを実施
h_pool2 = max_pool_2x2(h_conv2)

# 全結合層
# この時点で画像サイズは7*7に減じられている
# アウトプットは第2層出力時点で64チャネル
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

# 第2層のプーリング出力を、先のチュートリアルのように1次元に変換
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

# 重みを(もう畳み込みではなく)乗算しバイアスを加算
# さらに活性化関数を適用
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 過学習を抑止するためにドロップアウトを適用
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 読み出し層の重みとバイアスを設定
# 先のチュートリアルと同じく、0～9のどの数字かの確率を算出する
# 全結合層の出力は1024チャネル、これを数字の種類(10種類)に重み付けする
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# 1024チャネルに対して重みを乗算しバイアスを加算
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# 交差エントロピーを算出するため、y_に正解情報を取得する
# [<0である確率>, <1である確率>, ...]となっており
# 正解箇所には1、それ以外には0が入っている(前述のone_hot表現)
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# 交差エントロピーを算出
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

# 学習を実施
# ここではAdam法を使用して交差エントロピーの最小化を行う
# 学習率を1e-4=0.0001に設定
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 確率の算出結果(y)と正解データ(y_)から
# argmaxを使って最も高い値を求める
# つまりyからは予測した数字、y_からは正解の数字が取れる
# それが一致しているかどうかをcorrect_predictionに格納
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

# correct_predictionはtrue/falseのリストなので
# 正解率を算出するためtrue=1、false=0の浮動小数点型数値として平均をとる
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# セッション開始
with tf.Session() as sess:
  # 変数を初期化
  sess.run(tf.global_variables_initializer())
  # 学習を20000回実施
  for i in range(20000):
    # 手書き文字データから50件を抽出
    batch = mnist.train.next_batch(50)
    # 100回ごとに(学習用データに対する)正解率を表示
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: batch[0], y_: batch[1], keep_prob: 1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy))
    # 学習を実行
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

  # 学習完了後、テストデータを使って正解率を検証
  print('test accuracy %g' % accuracy.eval(feed_dict={
      x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
