from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tensorflow.contrib import rnn

# 再帰的ニューラルネットワークのモデルを定義
def RNN(x):
    x = tf.unstack(x, 28, 1)

    # LSTMセルを定義する
    lstm_cell = rnn.BasicLSTMCell(128, forget_bias=1.0)

    # モデルの定義。各タイムステップの出力値と状態が返される
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # 変数定義
    weight = tf.Variable(tf.random_normal([128, 10]))
    bias = tf.Variable(tf.random_normal([10]))

    return tf.matmul(outputs[-1], weight) + bias


# 手書き文字データを入手
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 手書き文字は28*28(=784ピクセル)のモノクロ画像
# 各要素はピクセルの濃度を0(白)～1(黒)としている(ゆえにfloat型を用いる)
# Noneとなっている個所は次元が任意であることを示す
# つまりここでは、任意の数の画像を扱うことを示す
x = tf.placeholder("float", [None, 28, 28])

# 交差エントロピーを算出するため、y_に正解情報を取得する
# [<0である確率>, <1である確率>, ...]となっており
# 正解箇所には1、それ以外には0が入っている(前述のone_hot表現)
y_ = tf.placeholder("float", [None, 10])

preds = RNN(x)

# 誤差を交差エントロピーで算出
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=y_))

# Adam法を使用して交差エントロピーの最小化を行う
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

# 確率の算出結果(preds)と正解データ(y_)から
# argmaxを使って最も高い値を求める
# つまりpredsからは予測した数字、y_からは正解の数字が取れる
# それが一致しているかどうかをcorrect_predに格納
correct_pred = tf.equal(tf.argmax(preds, 1), tf.argmax(y_, 1))

# correct_predはtrue/falseのリストなので
# 正解率を算出するためtrue=1、false=0の浮動小数点型数値として平均をとる
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 手書き文字データから120件を抽出して学習を行う
# 学習に用いたデータが10万件を超えたら学習完了とする
batch_size = 128
n_training_iters = 100000

with tf.Session() as sess:
    # 変数を初期化
    sess.run(tf.global_variables_initializer())
    # stepに実行回数を設定し、10万件の学習を行うまでループ
    step = 1
    while step * batch_size < n_training_iters:
        # 手書き文字データから抽出し120剣のデータを取得
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # next_batchで返されるbatch_xは[batch_size, 784]のテンソルなので
        # batch_size×28×28に変換
        batch_x = batch_x.reshape((batch_size, 28, 28))
        # 学習を実行
        sess.run(optimizer, feed_dict={x: batch_x, y_: batch_y})
        # 10回ごとに現状の正解率と誤差を出力
        if step % 10 == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y_: batch_y})
            loss = sess.run(cost, feed_dict={x: batch_x, y_: batch_y})
            print('step: {} / loss: {:.6f} / acc: {:.5f}'.format(step, loss, acc))
        step += 1

    # 学習完了したので、最終的な正解率を算出
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, 28, 28))
    test_label = mnist.test.labels[:test_len]
    test_acc = sess.run(accuracy, feed_dict={x: test_data, y_: test_label})
    print("Test Accuracy: {}".format(test_acc))
