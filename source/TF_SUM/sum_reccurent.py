import tensorflow as tf
import numpy as np
import random
from tensorflow.contrib import rnn

num_of_input_nodes = 1
num_of_hidden_nodes = 80
num_of_output_nodes = 1
length_of_sequences = 10
num_of_training_epochs = 5000
size_of_mini_batch = 100
num_of_prediction_epochs = 100
learning_rate = 0.01
forget_bias = 0.8
num_of_sample = 1000

# 生成した数列から学習用のデータを抽出する
# 引数
#  batch_size: Xから何個の数列を抽出するか指定(今回は100個)
#  X: 抽出元となる数列の集まり
#  t: 抽出元となる数列の合計値のリスト
#   ※Xとtはcreate_data関数で生成したものを使っている
# 戻り値
#  xs: 抽出した数列(batch_size個分)
#  ts: 抽出した数列の各々の合計値
def get_batch(batch_size, X, t):
    # どの数列を取得するかをrnumに設定
    rnum = [random.randint(0, len(X) - 1) for x in range(batch_size)]
    # 数列の各要素と合計値を、抽出用の数列に格納
    xs = np.array([[[y] for y in list(X[r])] for r in rnum])
    ts = np.array([[t[r]] for r in rnum])
    return xs, ts


# 合計値算出に使用する数列と合計値を生成する
# 数列は第1引数に指定した数だけ生成する
# 引数
#  nb_of_samples: 数列をいくつ作成するかを指定(今回は1000個)
#  sequence_len: 数列の要素数(今回は10)
# 戻り値
#  X: 生成した数列(nb_of_samples個分)
#  t: 数列の合計値のリスト
def create_data(nb_of_samples, sequence_len):
    X = np.zeros((nb_of_samples, sequence_len))
    for row_idx in range(nb_of_samples):
        # rand関数で要素数の分だけ乱数を生成する
        # 乱数は0～1の範囲なので、around関数で四捨五入して
        # 戻り値となるXに格納すする
        X[row_idx, :] = np.around(np.random.rand(sequence_len)).astype(int)
    # np.sum関数で数列ごとの合計値を算出
    t = np.sum(X, axis=1)
    return X, t

# テスト用の数列を生成する
# 引数
#  nb_of_samples: 数列をいくつ作成するかを指定(今回は100個)
# 戻り値
#  1つめ: 生成した数列(nb_of_samples個分)
#  2つめ: 数列の合計値のリスト
# ※get_batch関数が学習用で、make_prediction関数がテスト用となっている
def make_prediction(nb_of_samples):
    sequence_len = 10
    xs, ts = create_data(nb_of_samples, sequence_len)
    return np.array([[[y] for y in x] for x in xs]), np.array([[x] for x in ts])

# 推論(inference)を行う
# 引数
#  input_ph: 学習に使用する数列
#  istate_ph: 再帰的ニューラルネットワークの状態初期値
# 戻り値
#  1つめ: 合計値の予測
#  2つめ: ネットワークの状態
#  3つめ: 使用した重みとバイアス
def inference(input_ph, istate_ph):
    with tf.name_scope("inference") as scope:
        # 入力と隠れ層の間の重みを設定
        # 今回使用するのは、入力が1ノード(数列から抽出した値)、隠れ層が80ノードの構造となっている
        # 初期値は、標準偏差の2倍までの正規分布からランダムな値を設定
        weight1_var = tf.Variable(tf.truncated_normal(
            [num_of_input_nodes, num_of_hidden_nodes], stddev=0.1), name="weight1")
        # 隠れ層と出力の間の重みを設定
        # 隠れ層は80ノード、出力は1ノード(合計値)
        weight2_var = tf.Variable(tf.truncated_normal(
            [num_of_hidden_nodes, num_of_output_nodes], stddev=0.1), name="weight2")
        # 隠れ層と出力の重みを設定
        bias1_var = tf.Variable(tf.truncated_normal([num_of_hidden_nodes], stddev=0.1), name="bias1")
        bias2_var = tf.Variable(tf.truncated_normal([num_of_output_nodes], stddev=0.1), name="bias2")

        # transposeメソッドでテンソルの順序を入れ替える
        # input_phは[100, 10, 1]となっており、これを[10, 100, 1]に変換(1次元目, 0次元目, 2次元目に並べ替え)
        in1 = tf.transpose(input_ph, [1, 0, 2])
        # 並べ替えた[10, 100, 1]の配列を[n, 1]の配列に成形
        # reshapeの「-1」は次元数を調整するためのワイルドカード
        in2 = tf.reshape(in1, [-1, num_of_input_nodes])
        # 入力から隠れ層へ重みとバイアスを計算
        in3 = tf.matmul(in2, weight1_var) + bias1_var
        # 再度テンソルを10個ごとに区切っていく
        # 最終的に[10, 100]のテンソルになっている(はず)
        in4 = tf.split(axis=0, num_or_size_splits=length_of_sequences, value=in3)

        # LSTMセルを構成
        # 注: state_is_tuple=Falseは非推奨(エラー回避のため今回はこのまま使用)
        cell = rnn.BasicLSTMCell(num_of_hidden_nodes, forget_bias=forget_bias, state_is_tuple=False)
        # 再帰的ニューラルネットワークを構成
        # セルには先ほど構成したLSTMセル、inputにテスト用数列から作成したin4を
        # 戻り値のrnn_output、states_opには最終的なstateが設定される
        rnn_output, states_op = rnn.static_rnn(cell, in4, initial_state=istate_ph)
        # 再帰的ニューラルネットワークの計算結果を合計値の予測に反映
        # 隠れ層の80ノードから合計値を示す1ノードへ重みとバイアスを計算
        output_op = tf.matmul(rnn_output[-1], weight2_var) + bias2_var

        # ここはグラフ描画のためのもの
        w1_hist = tf.summary.histogram("weights1", weight1_var)
        w2_hist = tf.summary.histogram("weights2", weight2_var)
        b1_hist = tf.summary.histogram("biases1", bias1_var)
        b2_hist = tf.summary.histogram("biases2", bias2_var)
        output_hist = tf.summary.histogram("output",  output_op)

        # 戻り値として合計値の予測(output_op)、ネットワークの状態(states_op)、使用した重みとバイアス(results)を渡す
        results = [weight1_var, weight2_var, bias1_var,  bias2_var]
        return output_op, states_op, results


def loss(output_op, supervisor_ph):
    with tf.name_scope("loss") as scope:
        square_error = tf.reduce_mean(tf.square(output_op - supervisor_ph))
        loss_op = square_error
        tf.summary.scalar("loss", loss_op)
        return loss_op


def training(loss_op):
    with tf.name_scope("training") as scope:
        training_op = optimizer.minimize(loss_op)
        return training_op


def calc_accuracy(output_op, prints=False):
    inputs, ts = make_prediction(num_of_prediction_epochs)
    pred_dict = {
        input_ph:  inputs,
        supervisor_ph: ts,
        istate_ph:    np.zeros((num_of_prediction_epochs, num_of_hidden_nodes * 2)),
    }
    output = sess.run([output_op], feed_dict=pred_dict)

    def print_result(i, p, q):
        [print(list(x)[0]) for x in i]
        print("output: %f, correct: %d" % (p, q))
    if prints:
        [print_result(i, p, q) for i, p, q in zip(inputs, output[0], ts)]

    opt = abs(output - ts)[0]
    total = sum([1 if x[0] < 0.05 else 0 for x in opt])
    print("accuracy %f" % (total / float(len(ts))))
    return output

random.seed(0)
np.random.seed(0)
tf.set_random_seed(0)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

# 数列を生成
# 10(=length_of_sequences)個の数字を持つ数列を1000(=num_of_sample)個生成
# Xに生成した数列、tには各数列の合計値を格納
X, t = create_data(num_of_sample, length_of_sequences)

with tf.Graph().as_default():
    # 数列のバッチを保持するプレースホルダー
    input_ph = tf.placeholder(tf.float32, [None, length_of_sequences, num_of_input_nodes], name="input")
    # 数列の合計値を保持するプレースホルダー
    supervisor_ph = tf.placeholder(tf.float32, [None, num_of_output_nodes], name="supervisor")

    istate_ph = tf.placeholder(tf.float32, [None, num_of_hidden_nodes * 2], name="istate")

    output_op, states_op, datas_op = inference(input_ph, istate_ph)
    loss_op = loss(output_op, supervisor_ph)
    training_op = training(loss_op)

    summary_op = tf.summary.merge_all()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        #saver = tf.train.Saver()
        #summary_writer = tf.summary.FileWriter("/tmp/tensorflow_log", graph=sess.graph)
        sess.run(init)

        # 学習を実施
        # num_of_training_epochsの回数分実施する(今回は5000回)
        for epoch in range(num_of_training_epochs):
            # 生成した1000個の数列から100個を抽出
            inputs, supervisors = get_batch(size_of_mini_batch, X, t)
            # プレースホルダーに値を設定
            train_dict = {
                input_ph:      inputs,  # 抽出した100個の数列
                supervisor_ph: supervisors,  # 抽出した各数列の合計値
                istate_ph:     np.zeros((size_of_mini_batch, num_of_hidden_nodes * 2)),  # (0で初期化)
            }
            sess.run(training_op, feed_dict=train_dict)

            # 100回ごとに損失関数の現在値を出力
            if (epoch) % 100 == 0:
                summary_str, train_loss = sess.run([summary_op, loss_op], feed_dict=train_dict)
                print("train#%d, train loss: %e" % (epoch, train_loss))
                #summary_writer.add_summary(summary_str, epoch)
                # 500回ごとに正解率を出力
                if (epoch) % 500 == 0:
                    calc_accuracy(output_op)

        # 学習が完了したので最終的な正解率を出力
        calc_accuracy(output_op, prints=True)
        datas = sess.run(datas_op)
        #saver.save(sess, "model.ckpt")
