# 自然言語解析についてのリンク集

## 自然言語処理一般

自然言語処理（NLP）ってなんだろう？  
https://qiita.com/MahoTakara/items/b3d719ed1a3665730826  

## 各種技法について

※自然言語処理の中に、機械学習を使うものとそうでないものがある。  
　目的に応じて適切なものを選択することが肝要。  

・TF-IDF(Term Frequency and Inverse Document Frequency) ※非機械学習  
単語の「その文書内での出現頻度」と「文書全体での出現頻度」に着目した方法。  
「その文書内での出現頻度」が高く、かつ「文書全体での出現頻度」が低い単語についてスコアが高くなる。  
このスコアを任意の単語リストで並べると、その文書の特徴を示す文書ベクトルができる。  

TF-IDFを使って企業紹介ページから各企業の特徴語を抽出してみた  
https://qiita.com/tfujiwar/items/8ea43aaaebf2ee7cf335  
TF-IDF Cos類似度推定法  
https://qiita.com/nmbakfm/items/6bb91b89571dd68fcea6  

・LDA(Latent Dirichlet Allocation) ※機械学習  
文書中に出てくる単語を機械学習の手法を使っていくつかの「トピック」に分類するもの。  
※トピックの数は手動で設定する。  
文書に含まれる単語から、その文書が各トピックに属する確率を算出し、同じトピックに属する確率の高い文書は類似しているというふうに考えることができる。  

データ解析: LDA  
https://openbook4.me/projects/193/sections/1153  

・Word2Vec(Doc2Vec) ※機械学習  
単語をベクトルで表現する手法。  
特徴として「王様」- 「男」+ 「女」= 「女王」というような演算を行うことができる。  
文書に対してベクトル化を行うdoc2vecも存在する。

Word2Vec：発明した本人も驚く単語ベクトルの驚異的な力  
https://deepage.net/bigdata/machine_learning/2016/09/02/word2vec_power_of_word_vector.html  
word2vec（Skip-Gram Model）の仕組みを恐らく日本一簡潔にまとめてみたつもり  
http://www.randpy.tokyo/entry/word2vec_skip_gram_model  
絵で理解するWord2vecの仕組み  
https://qiita.com/Hironsan/items/11b388575a058dc8a46a  

・クラスタリング ※機械学習  
ベクトル化したデータを分類するもので、Word2Vecなどの結果に適用することで文書の分類に使用することができる。  
階層型と非階層型の手法がある。  
　階層型手法: ウォード法 など  
　非階層型手法: k平均法 など  

ビッグデータ解析にも使われるクラスタリングを解説！  
https://udemy.benesse.co.jp/ai/clustering.html  
クラスタリング手法のクラスタリング  
https://qiita.com/suecharo/items/20bad5f0bb2079257568  

<文書のクラスタリングの事例>  
Doc2Vecを使って小説家になろうで自分好みの小説を見つけたい話  
https://blog.aidemy.net/entry/2018/10/16/125954  
※Doc2Vecでベクトル化し、ウォード法で分類  

## 文章解析のサービスいろいろ

・ワトソン(IBM)  
Natural Language Classifier  
https://www.ibm.com/watson/jp-ja/developercloud/nl-classifier.html  
IBM Cloudで一部無料で使用できる模様。  

・ZINRAI(富士通)  
FAQ検索  
http://www.fujitsu.com/jp/solutions/business-technology/ai/ai-zinrai/services/platform/faq-search/index.html  
文章分類（自然文解析）  
http://www.fujitsu.com/jp/solutions/business-technology/ai/ai-zinrai/services/platform/text-classification/index.html  

・Cloud Natural Language(Google)  
https://cloud.google.com/natural-language/?hl=ja  

・TRAINA(野村総研)  
https://www.traina.ai/  

・COTOHA API Portal(NTT Com)  
https://api.ce-cotoha.com/contents/index.html  
※検証用環境なら無料で使用できる。商用環境は13万円～。  
