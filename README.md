Synbio2024 早稲田チーム　コードとデザインのアイデア
240610 nishipippi

早稲田チームは今回以下の4つのコードによって変異体の活性を評価した。
synbio_re.py
dupdelete_ver2.py
mutationcombo_ver2.py
knn_nishi_with_GPU.py

それぞれのコードが行う内容を下記に記す。
1.	synbio_re.py
古典的なアミノ酸変異の書き方(例:A129D)を野生型シーケンスに適用し、指定のアミノ酸を書き換えるコード。これによってデータセットを機械学習に用いやすい（ACDEFGHIKLMNPQRSTVWYによって構成される）アミノ酸シーケンスに変換する。
2.	dupdelete_ver2.py
このプログラムは変異点が重複している変異を削除するものである。例えばA129DとA129GがありA129Gの活性がより高い場合、データセットからA129Dの行を削除する。今回早稲田チームでは野生型から1変異したもの及び2変異したもので活性が野生型よりも向上した変異データをすべて抜き出し、このプログラムで処理することによってより効果の高い変異のみをデータセットに残した。
3.	mutationcombo_ver2.py
このプログラムはdupdelete_ver2.pyで処理された変異のデータセットを取得し、それらを最大3種類組み合わせる。野生型との活性の違いをscoreとし、組み合わせた変異ごとにscoreを足し合わせ、scoreが上位100個である変異を出力する。
4.	knn_nishi_with_GPU.py
5.	このプログラムはk近傍法を使い未知の配列の活性を推定する。データセットとして与えられたアミノ酸データをsynbio_re.pyを用いてアミノ酸シーケンスに変換し、それをone-hot encodingする。そのデータをsklearnのKNeighborsRegressorに学習させ、そのモデルによってmutationcombo_ver2.pyで出力された変異体データの活性を推定する。





We, Team Waseda, evaluate the mutant’s activity with following 4 codes.
1.	synbio_re.py
2.	dupdelete_ver2.py
3.	mutationcombo_ver2.py
4.	knn_nishi_with_GPU.py


1.	synbio_re.py
It converts the dataset from the format written in the amino acid mutation points such as A129D, into the format that is easy to use for machine learning, such as MACDEFGHIKL.
2.	dupdelete_ver2.py
This code removes mutations with duplicate mutation points, for example it removes A129D if there are mutants, such as A129D and A129G, which are A 129G is more active than A129D. In this challenge, we investigate all mutants which has one or two mutations and has more activity than that of wild type. Then we chose the mutants that has effectivity mutants.

3.	mutationcombo_ver2.py
This code combines a mutation data set processed by “dupdelete_ver2.py” with a mutant set of 3 or less mutants. This code outputs the difference between mutant and wild type as “score”, and adds “score” for each mutation in the mutant.

4.	knn_nishi_with_GPU.py
It estimates the activity of unknown sequences with k-nearest neighbor method. 
In this code, it converts the mutation data data given as this dataset into amino acid sequence, and it outputs the data with One-hot encoding. These data are trained by KNeighborsRegressor in sklearn. Finally, it estimates the activity of mutants from mutationcombo_ver2.py.

