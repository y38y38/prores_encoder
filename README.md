# prores_encoder
prores encoder


・書き出しは別途にする。
・全部終わったら書き出す。

・全スレットが終わったことをどう検出するか？

・データを書き出し用と、処理用に2個分持つか。

・

thread数と時間

1 :14
2 :12
3 : 5
4 : 4
5 : 5
6 : 5
7 : 4
8 : 4
12 : 5
16:4 
24:4


2枚目以降は省略できそうなところありそう。

8kで、5秒なので、マルチスレッドで1秒を目指す。

-----------------
1 sliceづつの処理からまとめてに変更する。




