
最急降下法による最適化ではなくNewton-Rapson法のような最適化法を使用して,
naiveCD,nomcCD,MLE,MPF,pseudo LEMの各評価関数に大して問題を解いている。
ただしこのディレクトリのみNewtonというディレクトリの中にNewton-Rapson法で解いたコードが入れてある。

また11/09の藤堂先生の助言より、CD法のバイアスをとりのぞけるかもしれない(？)手法に関してテストしてみた。
バイアスを無くせる可能性のある手法とは以下のような手法である。
藤堂先生は初めに、このような考えから始めた。
\partial CD = <g>0 - <g>1
であったが、標本について真の確率分布で平均をとったときにパラメータに一致しない
E[\hat\theta] \not= \theta_0
<< >>:標本についての平均とする
<< <g>0 - <g>1 >> \not= 0 
であるが
<< <g>0 >> = 0
<< <g>1 >> = 0
である。
上で、\partial CD の標本平均が０になることと不偏推定になることが同値になるかは、詳細な議論が必要ではある。

解決の方法としてこのような手法を提示していた。
標本を半分に分けて(group-1, group-2とする。いずれも同一の確率分布から出現している)、group-1を経験分布による平均量を作る操作に使用し、group-2を経験分布から1MC-step遷移した平均量を計算するのに使用する。
>subset-cd.pyというファイル名

