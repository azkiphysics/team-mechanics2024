## 力学系ゼミ 第6回 プログラミング課題
### 概要
第6回は，数値計算法として，ニュートン法，オイラー法，ルンゲクッタ法を扱います．力学系に限らず宇宙系，複雑系でも頻繁に使用するアルゴリズムですので，是非この機会に使えるようにしていただければと思います．

課題を作成する際は，hw6ディレクトリ内にフォルダ(フォルダ名: `(名前)`)を作成し (e.g., `ito`)，作成したフォルダ内に課題ごとのファイルを`answer(課題番号).py`として作成してください．(e.g., `answer1.py`, `answer2-1.py`)

課題を作成する際は，必ずブランチを切り，作成したブランチ上で作業を行うようにしてください ([ブランチの作成](https://github.com/azkiphysics/team-mechanics2024?tab=readme-ov-file#ブランチの作成))．

課題が作成できたら，GitHub上でプルリクエストを開き，伊藤(ユーザー名: azkiphysics)にマージの許可を得てください．伊藤が提出した課題のコードレビューを行い，コードの修正をしていただきます．修正が完了したらマージを行い，その週の課題は終了となります．

### 準備
今回は，シミュレーション動画を作成するために，opencvと呼ばれるライブラリを使用します．なので，課題を行う前にまず以下のコマンドを仮想環境内で実行して，opencvをインストールしておいてください．

```zsh
pip install opencv-python opencv-contrib-python
```

### 課題1 (数値計算法の実装)

### 課題2 (数値計算法の比較)

## 解説
### ニュートンラフソン法
ニュートンラフソン法は，ある方程式 $\boldsymbol{f}(\boldsymbol{x}) = \boldsymbol{0}$ の解を数値的に求めるための手法となります．

以下では，ニュートンラフソン法のアルゴリズムを導出方法について説明します．まず， $\boldsymbol{f}(\boldsymbol{x}+\Delta \boldsymbol{x})$ を1次の項までテイラー展開すると，次式が得られます．

$$
\boldsymbol{f}(\boldsymbol{x}+\Delta \boldsymbol{x}) \simeq \boldsymbol{f}(\boldsymbol{x}) + \frac{\partial \boldsymbol{f}}{\partial \boldsymbol{x}}\Delta \boldsymbol{x}
$$

ニュートンラフソン法では, $\boldsymbol{f}(\boldsymbol{x}+\Delta \boldsymbol{x}) = \boldsymbol{0}$ となるように $\Delta x$ を決定します．上式と $\boldsymbol{f}(\boldsymbol{x}+\Delta \boldsymbol{x}) = \boldsymbol{0}$ を用いて $\Delta x$ を求めると，次のようになります．

$$
\Delta \boldsymbol{x} = -\Bigg(\frac{\partial \boldsymbol{f}}{\partial \boldsymbol{x}}\Bigg)^{-1}\boldsymbol{f}(\boldsymbol{x})
$$

上記の $\Delta\boldsymbol{x}$ を用いて，以下のように $\boldsymbol{x}$ を更新します．

$$
\boldsymbol{x} \leftarrow \boldsymbol{x} + \Delta \boldsymbol{x}
$$

この更新式を $\||\Delta\boldsymbol{x}\|| < \epsilon$ ($\epsilon$ : 閾値)となるまで繰り返し適用し，解を求めます．

### オイラー法
オイラー法は，常微分方程式 $\dot{\boldsymbol{x}} = \boldsymbol{f}(\boldsymbol{x})$ の数値解法の一つです．時刻 $t$ の状態変数 $\boldsymbol{x}(t)$ から時刻 $t + dt$ の状態変数 $\boldsymbol{x}(t + dt)$ を求めるために，以下の更新式を利用します．

$$
\boldsymbol{x}(t + dt) = \boldsymbol{x}(t) + \boldsymbol{f}(\boldsymbol{x})dt
$$

初期値 $t = t_0$ , $\boldsymbol{x} = \boldsymbol{x}_0$ から始めて，上式を繰り返し実行することにより，状態変数 $\boldsymbol{x}$ の時間発展を求めることができます．

### ルンゲクッタ法
ルンゲクッタ法は，常微分方程式 $\dot{\boldsymbol{x}} = \boldsymbol{f}(\boldsymbol{x})$ の数値解法の一つです．時刻 $t$ の状態変数 $\boldsymbol{x}(t)$ から時刻 $t + dt$ の状態変数 $\boldsymbol{x}(t + dt)$ を求めるために，以下の更新式を利用します．

$$
\boldsymbol{x}(t + dt) = \frac{dt}{6}(\boldsymbol{k}_1 + 2\boldsymbol{k}_2 + 2\boldsymbol{k}_3 + \boldsymbol{k}_4)
$$

$$
\begin{eqnarray}
    \boldsymbol{k}_1 &=& \boldsymbol{f}(\boldsymbol{x})\\
    \boldsymbol{k}_2 &=& \boldsymbol{f}\Big(\boldsymbol{x} + \frac{dt}{2}\boldsymbol{k}_1 \Big)\\
    \boldsymbol{k}_3 &=& \boldsymbol{f}\Big(\boldsymbol{x} + \frac{dt}{2}\boldsymbol{k}_2 \Big)\\
    \boldsymbol{k}_4 &=& \boldsymbol{f}\Big(\boldsymbol{x} + dt\boldsymbol{k}_3 \Big)
\end{eqnarray}
$$

初期値 $t = t_0$ , $\boldsymbol{x} = \boldsymbol{x}_0$ から始めて，上式を繰り返し実行することにより，状態変数 $\boldsymbol{x}$ の時間発展を求めることができます．
