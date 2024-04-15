## 力学系ゼミ 第6回 プログラミング課題
### 概要
第6回は，数値計算法として，ニュートン法，オイラー法，ルンゲクッタ法を扱います．力学系に限らず宇宙系，複雑系でも頻繁に使用するアルゴリズムですので，是非この機会に使えるようにしていただければと思います．

課題を作成する際は，hw6ディレクトリ内にフォルダ(フォルダ名: `(名前)`)を作成し (e.g., `ito`)，作成したフォルダ内に課題ごとのファイルを`answer(課題番号).py`として作成してください．(e.g., `answer1.py`, `answer2-1.py`)

課題を作成する際は，必ずブランチを切り，作成したブランチ上で作業を行うようにしてください ([ブランチの作成](https://github.com/azkiphysics/team-mechanics2024?tab=readme-ov-file#ブランチの作成))．

課題が作成できたら，GitHub上でプルリクエストを開き，伊藤(ユーザー名: azkiphysics)にマージの許可を得てください．伊藤が提出した課題のコードレビューを行い，コードの修正をしていただきます．修正が完了したらマージを行い，その週の課題は終了となります．([プルリクエストの作成](https://github.com/azkiphysics/team-mechanics2024?tab=readme-ov-file#プルリクエストの作成))

### 準備
今回は，シミュレーション動画を作成するために，opencvと呼ばれるライブラリを使用します．なので，課題を行う前にまず以下のコマンドを仮想環境内で実行して，opencvをインストールしておいてください．

```zsh
pip install opencv-python opencv-contrib-python
```

### 課題1 (数値計算法の実装)
`template1.py`を利用して，オイラー法，ルンゲクッタ法を実装してください (`template1.py`をご自身のディレクトリに`answer1.py`という名前でコピーして課題を行ってください)．オイラー法は`Env`クラスの`euler_method`メソッド, ルンゲクッタ法は`Env`クラスの`runge_kutta_method`メソッドを編集してください．実装の完了後，以下の2つのコマンドを実行してください．プログラムを実行する際は必ずカレントディレクトリを`hw6/(名前)`にするようにしてください (カレントディレクトリが`team-mechanics2024`の場合はコマンドプロンプト上で`cd hw6/(名前)`を実行してカレントディレクトリを変更してください)．

**オイラー法を利用した数値計算**

```zsh
python answer1.py euler_method
```

**ルンゲクッタ法を利用した数値計算**

```zsh
python answer1.py runge_kutta_method
```

上記2つのプログラムを実行すると，`result/euler_method`, `result/runge_kutta`ディレクトリ内にそれぞれの結果が保存されます．

### 課題2 (数値計算法の比較)
`template2.py`をご自身のディレクトリに`answer2.py`という名前でコピーしてプログラムを実行してください．プログラムを実行する際は必ずカレントディレクトリを`hw6/(名前)`にするようにしてください．

```zsh
python answer2.py
```

プログラムの実行後，`result`ディレクトリに`compare_integral_method.png`が保存されます．この図を用いて，オイラー法とルンゲクッタ法の数値誤差(`Mean absolute error (MAE) of total energy`)と計算速度(`Calculation speed`)の比較を行ってください．

## 解説
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

## 補足
### 型アノテーション
型アノテーションは，変数や関数，メソッドの引数，戻り値のデータの型を明示するための構文になります．型アノテーションを利用することで，コードが読みやすくなりバグが少なくなるため，チームで開発する際は必須の構文です．

型アノテーションの例を以下に示します．例えば，変数`x`に整数型`10`を定義する場合は以下のように書きます．

```python
x: int = 10
```

上の例では`int`が整数の型アノテーションとなります．他にも小数の場合は`float`, 複素数型の場合は`complex`, 文字列の場合は`str`, リストの場合は`list`, タプルの場合は`tuple`, 辞書の場合は`dict`が型アノテーションとなります．

関数に型アノテーションを利用する場合は，例えば以下のように引数と戻り値に対してそれぞれ型アノテーションを書きます．ここでは，2つの数値型オブジェクトの和を計算する`add`関数を実装しています．したがって，引数と戻り値は整数型，小数型，複素数型のいずれかとなります．

```python
def add(x: int | float | complex, y: int | float | complex) -> int | float | complex:
    return x + y
```

上の例にあるように，型アノテーションの候補が複数ある場合は`|`を用いて型アノテーションを並べます．また，戻り値の型アノテーションは`->`の後に書きます．

リスト型やタプル型，辞書型のようなシーケンス型の場合，要素の型を定義したい時があります．その場合，python標準ライブラリの`typing`モジュールを利用します．例えば，小数型を要素にもつリストを引数とし，そのリストの要素の和を戻り値とする`sum`関数を実装すると以下のようになります．

```python
from typing import List

def sum(x: List[float]) -> float:
    return sum(x)
```

上の例のように，小数型を要素にもつリストの型アノテーションは`List[float]`となります．`List`は`typing`モジュールからインポートします．一般に，シーケンス型の型アノテーションは`(シーケンス型の型アノテーションの最初の文字を大文字にしたもの)[型アノテーション]` (e.g., `List[float]`, `Dict[str, int | float]`, `Tuple[int, int]`)と書きます．ここで，`Dict`や`Tuple`についても，`List`の時と同様，`typing`モジュールからインポートします．また，辞書型はキーと値を持つので，型アノテーション`Dict`の`[]`の中にはキーと値の型をそれぞれ書きます．

### defaultdictオブジェクト
`defaultdict`はpython標準ライブラリの`collections`モジュールで実装されている辞書型のサブクラスになります．`defaultdict`は辞書型のデフォルトの機能に加え，初期化の機能があります．例えば，`buffer`というリストを要素にもつ辞書型に，新しいキー`"t"`とそのキーに対応するリストの要素を追加するというプログラムを実装すると，以下のようになります．

```python
buffer = {}
t = 10.0
if "t" in buffer:
    buffer["t"] = []
buffer["t"].append(t)
```

上の例では，キー`"t"`に対応するリストを`[]`で初期化した後，リストの要素を`append`メソッドで追加しています．一方，上のプログラムを`defaultdict`を使って実装すると以下のようになります．

```python
from collections import defaultdict

buffer = defaultdict(list)
t = 10.0
buffer["t"].append(t)
```

上の例からわかるように，`defaultdict`では初期化の処理を行わずとも，`append`メソッドを実行した時に自動で初期化が行われます．`defaultdict`のキーに対応する要素の型については，`defaultdict`オブジェクトを定義するときに指定します．リストの場合は`defaultdict(list)`, 整数の場合は`defaultdict(int)`, 文字列の場合は`defaultdict(str)`といった具合に，要素の型に応じて`defaultdict(型)`と定義します．

### ニュートンラフソン法
ニュートンラフソン法は，ある方程式 $\boldsymbol{f}(\boldsymbol{x}) = \boldsymbol{0}$ の解を数値的に求めるための手法となります．ニュートンラフソン法は，マルチボディシステムの状態 $\boldsymbol{x}$ の精度を高めるために利用しています．

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

### Cart pole問題の運動方程式
マルチボディシステムの微分代数方程式は以下の式で表されます．

$$
\begin{bmatrix}
    \mathrm{M} & \boldsymbol{C}_{\boldsymbol{q}}^T\\
    \boldsymbol{C}_{\boldsymbol{q}} & \boldsymbol{0}
\end{bmatrix}

\begin{bmatrix}
    \boldsymbol{\ddot{q}}\\
    \boldsymbol{\lambda}
\end{bmatrix}

=

\begin{bmatrix}
    \boldsymbol{Q}_e\\
    \boldsymbol{Q}_d
\end{bmatrix}
$$

$\mathrm{M}$は質量慣性行列, $\boldsymbol{C}_{\boldsymbol{q}}$ は拘束条件式 $\boldsymbol{C}$ のヤコビ行列, $\boldsymbol{q}$ は一般座標, $\boldsymbol{\lambda}$ はラグランジュの未定乗数, $\boldsymbol{Q}_e$ は一般外力, $\boldsymbol{Q}_d$ は次式で定義されるベクトルとなります．

$$
\boldsymbol{Q}_d = -\boldsymbol{C}_{tt} - (\boldsymbol{C}_{\boldsymbol{q}}\dot{\boldsymbol{q}})_{\boldsymbol{q}}\dot{\boldsymbol{q}} - 2\boldsymbol{C}_{\boldsymbol{q}t}\dot{\boldsymbol{q}}
$$

Cart pole問題では，上記のベクトル，行列はそれぞれ以下のように表されます．


$$
\boldsymbol{q} = \begin{bmatrix}
    x_{\mathrm{cart}}\\
    x_{\mathrm{ball}}\\
    x_{\mathrm{ball}}\\
    y_{\mathrm{ball}}\\
    \theta_{\mathrm{pole}}\\
\end{bmatrix}
$$

$$
M = \begin{bmatrix}
    m_{\mathrm{cart}} & 0 & 0 & 0\\
    0 & m_{\mathrm{ball}} & 0 & 0\\
    0 & 0 & 0 & 0
\end{bmatrix}
$$

$$
\boldsymbol{C} = \begin{bmatrix}
    x_{\mathrm{ball}} - x_{\mathrm{cart}} - l_{\mathrm{pole}}\cos(\theta_{\mathrm{pole}})\\
    y_{\mathrm{ball}} - l_{\mathrm{pole}}\sin(\theta_{\mathrm{pole}})
\end{bmatrix}
$$

$$
\boldsymbol{C}_{\boldsymbol{q}} = \begin{bmatrix}
    -1 & 1 & 0 & l_{\mathrm{pole}}\sin(\theta_{\mathrm{pole}})\\
    0 & 0 & 1 & -l_{\mathrm{pole}}\cos(\theta_{\mathrm{pole}})
\end{bmatrix}
$$

$$
\boldsymbol{C}_{tt} = \begin{bmatrix}
    0\\
    0
\end{bmatrix}
$$

$$
\boldsymbol{C}_{\boldsymbol{q}t} = \begin{bmatrix}
    0 & 0 & 0 & 0\\
    0 & 0 & 0 & 0
\end{bmatrix}
$$

$$
(\boldsymbol{C}_{\boldsymbol{q}}\dot{\boldsymbol{q}})_{\boldsymbol{q}} = \begin{bmatrix}
    0 & 0 & 0 & l_{\mathrm{pole}}\dot{\theta}_{\mathrm{pole}}\cos(\theta_{\mathrm{pole}})\\
    0 & 0 & 0 & l_{\mathrm{pole}}\dot{\theta}_{\mathrm{pole}}\sin(\theta_{\mathrm{pole}})
\end{bmatrix}
$$
