## 力学系ゼミ 第5回 プログラミング課題
### 概要
第5回は，研究で頻繁に利用するnumpyとmatplotlibを扱います．numpyは，pythonで数値計算を行う際には必須のライブラリとなっています．matplotlibは，シミュレーション結果の図の描画を行うのに適したライブラリとなっています．第1回~第5回までの課題を全て行うと，Pythonでシミュレーションを行う方法を一通り学んだことになります．第6回以降は，第5回までの応用となっていますので，課題をした後も復習するようにしてください．

課題を作成する際は，hw5ディレクトリ内にフォルダ(フォルダ名: `(名前)`)を作成し (e.g., `ito`)，作成したフォルダ内に課題ごとのファイルを`answer(課題番号).py`として作成してください．(e.g., `answer1.py`, `answer2-1.py`)

課題を作成する際は，必ずブランチを切り，作成したブランチ上で作業を行うようにしてください ([ブランチの作成](https://github.com/azkiphysics/team-mechanics2024?tab=readme-ov-file#ブランチの作成))．

課題が作成できたら，GitHub上でプルリクエストを開き，伊藤(ユーザー名: azkiphysics)にマージの許可を得てください．伊藤が提出した課題のコードレビューを行い，コードの修正をしていただきます．修正が完了したらマージを行い，その週の課題は終了となります．

### 課題1 (numpy)
hw3の課題3で実装したローレンツ系を記述したクラス`LorenzEnv`を`numpy`を使って書き直します．以下のテンプレートを利用して，`LorenzEnv`の`motion_equation`メソッドを`numpy`を利用して実装し，プログラムを実行して，カレントディレクトリ上に`result/trajectory.pickle`ファイルが作成されていることを確認してください．(プログラムを実行する際は，カレントディレクトリが`hw5/(名前)`にあることを確認してください．)

**テンプレート**
```python
import os
import pickle

import numpy as np


class Buffer(object):
    def __init__(self) -> None:
        self.buffer = None

    def reset(self):
        self.buffer = {}

    def push(self, data):
        for key, value in data.items():
            if key not in self.buffer:
                self.buffer[key] = []
            self.buffer[key].append(value)

    def get(self):
        return self.buffer

    def save(self, savedir, savefile):
        buffer = self.get()
        os.makedirs(savedir, exist_ok=True)
        path = os.path.join(savedir, savefile)
        with open(path, "wb") as f:
            pickle.dump(buffer, f)


class Env(object):
    def __init__(self, t_max, dt=1e-3):
        self.t_max = t_max
        self.dt = dt

        self.t = None
        self.x = None

    def integral(self, x):
        k1 = self.motion_equation(x)
        k2 = self.motion_equation(x + self.dt / 2 * k1)
        k3 = self.motion_equation(x + self.dt / 2 * k2)
        k4 = self.motion_equation(x + self.dt * k3)
        next_x = x + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return next_x

    def motion_equation(self, x):
        return np.zeros_like(x, dtype=np.float64)

    def reset(self, initial_t, initial_x):
        self.t = initial_t
        self.x = initial_x.copy()
        done = self.t >= self.t_max
        info = {"t": self.t, "x": self.x.copy(), "done": done}
        return info

    def step(self):
        self.t += self.dt
        self.x = self.integral(self.x)
        done = self.t >= self.t_max
        info = {"t": self.t, "x": self.x.copy(), "done": done}
        return info


class LorenzEnv(Env):
    def __init__(self, t_max, dt=1e-3, rho=28.0, sigma=10.0, beta=8.0/3.0):
        super().__init__(t_max, dt=dt)
        self.rho = rho
        self.sigma = sigma
        self.beta = beta

    def motion_equation(self, x):
        """
        numpyを使って，以下にLorenz方程式を実装してください．
        """


if __name__ == "__main__":
    t_max = 50.0
    dt = 1e-3
    initial_t = 0.0
    initial_x = np.ones(3, dtype=np.float64)

    env = LorenzEnv(t_max, dt=dt)
    buffer = Buffer()

    info = env.reset(initial_t, initial_x)
    done = info.pop("done")
    buffer.reset()
    buffer.push(info)
    while not done:
        info = env.step()
        done = info.pop("done")
        buffer.push(info)

    savedir = "result"
    savefile = "trajectory.pickle"
    buffer.save(savedir, savefile)

```

**解答例**
```python
import os
import pickle

import numpy as np


class Buffer(object):
    def __init__(self) -> None:
        self.buffer = None

    def reset(self):
        self.buffer = {}

    def push(self, data):
        for key, value in data.items():
            if key not in self.buffer:
                self.buffer[key] = []
            self.buffer[key].append(value)

    def get(self):
        return self.buffer

    def save(self, savedir, savefile):
        buffer = self.get()
        os.makedirs(savedir, exist_ok=True)
        path = os.path.join(savedir, savefile)
        with open(path, "wb") as f:
            pickle.dump(buffer, f)


class Env(object):
    def __init__(self, t_max, dt=1e-3):
        self.t_max = t_max
        self.dt = dt

        self.t = None
        self.x = None

    def integral(self, x):
        k1 = self.motion_equation(x)
        k2 = self.motion_equation(x + self.dt / 2 * k1)
        k3 = self.motion_equation(x + self.dt / 2 * k2)
        k4 = self.motion_equation(x + self.dt * k3)
        next_x = x + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return next_x

    def motion_equation(self, x):
        return np.zeros_like(x, dtype=np.float64)

    def reset(self, initial_t, initial_x):
        self.t = initial_t
        self.x = initial_x.copy()
        done = self.t >= self.t_max
        info = {"t": self.t, "x": self.x.copy(), "done": done}
        return info

    def step(self):
        self.t += self.dt
        self.x = self.integral(self.x)
        done = self.t >= self.t_max
        info = {"t": self.t, "x": self.x.copy(), "done": done}
        return info


class LorenzEnv(Env):
    def __init__(self, t_max, dt=1e-3, rho=28.0, sigma=10.0, beta=8.0/3.0):
        super().__init__(t_max, dt=dt)
        self.rho = rho
        self.sigma = sigma
        self.beta = beta

    def motion_equation(self, x):
        dx_dt = np.zeros_like(x, dtype=np.float64)
        dx_dt[0] = self.sigma * (x[1] - x[0])
        dx_dt[1] = x[0] * (self.rho - x[2]) - x[1]
        dx_dt[2] = x[0] * x[1] - self.beta * x[2]
        return dx_dt


if __name__ == "__main__":
    t_max = 50.0
    dt = 1e-3
    initial_t = 0.0
    initial_x = np.ones(3, dtype=np.float64)

    env = LorenzEnv(t_max, dt=dt)
    buffer = Buffer()

    info = env.reset(initial_t, initial_x)
    done = info.pop("done")
    buffer.reset()
    buffer.push(info)
    while not done:
        info = env.step()
        done = info.pop("done")
        buffer.push(info)

    savedir = "result"
    savefile = "trajectory.pickle"
    buffer.save(savedir, savefile)

```

### 課題2 (matplotlib)
以下のソースコードを利用して，課題1で得られたシミュレーション結果の図を作成してください．(プログラムを実行する際は必ずカレントディレクトリが`hw5/(名前)`であることを確認してください．)


```python
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


# Matplotlibで綺麗な論文用のグラフを作る
# https://qiita.com/MENDY/items/fe9b0c50383d8b2fd919
plt.rcParams['font.family'] = 'Times New Roman' # font familyの設定
plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
plt.rcParams["font.size"] = 15 # 全体のフォントサイズが変更されます。
plt.rcParams['xtick.labelsize'] = 15 # 軸だけ変更されます。
plt.rcParams['ytick.labelsize'] = 15 # 軸だけ変更されます
plt.rcParams['xtick.direction'] = 'in' # x axis in
plt.rcParams['ytick.direction'] = 'in' # y axis in 
plt.rcParams['axes.linewidth'] = 1.0 # axis line width
plt.rcParams['axes.grid'] = True # make grid


if __name__ == "__main__":
    loaddir = "result"
    loadfile = "trajectory.pickle"
    loadpath = os.path.join(loaddir, loadfile)
    with open(loadpath, "rb") as f:
        result = pickle.load(f)

    t = result["t"]
    x, y, z = np.split(np.vstack(result["x"]), 3, axis=1)

    fig = plt.figure(figsize=(9, 6), layout="constrained")
    axs = fig.subplot_mosaic(
        [
            ["trajectory", "time_series_x"],
            ["trajectory", "time_series_y"],
            ["trajectory", "time_series_z"],
        ],
        per_subplot_kw={("trajectory",): {"projection": "3d"}},
        gridspec_kw={"width_ratios": [2, 1], "wspace": 0.15, "hspace": 0.05},
    )
    axs["trajectory"].plot(x, y, z)
    axs["trajectory"].set_xlabel("x")
    axs["trajectory"].set_ylabel("y")
    axs["trajectory"].set_zlabel("z")

    axs["time_series_x"].plot(t, x)
    axs["time_series_x"].set_xlabel("Time $t$ s")
    axs["time_series_x"].set_ylabel("$x$")

    axs["time_series_y"].plot(t, y)
    axs["time_series_y"].set_xlabel("Time $t$ s")
    axs["time_series_y"].set_ylabel("$y$")

    axs["time_series_z"].plot(t, z)
    axs["time_series_z"].set_xlabel("Time $t$ s")
    axs["time_series_z"].set_ylabel("$z$")

    savedir = loaddir
    savefile = "trajectory.png"
    savepath = os.path.join(savedir, savefile)
    fig.savefig(savepath, dpi=300)
    plt.close()
```

プログラムを実行すると次の図が`result`ディレクトリに出力されます．

![ローレンツ系のシミュレーション結果](./ito/result/trajectory.png)


## 解説
### numpy
#### numpyとは
numpyはベクトルや行列計算を高速化することを目的として開発されているサードパーディー製のPythonライブラリです．
#### numpyのインストール
Anacondaの仮想環境を利用している場合は，以下のコマンドを入力することでnumpyをインストールすることができます．
```zsh
conda install numpy
```
それ以外の場合は，`pip`を利用してnumpyをインストールします．
```zsh
pip install numpy
```
#### numpyのインポート
numpyは`import numpy`によってインポートすることができます．numpyのインポートは，一般にPython標準ライブラリの下に記載します．また，`import numpy as np`と書くことで，`np.(関数名)`で関数の呼び出しができます．

```python
(python標準ライブラリのインポート)

# python標準ライブラリの間は一行空ける
import numpy as np # as npを加えることで，np.(関数名)と記載できる．
```
#### numpyの初期化
numpyでは，`array`というデータ構造を扱います．`array`はリストやタプルのように複数の要素を格納できるオブジェクトですが，それに加えて`array`は要素のデータ構造(データの型, データのランク(次元数))がすべて同じという特徴があります．`array`は一般にnumpy配列と呼ばれ，以降は`array`のことをnumpy配列と呼ぶことにします．

numpy配列はリストやタプルを利用して初期化することができます．例えば，`[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]`というリストをnumpy配列にすると次のようになります．

```python
a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
```

`np.array`の第１引数はnumpy配列に変換するオブジェクト，第2引数`dtype`はデータのタイプを表しており，整数の場合のデフォルト値は`np.int64`，小数の場合のデフォルト値は`np.float64`です．(c.f., ニューラルネットワークの入力としてnumpy配列を利用する場合，`dtype`は`np.float32`に設定することが多いです．)

numpy配列は多次元配列(=行列)も利用することができます．例えば，次のような他次元配列が定義できます．

```python
b = np.array([[1, 2, 3], [4, 5, 5]], dtype=np.float64)
```

numpyの別の初期化方法として，`np.zeros`や`np.ones`を用いる方法があります．`np.zeros`は0を要素にもつ配列を作成する関数，`np.ones`は1を要素にもつ配列を作成する関数となります．それぞれ第１引数に配列の形状`shape`(=行数, 列数)を設定し，第２引数に配列のタイプ`dtype`を設定します．例えば， 2次元の全要素1のベクトルは次のように定義できます．

```python
np.ones(2, dtype=np.float64)
```

多次元配列も定義することができ，$2\times 3$ のゼロ行列や $2\times 3\times 4$ のゼロ行列は次のように定義できます．

```python
np.zeros((2, 3), dtype=np.float64)
np.zeros((2, 3, 4), dtype=np.float64)
```

他にも，`np.arange`関数を使うと`range`関数のnumpy配列を作成することができ，`np.linspace`を使うとある特定の範囲内の線形な配列を作成することができます．

```python
np.arange(2, 9, 2) # array([2, 4, 6, 8])
np.linspace(0, 10, num=5) # array([0., 2.5, 5., 7.5, 10.])
```

#### numpyのインデックスとスライス
numpy配列の要素はシーケンス型の時と同様，インデックスを指定することでアクセスすることができます．例えば，[numpyの初期化](#numpyの初期化)の中で定義した変数`a`の2番目の要素を取得したい場合，次のように書くことができます．

```python
a[1] # 2.0
```

多次元配列の場合，要素は`(変数)[(行のインデックス), (列のインデックス)]`によってアクセスできます．例えば，[numpyの初期化](#numpyの初期化)の中で定義した変数`b`の2行3列目の要素を取得したい場合，次のように書くことができます．

```python
b[1, 2] # 5.0
```

また，numpy配列はスライスも行うことができ，例えば変数`a`の2番目から4番目のnumpy配列を取得したい場合は次のように書くことができます．

```python
a[1: 4] # array([2., 3., 4.])
```

また，`b`の2行目の2列目から3列目のnumpy配列を取得したい場合は次のように書くことができます．

```python
b[1, 2:4] # array([5., 5.])
```

また，numpy配列は条件式を与えることで，条件式を満たすような要素のみを取り出すこともできます．例えば，`b[b < 3]`と書くと，`b`の要素のうち，3よりも小さい値が1次元配列で取得できます．すなわち，`b[b < 3]`は`array([1, 2])`となります．

複数の条件式を満たすような要素を取り出す場合は`&`や`|`を利用します．`&`は`and`を表し，`|`は`or`を表します．例えば，`b`から2以上4以下の要素を取り出したい場合は，次のように書きます．

```python
b[(b >= 2) & (b <= 4)] # array([2, 3, 4])
```

ここで，複数の条件式を`&`や`|`で繋ぐ場合は必ず`()`の中に条件式を記載するようにしてください．

#### numpyの要素のソートと結合
numpyの要素をソートするためには`np.sort`関数を利用します．例えば，次のような配列が定義されているとします．

```python
arr = np.array([2, 1, 5, 3, 7, 4, 6, 8])
```

このとき，`np.sort`関数を利用することで，昇順に配列の要素を並べ変えることができます．

```python
np.sort(arr) # array([1, 2, 3, 4, 5, 6, 7, 8])
```

複数のnumpy配列を結合するには`np.concatenate`関数を利用します．例えば，次のような配列が定義されているとします．

```python
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])
```

このとき，`np.concatenate`関数を利用して，配列`a`と`b`を結合すると次のようになります．

```python
np.concatenate((a, b)) # array([1, 2, 3, 4, 5, 6, 7, 8])
```

`np.concatenate`は多次元配列にも利用でき，例えば2つの多次元配列を結合して次のような新しい多次元配列を作成することができます．

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])

np.concatenate((x, y), axis=0) # array([[1, 2], [3, 4], [5, 6]])
```

ここで，多次元配列を結合する場合は，第２引数の`axis`を指定してあげることで，結合する軸を決めることができます．

#### numpy配列の次元・サイズ・形状
numpy配列の次元・サイズ・形状の取得にはそれぞれ表に記載している関数を利用します．

|  取得する対象   |  方法   |
| --- | --- |
|  numpy配列の次元   |  `(numpy配列).ndim`   |
|  numpy配列のサイズ (=要素数)   |  `(numpy配列).size`   |
|  numpy配列の形状   |  `(numpy配列).shape`   |

例えば，ある配列の次元，サイズ，形状はそれぞれ次のように書けます．

```python
arr = np.zeros((10, 5, 4), dtype=np.float64)
arr.ndim # 次元数 3
arr.size # サイズ 200
arr.shape # 形状 (10, 5, 4)
```

#### numpy配列の形状変更
numpy配列の形状変更には`(numpy配列).reshape`関数を利用します．例えば，ある1次元numpy配列を $3\times 2$ のnumpy配列に変えるには次のように書きます．

```python
arr = np.arange(6)
new_arr = arr.reshape(3, 2) # array([[0, 1], [2, 3], [4, 5]])
```

#### numpy配列の新しい軸の追加
numpy配列に新しい軸を追加するためには`np.newaxis`を利用します．以下に例を示します．

```python
a= np.array([1, 2, 3, 4, 5, 6])
row_vector = a[np.newaxis, :] # array([[1, 2, 3, 4, 5, 6]])
col_vector = a[:, np.newaxis] # array([[1], [2], [3], [4], [5], [6]])
```

#### numpyの基本演算
##### 四則演算

numpy配列の四則演算(`+, -, *, /`)を行う場合，要素ごとの四則演算となります．以下に例を示します．

```python
a = np.array([[1, 2], [3, 4], [5, 6]])
b = np.array([[3, 2], [4, 3], [6, 5]])
print(a + b)
print(a - b)
print(a * b)
print(a / b)
```

それぞれの出力結果は以下のようになります．

```zsh
array([[ 4,  4],
       [ 7,  7],
       [11, 11]])
array([[-2,  0],
       [-1,  1],
       [-1,  1]])
array([[ 3,  4],
       [12, 12],
       [30, 30]])
array([[0.33333333, 1.        ],
       [0.75      , 1.33333333],
       [0.83333333, 1.2       ]])
```


##### ブロードキャスティング
上記の例のように，numpy配列の四則演算を行う場合，配列のshapeは基本的には同じである必要があります．

一方で，numpy配列の要素すべてを定数倍するときなどは，numpy配列と数値型(スカラー)の積ができれば簡単にプログラムを書くことができます．そこで，numpyでは，**ブロードキャスティング**という機能があります．ブロードキャスティングは異なる形状の配列の四則演算を行うためのメカニズムとなります．例えば，`arr = np.array([2, 3, 4])`の各要素を2倍にする時は以下のように書くことができます．

```python
arr = np.array([2, 3, 4])
arr_twice = arr * 2 # array([4, 6, 8])
```

上記の計算では，`2`はブロードキャスティングにより`np.array([2, 2, 2])`となり，arrとの演算が可能となります．

また，配列どうしの演算でもブロードキャスティングが適用されます．

```python
a = np.array([2, 3, 4])
b = np.array([[8, 3, 4], [3, 1, 4]])
c = a + b # array([[10, 6, 8], [5, 4, 8]])
```

上記の計算では，`a`がブロードキャスティングにより`np.array([[2, 3, 4], [2, 3, 4]])`となり，`a`と`b`の演算が可能となります．

ただし，上記の2つの例のように，ブロードキャスティングする軸方向が0または1である場合，ブロードキャスティングが可能ですが，それ以外の場合はブロードキャスティングできず，`ValueError`となります．

##### numpy配列の要素の最大値，最小値，和
numpy配列の最大値，最小値，和はそれぞれ`np.max`, `np.min`, `np.sum`関数を利用します．以下に例を示します．

```python
arr = np.array([[8, 3, 4], [3, 1, 4]])

arr_max = np.max(arr) # 最大値: 8
arr_min = np.min(arr) # 最小値: 1
arr_sum = np.sum(arr) # 和: 23
```

`np.max`, `np.min`, `np.sum`は第2引数の`axis`を指定することで，指定された軸方向に対して演算を行うことができます．

```python
arr = np.array([[8, 3, 4], [3, 1, 4]])

arr_max = np.max(arr, axis=1) # 最大値: array([8, 4])
arr_min = np.min(arr, axis=1) # 最小値: array([3, 1])
arr_sum = np.sum(arr, axis=1) # 和: array([15, 8])
```

##### numpy配列の分割
numpy配列を複数のnumpy配列に分割するためには`np.split`関数を利用します．以下に例を示します．

```python
arr = np.array([[8, 3, 4], [3, 1, 4]])

arr_split = np.split(arr, 3, axis=1) # (array([[8], [3]]), array([[3], [1]]), array([[4], [4]]))
```

`np.split`の第１引数は分割するnumpy配列，第２引数は分割する位置，もしくは分割個数，第3引数は分割する軸方向となっています．第3引数`axis`のデフォルト値は0となっています．

#### numpyの線形代数の演算
##### numpy配列の行列積
numpy配列の行列積は`np.dot`関数もしくは`@`を利用することで計算が可能です．以下に例を示します．

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[3, 5], [8, 9]])
b = np.array([1, 2])

x = np.dot(A, b) # array([ 5, 11])
y = A @ b # array([ 5, 11])

C = np.dot(A, B) # array([[19, 23], [41, 51]])
D = A @ B # array([[19, 23], [41, 51]])
```

##### numpy配列の外積
2または3次元ベクトルのnumpy配列の外積は`np.cross`を利用することで計算が可能です．以下に例を示します．

```python
a = np.array([1, 2, 3])
b = np.array([1, 2, -1])

ab_cross = np.cross(a, b) # array([-8,  4,  0])
```

##### numpy配列のノルム
numpy配列のノルムは`np.linalg.norm`を利用することで計算が可能です．以下に例を示します．

```python
arr = np.array([[8, 3, 4], [3, 1, 4]])

arr_norm = np.linalg.norm(arr) # 10.723805294763608
arr_norm0 = np.linalg.norm(arr, axis=0) # array([8.54400375, 3.16227766, 5.65685425])
arr_norm1 = np.linalg.norm(arr, axis=1) # array([9.43398113, 5.09901951])
```

`np.linalg.norm`関数は第2引数の`axis`に軸を指定することで，軸方向のノルムを計算することができます．

##### numpy配列の逆行列
次元数が2の正方行列の逆行列は`np.linalg.inv`によって求めることができ，次元数が2で正方行列でない行列の逆行列は`np.linalg..pinv`によって求めることができます．以下に例を示します．

```python
A = np.array([[8, 1], [3, 1]])
B = np.array([[8, 3, 4], [3, 1, 4]])

inv_A = np.linalg.inv(A) # array([[0.2, -0.2], [-0.6, 1.6]])
inv_B = np.linalg.pinv(B) # array([[ 0.16989247, -0.1655914 ], [ 0.07526882, -0.08602151], [-0.14623656,  0.39569892]])
```

##### numpy配列の固有値分解
次元数2の正方行列の固有値・固有ベクトルは`np.linalg.eig`によって求めることができます．以下に例を示します．

```python
A = np.array([[8, 1], [3, 1]])

eig_values, eig_vectors = np.linalg.eig(A)
print(eig_values) # array([8.40512484, 0.59487516])
print(eig_vectors) # array([[ 0.92682978, -0.13382688], [ 0.37548177,  0.99100473]])
```

`eig_values`は固有値，`eig_vectors`は固有ベクトルを表しています．

##### numpy配列の特異値分解
次元数2の正方行列の特異値・特異ベクトルは`np.linalg.svd`によって求めることができます．以下に例を示します．

```python
A = np.array([[8, 3, 4], [3, 1, 4]])

U, S, Vh = np.linalg.svh(A)
print(U) # array([[ 0.89189599, -0.45224057], [ 0.45224057,  0.89189599]])
print(S) # array([10.5263183 ,  2.04856608])
print(Vh) # array([[ 0.80672933,  0.29715314,  0.51077177], [-0.45994933, -0.22690297,  0.85846471], [-0.37099112,  0.92747779,  0.04637389]])
```

`U`は左特異ベクトル，`S`は特異値，`Vh`は右特異ベクトルの随伴行列を表しています．

#### numpyのより詳しく知りたい方へ
今回は，研究でよく使う関数のみ解説しましたが，numpyにはまだまだたくさんの関数が存在します．numpyをより詳しく知りたい方は，numpyのドキュメントをご確認いただければと思います．

[Numpy documentation](https://numpy.org/doc/stable/)

### matplotlib
#### matplotlibとは
matplotlibはPythonでグラフを描画するためのライブラリです．2D, 3Dともに描画することができます．
#### matplotlibのインストール
Anacondaの仮想環境を利用している場合は，以下のコマンドを入力することでmatplotlibをインストールすることができます．
```zsh
conda install matplotlib
```
それ以外の場合は，`pip`を利用してmatplotlibをインストールします．
```zsh
pip install matplotlib
```
#### 描画の基本
グラフの描画をするためには，まず`matplotlib.pyplot`をインポートします．
```python
import matplotlib.pyplot as plt
```
`matplotlib.pyplot`は一般に`plt`という名前で利用するため，`as plt`としています．

ここでは，正弦波を閉区間 $[0, 2\pi]$ の範囲で描画することにします．numpyを使って正弦波を生成するプログラムは次のとおりです．

```python
import numpy as np

x = np.linspace(0, 2 * np.pi, 200)
y = np.sin(x)
```

次に`matplotlib`を使って上記の正弦波を描画するプログラムを作成すると，次のようになります．

```python
fig, ax = plt.subplots()
ax.plot(x, y)
plt.show()
```

上記の`fig`はキャンパス全体を表し，`ax`は数値データ(正弦波)を描画する領域を表しています．上記の例では`ax`の領域が設定されていないため，`fig`全体にデータの描画を行う設定となっています．

正弦波のプロットには，`ax.plot`関数を利用します．第１引数と第２引数にはそれぞれ図の横軸，縦軸のデータを入力します．そして，`plt.show`関数を書くことで，図の描画結果が表示されます．

![](https://matplotlib.org/stable/_images/users-getting_started-index-1.2x.png)

図の保存をする場合は，以下のように`fig.savefig((図のパス), dpi=(解像度))`を追記します．

```python
fig.savefig("figure.png", dpi=300)
```

図はpngの他にepsやpdf形式でも保存できます．

より詳しい説明についてはmatplotlibのドキュメントを確認していただければと思います．

[Matplotlib documentation](https://matplotlib.org/stable/)
