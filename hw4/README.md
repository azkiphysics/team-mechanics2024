## 力学系ゼミ 第4回 プログラミング課題
### 概要
第4回は，ファイルの入出力を扱います．

課題を作成する際は，hw4ディレクトリ内にフォルダ(フォルダ名: `(名前)`)を作成し (e.g., `ito`)，作成したフォルダ内に課題ごとのファイルを`answer(課題番号).py`として作成してください．(e.g., `answer1.py`, `answer2-1.py`)

課題を作成する際は，必ずブランチを切り，作成したブランチ上で作業を行うようにしてください ([ブランチの作成](https://github.com/azkiphysics/team-mechanics2024?tab=readme-ov-file#ブランチの作成))．

課題が作成できたら，GitHub上でプルリクエストを開き，伊藤(ユーザー名: azkiphysics)にマージの許可を得てください．伊藤が提出した課題のコードレビューを行い，コードの修正をしていただきます．修正が完了したらマージを行い，その週の課題は終了となります．

### 課題1 (ファイルの出力)
課題1では，hw2の課題3のローレンツ系のシミュレーション結果をcsvファイルとpickleファイルに保存するプログラムを作成してもらいます．ここでは，以下のテンプレートを利用します．csvファイルの保存にはcsvモジュール，pickleファイルの保存にはpickleモジュールを利用してください．モジュールとは，ソースコードをまとめたファイルのことを指し，`import`を用いることで呼び出すことができます (e.g., `import csv`, `import pickle`)．(プログラムを実行する際は，カレントディレクトリが`hw4/(名前)`にあることを確認してください．)

**テンプレート**
```python
import csv
import pickle


def add_data(result, info):
    for key, value in info.items():
        if key in result:
            result[key].append(value)


class Env(object):
    def __init__(self, t_max, dt=1e-3):
        self.t_max = t_max
        self.dt = dt

        self.t = None
        self.x = None
        self.y = None
        self.z = None

    def integral(self, x, y, z):
        k1_x, k1_y, k1_z = self.motion_equation(x, y, z)
        k2_x, k2_y, k2_z = self.motion_equation(x + self.dt / 2 * k1_x, y + self.dt / 2 * k1_y, z + self.dt / 2 * k1_z)
        k3_x, k3_y, k3_z = self.motion_equation(x + self.dt / 2 * k2_x, y + self.dt / 2 * k2_y, z + self.dt / 2 * k2_z)
        k4_x, k4_y, k4_z = self.motion_equation(x + self.dt * k3_x, y + self.dt * k3_y, z + self.dt * k3_z)
        next_x = x + self.dt / 6 * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        next_y = y + self.dt / 6 * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)
        next_z = x + self.dt / 6 * (k1_z + 2 * k2_z + 2 * k3_z + k4_z)
        return next_x, next_y, next_z

    def motion_equation(self, x, y, z):
        return 0.0, 0.0, 0.0

    def reset(self, initial_t, initial_x, initial_y, initial_z):
        self.t = initial_t
        self.x, self.y, self.z = initial_x, initial_y, initial_z
        done = self.t >= self.t_max
        info = {"t": self.t, "x": self.x, "y": self.y, "z": self.z, "done": done}
        return info

    def step(self):
        self.t += self.dt
        self.x, self.y, self.z = self.integral(self.x, self.y, self.z)
        done = self.t >= self.t_max
        info = {"t": self.t, "x": self.x, "y": self.y, "z": self.z, "done": done}
        return info


class LorenzEnv(Env):
    def __init__(self, t_max, dt=1e-3, rho=28.0, sigma=10.0, beta=8.0/3.0):
        super().__init__(t_max, dt=dt)
        self.rho = rho
        self.sigma = sigma
        self.beta = beta

    def motion_equation(self, x, y, z):
        dx_dt = self.sigma * (y - x)
        dy_dt = x * (self.rho - z) - y
        dz_dt = x * y - self.beta * z
        return dx_dt, dy_dt, dz_dt


if __name__ == "__main__":
    t_max = 0.1
    dt = 1e-3
    initial_t = 0.0
    initial_x = 1.0
    initial_y = 1.0
    initial_z = 1.0

    result = {"t": [], "x": [], "y": [], "z": []}

    env = LorenzEnv(t_max, dt=dt)

    info = env.reset(initial_t, initial_x, initial_y, initial_z)
    done = info.pop("done")
    add_data(result, info) # データの保存
    while not done:
        info = env.step()
        done = info.pop("done")
        add_data(result, info) # データの保存

    # csvファイルの保存
    path = "hw1_result.csv"
    """
    以下にcsvファイルの保存を行うプログラムを実装してください．
    """
    
    # pickleファイルの保存
    path = "hw1_result.pickle"
    """
    以下にpickleファイルの保存を行うプログラムを実装してください．
    """
```

**解答例**
```python
import csv
import pickle


def add_data(result, info):
    for key, value in info.items():
        if key in result:
            result[key].append(value)


class Env(object):
    def __init__(self, t_max, dt=1e-3):
        self.t_max = t_max
        self.dt = dt

        self.t = None
        self.x = None
        self.y = None
        self.z = None

    def integral(self, x, y, z):
        k1_x, k1_y, k1_z = self.motion_equation(x, y, z)
        k2_x, k2_y, k2_z = self.motion_equation(x + self.dt / 2 * k1_x, y + self.dt / 2 * k1_y, z + self.dt / 2 * k1_z)
        k3_x, k3_y, k3_z = self.motion_equation(x + self.dt / 2 * k2_x, y + self.dt / 2 * k2_y, z + self.dt / 2 * k2_z)
        k4_x, k4_y, k4_z = self.motion_equation(x + self.dt * k3_x, y + self.dt * k3_y, z + self.dt * k3_z)
        next_x = x + self.dt / 6 * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        next_y = y + self.dt / 6 * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)
        next_z = x + self.dt / 6 * (k1_z + 2 * k2_z + 2 * k3_z + k4_z)
        return next_x, next_y, next_z

    def motion_equation(self, x, y, z):
        return 0.0, 0.0, 0.0

    def reset(self, initial_t, initial_x, initial_y, initial_z):
        self.t = initial_t
        self.x, self.y, self.z = initial_x, initial_y, initial_z
        done = self.t >= self.t_max
        info = {"t": self.t, "x": self.x, "y": self.y, "z": self.z, "done": done}
        return info

    def step(self):
        self.t += self.dt
        self.x, self.y, self.z = self.integral(self.x, self.y, self.z)
        done = self.t >= self.t_max
        info = {"t": self.t, "x": self.x, "y": self.y, "z": self.z, "done": done}
        return info


class LorenzEnv(Env):
    def __init__(self, t_max, dt=1e-3, rho=28.0, sigma=10.0, beta=8.0/3.0):
        super().__init__(t_max, dt=dt)
        self.rho = rho
        self.sigma = sigma
        self.beta = beta

    def motion_equation(self, x, y, z):
        dx_dt = self.sigma * (y - x)
        dy_dt = x * (self.rho - z) - y
        dz_dt = x * y - self.beta * z
        return dx_dt, dy_dt, dz_dt


if __name__ == "__main__":
    t_max = 0.1
    dt = 1e-3
    initial_t = 0.0
    initial_x = 1.0
    initial_y = 1.0
    initial_z = 1.0

    result = {"t": [], "x": [], "y": [], "z": []}

    env = LorenzEnv(t_max, dt=dt)

    info = env.reset(initial_t, initial_x, initial_y, initial_z)
    done = info.pop("done")
    add_data(result, info) # データの保存
    while not done:
        info = env.step()
        done = info.pop("done")
        add_data(result, info) # データの保存

    # csvファイルの保存
    path = "hw1_result.csv"
    keys = list(result.keys())
    data = [[result[key][i] for key in keys] for i in range(len(result[keys[0]]))]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows([keys] + data)
    
    # pickleファイルの保存
    path = "hw1_result.pickle"
    with open(path, "wb") as f:
        pickle.dump(result, f)
```

### 課題2 (ファイルの入力)
課題1で保存したhw1_result.csvとhw1_result.pickleを読み込むプログラムを実装してください．csvファイルの読み込みにはcsvモジュール，pickleファイルの読み込みにはpickleモジュールを利用してください．(プログラムを実行する際は，カレントディレクトリが`hw4/(名前)`にあることを確認してください．)

**テンプレート**
```python
import csv
import pickle


if __name__ == "__main__":
    # csvファイルの読み込み
    print("csvファイルの読み込み")
    path = "hw1_result.csv"
    """
    以下にcsvファイルの読み込みを行うプログラムを実装してください．
    """
    

    # pickleファイルの読み込み
    print("pickleファイルの読み込み")
    path = "hw1_result.pickle"
    """
    以下にpickleファイルの読み込みを行うプログラムを実装してください．
    """
```

**解答例**
```python
import csv
import pickle


if __name__ == "__main__":
    # csvファイルの読み込み
    print("csvファイルの読み込み")
    path = "hw1_result.csv"
    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            for value in row:
                print(value, end=", ")
            print()

    # pickleファイルの読み込み
    print("pickleファイルの読み込み")
    path = "hw1_result.pickle"
    with open(path, "rb") as f:
        result = pickle.load(f)
        keys = list(result.keys())
        print(*keys)
        for i in range(len(result[keys[0]])):
            for key in keys:
                print(result[key][i], end=", ")
            print()
```

## 解説
### ファイルの入出力
ファイルから文字列を読み込んだり，ファイルに書き込んだりするには，`open`関数によってファイルを開く必要があります．ファイルを開くには一般に, `with`文を利用します．`with`文はファイル操作の開始時の前処理と終了時の後処理を自動で実行してくれるものになります．そして，`with`文のブロック内でファイル内部の読み込みや編集を行います．

```python
with open((パス名), (ファイルの読み書きモード)) as f:
    (ファイル内部へのアクセスと追加の処理)
```

`open`関数の第1引数にはファイルのパスを指定し，第2引数にはファイルの読み書きモードを設定します．ファイルの読み書きモードで代表的なものは次の表の通りです．

|  読み書きモード   |   説明  |
| --- | --- |
|  `r`   |  読み込み用に開く (デフォルト)．   |
|  `w`   |  書き込み用に開き，ファイルを上書きする．   |
|  `a`   |  書き込み用に開き，ファイルが存在する場合には末尾に追記する．   |
|  `b`   |  バイナリモード (`r`や`w`と併用する e.g., `rb`, `wb`)   |
|  `t`   |  テキストモード (デフォルト, `r`や`w`と併用する). `rt`は`r`と同義．  |
|  `+`   |  更新用に開く (読み込み・書き込み用, `r`や`w`と併用する). `r+`と`r+b`はファイルを上書きせずに開く．  |

### csvファイルの入出力
#### csvモジュール
csvの読み書きを行うときは，python標準ライブラリの`csv`モジュールを利用します．python標準ライブラリとは，pythonをダウンロードしたときに初めから入っているプログラムファイルをまとめたものを表しており，`csv`モジュールはそのライブラリの中に入っているプログラムファイルになります．`csv`モジュールを利用するためには，`import`を用いて次のように書きます．

```python
import csv
```

#### csvファイルの書き込み
csvファイルの書き込みには`csv`モジュール内の`writer`関数を利用します．モジュール内の関数を利用する際は，`(モジュール名).(関数名)`という形で書く必要があります．例えば，`writer`関数を利用する場合は`csv.writer(f)` (`f`: ファイルデータ)と書くことによってファイルの読み込みができます．そして，各行の数値を書き込む際は`writer.writerow([1次元リスト])`を利用し，複数行の数値を書き込む際は`writer.writerows([2次元リスト])`を利用します．以下では，`csv`モジュールを利用して，result.csvファイルに書き込むプログラムを実装しています．

```python
import csv

result = {"t": [0, 1, 2, 3], "x": [0, 0.2, -1.0, 1.0]}
keys = list(result.keys())
values = list(zip(t, x))

with open("result.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(keys)
    writer.writerows(values)
```

上記のプログラムを実行すると，result.csvファイルが作成され、ファイルの中身は次のようになります．

```
t, x
0, 0
1, 0.2
2, -1.0
3, 1.0
```

#### csvファイルの読み込み
csvファイルの読み込みには`csv`モジュール内の`reader`関数を利用します．result.csvというcsvファイルを読み込む場合，次のように書くことができます．

```python
import csv

with open("result.csv", "r") as f:
    reader = csv.reader(f)
```

`reader`はイテレータで，`for`文や`next`関数を利用して各行の要素を呼び出します．例えば，result.csvが次のようなファイルであったとします．

|  t   |   x  |
| --- | --- |
| 0 | 0 |
| 1 | 0.2 |
| 2 | -1.0 |
| 3 | 1.0 |

上記のファイルの各行を`print`文で出力するプログラムを作成すると，次のようになります．

```python
import csv

with open("result.csv", "r") as f:
    reader = csv.reader(f)

    for row in reader:
        print(row)
```

このプログラムを実行すると，コンソール上には次のような出力が表示されます．

```
t x
0 0
1 0.2
2 -1.0
3 1.0
```

ここで，`reader`の要素である`row`は文字列を要素にもつリストとなっており，`row`の要素は数値型になっていない点については注意してください．

### pickleファイルの入出力
#### pickleファイルとは
pickleファイルは，pythonオブジェクトをファイルとして保存することのできるファイルのことで，数値型や文字列型だけでなくリストや辞書型といった様々なオブジェクトを保存することができます．
#### pickleファイルの書き込み
pickleファイルの書き込みにはpython標準ライブラリの`pickle`モジュールの`dump`関数を利用します．`dump`関数は第１引数は保存するデータ，第2引数はファイルデータ(`f`)となります．ここで，pickleファイルを開く際は，書き込みモードを`wb`に設定することを忘れないでください．以下にプログラム例を記載します．

```python
import pickle

result = {"t": [0, 1, 2, 3], "x": [0, 0.2, -1.0, 1.0]}

with open("result.pickle", "wb") as f:
    result = pickle.dump(result, f)
```

上記のプログラムを実行することで，result.pickleファイルに辞書型オブジェクト`result`が保存されます．

#### pickleファイルの読み込み
pickleファイルの読み込みにはpython標準ライブラリの`pickle`モジュールの`load`関数を利用します．`load`関数の第１引数は，ファイルデータ(`f`)となります．ここで，pickleファイルを開く際は，読み込みモードを`rb`に設定することを忘れないでください．以下にプログラム例を記載します．

```python
import pickle

with open("result.pickle", "rb") as f:
    result = pickle.load(f)

print(result) # {"t": [0, 1, 2, 3], "x": [0, 0.2, -1.0, 1.0]}
```

上記のプログラムを実行すると，コンソール上には次のような出力が表示されます．

```
{"t": [0, 1, 2, 3], "x": [0, 0.2, -1.0, 1.0]}
```
