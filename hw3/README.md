## 力学系ゼミ 第3回 プログラミング課題
### 概要
第3回は，関数とクラスを扱います．

課題を作成する際は，hw3ディレクトリ内にフォルダ(フォルダ名: `(名前)`)を作成し (e.g., `ito`)，作成したフォルダ内に課題ごとのファイルを`answer(課題番号).py`として作成してください．(e.g., `answer1.py`, `answer2-1.py`)

課題を作成する際は，必ずブランチを切り，作成したブランチ上で作業を行うようにしてください ([ブランチの作成](https://github.com/azkiphysics/team-mechanics2024?tab=readme-ov-file#ブランチの作成))．

課題が作成できたら，GitHub上でプルリクエストを開き，伊藤(ユーザー名: azkiphysics)にマージの許可を得てください．伊藤が提出した課題のコードレビューを行い，コードの修正をしていただきます．修正が完了したらマージを行い，その週の課題は終了となります．

### 課題1 (関数)
フィボナッチ数列 ${F_n}$ は次の漸化式で定義されます．

$$
\begin{eqnarray}
    && F_0 = 0\\
    && F_1 = 1\\
    && F_{n + 2} = F_n + F_{n + 1}\ (n \geq 0)
\end{eqnarray}
$$

このとき， $F_{10}$ を`print`文で出力してください．

**テンプレート**
```python
def fibonacci(n):
    """
    以下に解答を書いてください．
    """

if __name__ == "__main__":
    answer = fibonacci(10)
    print(answer)
```

**解答例**
```python
def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 2) + fibonacci(n - 1)

if __name__ == "__main__":
    answer = fibonacci(10)
    print(answer)
```

### 課題2 (クラス)
以下のソースコードをコピペしてプログラムを実行してください．ここでは，クラスの書き方とプログラムの処理の流れをざっくり理解してもらえればOKです．

```python
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


if __name__ == "__main__":
    t_max = 0.1
    dt = 1e-3
    initial_t = 0.0
    initial_x = 1.0
    initial_y = 1.0
    initial_z = 1.0

    env = Env(t_max, dt=dt)

    info = env.reset(initial_t, initial_x, initial_y, initial_z)
    done = info.pop("done")
    print(f"t={info['t']:.3f}, x={info['x']:.3f}, y={info['y']:.3f}, z={info['z']:.3f}")
    while not done:
        info = env.step()
        done = info.pop("done")
        print(f"t={info['t']:.3f}, x={info['x']:.3f}, y={info['y']:.3f}, z={info['z']:.3f}")
```

### 課題3 (クラスの継承)
カオス的な挙動を示すシステムの代表例として，ローレンツ系がよく用いられます．ローレンツ系は，大気挙動の簡易数学モデルとして，1963年に数学者・気象学者であるエドワード・ローレンツによって提案されました．ローレンツ方程式は以下の3つの方程式によって表されます．

$$
\begin{eqnarray}
    \frac{dx}{dt} &=& \sigma (y - x)\\
    \frac{dy}{dt} &=& x(\rho - z) - y\\
    \frac{dz}{dt} &=& xy - \beta z
\end{eqnarray}
$$

ここでは，ローレンツ系のクラスを課題2で作成した`Env`クラスを継承して作成してもらいます．下のテンプレートを利用して，ローレンツ系のクラス(`LorenzEnv`)を作成し，プログラムを実行してください．

**テンプレート**
```python
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
        """
        motion_equationメソッド内にローレンツ方程式を実装してください．
        戻り値はdx_dt, dy_dt, dz_dtの値を返すようにしてください．
        """


if __name__ == "__main__":
    t_max = 0.1
    dt = 1e-3
    initial_t = 0.0
    initial_x = 1.0
    initial_y = 1.0
    initial_z = 1.0

    env = LorenzEnv(t_max, dt=dt)

    info = env.reset(initial_t, initial_x, initial_y, initial_z)
    done = info.pop("done")
    print(f"t={info['t']:.3f}, x={info['x']:.3f}, y={info['y']:.3f}, z={info['z']:.3f}")
    while not done:
        info = env.step()
        done = info.pop("done")
        print(f"t={info['t']:.3f}, x={info['x']:.3f}, y={info['y']:.3f}, z={info['z']:.3f}")
```

**解答例**
```python
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

    env = LorenzEnv(t_max, dt=dt)

    info = env.reset(initial_t, initial_x, initial_y, initial_z)
    done = info.pop("done")
    print(f"t={info['t']:.3f}, x={info['x']:.3f}, y={info['y']:.3f}, z={info['z']:.3f}")
    while not done:
        info = env.step()
        done = info.pop("done")
        print(f"t={info['t']:.3f}, x={info['x']:.3f}, y={info['y']:.3f}, z={info['z']:.3f}")
```

## 解説
### 関数
関数は，複数の処理をひとまとめにしたものになります．関数を定義することで，可読性が向上するだけでなく，再帰構造を利用する場合や関数をコンパイルして高速化する場合などに役立ちます．

関数は次のような形式で書かれます．関数は`:`(コロン)で終了し，関数内のブロックの中はインデントして記述します．

```python
def function_name(variable):
    (処理内容)
    return result
```

一般にクラス名は単語間を_(アンダースコア)で結び，単語の文字はすべて小文字にします．

### クラス
クラスは，変数と関数をひとまとめにしたものになります．クラスを定義することで，変数と関数を用途ごとに分類することができるため，大規模なプログラムを開発する時によく用いられます．クラス内の変数は"アトリビュート"，関数は"メソッド"と呼ばれます．

クラスは次のような形式で書かれます．クラスは`:`(コロン)で終了し，クラス内やメソッド内のブロックの中はインデントして記述します．

```python
class ClassName:
    def __init__(self, attribute):
        self.attribute = attribute # アトリビュートの定義

    def method(self, attribute):
        (処理内容)
        return result
```

一般にクラス名は単語の初めの文字を大文字にし，他の文字は小文字にします．クラス内アトリビュートは`self`を用いて，`self.(アトリビュート名)`として定義されます (e.g., `self.attribute`). またクラス内メソッドの多くは後から定義しますが，もともと実装されているメソッドも存在します．`__init__`メソッド(=コンストラクタ)もあらかじめ実装されているメソッドであり，クラスを用いてインスタンスを生成された際にまず最初に呼び出されます．ここで，インスタンスとは，クラスから作成したオブジェクトのことを指します．例えば，次のような`Person`クラスがあったとします．ここでは，コンストラクタを上書きして，外部から与えられた変数`name`をアトリビュート`self.name`に定義しています．

```python
class Person:
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name
```

上記の`Person`クラスを用いて`Kevin`という名前のオブジェクト`kevin`を作成した場合，`kevin`は`Person`クラスのインスタンスとなります．

```python
kevin = Person("Kevin")
print(kevin.get_name()) # Kevin
```

ここで，クラス内のメソッドの第一引数には`self`という変数が入っていますが，これはメソッドを書くときには必ず必要となるので，書き忘れがないように気を付けてください．この`self`という変数があることでコンストラクタ(`__init__`)や別のメソッドで定義されたアトリビュートにアクセスすることができます．例えば，上の`Person`クラスの場合だと，コンストラクタ内で定義された`self.name`が`get_name`メソッド内で呼び出されています．このように，クラス内で定義されたアトリビュートはクラス内で定義されたメソッドであればどのメソッドでも利用できるため，アトリビュートをメソッドの引数に入れる必要がありません (ただし，状況によってはアトリビュートをメソッドの引数に入れる場合もあります)．

### クラスの継承
既存のクラスをもとに新しいクラスを作成し，既存のクラスのアトリビュートやメソッドを受け継がせることを継承といいます．このとき，もとにしたクラスをスーパークラス(親クラス)と呼び，新しいクラスをサブクラス(子クラス)と呼びます．

例えば，親クラスとして次の`Env`クラスがあったとします．

```python
class Env:
    def __init__(self, name, dt=1e-3):
        self.name = name
        self.dt = dt

        self.t = None
        self.x = None
        self.vx = None
        self.theta = None

    def integral(self, u):
        ...

    def reset(self):
        self.t = 0.0
        self.x = 0.0
        self.vx = 0.0
        self.theta = 0.1
        return {"t": self.t, "x": self.x, "vx": self.vx, "theta": self.theta}

    def step(self, u):
        self.t += self.dt
        self.integral(u)
        return {"t": self.t, "x": self.x, "vx": self.vx, "theta": self.theta}
```

上記の`Env`クラスから`CartPoleEnv`クラスを作成する場合，コンストラクタと`integral`メソッドだけ上書きすることで実装することができます．

```python
class CartPoleEnv(Env):
    def __init__(self, dt=1e-3):
        name = "CartPoleEnv"
        super().__init__(name, dt=dt)

    def integral(self, u):
        (CartPole問題の数値計算に関する処理)
```

ここで，`super().(メソッド名)`は，親クラスのメソッドを呼び出す場合に使用され，上記の場合だと，`CartPoleEnv`のコンストラクタ内で`Env`のコンストラクタを呼び出すために，`super().__init__`を用いています．また，`CartPoleEnv`のインスタンスから`Env`のメソッドを呼び出す場合は単純に`(CartPoleEnvのインスタンス名).(Envのメソッド名)`と書けば呼び出すことができます．

```python
env = CartPoleEnv()
print(env.reset()) # {"t": 0.0, "x": 0.0, "vx": 0.0, "theta": 0.1}
```

このように，クラスの継承を利用することで，少ないコード量で新しいクラスを作成することができます．また，クラスのメソッドを共通させることで，より拡張性の高いプログラムを書くことができます．したがって，クラスと継承をマスターすれば，何度も似たようなプログラムを書くことなく，一つのプログラムだけで複数の力学系のシミュレーションを行うことが可能となるので，ぜひクラスを使いこなせるようにしてみてください．
