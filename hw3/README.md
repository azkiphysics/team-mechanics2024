## 力学系ゼミ 第3回 プログラミング課題
### 概要
第3回は，関数とクラスを扱います．

課題を作成する際は，hw2ディレクトリ内にフォルダ(フォルダ名: `(名前)`)を作成し (e.g., `ito`)，作成したフォルダ内に課題ごとのファイルを`answer(課題番号).py`として作成してください．(e.g., `answer1.py`, `answer2-1.py`)

課題を作成する際は，必ずブランチを切り，作成したブランチ上で作業を行うようにしてください ([ブランチの作成](https://github.com/azkiphysics/team-mechanics2024?tab=readme-ov-file#ブランチの作成))．

課題が作成できたら，GitHub上でプルリクエストを開き，伊藤(ユーザー名: azkiphysics)にマージの許可を得てください．伊藤が提出した課題のコードレビューを行い，コードの修正をしていただきます．修正が完了したらマージを行い，その週の課題は終了となります．

### 課題1 (関数)
フィボナッチ数列 ${F_n}$ は次の漸化式で定義されます．

$$
\begin{eqnarray}
    F_0 &=& 0\\
    F_1 &=& 1\\
    F_{n + 2} &=& F_n + F_{n + 1} (n \geq 0)
\end{eqnarray}
$$

このとき，$F_{10}$ を`print`文で出力してください．

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
