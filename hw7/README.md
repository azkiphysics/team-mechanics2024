## 力学系ゼミ 第7回 プログラミング課題
### 概要
第7回は，第6回で実装した倒立振子環境(`CartPoleEnv`)を用いて，LQR制御と呼ばれる現代制御手法によりカート上のポールの倒立状態を維持する課題を行います．

課題を作成する際は，hw6ディレクトリ内にフォルダ(フォルダ名: `(名前)`)を作成し (e.g., `ito`)，作成したフォルダ内に課題ごとのファイルを`answer(課題番号).py`として作成してください．(e.g., `answer1.py`, `answer2-1.py`)

課題を作成する際は，必ずブランチを切り，作成したブランチ上で作業を行うようにしてください ([ブランチの作成](https://github.com/azkiphysics/team-mechanics2024?tab=readme-ov-file#ブランチの作成))．

課題が作成できたら，GitHub上でプルリクエストを開き，伊藤(ユーザー名: azkiphysics)にマージの許可を得てください．伊藤が提出した課題のコードレビューを行い，コードの修正をしていただきます．修正が完了したらマージを行い，その週の課題は終了となります．([プルリクエストの作成](https://github.com/azkiphysics/team-mechanics2024?tab=readme-ov-file#プルリクエストの作成))

### 準備
今回と次回で使用するライブラリのインストールを行います．以下のコマンドを入力して`pyyaml`, `tqdm`, `pytorch`ライブラリをインストールしてください．

```zsh
conda install pyyaml tqdm
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

もし，お使いのPCに`GeForce`のグラボが搭載されている場合は，上記2行目のコマンドを以下のように書き換えてください．

```zsh
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia # CUDA12.1の場合のコマンド．CUDA 11.8の場合は12.1を11.8に変更する．
```

### 課題1 (倒立振子のLQR制御)
課題1では，LQR制御により倒立振子環境の外力項 $u$ に適切な値を入力することで，倒立維持を行います．

### 課題2 (LQR制御のパレート解)

### 解説
#### 倒立振子の運動方程式の線形化
LQR制御では，線形な運動方程式を利用します．そこで，倒立振子環境を平衡点まわりで線形化することを考えます．いま，倒立振子の独立変数を $\boldsymbol{x}$ とし，平衡点を $\boldsymbol{x}_e$ , $\boldsymbol{u}_e$ とします． 独立変数 $\boldsymbol{x}$ に対する運動方程式を $\dot{\boldsymbol{x}} = \boldsymbol{f}(\boldsymbol{x})$ としたとき， $\boldsymbol{x} = \boldsymbol{x}_e$ , $\boldsymbol{u}=\boldsymbol{u}_e$ まわりでテイラー展開すると次式のように表されます．

$$
\frac{d}{dt}(\boldsymbol{x} - \boldsymbol{x}_e) = \frac{\partial \boldsymbol{f}}{\partial \boldsymbol{x}}\Bigg|_{(\boldsymbol{x}_e, \boldsymbol{u}_e)}(\boldsymbol{x} - \boldsymbol{x}_e) + \frac{\partial \boldsymbol{f}}{\partial \boldsymbol{u}}\Bigg|_{(\boldsymbol{x}_e, \boldsymbol{u}_e)}(\boldsymbol{u} - \boldsymbol{u}_e)
$$

$\bar{\boldsymbol{x}} = \boldsymbol{x} - \boldsymbol{x}_e$ , $\mathrm{A} = \frac{\partial \boldsymbol{f}}{\partial \boldsymbol{x}}\big|_{(\boldsymbol{x}_e, \boldsymbol{u}_e)}$ , $\mathrm{B} = \frac{\partial \boldsymbol{f}}{\partial \boldsymbol{u}}\big|_{(\boldsymbol{x}_e, \boldsymbol{u}_e)}$ とすると，以下のように線形方程式を書き換えることができます．

$$
\dot{\bar{\boldsymbol{x}}} = \mathrm{A}\bar{\boldsymbol{x}} + \mathrm{B}\bar{\boldsymbol{u}}
$$

以下ではこの線形運動方程式を利用してLQR制御の導出を行います．

#### LQR制御
LQR制御では，以下の目的関数を最小にするように制御入力を決定します．ここで, $\mathrm{Q}$ , $\mathrm{Q}_f$ は半正定値対角行列, $\mathrm{R}$ は正定値対角行列を表します．

$$
J = \int_{t=0}^{t_{\mathrm{max}}}\frac{1}{2}\Big(\bar{\boldsymbol{x}}^T\mathrm{Q}\bar{\boldsymbol{x}} + \bar{\boldsymbol{u}}^T\mathrm{R}\bar{\boldsymbol{u}}\Big)dt + \frac{1}{2}\bar{\boldsymbol{x}}(t_f)^T\mathrm{Q}_f\bar{\boldsymbol{x}}(t_f)
$$

ここでは，線型方程式 $\dot{\bar{\boldsymbol{x}}} = \mathrm{A}\bar{\boldsymbol{x}} + \mathrm{B}\bar{\boldsymbol{u}}$ を制約条件として, 目的関数 $J$ を最小化するので，実際には以下の目的関数 $J_{\mathrm{aug}}$ を最小化します．ここで，変数 $\boldsymbol{\lambda}$ はラグランジュの未定乗数を表しており，下式の右辺第1項の被積分関数に $\boldsymbol{\lambda}^T(\mathrm{A}\bar{\boldsymbol{x}} + \mathrm{B}\bar{\boldsymbol{u}} - \dot{\bar{\boldsymbol{x}}})$ を加えることで，運動制約を考慮した最適化が可能となります．

$$
J_{\mathrm{aug}} = \int_{t=0}^{t_{\mathrm{max}}}\Bigg\{\frac{1}{2}\Big(\bar{\boldsymbol{x}}^T\mathrm{Q}\bar{\boldsymbol{x}} + \bar{\boldsymbol{u}}^T\mathrm{R}\bar{\boldsymbol{u}}\Big) + \boldsymbol{\lambda}^T(\mathrm{A}\bar{\boldsymbol{x}} + \mathrm{B}\bar{\boldsymbol{u}} - \dot{\bar{\boldsymbol{x}}})\Bigg\}dt + \frac{1}{2}\bar{\boldsymbol{x}}(t_f)^T\mathrm{Q}_f\bar{\boldsymbol{x}}(t_f)
$$

目的関数 $J_{\mathrm{aug}}$ を最小にするための必要条件は，変分 $\delta J_{\mathrm{aug}}$ がゼロになることです．そこで，変分 $\delta J_{\mathrm{aug}}$ を計算すると

$$
\begin{eqnarray}
    \delta J_{\mathrm{aug}} &=& \int_{t=0}^{t_{\mathrm{max}}}\Bigg\{\Big(\bar{\boldsymbol{x}}^T\mathrm{Q}\delta\bar{\boldsymbol{x}} + \bar{\boldsymbol{u}}^T\mathrm{R}\delta\bar{\boldsymbol{u}}\Big) + \delta\boldsymbol{\lambda}^T(\mathrm{A}\bar{\boldsymbol{x}} + \mathrm{B}\bar{\boldsymbol{u}} - \dot{\bar{\boldsymbol{x}}}) + \boldsymbol{\lambda}^T(\mathrm{A}\delta\bar{\boldsymbol{x}} + \mathrm{B}\delta\bar{\boldsymbol{u}} - \delta\dot{\bar{\boldsymbol{x}}})\Bigg\}dt + \bar{\boldsymbol{x}}(t_f)^T\mathrm{Q}_f\delta\bar{\boldsymbol{x}}(t_f)

    &=& \int_{t=0}^{t_{\mathrm{max}}}\Big\{(\bar{\boldsymbol{x}}^T\mathrm{Q} + \boldsymbol{\lambda}^T\mathrm{A} + \dot{\boldsymbol{\lambda}}^T)\delta\bar{\boldsymbol{x}} + (\bar{\boldsymbol{u}}^T\mathrm{R} + \boldsymbol{\lambda}^T\mathrm{B})\delta\bar{\boldsymbol{u}} + (\mathrm{A}\bar{\boldsymbol{x}} + \mathrm{B}\bar{\boldsymbol{u}})^T\delta\boldsymbol{\lambda}\Big\}dt + \Big\{\bar{\boldsymbol{x}}(t_f)^T\mathrm{Q}_f - \boldsymbol{\lambda}^T(t_f)\Big\}\delta\bar{\boldsymbol{x}}(t_f)
\end{eqnarray}
$$

となるので，$\delta J_{\mathrm{aug}} = 0$ となるための条件は，

$$
\begin{cases}
    \bar{\boldsymbol{x}}^T\mathrm{Q} + \boldsymbol{\lambda}^T\mathrm{A} + \dot{\boldsymbol{\lambda}}^T = \boldsymbol{0}\\
    \bar{\boldsymbol{u}}^T\mathrm{R} + \boldsymbol{\lambda}^T\mathrm{B} = \boldsymbol{0}\\
    \dot{\bar{\boldsymbol{x}}} = \mathrm{A}\bar{\boldsymbol{x}} + \mathrm{B}\bar{\boldsymbol{u}}\\
    \bar{\boldsymbol{x}}(t_f)^T\mathrm{Q}_f - \boldsymbol{\lambda}^T(t_f) = \boldsymbol{0}
\end{cases}
$$
