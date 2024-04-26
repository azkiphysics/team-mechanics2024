## 力学系ゼミ 第8回 プログラミング課題
### 概要
第8回は，第6回で実装した倒立振子環境(`CartPoleEnv`)を用いて，機械学習手法の一つである深層強化学習により, カート上のポールの倒立状態を目標地点で維持するという課題を行います．

課題を作成する際は，hw8ディレクトリ内にフォルダ(フォルダ名: `(名前)`)を作成し (e.g., `ito`)，作成したフォルダ内に課題ごとのファイルを`answer(課題番号).py`として作成してください．(e.g., `answer1.py`, `answer2-1.py`)

課題を作成する際は，必ずブランチを切り，作成したブランチ上で作業を行うようにしてください ([ブランチの作成](https://github.com/azkiphysics/team-mechanics2024?tab=readme-ov-file#ブランチの作成))．

課題が作成できたら，GitHub上でプルリクエストを開き，伊藤(ユーザー名: azkiphysics)にマージの許可を得てください．伊藤が提出した課題のコードレビューを行い，コードの修正をしていただきます．修正が完了したらマージを行い，その週の課題は終了となります．([プルリクエストの作成](https://github.com/azkiphysics/team-mechanics2024?tab=readme-ov-file#プルリクエストの作成))

### 準備
#### ライブラリのインストール
今回使用するライブラリのインストールを行います．以下のコマンドを入力して`pytorch`ライブラリをインストールしてください．

```zsh
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

もし，お使いのPCに`GeForce`のグラボが搭載されている場合は，上記2行目のコマンドを以下のように書き換えてください．

```zsh
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia # CUDA12.1の場合のコマンド．CUDA 11.8の場合は12.1を11.8に変更する．
```

#### テンプレートのコピー&ペースト
ご自身で作成したディレクトリに`templates`内のファイルとフォルダをコピー&ペーストしてください．ファイルの構成はそのままでコピペするようにしてください．(ファイル名を変えるとエラーがでます．)

コピペ後のファイルの構成が以下のようになっていれば問題ありません．

```
hw7
├── README.md
...
└── (ご自身で作成したディレクトリ)
    ├── common
    |   ├── agents.py
    |   ├── buffers.py
    |   ├── envs.py
    |   ├── utils.py
    |   └── wrappers.py
    ├── configs
    |   ├── CartPoleEnv
    |   |   ├── DDPG.yaml
    |   |   ├── LQR.yaml
    |   |   └── TD3.yaml
    |   └── gym
    |       ├── CartPole-v1
    |       |   └── DQN.yaml
    |       └── Pendulum-v1
    |           ├── DDPG.yaml
    |           └── TD3.yaml
    ├── answer1.py
    └── answer2.py
```

#### プログラム実行時の注意点
プログラムを実行する場合は，必ずご自身のディレクトリに移動してください．相対パスでモジュールをインポートしているため，別のディレクトリからプログラムを実行すると，エラーが出る可能性があります．

カレントディレクトリが`team-mechanics`の場合は以下のコマンドを実行すると，カレントディレクトリを移動できます．

**Windowsの場合**

```zsh
cd hw7\(ご自身で作成したディレクトリ)
```

**Mac/Linuxの場合**

```zsh
cd hw7/(ご自身で作成したディレクトリ)
```