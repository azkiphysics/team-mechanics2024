## 力学系ゼミ プログラミング課題
### 目的と扱う内容
本課題では，最終的に研究でPythonを使いこなせるようになることを目的としています．なので，数値計算や機械学習で利用するものしかここでは扱いません．具体的には以下を扱います．
- Python文法
    - 型
    - 制御構文(if文, for文, while文)
    - 関数とクラス
    - ファイルの入出力
    - モジュールの利用 (numpy, matplotlib)
- 数値計算法
    - ニュートン法
    - オイラー法
    - ルンゲクッタ法
- 最適制御
    - LQR制御
    - 強化学習

力学系ゼミのシラバスにしたがって，以下のプログラミング課題を出します．
1. Python文法 (型)
2. Python文法 (制御構文 (if文, for文, while文))
3. Python文法 (関数とクラス)
4. Python文法 (ファイルの入出力)
5. Python文法 (numpy, matplotlib)
6. 数値計算法 (オイラー法, ルンゲクッタ法)
8. LQR制御
9. 深層強化学習 (DQN, DDPG, TD3)

## 準備
本課題をするにあたり，次の3点を行ってください．
### Visual Studio Codeのインストール
#### Visual Studio Codeとは
Visual Studio Code (VS Code)はMicrosoftが開発しているソースコードエディタです．VS Codeを利用することで，効率よくコードを書くことができます．
#### インストール
[VS Code](https://code.visualstudio.com/download)のサイトからインストールしてください．
### Anacondaのインストール
#### Anacondaとは
AnacondaはPythonで科学計算を行うためのディストリビューションです．一般に通常のPython環境よりも計算の高速化が可能であるといわれています．また，Anacondaでは，仮想環境を利用することでバージョンの異なるPythonを簡単に利用することができます．
#### インストール
[Anaconda](https://www.anaconda.com/download)のサイトからインストールしてください．
### GitHub Desktopのインストール
#### Gitとは
Gitはソースコードや変更履歴を管理するために使われるバージョン管理システムです．仮にソースコードに意図しない変更が加えられたとしても，容易に元に戻すことができるため，複数のユーザーと共有して共同作業を行ったり，ソフトウェア開発プロジェクトを管理するのに利用されます．
#### GitHubとは
[GitHub](https://github.co.jp/)はGitの仕組みを利用したソフトウェア開発のプラットフォームです．オンラインで無料で利用できるため，個人，企業問わず様々なところで利用されています．
#### GitHub Desktopとは
通常，GitHubはコマンドプロンプト(= Command User Interface, CUI)を利用し，gitコマンドをGitHubに送受信することでGitHub上のレポジトリ(≒フォルダ)の更新を行います (e.g., `git push REMOTE-NAME BRANCH-NAME`)．一方，GitHub Desktopでは，Github Desktopの専用アプリ(=Graphic User Interface, GUI)を利用し，GitHubとのやり取りを行います．これにより，gitコマンドを知らなくても簡単にレポジトリの更新を行うことができます．本課題では基本的にGitHub Desktopを利用して，課題の提出を行っていただきます．
#### インストール
[Git](https://git-scm.com/book/ja/v2/使い始める-Gitのインストール)のインストールと[GitHub Desktop](https://desktop.github.com/)のインストールを行なってください．

## Pythonの使い方
### 仮想環境の作成と有効化
本課題を行うにあたり，condaの仮想環境を作成します．コマンドプロンプト(端末, ターミナル)から以下のコマンドを実行することでteam_mechanics2024という仮想環境を作成します．ここでは，Pythonのバージョンが3.11.2のものを利用します．
```zsh
conda create -n team_mechanics2024 python==3.11.2
```

team_mechanics2024の仮想環境を有効化するには以下のコマンドを実行します．
```zsh
conda activate team_mechanics2024
```

team_mechanics2024の仮想環境を無効化するには以下のコマンドを実行します．
```zsh
conda deactivate
```

### プログラムの作成と実行
Pythonファイルは*.py (*は任意の文字列)により作成することができます．ここでは，以下のソースコード(hello.py)を作成したとします．
```python:hello.py
print("Hello, world!")
```
上記のソースコードを実行するためには，コマンドプロンプトから以下のコマンドを実行します．プログラムを実行する前に必ず前節で作成したteam_mechanics2024の仮想環境を有効化するようにしてください．
```zsh
python hello.py
```
コマンドを実行すると，コマンドプロンプトに`Hello, world!`と表示されます．

## GitHub Desktopの使い方
### team-mechanics2024レポジトリのクローン
レポジトリをPCにダウンロードすることをクローンと言います．レポジトリのクローンは以下の手順で行います．
1. GitHub Desktopアプリ左上のCurrent Repositoryを押し，検索欄の右のAddより，clone repository...を選択します．
2. `azkiphysics/team-mechanics2024`という名前のレポジトリを選択し，右下のCloneよりレポジトリをローカルにクローンします．

**参考文献**
1. [【入門】Github Desktopとは？インストール・使い方](https://www.kagoya.jp/howto/it-glossary/develop/githubdesktop/)

### ブランチの作成
本課題を行うときは，新たにブランチを作成し，作成したブランチ先で作業を行ってもらいます．ブランチの切り方は以下の通りです．
1. GitHub Desktopアプリ真ん中上のCurrent Branchを押し，検索欄の右のNew Branchより，ブランチを作成します．ブランチ名は`feature-hw(課題番号)-(名前)`という命名規則のもと作成するようにしてください (e.g., `feature-hw1-ito`)．
2. GitHub Desktopアプリ真ん中上のCurrent Branchを押すと，Default Branch内に1.で作成したブランチがあるので，そのブランチを選択します．

**参考文献**
1. [【入門】Github Desktopとは？インストール・使い方](https://www.kagoya.jp/howto/it-glossary/develop/githubdesktop/)

### GitHubへの変更のプッシュ
GitHub上のレポジトリにPC上のローカルレポジトリの変更を反映することをプッシュと言います．GitHubへの変更のプッシュは以下の手順で行います．
1. GitHub Desktopアプリの真ん中上のCurrent Branchが自分の作成したブランチであることを確認してください．
2. GitHub Desktopアプリ上で，変更したファイルは左側に表示されていますので，変更内容が問題ないかと，ボックスにチェックが入っているか確認します．
3. 左下にコメントを記載し，`Commit to (ブランチ名)`を押して，変更をコミットします．
4. GitHub Desktopアプリの右側のボタン`Push origin`を押してGitHubへ変更をプッシュする．

**参考文献**
1. [GitHub Desktop から GitHub に変更をプッシュする](https://docs.github.com/ja/desktop/making-changes-in-a-branch/pushing-changes-to-github-from-github-desktop)
2. [【入門】Github Desktopとは？インストール・使い方](https://www.kagoya.jp/howto/it-glossary/develop/githubdesktop/)

### ブランチのマージ
作成したブランチで作業した後，他のブランチへ更新内容を反映させたい場合，ブランチのマージを行います．本課題では，主にmainブランチの更新内容を反映するために，mainブランチから作成したブランチへマージを行います．マージの手順は以下の通りです．
1. GitHub Desktopアプリの真ん中上のCurrent Branchから，`Choose a branch to merge into (ブランチ名)`を選択します．
2. 更新内容を取り込みたいブランチ(本課題では`main`ブランチ)を選択し，画面下部の`Create a merge commit`を選択します．これにより，マージが完了します．

**参考文献**
1. [【入門】Github Desktopとは？インストール・使い方](https://www.kagoya.jp/howto/it-glossary/develop/githubdesktop/)

### プルリクエストの作成
作成したブランチの変更内容を別のブランチに反映させる場合には，プルリクエストを開き，別の共同作業者にマージの許可をもらいます．プルリクエストの開き方は以下の手順で行います．
1. [GitHubのteam-mechanics2024レポジトリ](https://github.com/azkiphysics/team-mechanics2024)を開きます．
2. 左上のブランチメニューを開き，作成したブランチを選択します．
3. ファイルの一覧の上にある黄色のバナーで，[compare & pull request] をクリックして，関連付けられているブランチ(本課題では`main`ブランチ)のpull requestを作成します．
4. 変更をマージする対象のブランチを [base] ブランチ (`main`ブランチ) ドロップダウン メニューで選択し、次に [compare] ブランチ ドロップダウン メニューを使用して、変更を行ったトピック ブランチ(`feature-hw(課題番号)-(名前)`)を選択します．
5. プルリクエストのタイトルと説明を入力します．
6. レビューワーを伊藤 (アカウント名: `azkiphysics`)にし，画面下部の[pull requestの作成]を選択することでpull requestを作成できます．

**参考文献**
1. [pull request の作成](https://docs.github.com/ja/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)
