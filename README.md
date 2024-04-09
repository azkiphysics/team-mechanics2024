## 力学系ゼミ
### 目的と扱う内容
本課題では，最終的に研究でPythonを使いこなせるようになることを目的としています．なので，数値計算や機械学習で利用するものしかここでは扱いません．具体的には以下を扱います．
- Python文法
    - 型
    - 制御構文(for文, while文)
    - 関数
    - クラス
    - ファイルの入出力
    - numpy
    - matplotlib
- 数値計算法
    - オイラー法
    - ルンゲクッタ法
- 最適制御
    - LQR制御
    - 強化学習

力学系ゼミのシラバスにしたがって，以下のプログラミング課題を出します．
1. Python文法 (型)
2. Python文法 (制御構文 (for文, while文))
3. Python文法 (関数とクラス)
4. Python文法 (ファイルの入出力)
5. Python文法 (numpy, matplotlib)
6. 数値計算法 (オイラー法, ルンゲクッタ法)
7. 倒立振子のシミュレーション
8. LQR制御
9. 強化学習 (Q-learning, DQN)

## 準備
本課題をするにあたり，次の3点を行ってください．
### Visual Studio Code
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
[GitHub Desktop](https://desktop.github.com/)のサイトからインストールしてください．
