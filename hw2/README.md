## 力学系ゼミ 第1回 プログラミング課題
### 概要
第2回は，制御構文 (if文, for文, while文)を扱います．

課題を作成する際は，hw2ディレクトリ内にフォルダ(フォルダ名: `(名前)`)を作成し (e.g., `ito`)，作成したフォルダ内に課題ごとのファイルを`answer(課題番号).py`として作成してください．(e.g., `answer1.py`, `answer2-1.py`)

課題を作成する際は，必ずブランチを切り，作成したブランチ上で作業を行うようにしてください ([ブランチの作成](https://github.com/azkiphysics/team-mechanics2024?tab=readme-ov-file#ブランチの作成))．

課題が作成できたら，GitHub上でプルリクエストを開き，伊藤(ユーザー名: `azkiphysics`)にマージの許可を得てください．マージが完了した時点でその週の課題は終了となります．

### 課題1 (if文)
コマンドプロンプトから入力された整数がゼロより小さければ0を，0以上で1より小さければ1を，1以上であれば2を出力するプログラムを作成してください．解答には以下のテンプレートを利用していただいて構いません．

**テンプレート**
```python
# 整数の入力
while True:
    x = input("Please enter an integer: ")
    if isinstance(x, int):
        break

"""
以下にif文を使った解答を作成してください．
"""
```

**解答例**
```python
# 整数の入力
while True:
    x = input("Please enter an integer: ")
    if isinstance(x, int):
        break

"""
以下にif文を使った解答を作成してください．
"""
if x < 0:
    print(0)
elif x < 1:
    print(1)
else:
    print(2)
```

### 課題2 (for文)
### 課題3 (while文)

