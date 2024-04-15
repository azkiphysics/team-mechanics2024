## 力学系ゼミ 第2回 プログラミング課題
### 概要
第2回は，制御構文 (if文, for文, while文)を扱います．制御構文はプログラミングをしたことがある人であれば聞いたことのある文法だと思います．Python特有の書き方もあったりするので，他の言語の制御構文と比較しつつ，勉強してもらえればと思います．

課題を作成する際は，hw2ディレクトリ内にフォルダ(フォルダ名: `(名前)`)を作成し (e.g., `ito`)，作成したフォルダ内に課題ごとのファイルを`answer(課題番号).py`として作成してください．(e.g., `answer1.py`, `answer2-1.py`)

課題を作成する際は，必ずブランチを切り，作成したブランチ上で作業を行うようにしてください ([ブランチの作成](https://github.com/azkiphysics/team-mechanics2024?tab=readme-ov-file#ブランチの作成))．

課題が作成できたら，GitHub上でプルリクエストを開き，伊藤(ユーザー名: azkiphysics)にマージの許可を得てください．伊藤が提出した課題のコードレビューを行い，コードの修正をしていただきます．修正が完了したらマージを行い，その週の課題は終了となります．([プルリクエストの作成](https://github.com/azkiphysics/team-mechanics2024?tab=readme-ov-file#プルリクエストの作成))

### 課題1 (if文)
コマンドプロンプトから入力された整数がゼロより小さければ0を，0以上で1より小さければ1を，1以上であれば2を出力するプログラムを作成してください．解答には以下のテンプレートを利用していただいて構いません．

**テンプレート**
```python
if __name__ == "__main__":
    # 整数の入力
    while True:
        x = input("Please enter an integer: ")
        try:
            assert "." not in x
            x = int(x)
            break
        except (ValueError, AssertionError):
            print(f"{x} is not an integer.")
            continue

    if x < 0:
        print(0)
    elif x < 1:
        print(1)
    else:
        print(2)
```

**解答例**
```python
if __name__ == "__main__":
    # 整数の入力
    while True:
        x = input("Please enter an integer: ")
        try:
            assert "." not in x
            x = int(x)
            break
        except (ValueError, AssertionError):
            print(f"{x} is not an integer.")
            continue

    if x < 0:
        print(0)
    elif x < 1:
        print(1)
    else:
        print(2)
```

### 課題2 (for文)
#### 課題2-1 (リストを使ったfor文)
リスト型オブジェクト`words = ["mechanics", "space_mission", "machine_learning", "complex_system"]`の要素である文字列とその文字列の長さを`print`関数を用いて出力してください．

**テンプレート**
```python
if __name__ == "__main__":
    words = ["mechanics", "space_mission", "machine_learning", "complex_system"]

    """
    以下にfor文を使った解答を作成してください．
    """
```

**解答例**
```python
if __name__ == "__main__":
    words = ["mechanics", "space_mission", "machine_learning", "complex_system"]

    for word in words:
        print(word, len(word))
```

#### 課題2-2 (`range`関数を使ったfor文)
`range`関数を用いて，0~9までの偶数を足し合わせて`print`文で出力してください．

**テンプレート**
```python
if __name__ == "__main__":
    answer = 0

    """
    以下にfor文とrange関数を使った解答を作成してください．
    """

    # answerの出力
    print(answer)
```

**解答例**
```python
if __name__ == "__main__":
    answer = 0
    
    for i in range(10):
        if i % 2 == 0:
            answer += i
    
    # answerの出力
    print(answer)
```

### 課題3 (while文)
$2^n > 1000$となる整数 $n$ を`while`文を用いて求めてください．

**テンプレート**
```python
if __name__ == "__main__":
    n = 0

    """
    以下にwhile文を使った解答を作成してください．
    """

    # 解答の出力
    print(n)
```

**解答例**
```python
if __name__ == "__main__":
    n = 0

    while 2 ** n <= 1000:
        n += 1

    print(n)
```

## 解説
### if文
`if`文は条件分岐を行うための制御文です．`if`文は`if (条件式): ... elif (条件式): ... else: ...`で構成されています．

`if`文は条件式が`True`である場合は`if`文内の処理を実行し，`False`である場合は，`if`文内の処理は実行されません．`if`文内の条件式が一致しない場合，次に`elif`文内の条件式が判定され，`True`である場合は`elif`文内の処理が実行され，そうでない場合は，`else`文内の処理が実行されます．

`if`文と`else`文は1回の条件分岐に対し，1つしか利用できませんが，`elif`文は1回の条件分岐に対し，何度も利用することができます．

`if`文の基本形は以下のとおりです．`elif`文と`else`文は必要に応じて記述します．また，条件式(条件A, B, ...)はブーリアン型が用いられます (比較演算子(`==`, `!=`, `>`, `<`, `in`, `is`)や論理演算子(`and`, `or`)，また整数が用いられることもあります)．条件式は`:`(コロン)で終了し，条件式内のブロックの中はインデントして記述します．

```python
if (条件式A):
    (処理A)
elif (条件式B):
    (処理B)
else:
    (処理C)
```

### for文
`for`文は, 繰り返し処理のことを指します．`for`文は，`for (イテラブルなオブジェクトの要素) in (イテラブルなオブジェクト): ...`の形で記述されます．ここで，イテラブルなオブジェクトとはシーケンス型オブジェクトなどの複数のオブジェクトが格納されたオブジェクトのことを指します．

`for`文の基本形は以下のとおりです．条件式は`:`(コロン)で終了し，条件式内のブロックの中はインデントして記述します．

```python
for (イテラブルなオブジェクトの要素) in (イテラブルなオブジェクト):
    (処理)
```

### while文
`while`文は，`for`文と同様，繰り返し処理のことを指します．`while`文は，`while (条件式): ...`の形で記述され，条件式が`True`である限り，`while`文内の処理が実行されます．

`while`文の基本形は以下のとおりです．条件式はブーリアン型が用いられます．条件式は`:`(コロン)で終了し，条件式内のブロックの中はインデントして記述します．

```python
while (条件式A):
    (処理A)
```

`for`文と`while`の違いは，`for`文は一定回数分だけ同じ繰り返し処理を行う一方，`while`文は条件式を満たしている間だけ同じ繰り返し処理を行います．

### 内包表記
内包表記とは，`for`文を用いてリストやタプル，辞書型といったシーケンス型オブジェクトの要素を定義する方法のことを指します．それぞれのシーケンス型の内包表記は以下のとおりです．

**リスト型**
```python
[value for (要素) in (イテラブルなオブジェクト)]
```

**タプル型**
```python
(value for (要素) in (イテラブルなオブジェクト))
```

**辞書型**
```python
{key: value for (要素) in (イテラブルなオブジェクト)}
```

内包表記を利用することでコードが簡潔，明解になるだけではなく，処理速度が高速になります．

内包表記を用いない場合，上記のコードは以下のように書けます．

**リスト型**

```python
a = []
for (要素) in (イテラブルなオブジェクト):
    ...
    a += [value,]
```

**タプル型**

```python
a = ()
for (要素) in (イテラブルなオブジェクト):
    ...
    a += (value,)
```

**辞書型**

```python
a = {}
for (要素) in (イテラブルなオブジェクト):
    ...
    a |= {key: value}
```

### if文を利用した内包表記
内包表記は`if`文を利用した条件分岐を行うことが可能です．`if`文を利用したシーケンス型の内包表記は以下のとおりです．

**リスト型**

```python
[value for (要素) in (イテラブルなオブジェクト) if (条件式)]
```

**タプル型**

```python
(value for (要素) in (イテラブルなオブジェクト) if (条件式))
```

**辞書型**

```python
{key: value for (要素) in (イテラブルなオブジェクト) if (条件式)}
```

内包表記を用いない場合，上記のコードは以下のように書けます．

**リスト型**

```python
a = []
for (要素) in (イテラブルなオブジェクト):
    ...
    if (条件式):
        a += [value,]
```

**タプル型**

```python
a = ()
for (要素) in (イテラブルなオブジェクト):
    ...
    if (条件式):
        a += (value,)
```

**辞書型**

```python
a = {}
for (要素) in (イテラブルなオブジェクト):
    ...
    if (条件式):
        a |= {key: value}
```

### if else文を利用した内包表記
内包表記は`if else`文を利用した条件分岐を行うことが可能です．`if`文を利用したシーケンス型の内包表記は以下のとおりです．

**リスト型**

```python
[value if (条件式) else other_value for (要素) in (イテラブルなオブジェクト)]
```

**タプル型**

```python
(value if (条件式) else other_value for (要素) in (イテラブルなオブジェクト))
```

**辞書型**

```python
{key: value if (条件式) else other_value for (要素) in (イテラブルなオブジェクト)}
```

内包表記を用いない場合，上記のコードは以下のように書けます．

**リスト型**

```python
a = []
for (要素) in (イテラブルなオブジェクト):
    ...
    if (条件式):
        a += [value,]
    else:
        a += [other_value,]
```

**タプル型**

```python
a = ()
for (要素) in (イテラブルなオブジェクト):
    ...
    if (条件式):
        a += (value,)
    else:
        a += (other_value,)
```

**辞書型**

```python
a = {}
for (要素) in (イテラブルなオブジェクト):
    ...
    if (条件式):
        a |= {key: value}
    else:
        a |= {key: other_value}
```
