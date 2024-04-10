import csv
import pickle


if __name__ == "__main__":
    # csvファイルの読み込み
    print("csvファイルの読み込み")
    path = "hw1_result.csv"
    with open(path, "r", encoding="utf-8", newline="") as f:
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
