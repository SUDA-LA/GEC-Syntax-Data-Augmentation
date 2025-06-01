from supar import Parser
import sys
import pickle
import torch
import os


def load(filename):
    with open(filename, 'r', encoding="utf8") as f:
        data = [line.split(" ") for line in f.read().split("\n") if line]
        return data


# dep = Parser.load(sys.argv[3])
con = Parser.load('crf-con-zh')

input_sentences = load(sys.argv[1])
n=10000
input_split = [input_sentences[i:i + n] for i in range(0, len(input_sentences), n)]
# 容易中断，下次从断点开始
idx = 0
while True:
    if idx >= len(input_split):
        break
    print(f"正在处理{idx}/{len(input_split)}")
    input_texts = input_split[idx]
    res = con.predict(input_sentences, verbose=False, buckets=25, batch_size=500, prob=False)
    # probs = []

    with open(sys.argv[2], 'a') as f:
        for tree in res.trees:
            f.write(str(tree))
            f.write("\n\n")
    # for r, t in zip(res, res.probs):
    #     f.write(str(r) + "\n")
        # t1, t2 = t.split([1, len(t[0])-1], dim=-1)
        # t = torch.cat((t2, t1), dim=-1)
        # t = torch.cat((t, t.new_zeros((1, len(t[0])))))
        # t.masked_fill_(torch.eye(len(t[0])) == 1.0, 1.0)
        # t_list = t.numpy()
        # probs.append(t_list)
    idx += 1

# with open(sys.argv[2] + ".probs", "wb") as o:
#     pickle.dump(probs, o)