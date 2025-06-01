import argparse
import json
import re


def main(args):
    data = open(args.input).read().strip().split("\n")
    f_out = open(args.output, "w")
    ref_paras = open("data/cgec/fcgec.test.para").read().strip().split("\n")
    ref_inputs = [line.split("\t")[1] for line in ref_paras]
    punctuation = re.escape(
        "～`~!@#！￥…（）—【】、；‘：“”，。、《》？%&_=;':/<>,\$^*()[]{}|.?")
    idx = 0
    for data_line in data:
        src = data_line.split("\t")[1]
        tgt = data_line.split("\t")[2].replace(" ", "")
        if src == "遍布亚洲的偷猎行为使得野生虎的数量急剧减少，将来老虎能否在大自然中继续生存取决于人类的实际行动。":
            new_src = "遍布亚洲的偷猎行为使得野生虎的数量急剧减少将来老虎能否在大自然中继续生存取决于人类的实际"
        else:
            new_src = re.sub(f"[{punctuation}]", "", src)
        if new_src in ref_inputs:
            assert new_src == ref_inputs[0]
            del ref_inputs[0]
            idx += 1
            new_tgt = re.sub(f"[{punctuation}]", "", tgt)
            print(idx, new_src, new_tgt, sep="\t", file=f_out)
    assert idx == 1654


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input para path", required=True)
    parser.add_argument("--output", help="output para path", required=True)
    args = parser.parse_args()
    main(args)
