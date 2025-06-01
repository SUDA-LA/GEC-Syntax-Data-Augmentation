import copy
import random
from copy import deepcopy
from string import punctuation

chinese_punct = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏"
english_punct = punctuation
punct = chinese_punct + english_punct

import os
from tqdm import tqdm
from opencc import OpenCC
import sys
import difflib
import re
import argparse
from multiprocessing import Pool

# CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]  # 当前目录
# config_path = CURRENT_DIR.rsplit('/', 2)[0]  # 上三级目录

sys.path.append("../../")

from utils.ChERRANT.modules.annotator import Annotator
from utils.ChERRANT.modules.tokenizer import Tokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

annotator, sentence_to_tokenized = None, None
cc = OpenCC("t2s")


class M2Processor():
    def __init__(self, src_sent, edit_lines):
        self.src_sent = src_sent
        self.edit_lines = edit_lines
        self.edits = {}
        self.trg_sents = []

    def conv_edit(self, line):
        line = line.strip().split("|||")
        edit_span = line[0].split(" ")
        edit_span = (int(edit_span[0]), int(edit_span[1]))
        edit_res = line[2]
        editor = line[-1]
        if edit_span[0] == -1:
            return None
        if edit_span[0] == edit_span[1]:
            edit_tag = "ADD"
        elif edit_res == "-NONE-" or edit_res == "":
            edit_tag = "DEL"
        else:
            edit_tag = "REP"
        return editor, edit_tag, edit_span, edit_res

    def get_edits(self):
        for line in self.edit_lines:
            if line:
                edit_item = self.conv_edit(line)
                if not edit_item:
                    continue
                editor, edit_tag, edit_span, edit_res = edit_item
                if editor not in self.edits:
                    self.edits[editor] = []
                self.edits[editor].append({"span": edit_span, "op": edit_tag, "res": edit_res})

    def get_para(self):
        self.get_edits()
        if self.edits:
            for editor in self.edits:
                sent = self.src_sent.split(" ")
                for edit_item in self.edits[editor]:
                    edit_span, edit_tag, trg_tokens = edit_item["span"], edit_item["op"], edit_item["res"]
                    if edit_tag == "DEL":
                        sent[edit_span[0]:edit_span[1]] = [" " for _ in range(edit_span[1] - edit_span[0])]
                    else:
                        if edit_tag == "ADD":
                            if edit_span[0] != 0:
                                sent[edit_span[0] - 1] += " " + trg_tokens
                            else:
                                sent[edit_span[0]] = trg_tokens + " " + sent[edit_span[0]]
                        elif edit_tag == "REP":
                            src_tokens_len = len(sent[edit_span[0]:edit_span[1]])
                            sent[edit_span[0]:edit_span[1]] = [trg_tokens] + [" " for _ in range(src_tokens_len - 1)]
                sent = " ".join(sent).strip()
                res_sent = re.sub(" +", " ", sent)
                self.trg_sents.append(res_sent)
            return self.trg_sents
        else:
            return [self.src_sent]


def align(src, tgts):
    d = difflib.Differ()
    tgts_process = []
    for tgt in tgts:
        l = list(d.compare(src, tgt))
        tgt = []
        for i in l:
            if i[0] == ' ':
                tgt.append([i[-1], 0])
            elif i[0] == '-':
                tgt.append([i[-1], 1])
            elif i[0] == '+':
                tgt.append([i[-1], 2])

        # 根据规则特殊显示差异内容
        tgt_process = ""
        dlt = ""
        apd = ""
        for idx, char in enumerate(tgt):
            if char[1] == 1:
                dlt += char[0]
            elif char[1] == 2:
                apd += char[0]
            else:
                if dlt or apd:
                    tgt_process += "["+dlt+"|"+apd+"]"
                    dlt = ""
                    apd = ""
                tgt_process += char[0]
        tgts_process.append(tgt_process)

    return src, tgts_process


def read_m2(m2_data):
    src_sent = None
    edit_lines = []
    fr = m2_data.split("\n")
    result = []
    for line in fr:
        line = line.strip()
        if line.startswith("S "):
            src_sent = line.replace("S ", "", 1)
        elif line.startswith("A "):
            edit_lines.append(line.replace("A ", "", 1))
        elif line == "":
            if edit_lines:
                result.append([src_sent, edit_lines[:]])
                edit_lines.clear()
    return result


def is_verb(idx, seq_dep):
    '''
    返回root id， seg【idx+1】是否为verb
    :param idx:
    :param seq_dep:
    :return:
    '''
    if int(idx) < 0 or int(idx) >= len(seq_dep):
        return [-1, False]
    dep_syntax = seq_dep[int(idx)]
    while dep_syntax[3] in ["conj", "dep"]:
        dep_syntax = seq_dep[int(dep_syntax[2]) - 1]
    if dep_syntax[3] == "root":
        return [int(dep_syntax[0])-1, True]
    return [-1, False]


def read_dep(dep_path, src_data=False):
    seq_dep = []
    with open(dep_path, "r", encoding="utf8") as f1:
        src = [line for line in f1.read().split("\n\n") if line]
        data = [[token.split("\t") for token in line.split("\n") if token] for line in src]

        for seq in data:
            seq_tokens = []
            for token_syn in seq:
                id, src_token, dep_id, syn = token_syn[0], token_syn[1], token_syn[6], token_syn[7]
                seq_tokens.append([id, src_token, dep_id, syn])
            seq_dep.append(seq_tokens)
    if src_data:
        return seq_dep, src
    return seq_dep


def convert_char2seg(char_id, src_word):
    """
    char_id从0开始
    返回其在词级别句子中的下标id
    注意：char级别中 数字按一个token算 如200，需要单独考虑
    :param char_id:
    :param seq_dep:
    :param src:
    :return:
    """
    count = -1   # count 表示前面已过的字数
    seg_id = -1
    for id, token in enumerate(src_word):
        if count < char_id:  # 还没到目标token
            have_digit = False
            seg_id = id
            for char in token:
                if char in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                    have_digit = True
                    continue
                count += 1
            if have_digit:
                count += 1
        else:
            break
    return seg_id


def get_root_token(seq_dep):
    for id, src_token, dep_id, syn in seq_dep:
        if syn == "root":
            return src_token
    return ""


def syntax_rule_search_old(m2_path, dep_path, tgt_path, output_path):
    """
    根据句法规则筛选出数据集中特定的错误
    :param m2_path:
    :param dep_path:
    :param tgt_path:
    :param output_path:
    :return:
    """
    ob_path_src = "../../data/exam_error/ob.src"
    ob_path_tgt = "../../data/exam_error/ob.tgt"

    seqs_dep, srcs_dep = read_dep(dep_path, src_data=True)
    with open(tgt_path, "r", encoding="utf8") as f1:
        tgts = [line for line in f1.read().split("\n") if line]
    with open(m2_path, "r", encoding="utf8") as f1, open(output_path, "w", encoding="utf8") as f2, open(ob_path_src, "w", encoding="utf8") as f3, open(ob_path_tgt, "w", encoding="utf8") as f4:
        m2_data = read_m2(f1.read())
        assert len(srcs_dep) == len(seqs_dep) == len(tgts) == len(m2_data), print(len(srcs_dep), len(seqs_dep), len(tgts), len(m2_data))
        for src_dep, seq_dep, tgt, (src_sent, edit_lines) in zip(srcs_dep, seqs_dep, tgts, m2_data):
            src = src_sent.split(" ")
            seq_seg = [i[1] for i in seq_dep]

            # 指定句法规则，如谓语替换类
            flag = 0
            for edit in edit_lines:
                tgt_content = "".join(edit.split("|||")[2].split(" "))
                src_span = edit.split("|||")[0].split(" ")
                src_content = "".join(src[int(src_span[0]):int(src_span[1])])
                idx, syntax = get_syntax(tgt_content, seq_dep)
                if idx == -1 and len(tgt_content) == 1:
                    idx, syntax = get_tgt_syntax(tgt_content, seq_dep, ["advmod"])
                error_type = edit.split("|||")[1]

                seg_span = [convert_char2seg(int(src_span[0]), seq_seg), convert_char2seg(int(src_span[1]), seq_seg)]
                if seg_span[1] == seg_span[0]:
                    seg_span[1] = seg_span[0] + 1

                # if src_content == "保障":
                #     print(src)
                #     print(edit)
                #     print(src_span)
                #     print(seq_seg)
                #     print(seg_span)
                #     exit(0)

                #替换类错误，且替换词是谓语,且长度相同（仅为1， 2， 4）
                if error_type == "S" and len(src_content) == len(tgt_content) and len(src_content) in [1, 2, 4]:
                # if error_type == "S":
                    for seg_id in range(seg_span[0], seg_span[1]):
                        idx, is_syn = is_verb(seg_id, seq_dep)

                        # 修饰语替换
                        # if idx+2 < len(seq_dep) and is_syn and seq_dep[idx+1][3] == "cpm":
                        # 谓语替换
                        if is_syn:
                            # 找到宾语
                            obj = "None"
                            for id, src_token, dep_id, syn in seq_dep[seg_id:]:
                                id = int(id)
                                if syn in ["dobj", "pobj"] and int(dep_id) == seg_id+1:
                                    obj = [src_token]
                                    while seq_dep[id - 2][3] == "nn":
                                        obj.insert(0, seq_dep[id - 2][1])
                                        id -= 1
                                    obj = "".join(obj)

                            f2.write(''.join(src)+"\n"+tgt+"\n")
                            f2.write(src_content + "\t" + tgt_content+"\n")
                            f2.write(seq_dep[seg_id][1]+"\t"+obj+"\t"+"-1"+"\n\n")
                            f3.write(''.join(src)+"\n")
                            f4.write(tgt+"\n")
                            flag = 1
                            break
                if flag:
                    break


def clean_edit(edit_lines, src_sent, seq_dep=[], error_type="M"):
    """
    根据编辑清洗数据

    :param edit_lines:
    :return:
    """
    clean_edit_lines = []
    src = src_sent.split(" ")
    seq_seg = [i[1] for i in seq_dep]
    for edit in edit_lines:
        tgt_content = edit.split("|||")[2]
        tgt_content = ''.join(tgt_content.split(" "))
        src_span = edit.split("|||")[0].split(" ")
        src_content = "".join(src[int(src_span[0]):int(src_span[1])])
        content = src_content + tgt_content
        error_type = edit.split("|||")[1]
        seg_span = [convert_char2seg(int(src_span[0]), seq_seg), convert_char2seg(int(src_span[1]), seq_seg)]

        if seg_span[0] == -1:
            continue
        if seg_span[1] == seg_span[0]:
            seg_span[1] = seg_span[0] + 1

        # # 仅保留谓语替换编辑，其他全删掉
        # # 替换类错误，且替换词是谓语,且长度相同（仅为1， 2， 4）
        # flag = 0
        # if error_type == "S" and len(src_content) == len(tgt_content) and len(src_content) in [1, 2, 4] and tgt_content != "-NONE-":
        #     for seg_id in range(seg_span[0], seg_span[1]):
        #         idx, is_syn = is_verb(seg_id, seq_dep)
        #         # 谓语替换
        #         if is_syn:
        #             flag = 1
        # if flag:
        #     continue

        # # 仅保留W类编辑
        # if error_type == "W":
        #     continue

        # 仅保留M类编辑
        if error_type == error_type:
            continue

        # # M类编辑筛选
        # if error_type == "M":
        #     # if seq_dep[seg_span[0]][3] in ["dobj", "pobj", "lobj", "attr"] and is_verb(dep_id-1, seq_dep)[1]:
        #     continue
        #     # if seq_dep[seg_span[0]][3] in ["dobj", "pobj", "lobj", "attr", "cpm", "dep"]:
        #     #     continue

        # # 英文筛掉， #筛掉, 引号筛掉， ;；标点转句号删掉
        # if re.findall("[a-zA-Z#]{1,}", content) or '#' in content:
        #     if (not re.findall("[a-zA-Z#]{1,}", src_content)) and tgt_content == "-NONE-":
        #         clean_edit_lines.append(edit)
        #     continue
        # if src_content and src_content in ";；" and tgt_content == "。":
        #     continue
        # if src_content and (src_content in "\"“”" or re.findall("[”“\"]{1,}", content)):
        #     continue
        clean_edit_lines.append(edit)
        # if clean_edit_lines:
        #     print(clean_edit_lines)
    return clean_edit_lines


def annotate(line, segmented=False, no_simplified=False):
    """
    :param no_simplified:
    :param segmented:
    :param line:
    :return:
    """
    sent_list = line.split("\t")[1:]
    source = sent_list[0]
    if segmented:
        source = source.strip()
    else:
        source = "".join(source.strip().split())
        # source = segment(source.strip())
    output_str = ""
    for idx, target in enumerate(sent_list[1:]):
        try:
            if segmented:
                target = target.strip()
            else:
                target = "".join(target.strip().split())
                # target = segment(target.strip())
            if not no_simplified:
                target = cc.convert(target)
            source_tokenized, target_tokenized = sentence_to_tokenized[source], sentence_to_tokenized[target]
            out, cors = annotator(source_tokenized, target_tokenized, idx)
            if idx == 0:
                output_str += "".join(out[:-1])
            else:
                output_str += "".join(out[1:-1])
        except Exception:
            print("|||", line)
            raise Exception
    return output_str


def para2m2(srcs, tgts, batch_size, path, segmented=False, no_simplified=False):
    device = "0"

    tokenizer = Tokenizer("char", device, segmented)
    global annotator, sentence_to_tokenized
    annotator = Annotator.create_default("char", "first")

    lines = []
    for idx, [src, tgt] in enumerate(zip(srcs,tgts)):
        if abs(len(src) - len(tgt)) > len(src)/2:
            tgt = src
        lines.append(str(idx) + '\t' + src + '\t' + tgt)

    print("预处理")
    count = 0
    sentence_set = set()
    sentence_to_tokenized = {}

    for line in lines:
        sent_list = line.split("\t")[1:]
        for idx, sent in enumerate(sent_list):
            if segmented:
                sent = sent.strip()
            else:
                sent = "".join(sent.split()).strip()
                # sent = segment(sent.strip())
            if idx >= 1:
                if not no_simplified:
                    sentence_set.add(cc.convert(sent))
                else:
                    sentence_set.add(sent)
            else:
                sentence_set.add(sent)
    batch = []
    for sent in tqdm(sentence_set):
        count += 1
        if sent:
            batch.append(sent)
        if count % batch_size == 0:
            results = tokenizer(batch)
            for s, r in zip(batch, results):
                sentence_to_tokenized[s] = r  # Get tokenization map.
            batch = []
    if batch:
        results = tokenizer(batch)
        for s, r in zip(batch, results):
            sentence_to_tokenized[s] = r  # Get tokenization map.

    print("m2转换")
    result = ""
    # 单进程模式
    # with open(path, "w", encoding="utf-8") as f:
    #     for line in tqdm(lines):
    #         ret = annotate(line, segmented, no_simplified)
    #         # result += ret + "\n"
    #         if ret:
    #             f.write(ret)
    #             f.write("\n")
    #             result += ret+"\n"
    #     print("转换完成")
    # return result

    # 多进程模式：仅在Linux环境下测试，建议在linux服务器上使用
    with open(path, "w", encoding="utf-8") as f:
        with Pool(20) as pool:
            for ret in pool.imap(annotate, tqdm(lines), chunksize=8):
                if ret:
                    f.write(ret)
                    f.write("\n")
                    result += ret + "\n"
        print("转换完成")
    return result


def m22para(result_m2_data):
    srcs, tgts = [], []
    for src_sent, edit_lines in result_m2_data:
        # edit_lines = clean_edit(edit_lines, src_sent)
        m2_item = M2Processor(src_sent, edit_lines)
        trg_sents = m2_item.get_para()
        srcs.append(src_sent)
        tgts.append(trg_sents[0])
    return srcs, tgts


def m22para_seq(m2_data):
    src_sent, edit_lines = m2_data
    m2_item = M2Processor(src_sent, edit_lines)
    trg_sents = m2_item.get_para()
    return ''.join(src_sent.split(" ")), tgt_sents[0]


def postprocess(src_path, predict_path, processed_path, dep_path):
    seqs_dep, srcs_dep = read_dep(dep_path, src_data=True)
    with open(predict_path, "r", encoding="utf-8") as f1:
        # 将所有繁体转为简体 cc.convert(line)
        tgts = [cc.convert(line.strip()) for line in f1.read().split("\n")]
        with open(src_path, "r", encoding="utf-8") as f2:
            srcs = [line.strip().replace(" ", "") for line in f2.read().split("\n") if line]
            for idx, [src, tgt] in enumerate(zip(srcs, tgts)):
                if not tgt:
                    tgts[idx] = src
                if abs(len(src)-len(tgt)) > len(src)/2:
                    tgts[idx] = src

            # 转为m2格式并抽取编辑
            # m2_data = open("temp.txt", "r", encoding="utf8").read()
            m2_data = para2m2(srcs, tgts, batch_size=2, path="temp.txt", segmented=False)
            m2_data = read_m2(m2_data)
            # 对编辑操作做后处理
            result_m2_data = []

            for seq_dep, (src, edit_lines) in zip(seqs_dep, m2_data):
                edit_lines = clean_edit(edit_lines, src, seq_dep)
                result_m2_data.append([src, edit_lines])
            srcs, tgts = m22para(result_m2_data)

            # 后处理结果写入文件
            with open(processed_path, "w", encoding="utf-8") as f3:
                for tgt in tgts:
                    # 去除BPE分词符
                    f3.write("".join(tgt.split(" ")).replace("##", "")+"\n")
    return srcs, tgts


def predict2m2(srcs_models, tgts_models, batch_size):
    save_path = "predict"
    count = 1
    for srcs, tgts in zip(srcs_models, tgts_models):
        length = len(srcs)
        # 获取m2格式的预测结果，按模型分

        path = save_path + str(count) + ".m2"
        count += 1
        if os.path.exists(path):
            print(path)
            continue
        print(path, len(srcs), len(tgts))
        para2m2(srcs, tgts, batch_size, path, segmented=False)


def read_m2(m2_data):
    src_sent = None
    edit_lines = []
    fr = m2_data.split("\n")
    result = []
    for line in fr:
        line = line.strip()
        if line.startswith("S "):
            src_sent = line.replace("S ", "", 1)
        elif line.startswith("A "):
            edit_lines.append(line.replace("A ", "", 1))
        elif line == "":
            if edit_lines:
                result.append([src_sent, edit_lines[:]])
                edit_lines.clear()
    return result


def read_dep(dep_path, src_data=False):
    seq_dep = []
    with open(dep_path, "r", encoding="utf8") as f1:
        src = [line for line in f1.read().split("\n\n") if line]
        data = [[token.split("\t") for token in line.split("\n") if token] for line in src]

        for seq in data:
            seq_tokens = []
            for token_syn in seq:
                id, src_token, dep_id, syn = token_syn[0], token_syn[1], token_syn[6], token_syn[7]
                seq_tokens.append([id, src_token, dep_id, syn])
            seq_dep.append(seq_tokens)
    if src_data:
        return seq_dep, src
    return seq_dep


def get_syntax(tgt_content, seq_dep):
    for id, src_token, dep_id, syn in seq_dep:
        if tgt_content == src_token:
            return int(id), syn
    return -1, "_"


def get_tgt_syntax(tgt_content, seq_dep, tgt_syntaxs):
    for id, src_token, dep_id, syn in seq_dep:
        if tgt_content in src_token and syn in tgt_syntaxs:
            return int(id), syn
    return -1, "_"


def edit_extract(m2_path, dep_path, tgt_path):
    seqs_errors = []

    seqs_dep = read_dep(dep_path)
    with open(tgt_path, "r", encoding="utf8") as f1:
        tgts = [line for line in f1.read().split("\n") if line]
    with open(m2_path, "r", encoding="utf8") as f1:
        for seq_dep, tgt, (src_sent, edit_lines) in zip(seqs_dep, tgts, read_m2(f1.read())):
            errors = []
            for edit in edit_lines:
                src = src_sent.split(" ")
                tgt_content = edit.split("|||")[2]
                src_span = edit.split("|||")[0].split(" ")
                src_content = "".join(src[int(src_span[0]):int(src_span[1])])
                idx, syntax = get_syntax(tgt_content, seq_dep)
                error_type = edit.split("|||")[1]

                if error_type == "R" and tgt_content == "-NONE-":
                    src.insert(int(src_span[0]), "【")
                    src.insert(int(src_span[1]) + 1, "】")
                    errors.append([error_type, src_content, syntax, "_"])
                elif error_type == "M":
                    src.insert(int(src_span[0]), "】")
                    src.insert(int(src_span[1]), "【")
                    errors.append([error_type, "_", "_", tgt_content])
                elif error_type == "S":
                    src.insert(int(src_span[0]), "【")
                    src.insert(int(src_span[1]) + 1, "】")
                    errors.append([error_type, src_content, syntax, tgt_content])
                elif error_type == "W":
                    src.insert(int(src_span[0]), "【")
                    src.insert(int(src_span[1]) + 1, "】")
                    errors.append([error_type, src_content, syntax, tgt_content])
            seqs_errors.append(["".join(src), tgt, seq_dep, errors])

    return seqs_errors


def is_verb(idx, seq_dep):
    '''
    返回root id， seg【idx+1】是否为verb
    :param idx:
    :param seq_dep:
    :return:
    '''
    if int(idx) < 0 or int(idx) >= len(seq_dep):
        return [-1, False]
    dep_syntax = seq_dep[int(idx)]

    if dep_syntax[3] not in ["conj", "dep", "root"]:
        return [-1, False]

    while dep_syntax[3] in ["conj", "dep"]:
        dep_syntax = seq_dep[int(dep_syntax[2]) - 1]
    if dep_syntax[3] == "root":
        return [int(dep_syntax[0])-1, True]
    return [-1, False]


def is_mod(idx, seq_dep):
    if int(idx)-1 < 0 or int(idx)-1 >= len(seq_dep):
        return [-1, False]
    dep_syntax = seq_dep[int(idx)]
    if dep_syntax[3] in ["advmod"]:
        return [int(dep_syntax[0])-1, True]
    return [-1, False]


def error_backbone_extract(m2_path, dep_path, tgt_path):
    seqs_errors = []

    seqs_dep = read_dep(dep_path)
    with open(tgt_path, "r", encoding="utf8") as f1:
        tgts = [line for line in f1.read().split("\n") if line]
    with open(m2_path, "r", encoding="utf8") as f1:
        for seq_dep, tgt, (src_sent, edit_lines) in zip(seqs_dep, tgts, read_m2(f1.read())):
            errors_backbone = []
            for edit in edit_lines:
                src = src_sent.split(" ")
                tgt_content = "".join(edit.split("|||")[2].split(" "))
                src_span = edit.split("|||")[0].split(" ")
                src_content = "".join(src[int(src_span[0]):int(src_span[1])])
                idx, syntax = get_syntax(tgt_content, seq_dep)
                if idx == -1 and len(tgt_content) == 1:
                    idx, syntax = get_tgt_syntax(tgt_content, seq_dep, ["root", "conj", "dep"])
                error_type = edit.split("|||")[1]

                # 先定位谓语，然后找到主谓宾结构(就近原则)
                # 如果找不到主语，则寻找并列谓语的主语
                if error_type == "S" and len(src_content)==len(tgt_content) and is_verb(idx, seq_dep)[1]:
                    subj, d_subj = "", len(seq_dep)
                    obj, d_obj = "", len(seq_dep)
                    root_subj, d_root_subj = "", len(seq_dep)

                    root_id = is_verb(idx, seq_dep)[0]
                    for id, src_token, dep_id, syn in seq_dep:
                        id = int(id)
                        d = abs(id-int(dep_id))

                        # if int(dep_id) == int(idx) and "subj" in syn and d < d_subj and int(id) < int(dep_id):
                        #     # 找前面的nn
                        #     subj = [src_token]
                        #     while seq_dep[id-2][3] == "nn":
                        #         subj.insert(0, seq_dep[id-2][1])
                        #         id -= 1
                        #     subj = "".join(subj)
                        #     d_subj = d

                        if int(dep_id) == int(idx) and "obj" in syn and d < d_obj and int(id) > int(dep_id):
                            # 找前面的nn
                            obj = [src_token]
                            while seq_dep[id - 2][3] == "nn":
                                obj.insert(0, seq_dep[id - 2][1])
                                id -= 1
                            obj = "".join(obj)
                            d_obj = d

                        # # 找并列谓语的主语
                        # if int(dep_id) == int(root_id) and "subj" in syn and d < d_root_subj:
                        #     # 找前面的nn
                        #     root_subj = [src_token]
                        #     while seq_dep[id-2][3] == "nn":
                        #         root_subj.insert(0, seq_dep[id-2][1])
                        #         id -= 1
                        #     root_subj = "".join(root_subj)
                        #     d_root_subj = d

                    if subj != "" and obj != "":
                        errors_backbone.append(["", seq_dep[idx-1][1], obj, src_content, tgt_content])
                    elif subj == "" and obj != "":
                        errors_backbone.append(["", seq_dep[idx-1][1], obj, src_content, tgt_content])
            seqs_errors.append(["".join(src), tgt, seq_dep, copy.deepcopy(errors_backbone)])
    return seqs_errors


def dataset_analyse(m2_path, dep_path, tgt_path, out_path, k=1):
    '''
    根据错误类型和句法信息统计分析数据集
    根据频率找出高频错误（k为阈值）
    输出成便于观察的形式,按编辑类型划分,输出原句子,目标句,m2编辑,以及句法分析信息
    :return:
    '''
    seqs_dep, srcs_dep = read_dep(dep_path, src_data=True)
    with open(tgt_path, "r", encoding="utf8") as f1:
        tgts = [line for line in f1.read().split("\n") if line]

    seqs_info_s = []
    seqs_info_m = []
    seqs_info_r = []
    seqs_info_w = []

    # 字典统计编辑频率
    edit_dct = dict()
    cnt=0

    with open(m2_path, "r", encoding="utf8") as f1:
        for src_dep, seq_dep, tgt, (src_sent, edit_lines) in zip(srcs_dep, seqs_dep, tgts, read_m2(f1.read())):
            src = src_sent.split(" ")
            seq_seg = [tmp[1] for tmp in seq_dep]

            for edit in edit_lines:
                error_type = edit.split("|||")[1]
                tgt_content = "".join(edit.split("|||")[2].split(" "))
                src_span = edit.split("|||")[0].split(" ")
                src_content = "".join(src[int(src_span[0]):int(src_span[1])])

                seg_span = [convert_char2seg(int(src_span[0]), seq_seg), convert_char2seg(int(src_span[1]), seq_seg)]
                if seg_span[1] == seg_span[0]:
                    seg_span[1] = seg_span[0] + 1

                # # 替换类错误需要完整token替换
                # tgt_content_token = ""
                # src_content_token = src_content
                # for token in seq_seg[seg_span[0]:seg_span[1]]:
                #     tgt_content_token += token
                #
                # if tgt_content_token != tgt_content:
                #     src_content_token = tgt_content_token.replace(tgt_content, src_content)
                #
                # if src_content_token == tgt_content_token:
                #     continue
                #
                # edit_dct[(src_content_token, tgt_content_token, error_type)] = edit_dct.setdefault((src_content_token, tgt_content_token, error_type), []) + [["".join(src), tgt, edit, src_dep]]

                # 其他错误只需要编辑内容即可
                edit_dct[(src_content, tgt_content, error_type)] = edit_dct.setdefault(
                    (src_content, tgt_content, error_type), []) + [["".join(src), tgt, edit, src_dep]]
        for key, value in edit_dct.items():
            # 筛选出现次数超过k的错误
            if len(value) >= k:
                src_content, tgt_content, error_type = key

                # value = random.sample(value, int(len(value)/4)+1)

                for src, tgt, edit, src_dep in value:
                    if error_type == "S":
                        if len(src_content) == len(tgt_content):
                            seqs_info_s.append([src, tgt, src_content+"\t"+tgt_content+"\t"+error_type])
                    elif error_type == "M":
                        seqs_info_m.append([src, tgt, src_content+"\t"+tgt_content+"\t"+error_type])
                    elif error_type == "R":
                        seqs_info_r.append([src, tgt, src_content+"\t"+tgt_content+"\t"+error_type])
                    elif error_type == "W":
                        if len(src_content) <= 15:
                            seqs_info_w.append([src, tgt, src_content+"\t"+tgt_content+"\t"+error_type])
                            cnt+=1
                            break
    print(cnt)

    with open(out_path+"_S", "w", encoding="utf8") as fo:
        for seq_info_s in seqs_info_s:
            for line in seq_info_s:
                fo.write(line.strip()+"\n")
            fo.write("\n")
    with open(out_path+"_M", "w", encoding="utf8") as fo:
        for seq_info_m in seqs_info_m:
            for line in seq_info_m:
                fo.write(line.strip()+"\n")
            fo.write("\n")
    with open(out_path+"_R", "w", encoding="utf8") as fo:
        for seq_info_r in seqs_info_r:
            for line in seq_info_r:
                fo.write(line.strip()+"\n")
            fo.write("\n")
    with open(out_path+"_W", "w", encoding="utf8") as fo:
        for seq_info_w in seqs_info_w:
            for line in seq_info_w:
                fo.write(line.strip()+"\n")
            fo.write("\n")


def convert_char2seg(char_id, src_word):
    """
    char_id从0开始
    返回其在词级别句子中的下标id
    注意：char级别中 数字按一个token算 如200，需要单独考虑
    :param char_id:
    :param seq_dep:
    :param src:
    :return:
    """
    count = -1   # count 表示前面已过的字数
    seg_id = -1
    for id, token in enumerate(src_word):
        if count < char_id:  # 还没到目标token
            have_digit = False
            seg_id = id
            for char in token:
                if char in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                    have_digit = True
                    continue
                count += 1
            if have_digit:
                count += 1
        else:
            break
    return seg_id


def get_root_token(seq_dep):
    for id, src_token, dep_id, syn in seq_dep:
        if syn == "root":
            return src_token
    return ""


def nopunct(content):
    for char in content:
        if char in punct:
            return False
    return True


def syntax_rule_search(m2_path, dep_path, tgt_path, output_path,para_path=None):
    """
    根据句法规则筛选出数据集中特定的错误
    :param m2_path:
    :param dep_path:
    :param tgt_path:
    :param output_path:
    :return:
    """
    ob_path_src = "../../data/exam_error/ob.src"
    ob_path_tgt = "../../data/exam_error/ob.tgt"

    seqs_dep, srcs_dep = read_dep(dep_path, src_data=True)
    f_out=open("w.hcjiang.v2.train.prog.clean","w")
    with open(tgt_path, "r", encoding="utf8") as f1:
        tgts = [line for line in f1.read().split("\n") if line]
    with open(m2_path, "r", encoding="utf8") as f1, open(output_path, "w", encoding="utf8") as f2, open(ob_path_src, "w", encoding="utf8") as f3, open(ob_path_tgt, "w", encoding="utf8") as f4:
        data_para=open(para_path).read().strip().split("\n")
        m2_data = read_m2(f1.read())
        assert len(srcs_dep) == len(seqs_dep) == len(tgts) == len(m2_data)==len(data_para), print(len(srcs_dep), len(seqs_dep), len(tgts), len(m2_data))
        for cur_para,src_dep, seq_dep, tgt, (src_sent, edit_lines) in zip(data_para,srcs_dep, seqs_dep, tgts, m2_data):
            src = src_sent.split(" ")
            seq_seg = [i[1] for i in seq_dep]

            # 编辑太多，则对句子修改幅度大，多为大规模词序类替换，会有bug暂不考虑
            if len(edit_lines) >= 4:
                continue
            # 提取各数据
            exit_flag = 0
            prep_flag = 0
            nsubj_flag = 0
            for edit in edit_lines:
                tgt_content = "".join(edit.split("|||")[2].split(" "))
                src_span = edit.split("|||")[0].split(" ")
                src_content = "".join(src[int(src_span[0]):int(src_span[1])])
                error_type = edit.split("|||")[1]

                seg_span = [convert_char2seg(int(src_span[0]), seq_seg), convert_char2seg(int(src_span[1]), seq_seg)]
                if seg_span[1] == seg_span[0]:
                    seg_span[1] = seg_span[0] + 1


                # if src_content == "保障":
                #     print(src)
                #     print(edit)
                #     print(src_span)
                #     print(seq_seg)
                #     print(seg_span)
                #     exit(0)

                # # 替换类错误，且替换词长度相同（仅为1， 2， 4）
                # if error_type == "S" and len(src_content) == len(tgt_content) and len(src_content) in [1, 2, 4]:
                # # if error_type == "S":
                #     for seg_id in range(seg_span[0], seg_span[1]):
                #         # 1. 谓语替换
                #         idx, is_syn = is_verb(seg_id, seq_dep)
                #         if is_syn:
                #             # 找到宾语
                #             obj = "None"
                #             for id, src_token, dep_id, syn in seq_dep[seg_id:]:
                #                 id = int(id)
                #                 if syn in ["dobj", "pobj"] and int(dep_id) == seg_id+1:
                #                     obj = [src_token]
                #                     while seq_dep[id - 2][3] == "nn":
                #                         obj.insert(0, seq_dep[id - 2][1])
                #                         id -= 1
                #                     obj = "".join(obj)
                #
                #             # 没找到宾语，或谓语长度为1，或替换内容为标点 筛掉
                #             if obj == "None" or len(seq_dep[seg_id][1]) == 1 or not nopunct(src_content+tgt_content):
                #                 break
                #
                #             # 写入文件
                #             f2.write("".join(src)+"\n"+tgt+"\n"+" ".join(seq_seg)+"\n")
                #             # f2.write("".join(src) + "\n" + tgt + "\n")
                #             f2.write(src_content + "\t" + tgt_content+"\n")
                #             # f2.write(seq_dep[seg_id][1]+"\t"+obj+"\t"+str(seg_id)+"\n\n")
                #             f3.write(''.join(src)+"\n")
                #             f4.write(tgt+"\n")
                #             exit_flag = 1
                #             break
                #
                #         # # 2. 修饰语替换
                #         # idx, is_syn = is_mod(seg_id, seq_dep)
                #         # if idx+2 < len(seq_dep) and is_syn and seq_dep[idx+1][3] == "cpm":
                #         #     obj = seq_dep[idx+2][1]
                #         #
                #         #     f2.write("".join(src)+"\n"+tgt+"\n"+" ".join(seq_seg)+"\n")
                #         #     f2.write(src_content + "\t" + tgt_content+"\n")
                #         #     f2.write(seq_dep[seg_id][1]+"\t"+obj+"\t"+str(seg_id)+"\n\n")
                #         #     f3.write(''.join(src)+"\n")
                #         #     f4.write(tgt+"\n")
                #         #     exit_flag = 1
                #         #     break
                #
                #         # # 3.
                #         # if "mod" in seq_dep[seg_id][3]:
                #         #     # 找到宾语
                #         #     obj = "None"
                #         #     for id, src_token, dep_id, syn in seq_dep[seg_id:]:
                #         #         id = int(id)
                #         #         if syn in ["dobj", "pobj"] and int(dep_id) == seg_id+1:
                #         #             obj = [src_token]
                #         #             while seq_dep[id - 2][3] == "nn":
                #         #                 obj.insert(0, seq_dep[id - 2][1])
                #         #                 id -= 1
                #         #             obj = "".join(obj)
                #         #
                #         #     # 没找到宾语，或修饰语长度为1，或替换内容为标点 筛掉
                #         #     if obj == "None" or len(seq_dep[seg_id][1]) == 1 or not nopunct(src_content+tgt_content):
                #         #         break
                #         #
                #         #     f2.write("".join(src)+"\n"+tgt+"\n"+" ".join(seq_seg)+"\n")
                #         #     # f2.write("".join(src) + "\n" + tgt + "\n")
                #         #     f2.write(src_content + "\t" + tgt_content+"\n")
                #         #     f2.write(seq_dep[seg_id][1]+"\t"+obj+"\t"+str(seg_id)+"\n\n")
                #         #     # f2.write(src_dep+"\n\n")
                #         #     exit_flag = 1
                #         #     break

                # 词序类错误例子搜索
                if error_type == "W":
                    # if seg_span[1]-seg_span[0] > 10:
                    #     continue
                    # for seg_id in range(seg_span[0], seg_span[1]):
                    #     if seq_dep[seg_id][1] in ["、"]:
                    #         prep_flag = 1
                    #         nsubj_flag = 1
                    #     # if seq_dep[seg_id][3] in ["advmod"]:
                    #     #     prep_flag = 1
                    #     # if seq_dep[seg_id][3] in ["nsubj"]:
                    #     #     nsubj_flag = 1
                    # if prep_flag and nsubj_flag:
                    syntax_seq = ' '.join([i[3] for i in seq_dep])

                    # conj/obj + cc/"、" + conj + cc/"、" + conj/advmod rcmod

                    # pattern = "conj cc conj cc conj rcmod"
                    pattern = "cc"
                    # print(syntax_seq)
                    # print(pattern)
                    syntax_seq_edit=' '.join([i[3] for i in seq_dep[seg_span[0]: seg_span[1]]])
                    if "cc" in syntax_seq_edit or "punct" in syntax_seq_edit:
                        print(cur_para,file=f_out)
                        f2.write("".join(src).replace("#", "") + "\t" + tgt + "\n")
                        # print("".join(src).replace("#", "") + "\t" + tgt + "\n")
                        # f2.write("".join(src).replace("#", "") + "\t" + tgt + "\n")
                        # f2.write(src_content + "\t" + tgt_content + "\n\n")
                        # f2.write(seq_dep[seg_id][1] + "\t" + obj + "\t" + str(seg_id) + "\n")
                        # f2.write(src_dep+"\n\n")
                        # f3.write(''.join(src) + "\n")
                        # f4.write(tgt + "\n")
                        exit_flag = 1
                        break

                # if error_type == "W":
                #     f2.write("".join(src).replace("#", "") + "\t" + tgt + "\n")
                #     break

                # # 缺失类
                # if error_type == "M":
                #     for seg_id in range(seg_span[0], seg_span[1]):
                #         if seq_dep[seg_id][3] in ["dobj", "pobj", "lobj", "attr"] and is_verb(int(seq_dep[seg_id][2]) - 1, seq_dep)[1]:
                #             f2.write("".join(src).replace("#", "") + "\t" + tgt + "\n")
                #             # f2.write(edit + "\n\n")
                #             # f2.write(src_dep+"\n\n")
                #             # f2.write(seq_dep[seg_id][1] + "\t" + obj + "\t" + str(seg_id) + "\n\n")
                #             f3.write(''.join(src) + "\n")
                #             f4.write(tgt + "\n")
                #             exit_flag = 1
                #             break

                # # 冗余类
                # if error_type == "R":
                #     f2.write("".join(src).replace("#", "") + "\t" + tgt + "\n")
                #     # f2.write(src_content + "\t" + tgt_content + "\n\n")
                #     break

                if exit_flag:
                    break


def m2togpt(m2_path, tgt_path, dep_path, gpt_input):
    seqs_dep, srcs_dep = read_dep(dep_path, src_data=True)
    with open(m2_path, "r", encoding="utf8") as f1, open(tgt_path, "r", encoding="utf8") as f2, open(gpt_input, "w", encoding="utf8") as f3:
        m2_data = read_m2(f1.read())
        tgts = [line for line in f2.read().split("\n") if line]
        assert len(srcs_dep) == len(seqs_dep) == len(tgts) == len(m2_data), print(len(srcs_dep), len(seqs_dep), len(tgts), len(m2_data))
        for src_dep, seq_dep, tgt, (src_sent, edit_lines) in zip(srcs_dep, seqs_dep, tgts, m2_data):
            src = src_sent.split(" ")
            seq_seg = [i[1] for i in seq_dep]

            # 编辑太多，则对句子修改幅度大，多为大规模词序类替换，会有bug暂不考虑
            if len(edit_lines) >= 4:
                continue
            f3.write(''.join(src) + "\n" + tgt + "\n")

            for edit in edit_lines:
                tgt_content = "".join(edit.split("|||")[2].split(" "))
                src_span = edit.split("|||")[0].split(" ")
                src_content = "".join(src[int(src_span[0]):int(src_span[1])])
                error_type = edit.split("|||")[1]

                seg_span = [convert_char2seg(int(src_span[0]), seq_seg), convert_char2seg(int(src_span[1]), seq_seg)]
                if seg_span[1] == seg_span[0]:
                    seg_span[1] = seg_span[0] + 1

                # 将单字的替换编辑，改为词级别
                if error_type == "S" and len(src_content) == len(tgt_content) == 1:
                    S_token = ''.join(seq_seg[seg_span[0]:seg_span[1]])

                    if src_content == "举" and tgt_content == "进":
                        print(seq_seg)
                        print(S_token)

                    if len(S_token) <= 1:
                        pass
                    else:
                        if S_token[0] == tgt_content:
                            src_content = src_content + ''.join(S_token[1:])
                            tgt_content = ''.join(S_token)
                        elif S_token[-1] == tgt_content:
                            src_content = ''.join(S_token[:-1]) + src_content
                            tgt_content = ''.join(S_token)

                f3.write(error_type+"\t"+src_content+"\t"+tgt_content+"\n")
            f3.write("\n")


def m2togpt_type(m2_path, tgt_path, dep_path, gpt_input, type="R"):
    seqs_dep, srcs_dep = read_dep(dep_path, src_data=True)
    with open(m2_path, "r", encoding="utf8") as f1, open(tgt_path, "r", encoding="utf8") as f2, open(gpt_input, "w",
                                                                                                     encoding="utf8") as f3:
        m2_data = read_m2(f1.read())
        tgts = [line for line in f2.read().split("\n") if line]
        assert len(srcs_dep) == len(seqs_dep) == len(tgts) == len(m2_data), print(len(srcs_dep), len(seqs_dep),
                                                                                  len(tgts), len(m2_data))
        for src_dep, seq_dep, tgt, (src_sent, edit_lines) in zip(srcs_dep, seqs_dep, tgts, m2_data):
            output = []
            src = src_sent.split(" ")
            seq_seg = [i[1] for i in seq_dep]

            output.append(''.join(src) + "\n" + tgt + "\n")

            # 首先清洗掉无关编辑
            edit_lines = clean_edit(edit_lines, src, seq_dep, error_type=type)
            clean_src, clean_tgt = m22para_seq([src, edit_lines])
            output.append(clean_tgt + "\n" + tgt + "\n")
            
            # 仅保留指定类型编辑，输出内容
            for edit in edit_lines:
                tgt_content = "".join(edit.split("|||")[2].split(" "))
                src_span = edit.split("|||")[0].split(" ")
                src_content = "".join(src[int(src_span[0]):int(src_span[1])])
                error_type = edit.split("|||")[1]

                seg_span = [convert_char2seg(int(src_span[0]), seq_seg), convert_char2seg(int(src_span[1]), seq_seg)]
                if seg_span[1] == seg_span[0]:
                    seg_span[1] = seg_span[0] + 1

                # 将单字的替换编辑，改为词级别
                if error_type == "S" and len(src_content) == len(tgt_content) == 1:
                    S_token = ''.join(seq_seg[seg_span[0]:seg_span[1]])

                    if len(S_token) <= 1:
                        pass
                    else:
                        if S_token[0] == tgt_content:
                            src_content = src_content + ''.join(S_token[1:])
                            tgt_content = ''.join(S_token)
                        elif S_token[-1] == tgt_content:
                            src_content = ''.join(S_token[:-1]) + src_content
                            tgt_content = ''.join(S_token)

                if error_type == type:
                    output.append(error_type + "\t" + src_content + "\t" + tgt_content + "\n")
            if len(output) > 1:
                for line in output:
                    f3.write(line)
            f3.write("\n")



if __name__=="__main__":
    # m2_path = "../data/FCGEC/valid/FCGEC_valid.m2.char"
    # dep_path = "../data/FCGEC/valid/FCGEC_valid.tgt.dep.conll_predict"
    # tgt_path = "../data/FCGEC/valid/FCGEC_valid.tgt"
    # out_path = "../data/FCGEC/valid/FCGEC_valid.W.2"

    # m2_path = "../../data/FCGEC/train/FCGEC_train.1000.para.m2.char"
    # dep_path = "../../data/FCGEC/train/FCGEC_train.1000.tgt.dep.conll_predict"
    # tgt_path = "../../data/FCGEC/train/FCGEC_train.1000.tgt"
    # out_path = "../../data/FCGEC/train/FCGEC_train.1000.R.gpt.input"
    #
    # m2_path = "/data3/hcjiang/data_augment/data/news/news.seq.100w.output.limit.m2.char"
    # dep_path = "/data3/hcjiang/data_augment/data/news/news.seq.100w.tgt.pattern.limit.dep.conll_predict"
    # tgt_path = "/data3/hcjiang/data_augment/data/news/news.seq.100w.tgt.pattern.limit"
    # out_path = "/data3/hcjiang/data_augment/data/news/news.seq.100w.tgt.pattern.limit.gpt.input"

    # m2_path = "../data/exam_error/train/exam.train.m2.char"
    # dep_path = "../data/exam_error/train/tgt.txt.dep.conll_predict"
    # tgt_path = "../data/exam_error/train/tgt.txt"
    # out_path = "../data/exam_error/train/exam.train.R.gpt.input"

    # m2_path = "../../data/news/news.seq.100w.m2.char"
    # dep_path = "../../data/news/news.seq.100w.txt.dep.conll_predict"
    # tgt_path = "../../data/news/news.seq.100w.txt"
    # out_path = "../../data/news/news.seq.100w.txt.read"

    # m2_path = "../../data/FCGEC/FCGEC_train.m2.char"
    # dep_path = "../../data/FCGEC/FCGEC_train.tgt.dep.conll_predict"
    # tgt_path = "../../data/FCGEC/FCGEC_train.tgt"
    # out_path = "../../data/FCGEC/read/FCGEC_train.read"

    m2_path = "w.hcjiang.v2.train.m2"
    dep_path = "w.hcjiang.v2.train.tgt.dep.conll_predict"
    tgt_path = "w.hcjiang.v2.train.tgt"
    out_path = "w.hcjiang.v2.train.read"
    para_path = "w.hcjiang.v2.train.para"

    # m2_path = "/data3/hcjiang/data_augment/data/hkht/train/hkht.4000.v2.train.para.m2.char"
    # dep_path = "/data3/hcjiang/data_augment/data/hkht/train/hkht.4000.v2.train.tgt.dep.conll_predict"
    # tgt_path = "/data3/hcjiang/data_augment/data/hkht/train/hkht.4000.v2.train.tgt"
    # gpt_input = "/data3/hcjiang/data_augment/data/hkht/train/hkht.4000.v2.train.gpt.input"

    syntax_rule_search(m2_path, dep_path, tgt_path, out_path,para_path)
    # dataset_analyse(m2_path, dep_path, tgt_path, out_path, k=5)
    # gpt_input = "../../data/exam_error/train/exam.train.gpt.input"
    # m2togpt(m2_path, tgt_path, dep_path, gpt_input)
    # m2togpt_type(m2_path, tgt_path, dep_path, gpt_input)

    # seqs_backbone_errors = error_backbone_extract(m2_path, dep_path, tgt_path)
    #
    # substitute_path = "../../data/FCGEC/edit.substitute.read"
    # # for src, tgt, seq_dep, errors in seqs_errors:
    # #     for error_type, src_content, syntax, tgt_content in errors:
    # #         if error_type == "S" and syntax in ["root", "conj"]:
    # #             output(src, tgt, errors, seq_dep, substitute_path)
    # #             break
    #
    # count = 0
    # for src, tgt, seq_dep, errors in seqs_backbone_errors:
    #     if errors:
    #         count += 1
    #         output_backbone(src, tgt, errors, seq_dep, substitute_path)
    # print(count)
    #
    #         # f1.write(src_content+"\t"+syntax+"\t"+tgt_content+"\n")
    #         # for i in seq_dep:
    #         #     f1.write("\t".join(i)+"\n")
    #         # f1.write("\n")
