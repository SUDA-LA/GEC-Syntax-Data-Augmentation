import copy
import random
import thulac
from transformers import BertForMaskedLM, BertTokenizer
import torch
import re
import pprint
from numpy.random import choice
from tqdm import tqdm
from copy import deepcopy
from multiprocessing import Pool
from GPU_autochoice import GPUManager
import time

torch.multiprocessing.set_start_method('forkserver', force=True)

def _is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    if len(cp) > 1:
        return False
    cp = ord(cp)
    if ((0x4E00 <= cp <= 0x9FFF) or  #
            (0x3400 <= cp <= 0x4DBF) or  #
            (0x20000 <= cp <= 0x2A6DF) or  #
            (0x2A700 <= cp <= 0x2B73F) or  #
            (0x2B740 <= cp <= 0x2B81F) or  #
            (0x2B820 <= cp <= 0x2CEAF) or
            (0xF900 <= cp <= 0xFAFF) or  #
            (0x2F800 <= cp <= 0x2FA1F)):  #
        return True
    return False


def is_all_chinese(seq_seg):
    for word in seq_seg:
        for ch in word:
            if ch in "．…%—:：，,《》.、。!?;！？；()（）”“0123456789\"":
                continue
            if not _is_chinese_char(ch):
                return False
    return True


def torch_index(tokens, token):
    for idx, i in enumerate(tokens):
        if token == i:
            return idx


def read_src(seg_path):
    with open(seg_path, "r", encoding="utf8") as f1:
        data = [line.strip().split(" ") for line in f1.read().split("\n") if line]
    return data


def read_dep(dep_path):
    seq_dep = []
    with open(dep_path, "r", encoding="utf8") as f1:
        data = [[token.split("\t") for token in line.split("\n") if token] for line in f1.read().split("\n\n") if line]
        for seq in data:
            seq_tokens = []
            for token_syn in seq:
                id, src_token, dep_id, syn = token_syn[0], token_syn[1], token_syn[6], token_syn[7]
                seq_tokens.append([id, src_token, dep_id, syn])
            seq_dep.append(seq_tokens)
    return seq_dep


def read_con(con_path):
    pass


def is_parataxis(idx, seq_dep):
    '''
    从idx下标开始，找、表示的并列关系
    单、 视为并列（大部分都是）
    多、 后有关键词["等", "之类", "或", "和", "以及"]，视为并列
    :param idx:
    :param seq_dep:
    :return:
    '''
    k = idx
    while k < len(seq_dep) and (seq_dep[k][3] != "punct" or seq_dep[k][1] == "、"):
        if seq_dep[k][1] in ["等", "之类", "或", "和", "以及"] or "等" in seq_dep[k][1]:
            return True
        k += 1

    # 同时存在两个、   不是并列关系
    i = idx - 1
    j = idx + 1
    while j < len(seq_dep):
        if seq_dep[j][1] == "、":
            return False
        elif seq_dep[j][3] == "punct":
            break
        j += 1

    while i >= 0:
        if seq_dep[i][1] == "、":
            return False
        elif seq_dep[i][3] == "punct":
            break
        i -= 1

    return True


def search_pre_pattern_test(seq_dep, id, patterns=[]):
    pre_lst = []
    for pattern in patterns:
        length = len(pattern)
        if id-length >= 0:
            value = tuple([i[3] for i in seq_dep[id-length:id]])
            print(value)
            print(pattern)
            print(len(pre_lst)<len(value))
            if value == pattern and len(pre_lst) < len(value):
                pre_lst = [idx for idx in range(id-length, id)]
    return pre_lst


def search_pre_pattern(seq_dep, id, patterns=[]):
    pre_lst = []
    for pattern in patterns:
        length = len(pattern)
        if id-length >= 0:
            value = tuple([i[3] for i in seq_dep[id-length:id]])
            if value == pattern and len(pre_lst) < len(value):
                pre_lst = [idx for idx in range(id-length, id)]
    return pre_lst


def search_suffix_pattern(seq_dep, id, patterns=[]):
    suf_lst = []
    for pattern in patterns:
        length = len(pattern)
        if id+length <= len(seq_dep):
            value = tuple([i[3] for i in seq_dep[id+1:id+length+1]])
            if value == pattern and len(suf_lst) < len(value):
                suf_lst = [idx for idx in range(id+1, id+length+1)]
    return suf_lst


def search_pre_syntax(seq_dep, id, syntaxs=[], end_syntaxs=[], end_tokens=[]):
    """
    找到nsubj，dobj等的前缀，即整个名词性短语
    若syntaxs为None，则匹配从句首开始的所有内容
    遇到end_中的内容，则停止
    无论何种模式，依附id不能超过原token
    :param seq_dep:
    :param id:
    :param syntaxs: 前缀可能包含的句法标签
    :return:
    """
    nsubj_lst = [id - 1]
    # 匹配标签 模式
    if syntaxs and id - 2 >= 0:
        pre_id = id - 2
        while seq_dep[pre_id][3] in syntaxs and int(seq_dep[pre_id][2]) <= id:
            nsubj_lst.insert(0, pre_id)
            if pre_id - 1 >= 0:
                pre_id -= 1
            else:
                break
    # 匹配截止标签 模式
    elif not syntaxs and id - 2 >= 0:
        pre_id = id - 2
        while int(seq_dep[pre_id][2]) <= id and seq_dep[pre_id][3] not in end_syntaxs and seq_dep[pre_id][1] not in end_tokens:
            nsubj_lst.insert(0, pre_id)
            if pre_id - 1 >= 0:
                pre_id -= 1
            else:
                break
    if id < len(seq_dep) and seq_dep[id][1] in ["”", "\"", "》"]:
        nsubj_lst.append(id)
    return nsubj_lst


def is_verb(idx, seq_dep):
    '''
    返回root id， seg【idx】是否为verb
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


def generate_redundant_error(seq_seg, seq_dep, flag, seq_con=None, ob=False):
    '''
    冗余类错误
    规律
        修饰语冗余
        1.rcmod+cpm 前后补充一个修饰语
            近三十年来，广州发生了翻天覆地的变化，成就令人惊喜，但不可否认的是，这种发展是伴随着环境污染，特别是水污染这一惨重代价的。
        ->  近三十年来，广州发生了巨大的翻天覆地的变化，成就令人惊喜，但不可否认的是，这种发展是伴随着环境污染，特别是水污染这一惨重代价的。

        2. nsubj+ccomp+punct nsubj后加一个“的”【偏润色】
            经过几年试行，实践证明，颁发“考试说明”有利于考生复习备考，也有利于实现考试科学化、标准化，减轻不必要的负担。
        ->  经过几年试行，实践证明，颁发“考试说明”有利于考生复习备考，也有利于实现考试的科学化、标准化，减轻不必要的负担。


    :param seq_seg:
    :param seq_dep:
    :param seq_con:
    :return:
    '''
    # 是否加入噪声
    # whether_add_noise = choice(a=[0, 1], p=[0.2, 0.8], size=1, replace=False)[0]
    # if not whether_add_noise:
    #     return seq_seg
    # 噪声数量
    add_noise_num = choice(a=[1, 2, 3], p=[0.8, 0.15, 0.05], size=1, replace=False)[0]

    # 定位目标句法结构
    src_seq = seq_seg[:]
    src_dep = seq_dep[:]
    append_idxs = []
    for token, token_syntax in zip(src_seq, src_dep):
        id, src_token, dep_id, syn = token_syntax
        id, dep_id = int(id), int(dep_id)
        dep_syntax = seq_dep[dep_id - 1]
        assert token == src_token, print(token, src_token)

        # # 1.root+nsubj+ccomp+punct【比例极低，不考虑】
        # if id+1 < len(seq_dep) and syn == "nsubj" and src_dep[id][3] == "ccomp" and src_dep[id+1][3] == "punct" and src_dep[id][1] not in ["是否"]:
        #     append_idxs.append((id, "的"))


    # # 去重
    tmp = []
    clean_append_idxs = []
    for id, content in append_idxs:
        if id not in tmp:
            tmp.append(id)
            clean_append_idxs.append([id, content])

    # # 随机选取 add_noise_num 个噪声进行加噪
    # clean_append_idxs = random.sample(clean_append_idxs, min(add_noise_num, len(clean_append_idxs)))
    clean_append_idxs = sorted(append_idxs, key=lambda x:x[0], reverse=True)

    if len(clean_append_idxs) == 0:
        flag[0] = 0
    else:
        flag[0] = 2

    for id, content in clean_append_idxs:
        seq_seg.insert(id, content)

    if ob:
        return ''.join(src_seq), seq_seg

    return seq_seg


def get_root_token(seq_dep):
    for id, src_token, dep_id, syn in seq_dep:
        if syn == "root":
            return src_token
    return ""


def generate_missing_error(seq_seg, seq_dep, flag, seq_con=None, ob=False):
    '''
    生成缺失类错误
    规则：
        1. punct+(...+nsubj): punct后短距离（3）定位到nsubj时，删除nsubj，形成状语前置的主语缺失类错误
                为防止重复，每句只删除最后一个符合该规律的主语【若主语存在修饰成分，则不做处理（改动范围过大）】
            俄罗斯马林斯基剧院代表团带来的歌剧《伊戈尔王》将成为中国国家大剧院歌剧院的揭幕剧目，同时该剧院还将带来三部经典芭蕾舞剧，连续上演十一天。
        ->  俄罗斯马林斯基剧院代表团带来的歌剧《伊戈尔王》将成为中国国家大剧院歌剧院的揭幕剧目，同时【】还将带来三部经典芭蕾舞剧，连续上演十一天。
        2. 宾语缺失
            root+（）+obj，obj依附于root
            a. obj前存在cpm，一起删除
            b. obj前存在cpm，但中间有其他成分，此时仅删除obj（amod，nn）
            c. obj前不存在cpm（todo）
        3.  双动宾结构，删除后一个谓语
            （root+dobj）+cc+（conj+dobj）
            打车软件为乘客和司机搭建起沟通平台，方便了市民打车，但出租车无论是否使用打车软件，均应遵守运营规则，这才能维护相关各方的合法权益和满足合理要求。
            ->打车软件为乘客和司机搭建起沟通平台，方便了市民打车，但出租车无论是否使用打车软件，均应遵守运营规则，这才能维护相关各方的合法权益和合理要求。
            （root+dobj）+punct+（conj+dobj）
            这次回到故乡，我又看到了那阔别多年的老师，听到了那熟悉的可爱的乡音和那爽朗的笑声。
            ->这次回到故乡，我又看到了那阔别多年的老师，那熟悉的可爱的乡音和那爽朗的笑声。
        
    :param seq_seg:
    :param seq_dep:
    :param seq_con:
    :return:
    '''
    # 噪声数量
    add_noise_num = choice(a=[1, 2, 3], p=[0.8, 0.15, 0.05], size=1, replace=False)[0]

    # 规则数n
    n = 3

    src_seq = seq_seg[:]
    src_dep = seq_dep[:]
    del_idx_split = [[] for _ in range(n)]
    candidate_del_idx1 = []
    candidate_del_idx2 = []
    results = [[] for _ in range(n)]
    # 【id, token, dep_id, syn】
    for token, token_syntax in zip(src_seq, src_dep):
        id, src_token, dep_id, syn = token_syntax
        id, dep_id = int(id), int(dep_id)
        assert token == src_token, print(token, src_token)

        # 寻找符合规律的句法机构
        # 1. punct后定位到nsubj时，删除nsubj
        # 只删除最后一个nsubj
        if src_token in ",，":
            candidate_tokens = src_dep[id:id+3]
            for idx, candidate_token in enumerate(candidate_tokens):
                if candidate_token[3] == "nsubj":
                    # 若nsubj的修饰性成分过长，则不删除（模型预测效果较差）
                    nsubj_lst = search_pre_syntax(seq_dep, int(candidate_token[0]), syntaxs=["nn", "det", "amod"])
                    if len(nsubj_lst) <= 2:
                        del_idx_split[0].append(tuple(nsubj_lst))

        # 2. 宾语缺失
        '''
        root+（）+obj，obj依附于root
            a. obj前存在cpm，一起删除
            b. obj前存在cpm，但中间有其他成分，此时仅删除obj（amod，nn）
            c. obj前不存在cpm，obj前存在nn，则删去obj（可能有噪音,优先级最低）
            优先考虑句末情况（后面有标点）
        '''
        if syn in ["dobj", "pobj", "lobj", "attr"] and is_verb(dep_id-1, seq_dep)[1]: # 找到obj，且依附于verb
            if id<len(seq_dep) and seq_dep[id][3] == "punct":
                if seq_dep[id-2][1] in ["的", "等"] and id - dep_id >= 5:
                    del_idx_split[1].append((id-2, id-1))
                else:
                    have_cpm = False
                    for tmp_idx in range(dep_id, id):
                        if seq_dep[tmp_idx][3] in ["的", "等"] and id - dep_id >= 6:
                            del_idx_split[1].append((id - 1,))
                            have_cpm = True
                            break
                    if not have_cpm:
                        if seq_dep[id - 2][3] in ["nn"]:
                            candidate_del_idx2.append((id - 1,))
            else:
                if seq_dep[id-2][1] in ["的", "等"] and id - dep_id >= 5:
                    candidate_del_idx1.append((id-2, id-1))
                else:
                    have_cpm = False
                    for tmp_idx in range(dep_id, id):
                        if seq_dep[tmp_idx][3] in ["的", "等"] and id - dep_id >= 6:
                            candidate_del_idx1.append((id - 1,))
                            have_cpm = True
                            break
                    if not have_cpm:
                        if seq_dep[id - 2][3] in ["nn"]:
                            candidate_del_idx2.append((id - 1,))

        # # 效果不好，暂时弃置
        # # 4. punct + prep
        # if syn == "punct" and src_token in ",，" and id < len(seq_dep):
        #     if src_dep[id][3] == "prep":
        #         if id+1 < len(seq_dep) and src_dep[id+1][3] == "asp":
        #             del_idx.append((id,id+1))
        #             del_idx_split[3].append((id, id+1))
        #         else:
        #             del_idx.append((id,))
        #             del_idx_split[3].append((id,))

    if len(del_idx_split[1]) == 0:
        if candidate_del_idx1:
            del_idx_split[1] = candidate_del_idx1
        elif candidate_del_idx2:
            del_idx_split[1] = candidate_del_idx2

    # 3. （root+dobj）+cc+（conj+dobj）
    #    （root + dobj）+punct +（conj + dobj）
    # 这条规则单独做
    # 先找出所有有直接宾语的谓语
    dep_verb_idx = []
    for token, token_syntax in zip(seq_seg, seq_dep):
        id, src_token, dep_id, syn = token_syntax
        id, dep_id = int(id), int(dep_id)
        assert token == src_token, print(token, src_token)

        if syn == "dobj":
            conjs = []
            deps = []
            dep_syntax = seq_dep[dep_id - 1]
            while dep_syntax[3] in ["dep", "conj"]:
                if dep_syntax[3] == "conj":
                    conjs.append(int(dep_syntax[0]) - 1)
                elif dep_syntax[3] == "dep":
                    deps.append(int(dep_syntax[0]) - 1)
                dep_syntax = seq_dep[int(dep_syntax[2]) - 1]
            if dep_syntax[3] == "root":
                if conjs:
                    dep_verb_idx.append(conjs[0])
                else:
                    dep_verb_idx.append(int(dep_syntax[0])-1)
                # dep 暂不加入

    dep_verb_idx = list(set(dep_verb_idx))
    # 先筛选出和cc/punct相连的verb2，然后找到cc/punct依附的verb1
    # cc前需要是dobj，pobj，lobj，且依附于verb1
    for verb_idx in dep_verb_idx:
        id, src_token, dep_id, syn = seq_dep[verb_idx - 1]
        id, dep_id = int(id), int(dep_id)
        # 定位verb2
        if syn == "cc" or (syn == "punct" and src_token in ",，"):
            # cc，punct的前一个成分是宾语
            if seq_dep[id-2][3] not in ["dobj", "pobj", "lobj"]:
                continue
            else:
                obj_dep_id = int(seq_dep[id-2][2])

            # 找到cc punct依附的verb1
            dep_syntax = seq_dep[dep_id - 1]
            verb1_idx = int(dep_syntax[0])

            while dep_syntax[3] in ["conj"] and verb1_idx < int(id):
                dep_syntax = seq_dep[int(dep_syntax[2]) - 1]
            if dep_syntax[3] == "root" and verb1_idx == obj_dep_id:
                if verb_idx+1 < len(seq_dep) and seq_dep[verb_idx+1][3] == "asp":
                    del_idx_split[2].append((verb_idx, verb_idx+1))
                else:
                    del_idx_split[2].append((verb_idx,))

    for idx, error in enumerate(del_idx_split):
        # 去重
        error = list(set(error))
        # 控制加噪数
        if len(error) > add_noise_num:
            error = random.sample(error, add_noise_num)
        clean_error = list(set([i for tup in error for i in tup]))
        # 从后往前删除
        clean_error = sorted(clean_error, reverse=True)

        seq_seg = deepcopy(src_seq)
        for id in clean_error:
            del seq_seg[id]
        results[idx] = seq_seg

    return results


def generate_verb_substitute_error(seq_seg, seq_dep, flag, seq_con=None, ob=False):
    '''
    生成替换类错误
    规则：
        谓语替换
        1. dobj/nsubj依附于谓语（可以是root，或conj） 替换谓语 形成动宾搭配不当
            最近又发动了全面的质量大检查运动，要在这个运动中完成建立与完善技术管理制度等一系列的工作。
            ->最近又发动了全面的质量大检查运动，要在这个运动中完成建立与【加强】技术管理制度等一系列的工作。

        宾语修饰语替换
        2. rcmod+cpm+dobj 替换rcmod 造 修饰语使用不当
            人们在会议上爆发了激烈的争论。
        ->  人们在会议上爆发了强烈的争论。
        谓语修饰语替换
        3. dvpmod+（comod）+dvpm+ root/conj conj依附于root 替换dvpmod
            要掌握走私犯的行动规律，以便更好地识别和打击他们。
        ->  要掌握走私犯的行动规律，以便稳准狠地识别和打击他们。

        4. 【测试】root + ... + cpm + （amod+attr） 替换宾语  发展社会主义文艺的【指导思想】
            attr依附于root， 寻找依附于attr的修饰成分一起替换，若较多则不修改

        并列谓语替换
        5. root/conj + cc + dep + ... + dobj  dobj依附于dep，dep依附于root/conj
            随机替换前后任一谓语
    :param seq_seg:
    :param seq_dep:
    :param seq_con:
    :return:
    '''
    # 是否加入噪声
    # whether_add_noise = choice(a=[0, 1], p=[0.2, 0.8], size=1, replace=False)[0]
    # if not whether_add_noise:
    #     return seq_seg
    # 噪声数量
    add_noise_num = choice(a=[1, 2, 3], p=[0.8, 0.15, 0.05], size=1, replace=False)[0]

    src_seq = seq_seg[:]
    src_dep = seq_dep[:]
    masked_tokens = []
    for token, token_syntax in zip(src_seq, src_dep):
        id, src_token, dep_id, syn = token_syntax
        id, dep_id = int(id), int(dep_id)
        assert token == src_token, print(token, src_token)

        # 1. dobj依附于root/conj   conj依附于root
        #todo 句法标签为dep的情况
        if syn in ["dobj", "pobj"]:
            conjs = []
            dep_syntax = seq_dep[dep_id - 1]
            while dep_syntax[3] in ["conj"]:
                if dep_syntax[3] == "conj":
                    conjs.append(int(dep_syntax[0])-1)
                dep_syntax = seq_dep[int(dep_syntax[2]) - 1]
            if dep_syntax[3] == "root":
                # 找出前面nn相连内容
                obj = [src_token]
                while seq_dep[id - 2][3] == "nn":
                    obj.insert(0, seq_dep[id - 2][1])
                    id -= 1
                obj = "".join(obj)

                if conjs:
                    masked_tokens.append([conjs[0], obj])
                else:
                    masked_tokens.append([int(dep_syntax[0])-1, obj])



        # # 2. rcmod+cpm+dobj 替换rcmod
        # if id + 1 < len(seq_dep):
        #     if syn == "rcmod" and src_dep[id][3] == "cpm" and src_dep[id+1][3] == "dobj":
        #         masked_tokens.append([id-1])
        #
        # # 3. dvpmod+（comod）+dvpm+ root/conj conj依附于root 替换dvpmod
        # if id+1 < len(seq_dep):
        #     if syn == "dvpmod" and src_dep[id][3] == "dvpm" or (src_dep[id][3] == "comod" and src_dep[id+1][3] == "dvpm"):
        #         # 验证是否是root或依附于root的conj
        #         if src_dep[id][3] == "dvpm":
        #             dep_syntax = seq_dep[id + 1]
        #         else:
        #             dep_syntax = seq_dep[id + 2]
        #         conjs = []
        #         while dep_syntax[3] in ["conj", "dep"]:
        #             if dep_syntax[3] == "conj":
        #                 conjs.append(int(dep_syntax[0]) - 1)
        #             dep_syntax = seq_dep[int(dep_syntax[2]) - 1]
        #         if dep_syntax[3] == "root":
        #             if src_dep[id][3] == "comod":
        #                 masked_tokens.append([id-1, id])
        #             else:
        #                 masked_tokens.append([id - 1])
        #
        # # 4. root + ... + cpm + （amod+attr）
        # if syn == "attr":
        #     dep_syntax = seq_dep[dep_id - 1]
        #     if dep_syntax[3] == "root":
        #         if seq_dep[id - 2][3] == "amod":
        #             masked_tokens.append([id-2, id-1])
        #         else:
        #             masked_tokens.append([id - 1])
        #
        # # 6 root/conj + cc + dep + ... + dobj  dobj依附于dep，dep依附于root/conj
        # if syn == "dobj":
        #     if seq_dep[dep_id - 1][3] == "dep" and seq_dep[dep_id - 2][3] == "cc":
        #         dep_syntax = seq_dep[dep_id - 3]
        #         conjs = []
        #         while dep_syntax[3] in ["conj", "dep"]:
        #             if dep_syntax[3] == "conj":
        #                 conjs.append(int(dep_syntax[0]) - 1)
        #             dep_syntax = seq_dep[int(dep_syntax[2]) - 1]
        #         if dep_syntax[3] == "root":
        #             masked_tokens.append([dep_id - 1])
        #             masked_tokens.append([dep_id - 3])

    # 去重,同时去掉过短或过长的内容(规则漏洞或句法分析错误导致)
    clean_masked_tokens = []
    tmp = []
    for lst in masked_tokens:
        if lst[0] not in tmp and 1 < len(seq_seg[lst[0]]) <= 4:
            clean_masked_tokens.append(lst)
            tmp.append(lst[0])

    # 去重, 随机选取 add_noise_num 个噪声进行加噪
    # clean_masked_tokens = list(set(tuple(x) for x in clean_masked_tokens))
    # clean_masked_tokens = random.sample(clean_masked_tokens, min(add_noise_num, len(clean_masked_tokens)))
    # clean_masked_tokens = list(set([j for i in clean_masked_tokens for j in i ]))
    clean_masked_tokens = sorted(clean_masked_tokens, reverse=True)

    verb_objs = []
    if len(clean_masked_tokens) == 0:
        flag[2] = 0
    else:
        flag[2] = 2
        for lst in clean_masked_tokens:
            verb_objs.append((seq_seg[lst[0]], lst[1], lst[0]))

    if ob:
        return src_seq, verb_objs

    return src_seq, verb_objs


def generate_mod_substitute_error(seq_seg, seq_dep, flag, seq_con=None, ob=False):
    '''
    生成替换类错误
    规则：
        谓语替换
        1. dobj/nsubj依附于谓语（可以是root，或conj） 替换谓语 形成动宾搭配不当
            最近又发动了全面的质量大检查运动，要在这个运动中完成建立与完善技术管理制度等一系列的工作。
            ->最近又发动了全面的质量大检查运动，要在这个运动中完成建立与【加强】技术管理制度等一系列的工作。

        宾语修饰语替换
        2. rcmod+cpm+dobj 替换rcmod 造 修饰语使用不当
            人们在会议上爆发了激烈的争论。
        ->  人们在会议上爆发了强烈的争论。
        谓语修饰语替换
        3. dvpmod+（comod）+dvpm+ root/conj conj依附于root 替换dvpmod
            要掌握走私犯的行动规律，以便更好地识别和打击他们。
        ->  要掌握走私犯的行动规律，以便稳准狠地识别和打击他们。

        4. 【测试】root + ... + cpm + （amod+attr） 替换宾语  发展社会主义文艺的【指导思想】
            attr依附于root， 寻找依附于attr的修饰成分一起替换，若较多则不修改

        并列谓语替换
        5. root/conj + cc + dep + ... + dobj  dobj依附于dep，dep依附于root/conj
            随机替换前后任一谓语
    :param seq_seg:
    :param seq_dep:
    :param seq_con:
    :return:
    '''
    # 是否加入噪声
    # whether_add_noise = choice(a=[0, 1], p=[0.2, 0.8], size=1, replace=False)[0]
    # if not whether_add_noise:
    #     return seq_seg
    # 噪声数量
    add_noise_num = choice(a=[1, 2, 3], p=[0.8, 0.15, 0.05], size=1, replace=False)[0]

    src_seq = seq_seg[:]
    src_dep = seq_dep[:]
    masked_tokens = []
    for token, token_syntax in zip(src_seq, src_dep):
        id, src_token, dep_id, syn = token_syntax
        id, dep_id = int(id), int(dep_id)
        assert token == src_token, print(token, src_token)

        # rcmod+cpm+(nn + dobj) 替换rcmod
        if id + 1 < len(seq_dep):
            if syn == "rcmod" and src_dep[id][3] == "cpm" and src_dep[id-2][3] != "advmod":
                obj_id = id+1
                while src_dep[obj_id][3] == "nn":
                    obj_id+=1
                if src_dep[obj_id][3] in ["dobj", "pobj", "lobj"]:
                    masked_tokens.append([id-1, ''.join(seq_seg[id+1:obj_id+1])])

        # # 3. dvpmod+（comod）+dvpm+ root/conj conj依附于root 替换dvpmod
        # if id+1 < len(seq_dep):
        #     if syn == "dvpmod" and src_dep[id][3] == "dvpm" or (src_dep[id][3] == "comod" and src_dep[id+1][3] == "dvpm"):
        #         # 验证是否是root或依附于root的conj
        #         if src_dep[id][3] == "dvpm":
        #             dep_syntax = seq_dep[id + 1]
        #         else:
        #             dep_syntax = seq_dep[id + 2]
        #         conjs = []
        #         while dep_syntax[3] in ["conj", "dep"]:
        #             if dep_syntax[3] == "conj":
        #                 conjs.append(int(dep_syntax[0]) - 1)
        #             dep_syntax = seq_dep[int(dep_syntax[2]) - 1]
        #         if dep_syntax[3] == "root":
        #             if src_dep[id][3] == "comod":
        #                 masked_tokens.append([id-1, id])
        #             else:
        #                 masked_tokens.append([id - 1])
        #
        # # 4. root + ... + cpm + （amod+attr）
        # if syn == "attr":
        #     dep_syntax = seq_dep[dep_id - 1]
        #     if dep_syntax[3] == "root":
        #         if seq_dep[id - 2][3] == "amod":
        #             masked_tokens.append([id-2, id-1])
        #         else:
        #             masked_tokens.append([id - 1])
        #
        # # 6 root/conj + cc + dep + ... + dobj  dobj依附于dep，dep依附于root/conj
        # if syn == "dobj":
        #     if seq_dep[dep_id - 1][3] == "dep" and seq_dep[dep_id - 2][3] == "cc":
        #         dep_syntax = seq_dep[dep_id - 3]
        #         conjs = []
        #         while dep_syntax[3] in ["conj", "dep"]:
        #             if dep_syntax[3] == "conj":
        #                 conjs.append(int(dep_syntax[0]) - 1)
        #             dep_syntax = seq_dep[int(dep_syntax[2]) - 1]
        #         if dep_syntax[3] == "root":
        #             masked_tokens.append([dep_id - 1])
        #             masked_tokens.append([dep_id - 3])

    # 去重,同时去掉过短或过长的内容(规则漏洞或句法分析错误导致)
    clean_masked_tokens = []
    tmp = []
    for lst in masked_tokens:
        if lst[0] not in tmp and 1 < len(seq_seg[lst[0]]) <= 4:
            clean_masked_tokens.append(lst)
            tmp.append(lst[0])

    clean_masked_tokens = sorted(clean_masked_tokens, key=lambda x:x[0], reverse=True)

    mod_objs = []
    if len(clean_masked_tokens) == 0:
        flag[2] = 0
    else:
        flag[2] = 2
        for lst in clean_masked_tokens:
            if len(seq_seg[lst[0]]) == 4:
                continue
            mod_objs.append((seq_seg[lst[0]], lst[1], lst[0]))

    if ob:
        return src_seq, mod_objs

    return src_seq, mod_objs


def generate_word_order_error(seq_seg, seq_dep, flag, seq_con=None, ob=False):
    '''
    生成词序类错误
    规则：
        1.  nsubj+(prep/advmod) 互换位置，得到主语缺失错误
            神探经过深入调查和缜密推理，终于将谜案破解，并在华莱士和帕里佐的婚礼上，怒指新郎官就是真正的杀人凶手。
        ->  【经过神探】深入调查和缜密推理，终于将谜案破解，并在华莱士和帕里佐的婚礼上，怒指新郎官就是真正的杀人凶手。
        2.  root+nn 互换位置 得到主语搭配错误【效果较差，弃置】
            这些颜色与草木的绿色配合，引起人们安静闲适的感觉。
        ->  这些颜色与草木的绿色配合，【人们引起】安静闲适的感觉。
        4.  rcmod+cpm+nsubj 交换rcmod与nsubj【50%噪音，且在数据集中占比较低，暂不考虑】
            观望的购房者   -> 购房者的观望 观望的购房者坚持几日就选择放弃了。
            从严的政策    -> 政策的从严
        5.  advmod + nsubj 将advmod换到nsubj后面 构成语序错误
            种种迹象表明，非但观望的购房者没有像往日那样，坚持几日就选择放弃，重新加入到购房大军中去，而且从严的政策，让购房者对房价下降充满信心。
        ->  种种迹象表明，【购房者的观望】【非但】没有像往日那样，坚持几日就选择放弃，重新加入到购房大军中去，而且【政策的从严】，让购房者对房价下降充满信心。
    :return:
    词序调换后的句子
    '''
    # 是否加入噪声
    # whether_add_noise = choice(a=[0, 1], p=[0.2, 0.8], size=1, replace=False)[0]
    # if not whether_add_noise:
    #     return seq_seg
    # 噪声数量
    add_noise_num = choice(a=[1, 2, 3], p=[0.8, 0.15, 0.05], size=1, replace=False)[0]
    add_noise_num = 1
    n = 4

    # 获取句法信息
    src_seq = seq_seg[:]
    src_dep = seq_dep[:]
    word_order_idx_list = [[] for i in range(n)]

    for token, token_syntax in zip(src_seq, src_dep):
        id, src_token, dep_id, syn = token_syntax
        id, dep_id = int(id), int(dep_id)
        assert token == src_token, print(token, src_token)

        """
        状语主语互换
        1. nsubj+(advmod)+prep
            该结构互换仅限出现在句首，或标点（，。！；）后
            若其后有obj：nsubj+prep+obj
                若prep为“对，对其，对于，与”,则交换主宾
                若prep为“关于”,则将 关于+obj 提前到句首
            obj 依附于 "prep"，nsubj前缀为任意 依附id < nsubj id 的(从 ，。 开始算)，长度小于5（词级别）
            补充：
            prep为“在”,需要在后面匹配到依附于在的 plmod
        2. prep+nsubj
        3. advmod+nsubj
        4. nsubj+advmod
        前两类放在word_order_idx_list[0],后两类放在[1]
        """
        # # 1. 找到nsubj+(advmod)+prep+(obj)结构
        # if syn == "nsubj":
        #     # 定位nsubj
        #     nsubj_lst = search_pre_syntax(seq_dep, int(id), end_syntaxs=["root"], end_tokens=[",", ".", "，", "。", "!", "！", ";", "；", "?", "？"])

        #     # 定位prep
        #     prep_lst = []
        #     if nsubj_lst[-1] + 1 < len(seq_dep) and seq_dep[nsubj_lst[-1] + 1][3] == "prep":
        #         prep_lst = [nsubj_lst[-1] + 1]
        #     elif nsubj_lst[-1] + 2 < len(seq_dep) and seq_dep[nsubj_lst[0] + 1][3] == "advmod" and seq_dep[nsubj_lst[0] + 2][3] == "prep":
        #         prep_lst = [nsubj_lst[-1] + 1, nsubj_lst[-1] + 2]

        #     # if seq_dep[0][1] == "人工智能" and seq_dep[1][1] == "由于":
        #     #     print(nsubj_lst, prep_lst)

        #     # "在...时“等情况单独处理
        #     if not prep_lst or (len(nsubj_lst) > 5 and seq_dep[prep_lst[-1]][1] not in ["对于"]) or seq_dep[prep_lst[-1]][1] in ["于", "从", "以", "如同"]:
        #         pass
        #     else:
        #         if seq_dep[prep_lst[-1]][1] == "在":
        #             # 找plmod
        #             for token_2, token_syntax_2 in zip(src_seq[prep_lst[-1] + 1:], src_dep[prep_lst[-1] + 1:]):
        #                 if token_syntax_2[3] in ["plmod"]:
        #                     word_order_idx_list[0].append([tuple(nsubj_lst + prep_lst), tuple(prep_lst + nsubj_lst)])
        #                     break
        #         else:
        #             # 定位obj
        #             obj_lst = []
        #             for token_2, token_syntax_2 in zip(src_seq[prep_lst[-1]+1:], src_dep[prep_lst[-1]+1:]):
        #                 # 找到依附于prep的obj
        #                 if token_syntax_2[3] in ["dobj", "pobj", "lobj"] and int(token_syntax_2[2])-1 == prep_lst[-1]:
        #                     obj_lst = [i for i in range(prep_lst[-1]+1, int(token_syntax_2[0]))]
        #                     if seq_dep[int(token_syntax_2[0])][1] in ["”", "\"", "》"]:
        #                         obj_lst.append(int(token_syntax_2[0]))
        #                     break

        #             if obj_lst:
        #                 # 根据prep指定交换方式
        #                 if "对" in src_dep[prep_lst[-1]][1] or src_dep[prep_lst[-1]][1] in ["与", "为", "给"]:
        #                     word_order_idx_list[0].append([tuple(nsubj_lst+prep_lst+obj_lst), tuple(obj_lst+prep_lst+nsubj_lst)])
        #                 elif src_dep[prep_lst[-1]][1] in ["关于", "自"]:
        #                     word_order_idx_list[0].append([tuple(nsubj_lst + prep_lst + obj_lst), tuple(prep_lst + obj_lst + nsubj_lst)])
        #                 else:
        #                     word_order_idx_list[0].append([tuple(nsubj_lst + prep_lst), tuple(prep_lst + nsubj_lst)])
        #             else:# 没有obj，只交换nsubj和prep
        #                 word_order_idx_list[0].append([tuple(nsubj_lst + prep_lst), tuple(prep_lst + nsubj_lst)])

        # # 2. prep+(obj)+nsubj
        # if syn == "nsubj":
        #     # 定位nsubj
        #     nsubj_lst = search_pre_syntax(seq_dep, int(id), end_syntaxs=["prep", "dobj", "lobj", "pobj", "root"])
        #     if len(nsubj_lst) > 5:
        #         pass
        #     else:
        #         prep_lst = []
        #         obj_lst = []
        #         if nsubj_lst[0] - 2 >= 0 and seq_dep[nsubj_lst[0] - 2][3] == "advmod" and seq_dep[nsubj_lst[0] - 1][3] == "prep":
        #             prep_lst = [nsubj_lst[0] - 2, nsubj_lst[0] - 1]
        #         elif nsubj_lst[0] - 1 >= 0 and seq_dep[nsubj_lst[0]-1][3] == "prep":
        #             prep_lst = [nsubj_lst[0] - 1]
        #         elif seq_dep[nsubj_lst[0]-1][3] in ["dobj", "pobj", "lobj"] and seq_dep[int(seq_dep[nsubj_lst[0]-1][2])-1][3] == "prep":
        #             obj_syn = seq_dep[nsubj_lst[0]-1]
        #             prep_syn = seq_dep[int(obj_syn[2])-1]
        #             if seq_dep[int(prep_syn[0])-2][3] == "advmod":
        #                 prep_lst = [int(prep_syn[0])-2, int(prep_syn[0])-1]
        #             else:
        #                 prep_lst = [int(prep_syn[0])-1]

        #             obj_lst = [i for i in range(prep_lst[-1] + 1, int(obj_syn[0]))]

        #         # if seq_dep[1][1] == "太阳岛" and seq_dep[2][1] == "国际":
        #         #     print(nsubj_lst, prep_lst)

        #         if prep_lst == []:
        #             pass
        #         else:
        #             if seq_dep[prep_lst[-1]][1] == "在" and obj_lst == []:
        #                 # 找plmod
        #                 for token_2, token_syntax_2 in zip(src_seq[prep_lst[-1] + 1:], src_dep[prep_lst[-1] + 1:]):
        #                     if token_syntax_2[3] in ["plmod"]:
        #                         word_order_idx_list[0].append([tuple(prep_lst + nsubj_lst), tuple(nsubj_lst + prep_lst)])
        #                         break
        #             elif seq_dep[prep_lst[-1]][1] in ["自从", "每当", "由于", "经过", "根据","因为","自","从"] and obj_lst == []:
        #                 word_order_idx_list[0].append([tuple(prep_lst + nsubj_lst), tuple(nsubj_lst + prep_lst)])
        #             elif seq_dep[prep_lst[-1]][1] in ["关于"] and obj_lst:
        #                 word_order_idx_list[0].append([tuple(prep_lst + obj_lst + nsubj_lst), tuple(nsubj_lst + prep_lst + obj_lst)])

        # # 3. nsubj+advmod
        # if syn == "nsubj":
        #     # 定位nsubj
        #     nsubj_lst = search_pre_syntax(seq_dep, int(id), end_syntaxs=["root"], end_tokens=[",", ".", "，", "。", "!", "！", ";", "；", "?", "？"])

        #     # 定位advmod
        #     advmod_lst = []
        #     if nsubj_lst[-1] + 2 < len(seq_dep) and seq_dep[nsubj_lst[-1] + 1][3] == "neg" and seq_dep[nsubj_lst[-1] + 2][3] == "advmod":
        #         advmod_lst = [nsubj_lst[-1] + 1, nsubj_lst[-1] + 2]
        #     elif nsubj_lst[-1] + 1 < len(seq_dep) and seq_dep[nsubj_lst[-1] + 1][1] in ["值得","尽管","突然","如果","虽然","即使","一方面","只有",
        #                                                                         "虽","之所以","即便","平均","就算","似乎","只要","将来",
        #                                                                         "不仅","不但","非但","一旦","假若","倘若","每天","往往","无论"]:
        #         advmod_lst = [nsubj_lst[-1] + 1]

        #     # if seq_dep[0][1] == "人工智能" and seq_dep[1][1] == "由于":
        #     #     print(nsubj_lst, prep_lst)

        #     if not advmod_lst:
        #         pass
        #     else:
        #         word_order_idx_list[1].append([tuple(nsubj_lst + advmod_lst), tuple(advmod_lst + nsubj_lst)])

        # # 4.advmod+nsubj
        # if syn == "nsubj":
        #     # 定位nsubj
        #     nsubj_lst = search_pre_syntax(seq_dep, int(id), end_syntaxs=["advmod"], end_tokens=[",", ".", "，", "。", "!", "！", ";", "；", "?", "？"])
        #     if len(nsubj_lst) > 10:
        #         pass
        #     else:
        #         advmod_lst = []
        #         obj_lst = []
        #         if nsubj_lst[0] - 2 >= 0 and seq_dep[nsubj_lst[0] - 2][3] == "neg" and seq_dep[nsubj_lst[0] - 1][3] == "advmod":
        #             advmod_lst = [nsubj_lst[0] - 2, nsubj_lst[0] - 1]
        #         elif nsubj_lst[0] - 1 >= 0 and seq_dep[nsubj_lst[0] - 1][1] in ["值得","尽管","突然","如果","虽然","即使","一方面","只有",
        #                                                                         "虽","之所以","即便","平均","就算","似乎","只要","将来",
        #                                                                         "不仅","不但","非但","一旦","假若","倘若","每天","往往","无论"]:
        #             advmod_lst = [nsubj_lst[0] - 1]

        #         if advmod_lst == []:
        #             pass
        #         else:
        #             word_order_idx_list[1].append([tuple(advmod_lst + nsubj_lst), tuple(nsubj_lst + advmod_lst)])

        """
        并列关系互换
        1. （conj/dep/root）+ cc + dep
            --后两者依附于前者
            --dep前可有前缀 neg+advmod
        2. conj/rcmod + punct + conj
            --三者都依附于同一个token
            --若为rcmod，都依附于punct（可选）
        3. conj + cc/punct + pobj/dobj/nn
            --conj，cc/punct依附于同一个token，强约束：均依附于punct
        放在word_order_idx_list[2]
        4. 三元素并列
            --conj/obj/ccomp + punct/cc + conj( + obj,仅当前面为obj时) + punct/cc + conj/ obj/ nn/ advmod, rcmod
            -- 左右两侧的成分可有一个不同，其余全部依附于同一个节点
        
        补充：conj+punct+conj结构，误识别的并列关系太多了，噪音比例较高,pass
        """
        # 1. （conj / dep / root）+ cc + dep（+dobj）
        # 均依附于第一个元素
        # 前缀 neg+advmod,advmod,conj,dep,root,advmod advmod,
        # 后缀 nn+dobj,dobj,dep,rcomp
        if syn in ['cc']:
            # 寻找该结构
            pre_lst = []
            if id-2 > 0 and seq_dep[id-2][3] in ["conj", "dep", "root"]:
                pre_lst = search_pre_pattern(seq_dep, id-2, [("neg", "advmod"), ("advmod",), ("conj",), ("dep",), ("root",), ("neg",), ("advmod", "advmod",)]) + [id-2]
            suf_lst = search_suffix_pattern(seq_dep, id - 1, [('neg', 'advmod', 'dep'), ('advmod', 'dep'), ('dep',),
                                                              ('neg', 'advmod', 'dep', 'dobj'), ('advmod', 'dep', 'dobj'), ('dep', 'dobj'),
                                                              ('neg', 'advmod', 'dep', 'nn', 'dobj'), ('advmod', 'dep', 'nn', 'dobj'), ('dep', 'nn', 'dobj'),
                                                              ('dep', 'dep'), ('dep', 'rcomp'),('dep', 'advmod', 'vmod', 'dobj')])
            if pre_lst and suf_lst:
                # 匹配依存关系
                pre_core = pre_lst[-1]
                suf_core = suf_lst[-1]
                if int(seq_dep[id-1][2]) == int(seq_dep[suf_core][2]) == int(seq_dep[pre_core][0]):
                    word_order_idx_list[2].append([tuple(pre_lst+[id-1]+suf_lst), tuple(suf_lst+[id-1]+pre_lst)])

        # 2.conj(+dobj) / rcmod + punct + conj(+dobj)
        #     --后面元素依附于第一个元素
        #     --若为rcmod，依附于punct（可选）
        if src_token == "、":
            # 寻找该结构
            pre_lst = search_pre_pattern(seq_dep, id - 1, [('rcmod',), ("advmod", "rcmod"),("neg", "advmod", "rcmod"),
                                                           ('conj',), ("advmod", "conj"),('neg',"advmod", "conj"),
                                                           ('conj','dobj'), ("advmod", "conj",'dobj'),('neg',"advmod", "conj", 'dobj')])
            suf_lst = search_suffix_pattern(seq_dep, id - 1, [('conj',), ('conj', "dobj"),('conj', "conj")])
            if pre_lst and suf_lst:
                # 匹配依存关系
                pre_core = pre_lst[-1]
                suf_core = suf_lst[0]
                if int(seq_dep[id-1][2]) == int(seq_dep[suf_core][2]) == int(seq_dep[pre_core][0]):
                    if seq_dep[int(seq_dep[pre_core][2])][3] == 'punct':
                        word_order_idx_list[2].append([tuple(pre_lst+[id-1]+suf_lst), tuple(suf_lst+[id-1]+pre_lst)])


        # 3.conj + cc / punct + pobj / dobj / nn
        #     --conj，cc / punct依附于同一个token，强约束：均依附于第二个元素
        # 前缀 conj conj，nn conj，nn nn conj，nsubj prtmod conj
        # 后缀 nn dobj，dobj dobj
        if src_token == "、" or syn in ['cc']:
            # 寻找该结构
            pre_lst = search_pre_pattern(seq_dep, id - 1, [("conj",), ("nn", "conj"), ("conj", "conj"),("nn", 'nn',"conj"),("nsubj", "prtmod", "conj")])
            suf_lst = search_suffix_pattern(seq_dep, id - 1, [('nn', 'dobj'),('pobj',), ('dobj',), ('dobj','dobj'), ('nn')])
            if pre_lst and suf_lst:
                # 匹配依存关系
                pre_core = pre_lst[-1]
                suf_core = suf_lst[-1]
                if int(seq_dep[pre_core][2]) == int(seq_dep[id-1][2]) == int(seq_dep[suf_core][0]):
                    word_order_idx_list[2].append(
                        [tuple(pre_lst + [id - 1] + suf_lst), tuple(suf_lst + [id - 1] + pre_lst)])

        # 4.三元素并列：
        # conj(+obj)/(rcmod+)obj/ccomp + punct/cc + conj( + obj,仅当前面为obj时) + punct/cc + conj(+obj)/ obj/ nn/ advmod, rcmod
        # --左右两侧的成分可有一个不同，其余全部依附于同一个节点
        # 60%交换1、3元素，20%分别交换12或23
        # 为排除并列关系，去除conj punct conj punct结构
        if src_token == "、" or syn in ['cc']:
            # 首先要确认为三元素并列结构
            suf_lst = []

            pre_lst = search_pre_pattern(seq_dep, id - 1, [("conj",),('nn','conj'),('conj','dobj'), ('dep','dobj'), ("dobj",),('rcmod','dobj'),("pobj",),("lobj",), ("ccomp",), ('dobj',"ccomp",)])
            if pre_lst and seq_dep[pre_lst[-1]][3] in ["dobj","pobj","lobj"]:
                mid_lst = search_suffix_pattern(seq_dep, id - 1, [('conj',), ('conj','dobj'),('conj','pobj'),('conj','lobj'),])
            else:
                mid_lst = search_suffix_pattern(seq_dep, id - 1, [('conj',)])

            if mid_lst and mid_lst[-1]+1 < len(seq_dep) and (seq_dep[mid_lst[-1]+1][1] == "、" or seq_dep[mid_lst[-1]+1][3] in ['cc']):
                suf_lst = search_suffix_pattern(seq_dep, mid_lst[-1]+1, [('conj',), ('nsubj','conj'),('conj','dobj'),('dobj',),('pobj',),('lobj',),("nn",),("advmod","rcmod"),('conj','comod'),('conj','dobj')])

            if pre_lst and mid_lst and suf_lst:
                # 确认依存关系
                lst = [idx for idx in range(id-1, mid_lst[-1]+2)]
                dep_syn = seq_dep[lst[0]][2]
                valid = True
                for i in lst:
                    if seq_dep[i][2] != dep_syn:
                        valid = False
                        break
                left_dep = seq_dep[pre_lst[-1]][2]
                right_dep = seq_dep[suf_lst[-1]][2]
                if left_dep != dep_syn and right_dep != dep_syn:
                    valid = False

                if valid:
                    # 60%交换1、3元素，20%分别交换12或23
                    mode = choice(a=[1, 2, 3], p=[0.6, 0.2, 0.2], size=1, replace=False)[0]
                    if mode == 1:
                        word_order_idx_list[2].append(
                        [tuple(pre_lst + [id - 1] + mid_lst + [mid_lst[-1]+1] + suf_lst),
                         tuple(suf_lst + [id - 1] + mid_lst + [mid_lst[-1]+1] + pre_lst)])
                    elif mode == 2:
                        word_order_idx_list[2].append(
                            [tuple(pre_lst + [id - 1] + mid_lst + [mid_lst[-1] + 1] + suf_lst),
                             tuple(mid_lst + [id - 1] + pre_lst + [mid_lst[-1] + 1] + suf_lst)])
                    elif mode == 3:
                        word_order_idx_list[2].append(
                            [tuple(pre_lst + [id - 1] + mid_lst + [mid_lst[-1] + 1] + suf_lst),
                             tuple(pre_lst + [id - 1] + suf_lst + [mid_lst[-1] + 1] + mid_lst)])


        # 6. (advmod+rcmod)+cpm+nsubj
        # 结构前面不能是nsubj, 不能依附于rcmod
        # 还是有一定噪音，大概50%，限制造错概率。
        # 在数据集中占比低，暂不加入
        # whether_add_noise = choice(a=[0, 1], p=[0.5, 0.5], size=1, replace=False)[0]
        # if whether_add_noise:
        #     if syn == "rcmod" and len(src_token) != 1:
        #         if seq_dep[id][3] == "cpm" and seq_dep[id+1][3] == "nsubj" and seq_dep[id+1][1] not in ["时候", "现象"]:
        #             # seq_seg[id - 1], seq_seg[id + 1] = src_seq[id+1], src_seq[id-1]
        #             if id-2 >= 0 and seq_dep[id-2][3] == "advmod" and seq_dep[id-2][1] not in ["而"]:
        #                 if id-3 >= 0 and (seq_dep[id - 3][3] in ["nsubj", "dobj", "lobj"] or (int(seq_dep[id - 3][2]) == id)):
        #                     continue
        #                 word_order_idx_list[3].append([(id-2, id - 1, id, id + 1), (id + 1, id, id-2, id - 1)])
        
        #             if id-2 >= 0 and (seq_dep[id - 2][3] in ["nsubj", "dobj", "lobj"] or (int(seq_dep[id - 3][2]) == id)):
        #                 continue
        #             word_order_idx_list[3].append([(id - 1, id+1), (id+1, id - 1)])


    results = [[] for i in range(n)]

    # if word_order_idx_list[2]:
    #     print(word_order_idx_list[2])

    # 去重，若有相同idx需要调换位置，也视为重复
    for idx, word_order_idx in enumerate(word_order_idx_list):
        temp = []
        word_order_idx_dedup = []
        for i in word_order_idx:
            dup = False
            tup1, tup2 = i
            for j in tup1:
                if j in temp:
                    dup = True
            if dup:
                continue
            else:
                temp += list(tup1+tup2)
                word_order_idx_dedup.append(i)
        word_order_idx_dedup = random.sample(word_order_idx_dedup, min(add_noise_num, len(word_order_idx_dedup)))
        seq_seg = src_seq[:]
        for tup1, tup2 in word_order_idx_dedup:
            if len(tup2) == 1:
                continue
            for i, j in zip(tup1,tup2):
                seq_seg[i] = src_seq[j]
        results[idx] = seq_seg
    return results


def generate_by_syntax_S(myargs):
    data, GM = myargs

    results = []
    for seq_seg, seq_dep in tqdm(data):

        # # 含有英文的暂不考虑
        # if not is_all_chinese(seq_seg):
        #     results.append((seq_seg, []))
        #     continue

        flag = [1, 1, 1, 1]
        src, verb_objs = generate_mod_substitute_error(copy.deepcopy(seq_seg), copy.deepcopy(seq_dep), flag)
        results.append((src, verb_objs))

    return results


def generate_by_syntax(myargs, model_name="../../plm/bert-base-chinese"):
    data, GM = myargs

    # time.sleep(random.randint(0,180))
    # device_id = GM.auto_choice()
    # # 加载模型
    # tokenizer = BertTokenizer.from_pretrained(model_name)
    # model = BertForMaskedLM.from_pretrained(model_name)
    # model.eval()
    #
    # device=torch.device("cuda", device_id)
    # model.to(device)
    
    generate_results = []
    for seq_seg, seq_dep in tqdm(data):
        results = []
        flag = [1,1,1,1]

        # error_type = choice(a=["R", "M", "S", "W"], p=[0.2, 0.35, 0.15, 0.3], size=1, replace=False)[0]
        error_type = "W"

        # # 含有英文的暂不考虑
        # result = ''.join(seq_seg)
        # if not is_all_chinese(seq_seg):
        #     generate_results.append([result, [result for _ in range(n)]])
        #     continue

        if error_type == "R":
            results = generate_redundant_error(copy.deepcopy(seq_seg), copy.deepcopy(seq_dep), flag)
        elif error_type == "M":
            results = generate_missing_error(copy.deepcopy(seq_seg), copy.deepcopy(seq_dep), flag)
        elif error_type == "S":
            results = generate_substitute_error(copy.deepcopy(seq_seg), copy.deepcopy(seq_dep), flag)
        elif error_type == "W":
            results = generate_word_order_error(copy.deepcopy(seq_seg), copy.deepcopy(seq_dep), flag)


        for idx, result in enumerate(results):
            result = "".join(result)
            result = result.replace("#", "")
            result = result.replace("[UNK]", "\"")
            results[idx] = result

        generate_results.append(["".join(seq_seg), results])

    # # 将模型model转到cpu
    # model.to('cpu')
    # # 删除模型，也就是删除引用
    # del model
    # # 释放GPU。
    # torch.cuda.empty_cache()

    return generate_results
    # f1.write(result + "\n")


if __name__=="__main__":
    # seg_path = "../../data/FCGEC/FCGEC_train.tgt.thulac"
    # dep_path = "../../data/FCGEC/FCGEC_train.tgt.dep.conll_predict"
    # con_path = ""
    # src_path = "../../data/FCGEC/pseudo/FCGEC_train.pseudo.correct.M."
    # pseudo_path = "../../data/FCGEC/pseudo/FCGEC_train.pseudo.error.M."
    # para_path = "../../data/FCGEC/pseudo/FCGEC_train.pseudo.para.M."

    # seg_path = "../../data/exam_error/exam.tgt.thulac"
    # dep_path = "../../data/exam_error/exam.tgt.dep.conll_predict"
    # con_path = ""
    # src_path = "../../data/exam_error/pseudo/exam.pseudo.correct.W."
    # pseudo_path = "../../data/exam_error/pseudo/exam.pseudo.error.W.2025.2.23."
    # para_path = "../../data/exam_error/pseudo/exam.pseudo.para.W.2025.2.23."

    # seg_path = "../../data/news/news.seq.400w.txt.thulac"
    # dep_path = "../../data/news/news.seq.400w.txt.dep.conll_predict"
    # con_path = ""
    # src_path = "../../data/news/pseudo/news.400w.correct.W."
    # pseudo_path = "../../data/news/pseudo/news.400w.error.W."
    # para_path = "../../data/news/pseudo/news.400w.para.W."


    seg_path = "../../data/news/news.seq.100w.txt.thulac"
    dep_path = "../../data/news/news.seq.100w.txt.dep.conll_predict"
    con_path = ""
    src_path = "../../data/news/pseudo/news.100w.correct.W."
    pseudo_path = "../../data/news/pseudo/news.100w.error.W.2025.2.26"
    para_path = "../../data/news/pseudo/news.100w.para.W.2025.2.26"

    # seg_path = "../../data/test/W/FCGEC_train.W.3.test.tgt.thulac"
    # dep_path = "../../data/test/W/FCGEC_train.W.3.test.tgt.dep.conll_predict"
    # con_path = ""
    # src_path = "../../data/test/W/FCGEC_train.W.3.test.tgt"
    # pseudo_path = "../../data/test/W/FCGEC_train.W.3.test.src.pseudo"
    # para_path = "../../data/test/W/FCGEC_train.W.3.test.pseudo.para"

    seqs_seg = read_src(seg_path)
    seqs_dep = read_dep(dep_path)
    assert len(seqs_seg) == len(seqs_dep), print(len(seqs_seg), len(seqs_dep))

    print("数据集大小:", len(seqs_seg))

    data = [list(tup) for tup in zip(seqs_seg, seqs_dep)]

    n=50000
    data_split = [data[i:i + n] for i in range(0, len(data), n)]


    # 多进程模式
    DEVICE = [0, 1, 2, 3, 4, 5, 6, 7]
    GM = GPUManager(DEVICE)
    myargs = [[chunk, GM] for chunk in data_split]

    generate_results = []
    with Pool(25) as pool:
        for results in pool.imap(generate_by_syntax, tqdm(myargs), chunksize=1):
            if results:
                generate_results += results

    n = len(generate_results[0][1])
    if generate_results:
        for i in range(n):
            with open(pseudo_path+str(i+1), "w", encoding="utf8") as f1, open(src_path+str(i+1), "w", encoding="utf8") as f2, open(para_path+str(i+1), "w", encoding="utf8") as f3:
                for src, tgts in generate_results:
                    tgt = tgts[i]
                    if src != tgt:
                        f2.write(src+"\n")
                        f1.write(tgt+"\n")
                        f3.write(tgt+"\t"+src+"\n")


# # 生成替换类错误
    # with open(pseudo_path, "w", encoding="utf8") as f1:
    #     # 多进程模式
    #     DEVICE = [0, 1, 2, 3, 4, 5, 6, 7]
    #     GM = GPUManager(DEVICE)
    #     myargs = [[chunk, GM] for chunk in data_split]
    #
    #     with Pool(25) as pool:
    #         for results in pool.imap(generate_by_syntax_S, tqdm(myargs), chunksize=1):
    #             if results:
    #                 for src, verb_objs in results:
    #                     if verb_objs:
    #                         f1.write(" ".join(src)+"\n")
    #                         for src_content, tgt_content, src_idx in verb_objs:
    #                             f1.write(src_content+"\t"+tgt_content+"\t"+str(src_idx)+"\n")
    #                         f1.write("\n")

