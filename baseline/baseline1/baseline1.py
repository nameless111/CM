import jieba.posseg as pseg


def rules_based_data_fp_exact(sentence):
    """
    BaseLine1-按照规则进行功能点抽取
    pattern1 - Verb Noun - v n
    Pattern2 - Verb Pronoun Noun - v r n
    Pattern3 - Verb Noun Noun - v n n
    Pattern4 - Verb Adjective Noun - v a n
    Pattern5 - Verb Deteminer Noun - v m n
    Pattern6 - Verb Noun Preposition Noun - v n p n
    Pattern7 - Verb Preposition Adjective Noun - v p a n
    Pattern8 - Verb Pronoun Adjective Noun - v r a n
    :param sentence:
    :return:
    """
    fps = []

    # pattern list
    patterns = []
    pattern1 = ['v', 'n']
    pattern2 = ['v', 'r', 'n']
    pattern3 = ['v', 'n', 'n']
    pattern4 = ['v', 'a', 'n']
    pattern5 = ['v', 'm', 'n']
    pattern6 = ['v', 'n', 'p', 'n']
    pattern7 = ['v', 'p', 'a', 'n']
    pattern8 = ['v', 'r', 'a', 'n']
    patterns.append(pattern1)
    patterns.append(pattern2)
    patterns.append(pattern3)
    patterns.append(pattern4)
    patterns.append(pattern5)
    patterns.append(pattern6)
    patterns.append(pattern7)
    patterns.append(pattern8)

    words = []
    pogs = []
    cut_results = pseg.cut(sentence)
    for word, pog in cut_results:
        words.append(str(word))
        pogs.append(str(pog))

    for pattern in patterns:
        index_infos = sub_list_index(pogs, pattern)
        for (start_index, end_index) in index_infos:
            fp_word_list = words[start_index:end_index]
            fps.append(''.join(fp_word_list))

    return fps


def sub_list_index(list, sub_list):
    """
    判断sub_list是否是list的子序列，如果是，则返回元祖列表，标记开始和结束所以
    :param list:
    :param sub_list:
    :return: list  [(start_index, end_index), (), (), ]
    """
    results = []

    for index, element_in_list in enumerate(list):
        if element_in_list != sub_list[0]:
            continue
        else:
            cut_list = list[index:index+len(sub_list)]
            if cut_list == sub_list:
                results.append((index, index+len(sub_list)))

    return results


if __name__ == '__main__':
    # get_all_labeled_data_lm_scores()
    fps = rules_based_data_fp_exact('我想要在界面中分享一张照片')
    for fp in fps:
        print(fp)
