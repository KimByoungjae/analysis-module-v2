from konlpy.tag import Komoran
#komoran = Komoran()
komoran = Komoran(userdic='Modules/keyword/data/dic.user')    


import sys
import os
import json

sys.path.append(os.path.abspath("./Modules/keyword/"))
from textrank_models import KeywordSummarizer
from WebAnalyzer.utils.media import frames_to_timecode
import pickle
import re

def cleanText(readData):
    #텍스트에 포함되어 있는 특수 문자 제거
    text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', readData)
    return text

class Keyword:

    with open('Modules/keyword/data/keyword_list.pkl', 'rb') as f:
        videos = pickle.load(f)

    with open('Modules/keyword/data/tourkeyword.pkl', 'rb') as f2:
        tourAPI_dict = pickle.load(f2)

    with open('Modules/keyword/data/per_keyword_ext_testset.pkl', 'rb') as f3:
        testset = pickle.load(f3)



    def open_ASR_file(file_name):
        ASR_file = open(file_name, "r", encoding='utf-8')
        ASR_lines = ASR_file.readlines()
        ASR_video_list = {}
        for line in ASR_lines:
            line = cleanText(line)
            line = line.split(' ')
            idx = line[0]
            text = line[1:]
            idx = idx.split('_')
            video_idx = idx[6][5:]
            text = ' '.join(text)
            video_idx = int(video_idx)
            if video_idx in ASR_video_list:
                ASR_video_list[video_idx].append(text)
            else:
                ASR_video_list[video_idx] = [text]

        return ASR_video_list

    def komoran_tokenize(sent):
        words = komoran.pos(sent, join=True)
        words = [w for w in words if ('/SL' in w  or '/NN' in w or '/XR' in w or '/VA' in w or '/VV' in w)]
        return words

    keyword_extractor = KeywordSummarizer(
        tokenize = komoran_tokenize,
        window = -1,
        verbose = False,
        min_count=2,
        min_cooccurrence=1
        )

    keywords_list = []
    tourAPI = []
    video_num = 1
    ############################
    put_ASR = 1

    sogang_asr = open_ASR_file("Modules/keyword/asr_data/asr_sogang_result.txt")
    google_asr = open_ASR_file("Modules/keyword/asr_data/asr_google_result.txt")

    for dic in tourAPI_dict:
        dic = cleanText(dic)
        dic_word = komoran.pos(dic)
        tourAPI += [w+'/'+pos for w, pos in dic_word if ('SL' in pos or 'NN' in pos or 'XR' in pos or 'VA' in pos or 'VV' in pos)]
    tourAPI = set(tourAPI)
    tourAPI = list(tourAPI)


    ### seoul tour dataset####
    model = None
    result = None
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):
        # TODO
        #   - initialize and load model here
        #model_path = os.path.join(self.path, "model.txt")
        #self.model = open(model_path, "r")
        pass

    def inference_by_text(self, data, video_info):
        result = {'text_result':[]}
        # TODO
        #   - Inference using aggregation result
        
        userdic= []
        userdicfile = open("Modules/keyword/data/dic.user", "r")
        lines = userdicfile.readlines()
        for line in lines:
            line = line.split('\t')
            userdic.append(line[0])
        ### test set ###
        keyword_test_list = []
        for c_i, case in enumerate(testset):
            keyword_test = {}
            try:
                key_test = keyword_extractor.summarize(case[1], topk=5)
            except:
                continue
            keyword_test['keyword'] = key_test
            keyword_test['answer'] = case[0]
            keyword_test_list.append(keyword_test)


        f1 = 0
        total = 1
        total_num = len(keyword_test_list)
        for t_k in keyword_test_list:
            compare_list = []
            score_list = []
            list_count = 0
            dic_flag = 0
            print(total)

            for word, rank in t_k['keyword']:
                print('{} ({:.3})'.format(word, rank))
                word = word.split('/')
                word = word[0]
                if list_count < 1:
                    compare_list.append(word)
                    score_list.append(rank)
                    list_count += 1
                if word in userdic:
                    compare_list.pop()
                    compare_list.append(word)
                    score_list.pop()
                    score_list.append(rank)
                    dic_flag = 1
                if dic_flag == 0:
                    for ud in userdic:
                        if word in ud:
                            compare_list.pop()
                            compare_list.append(word)
                            score_list.pop()
                            score_list.append(rank)
                result_element = {'label': [{'description': str(compare_list[0]), 'score': score_list[0]}]}
                result['aggregation_result'].append(result_element)

            print("keyword선택 단어: " + str(compare_list[0]))
            print("Topic단어: "+str(t_k['answer']))
            print("")
            total += 1
            for word in compare_list:
                if word == t_k['answer']:
                    f1 += 1
                    break

        print("acc(f1) : " + str(f1/total_num*100))
        
        """
        #########output format########
        result = {"aggregation_result": [
            {
                # 1 timestamp & multiple class
                'label': [
                    {'description': 'word_name', 'score': 1.0},
                    {'description': 'word_name', 'score': 1.0}
                ],
            },
            {
                # 1 timestamp & 1 class
                'label': [
                    {'description': 'word_name', 'score': 1.0}
                ],
            }
        ]}
        """
        self.result = result

        return self.result








