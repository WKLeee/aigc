import re
import pandas as pd

def question_mapper(questions):

    mapper_list = []
    for q in questions:
        mapper = {
            '[0]':None, '[1]':None, '[2]':None, '[3]':None,
            '[4]':None, '[5]':None, '[6]':None, '[7]':None,
            '[8]':None, '[9]':None, '[10]':None
        }

        print(q)
        nums = re.findall(r'\d*[\.|\,]\d+|\d+',q)
        
        for i, n in enumerate(nums):
            if '.' in n:
                n = float(n)
            elif ',' in n:
                n = int(n.split(',')[0]+n.split(',')[1])
            else:
                n = int(n)
            mapper['['+str(i)+']']=n

        print(mapper)
        for k in mapper:
            if mapper[k]:
                print(mapper[k])
                
            else:
                break
        input()        
        mapper_list.append(mapper)
        #break

if __name__=="__main__":

    total_data = pd.read_csv('test.csv',header=None, names=['ques', 'equation'])

    questions = total_data['ques']
    questions = questions.tolist()
    equations = total_data['equation']
    equations = equations.tolist()

    question_mapper(questions)
