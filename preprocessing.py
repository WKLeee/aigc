import re
import pandas as pd

def mapper_initializer(mapper_size):
    mapper = {}
    for i in range(mapper_size):
        mapper['['+str(i)+']'] = None
    return mapper

def question_mapper(questions, mapper_size):

    mapper_list = []
    new_questions = []

    #max_count = 0
    for q in questions:
        mapper = mapper_initializer(mapper_size)

        nums = re.findall(r'\d*[\.|\,]\d+|\d+',q)
 
        #if max_count < len(nums):
        #    max_count = len(nums)

        new_q = ''
        temp_q = q
        for i, n in enumerate(nums):
            num_token = '['+str(i)+']'
            new_q = new_q + temp_q.split(n,1)[0] + num_token
            temp_q = temp_q.split(n,1)[1]

            if '.' in n:
                n = float(n)
            elif ',' in n:
                n = int(n.split(',')[0]+n.split(',')[1])
            else:
                n = int(n)
            mapper[num_token]=n

        new_q += temp_q
        new_questions.append(new_q)
        mapper_list.append(mapper)
    #print(max_count)

    return(new_questions, mapper_list)

if __name__=="__main__":
    mapper_size = 10

    total_data = pd.read_csv('test.csv',header=None, names=['ques', 'equation'])

    questions = total_data['ques']
    questions = questions.tolist()
    equations = total_data['equation']
    equations = equations.tolist()

    new_questions, mapper_list = question_mapper(questions, mapper_size)
    print(len(questions))
    print(len(new_questions), len(mapper_list))

    rng = 10
    for r in zip(new_questions,mapper_list, questions):
        print(r[2])
        print(r[0])
        print(r[1])
        if rng==0:
            break
        else:
            rng-=1