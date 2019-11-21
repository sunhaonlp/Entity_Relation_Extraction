import json
import codecs
from goto import with_goto


@with_goto
def main():
    file = open("filter_data_processed.txt", "r",encoding= 'utf-8')

    entity_type=[1,2,3,4]   # 1.企业 2.人物 3.产品 4.有关部门
    relation_type=[1,2,3,4,5]   # 1.合作（企业和企业之间） 2.竞争（企业和企业之间） 3.属于（企业和产品之间） 4.监管（有关部门和企业之间）5.管理（人物和企业之间）
    # bert=[]
    num_=0  # 记录已经产生了多少个数据
    for line in file.readlines():  # 依次读取每行
        num_+=1
        label .begin1
        line=line.replace("\n","")
        line=line.replace("\\","")
        line=line.replace("/","")
        line=line.replace("：","")
        line=line.replace(":","")
        line=line.replace("*","")
        line=line.replace("'","")
        line=line.replace("<","")
        line=line.replace(">","")
        line=line.replace("|","")
        line=line.replace("?","")
        line=line.replace("？","")
        print("句子: "+line)
        print("是否有实体")
        input_1 = input("请输入(y/n):")
        if(input_1=='y' or input_1=='Y'):  # 如果存在实体
            dict_bert = {}
            dict_bert['entity-mentions'] = []
            dict_bert['sentence'] = line
            try:
                num = int(input("请输入实体的个数："))
            except:
                num = int(input("输入错误！请重新请输入实体的个数："))

            ## 处理实体
            for i in range(num):  # 对每个实体操作
                tmp_dict={}
                ####################
                ## 确定实体名
                ####################
                input_2 = input("请输入第"+str(i+1)+"个实体名：")
                while(input_2 not in line):
                    input_2 = input("未在该句中检测到 "+input_2+" 请重新输入实体名：")
                tmp_dict['text']=input_2
                ####################
                ## 确定实体类型
                ####################
                try:
                    input_3 = int(input("请输入 " + input_2 + " 的实体类型(1.企业 2.人物 3.产品 4.有关部门):\n"))
                except:
                    input_3 = int(input("输入错误！请重新输入 " + input_2 + " 的实体类型(1.企业 2.人物 3.产品 4.有关部门):\n"))

                while (input_3 not in entity_type):
                    input_3 = int(input("不存在该种类型，请输入1-4的整数!(1.企业 2.人物 3.产品 4.有关部门)\n"))
                if(input_3 == 1):
                    tmp_dict['entity-type'] = "company"
                if(input_3 == 2):
                    tmp_dict['entity-type'] = "people"
                if(input_3 == 3):
                    tmp_dict['entity-type'] = "product"
                if(input_3 == 4):
                    tmp_dict['entity-type'] = "department"
                ####################
                ##确定实体位置
                ####################
                tmp_dict['start']=line.find(input_2)
                tmp_dict['end']=tmp_dict['start']+len(input_2)
                dict_bert['entity-mentions'].append(tmp_dict)

            decision  = input("是否确认写入实体?(y/n):")
            if(decision == 'y' or decision == 'Y'):
                print("已写入")
            else:
                print("再来一次：")
                goto .begin1

            final_json = json.dumps(dict_bert, indent=4 ,ensure_ascii=False)
            with codecs.open("entity/"+line[:10]+".json", 'w', 'utf-8') as file:
                file.write(final_json)

            ## 处理关系
            input_4 = input("实体间是否存在关系?请输入(y/n):")
            while(input_4 == 'y'or input_4=='Y'):
                label .begin2

                entity_1 = input("请输入第一个实体:")
                while(entity_1 not in line):
                    entity_1 = input("未在该句中检测到 "+ entity_1 +" 请重新输入实体名：")

                entity_2 = input("请输入第二个实体:")
                while (entity_2 not in line):
                    entity_2 = input("未在该句中检测到 " + entity_2 + " 请重新输入实体名：")

                try:
                    relation = int(input("请输入 " + entity_1 + " 和 " + entity_2 + " 间的关系(1.合作 2.竞争 3.属于 4.监管 5.管理)"))
                except:
                    relation = int(input("输入错误，请重新输入 " + entity_1 + " 和 " + entity_2 + " 间的关系(1.合作 2.竞争 3.属于 4.监管 5.管理)"))

                while(relation not in relation_type):
                    relation = int(input("请重新输入(1.合作 2.竞争 3.属于 4.监管 5.管理)"))
                if(relation == 1):
                    relation_type_str='合作'
                if(relation == 2):
                    relation_type_str='竞争'
                if(relation == 3):
                    relation_type_str='属于'
                if(relation == 4):
                    relation_type_str='监管'
                if(relation == 5):
                    relation_type_str='管理'

                decision  = input("是否确认写入关系?(y/n):")
                if(decision == 'y' or decision == 'Y'):
                    print("已写入")
                else:
                    print("再来一次：")
                    goto .begin2

                file_tmp = open("relation/"+entity_1+"_"+entity_2+".txt", "w", encoding='utf-8')
                sequence = (entity_1,entity_2,relation_type_str,line)
                file_tmp.write('\t'.join(sequence))
                file_tmp.flush()

                input_4 = input("实体间是否还存在其它关系?请输入(y/n):")

            # bert.append(dict_bert)##########
            print('#####################################################################################')
            print('##############################你已经看了'+str(num_)+'条数据啦####################################')
            print('#####################################################################################')
        else:  #如果不存在实体
            print('#####################################################################################')
            print('##############################你已经看了'+str(num_)+'条数据啦####################################')
            print('#####################################################################################')
            continue

if __name__ == '__main__':
  main()