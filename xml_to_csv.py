import glob
import pandas as pd
import xml.etree.ElementTree as ET

def changeXmlToCsv(path):
    xml_list = []
    # 读取path下以.xml结尾的文件，并对xml文件进行解析
    for xml_file in glob.glob(path + '/*.xml'):
        xmlTree = ET.parse(xml_file)
        xmlRoot = xmlTree.getroot()
        for xmlMember in xmlRoot.findall('object'):
            # value分别对应filename，width，height，class，xmin，ymin，xmax，yamx的值
            value = (xmlRoot.find('filename').text.split('.')[0] + '.jpg',
                     int(xmlRoot.find('size')[0].text),
                     int(xmlRoot.find('size')[1].text),
                     xmlMember[0].text,
                     int(xmlMember[4][0].text),
                     int(xmlMember[4][1].text),
                     int(xmlMember[4][2].text),
                     int(xmlMember[4][3].text)
                     )
            xml_list.append(value)
            column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
            # 按照3：1的比例将所有数据分为样本集和验证集
            train_list = xml_list[0: int(len(xml_list) * 0.67)]
            eval_list = xml_list[int(len(xml_list) * 0.67) + 1:]
            # 利用padas库下的相关函数将文件保存为CSV格式
            pd.DataFrame(train_list, columns=column_name).to_csv('data/train.csv', index=None)
            pd.DataFrame(eval_list, columns=column_name).to_csv('data/eval.csv', index=None)

def main():
    path = r'E:\datas\datas\xml'  # xml文件的路径
    changeXmlToCsv(path)
    print('成功将xml文件转换为csv文件！！！')
main()
