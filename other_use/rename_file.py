import os, sys
from tqdm import tqdm
from googletrans import Translator
from pprint import pprint

def rename_file(file_name):
    # location_list = ['高雄','澎湖','新北','花蓮','七星潭','嘉義']  hash
    # translator = Translator(service_urls=['translate.googleapis.com'])
    # translator = Translator()
    file_name = file_name.replace("_"," ")
    file_name = file_name.replace("-"," ")
    file_name = file_name.replace(" "," ")
    file_name = file_name.replace("~"," ")
    
    file_name = file_name.split(" ")
    
    times = len(file_name)
    for i in range(times)[0:-1]:
        
        # file_name[i] = translator.translate(file_name[i], dest = 'en').text
        file_name[i] = hash(file_name[i])
        # print(file_name[i])
    
    new_name = ""
    for i in range(times):
        new_name = str(new_name)+ "_" + str(file_name[i])
    # print(new_name)
    
    return new_name

source_img_folder = r"D:\Yun\sea_trash\corrupt_data\valid\images"
source_label_folder = r"D:\Yun\sea_trash\corrupt_data\valid\labels"

file_img_list = os.listdir(path = source_img_folder)
file_count = len(file_img_list)
error_list = []
error_count = 0

for i in tqdm(range(file_count)):
    # print(file_img_list[i])
    
    
    file_img_name = file_img_list[i].replace(".JPG","")
    # file_img_name = file_img_name.replace(".jpg","")
            
    source_img_file = source_img_folder+ "\\"+ file_img_name+ ".JPG"
    source_label_file = source_label_folder+ "\\"+ file_img_name+ ".JPG.txt"
            
    new_filename = rename_file(file_img_name)
    new_img_name = source_img_folder+ "\\"+ new_filename + ".jpg"
    new_label_name = source_label_folder+ "\\"+ new_filename + ".txt"
            
    os.rename(source_img_file,new_img_name)
    os.rename(source_label_file,new_label_name)
    
    # except:
    #     error_list.append(file_img_name)
    #     error_count += 1
        
print(error_count) 