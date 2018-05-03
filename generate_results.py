import clean_data
import make_complex_prediction

results = make_complex_prediction.predict()
result_list = []
for item in results:
    result_list.append(item)

big_list = []
with open("result_sw_analysis_en.file", "r") as fp:
    lines = fp.readlines()
    curr_list = []
    for x in range(len(lines) - 1):
        if lines[x].split()[0] != lines[x+1].split()[0]:
            curr_list.append(lines[x])
            if len(curr_list) == 20:
                big_list.append(curr_list)
            curr_list = []
        else:
            curr_list.append(lines[x])
for item in big_list:
    if len(item) != 20:
        big_list.remove(item)

print len(big_list), len(result_list)
with open("my_result_sw.file", "a+") as fp:
    for y in range(len(big_list)-5):
        #print big_list[y]
        for z in range(0, result_list[y]+1):
            fp.write(big_list[y][z])

