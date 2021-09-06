import os
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import copy
from scipy.stats import spearmanr
from scipy.stats import pearsonr

def find_files(top_level_dir, mode):
    decade_path = []
    fif_files = []
    six_files = []
    sev_files = []
    eig_files = []
    nin_files = []
    ten_files = []
    zer_files = []

    if str(mode) == 'crossent':
        mode = 'CrossEnt'
    if str(mode) == 'kl':
        mode = 'TS'

    for path, dirlist, filelist in os.walk(top_level_dir):
        decade_path.append([path + '/'])
    decade_path = decade_path[1:]
    # print(decade_path)
    for i in decade_path:
        # print(i)
        # print(i[0][-10:])
        # if 'six' in i[0][-10:]:
        # print(i[0][-10:])
        for path, dirlist, filelist in os.walk(i[0]):
            # print(path)
            # print(dirlist)
            # files = filelist
            # print(filelist)
            for j in range(len(filelist)):
                # print(filelist[i])
                if mode in filelist[j]:
                    if 'fif' in i[0][-10:]:
                        fif_files.append(filelist[j])
                    if 'six' in i[0][-10:]:
                        six_files.append(filelist[j])
                    if 'sev' in i[0][-10:]:
                        sev_files.append(filelist[j])
                    if 'eig' in i[0][-10:]:
                        eig_files.append(filelist[j])
                    if 'nin' in i[0][-10:]:
                        nin_files.append(filelist[j])
                    if 'ten' in i[0][-10:]:
                        ten_files.append(filelist[j])
                    if 'zer' in i[0][-10:]:
                        zer_files.append(filelist[j])

    return fif_files, six_files, sev_files, eig_files, nin_files, ten_files, zer_files


def find_both_file(id_path, full_path, mode):
    id_file = find_files(id_path, mode)
    full_file = find_files(full_path, mode)
    return id_file, full_file


def get_LogFile(path, file):
    # decade_path = path + '/' + decade + '/'
    file = open(path + file, 'r')
    if "id" in str(file):
        method = 'id'
    else:
        method = 'full'
    lines = file.read().splitlines()
    file.close()
    # print(method)
    return get_intervals_year(lines, method)


def get_intervals_year(lines, method):
    l = []
    year = []
    i = 0
    for line in lines:
        i += 1
        if i == 3:
            l = line
        if i == 9:
            new = line[line.rfind("timeslice") + 17:line.rfind("region") - 2]
            year.append(new)
    l = eval(l)
    l = [list(i) for i in l]
    [l[i].append(method) for i in range(len(l))]
    [l[i].append(i) for i in range(len(l))]  # add the fouth element "ranking"
    return l


def get_decade_data(id_path, full_path, id_, full_):
    # generate two big dictionary that contains decades: from fiftess to zeros
    id_ind = []
    full_ind = []
    empt = [[], [], [], [], [], [], []]
    years = ['fiftees', 'sixtees', 'seventees', 'eightees', 'ninetees', 'tens', 'zeroes']

    for num in range(len(years)):
        id_ind.append(years[num])
        full_ind.append(years[num])
        id_dict = dict(zip(id_ind, empt))
        full_dict = dict(zip(full_ind, empt))

    emp = [[], [], [], [], [], [], [], []]

    for i in range(len(id_dict)):
        id_index = []
        full_index = []
        for num in id_[i]:
            id_index.append(num)
            full_index.append(num)
            if i == 0:
                id_dict['fiftees'] = dict(zip(id_index, emp))
                full_dict['fiftees'] = dict(zip(full_index, emp))
            if i == 1:
                id_dict['sixtees'] = dict(zip(id_index, emp))
                full_dict['sixtees'] = dict(zip(full_index, emp))
            if i == 2:
                id_dict['seventees'] = dict(zip(id_index, emp))
                full_dict['seventees'] = dict(zip(full_index, emp))
            if i == 3:
                id_dict['eightees'] = dict(zip(id_index, emp))
                full_dict['eightees'] = dict(zip(full_index, emp))
            if i == 4:
                id_dict['ninetees'] = dict(zip(id_index, emp))
                full_dict['ninetees'] = dict(zip(full_index, emp))
            if i == 5:
                id_dict['tens'] = dict(zip(id_index, emp))
                full_dict['tens'] = dict(zip(full_index, emp))
            if i == 6:
                id_dict['zeroes'] = dict(zip(id_index, emp))
                full_dict['zeroes'] = dict(zip(full_index, emp))

    for i in id_dict:
        for j in id_dict[i]:
            id_dict[i][j] = get_LogFile(id_path + '/' + i + '/', j)
    for i in full_dict:
        for j in full_dict[i]:
            full_dict[i][j] = get_LogFile(full_path + '/' + i + '/', j)

    return id_dict, full_dict


def get_intervals_year(lines, method):
    l = []
    year = []
    timing = []
    i = 0
    for line in lines:
        i += 1
        if i == 3:
            l = line
        if i == 9:
            new = line[line.rfind("timeslice") + 17:line.rfind("region") - 2]
            year.append(new)
        if i == 5:
            # print(line)
            timing = float(line[20:])
    l = eval(l)
    l = [list(i) for i in l]
    [l[i].append(method) for i in range(len(l))]
    [l[i].append(i) for i in range(len(l))]  # add the fouth element "ranking"
    [l[i].append(timing) for i in range(len(l))]  # add the fifth element "timing"
    return l


def count_time_pre():
    id_path = 'G:/thesis-gaussian_id/MDI/libmaxdiv/maxdiv/output/final_experiments/time_only/full_region/ERA5_summed/decades/'
    full_path = 'G:/thesis-master/MDI/libmaxdiv/maxdiv/output/final_experiments/time_only/full_region/ERA5_summed/decades/'

    id_kl, full_kl = find_both_file(id_path, full_path, 'kl')
    id_dict_kl, full_dict_kl = get_decade_data(id_path, full_path, id_kl, full_kl)

    id_ce, full_ce = find_both_file(id_path, full_path, 'crossent')
    id_dict_ce, full_dict_ce = get_decade_data(id_path, full_path, id_ce, full_ce)

    return id_dict_kl, full_dict_kl, id_dict_ce, full_dict_ce


def count_time(dict_):
    time_ = 0
    for decade in dict_:
        for file in dict_[decade]:
            for i in dict_[decade][file]:
                time_ += i[5]
    return time_

def plot_running_time():
    id_dict_kl, full_dict_kl, id_dict_ce, full_dict_ce = count_time_pre()
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    names = ['ID method CE', 'ID method KL', 'FULL method CE', 'FULL method KL']

    timing = [count_time(id_dict_ce), count_time(id_dict_kl), count_time(full_dict_ce), count_time(full_dict_kl)]
    ax.bar(names, timing, color='green')
    ax.set_ylabel('Total running time (seconds)', fontsize=12)
    ax.set_xlabel('Methods', fontsize=12)
    ax.set_title('Running time of algorithms', fontsize=15)
    for p in ax.patches[0:]:
        h = p.get_height()
        x = p.get_x() + p.get_width() / 2.
        if h != 0:
            ax.annotate("%g" % p.get_height(), xy=(x, h), xytext=(0, 1),
                        textcoords="offset points", ha="center", va="bottom")
    plt.show()
    return


def check_sublist(l, check):
    if len(l) == 0:
        return False
    for ele in l:
        count = 0
        if check in ele:
            count += 1
    if count == 0:
        return False
    else:
        return True


def Iou_shaomu(R, B):
    start1 = R[0]
    len1 = R[1] - R[0]
    start2 = B[0]
    len2 = B[1] - B[0]
    intersection = max(0, min(start1 + len1, start2 + len2) - max(start1, start2))
    return float(intersection) / (len1 + len2 - intersection)


def get_result(R, B, name_R, name_B):
    """ get_Sets function is capable to find Subsets and Intersections(within the setting distance)
        Distance: hyper-parameter that determine whether R and B are overlapping
        Return: A list that contains the subsets and Intersections but without merging the same elements together,
    """
    # get_sub(equal situation included)
    n = []  # List store all overlapping outlier intervals
    m = []
    for i in R:
        for j in B:
            iou_score = Iou_shaomu(i, j)
            if iou_score > 0.:
                n.append([i, j, iou_score])
                # m.append(iou_score)

    info = str(name_R) + ' and ' + str(name_B)

    return [info, n, len(n)]  # n,m,best_subset,Acce_subset,best_inter,Acce_inter


def gen_eval_dict(id_path, id_):
    id_ind = []
    empt = [[], [], [], [], [], [], []]
    years = ['fiftees', 'sixtees', 'seventees', 'eightees', 'ninetees', 'tens', 'zeroes']
    emp = [[], [], [], [], [], [], [], []]

    for num in range(len(years)):
        id_ind.append(years[num])
        id_dict = dict(zip(id_ind, empt))

    for i in range(len(id_dict)):
        id_index = []
        for num in id_[i]:
            id_index.append(num)
            if i == 0:
                id_dict['fiftees'] = dict(zip(id_index, emp))
            if i == 1:
                id_dict['sixtees'] = dict(zip(id_index, emp))
            if i == 2:
                id_dict['seventees'] = dict(zip(id_index, emp))
            if i == 3:
                id_dict['eightees'] = dict(zip(id_index, emp))
            if i == 4:
                id_dict['ninetees'] = dict(zip(id_index, emp))
            if i == 5:
                id_dict['tens'] = dict(zip(id_index, emp))
            if i == 6:
                id_dict['zeroes'] = dict(zip(id_index, emp))

    return id_dict


def eval_and_result(id_dict, full_dict, eval_dict):
    eval_dict_re = copy.deepcopy(eval_dict)
    n = []
    m = []
    for i in id_dict:
        for j in id_dict[i]:
            eval_dict[i][j] = get_result(id_dict[i][j], full_dict[i][j], 'id+' + j, 'full+' + j)

    for i in id_dict:
        for j in id_dict[i]:
            eval_dict_re[i][j] = get_result(full_dict[i][j], id_dict[i][j], 'full+' + j, 'id+' + j)

    result = {
        'fiftees': {'id2full': [], 'full2id': []},
        'sixtees': {'id2full': [], 'full2id': []},
        'seventees': {'id2full': [], 'full2id': []},
        'eightees': {'id2full': [], 'full2id': []},
        'ninetees': {'id2full': [], 'full2id': []},
        'tens': {'id2full': [], 'full2id': []},
        'zeroes': {'id2full': [], 'full2id': []}}

    for i in eval_dict:
        decade_id2full = 0
        decade_full2id = 0
        for j in eval_dict[i]:
            decade_id2full += eval_dict[i][j][2]
            decade_full2id += eval_dict_re[i][j][2]

        result[i]['id2full'].append(decade_id2full)
        result[i]['full2id'].append(decade_full2id)
    return eval_dict, eval_dict_re, result


def run_eval(id_path, full_path, mode):
    id_, full_ = find_both_file(id_path, full_path, mode)
    id_dict, full_dict = get_decade_data(id_path, full_path, id_, full_)
    eval_dict = gen_eval_dict(id_path, id_)
    duplicate = gen_eval_dict(id_path, id_)
    # inter_result = gen_eval_dict(id_path, id_)
    eval_, eval_re, result = eval_and_result(id_dict, full_dict, eval_dict)

    return eval_, eval_re, result, duplicate

def run(mode):
    #print('Please input the path of the results of Identity covariance gaussian MDI: ')
    #path = str(input())
    #print(path)
    #print('Please input the path of the results of Full covariance gaussian MDI: ')
    #path1 = str(input())
    #print(path1)

    path = 'G:/thesis-gaussian_id/MDI/libmaxdiv/maxdiv/output/final_experiments/time_only/full_region/ERA5_summed/decades/'
    path1 = 'G:/thesis-master/MDI/libmaxdiv/maxdiv/output/final_experiments/time_only/full_region/ERA5_summed/decades/'

    eval_,eval_re,result,duplicate = run_eval(path, path1,mode)

    #print('-----------------------Eval_-----------------------')
    #print(eval_,'\n\n')

    #print('-----------------------eval_re-----------------------')
    #print(eval_re, '\n\n')

    #print('-----------------------Result_-----------------------')
    #print(result,'\n\n')

    #print('Now running the visualization')
    return eval_,eval_re,result,duplicate



#Remove non-monogamy
def get_dup(x):
    id_ = []
    for sub in x:
        id_.append(sub[0])

    dups = list()
    ind = list()
    index = -1
    for e in id_:
        index += 1
        if id_.count(e) > 1:
            ind.append(index)
            dups.append([e, index])

    for iou in ind:
        print(x[iou])

    del_2 = []
    if len(ind) == 2:
        # print(x[ind[0]][2])
        if x[ind[0]][2] > x[ind[1]][2]:
            del_2.append(x[ind[1]])

        else:
            del_2.append(x[ind[0]])
        # print('2:',del_2)

    del_4 = []
    if len(ind) == 4:
        # print(x[ind[0]][2])
        if x[ind[0]][2] > x[ind[1]][2]:
            del_4.append(x[ind[1]])
            # print('yes')
            # print(del_4)
            #
        else:
            del_4.append(x[ind[0]])
            # print('no')
            # print(del_4)

        if x[ind[2]][2] > x[ind[3]][2]:
            del_4.append(x[ind[3]])
            # print('yes')
            # print(del_4)

        else:
            del_4.append(x[ind[2]])
            # print('no')
            # print(del_4)

        # print('4:',del_4)

    del_6 = []
    if len(ind) == 6:
        # print(x[ind[0]][2])
        if x[ind[0]][2] > x[ind[1]][2]:
            del_6.append(x[ind[1]])


        else:
            del_6.append(x[ind[0]])

        if x[ind[2]][2] > x[ind[3]][2]:
            del_6.append(x[ind[3]])


        else:
            del_6.append(x[ind[2]])

        if x[ind[4]][2] > x[ind[5]][2]:
            del_6.append(x[ind[5]])


        else:
            del_6.append(x[ind[4]])

        # print('6:',del_6)

    return del_2, del_4, del_6


def remove_non_monogamy(eval_):
    test1 = copy.deepcopy(eval_)
    test = copy.deepcopy(eval_)
    _two = []
    _four = []
    _six = []
    for i in test:
        for j in test[i]:
            # print(i)
            # print(j)
            _two, _four, _six = get_dup(test[i][j][1])

            if len(_two) > 0:
                # print(_two)
                test1[i][j][1].remove(_two[0])
                test1[i][j][2] = test1[i][j][2] - 1
            elif len(_four) > 0:
                # print(_four)
                test1[i][j][1].remove(_four[0])
                test1[i][j][1].remove(_four[1])
                test1[i][j][2] = test1[i][j][2] - 2
            elif len(_six) > 0:
                # print(_six)
                test1[i][j][1].remove(_six[0])
                test1[i][j][1].remove(_six[1])
                test1[i][j][1].remove(_six[2])
                test1[i][j][2] = test1[i][j][2] - 3
            # print('-----------------------------')
    return test1


def return_new_result(eval_dict, eval_dict_re):
    result_remove = {
        'fiftees': {'id2full': [], 'full2id': []},
        'sixtees': {'id2full': [], 'full2id': []},
        'seventees': {'id2full': [], 'full2id': []},
        'eightees': {'id2full': [], 'full2id': []},
        'ninetees': {'id2full': [], 'full2id': []},
        'tens': {'id2full': [], 'full2id': []},
        'zeroes': {'id2full': [], 'full2id': []}}

    for i in eval_dict:
        decade_id2full = 0
        decade_full2id = 0
        for j in eval_dict[i]:
            decade_id2full += eval_dict[i][j][2]
            decade_full2id += eval_dict_re[i][j][2]

        result_remove[i]['id2full'].append(decade_id2full)
        result_remove[i]['full2id'].append(decade_full2id)
    return result_remove


def count_non_monogamy(l, duplicate):
    for i in l:
        for j in l[i]:
            count = []
            for k in l[i][j][1]:
                for number in range(50):
                    elm_count = k[0].count(number)
                    if elm_count != 0:
                        count.append(number)
            s = set()
            duplicates = set(x for x in count if x in s or s.add(x))
            duplicate[i][j] = list(duplicates)

    return duplicate


def plot_duplicate(duplicate):
    duplicate_sum = {'fiftees': [], 'sixtees': [], 'seventees': [], 'eightees': [], 'ninetees': [], 'tens': [],
                     'zeroes': []}
    for i in duplicate:
        sum_ = 0
        for j in duplicate[i]:
            # print(duplicate[i][j])
            # print(len(duplicate[i][j]))
            if len(duplicate[i][j]) != 0:
                # print(len(duplicate[i][j]))
                sum_ += len(duplicate[i][j])
            duplicate_sum[i] = sum_

    labels = ['50s', '60s', '70s', '80s', '90s', '00s', '10s']
    overall = list(duplicate_sum.values())

    x = np.arange(len(labels))  # the label locations
    width = 0.5  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x, overall, width, label='Num_non-monogamy')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Number of non-monogamy cases')
    # ax.set_title('Figure of monogamy analysis')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Decades')
    # ax.legend()
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.99),
    #      fancybox=True, shadow=True, ncol=5)
    plt.legend(bbox_to_anchor=(0, -0.17, 0.4, 0.5), loc="lower left",
               mode="expand", borderaxespad=0, ncol=3)
    plt.title('Figure of monogamy analysis')

    for p in ax.patches[0:]:
        h = p.get_height() - 0.05
        x = p.get_x() + p.get_width() / 3.
        if h != 0:
            ax.annotate("%g" % p.get_height(), xy=(x, h), xytext=(0, 1),
                        textcoords="offset points", ha="center", va="bottom")

    fig.tight_layout()

    plt.show()

    return


def run_remove_non_nomogamy(eval_, eval_re, duplicate, mode):
    eval_remove = remove_non_monogamy(eval_)

    eval_re_remove = remove_non_monogamy(eval_re)

    result_remove = return_new_result(eval_remove, eval_re_remove)

    if str(mode) == 'crossent':
        print('For Cross Entropy criterion')
    if str(mode) == 'kl':
        print('For Kullback-Leibler criterion')

    print('Before removing non-monogamy for id2full:')
    duplicates = count_non_monogamy(eval_, duplicate)
    plot_duplicate(duplicates)

    print('Before removing non-monogamy for full2id:')
    duplicates = count_non_monogamy(eval_re, duplicate)
    plot_duplicate(duplicates)

    print('After removing non-monogamy for id2full:')
    duplicates = count_non_monogamy(eval_remove, duplicate)
    plot_duplicate(duplicates)

    print('After removing non-monogamy for full2id:')
    duplicates = count_non_monogamy(eval_re_remove, duplicate)
    plot_duplicate(duplicates)

    return eval_remove, eval_re_remove, result_remove

#Visualization for result (Bar,pie charts)
def plot_Num_cases(result_, index):
    if str(index) == 'id2full':
        ind = 'id2full'
    else:
        ind = 'full2id'

    overall_50s = result_['fiftees'][ind][0]
    overall_60s = result_['sixtees'][ind][0]
    overall_70s = result_['seventees'][ind][0]
    overall_80s = result_['eightees'][ind][0]
    overall_90s = result_['ninetees'][ind][0]
    overall_00s = result_['zeroes'][ind][0]
    overall_10s = result_['tens'][ind][0]

    labels = ['50s', '60s', '70s', '80s', '90s', '00s', '10s']
    overall = [overall_50s, overall_60s, overall_70s, overall_80s, overall_90s, overall_00s, overall_10s]

    x = np.arange(len(labels))  # the label locations
    width = 0.5  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x, overall, width, label='overall')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Number of Matching events')
    ax.set_title('Figure of matching algorithm result', fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Decades')
    ax.legend()

    for p in ax.patches[0:]:
        h = p.get_height() - 4
        x = p.get_x() + p.get_width() / 2.
        if h != 0:
            ax.annotate("%g" % p.get_height(), xy=(x, h), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom")

    fig.tight_layout()

    plt.show()
    return


def plot_P_cases(result_, index):
    if str(index) == 'id2full':
        ind = 'id2full'
    else:
        ind = 'full2id'
    overall_50s = result_['fiftees'][ind][0] / 400.
    overall_60s = result_['sixtees'][ind][0] / 400.
    overall_70s = result_['seventees'][ind][0] / 400.
    overall_80s = result_['eightees'][ind][0] / 400.
    overall_90s = result_['ninetees'][ind][0] / 400.
    overall_00s = result_['zeroes'][ind][0] / 400.
    overall_10s = result_['tens'][ind][0] / 400.

    def dec(a):
        return "%.2f" % a

    overall_50s_dec = dec(overall_50s)
    overall_60s_dec = dec(overall_60s)
    overall_70s_dec = dec(overall_70s)
    overall_80s_dec = dec(overall_80s)
    overall_90s_dec = dec(overall_90s)
    overall_00s_dec = dec(overall_00s)
    overall_10s_dec = dec(overall_10s)

    labels = ['50s', '60s', '70s', '80s', '90s', '00s', '10s']
    overall = [overall_50s, overall_60s, overall_70s, overall_80s, overall_90s, overall_00s, overall_10s]

    x = np.arange(len(labels))  # the label locations
    width = 0.6  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x, overall, width, label='overall', color='orange')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Percentage of Matching events')
    ax.set_title('')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title('Figure of matching algorithm result', fontsize=15)
    ax.set_xlabel('Decades')
    ax.legend()

    for p in ax.patches[0:]:
        h = p.get_height() - 0.01
        x = p.get_x() + p.get_width() / 2.
        if h != 0:
            ax.annotate("%g" % p.get_height(), xy=(x, h), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom")

    fig.tight_layout()

    plt.show()
    return


def plot_(result_, index):
    if str(index) == 'id2full':
        ind = 'id2full'
    else:
        ind = 'full2id'

    labels = ['Percentage of Matching events', 'Percentage of Non-matching events']

    overall = result_['fiftees'][ind][0] + result_['sixtees'][ind][0] + result_['seventees'][ind][0] + \
              result_['eightees'][ind][0] + result_['ninetees'][ind][0] + result_['zeroes'][ind][0] + \
              result_['tens'][ind][0]
    non = 400 * 7 - overall
    sizes = [overall, non]
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

    patches, texts, percentage = plt.pie(sizes, colors=colors, startangle=90, autopct='%1.1f%%')
    plt.legend(patches, labels, loc="best")
    # Set aspect ratio to be equal so that pie is drawn as a circle.
    plt.title('Figure of {} matching algorithm result'.format(ind), fontsize=15)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    return


def plot_visual(result_, result_remove, mode):
    if str(mode) == 'crossent':
        print('For Cross Entropy criterion')
    if str(mode) == 'kl':
        print('For Kullback-Leibler criterion')

    print('Visualization for result before removing non-monogamy')
    print('Visualization for id2full result')
    plot_Num_cases(result_, 'id2full')
    plot_P_cases(result_, 'id2full')

    print('Visualization for full2id result')
    plot_Num_cases(result_, 'full2id')
    plot_P_cases(result_, 'full2id')

    print('Visualization for result after removing non-monogamy')
    print('Visualization for id2full result')
    plot_Num_cases(result_remove, 'id2full')
    plot_P_cases(result_remove, 'id2full')

    print('Visualization for full2id result')
    plot_Num_cases(result_remove, 'full2id')
    plot_P_cases(result_remove, 'full2id')

    print('Visualization before removing non_monogamy cases')
    plot_(result_, 'id2full')
    plot_(result_, 'full2id')

    print('Visualization after removing non_monogamy cases')
    plot_(result_remove, 'id2full')
    plot_(result_remove, 'full2id')

    return



#ranking correlation
def plot_4_(x, y, rank_dis, l, decade):
    a_, b_, c_, d_ = l

    #color_map = plt.cm.get_cmap('Greens')
    #reversed_color_map = color_map.reversed()

    fig = plt.figure(figsize=(6, 4))
    fig.suptitle('Visualization for {}'.format(decade), fontsize=15)

    sub1 = fig.add_subplot(221)  # instead of plt.subplot(2, 2, 1)
    # sub1.set_title('The function f') # non OOP: plt.title('The function f')
    ax = sub1.scatter(x[a_], y[a_], s=200, c='green')
    # sub1.set_xlabel('Rank of ID results')
    # sub1.set_ylabel('Rank of Full results')

    sub2 = fig.add_subplot(222)
    # sub2.set_title('The function f') # non OOP: plt.title('The function f')
    sub2.scatter(x[b_], y[b_], s=200, c='green')
    # sub2.set_xlabel('Rank of ID results')
    # sub2.set_ylabel('Rank of Full results')

    sub3 = fig.add_subplot(223)
    # sub3.set_title('The function f') # non OOP: plt.title('The function f')
    sub3.scatter(x[c_], y[c_], s=200, c='green')
    # sub3.set_xlabel('Rank of ID results')
    # sub3.set_ylabel('Rank of Full results')

    sub4 = fig.add_subplot(224)
    # sub4.set_title('The function f') # non OOP: plt.title('The function f')
    sub4.scatter(x[d_], y[d_], s=200, c='green')
    # sub4.set_xlabel('Rank of ID results')
    # sub4.set_ylabel('Rank of Full results')

    fig.text(0.5, 0.001, "Rank of ID method results", ha="center", va="center", fontsize=13.5)
    fig.text(0.005, 0.5, "Rank of Full method results", ha="center", va="center", rotation=90, fontsize=13.5)

    # cbar_ax = fig.add_axes([0.999, 0.1, 0.05, 0.75])
    # fig.colorbar(ax,cbar_ax)

    # plt.tight_layout()
    plt.show()
    return


def plot_rank(duplicate, eval_):
    y = copy.deepcopy(duplicate)
    x = copy.deepcopy(y)
    rank_dis = copy.deepcopy(y)
    iou = copy.deepcopy(y)

    for decade in eval_:
        for item in eval_[decade]:

            full_y = []
            id_x = []
            rank_dis_ = []
            iou_score = []

            for i in eval_[decade][item][1]:
                sc = 0
                full_y.append(i[0][4])
                id_x.append(i[1][4])
                rank_dis_.append(abs(i[1][4] - i[0][4]))
                sc = 30 * i[2]
                iou_score.append(sc)

            y[decade][item] = full_y
            x[decade][item] = id_x
            rank_dis[decade][item] = rank_dis_
            iou[decade][item] = iou_score

    for decade in rank_dis:
        target = list(rank_dis[decade].keys())
        plot_4_(x[decade], y[decade], rank_dis[decade], target[:4], decade)
        plot_4_(x[decade], y[decade], rank_dis[decade], target[-4:], decade)

    return id_x, full_y, rank_dis_



#Merge 8 files approach
def merge_and_rerank(decade):
    de = []
    ind = -1
    for i in decade:
        for j in i:
            ind += 1
            j[4] = ind
            de.append(j)
    return de


def get_merged_decades_data(id_dict):
    fiftees_all = []
    sixtees_all = []
    seventees_all = []
    eightees_all = []
    ninetees_all = []
    tens_all = []
    zeroes_all = []

    for i in id_dict:
        for j in id_dict[i]:
            if i == 'fiftees':
                fiftees_all.append(id_dict[i][j])
            if i == 'sixtees':
                sixtees_all.append(id_dict[i][j])
            if i == 'seventees':
                seventees_all.append(id_dict[i][j])
            if i == 'eightees':
                eightees_all.append(id_dict[i][j])
            if i == 'ninetees':
                ninetees_all.append(id_dict[i][j])
            if i == 'tens':
                tens_all.append(id_dict[i][j])
            if i == 'zeroes':
                zeroes_all.append(id_dict[i][j])

    return [merge_and_rerank(fiftees_all), merge_and_rerank(sixtees_all), merge_and_rerank(seventees_all),
            merge_and_rerank(eightees_all), merge_and_rerank(ninetees_all), merge_and_rerank(tens_all),
            merge_and_rerank(zeroes_all)]


def merge_pre(mode):
    path = 'G:/thesis-gaussian_id/MDI/libmaxdiv/maxdiv/output/final_experiments/time_only/full_region/ERA5_summed/decades/'
    path1 = 'G:/thesis-master/MDI/libmaxdiv/maxdiv/output/final_experiments/time_only/full_region/ERA5_summed/decades/'
    id_, full_ = find_both_file(path, path1, mode)
    id_dict, full_dict = get_decade_data(path, path1, id_, full_)

    id_ = get_merged_decades_data(id_dict)
    full_ = get_merged_decades_data(full_dict)
    return id_, full_


def merge_and_rerank(decade):
    de = []
    ind = -1
    for i in decade:
        for j in i:
            ind += 1
            j[4] = ind
            de.append(j)
    return de


def get_merged_decades_data(id_dict):
    fiftees_all = []
    sixtees_all = []
    seventees_all = []
    eightees_all = []
    ninetees_all = []
    tens_all = []
    zeroes_all = []

    for i in id_dict:
        for j in id_dict[i]:
            if i == 'fiftees':
                fiftees_all.append(id_dict[i][j])
            if i == 'sixtees':
                sixtees_all.append(id_dict[i][j])
            if i == 'seventees':
                seventees_all.append(id_dict[i][j])
            if i == 'eightees':
                eightees_all.append(id_dict[i][j])
            if i == 'ninetees':
                ninetees_all.append(id_dict[i][j])
            if i == 'tens':
                tens_all.append(id_dict[i][j])
            if i == 'zeroes':
                zeroes_all.append(id_dict[i][j])

    return [merge_and_rerank(fiftees_all), merge_and_rerank(sixtees_all), merge_and_rerank(seventees_all),
            merge_and_rerank(eightees_all), merge_and_rerank(ninetees_all), merge_and_rerank(tens_all),
            merge_and_rerank(zeroes_all)]


# merge them together instead of list of lists form
def merge_whole(decade):
    de = []

    for i in decade:
        for j in i:
            de.append(j)
    return de


def Iou_shaomu(R, B):
    start1 = R[0]
    len1 = R[1] - R[0]
    start2 = B[0]
    len2 = B[1] - B[0]
    intersection = max(0, min(start1 + len1, start2 + len2) - max(start1, start2))
    return float(intersection) / (len1 + len2 - intersection)


def iou_filter_perdecade(id_ts_fif):
    Iou_ = []
    # after = []
    for i in range(len(id_ts_fif)):
        for j in range(len(id_ts_fif)):
            if id_ts_fif[i] != id_ts_fif[j]:
                iou = Iou_shaomu(id_ts_fif[i], id_ts_fif[j])
                if iou > 0.:
                    if ([id_ts_fif[i], id_ts_fif[j], iou] and [id_ts_fif[j], id_ts_fif[i], iou] not in Iou_) == True:
                        Iou_.append([id_ts_fif[i], id_ts_fif[j], iou])

                        # if id_ts_fif[i] not in after:
                        #    if id_ts_fif[j] not in after:
                        #        if id_ts_fif[i][2] > id_ts_fif[j][2]:
                        #            after.append(id_ts_fif[i])
                        #        else:
                        #            after.append(id_ts_fif[j])

    a = {}
    for i in Iou_:
        key = str(i[0])
        a[key] = []

    for i in Iou_:
        key = str(i[0])
        if i[0] not in a[key]:
            a[key].append(i[0])

    for i in Iou_:
        key = str(i[0])
        if i[1] not in a[key]:
            a[key].append(i[1])

    for key in a:
        a[key] = sorted(a[key], key=lambda x: x[2], reverse=True)

    drop = []
    for key in a:
        for item in a[key][1:]:
            if item not in drop:
                drop.append(item)

    for i in drop:
        id_ts_fif.remove(i)
    #26 April updated version
    id_ts_fif = sorted(id_ts_fif, key=lambda x: x[2], reverse=True)
    for i in range(len(id_ts_fif)):
        id_ts_fif[i][4] = i
    return id_ts_fif


def iou_filter(whole):
    whole_new = [[], [], [], [], [], [], []]

    id_ce_fif = copy.deepcopy(whole[0:400])
    id_ce_six = copy.deepcopy(whole[400:800])
    id_ce_sev = copy.deepcopy(whole[800:1200])
    id_ce_eig = copy.deepcopy(whole[1200:1600])
    id_ce_nin = copy.deepcopy(whole[1600:2000])
    id_ce_zer = copy.deepcopy(whole[2000:2400])
    id_ce_ten = copy.deepcopy(whole[2400:2800])

    id_ce_fif = iou_filter_perdecade(id_ce_fif)
    id_ce_six = iou_filter_perdecade(id_ce_six)
    id_ce_sev = iou_filter_perdecade(id_ce_sev)
    id_ce_eig = iou_filter_perdecade(id_ce_eig)
    id_ce_nin = iou_filter_perdecade(id_ce_nin)
    id_ce_zer = iou_filter_perdecade(id_ce_zer)
    id_ce_ten = iou_filter_perdecade(id_ce_ten)

    whole_new[0] = id_ce_fif
    whole_new[1] = id_ce_six
    whole_new[2] = id_ce_sev
    whole_new[3] = id_ce_eig
    whole_new[4] = id_ce_nin
    whole_new[5] = id_ce_zer
    whole_new[6] = id_ce_ten

    return whole_new

def np_rank(merge_decade):
    id_iou = {'fiftees': [], 'sixtees': [], 'seventees': [], 'eightees': [], 'ninetees': [], 'zeroes': [], 'tens': []}
    full_iou = {'fiftees': [], 'sixtees': [], 'seventees': [], 'eightees': [], 'ninetees': [], 'zeroes': [], 'tens': []}
    for decade in merge_decade:
        for i in merge_decade[decade]:
            id_iou[decade].append(i[0][4])
            full_iou[decade].append(i[1][4])

    np_id_iou = []
    np_full_iou = []
    for decade in id_iou:
        np_id_iou.append(np.array(id_iou[decade]))
    for decade in full_iou:
        np_full_iou.append(np.array(full_iou[decade]))
    return np_id_iou,np_full_iou

def get_whole_result(id_, full_, iou_threshold):
    cap_list = []
    for i in id_:
        for j in full_:
            score = Iou_shaomu(i, j)
            if score > iou_threshold:
                cap_list.append([i, j, score])

    return cap_list


def merge_process_decade(capzero):
    compare_t = copy.deepcopy(capzero)
    comt = {}

    for i in compare_t:
        name = str(i[0][0]) + ' ' + str(i[0][1])
        comt[name] = []

    for i in compare_t:
        for j in capzero:
            if i != j:
                if i[0] == j[0]:
                    name = str(i[0][0]) + ' ' + str(i[0][1])
                    if j not in comt[name]:
                        comt[name].append(j)

    pop = []
    new = []
    for key in comt:
        comt[key] = sorted(comt[key], key=lambda x: x[1][2], reverse=True)
        if len(comt[key]) > 0:
            new.append(comt[key][0])
    #        comt[key] = comt[key][0]
    #    if len(comt[key])==0:
    #        pop.append(key)
    # for i in pop:
    #    comt.pop(i)

    return new


def plot_mer_per_de(merge_decade, mode, reverse, criterion,s_coef, s_p, p_coef, p_p):
    import matplotlib as mpl
    cmap = mpl.cm.cool

    x = {'fiftees': [], 'sixtees': [], 'seventees': [], 'eightees': [], 'ninetees': [], 'zeroes': [], 'tens': []}
    y = {'fiftees': [], 'sixtees': [], 'seventees': [], 'eightees': [], 'ninetees': [], 'zeroes': [], 'tens': []}
    rank_dis = {'fiftees': [], 'sixtees': [], 'seventees': [], 'eightees': [], 'ninetees': [], 'zeroes': [], 'tens': []}
    iou_score = {'fiftees': [], 'sixtees': [], 'seventees': [], 'eightees': [], 'ninetees': [], 'zeroes': [],
                 'tens': []}
    iou_score_mf = {'fiftees': [], 'sixtees': [], 'seventees': [], 'eightees': [], 'ninetees': [], 'zeroes': [],
                    'tens': []}

    for decade in merge_decade:
        for i in merge_decade[decade]:
            x[decade].append(i[0][4])
            y[decade].append(i[1][4])
            rank_dis[decade].append(abs(i[1][4] - i[0][4]))
            iou_score[decade].append(i[2])
            iou_score_mf[decade].append(i[2] * 50)

    if reverse == True:
        xlabel = 'Rank of ID method results'
        ylabel = 'Rank of Full method results'
    else:
        ylabel = 'Rank of ID method results'
        xlabel = 'Rank of Full method results'

    if criterion == 'ce':
        cri_ = 'For Cross Entropy criterion'

    if criterion == 'kl':
        cri_ = 'For Kullback-Leibler criterion'

    if str(mode) == 'default':
        fig = plt.figure(figsize=(15, 8))
        plt.subplots_adjust(hspace = 0.5)
        fig.suptitle('Visualization for Merged decades' + '(' + cri_ + ')', fontsize=15)

        sub1 = fig.add_subplot(241)
        sub1.scatter(x['fiftees'], y['fiftees'], s=40, cmap='Greens')
        per_s1 = '{:.2%}'.format(s_coef[0])
        s_p1 = '{:.3f}'.format(s_p[0])
        per_p1 = '{:.2%}'.format(p_coef[0])
        p_p1 = '{:.3f}'.format(p_p[0])
        sub1.set_xlabel('\nSpearmans coef: {}, P-value: {}\n\nPearson coef: {}, P-value: {}'.format(per_s1,s_p1,per_p1,p_p1))


        sub2 = fig.add_subplot(242)
        sub2.scatter(x['sixtees'], y['sixtees'], s=40, cmap='Greens')
        per_s2 = '{:.2%}'.format(s_coef[1])
        s_p2 = '{:.3f}'.format(s_p[1])
        per_p2 = '{:.2%}'.format(p_coef[1])
        p_p2 = '{:.3f}'.format(p_p[1])
        sub2.set_xlabel('\nSpearmans coef: {}, P-value: {}\n\nPearson coef: {}, P-value: {}'.format(per_s2,s_p2,per_p2,p_p2))

        sub3 = fig.add_subplot(243)
        sub3.scatter(x['seventees'], y['seventees'], s=40, cmap='Greens')
        per_s3 = '{:.2%}'.format(s_coef[2])
        s_p3 = '{:.3f}'.format(s_p[2])
        per_p3 = '{:.2%}'.format(p_coef[2])
        p_p3 = '{:.3f}'.format(p_p[2])
        sub3.set_xlabel('\nSpearmans coef: {}, P-value: {}\n\nPearson coef: {}, P-value: {}'.format(per_s3,s_p3,per_p3,p_p3))

        sub4 = fig.add_subplot(244)
        sub4.scatter(x['eightees'], y['eightees'], s=40, cmap='Greens')
        per_s4 = '{:.2%}'.format(s_coef[3])
        s_p4 = '{:.3f}'.format(s_p[3])
        per_p4 = '{:.2%}'.format(p_coef[3])
        p_p4 = '{:.3f}'.format(p_p[3])
        sub4.set_xlabel('\nSpearmans coef: {}, P-value: {}\n\nPearson coef: {}, P-value: {}'.format(per_s4,s_p4,per_p4,p_p4))

        sub5 = fig.add_subplot(245)
        sub5.scatter(x['ninetees'], y['ninetees'], s=40, cmap='Greens')
        per_s5 = '{:.2%}'.format(s_coef[4])
        s_p5 = '{:.3f}'.format(s_p[4])
        per_p5 = '{:.2%}'.format(p_coef[4])
        p_p5 = '{:.3f}'.format(p_p[4])
        sub5.set_xlabel('\nSpearmans coef: {}, P-value: {}\n\nPearson coef: {}, P-value: {}'.format(per_s5,s_p5,per_p5,p_p5))

        sub6 = fig.add_subplot(246)
        sub6.scatter(x['zeroes'], y['zeroes'], s=40, cmap='Greens')
        per_s6 = '{:.2%}'.format(s_coef[5])
        s_p6 = '{:.3f}'.format(s_p[5])
        per_p6 = '{:.2%}'.format(p_coef[5])
        p_p6 = '{:.3f}'.format(p_p[5])
        sub6.set_xlabel('\nSpearmans coef: {}, P-value: {}\n\nPearson coef: {}, P-value: {}'.format(per_s6,s_p6,per_p6,p_p6))

        sub7 = fig.add_subplot(247)
        sub7.scatter(x['tens'], y['tens'], s=40, cmap='Greens')
        per_s7 = '{:.2%}'.format(s_coef[6])
        s_p7 = '{:.3f}'.format(s_p[6])
        per_p7 = '{:.2%}'.format(p_coef[6])
        p_p7 = '{:.3f}'.format(p_p[6])
        sub7.set_xlabel('\nSpearmans coef: {}, P-value: {}\n\nPearson coef: {}, P-value: {}'.format(per_s7,s_p7,per_p7,p_p7))

        avg_sp = '{:.2%}'.format(np.mean(s_coef))
        avg_pearson = '{:.2%}'.format(np.mean(p_coef))

        fig.text(0.5, -0.05, xlabel, ha="center", va="center", fontsize=13.5)
        fig.text(0.08, 0.5, ylabel, ha="center", va="center", rotation=90, fontsize=13.5)
        fig.text(0.8, 0.25,'Averaged Spearmans coef: {}\n\nAveraged Pearson coef: {}'.format(avg_sp,avg_pearson), ha="center", va="center", fontsize=11)

    if str(mode) == 'color_iou':
        fig = plt.figure(figsize=(15, 8))
        plt.subplots_adjust(hspace=0.5)
        fig.suptitle('Visualization for Merged decades' + '(' + cri_ + ')', fontsize=15)

        sub1 = fig.add_subplot(241)
        sub1.scatter(x['fiftees'], y['fiftees'], s=40, c=iou_score['fiftees'], cmap=cmap)
        per_s1 = '{:.2%}'.format(s_coef[0])
        s_p1 = '{:.3f}'.format(s_p[0])
        per_p1 = '{:.2%}'.format(p_coef[0])
        p_p1 = '{:.3f}'.format(p_p[0])
        sub1.set_xlabel('\nSpearmans coef: {}, P-value: {}\n\nPearson coef: {}, P-value: {}'.format(per_s1, s_p1, per_p1, p_p1))


        sub2 = fig.add_subplot(242)
        sub2.scatter(x['sixtees'], y['sixtees'], s=40, c=iou_score['sixtees'], cmap=cmap)
        per_s2 = '{:.2%}'.format(s_coef[1])
        s_p2 = '{:.3f}'.format(s_p[1])
        per_p2 = '{:.2%}'.format(p_coef[1])
        p_p2 = '{:.3f}'.format(p_p[1])
        sub2.set_xlabel('\nSpearmans coef: {}, P-value: {}\n\nPearson coef: {}, P-value: {}'.format(per_s2, s_p2, per_p2, p_p2))

        sub3 = fig.add_subplot(243)
        sub3.scatter(x['seventees'], y['seventees'], s=40, c=iou_score['seventees'], cmap=cmap)
        per_s3 = '{:.2%}'.format(s_coef[2])
        s_p3 = '{:.3f}'.format(s_p[2])
        per_p3 = '{:.2%}'.format(p_coef[2])
        p_p3 = '{:.3f}'.format(p_p[2])
        sub3.set_xlabel('\nSpearmans coef: {}, P-value: {}\n\nPearson coef: {}, P-value: {}'.format(per_s3, s_p3, per_p3, p_p3))

        sub4 = fig.add_subplot(244)
        sub4.scatter(x['eightees'], y['eightees'], s=40, c=iou_score['eightees'], cmap=cmap)
        per_s4 = '{:.2%}'.format(s_coef[3])
        s_p4 = '{:.3f}'.format(s_p[3])
        per_p4 = '{:.2%}'.format(p_coef[3])
        p_p4 = '{:.3f}'.format(p_p[3])
        sub4.set_xlabel('\nSpearmans coef: {}, P-value: {}\n\nPearson coef: {}, P-value: {}'.format(per_s4, s_p4, per_p4, p_p4))

        sub5 = fig.add_subplot(245)
        sub5.scatter(x['ninetees'], y['ninetees'], s=40, c=iou_score['ninetees'], cmap=cmap)
        per_s5 = '{:.2%}'.format(s_coef[4])
        s_p5 = '{:.3f}'.format(s_p[4])
        per_p5 = '{:.2%}'.format(p_coef[4])
        p_p5 = '{:.3f}'.format(p_p[4])
        sub5.set_xlabel('\nSpearmans coef: {}, P-value: {}\n\nPearson coef: {}, P-value: {}'.format(per_s5, s_p5, per_p5, p_p5))

        sub6 = fig.add_subplot(246)
        sub6.scatter(x['zeroes'], y['zeroes'], s=40, c=iou_score['zeroes'], cmap=cmap)
        per_s6 = '{:.2%}'.format(s_coef[5])
        s_p6 = '{:.3f}'.format(s_p[5])
        per_p6 = '{:.2%}'.format(p_coef[5])
        p_p6 = '{:.3f}'.format(p_p[5])
        sub6.set_xlabel('\nSpearmans coef: {}, P-value: {}\n\nPearson coef: {}, P-value: {}'.format(per_s6, s_p6, per_p6, p_p6))

        sub7 = fig.add_subplot(247)
        ax = sub7.scatter(x['tens'], y['tens'], s=40, c=iou_score['tens'], cmap=cmap)
        per_s7 = '{:.2%}'.format(s_coef[6])
        s_p7 = '{:.3f}'.format(s_p[6])
        per_p7 = '{:.2%}'.format(p_coef[6])
        p_p7 = '{:.3f}'.format(p_p[6])
        sub7.set_xlabel('\nSpearmans coef: {}, P-value: {}\n\nPearson coef: {}, P-value: {}'.format(per_s7, s_p7, per_p7, p_p7))

        cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.75])
        cb = fig.colorbar(ax, cbar_ax)
        cb.set_label('IoU score')

        avg_sp = '{:.2%}'.format(np.mean(s_coef))
        avg_pearson = '{:.2%}'.format(np.mean(p_coef))

        fig.text(0.5, -0.05, xlabel, ha="center", va="center", fontsize=13.5)
        fig.text(0.08, 0.5, ylabel, ha="center", va="center", rotation=90, fontsize=13.5)
        fig.text(0.8, 0.25, 'Averaged Spearmans coef: {}\n\nAveraged Pearson coef: {}'.format(avg_sp, avg_pearson),
                 ha="center", va="center", fontsize=11)

    if str(mode) == 'size_iou':
        fig = plt.figure(figsize=(15, 6))
        plt.subplots_adjust(hspace=0.6)
        fig.suptitle('Visualization for Merged decades' + '(' + cri_ + ')', fontsize=15)

        sub1 = fig.add_subplot(241)
        sub1.scatter(x['fiftees'], y['fiftees'], s=iou_score_mf['fiftees'], cmap='Greens')
        per_s1 = '{:.2%}'.format(s_coef[0])
        s_p1 = '{:.3f}'.format(s_p[0])
        per_p1 = '{:.2%}'.format(p_coef[0])
        p_p1 = '{:.3f}'.format(p_p[0])

        sub1.set_xlabel('\nSpearmans coef: {}, P-value: {}\n\nPearson coef: {}, P-value: {}'.format(per_s1, s_p1, per_p1, p_p1))

        sub2 = fig.add_subplot(242)
        sub2.scatter(x['sixtees'], y['sixtees'], s=iou_score_mf['sixtees'], cmap='Greens')
        per_s2 = '{:.2%}'.format(s_coef[1])
        s_p2 = '{:.3f}'.format(s_p[1])
        per_p2 = '{:.2%}'.format(p_coef[1])
        p_p2 = '{:.3f}'.format(p_p[1])
        sub2.set_xlabel('\nSpearmans coef: {}, P-value: {}\n\nPearson coef: {}, P-value: {}'.format(per_s2, s_p2, per_p2, p_p2))

        sub3 = fig.add_subplot(243)
        sub3.scatter(x['seventees'], y['seventees'], s=iou_score_mf['seventees'], cmap='Greens')
        per_s3 = '{:.2%}'.format(s_coef[2])
        s_p3 = '{:.3f}'.format(s_p[2])
        per_p3 = '{:.2%}'.format(p_coef[2])
        p_p3 = '{:.3f}'.format(p_p[2])
        sub3.set_xlabel('\nSpearmans coef: {}, P-value: {}\n\nPearson coef: {}, P-value: {}'.format(per_s3, s_p3, per_p3, p_p3))

        sub4 = fig.add_subplot(244)
        sub4.scatter(x['eightees'], y['eightees'], s=iou_score_mf['eightees'], cmap='Greens')
        per_s4 = '{:.2%}'.format(s_coef[3])
        s_p4 = '{:.3f}'.format(s_p[3])
        per_p4 = '{:.2%}'.format(p_coef[3])
        p_p4 = '{:.3f}'.format(p_p[3])
        sub4.set_xlabel('\nSpearmans coef: {}, P-value: {}\n\nPearson coef: {}, P-value: {}'.format(per_s4, s_p4, per_p4, p_p4))

        sub5 = fig.add_subplot(245)
        sub5.scatter(x['ninetees'], y['ninetees'], s=iou_score_mf['ninetees'], cmap='Greens')
        per_s5 = '{:.2%}'.format(s_coef[4])
        s_p5 = '{:.3f}'.format(s_p[4])
        per_p5 = '{:.2%}'.format(p_coef[4])
        p_p5 = '{:.3f}'.format(p_p[4])
        sub5.set_xlabel('\nSpearmans coef: {}, P-value: {}\n\nPearson coef: {}, P-value: {}'.format(per_s5, s_p5, per_p5, p_p5))

        sub6 = fig.add_subplot(246)
        sub6.scatter(x['zeroes'], y['zeroes'], s=iou_score_mf['zeroes'], cmap='Greens')
        per_s6 = '{:.2%}'.format(s_coef[5])
        s_p6 = '{:.3f}'.format(s_p[5])
        per_p6 = '{:.2%}'.format(p_coef[5])
        p_p6 = '{:.3f}'.format(p_p[5])
        sub6.set_xlabel('\nSpearmans coef: {}, P-value: {}\n\nPearson coef: {}, P-value: {}'.format(per_s6, s_p6, per_p6, p_p6))

        sub7 = fig.add_subplot(247)
        sub7.scatter(x['tens'], y['tens'], s=iou_score_mf['tens'], cmap='Greens')
        per_s7 = '{:.2%}'.format(s_coef[6])
        s_p7 = '{:.3f}'.format(s_p[6])
        per_p7 = '{:.2%}'.format(p_coef[6])
        p_p7 = '{:.3f}'.format(p_p[6])
        sub7.set_xlabel('\nSpearmans coef: {}, P-value: {}\n\nPearson coef: {}, P-value: {}'.format(per_s7, s_p7, per_p7, p_p7))

        fig.text(0.5, 0.01, xlabel, ha="center", va="center", fontsize=13.5)
        fig.text(0.08, 0.5, ylabel, ha="center", va="center", rotation=90, fontsize=13.5)
        fig.text(0.8, 0.3, "Size determined by IoU scores", ha="center", va="center", fontsize=12)

    return

def np_iou(merge_decade,iou_threshold):
    iou = {'fiftees': [], 'sixtees': [], 'seventees': [], 'eightees': [], 'ninetees': [], 'zeroes': [], 'tens': []}
    for decade in merge_decade:
        for i in merge_decade[decade]:
            if  i[-1] >= iou_threshold:
                iou[decade].append(i[-1])

    np_iou = []
    for decade in iou:
        np_iou.append(np.array(iou[decade]))
    return np_iou


def spearman_pearson(ranking_id, ranking_full):
    s_coef = []
    s_p = []
    p_coef = []
    p_p = []
    spear_coef = 0
    spear_p = 0
    pearson_coef = 0
    pearson_p = 0

    for i in range(len(ranking_id)):
        # calculate spearman's correlation
        spear_coef, spear_p = spearmanr(ranking_id[i], ranking_full[i])
        s_coef.append(spear_coef)
        s_p.append(spear_p)
        # calculate spearman's correlation
        pearson_coef, pearson_p = pearsonr(ranking_id[i], ranking_full[i])
        p_coef.append(pearson_coef)
        p_p.append(pearson_p)
    return s_coef, s_p, p_coef, p_p

def plot_mean_median(np_iou):
    mean = 0
    for i in np_iou:
        mean += np.mean(i)
    mean = mean / 7.

    median = 0
    for i in np_iou:
        median += np.median(i)
    median = median / 7.

    return mean, median

def plot_single_volin_box(np_iou, criterion):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))

    if criterion == 'ce':
        cri_ = 'Cross Entropy criterion'

    if criterion == 'kl':
        cri_ = 'Kullback-Leibler criterion'

    fig.suptitle('Distribution of overlaps' + '(' + cri_ + ') \n', fontsize=15)

    mean,median = plot_mean_median(np_iou)

    # plot violin plot
    axs[0].violinplot(np_iou,
                      showmeans=True,
                      showmedians=False)

    axs[0].plot((0.7, 7.3), (mean, mean), linestyle='--', color='mediumorchid')
    axs[0].legend(['Avg.Mean'], loc=2)

    mean_p = float("{:.2f}".format(mean))

    axs[0].text(0.2, mean , mean_p,color='mediumorchid')

    axs[0].set_title('show Means ')

    # plot box plot
    axs[1].plot((0.7, 7.3), (median, median), linestyle='--', color='darkgreen')
    median_p = float("{:.2f}".format(median))

    axs[1].text(0.03, median , median_p, color='darkgreen')

    axs[1].legend(['Avg.Median'], loc=2)
    axs[1].boxplot(np_iou)
    axs[1].set_title('show Medians')



    # adding horizontal grid lines
    for ax in axs:
        ax.yaxis.grid(True)
        ax.set_xticks([y + 1 for y in range(len(np_iou))])
        ax.set_xlabel('Decades')
        ax.set_ylabel('Observed IoU values')

    # add x-tick labels
    plt.setp(axs, xticks=[y + 1 for y in range(len(np_iou))],
             xticklabels=['50s', '60s', '70s', '80s', '90s', '00s', '10s'])
    plt.show()
    return

def visual_md(id_, full_, method, iou_threshold, cri):
    merge_decade = {'fiftees': [], 'sixtees': [], 'seventees': [], 'eightees': [], 'ninetees': [], 'zeroes': [],
                    'tens': []}
    count_ = -1
    for i in merge_decade:
        count_ += 1
        if str(method) == 'id2full':
            cap = get_whole_result(id_[count_], full_[count_], iou_threshold)
            merge_decade[i] = cap
            # merge_decade[i] = merge_process_decade(cap)
        if str(method) == 'full2id':
            cap = get_whole_result(full_[count_], id_[count_], iou_threshold)
            merge_decade[i] = cap
            # merge_decade[i] = merge_process_decade(cap)

    id_, full_ = np_rank(merge_decade)
    s_coef, s_p, p_coef, p_p = spearman_pearson(id_, full_)

    if str(method) == 'id2full':
        plot_mer_per_de(merge_decade, 'default', False, cri,s_coef, s_p, p_coef, p_p)
        # plot_mer_per_de(merge_decade, 'size_iou', False, cri)
        plot_mer_per_de(merge_decade, 'color_iou', False, cri,s_coef, s_p, p_coef, p_p)

    if str(method) == 'full2id':
        plot_mer_per_de(merge_decade, 'default', True, cri,s_coef, s_p, p_coef, p_p)
        # plot_mer_per_de(merge_decade, 'size_iou', True, cri)
        plot_mer_per_de(merge_decade, 'color_iou', True, cri,s_coef, s_p, p_coef, p_p)

    return merge_decade

def plot_volin_box(merge_decade,cri,iou_threshold):
    iou = np_iou(merge_decade,iou_threshold)
    #id_, full_ = np_rank(merge_decade)
    #s_coef, s_p, p_coef, p_p = spearman_pearson(id_, full_)
    plot_single_volin_box(iou, cri)
    return


def compute_mean_std(whole):
    whole = merge_whole(whole)
    np_whole_score = []
    for data in whole:
        np_whole_score.append(np.array(data[2]))
    mean = np.mean(np_whole_score)
    std = np.std(np_whole_score)
    return mean, std


def plot_mean_outlier_intervals(id_whole_ce, id_whole_ts, full_whole_ce, full_whole_ts):
    id_ce_mean, id_ce_std = compute_mean_std(id_whole_ce)
    id_ts_mean, id_ts_std = compute_mean_std(id_whole_ts)
    full_ce_mean, full_ce_std = compute_mean_std(full_whole_ce)
    full_ts_mean, full_ts_std = compute_mean_std(full_whole_ts)

    x_ce = ['id_ce', 'full_ce']
    y_ce = [id_ce_mean, full_ce_mean]
    dy_ce = [id_ce_std, full_ce_std]

    x_ts = ['id_ts', 'full_ts']
    y_ts = [id_ts_mean, full_ts_mean]
    dy_ts = [id_ts_std, full_ts_std]

    fig = plt.figure(figsize=(10, 5))
    fig.suptitle('Visualization for Mean outlier scores', fontsize=15)

    ax0 = fig.add_subplot(121)

    ax0.errorbar(x_ce, y_ce, yerr=dy_ce, fmt='go')
    ax0.set_title('Mean value of outlier scores (Cross Entropy)')

    ax1 = fig.add_subplot(122)
    ax1.errorbar(x_ts, y_ts, yerr=dy_ts, fmt='bo')
    ax1.set_title('Mean value of outlier scores (Kullback Leibler)')
    plt.show()
    return


from scipy.stats import spearmanr
from scipy.stats import pearsonr


def add_another_score(id_whole_fil_ts,ids_full_ts ):
    id_whole_fil_ts = copy.deepcopy(id_whole_fil_ts)

    id_full_score = [[], [], [], [], [], [], []]
    # id intervals add full score
    for decade in range(5):
        for data in range(len(id_whole_fil_ts[decade])):
            # print(shaomu_decade_inter_scores_ts_id[decade][0][data][2])
            # id_whole_fil_ts[decade][data].append(shaomu_decade_inter_scores_ts_id[decade][0][data][2])
            id_full_score[decade].append(
                [id_whole_fil_ts[decade][data][2], ids_full_ts[decade][0][data][2], id_whole_fil_ts[decade][data][4]])
    for data in range(len(id_whole_fil_ts[5])):
        id_full_score[5].append([id_whole_fil_ts[5][data][2], ids_full_ts[6][0][data][2], id_whole_fil_ts[5][data][4]])
    #
    for data in range(len(id_whole_fil_ts[6])):
        id_full_score[6].append([id_whole_fil_ts[6][data][2], ids_full_ts[5][0][data][2], id_whole_fil_ts[6][data][4]])

    return id_full_score  # id_whole_fil_ts


def plot_id_full_pre(id_whole_fil_ce_add):
    #from sklearn.preprocessing import MinMaxScaler
    #scaler = MinMaxScaler()
    id_full_ce = copy.deepcopy(id_whole_fil_ce_add)
    #for decade in range(len(id_full_ce)):
        #id_full_ce[decade] = np.array(tuple(id_full_ce[decade]))

        #scaler.fit(id_full_ce[decade])

        #id_full_ce[decade] = scaler.transform(id_full_ce[decade])

    id_s = {'fiftees': [], 'sixtees': [], 'seventees': [], 'eightees': [], 'ninetees': [], 'zeroes': [], 'tens': []}
    full_s = {'fiftees': [], 'sixtees': [], 'seventees': [], 'eightees': [], 'ninetees': [], 'zeroes': [], 'tens': []}
    id_rank = {'fiftees': [], 'sixtees': [], 'seventees': [], 'eightees': [], 'ninetees': [], 'zeroes': [], 'tens': []}
    
    for decade in range(len(id_full_ce)):
        for data in id_full_ce[decade]:
            if decade == 0:
                id_s['fiftees'].append(data[0])
                full_s['fiftees'].append(data[1])
                id_rank['fiftees'].append(data[2])
            if decade == 1:
                id_s['sixtees'].append(data[0])
                full_s['sixtees'].append(data[1])
                id_rank['sixtees'].append(data[2])
            if decade == 2:
                id_s['seventees'].append(data[0])
                full_s['seventees'].append(data[1])
                id_rank['seventees'].append(data[2])
            if decade == 3:
                id_s['eightees'].append(data[0])
                full_s['eightees'].append(data[1])
                id_rank['eightees'].append(data[2])
            if decade == 4:
                id_s['ninetees'].append(data[0])
                full_s['ninetees'].append(data[1])
                id_rank['ninetees'].append(data[2])
            if decade == 5:
                id_s['zeroes'].append(data[0])
                full_s['zeroes'].append(data[1])
                id_rank['zeroes'].append(data[2])
            if decade == 6:
                id_s['tens'].append(data[0])
                full_s['tens'].append(data[1])
                id_rank['tens'].append(data[2])
    return id_s, full_s, id_rank


def plot_id_full_score(x, y, rank, method, computed_score, cri, mode):
    spearman_coef_ = []
    pearson_coef_ = []

    import matplotlib as mpl
    cmap = mpl.cm.cool_r

    if str(mode) == 'default':
        fig = plt.figure(figsize=(15, 8))
        plt.subplots_adjust(hspace=0.5)
        fig.suptitle('Visualization for ' + method + ' and its corresponding ' + computed_score + '(' + cri + ')',
                     fontsize=15)

        if method == 'full method':
            xlabel = 'FULL outlier scores'
            ylabel = 'ID outlier scores'
        else:
            ylabel = 'FULL outlier scores'
            xlabel = 'ID outlier scores'

        sub1 = fig.add_subplot(241)
        sub1.scatter(x['fiftees'], y['fiftees'], s=40, cmap='Greens')
        spear_coef, spear_p = spearmanr(x['fiftees'], y['fiftees'])
        pearson_coef, pearson_p = pearsonr(x['fiftees'], y['fiftees'])
        spearman_coef_.append(spear_coef)
        pearson_coef_.append(pearson_coef)
        per_s1 = '{:.2%}'.format(spear_coef)
        s_p1 = '{:.3f}'.format(spear_p)
        per_p1 = '{:.2%}'.format(pearson_coef)
        p_p1 = '{:.3f}'.format(pearson_p)
        sub1.set_xlabel(
            '\nSpearmans coef: {}, P-value: {}\n\nPearson coef: {}, P-value: {}'.format(per_s1, s_p1, per_p1, p_p1))

        sub2 = fig.add_subplot(242)
        ax = sub2.scatter(x['sixtees'], y['sixtees'], s=40, cmap='Greens')
        spear_coef, spear_p = spearmanr(x['fiftees'], y['fiftees'])
        pearson_coef, pearson_p = pearsonr(x['fiftees'], y['fiftees'])
        spearman_coef_.append(spear_coef)
        pearson_coef_.append(pearson_coef)
        per_s2 = '{:.2%}'.format(spear_coef)
        s_p2 = '{:.3f}'.format(spear_p)
        per_p2 = '{:.2%}'.format(pearson_coef)
        p_p2 = '{:.3f}'.format(pearson_p)
        sub2.set_xlabel(
            '\nSpearmans coef: {}, P-value: {}\n\nPearson coef: {}, P-value: {}'.format(per_s2, s_p2, per_p2, p_p2))

        sub3 = fig.add_subplot(243)
        sub3.scatter(x['seventees'], y['seventees'], s=40, cmap='Greens')
        spear_coef, spear_p = spearmanr(x['seventees'], y['seventees'])
        pearson_coef, pearson_p = pearsonr(x['seventees'], y['seventees'])
        spearman_coef_.append(spear_coef)
        pearson_coef_.append(pearson_coef)
        per_s3 = '{:.2%}'.format(spear_coef)
        s_p3 = '{:.3f}'.format(spear_p)
        per_p3 = '{:.2%}'.format(pearson_coef)
        p_p3 = '{:.3f}'.format(pearson_p)
        sub3.set_xlabel(
            '\nSpearmans coef: {}, P-value: {}\n\nPearson coef: {}, P-value: {}'.format(per_s3, s_p3, per_p3, p_p3))

        sub4 = fig.add_subplot(244)
        sub4.scatter(x['eightees'], y['eightees'], s=40, cmap='Greens')
        spear_coef, spear_p = spearmanr(x['eightees'], y['eightees'])
        pearson_coef, pearson_p = pearsonr(x['eightees'], y['eightees'])
        spearman_coef_.append(spear_coef)
        pearson_coef_.append(pearson_coef)
        per_s4 = '{:.2%}'.format(spear_coef)
        s_p4 = '{:.3f}'.format(spear_p)
        per_p4 = '{:.2%}'.format(pearson_coef)
        p_p4 = '{:.3f}'.format(pearson_p)
        sub4.set_xlabel(
            '\nSpearmans coef: {}, P-value: {}\n\nPearson coef: {}, P-value: {}'.format(per_s4, s_p4, per_p4, p_p4))

        sub5 = fig.add_subplot(245)
        sub5.scatter(x['ninetees'], y['ninetees'], s=40, cmap='Greens')
        spear_coef, spear_p = spearmanr(x['ninetees'], y['ninetees'])
        pearson_coef, pearson_p = pearsonr(x['ninetees'], y['ninetees'])
        spearman_coef_.append(spear_coef)
        pearson_coef_.append(pearson_coef)
        per_s5 = '{:.2%}'.format(spear_coef)
        s_p5 = '{:.3f}'.format(spear_p)
        per_p5 = '{:.2%}'.format(pearson_coef)
        p_p5 = '{:.3f}'.format(pearson_p)
        sub5.set_xlabel(
            '\nSpearmans coef: {}, P-value: {}\n\nPearson coef: {}, P-value: {}'.format(per_s5, s_p5, per_p5, p_p5))

        sub6 = fig.add_subplot(246)
        sub6.scatter(x['zeroes'], y['zeroes'], s=40, cmap='Greens')
        spear_coef, spear_p = spearmanr(x['zeroes'], y['zeroes'])
        pearson_coef, pearson_p = pearsonr(x['zeroes'], y['zeroes'])
        spearman_coef_.append(spear_coef)
        pearson_coef_.append(pearson_coef)
        per_s6 = '{:.2%}'.format(spear_coef)
        s_p6 = '{:.3f}'.format(spear_p)
        per_p6 = '{:.2%}'.format(pearson_coef)
        p_p6 = '{:.3f}'.format(pearson_p)
        sub6.set_xlabel(
            '\nSpearmans coef: {}, P-value: {}\n\nPearson coef: {}, P-value: {}'.format(per_s6, s_p6, per_p6, p_p6))

        sub7 = fig.add_subplot(247)
        sub7.scatter(x['tens'], y['tens'], s=40, cmap='Greens')
        spear_coef, spear_p = spearmanr(x['tens'], y['tens'])
        pearson_coef, pearson_p = pearsonr(x['tens'], y['tens'])
        spearman_coef_.append(spear_coef)
        pearson_coef_.append(pearson_coef)
        per_s7 = '{:.2%}'.format(spear_coef)
        s_p7 = '{:.3f}'.format(spear_p)
        per_p7 = '{:.2%}'.format(pearson_coef)
        p_p7 = '{:.3f}'.format(pearson_p)
        sub7.set_xlabel(
            '\nSpearmans coef: {}, P-value: {}\n\nPearson coef: {}, P-value: {}'.format(per_s7, s_p7, per_p7, p_p7))

        avg_sp = '{:.2%}'.format(np.mean(spearman_coef_))
        avg_pearson = '{:.2%}'.format(np.mean(pearson_coef_))

        fig.text(0.5, -0.05, xlabel, ha="center", va="center", fontsize=13.5)
        fig.text(0.08, 0.5, ylabel, ha="center", va="center", rotation=90, fontsize=13.5)
        fig.text(0.8, 0.25, 'Averaged Spearmans coef: {}\n\nAveraged Pearson coef: {}'.format(avg_sp, avg_pearson),
                 ha="center", va="center", fontsize=11)

    if str(mode) == 'color':
        fig = plt.figure(figsize=(15, 8))
        plt.subplots_adjust(hspace=0.5)
        fig.suptitle('Visualization for ' + method + ' and its corresponding ' + computed_score + '(' + cri + ')',
                     fontsize=15)

        if method == 'full method':
            xlabel = 'FULL outlier scores'
            ylabel = 'ID outlier scores'
        else:
            ylabel = 'FULL outlier scores'
            xlabel = 'ID outlier scores'

        sub1 = fig.add_subplot(241)
        sub1.scatter(x['fiftees'], y['fiftees'], s=40, c=rank['fiftees'], cmap=cmap)
        spear_coef, spear_p = spearmanr(x['fiftees'], y['fiftees'])
        pearson_coef, pearson_p = pearsonr(x['fiftees'], y['fiftees'])
        spearman_coef_.append(spear_coef)
        pearson_coef_.append(pearson_coef)
        per_s1 = '{:.2%}'.format(spear_coef)
        s_p1 = '{:.3f}'.format(spear_p)
        per_p1 = '{:.2%}'.format(pearson_coef)
        p_p1 = '{:.3f}'.format(pearson_p)
        sub1.set_xlabel(
            '\nSpearmans coef: {}, P-value: {}\n\nPearson coef: {}, P-value: {}'.format(per_s1, s_p1, per_p1, p_p1))

        sub2 = fig.add_subplot(242)
        ax = sub2.scatter(x['sixtees'], y['sixtees'], s=40, c=rank['sixtees'], cmap=cmap)
        spear_coef, spear_p = spearmanr(x['fiftees'], y['fiftees'])
        pearson_coef, pearson_p = pearsonr(x['fiftees'], y['fiftees'])
        spearman_coef_.append(spear_coef)
        pearson_coef_.append(pearson_coef)
        per_s2 = '{:.2%}'.format(spear_coef)
        s_p2 = '{:.3f}'.format(spear_p)
        per_p2 = '{:.2%}'.format(pearson_coef)
        p_p2 = '{:.3f}'.format(pearson_p)
        sub2.set_xlabel(
            '\nSpearmans coef: {}, P-value: {}\n\nPearson coef: {}, P-value: {}'.format(per_s2, s_p2, per_p2, p_p2))

        sub3 = fig.add_subplot(243)
        sub3.scatter(x['seventees'], y['seventees'], s=40, c=rank['seventees'], cmap=cmap)
        spear_coef, spear_p = spearmanr(x['seventees'], y['seventees'])
        pearson_coef, pearson_p = pearsonr(x['seventees'], y['seventees'])
        spearman_coef_.append(spear_coef)
        pearson_coef_.append(pearson_coef)
        per_s3 = '{:.2%}'.format(spear_coef)
        s_p3 = '{:.3f}'.format(spear_p)
        per_p3 = '{:.2%}'.format(pearson_coef)
        p_p3 = '{:.3f}'.format(pearson_p)
        sub3.set_xlabel(
            '\nSpearmans coef: {}, P-value: {}\n\nPearson coef: {}, P-value: {}'.format(per_s3, s_p3, per_p3, p_p3))

        sub4 = fig.add_subplot(244)
        sub4.scatter(x['eightees'], y['eightees'], s=40, c=rank['eightees'], cmap=cmap)
        spear_coef, spear_p = spearmanr(x['eightees'], y['eightees'])
        pearson_coef, pearson_p = pearsonr(x['eightees'], y['eightees'])
        spearman_coef_.append(spear_coef)
        pearson_coef_.append(pearson_coef)
        per_s4 = '{:.2%}'.format(spear_coef)
        s_p4 = '{:.3f}'.format(spear_p)
        per_p4 = '{:.2%}'.format(pearson_coef)
        p_p4 = '{:.3f}'.format(pearson_p)
        sub4.set_xlabel(
            '\nSpearmans coef: {}, P-value: {}\n\nPearson coef: {}, P-value: {}'.format(per_s4, s_p4, per_p4, p_p4))

        sub5 = fig.add_subplot(245)
        sub5.scatter(x['ninetees'], y['ninetees'], s=40, c=rank['ninetees'], cmap=cmap)
        spear_coef, spear_p = spearmanr(x['ninetees'], y['ninetees'])
        pearson_coef, pearson_p = pearsonr(x['ninetees'], y['ninetees'])
        spearman_coef_.append(spear_coef)
        pearson_coef_.append(pearson_coef)
        per_s5 = '{:.2%}'.format(spear_coef)
        s_p5 = '{:.3f}'.format(spear_p)
        per_p5 = '{:.2%}'.format(pearson_coef)
        p_p5 = '{:.3f}'.format(pearson_p)
        sub5.set_xlabel(
            '\nSpearmans coef: {}, P-value: {}\n\nPearson coef: {}, P-value: {}'.format(per_s5, s_p5, per_p5, p_p5))

        sub6 = fig.add_subplot(246)
        sub6.scatter(x['zeroes'], y['zeroes'], s=40, c=rank['zeroes'], cmap=cmap)
        spear_coef, spear_p = spearmanr(x['zeroes'], y['zeroes'])
        pearson_coef, pearson_p = pearsonr(x['zeroes'], y['zeroes'])
        spearman_coef_.append(spear_coef)
        pearson_coef_.append(pearson_coef)
        per_s6 = '{:.2%}'.format(spear_coef)
        s_p6 = '{:.3f}'.format(spear_p)
        per_p6 = '{:.2%}'.format(pearson_coef)
        p_p6 = '{:.3f}'.format(pearson_p)
        sub6.set_xlabel(
            '\nSpearmans coef: {}, P-value: {}\n\nPearson coef: {}, P-value: {}'.format(per_s6, s_p6, per_p6, p_p6))

        sub7 = fig.add_subplot(247)
        sub7.scatter(x['tens'], y['tens'], s=40, c=rank['tens'], cmap=cmap)
        spear_coef, spear_p = spearmanr(x['tens'], y['tens'])
        pearson_coef, pearson_p = pearsonr(x['tens'], y['tens'])
        spearman_coef_.append(spear_coef)
        pearson_coef_.append(pearson_coef)
        per_s7 = '{:.2%}'.format(spear_coef)
        s_p7 = '{:.3f}'.format(spear_p)
        per_p7 = '{:.2%}'.format(pearson_coef)
        p_p7 = '{:.3f}'.format(pearson_p)
        sub7.set_xlabel(
            '\nSpearmans coef: {}, P-value: {}\n\nPearson coef: {}, P-value: {}'.format(per_s7, s_p7, per_p7, p_p7))

        avg_sp = '{:.2%}'.format(np.mean(spearman_coef_))
        avg_pearson = '{:.2%}'.format(np.mean(pearson_coef_))

        cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.75])
        cb = fig.colorbar(ax, cbar_ax)
        cb.set_label('Rank')
        fig.text(0.5, -0.05, xlabel, ha="center", va="center", fontsize=13.5)
        fig.text(0.08, 0.5, ylabel, ha="center", va="center", rotation=90, fontsize=13.5)
        fig.text(0.8, 0.25, 'Averaged Spearmans coef: {}\n\nAveraged Pearson coef: {}'.format(avg_sp, avg_pearson),
                 ha="center", va="center", fontsize=11)
    return


def run_test(iou_threshold):
    id_ce, full_ce = merge_pre('crossent')
    id_ts, full_ts = merge_pre('kl')

    id_whole_ce = merge_whole(id_ce)
    full_whole_ce = merge_whole(full_ce)
    id_whole_ts = merge_whole(id_ts)
    full_whole_ts = merge_whole(full_ts)

    print('Mean values before filtering:\n')

    print('Mean values of ID outliers (Cross Entropy):')
    print( (  len(id_ce[0]) + len(id_ce[1]) + len(id_ce[2]) + len(id_ce[3]) + len(id_ce[4]) + len(id_ce[5]) + len(id_ce[6]) )/7. )
    print('\nMean values of Full outliers (Cross Entropy):')
    print( (  len(full_ce[0]) + len(full_ce[1]) + len(full_ce[2]) + len(full_ce[3]) + len(full_ce[4]) + len(full_ce[5]) + len(full_ce[6]) )/7. )
    print('\nMean values of ID outliers (Kullback Leibler):')
    print((len(id_ts[0]) + len(id_ts[1]) + len(id_ts[2]) + len(id_ts[3]) + len(id_ts[4]) + len(id_ts[5]) + len(id_ts[6])) / 7.)
    print('\nMean values of Full outliers (Kullback Leibler):')
    print((len(full_ts[0]) + len(full_ts[1]) + len(full_ts[2]) + len(full_ts[3]) + len(full_ts[4]) + len(full_ts[5]) + len(full_ts[6])) / 7.)



    id_whole_fil_ce = iou_filter(id_whole_ce)
    full_whole_fil_ce = iou_filter(full_whole_ce)
    id_whole_fil_ts = iou_filter(id_whole_ts)
    full_whole_fil_ts = iou_filter(full_whole_ts)

    print('\nMean values after filtering:\n')

    print('Mean values of ID outliers (Cross Entropy):')
    print((len(id_whole_fil_ce[0]) + len(id_whole_fil_ce[1]) + len(id_whole_fil_ce[2]) + len(id_whole_fil_ce[3]) + len(id_whole_fil_ce[4]) + len(
        id_whole_fil_ce[5]) + len(id_whole_fil_ce[6])) / 7.)
    print('\nMean values of Full outliers (Cross Entropy):')
    print((len(full_whole_fil_ce[0]) + len(full_whole_fil_ce[1]) + len(full_whole_fil_ce[2]) + len(full_whole_fil_ce[3]) + len(
        full_whole_fil_ce[4]) + len(full_whole_fil_ce[5]) + len(full_whole_fil_ce[6])) / 7.)
    print('\nMean values of ID outliers (Kullback Leibler):')
    print((len(id_whole_fil_ts[0]) + len(id_whole_fil_ts[1]) + len(id_whole_fil_ts[2]) + len(id_whole_fil_ts[3]) + len(
        id_whole_fil_ts[4]) + len(id_whole_fil_ts[5]) + len(id_whole_fil_ts[6])) / 7.)
    print('\nMean values of Full outliers (Kullback Leibler):')
    print((len(full_whole_fil_ts[0]) + len(full_whole_fil_ts[1]) + len(full_whole_fil_ts[2]) + len(full_whole_fil_ts[3]) + len(
        full_whole_fil_ts[4]) + len(full_whole_fil_ts[5]) + len(full_whole_fil_ts[6])) / 7.)


    #print(len(id_whole_fil_ce[0]))
    #print(len(full_whole_fil_ce[0]))
    #print(len(id_whole_fil_ts[0]))
    #print(len(full_whole_fil_ts[0]))

    return

def run_merge_analysis(iou_threshold):
    id_ce, full_ce = merge_pre('crossent')
    id_ts, full_ts = merge_pre('kl')

    id_whole_ce = merge_whole(id_ce)
    full_whole_ce = merge_whole(full_ce)
    id_whole_ts = merge_whole(id_ts)
    full_whole_ts = merge_whole(full_ts)

    id_whole_fil_ce = iou_filter(id_whole_ce)
    full_whole_fil_ce = iou_filter(full_whole_ce)
    id_whole_fil_ts = iou_filter(id_whole_ts)
    full_whole_fil_ts = iou_filter(full_whole_ts)

    m_ce = visual_md(id_whole_fil_ce, full_whole_fil_ce, 'id2full', iou_threshold, 'ce')
    visual_md(id_whole_fil_ce, full_whole_fil_ce, 'full2id', iou_threshold, 'ce')
    plot_volin_box(m_ce, 'ce',iou_threshold)

    m_kl = visual_md(id_whole_fil_ts, full_whole_fil_ts, 'id2full', iou_threshold, 'kl')
    visual_md(id_whole_fil_ts, full_whole_fil_ts, 'full2id', iou_threshold, 'kl')
    plot_volin_box(m_kl, 'kl',iou_threshold)
    return

def compute_other_criterion_outlier_score(mode):
    id_ce, full_ce = merge_pre('crossent')
    id_ts, full_ts = merge_pre('kl')

    id_whole_ce = merge_whole(id_ce)
    full_whole_ce = merge_whole(full_ce)
    id_whole_ts = merge_whole(id_ts)
    full_whole_ts = merge_whole(full_ts)

    id_whole_fil_ce = iou_filter(id_whole_ce)
    full_whole_fil_ce = iou_filter(full_whole_ce)
    id_whole_fil_ts = iou_filter(id_whole_ts)
    full_whole_fil_ts = iou_filter(full_whole_ts)

    #plot_mean_outlier_intervals(id_whole_fil_ce, id_whole_fil_ts, full_whole_fil_ce, full_whole_fil_ts)

    import pickle
    with open("F:/PythonJupyterStudy/thesis-master/shaomu_ID's_full_scores_ce.txt", "rb") as fp:  # Pickling
        ids_full_ce = pickle.load(fp)
    with open("F:/PythonJupyterStudy/thesis-master/shaomu_ID's_full_scores_ts.txt",
              "rb") as fp:  # Pickling
        ids_full_ts = pickle.load(fp)

    with open("F:/PythonJupyterStudy/thesis-master/shaomu_Full's_id_scores_ce.txt",
              "rb") as fp:  # Pickling
        fulls_id_ce = pickle.load(fp)
    with open("F:/PythonJupyterStudy/thesis-master/shaomu_Full's_id_scores_ts.txt",
              "rb") as fp:  # Pickling
        fulls_id_ts = pickle.load(fp)

    id_whole_fil_ts_add = add_another_score(id_whole_fil_ts, ids_full_ts)
    id_whole_fil_ce_add = add_another_score(id_whole_fil_ce, ids_full_ce)
    x1, y1,rank1 = plot_id_full_pre(id_whole_fil_ce_add)
    plot_id_full_score(x1, y1,rank1, 'id method', 'full method scores', 'Cross Entropy',mode)


    x2,y2,rank2 = plot_id_full_pre(id_whole_fil_ts_add)
    plot_id_full_score(x2,y2,rank2,'id method','full method scores','Kullback Leibler',mode)

    full_whole_fil_ts_add = add_another_score(full_whole_fil_ts, fulls_id_ts)
    full_whole_fil_ce_add = add_another_score(full_whole_fil_ce, fulls_id_ce)
    x3, y3,rank3 = plot_id_full_pre(full_whole_fil_ce_add)
    plot_id_full_score(x3, y3,rank3, 'full method', 'ID method scores', 'Cross Entropy',mode)

    x4, y4,rank4 = plot_id_full_pre(full_whole_fil_ts_add)
    plot_id_full_score(x4, y4,rank4, 'full method', 'id method scores', 'Kullback Leibler',mode)

    return id_whole_fil_ce_add,id_whole_fil_ts_add,full_whole_fil_ce_add,full_whole_fil_ts_add#x1,y1,x2,y2,x3,y3,x4,y4,



