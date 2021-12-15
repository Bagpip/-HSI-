import os


def file_name(file_dir):
    R = []
    H = []
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.raw':
                R.append(os.path.join(root, file))
            if os.path.splitext(file)[1] == '.hdr':
                H.append(os.path.join(root, file))
            if os.path.splitext(file)[1] == '.xml':
                L.append(os.path.join(root, file))
    return R, H, L


classic = ['blank', 'roi']
path = 'G:/restorepaper/date/img_raw'
X = []


def date_path(classic, path):
    path_blank_raw = []
    path_raw = []
    path_blank_hdr = []
    path_hdr = []
    label_path = []
    path_all = []
    path_list = []
    R, H, L = file_name(path)

    for R_1 in R:
        R_1 = os.path.splitext(R_1)[0]
        R_1 = os.path.split(R_1)[1]
        if R_1.endswith(classic[0]):
            path_blank_raw.append(R_1)
        if not R_1.endswith(classic[0]):
            path_raw.append(R_1)

    # temp = R_1.find('roi')
    # R_1[0:temp]

    for H_1 in H:
        H_1 = os.path.splitext(H_1)[0]
        H_1 = os.path.split(H_1)[1]
        if H_1.endswith(classic[0]):
            path_blank_hdr.append(H_1)
        if not H_1.endswith(classic[0]):
            path_hdr.append(H_1)

    for L_1 in L:
        L_1 = os.path.splitext(L_1)[0]
        L_1 = os.path.split(L_1)[1]
        label_path.append(L_1)

    for name in path_raw:
        temp0 = name.find(classic[1])
        for name_l in label_path:
            if not name[0: temp0] == name_l[0: temp0]:
                continue
            temp1 = name_l.find(classic[1])
            if name_l[temp1 + 3] == name[temp1 + 3]:
                for name_b in path_blank_raw:
                    temp2 = name_b.find(classic[0])
                    if name_l[0: temp2] == name_b[0: temp2]:
                        path_in_blank_hdr = path + '/' + name_b + '.hdr'
                        path_in_blank_raw = path + '/' + name_b + '.raw'
                        path_in_hdr = path + '/' + name + '.hdr'
                        path_in_raw = path + '/' + name + '.raw'
                        path_in_label = path + '/' + name + '.xml'
                        path_all= [path_in_blank_hdr, path_in_blank_raw, path_in_hdr, path_in_raw, path_in_label]
                        path_list.append(path_all)
    return path_list


X = date_path(classic, path)
print(X)