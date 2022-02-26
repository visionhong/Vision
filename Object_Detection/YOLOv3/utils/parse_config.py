import pdb
#
# config/yolov3.cfg
def parse_model_config(path):
    ''' Parses the yolo-v3 layer configuration file and returns module definitions'''
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]  # 값이 있는 줄이면서 #으로 시작하지 않는것만 가져옴
    lines = [x.rstrip().lstrip() for x in lines]  # 좌우 공백 제거
    module_defs = []
    for line in lines:
        if line.startswith('['):
            module_defs.append({})  # 다음 타입이 들어올때마다 새로운 딕셔너리 생성
            module_defs[-1]['type'] = line[1:-1].rstrip()  # 딕셔너리에 type 키값에 []제거한 값추가
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0  # batch_normalize가 써있지 않은 conv에서 0으로 따로 지정해놓음
        else:
            key, value = line.split('=')
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()  # 해당 type에 keys, values 추가

    return module_defs

#config/coco.data
def parse_data_config(path):
    '''Parses the data configuration file'''
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '6'  # 내컴퓨터 6개임 github : '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()  # 양끝 공백제거 + \n 제거
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options




if __name__ == '__main__':
    a = parse_model_config('../config/yolov3.cfg')
    print(a)
    print(len(a))

    b = parse_data_config('../config/coco.data')
    print(b)