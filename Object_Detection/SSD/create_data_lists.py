from SSD.utils import create_data_lists

if __name__ == '__main__':
    # VOC 폴더경로와 json을 담을 output folder 경로
    create_data_lists(voc07_path='E:\\Object Detection\\data\\VOC\\VOC2007',
                      voc12_path='E:\\Object Detection\\data\\VOC\\VOC2012',
                      output_folder='E:\\Object Detection\\data\\VOC\\Output')
