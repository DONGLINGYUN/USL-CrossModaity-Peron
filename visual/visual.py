import cv2 as cv
import os
import numpy as np

from matplotlib import pyplot as plt


class Visualization:
    """
    GitHub: https://github.com/asd123pwj/ReID-visualize
    Blog: https://mwhls.top/2858.html
    """

    def __init__(self):
        self.gallery_path = np.array([],dtype='<U48')
        self.gallery_match = np.array([],dtype='bool')
        self.query_path = np.array([],dtype='str')

        self._output_path = './logs/visualize'
        self._style_list = ['border', 'top', 'description']
        self._style = self._style_list[1]
        self._color_match = [0, 255, 0]
        self._color_mismatch = [255, 0, 0]
        self._index = 0

        self._gallery_img = []
        self._gallery_match = np.array([],dtype='bool')
        self._query_img = []

        self._output_name = "helloworld"
        self._output_img = None

    def help(self):
        help = """
        #   code sample
        #   代码示例
        from ReID_visualize import Visualization as vis
        query_path = './query/query1.jpg'
        gallery_path = ['./load/method1/gallery/1.jpg',
                        './load/method1/gallery/2.jpg',
                        './load/method1/gallery/3.jpg',
                        './load/method1/gallery/4.jpg' ]
        gallery_match = [True, True, False, True]

        v = vis()

        v.load_query(query_path)
        v.load_gallery(gallery_path, gallery_match)
        v.visualize()   #   visualize the data above, and index +1 to 1
        #   output:   (1: query1.jpg), (1: query1.log)

        v.clear()       #   clear the data above, prepare for next visualization.
        v.load_query(query_path)
        v.load_gallery(gallery_path, gallery_match)
        v.load_gallery('./load/method2/gallery/1.jpg', True)
        v.load_gallery('./load/method2/gallery/2.jpg', False)
        v.visualize()   #   visualize the data above, and index +1 to 2
        #   output:   (2: query1.jpg), (2: query1.log)

        """
        print(help)

    def load_gallery(self, gallery_path, gallery_match):
        """
        load path of gallery, and match of them
        加载gallery路径，以及是否成功匹配。
        """
        # if isinstance(gallery_path, list) is False:
        #     gallery_path = [gallery_path]
        #     gallery_match = [gallery_match]
        self.gallery_path = gallery_path
        self.gallery_match = gallery_match

    def load_query(self, query_path):
        """
        load path of query
        加载query的路径。
        """
        self.query_path = query_path

    def clear(self):
        """
        clear cache, prepare to visualize the next (query and gallery)
        清除上一次的query与gallery，保持已可视化的数量(index)不变，为下一次可视化做准备
        """
        self.gallery_path = []
        self.gallery_match = []
        self.query_path = None
        self._gallery_img = []
        self._gallery_match = []
        self._query_img = []

    def set_style(self, style=1, match_color=[0, 255, 0], mismatch_color=[255, 0, 0]):
        """
        style:
            0:  framed by a rectangle
                被矩形框框着
            1:  a line in the top
                顶部有一条线标注
            2:  only index and True/False
                只显示序号以及True/False
        match_color:
            the color of match, default is [0, 255, 0](green)
            匹配的标注颜色，默认为[0, 255, 0]绿色
        mismatch_color:
            the color of mismatch, default is [255, 0, 0](red)
            不匹配的标注颜色，默认为[255, 0, 0]红色
        """
        self._style = self._style_list[style]
        self._color_match = match_color
        self._color_mismatch = mismatch_color

    def set_output_path(self, output='./log/visualize'):
        """
        set output path, default is './log/visualize'
        设置输出路径，默认为'./log/visualize'
        """
        self._output_path = output

    def _data_process(self):
        # 读取gallery的所有地址对应的图片
        for i in self.gallery_path:
            for j in i:
                img = cv.resize(cv.imread(j), (300, 800), interpolation=cv.INTER_LINEAR)
                b, g, r = cv.split(img)
                img_n = cv.merge([r, g, b])
                self._gallery_img.append(img_n)

        # 得到击中目标矩阵
        self._gallery_match = self.gallery_match.reshape(-1)
        # for path in self.gallery_path:
        #     img = cv.resize(cv.imread(path), (300, 800), interpolation=cv.INTER_LINEAR)
        #     b, g, r = cv.split(img)
        #     img_n = cv.merge([r, g, b])
        #     self._gallery_img.append(img_n)
        # self._gallery_match = self.gallery_match

        # 读取querypath的所有地址对应的图片
        for path in self.query_path:
            b, g, r = cv.split(cv.resize(cv.imread(path, 3), (300, 800), interpolation=cv.INTER_LINEAR))
            self._query_img.append(cv.merge([r, g, b]))
            # query_name = os.path.splitext(os.path.split(self.query_path)[1])[0]


        # self._index += 1
        # self._output_name = str(self._index) + '_' + query_name

    def _gallery_process(self):
        for num in range(len(self._gallery_img)):
            if self._gallery_match[num] == True:
                color = self._color_match
            else:
                color = self._color_mismatch
            img = self._gallery_img[num]
            if self._style == 'border':
                self._gallery_img[num] = cv.copyMakeBorder(img, 10, 10, 10, 10, cv.BORDER_CONSTANT, value=color)
            elif self._style == 'top':
                self._gallery_img[num] = cv.line(img, (0, 0), (img.shape[0], 0), color, 2)
            elif self._style == 'description':
                pass

    def _joint(self):
        self._output_img = None
        row = len(self._query_img)
        if len(self._gallery_img)%row ==0:
            col = len(self._gallery_img) // row + 1
        else:
            col = len(self._gallery_img) // row + 2
        # if num > 16:
        #     col = 11
        #     row = num // col + 1

        # plt.figure(figsize=(6, 8))

        fig,axs = plt.subplots(row,col,figsize=(10, 16))
        plt.subplots_adjust(hspace=0.03, wspace=0.003)
        k = 0
        z = 0
        for i in range(row):
            for j in range(col):
                ax = axs[i, j]
                ax.axis('off')

                if j == 0:
                    # axs[i, j].plot(self._query_img[k])
                    ax.imshow(self._query_img[k])
                    # axs.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
                    k=k+1
                else:
                    # title = str(n + 1) + '-' + str(self._gallery_match[n])
                    # axs[i, j].plot(self._gallery_img[z])
                    ax.imshow(self._gallery_img[z])
                    z=z+1

        plt.show()
        # plt.savefig(output_path, bbox_inches='tight')

        # for i in range(row):
        #     for j in range(col):
        #         plt.subplot(i, j, 1), plt.imshow(self._query_img[i]), plt.title('query'), plt.axis('off')
        #         plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
        #
        #
        # pos = 2
        # for n in range(num - 1):
        #
        #
        #
        #     img = self._gallery_img[n]
        #     title = str(n + 1) + '-' + str(self._gallery_match[n])
        #     # plt.figure(figsize=(6, 8))
        #     plt.subplot(row, col, n + n // (col - 1) + 2), plt.imshow(img), plt.title(title), plt.axis('off')



    def _save_log(self):
        output_path = os.path.join(self._output_path, (self._output_name + '.log'))
        log = ""
        log += 'query:\t\t' + self.query_path + '\n'
        for num in range(len(self.gallery_path)):
            log += str(num) + ': ' + str(self._gallery_match[num]) + ' \t' + self.gallery_path[num] + '\n'
        with open(output_path, 'w') as f:
            f.write(log)

    def visualize(self):
        """
        visualize data from {(v = vis()) or (v.clear())} to now
        将v.clear()之后加载的数据进行可视化。
        """
        if os.path.exists(self._output_path) is False:
            os.makedirs(self._output_path)
        self._data_process()
        self._gallery_process()
        self._joint()
        # self._save_log()
        output_path = os.path.join(self._output_path, (self._output_name + '.jpg'))

        plt.savefig(output_path, bbox_inches='tight')
        plt.clf()

