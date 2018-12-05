# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 18:39:19 2018

@author: yiyuezhuo
"""
#https://blog.gtwang.org/programming/selective-search-for-object-detection/

import cv2

# 讀取圖檔
im = cv2.imread('ab.jpg')

# 建立 Selective Search 分割器
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

# 設定要進行分割的圖形
ss.setBaseImage(im)

# 使用快速模式（精準度較差）
ss.switchToSelectiveSearchFast()

# 使用精準模式（速度較慢）
# ss.switchToSelectiveSearchQuality()

# 執行 Selective Search 分割
rects = ss.process()

print('候選區域總數量： {}'.format(len(rects)))

gs = cv2.ximgproc.segmentation.createGraphSegmentation()

dst = gs.processImage(im)

gs.setK(300)
#gs.setMinSize(100)
gs.setMinSize(5000)
gs.setSigma(0.5)

dst = gs.processImage(im)

import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(16,9))
plt.imshow(dst, cmap='gray')
plt.show()

b,g,r = cv2.split(im)
rgb_img = cv2.merge([r,g,b]) 
for i in range(int(dst.max())+1):
    masked_img = rgb_img.copy()
    masked_img[dst != i] = np.zeros(3)
    plt.imshow(masked_img)
    plt.show()
    
import imageio
import os

out_img = np.ones(rgb_img.shape[:2]+(4,),dtype=np.uint8) * 255
out_img[:,:,:3] = rgb_img
out_img[dst != i] = np.zeros(4)
imageio.imsave('ab_test.png', out_img)

def split_image(im_path, K=300, MinSize = 100, Sigma=0.5, show=False,
                save=True, dir_name = 'test_split',
                crop=False, suffix=True, scale=None):
    im = cv2.imread(im_path)
    if scale:
        MinSize = int(np.prod(im.shape[:2])/scale)
    if suffix:
        dir_name = dir_name+f'K={K}_MinSize={MinSize}_Sigma={Sigma}'

    
    b,g,r = cv2.split(im)
    rgb_img = cv2.merge([r,g,b]) 
    
    gs = cv2.ximgproc.segmentation.createGraphSegmentation()
    gs.setK(K)
    gs.setMinSize(MinSize)
    gs.setSigma(Sigma)
    dst = gs.processImage(im)
    
    #raise Exception
    if save:
        os.makedirs(dir_name, exist_ok=True)
    for i in range(int(dst.max())+1):
        mask = dst != i
        if show:
            masked_img = rgb_img.copy()
            masked_img[mask] = np.zeros(3)
            
            if crop:
                y,x = np.where(~mask)
                masked_img = masked_img[y.min():y.max()+1,x.min():x.max()+1]

            plt.imshow(masked_img)
            plt.show()
        if save:
            out_img = np.ones(rgb_img.shape[:2]+(4,),dtype=np.uint8) * 255
            out_img[:,:,:3] = rgb_img[:,:,:3]
            out_img[mask] = np.zeros(4)
            
            if crop:
                y,x = np.where(~mask)
                out_img = out_img[y.min():y.max()+1,x.min():x.max()+1]

            
            name = 'section_'+str(i)+'.png'
            imageio.imsave(os.path.join(dir_name,name), out_img)

'''
for i,name in enumerate(os.listdir('test_img')):
    #im = cv2.imread(name)
    path = os.path.join('test_img',name)
    split_image(path,MinSize=5000,crop=True,dir_name='hp'+str(i),scale=60)
'''
    
'''
# 要顯示的候選區域數量
numShowRects = 100

# 每次增加或減少顯示的候選區域數量
increment = 50

while True:
  # 複製一份原始影像
  imOut = im.copy()

  # 以迴圈處理每一個候選區域
  for i, rect in enumerate(rects):
      # 以方框標示候選區域
      if (i < numShowRects):
          x, y, w, h = rect
          cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
      else:
          break

  # 顯示結果
  cv2.imshow("Output", imOut)

  # 讀取使用者所按下的鍵
  k = cv2.waitKey(0) & 0xFF

  # 若按下 m 鍵，則增加 numShowRects
  if k == 109:
      numShowRects += increment
  # 若按下 l 鍵，則減少 numShowRects
  elif k == 108 and numShowRects > increment:
      numShowRects -= increment
  # 若按下 q 鍵，則離開
  elif k == 113:
      break

# 關閉圖形顯示視窗
cv2.destroyAllWindows()
'''