import numpy as np
import math
import cv2
import os
import json
#from scipy.special import expit
#from utils.box import BoundBox, box_iou, prob_compare
#from utils.box import prob_compare2, box_intersection
from ...utils.box import BoundBox
from ...cython_utils.cy_yolo2_findboxes import box_constructor

def expit(x):
	return 1. / (1. + np.exp(-x))

def _softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def findboxes(self, net_out):
	# meta
	meta = self.meta
	boxes = list()
	boxes=box_constructor(meta,net_out)
	return boxes

def postprocess(self, CC,net_out, im, save = True):
	"""
	Takes net output, draw net_out, save to disk
	"""
	boxes = self.findboxes(net_out)

	# meta
	meta = self.meta
	threshold = meta['thresh']
	colors = meta['colors']
	labels = meta['labels']
	answer_path ="content/result/" 
	if type(im) is not np.ndarray:
		imgcv = cv2.imread(im)
		astr = im.split('/')
		answer_path = answer_path+os.path.splitext(astr[2])[0]+".txt"
	else: imgcv = im
	h, w, _ = imgcv.shape
	f = open(answer_path)
	count = 0
	temp = np.array([])
	for line in f:
		context = line.split(' ')
		#print(context[1],"   ",context[2],"   ",context[3],"   ",context[4])
		x_plot = np.float64(context[1]) *320
		y_plot = np.float64(context[2]) *224
		w_long = (np.float64(context[3]) *320)/2
		h_long = (np.float64(context[4]) *224)/2
		gt_left = round(x_plot-w_long)
		gt_top = round(y_plot-h_long)
		gt_right = round(x_plot+w_long)
		gt_bot = round(y_plot+h_long)
		buff = [[gt_left,gt_top,gt_right,gt_bot]]
		
		if (count==0):
			temp = np.array([[gt_left,gt_top,gt_right,gt_bot]])
		else:
			temp = np.row_stack((temp,buff))
		count = count + 1
	
	resultsForJSON = []
	sum_IOU = 0
	for b in boxes:
		boxResults = self.process_box(b, h, w, threshold)
		if boxResults is None:
			continue
		left, right, top, bot, mess, max_indx, confidence = boxResults
		thick = int((h + w) // 300)
		max_IOU = 0
		if self.FLAGS.json:
			resultsForJSON.append({"label": mess, "confidence": float('%.2f' % confidence), "topleft": {"x": left, "y": top}, "bottomright": {"x": right, "y": bot}})
			continue
		for t in temp:
			#print("t:",t)
			
			area = (min(right,t[2])-max(left,t[0]))*(min(bot,t[3])-max(top,t[1]))
			area1=(bot-top)*(right-left)
			area2=(t[2]-t[0])*(t[3]-t[1])
			Union = area1+area2-area
			
			
			IOU = area/Union
			if (IOU > max_IOU):
				max_IOU = IOU
			
		cv2.rectangle(imgcv,
			(left, top), (right, bot),
			colors[max_indx], thick)
		cv2.putText(imgcv, mess, (left, top - 12),
			0, 1e-3 * h, colors[max_indx],thick//3)
		print("images:",im,"IOU:",max_IOU*100,"C:",area,"Union:",Union)
		CC.count +=  max_IOU
	
	if not save: return imgcv

	outfolder = os.path.join(self.FLAGS.imgdir, 'out')
	img_name = os.path.join(outfolder, os.path.basename(im))
	if self.FLAGS.json:
		textJSON = json.dumps(resultsForJSON)
		textFile = os.path.splitext(img_name)[0] + ".json"
		with open(textFile, 'w') as f:
			f.write(textJSON)
		return

	cv2.imwrite(img_name, imgcv)
	
