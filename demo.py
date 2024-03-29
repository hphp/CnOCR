#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2024 BetterWorld.ai.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: demo.py
Author: jiatong.han@BetterWorld.ai.com
Date: 2024/03/29 08:55:48
"""
import sys, os
sys.path.append(os.path.dirname(os.getcwd()))
from PIL import Image
from cnocr import app

def demo(img_fp):
	from cnocr import CnOcr
	det_model_name='en_PP-OCRv3_det'
	ocr = CnOcr(det_model_name=det_model_name, rec_model_name='en_PP-OCRv3')
	out = ocr.ocr(img_fp, return_cropped_image=True)
	print(out)
	img = Image.open(img_fp).convert('RGB')
	vis_fp_header = img_fp.replace('.png', '')
	app.visualize_naive_result(img, det_model_name, out, 0, vis_fp_header)
	app.visualize_result(img, ocr, vis_fp_header)

def demo_dp(dp):
	from daVinci.Wheels import util
	from cnocr import CnOcr
	det_model_name='en_PP-OCRv3_det'
	ocr = CnOcr(det_model_name=det_model_name, rec_model_name='en_PP-OCRv3')
	for img_fp in util.lstfile(dp):
		out = ocr.ocr(img_fp, return_cropped_image=True)
		img = Image.open(img_fp).convert('RGB')
		vis_fp_header = img_fp.replace('.png', '')
		app.visualize_naive_result(img, det_model_name, out, 0, vis_fp_header)
		app.visualize_result(img, out, vis_fp_header)

if __name__ == "__main__":
    import inspect, sys
    current_module = sys.modules[__name__]
    funnamelst = [item[0] for item in inspect.getmembers(current_module, inspect.isfunction)]
    if len(sys.argv) > 1:
        index = 1
        while index < len(sys.argv):
            if '--' in sys.argv[index]:
   	            index += 2
            else:
                break
        func = getattr(sys.modules[__name__], sys.argv[index])
        func(*sys.argv[index+1:])
    else:
        print('	'.join((__file__, "/".join(funnamelst), "args"))
)

