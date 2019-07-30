#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import re
import time
import sys
def detectGPU_exec(exec_filename, sleep_second):
    exec_return = None
    while(exec_return==None):
        gpustatus = os.popen("nvidia-smi").read()
        gpustatus = gpustatus.replace("\n", "")
        pattern = "(\d)  TITAN Xp.*?P(\d)"  # 匹配cuda_idx 和 是否在使用
        res = re.findall(pattern, gpustatus)  # dict(str, str)
        for cuda_idx, Perf_status in res:
            if Perf_status == '0':
                # 这里执行我们占用GPU资源的方法
                exec_return = os.system("CUDA_VISIBLE_DEVICES=%d python %s" % (int(cuda_idx), exec_filename))
                if exec_return != 0:
                    print("error_id:%d"%(exec_return))
                    exec_return = None
                break
        time.sleep(sleep_second)

if __name__=="__main__":
    exec_filename = sys.argv[1]
    sleep_second = sys.argv[2]
    detectGPU_exec(exec_filename, int(sleep_second))