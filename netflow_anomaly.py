import os
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import numpy as np
from f_dygat.read_data import Dataset
from f_dygat.f_dygat import Flow_DyGAT
from f_dygat.utils import set_seed
from maegan.maegan import MAEGAN
import pandas as pd
import datetime
from preprocess import Preprocess
import shutil

class NetFlowAnomaly:
    def __init__(self, config):
        self.config = config
        self.preprocess = Preprocess(config)

    def get_job_info(self):
        now = datetime.datetime.now()
        timestamp = datetime.datetime.timestamp(now)
        timestamp = datetime.datetime.fromtimestamp(timestamp - timestamp % 300 - 1800) # 获取半个小时前的数据
        date_dir = timestamp.strftime('%Y%m%d')
        file_name = timestamp.strftime('%Y%m%d.%H%M.tar.gz')
        file_path = os.path.join(self.config["netflow_datasource_path"], date_dir, file_name)
        job_name = timestamp.strftime('job_%Y%m%d_%H%M')
        return file_path, job_name
    
    def run_detect_job(self):
        t0=time()
        file_path, job_name = self.get_job_info()
        output_dir = os.path.join(self.config["output_dir"], job_name)
        os.makedirs(output_dir, exist_ok=True)
        csv_temp_output_dir = os.path.join(output_dir, "csv_temp")
        self.preprocess.process(
            output_dir=csv_temp_output_dir, 
            encoder_path=os.path.join(csv_temp_output_dir, "csv_temp"),
            input_file_path_list=[file_path]
            )

        # Flow_DyGAT
        # 图特征提取
        flow_dygat = Flow_DyGAT(self.config)
        # 构造图结构
        data = flow_dygat.read_data(csv_temp_output_dir)
        # 图嵌入计算
        data_embs, flow_with_preds = flow_dygat.predict_with_flow_anomaly(data)

        # MAEGAN
        # 加载模型
        maegan = MAEGAN.load(self.config)
        # 异常检测
        scores = maegan.detect(data_embs, output_dir)
        # 按比例选择结果输出
        percentage = self.config["percentage"]
        num_samples = int(len(scores) * percentage)
        anomaly_indices = np.argsort(scores)[-num_samples:] #选择异常概率最大的几个样本
        anomaly_flow_with_preds = flow_with_preds[anomaly_indices]
        self.find_anomaly_ip(anomaly_flow_with_preds, os.path.join(output_dir, 'top_ips.csv'))

        shutil.rmtree(csv_temp_output_dir)
        print('总耗时', time()-t0)

    def train(self, input_dir):
        # Flow_DyGAT训练
        flow_dygat = Flow_DyGAT(self.config)
        data = flow_dygat.read_data(input_dir)
        flow_dygat.train(data)
        data_embs, _ = flow_dygat.predict(data)
        # MAEGAN训练
        maegan = MAEGAN(self.config)
        maegan.train(data_embs)

    def find_anomaly_ip(self, anomaly_flow_with_preds, save_path):
        df = pd.DataFrame(
            anomaly_flow_with_preds.reshape(-1, 5),  # 展平到 2D 形状，每个样本有 5 个属性
            columns=["ipv4_initiator", "ipv4_responder", "start_time", "0_pred", "1_pred"]
        )
        df["1_pred"] = pd.to_numeric(df["1_pred"], errors="coerce")

        top_ips = (
            df.groupby("ipv4_initiator")["1_pred"]
            .mean()
            .nlargest(10)  # 获取 1_pred 累计值最大的前 top_n 个 IP
            .reset_index()
        )
        print(top_ips)
        top_ips.to_csv(save_path, index=False)
        print(f"Top IPs saved to {save_path}")