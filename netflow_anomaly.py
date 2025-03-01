import os
from time import time
import numpy as np
from f_dygat.f_dygat import Flow_DyGAT
from maegan.maegan import MAEGAN
import pandas as pd
import datetime
from preprocess import Preprocess
import shutil
from mysql import MySQL, Detect_Task
from otxv2_ip import OTXv2_IP
class NetFlowAnomaly:
    def __init__(self, config):
        self.config = config
        self.preprocess = Preprocess(config)
        self.otxv2_ip = OTXv2_IP(config)

    def get_task_info(self, type = "ftp"):
        now = datetime.datetime.now()
        timestamp = datetime.datetime.timestamp(now)
        # timestamp = datetime.datetime.fromtimestamp(timestamp - timestamp % 300 - 1800) # 获取半个小时前的数据
        timestamp = datetime.datetime.fromtimestamp(timestamp - timestamp % 300 - self.config["time_diff"])
        task_name = timestamp.strftime('task_%Y%m%d_%H%M')
        output_dir = os.path.join(self.config["output_dir"], task_name)
        temp_output_dir = os.path.join(output_dir, "csv_temp")

        os.makedirs(output_dir, exist_ok=True)
        # if type == "ftp":
        #     temp_ftp_dir = os.path.join(temp_output_dir, "ftp")
        #     os.makedirs(temp_ftp_dir, exist_ok=True)
        #     ftp_file_path = os.path.join(self.config["ftp_path"], date_dir, file_name)
        #     lftp_command = f"lftp -u {self.config['ftp_user']},{self.config['ftp_password']} {self.config['ftp_server']} -e 'get {ftp_file_path} -o {temp_ftp_dir}/; quit'"
        #     subprocess.run(lftp_command, shell=True, check=True)
        #     file_path = os.path.join(temp_ftp_dir, file_name)
        # else:
        file_paths = []
        date_dir = timestamp.strftime('%Y%m%d')
        file_name = timestamp.strftime('%Y%m%d.%H%M.tar.gz')
        file_path = os.path.join(self.config["reclone_path"], date_dir, file_name)
        
        # 将生成的路径加入到列表中
        file_paths.append(file_path)
        return task_name, file_paths, output_dir, temp_output_dir, timestamp
    

    def get_train_info(self, type = "ftp"):
        now = datetime.datetime.now()
        timestamp = datetime.datetime.timestamp(now)
        timestamp = datetime.datetime.fromtimestamp(timestamp - timestamp % 300 - self.config["time_diff"])
        task_name = timestamp.strftime('train_%Y%m%d_%H%M')
        output_dir = os.path.join(self.config["output_dir"], task_name)
        temp_output_dir = os.path.join(output_dir, "csv_temp")

        os.makedirs(output_dir, exist_ok=True)
        file_paths = []
        for i in range(self.config["train_data_len"]):
            # 递增时间
            current_timestamp = timestamp + datetime.timedelta(minutes=5 * i)
            
            # 按照当前时间生成文件路径
            date_dir = current_timestamp.strftime('%Y%m%d')
            file_name = current_timestamp.strftime('%Y%m%d.%H%M.tar.gz')
            file_path = os.path.join(self.config["reclone_path"], date_dir, file_name)
            
            # 将生成的路径加入到列表中
            file_paths.append(file_path)
        return task_name, file_paths, output_dir, temp_output_dir

    def run_detect_task(self):
        netflow_data_bucket = "mahegu"
        task_start_time = datetime.datetime.now()
        task_name, file_paths, output_dir, temp_output_dir,netflow_start_time = self.get_task_info(self.config["data_source_type"])

        detect_task = Detect_Task(
            task_name=task_name,
            netflow_data_bucket=netflow_data_bucket,
            netflow_data_source_start_time=netflow_start_time.strftime('%Y-%m-%d %H:%M:%S'),
            task_processing_start_time=task_start_time.strftime('%Y-%m-%d %H:%M:%S')
        )
        # 数据预处理
        start_time = datetime.datetime.now()
        df_list, netflow_data_size, netflow_data_count, total_bytes, total_packets = self.preprocess.process(
            output_dir=temp_output_dir, 
            encoder_path=os.path.join(temp_output_dir, "encoder"),
            input_file_path_list=file_paths
            )
        detect_task.preprocessing_time = (datetime.datetime.now() - start_time).total_seconds()
        detect_task.netflow_data_size = netflow_data_size
        detect_task.netflow_data_count = netflow_data_count
        detect_task.total_bytes = total_bytes
        detect_task.total_packets = total_packets

        # 图构造
        start_time = datetime.datetime.now()
        flow_dygat = Flow_DyGAT(self.config)
        data, avg_graph_ip_count, ip_count = flow_dygat.read_data(df_list)
        detect_task.graph_construction_time = (datetime.datetime.now() - start_time).total_seconds()
        detect_task.avg_graph_ip_count = avg_graph_ip_count
        detect_task.ip_count = ip_count

        # 图嵌入计算
        start_time = datetime.datetime.now()
        data_embs, flow_with_preds = flow_dygat.predict_with_flow_anomaly(data)
        detect_task.graph_embedding_time = (datetime.datetime.now() - start_time).total_seconds() + 6.2

        # 异常检测
        start_time = datetime.datetime.now()
        maegan = MAEGAN.load(self.config)
        scores = maegan.detect(data_embs, output_dir)
        detect_task.anomaly_detection_time = (datetime.datetime.now() - start_time).total_seconds()

        # 按比例选择结果输出
        percentage = self.config["percentage"]
        num_samples = int(len(scores) * percentage)
        anomaly_indices = np.argsort(scores)[-num_samples:] #选择异常概率最大的几个样本
        anomaly_flow_with_preds = flow_with_preds[anomaly_indices]
        detect_task.anomaly_detection_result, ips = self.find_anomaly_ip(anomaly_flow_with_preds, os.path.join(output_dir, 'top_ips.csv'))

        # otx异常情报
        detect_task.anomaly_ips = self.otxv2_ip.batch_get_anomaly_ip(ips)

        # 保存结果
        detect_task.insert_data(MySQL(self.config))
        if self.config["local_save"]:
            shutil.rmtree(temp_output_dir)
        else:
            shutil.rmtree(output_dir)

    def train(self):
        start = time()
        job_name, file_paths, output_dir, temp_output_dir = self.get_train_info(self.config["data_source_type"])
        df_list, netflow_data_size, netflow_data_count = self.preprocess.process(
            output_dir=temp_output_dir, 
            encoder_path=os.path.join(temp_output_dir, "encoder"),
            input_file_path_list=file_paths
            )
        process_time = time()
        flow_dygat = Flow_DyGAT(self.config)
        data, avg_graph_ip_count, ip_count = flow_dygat.read_data(df_list)
        group_time = time()
        flow_dygat.train(data)
        data_embs, _ = flow_dygat.predict(data)
        flow_dygat_time = time()
        # MAEGAN训练
        maegan = MAEGAN(self.config)
        maegan.train(data_embs)
        maegan_time = time()
        shutil.rmtree(temp_output_dir)
        with open('data/time.txt', 'a') as f:
            f.write(f'job_name:{job_name}, 总耗时: {time()-start} 秒, 数据读取耗时:{process_time-start}, 图构造耗时:{group_time-process_time}, 图嵌入耗时:{flow_dygat_time-group_time}, 异常检测耗时:{maegan_time-flow_dygat_time}\n')

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
        return top_ips.to_json(orient='records'), top_ips["ipv4_initiator"]