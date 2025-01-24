import argparse
import os
import sys
import glob
import shutil
import torch
from f_dygat.utils import set_seed
import pandas as pd
import yaml
from netflow_anomaly import NetFlowAnomaly

def parse():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config', default="config.yaml", type=str, help="Path to the YAML configuration file")
    parser.add_argument('--mode', default="test", type=str, help="运行模式")
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    return config, args.mode
# te={0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0, 24: 0.0, 25: 0.0, 26: 0.0, 27: 0.0, 28: 0.017871144729886424, 29: 0.0, 30: 0.0, 31: 0.0, 32: 0.006274654548248016, 33: 0.0, 34: 0.0, 35: 0.0, 36: 0.0, 37: 0.03762527233115466, 38: 0.0666615330678887, 39: 0.13764579497983095, 40: 0.13090638165804516, 41: 0.05306513409961686, 42: 0.2753630579847618, 43: 0.05311302681992337, 44: 0.00934037280301648, 45: 0.32935394784746264, 46: 0.41141163499936306, 47: 0.2013868842659629, 48: 0.0, 49: 0.0005695790449002075, 50: 0.3649125313746678, 51: 0.0, 52: 0.0, 53: 0.0, 54: 0.0, 55: 0.0, 56: 0.0, 57: 0.0, 58: 0.0, 59: 0.0, 60: 0.06147427372094644, 61: 0.0, 62: 0.001628352490421456, 63: 0.0067519678681203146, 64: 0.001628352490421456, 65: 0.0, 66: 0.0, 67: 0.0, 68: 0.0022932490869574966, 69: 0.0, 70: 0.0, 71: 0.0, 72: 0.0, 73: 0.0, 74: 0.0, 75: 0.0, 76: 0.0, 77: 0.0, 78: 0.0, 79: 0.0, 80: 0.0, 81: 0.0, 82: 0.008003505731122181, 83: 0.0, 84: 0.0, 85: 0.0, 86: 0.0, 87: 0.0, 88: 0.0, 89: 0.026875588038535996, 90: 0.0, 91: 0.0, 92: 0.0, 93: 0.0, 94: 0.0, 95: 0.0, 96: 0.0, 97: 0.02112006509768155, 98: 0.0, 99: 0.0, 100: 0.0, 101: 0.0, 102: 0.0, 103: 0.0, 104: 0.0, 105: 0.0, 106: 0.0, 107: 0.0, 108: 0.0, 109: 0.0, 110: 0.0, 111: 0.0, 112: 0.0, 113: 0.0, 114: 0.0, 115: 0.0, 116: 0.0, 117: 0.0, 118: 0.0, 119: 0.0, 120: 0.0, 121: 0.0, 122: 0.0, 123: 0.0, 124: 0.053746683426810465, 125: 0.0, 126: 0.0, 127: 0.0, 128: 0.0, 129: 0.0, 130: 0.0, 131: 0.0, 132: 0.0, 133: 0.0, 134: 0.0, 135: 0.0, 136: 0.0, 137: 0.0, 138: 0.0, 139: 0.0, 140: 0.0, 141: 0.0, 142: 0.0, 143: 0.0, 144: 0.0, 145: 0.0}
# print(max(te.values()))

def find_anomaly_ip(anomaly_flow_with_preds, save_path):
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


if __name__ =='__main__':
    os.chdir('/root/NetFlowAnomaly')
    
    config, mode=parse()
    set_seed(config["seed"]) # 设置随机数种子

    device=torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    config["device"]=device
    netFlow_anomaly = NetFlowAnomaly(config)
    # res = netFlow_anomaly.otxv2_ip.batch_get_anomaly_ip(["173.236.163.15", "147.185.133.29", "139.162.54.180"])
    res = netFlow_anomaly.otxv2_ip.batch_get_anomaly_ip(["147.185.133.29"])
    print(res)
