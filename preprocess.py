import pandas as pd
import os
import glob
from sklearn.preprocessing import LabelEncoder
import joblib  # 用于保存和加载编码器
import tarfile
# Define the input and output directories
import tempfile


def process_file(file_path, ip_protocol_encoder):
    df = pd.read_csv(file_path)
    df = df[df['ethernet_protocol'] == 'IPv4']
    df.replace(to_replace='\\N', value=0, inplace=True)
    df.fillna(0, inplace=True)

    if 'ip_protocol' in df.columns:
        df['ip_protocol'] = ip_protocol_encoder.transform(df['ip_protocol'])

    # 选择需要的列
    processed_df = df[[
        'start_time', 'duration', 'flow_continued', 'upstream_bytes', 'downstream_bytes',
        'total_bytes', 'upstream_packets', 'downstream_packets', 'total_packets', 'upstream_payload_bytes',
        'downstream_payload_bytes', 'total_payload_bytes', 'upstream_payload_packets', 'downstream_payload_packets',
        'total_payload_packets', 'tcp_client_network_latency', 'tcp_client_network_latency_flag',
        'tcp_server_network_latency', 'tcp_server_network_latency_flag', 'server_response_latency',
        'server_response_latency_flag', 'tcp_client_loss_bytes', 'tcp_server_loss_bytes',
        'tcp_client_zero_window_packets', 'tcp_server_zero_window_packets', 'tcp_session_state',
        'tcp_established_success_flag', 'tcp_established_fail_flag', 'established_sessions', 'tcp_syn_packets',
        'tcp_syn_ack_packets', 'tcp_syn_rst_packets', 'tcp_client_packets', 'tcp_server_packets',
        'tcp_client_retransmission_packets', 'tcp_server_retransmission_packets', 'ethernet_type',
        'ip_locality_initiator', 'ip_locality_responder', 'port_initiator', 'port_responder',
        'port_nat_initiator', 'port_nat_responder', 'l7_protocol_id', 'application_category_id',
        'application_subcategory_id', 'application_id', 'malicious_application_id', 'country_id_initiator',
        'province_id_initiator', 'city_id_initiator', 'district_id_initiator', 'continent_id_initiator',
        'isp_id_initiator', 'asn_initiator', 'area_code_initiator', 'longitude_initiator', 'latitude_initiator',
        'country_id_responder', 'province_id_responder', 'city_id_responder', 'district_id_responder',
        'continent_id_responder', 'isp_id_responder', 'asn_responder', 'area_code_responder',
        'longitude_responder', 'latitude_responder', 'tcp_client_retransmission_rate', 'tcp_server_retransmission_rate',
        'ipv4_initiator', 'ipv4_responder'
    ]]

    return processed_df, os.path.basename(file_path)

class Preprocess:
    def __init__(self, config):
        self.config = config

    def process(self, output_dir, encoder_path, input_dir = None, input_file_path_list = []):
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        # 加载或创建 LabelEncoder
        if os.path.exists(encoder_path):
            ip_protocol_encoder = joblib.load(encoder_path)
        else:
            ip_protocol_encoder = LabelEncoder()

        # 处理每个文件
        file_path_list = []
        if input_dir is not None:
            file_path_list.extend(glob.glob(os.path.join(input_dir, '*')))
        file_path_list.extend(input_file_path_list)

        print(f"发现{len(file_path_list)}个原始文件")

        result = []
        netflow_data_size = 0
        netflow_data_count = 0
        for tar_file_path in file_path_list:
            print(f"开始解压{tar_file_path}")

            # 创建一个临时目录来解压缩文件
            with tarfile.open(tar_file_path, 'r') as tar_ref:
                # 解压缩到临时目录
                with tempfile.TemporaryDirectory() as temp_dir:
                    tar_ref.extractall(temp_dir)
                    # 处理解压缩后的每个 CSV 文件
                    for file_path in glob.glob(os.path.join(temp_dir, '*')):
                        print(f"开始处理解压缩后的每个 CSV 文件{file_path}")
                        # 读取数据
                        df = pd.read_csv(file_path)
                        netflow_data_size = netflow_data_size + os.path.getsize(file_path)
                        netflow_data_count = netflow_data_count + len(df)

                        df = df[df['ethernet_protocol'] == 'IPv4']
                        # 更新 LabelEncoder
                        if 'ip_protocol' in df.columns:
                            ip_protocol_encoder.fit(df['ip_protocol'])
                        
                        # 使用 LabelEncoder 转换 ip_protocol
                        df['ip_protocol'] = ip_protocol_encoder.transform(df['ip_protocol'])
                        df.replace(to_replace='\\N', value=0, inplace=True)
                        df.fillna(0, inplace=True)
                        # 选择需要的列（避免重复创建 DataFrame）
                        required_columns = [
                            'start_time', 'duration', 'flow_continued', 'upstream_bytes', 'downstream_bytes', 'total_bytes',
                            'upstream_packets', 'downstream_packets', 'total_packets', 'upstream_payload_bytes', 
                            'downstream_payload_bytes', 'total_payload_bytes', 'upstream_payload_packets', 
                            'downstream_payload_packets', 'total_payload_packets', 'tcp_client_network_latency', 
                            'tcp_client_network_latency_flag', 'tcp_server_network_latency', 'tcp_server_network_latency_flag', 
                            'server_response_latency', 'server_response_latency_flag', 'tcp_client_loss_bytes', 
                            'tcp_server_loss_bytes', 'tcp_client_zero_window_packets', 'tcp_server_zero_window_packets', 
                            'tcp_session_state', 'tcp_established_success_flag', 'tcp_established_fail_flag', 
                            'established_sessions', 'tcp_syn_packets', 'tcp_syn_ack_packets', 'tcp_syn_rst_packets', 
                            'tcp_client_packets', 'tcp_server_packets', 'tcp_client_retransmission_packets', 
                            'tcp_server_retransmission_packets', 'ethernet_type', 'ip_locality_initiator', 
                            'ip_locality_responder', 'port_initiator', 'port_responder', 'port_nat_initiator', 
                            'port_nat_responder', 'l7_protocol_id', 'application_category_id', 'application_subcategory_id', 
                            'application_id', 'malicious_application_id', 'country_id_initiator', 'province_id_initiator', 
                            'city_id_initiator', 'district_id_initiator', 'continent_id_initiator', 'isp_id_initiator', 
                            'asn_initiator', 'area_code_initiator', 'longitude_initiator', 'latitude_initiator', 
                            'country_id_responder', 'province_id_responder', 'city_id_responder', 'district_id_responder', 
                            'continent_id_responder', 'isp_id_responder', 'asn_responder', 'area_code_responder', 
                            'longitude_responder', 'latitude_responder', 'tcp_client_retransmission_rate', 
                            'tcp_server_retransmission_rate', 'ipv4_initiator', 'ipv4_responder'
                        ]

                        processed_df = df[required_columns]

                        result.append(processed_df)
                        # 保存处理后的数据到新的 CSV 文件
                        # output_file_name = os.path.basename(file_path) + '.csv'
                        # print(processed_df.shape)
                        # processed_df.to_csv(os.path.join(output_dir, output_file_name), index=False)
        
        return result, netflow_data_size, netflow_data_count

if __name__ =='__main__':
    # input_dir = '/vdb2/cst/GCN/data/cst'
    # output_dir = '/vdb2/cst/GCN/data/cst_out'
    # encoder_path = '/vdb2/cst/GCN/data/ip_protocol_encoder.pkl'
    # process(input_dir, output_dir, encoder_path)
    # input_dir = '/vdb2/cst/GCN/data/ttt'
    # output_dir = '/vdb2/cst/GCN/data/ttt_out'
    # encoder_path = '/vdb2/cst/GCN/data/ttt_out/ip_protocol_encoder.pkl'
    # process(input_dir, output_dir, encoder_path)
    pass
