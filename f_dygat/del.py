import pandas as pd
import os
import glob
from sklearn.preprocessing import LabelEncoder
import joblib  # 用于保存和加载编码器
import tarfile
# Define the input and output directories
import tempfile

def process(input_dir, output_dir, encoder_path):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    # 加载或创建 LabelEncoder
    if os.path.exists(encoder_path):
        ip_protocol_encoder = joblib.load(encoder_path)
    else:
        ip_protocol_encoder = LabelEncoder()

    # 处理每个文件
    print(f"发现{len(glob.glob(os.path.join(input_dir, '*')))}个原始文件")
    for tar_file_path in glob.glob(os.path.join(input_dir, '*')):
        print(f"开始处理{tar_file_path}")

        # 创建一个临时目录来解压缩文件
        with tarfile.open(tar_file_path, 'r') as tar_ref:
            # 解压缩到临时目录
            with tempfile.TemporaryDirectory() as temp_dir:
                tar_ref.extractall(temp_dir)
                # 处理解压缩后的每个 CSV 文件
                for file_path in glob.glob(os.path.join(temp_dir, '*')):
                    # 读取数据
                    df = pd.read_csv(file_path)
                    df = df[df['ethernet_protocol'] == 'IPv4']
                    df.replace(to_replace='\\N', value=0, inplace=True)
                    df.fillna(0, inplace=True)
                    # 更新 LabelEncoder
                    if 'ip_protocol' in df.columns:
                        ip_protocol_encoder.fit(df['ip_protocol'])
                    
                    # 使用 LabelEncoder 转换 ip_protocol
                    df['ip_protocol'] = ip_protocol_encoder.transform(df['ip_protocol'])

                    # 选择需要的列
                    processed_df = pd.DataFrame({
                        'start_time': df['start_time'],
                        'duration': df['duration'],
                        'flow_continued': df['flow_continued'],
                        'upstream_bytes': df['upstream_bytes'],
                        'downstream_bytes': df['downstream_bytes'],
                        'total_bytes': df['total_bytes'],
                        'upstream_packets': df['upstream_packets'],
                        'downstream_packets': df['downstream_packets'],
                        'total_packets': df['total_packets'],
                        'upstream_payload_bytes': df['upstream_payload_bytes'],
                        'downstream_payload_bytes': df['downstream_payload_bytes'],
                        'total_payload_bytes': df['total_payload_bytes'],
                        'upstream_payload_packets': df['upstream_payload_packets'],
                        'downstream_payload_packets': df['downstream_payload_packets'],
                        'total_payload_packets': df['total_payload_packets'],
                        'tcp_client_network_latency': df['tcp_client_network_latency'],
                        'tcp_client_network_latency_flag': df['tcp_client_network_latency_flag'],
                        'tcp_server_network_latency': df['tcp_server_network_latency'],
                        'tcp_server_network_latency_flag': df['tcp_server_network_latency_flag'],
                        'server_response_latency': df['server_response_latency'],
                        'server_response_latency_flag': df['server_response_latency_flag'],
                        'tcp_client_loss_bytes': df['tcp_client_loss_bytes'],
                        'tcp_server_loss_bytes': df['tcp_server_loss_bytes'],
                        'tcp_client_zero_window_packets': df['tcp_client_zero_window_packets'],
                        'tcp_server_zero_window_packets': df['tcp_server_zero_window_packets'],
                        'tcp_session_state': df['tcp_session_state'],
                        'tcp_established_success_flag': df['tcp_established_success_flag'],
                        'tcp_established_fail_flag': df['tcp_established_fail_flag'],
                        'established_sessions': df['established_sessions'],
                        'tcp_syn_packets': df['tcp_syn_packets'],
                        'tcp_syn_ack_packets': df['tcp_syn_ack_packets'],
                        'tcp_syn_rst_packets': df['tcp_syn_rst_packets'],
                        'tcp_client_packets': df['tcp_client_packets'],
                        'tcp_server_packets': df['tcp_server_packets'],
                        'tcp_client_retransmission_packets': df['tcp_client_retransmission_packets'],
                        'tcp_server_retransmission_packets': df['tcp_server_retransmission_packets'],
                        'ethernet_type': df['ethernet_type'],
                        'ip_locality_initiator': df['ip_locality_initiator'],
                        'ip_locality_responder': df['ip_locality_responder'],
                        'port_initiator': df['port_initiator'],
                        'port_responder': df['port_responder'],
                        'port_nat_initiator': df['port_nat_initiator'],
                        'port_nat_responder': df['port_nat_responder'],
                        'l7_protocol_id': df['l7_protocol_id'],
                        'application_category_id': df['application_category_id'],
                        'application_subcategory_id': df['application_subcategory_id'],
                        'application_id': df['application_id'],
                        'malicious_application_id': df['malicious_application_id'],
                        'country_id_initiator': df['country_id_initiator'],
                        'province_id_initiator': df['province_id_initiator'],
                        'city_id_initiator': df['city_id_initiator'],
                        'district_id_initiator': df['district_id_initiator'],
                        'continent_id_initiator': df['continent_id_initiator'],
                        'isp_id_initiator': df['isp_id_initiator'],
                        'asn_initiator': df['asn_initiator'],
                        'area_code_initiator': df['area_code_initiator'],
                        'longitude_initiator': df['longitude_initiator'],
                        'latitude_initiator': df['latitude_initiator'],
                        'country_id_responder': df['country_id_responder'],
                        'province_id_responder': df['province_id_responder'],
                        'city_id_responder': df['city_id_responder'],
                        'district_id_responder': df['district_id_responder'],
                        'continent_id_responder': df['continent_id_responder'],
                        'isp_id_responder': df['isp_id_responder'],
                        'asn_responder': df['asn_responder'],
                        'area_code_responder': df['area_code_responder'],
                        'longitude_responder': df['longitude_responder'],
                        'latitude_responder': df['latitude_responder'],
                        'tcp_client_retransmission_rate': df['tcp_client_retransmission_rate'],
                        'tcp_server_retransmission_rate': df['tcp_server_retransmission_rate'],
                        'ipv4_initiator': df['ipv4_initiator'],
                        'ipv4_responder': df['ipv4_responder']
                    })

                    # 保存处理后的数据到新的 CSV 文件
                    output_file_name = os.path.basename(file_path) + '.csv'
                    processed_df.to_csv(os.path.join(output_dir, output_file_name), index=False)

    # 保存 LabelEncoder
    joblib.dump(ip_protocol_encoder, encoder_path)

if __name__ =='__main__':
    # input_dir = '/vdb2/cst/GCN/data/cst'
    # output_dir = '/vdb2/cst/GCN/data/cst_out'
    # encoder_path = '/vdb2/cst/GCN/data/ip_protocol_encoder.pkl'
    # process(input_dir, output_dir, encoder_path)
    input_dir = '/vdb2/cst/GCN/data/ttt'
    output_dir = '/vdb2/cst/GCN/data/ttt_out'
    encoder_path = '/vdb2/cst/GCN/data/ttt_out/ip_protocol_encoder.pkl'
    process(input_dir, output_dir, encoder_path)
