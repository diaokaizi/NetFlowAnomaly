import pymysql
import json
class MySQL:
    def __init__(self, config):
        self.connc = pymysql.Connect(
                    host=config['mysql']['host'],
                    user=config['mysql']['user'],
                    password=config['mysql']['password'],
                    database=config['mysql']['database'],
                    port=int(config['mysql']['port']),
                    charset="utf8",
        )
    
    def execute(self, sql:str):
        cursor = self.connc.cursor()
        try:
            cursor.execute(sql)
            self.connc.commit()
            data = cursor.fetchall()
        except Exception as e:
            self.connc.rollback()
            print(e)
        return data
    
        
    def execute_many(self, sql:str, data:list):
        cursor = self.connc.cursor()
        try:
            cursor.executemany(sql, data)
            self.connc.commit()
        except Exception as e:
            self.connc.rollback()
            print(e)

class Detect_Task:
    def __init__(self, task_name, netflow_data_bucket, netflow_data_source_start_time, task_processing_start_time):
        """
        初始化Detect_Task对象，接收各项任务相关参数
        """
        self.task_name = task_name
        self.netflow_data_bucket = netflow_data_bucket
        self.netflow_data_count = 0
        self.netflow_data_size = 0
        self.avg_graph_ip_count = 0
        self.ip_count = 0
        self.netflow_data_source_start_time = netflow_data_source_start_time
        self.task_processing_start_time = task_processing_start_time
        self.preprocessing_time = 0
        self.graph_construction_time = 0
        self.graph_embedding_time = 0
        self.anomaly_detection_time = 0
        self.anomaly_detection_result = ""
        self.extend = ""
        self.anomaly_ips = []
        self.total_bytes = 0
        self.total_packets = 0

    def insert_detect_task_sql(self):
        """
        构造插入数据的SQL语句
        """
        sql = """
            INSERT INTO detect_task (
                task_name,
                netflow_data_bucket,
                netflow_data_count,
                netflow_data_size,
                avg_graph_ip_count,
                ip_count,
                netflow_data_source_start_time,
                task_processing_start_time,
                preprocessing_time,
                graph_construction_time,
                graph_embedding_time,
                anomaly_detection_time,
                anomaly_detection_result,
                extend,
                anomaly_ip_count,
                total_bytes,
                total_packets
            ) VALUES ('%s','%s',%s,%s,%s,%s,'%s','%s',%s,%s,%s,%s,'%s','%s', %s,%s,%s)
        """
        data = (
            self.task_name,
            self.netflow_data_bucket,
            self.netflow_data_count,
            self.netflow_data_size,
            self.avg_graph_ip_count,
            self.ip_count,
            self.netflow_data_source_start_time,
            self.task_processing_start_time,
            self.preprocessing_time,
            self.graph_construction_time,
            self.graph_embedding_time,
            self.anomaly_detection_time,
            self.anomaly_detection_result,
            self.extend,
            len(self.anomaly_ips),
            self.total_bytes,
            self.total_packets,
        )
        return sql % data

    def insert_ip_anomaly_details_sql(self):
        sql = """
            INSERT INTO ip_anomaly_details (ip, anomaly_count, anomaly_details)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE 
                anomaly_count = VALUES(anomaly_count),
                anomaly_details = VALUES(anomaly_details);
        """
        return sql

    def insert_data(self, mysql_obj):
        """
        使用传入的MySQL类对象执行插入数据操作
        """
        sql = self.insert_detect_task_sql()
        mysql_obj.execute(sql)

        sql = self.insert_ip_anomaly_details_sql()
        mysql_obj.execute_many(
            sql,
            [(ip, len(details), json.dumps(details)) for ip, details in self.anomaly_ips]
        )


# CREATE TABLE IF NOT EXISTS detect_task (
#     -- 主键，使用自增长整数类型作为主键，唯一标识每条记录
#     id INT AUTO_INCREMENT PRIMARY KEY,
#     -- 任务名，使用合适长度的可变长字符串类型，根据实际任务名长度预估来确定varchar长度，这里假设最长50个字符
#     task_name VARCHAR(50) NOT NULL,
#     -- netflow数据桶
#     netflow_data_bucket VARCHAR(50) NOT NULL,
#     -- netflow数据量，整型记录数据量情况
#     netflow_data_count INT NOT NULL,
#     -- netflow数据大小，以字节等单位衡量，使用整型记录
#     netflow_data_size INT NOT NULL,
#     -- netflow数据源起始时间，设置为timestamp类型，并添加索引方便按此时间查询
#     netflow_data_source_start_time TIMESTAMP NOT NULL,
#     -- 任务处理开始时间，timestamp类型记录任务何时开始处理
#     task_processing_start_time TIMESTAMP NOT NULL,
#     -- 预处理耗时，比如可以以秒为单位，使用浮点型记录耗时情况
#     preprocessing_time FLOAT NOT NULL,
#     -- 图构造耗时，同样以合适时间单位对应的数值类型记录，这里用浮点型示例
#     graph_construction_time FLOAT NOT NULL,
#     -- 图嵌入耗时，浮点型记录该阶段耗时
#     graph_embedding_time FLOAT NOT NULL,
#     -- 异常检测耗时，浮点型表示耗时长短
#     anomaly_detection_time FLOAT NOT NULL,
#     -- 异常检测结果，以字符串形式保存json序列化后的内容，使用较长的可变长字符串类型，根据实际可能的最长长度预估设置长度，这里假设最长5000字符
#     anomaly_detection_result TEXT NOT NULL,
#     -- 备用字段，以字符串形式保存其他相关信息，同样预估合适长度，这里假设最长2000字符
#     extend TEXT NOT NULL,
#     -- 创建时间，设置默认值为当前时间戳，插入记录时自动记录创建时间
#     create_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
#     -- 更新时间，默认当前时间戳且在记录更新时自动更新该时间戳
#     update_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
# ) ENGINE=InnoDB DEFAULT CHARSET=utf8;

# CREATE INDEX idx_netflow_data_source_start_time ON detect_task (netflow_data_source_start_time);
