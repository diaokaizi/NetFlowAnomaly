import pymysql

# 配置旧数据库和新数据库的连接信息
old_db_config = {
    'host': '127.0.0.1',  # 旧数据库地址
    'user': 'root',  # 旧数据库用户名
    'password': '123456',  # 旧数据库密码
    'database': 'netflow',  # 旧数据库名称
    'port': 3306  # 默认 MySQL 端口
}

new_db_config = {
    'host': '223.193.36.157',  # 旧数据库地址
    'user': 'root',  # 旧数据库用户名
    'password': '123456',  # 旧数据库密码
    'database': 'netflow',  # 旧数据库名称
    'port': 3306  # 默认 MySQL 端口
}

# 查询旧表数据
def fetch_old_data():
    connection = pymysql.connect(**old_db_config)
    cursor = connection.cursor(pymysql.cursors.DictCursor)
    query = "SELECT * FROM detect_task"  # 替换为旧表的表名
    cursor.execute(query)
    data = cursor.fetchall()
    cursor.close()
    connection.close()
    return data

# 插入数据到新表
def insert_into_new_table(data):
    connection = pymysql.connect(**new_db_config)
    cursor = connection.cursor()
    insert_query = """
    INSERT INTO detect_task (
        id, task_name, netflow_data_bucket, netflow_data_count, netflow_data_size,
        netflow_data_source_start_time, task_processing_start_time,
        preprocessing_time, graph_construction_time, graph_embedding_time,
        anomaly_detection_time, anomaly_detection_result, extend,
        create_time, update_time
    ) VALUES (
        %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s
    )
    """
    for row in data:
        # 如果需要转换字段格式，可以在这里处理，例如时间字段的转换
        values = (
            row['id'],
            row['task_name'],
            row['netflow_data_bucket'],
            row['netflow_data_count'],
            row['netflow_data_size'],
            row['netflow_data_source_start_time'],
            row['task_processing_start_time'],
            row['preprocessing_time'],
            row['graph_construction_time'],
            row['graph_embedding_time'],
            row['anomaly_detection_time'],
            row['anomaly_detection_result'],
            row['extend'],
            row['create_time'],
            row['update_time']
        )
        cursor.execute(insert_query, values)

    connection.commit()
    cursor.close()
    connection.close()

def main():
    # 获取旧表数据
    old_data = fetch_old_data()
    print(old_data)
    if old_data:
        print(f"Fetched {len(old_data)} rows from old table.")
        # 插入到新表
        insert_into_new_table(old_data)
        print("Data migration completed successfully.")
    else:
        print("No data found in old table.")

if __name__ == "__main__":
    main()
