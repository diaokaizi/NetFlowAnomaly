import subprocess

# 定义FTP连接信息
ftp_server = "10.16.1.2"
ftp_user = "mahegu"
ftp_password = "6904b83623"
remote_file = "/t_fpc_flow_log_record/20241207/20241207.1425.tar.gz"  # 替换为你要获取的目录
local_dir = "/vdb2/NetFlowAnomaly/data"     # 替换为你存放文件的本地目录

# 构建lftp命令
lftp_command = f"lftp -u {ftp_user},{ftp_password} {ftp_server} -e 'get {remote_file} -o {local_dir}/; quit'"

# 执行lftp命令
try:
    subprocess.run(lftp_command, shell=True, check=True)
    print("文件下载成功！")
except subprocess.CalledProcessError as e:
    print(f"执行命令时出错: {e}")

    