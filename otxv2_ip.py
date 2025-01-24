from OTXv2 import OTXv2
import IndicatorTypes

class OTXv2_IP:
    def __init__(self, config):
        self.API_KEY = config['otx_key']
        self.OTX_SERVER = 'https://otx.alienvault.com/'
        self.otx = OTXv2(self.API_KEY, server=self.OTX_SERVER)

    @staticmethod
    def getValue(results, keys):
        if type(keys) is list and len(keys) > 0:

            if type(results) is dict:
                key = keys.pop(0)
                if key in results:
                    return OTXv2_IP.getValue(results[key], keys)
                else:
                    return None
            else:
                if type(results) is list and len(results) > 0:
                    return OTXv2_IP.getValue(results[0], keys)
                else:
                    return results
        else:
            return results
    
    def get_ip_detail(self, ip):
        alerts = []
        result = self.otx.get_indicator_details_by_section(IndicatorTypes.IPv4, ip, 'general')
        # Return nothing if it's in the whitelist
        validation = self.getValue(result, ['validation'])
        if not validation:
            pulses = self.getValue(result, ['pulse_info', 'pulses'])
            if pulses:
                for pulse in pulses:
                    info = {}
                    if 'name' in pulse:
                        info['name'] = pulse['name']
                    if 'modified' in pulse:
                        info['modified'] = pulse['modified']
                    if 'created' in pulse:
                        info['created'] = pulse['created']
                    alerts.append(info)

        return alerts
    
    def batch_get_anomaly_ip(self, ips):
        results = []
        try:
            for ip in ips:
                # 调用 get_ip_detail 方法获取单个 IP 的异常情报
                details = self.get_ip_detail(ip)
                # 如果异常情报非空，则将 IP 和情报加入结果
                if len(details) != 0:
                    results.append([ip, details])
        except Exception as e:
            print(f"Error batch_get_anomaly_ip: {e}")
            return []
        return results

