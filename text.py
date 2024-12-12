from OTXv2 import OTXv2

otx = OTXv2("2d77d87738dbd8b0fe8ecb538a7b129aed13576f5dda2813b40c9b720aa3e2f2")
name = 'Test Pulse'
indicators = [
    {'indicator': '65.49.20.68', 'type': 'IPv4'},
]
response = otx.create_pulse(name=name ,public=True ,indicators=indicators ,tags=[] , references=[])
print(str(response))