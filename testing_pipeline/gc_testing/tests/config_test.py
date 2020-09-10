import configparser

config = configparser.ConfigParser()
config.read('config.ini')
cfg_dict = dict()

print(config.sections())
for sec in config.sections():
    cfg_dict[sec] = dict()
    for key in config[sec]:
        cfg_dict[sec][key] = config[sec][key]

print(cfg_dict)