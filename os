import os

path = os.path.join('folder', 'subfolder', 'file.txt')    #用于将多个路径片段组合成一个完整的路径
#  '..'上级目录  '.'当前目录    
os.path.exists(path)    #检查当前path路径是否存在


os.makedirs(name, mode=0o777, exist_ok=False)    #创建目录
#name目录名称   mode目录权限，默认是0o777（可以省略）    exist_ok如果为True，当目标目录已经存在时不会异常。如果为False默认，当目录已经存在时会异常
