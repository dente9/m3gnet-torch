from setuptools import setup, find_packages

setup(
    name='m3gnet-c',  # 包的名称，可以和文件夹不同
    version='0.1.0',
    packages=find_packages(), # 自动查找所有包含 __init__.py 的目录作为包
    author='Your Name', # 你的名字
    author_email='your.email@example.com', # 你的邮箱
)