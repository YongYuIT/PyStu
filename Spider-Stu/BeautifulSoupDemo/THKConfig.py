# 为了避免侵权问题，所有示例中地址、配置等均加密

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import base64


def str_to_bytes(key_str):
    key = key_str.encode('utf-8')
    key = key.ljust(32, b'\0')  # 使用0填充至32字节
    return key


currentKey = str_to_bytes('741236589')


# PKCS7填充函数
def pkcs7_pad(data):
    block_size = 16
    padding_size = block_size - (len(data) % block_size)
    return data + bytes([padding_size] * padding_size)


# PKCS7去除填充函数
def pkcs7_unpad(data):
    padding_size = data[-1]
    return data[:-padding_size]


# 加密函数
def encrypt(message, key=currentKey):
    backend = default_backend()
    cipher = Cipher(algorithms.AES(key), modes.ECB(), backend=backend)
    encryptor = cipher.encryptor()
    # 将字符串编码为字节形式并进行PKCS7填充
    message_bytes = pkcs7_pad(message.encode('utf-8'))
    ct = encryptor.update(message_bytes) + encryptor.finalize()
    # 嵌套一层base64编码
    return base64.b64encode(ct).decode('utf-8')


# 解密函数
def decrypt(ciphertext, key=currentKey):
    # base64解码
    decoded_message = base64.b64decode(ciphertext)
    backend = default_backend()
    cipher = Cipher(algorithms.AES(key), modes.ECB(), backend=backend)
    decryptor = cipher.decryptor()
    pt = decryptor.update(decoded_message) + decryptor.finalize()
    # 去除PKCS7填充并将字节形式解码为字符串
    decrypted_message = pkcs7_unpad(pt).decode('utf-8')
    return decrypted_message
