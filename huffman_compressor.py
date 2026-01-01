"""
哈夫曼编码压缩工具
基于贪心算法实现文件压缩和解压功能
时间复杂度: O(nlogn)
"""

import heapq
import os
import pickle
import json
from collections import Counter
from typing import Dict, Tuple, Optional, List
import struct


class HuffmanNode:
    """哈夫曼树节点类"""

    def __init__(self, char: Optional[str] = None, freq: int = 0,
                 left: Optional['HuffmanNode'] = None,
                 right: Optional['HuffmanNode'] = None):
        """
        初始化哈夫曼节点

        Args:
            char: 字符（叶子节点）
            freq: 频率
            left: 左子节点
            right: 右子节点
        """
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other: 'HuffmanNode') -> bool:
        """用于堆排序的比较函数"""
        return self.freq < other.freq

    def __eq__(self, other: 'HuffmanNode') -> bool:
        """相等比较"""
        if other is None:
            return False
        return self.freq == other.freq

    def is_leaf(self) -> bool:
        """判断是否为叶子节点"""
        return self.left is None and self.right is None


class HuffmanEncoder:
    """哈夫曼编码器"""

    def __init__(self):
        """初始化编码器"""
        self.frequency_table: Dict[str, int] = {}
        self.huffman_tree: Optional[HuffmanNode] = None
        self.encoding_table: Dict[str, str] = {}
        self.decoding_table: Dict[str, str] = {}
        self.stats = {
            'original_size': 0,
            'compressed_size': 0,
            'compression_ratio': 0.0,
            'tree_size': 0,
            'unique_chars': 0
        }

    def build_frequency_table(self, data: bytes) -> Dict[str, int]:
        """
        构建字符频率表

        Args:
            data: 输入数据（字节）

        Returns:
            字符频率字典
        """
        self.frequency_table = Counter(data)
        self.stats['unique_chars'] = len(self.frequency_table)
        return self.frequency_table

    def build_huffman_tree(self) -> HuffmanNode:
        """
        构建哈夫曼树（贪心算法核心）
        使用最小堆选择频率最小的两个节点合并

        Returns:
            哈夫曼树根节点
        """
        if not self.frequency_table:
            raise ValueError("频率表为空，请先构建频率表")

        # 创建优先队列（最小堆）
        priority_queue = []
        for char, freq in self.frequency_table.items():
            node = HuffmanNode(char=char, freq=freq)
            heapq.heappush(priority_queue, node)

        # 贪心策略：每次选择频率最小的两个节点合并
        while len(priority_queue) > 1:
            # 取出频率最小的两个节点
            left = heapq.heappop(priority_queue)
            right = heapq.heappop(priority_queue)

            # 创建新节点，频率为两个子节点之和
            merged = HuffmanNode(
                freq=left.freq + right.freq,
                left=left,
                right=right
            )

            # 将新节点加入堆
            heapq.heappush(priority_queue, merged)

        # 堆中最后一个节点就是根节点
        self.huffman_tree = priority_queue[0] if priority_queue else None
        return self.huffman_tree

    def _generate_codes(self, node: Optional[HuffmanNode],
                        code: str = "", codes: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        递归生成编码表

        Args:
            node: 当前节点
            code: 当前编码字符串
            codes: 编码字典（递归使用）

        Returns:
            字符到编码的映射字典
        """
        if codes is None:
            codes = {}

        if node is None:
            return codes

        # 如果是叶子节点，记录编码
        if node.is_leaf():
            if code:  # 非空编码
                codes[node.char] = code
            else:  # 只有一个字符的情况
                codes[node.char] = "0"
            return codes

        # 递归遍历左子树（编码加"0"）
        if node.left:
            self._generate_codes(node.left, code + "0", codes)

        # 递归遍历右子树（编码加"1"）
        if node.right:
            self._generate_codes(node.right, code + "1", codes)

        return codes

    def build_encoding_table(self) -> Dict[str, str]:
        """
        构建编码表

        Returns:
            字符到编码的映射字典
        """
        if self.huffman_tree is None:
            raise ValueError("哈夫曼树未构建，请先构建树")

        self.encoding_table = self._generate_codes(self.huffman_tree)

        # 构建解码表（编码到字符的映射）
        self.decoding_table = {v: k for k, v in self.encoding_table.items()}

        return self.encoding_table

    def encode_data(self, data: bytes) -> Tuple[str, int]:
        """
        编码数据

        Args:
            data: 原始数据（字节）

        Returns:
            (编码后的二进制字符串, 填充位数)
        """
        encoded_bits = ""
        for byte in data:
            encoded_bits += self.encoding_table[byte]

        # 计算需要填充的位数（使总位数是8的倍数）
        padding = (8 - len(encoded_bits) % 8) % 8
        encoded_bits += "0" * padding

        return encoded_bits, padding

    def bits_to_bytes(self, bits: str) -> bytes:
        """
        将二进制字符串转换为字节

        Args:
            bits: 二进制字符串

        Returns:
            字节数据
        """
        byte_array = bytearray()
        for i in range(0, len(bits), 8):
            byte = bits[i:i + 8]
            byte_array.append(int(byte, 2))
        return bytes(byte_array)

    def bytes_to_bits(self, data: bytes, padding: int) -> str:
        """
        将字节转换为二进制字符串

        Args:
            data: 字节数据
            padding: 填充位数

        Returns:
            二进制字符串
        """
        bits = ""
        for byte in data:
            bits += format(byte, '08b')

        # 移除填充
        if padding > 0:
            bits = bits[:-padding]

        return bits

    def decode_data(self, bits: str) -> bytes:
        """
        解码数据

        Args:
            bits: 二进制字符串

        Returns:
            解码后的字节数据
        """
        decoded_bytes = bytearray()
        current_code = ""

        for bit in bits:
            current_code += bit
            if current_code in self.decoding_table:
                decoded_bytes.append(self.decoding_table[current_code])
                current_code = ""

        return bytes(decoded_bytes)

    def serialize_tree(self) -> bytes:
        """
        序列化哈夫曼树（用于保存到压缩文件）

        Returns:
            序列化后的字节数据
        """

        def serialize_node(node: Optional[HuffmanNode]) -> List:
            """递归序列化节点"""
            if node is None:
                return None

            if node.is_leaf():
                return {'char': node.char, 'freq': node.freq}
            else:
                return {
                    'freq': node.freq,
                    'left': serialize_node(node.left),
                    'right': serialize_node(node.right)
                }

        tree_dict = serialize_node(self.huffman_tree)
        tree_json = json.dumps(tree_dict, default=str)
        return tree_json.encode('utf-8')

    def deserialize_tree(self, data: bytes) -> HuffmanNode:
        """
        反序列化哈夫曼树

        Args:
            data: 序列化后的字节数据

        Returns:
            哈夫曼树根节点
        """
        tree_dict = json.loads(data.decode('utf-8'))

        def deserialize_node(node_dict: dict) -> Optional[HuffmanNode]:
            """递归反序列化节点"""
            if node_dict is None:
                return None

            if 'char' in node_dict:
                # 叶子节点
                return HuffmanNode(
                    char=node_dict['char'],
                    freq=node_dict['freq']
                )
            else:
                # 内部节点
                return HuffmanNode(
                    freq=node_dict['freq'],
                    left=deserialize_node(node_dict.get('left')),
                    right=deserialize_node(node_dict.get('right'))
                )

        self.huffman_tree = deserialize_node(tree_dict)
        return self.huffman_tree

    def get_tree_visualization(self, node: Optional[HuffmanNode] = None,
                               prefix: str = "", is_last: bool = True) -> str:
        """
        获取树的文本可视化

        Args:
            node: 当前节点
            prefix: 前缀字符串
            is_last: 是否为最后一个子节点

        Returns:
            树的可视化字符串
        """
        if node is None:
            node = self.huffman_tree

        if node is None:
            return "空树"

        result = ""
        connector = "└── " if is_last else "├── "

        if node.is_leaf():
            char_repr = f"'{chr(node.char)}'" if 32 <= node.char < 127 else f"0x{node.char:02x}"
            result += prefix + connector + f"{char_repr} (freq: {node.freq})\n"
        else:
            result += prefix + connector + f"Internal (freq: {node.freq})\n"
            new_prefix = prefix + ("    " if is_last else "│   ")

            if node.left:
                result += self.get_tree_visualization(
                    node.left, new_prefix, node.right is None
                )
            if node.right:
                result += self.get_tree_visualization(
                    node.right, new_prefix, True
                )

        return result

    def print_statistics(self):
        """打印统计信息"""
        print("\n" + "=" * 60)
        print("压缩统计信息")
        print("=" * 60)
        print(f"原始文件大小:     {self.stats['original_size']:,} 字节")
        print(f"压缩后大小:       {self.stats['compressed_size']:,} 字节")
        print(f"压缩率:           {self.stats['compression_ratio']:.2%}")
        print(f"节省空间:         {self.stats['original_size'] - self.stats['compressed_size']:,} 字节")
        print(f"唯一字符数:       {self.stats['unique_chars']}")
        print(f"树结构大小:       {self.stats['tree_size']:,} 字节")
        print("=" * 60)

        # 显示部分编码表
        print("\n编码表（前20个字符）:")
        print("-" * 60)
        sorted_chars = sorted(self.encoding_table.items(),
                              key=lambda x: self.frequency_table.get(x[0], 0),
                              reverse=True)
        for i, (char, code) in enumerate(sorted_chars[:20]):
            char_repr = f"'{chr(char)}'" if 32 <= char < 127 else f"0x{char:02x}"
            print(f"{char_repr:10} -> {code:20} (freq: {self.frequency_table[char]})")
        if len(sorted_chars) > 20:
            print(f"... 还有 {len(sorted_chars) - 20} 个字符")
        print("=" * 60)


class HuffmanCompressor:
    """哈夫曼压缩器主类"""

    def __init__(self):
        """初始化压缩器"""
        self.encoder = HuffmanEncoder()

    def compress_file(self, input_path: str, output_path: str) -> bool:
        """
        压缩文件

        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径

        Returns:
            是否成功
        """
        try:
            # 读取原始文件
            with open(input_path, 'rb') as f:
                original_data = f.read()

            if len(original_data) == 0:
                print("错误: 文件为空")
                return False

            # 构建频率表
            print("正在分析文件...")
            self.encoder.build_frequency_table(original_data)

            # 构建哈夫曼树
            print("正在构建哈夫曼树...")
            self.encoder.build_huffman_tree()

            # 构建编码表
            print("正在生成编码表...")
            self.encoder.build_encoding_table()

            # 编码数据
            print("正在编码数据...")
            encoded_bits, padding = self.encoder.encode_data(original_data)
            encoded_bytes = self.encoder.bits_to_bytes(encoded_bits)

            # 序列化树
            tree_data = self.encoder.serialize_tree()

            # 写入压缩文件
            print("正在写入压缩文件...")
            with open(output_path, 'wb') as f:
                # 文件头：填充位数（1字节）+ 树大小（4字节）+ 编码数据大小（4字节）
                # 使用小端序确保跨平台兼容性
                header = struct.pack('<BII', padding, len(tree_data), len(encoded_bytes))
                f.write(header)
                f.write(tree_data)
                f.write(encoded_bytes)

            # 计算统计信息
            compressed_size = os.path.getsize(output_path)
            self.encoder.stats['original_size'] = len(original_data)
            self.encoder.stats['compressed_size'] = compressed_size
            self.encoder.stats['compression_ratio'] = (
                    1 - compressed_size / len(original_data)
            ) if len(original_data) > 0 else 0
            self.encoder.stats['tree_size'] = len(tree_data)

            print(f"\n压缩完成: {input_path} -> {output_path}")
            return True

        except Exception as e:
            print(f"压缩失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def decompress_file(self, input_path: str, output_path: str) -> bool:
        """
        解压文件

        Args:
            input_path: 输入文件路径（压缩文件）
            output_path: 输出文件路径

        Returns:
            是否成功
        """
        try:
            # 读取压缩文件
            with open(input_path, 'rb') as f:
                # 读取文件头
                header = f.read(9)  # 1 + 4 + 4 = 9 字节
                if len(header) < 9:
                    print("错误: 压缩文件格式不正确")
                    return False

                # 使用小端序解析，与写入时保持一致
                padding, tree_size, data_size = struct.unpack('<BII', header)

                # 读取树数据
                tree_data = f.read(tree_size)
                if len(tree_data) != tree_size:
                    print("错误: 无法读取完整的树数据")
                    return False

                # 读取编码数据
                encoded_bytes = f.read(data_size)
                if len(encoded_bytes) != data_size:
                    print("错误: 无法读取完整的编码数据")
                    return False

            # 反序列化树
            print("正在重建哈夫曼树...")
            self.encoder.deserialize_tree(tree_data)

            # 重建编码表
            print("正在重建编码表...")
            self.encoder.build_encoding_table()

            # 解码数据
            print("正在解码数据...")
            encoded_bits = self.encoder.bytes_to_bits(encoded_bytes, padding)
            decoded_data = self.encoder.decode_data(encoded_bits)

            # 写入解压文件
            print("正在写入解压文件...")
            with open(output_path, 'wb') as f:
                f.write(decoded_data)

            print(f"\n解压完成: {input_path} -> {output_path}")
            return True

        except Exception as e:
            print(f"解压失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def show_tree(self):
        """显示哈夫曼树结构"""
        if self.encoder.huffman_tree is None:
            print("错误: 哈夫曼树未构建")
            return

        print("\n" + "=" * 60)
        print("哈夫曼树结构")
        print("=" * 60)
        print(self.encoder.get_tree_visualization())
        print("=" * 60)


def main():
    """主函数"""
    import sys

    print("=" * 60)
    print("哈夫曼编码压缩工具")
    print("基于贪心算法的文件压缩/解压程序")
    print("=" * 60)

    if len(sys.argv) < 2:
        print("\n使用方法:")
        print("  压缩: python huffman_compressor.py compress <输入文件> [输出文件]")
        print("  解压: python huffman_compressor.py decompress <输入文件> [输出文件]")
        print("  示例: python huffman_compressor.py compress test.txt test.huff")
        print("        python huffman_compressor.py decompress test.huff test_decompressed.txt")
        return

    compressor = HuffmanCompressor()
    command = sys.argv[1].lower()

    if command == 'compress':
        if len(sys.argv) < 3:
            print("错误: 请指定输入文件")
            return

        input_file = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) > 3 else input_file + '.huff'

        if not os.path.exists(input_file):
            print(f"错误: 文件不存在: {input_file}")
            return

        success = compressor.compress_file(input_file, output_file)
        if success:
            compressor.encoder.print_statistics()
            # 可选：显示树结构（对于小文件）
            if compressor.encoder.stats['original_size'] < 10000:
                try:
                    show_tree = input("\n是否显示哈夫曼树结构? (y/n): ").lower()
                    if show_tree == 'y':
                        compressor.show_tree()
                except (EOFError, KeyboardInterrupt):
                    # 非交互式环境，跳过用户输入
                    pass

    elif command == 'decompress':
        if len(sys.argv) < 3:
            print("错误: 请指定输入文件")
            return

        input_file = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) > 3 else input_file.replace('.huff', '_decompressed')

        if not os.path.exists(input_file):
            print(f"错误: 文件不存在: {input_file}")
            return

        compressor.decompress_file(input_file, output_file)

    else:
        print(f"错误: 未知命令 '{command}'")
        print("支持的命令: compress, decompress")


if __name__ == '__main__':
    main()
