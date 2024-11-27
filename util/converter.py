import torch
import unicodedata as ud

# keys = ' 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ@!"#$%&\'[]()+,-./:;=?´ÉÈ'
# keys = '0123456789.'
with open('./util/codec.txt', 'r') as f:
    keys = f.readlines()[0]


class StringLabelConverter(object):
    """Convert between str and label.
    NOTE:
        Insert `blank` to the alphabet for CTC.
    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text):
        """
        将给定的文本列表转换为它们在字典中的索引，并对它们进行适当的填充以使它们具有相同的长度。

        参数:
        text (list of str): 待编码的文本列表，每个元素代表一个文本项。

        返回:
        torch.LongTensor: 编码后的文本张量。
        torch.LongTensor: 每个文本项原始长度的张量。
        """
        # 初始化长度列表和结果列表
        length = []
        result = []

        # 遍历每个文本项
        for item in text:
            # 检查文本项是否以阿拉伯字符开头，如果是，则将其反转
            if len(item) > 0 and 'ARABIC' in ud.name(item[0]):
                item = item[::-1]

            # 记录每个文本项的长度
            length.append(len(item))

            # 初始化字符索引列表
            r = []
            for char in item:
                # 获取字符在字典中的索引，并添加到列表中
                index = self.dict[char]
                r.append(index)
            result.append(r)

        # 初始化最大长度为0
        max_len = 0
        # 遍历结果列表，找到最大的文本项长度
        for r in result:
            if len(r) > max_len:
                max_len = len(r)

        # 初始化临时结果列表
        result_temp = []
        # 对每个文本项进行填充，使其长度与最大长度相同
        for r in result:
            for i in range(max_len - len(r)):
                r.append(0)
            result_temp.append(r)

        # 将临时结果赋值给text变量
        text = result_temp
        # 返回编码后的文本张量和长度张量
        return torch.LongTensor(text), torch.LongTensor(length)

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.LongTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.LongTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(),
                                                                                                         length)
            if raw:
                output = ''.join([self.alphabet[i - 1] for i in t])
                if len(output) > 0 and 'ARABIC' in ud.name(output[0]):
                    output = output[::-1]
                return output
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])

                output = ''.join(char_list)
                if len(output) > 0 and 'ARABIC' in ud.name(output[0]):
                    output = output[::-1]
                return output
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(
                t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(t[index:index + l], torch.LongTensor([l]), raw=raw))
                index += l
            return texts
