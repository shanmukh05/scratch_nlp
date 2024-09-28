import logging

class Node:
    def __init__(self, word_idx, freq, left=None, right=None):
        self.word_idx = word_idx
        self.freq = freq
        self.huffman_code = []
        self.huffman_path = []
        self.left = left
        self.right = right

class HuffmanBTree():
    def __init__(self, vocab_freq_dict):
        self.logger = logging.getLogger(__name__)
        self.vocab = list(vocab_freq_dict.keys())
        self.freq = list(vocab_freq_dict.values())

        self.construct_tree()
        self.word_code, self.word_path = {}, {}
        self.generate_huffman_code(self.huffman_tree, [], [])
        self.logger.info("Generated Huffman code for all the vocabulary")
        self.left_huff_dict, self.right_huff_dict = {}, {}
        self.separate_left_right_path()

    def construct_tree(self):
        node_list = []
        for w,f in zip(self.vocab, self.freq):
            node_list.append(Node(w,f))

        count = len(self.vocab)
        while len(node_list) > 1:
            node_list = sorted(node_list, key = lambda a:a.freq, reverse=True)
        
            left = node_list[-2]
            right = node_list[-1]

            freq = left.freq + right.freq
            word_idx = count

            self.huffman_tree = Node(word_idx, freq, left, right)
            node_list = node_list[:-2]

            node_list.append(self.huffman_tree)
            count+=1
        self.logger.info("Constructed Huffman Tree")

    def generate_huffman_code(self, tree, code, path):
        
        if tree.left is None and tree.right is None:
            self.word_code[tree.word_idx] = code
            self.word_path[tree.word_idx] = path
        else:
            self.generate_huffman_code(tree.left, code+[1], path+[tree.word_idx])
            self.generate_huffman_code(tree.right, code+[0], path+[tree.word_idx])

    def separate_left_right_path(self):

        for widx, code, path in zip(self.word_code.keys(), self.word_code.values(), self.word_path.values()):
            left, right = [], []
            for c, p in zip(code, path):
                if c == 1:
                    left.append(p)
                else:
                    right.append(p)
            self.left_huff_dict[widx] = left
            self.right_huff_dict[widx] = right
        self.logger.info("Separated Paths into Left and Right branches")