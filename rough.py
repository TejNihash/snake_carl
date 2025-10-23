class WordDictionary:

    class TrieNode:
        def __init__(self):
            self.data = {}
            self.is_word = False

    def __init__(self):
        self.root = self.TrieNode()
        

    def addWord(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.data:
                node.data[char]=self.TrieNode()
            
            node = node.data[char]
        node.status = True
        

    def search(self, word: str) -> bool:

        self.is_found = False

        node = self.root
        def dfs(node,word_sub):


            char = word_sub[0]
            if len(word_sub)==1:
                if char in node.data and node.data[char].status:
                    self.is_found = True
                    return
            if not node.data:
                return

            if char in node.data:
                node = node.data[char]
                dfs(node,word_sub[1:])
            elif char == '.':
                key_list = list(node.data.keys())
                for key in key_list:
                    dfs(node.data[key],word_sub[1:])
            
        dfs(node,word)
        return self.is_found


        
listt = ''
print(listt[0])