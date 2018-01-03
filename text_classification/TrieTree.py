#!/usr/bin/python
# -*- coding: utf-8 -*-


import sys
reload(sys)
sys.setdefaultencoding('utf-8')

'''
key和text都得是unicode编码模式
'''


class Node:
    def __init__(self):
        self.value = None
        self.children = {}    # children is of type {char, Node}

class Trie:
    def __init__(self):
        self.root = Node()

    def insert(self, key):      # key type is unicode, not a str
        # key should be a low-case string, this must be checked here!
        key = unicode(key)
        node = self.root
        for char in key:
            if char not in node.children:
                child = Node()
                node.children[char] = child
                node = child
            else:
                node = node.children[char]
        node.value = key

    def search(self, key):
        key = unicode(key)
        node = self.root
        for char in key:
            if char not in node.children:
                return None
            else:
                node = node.children[char]
        return node.value

    def searchphrase(self, text):
        text = unicode(text)
        length = len(text)
        node = self.root
        dict = {}
        for character in text:
            if character not in node.children:
                # print node.value
                if node.value != None:
                    dict[node.value] = node.value
                node = self.root
                if character in node.children:
                    node = node.children[character]
            else:
                node = node.children[character]

        if node.value != None:
            dict[node.value] = node.value

        return dict

    def display_node(self, node):
        if (node.value != None):
            print node.value
        for char in 'abcdefghijklmnopqrstuvwxyz':
            if char in node.children:
                self.display_node(node.children[char])
        return

    def display(self):
        self.display_node(self.root)
