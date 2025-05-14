import unittest

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class TestListNode(unittest.TestCase):
    def test_init_with_values(self):
        # 测试初始化带有特定值的链表节点
        node = ListNode(1, ListNode(2, ListNode(3)))
        self.assertEqual(node.val, 1)
        self.assertIsNotNone(node.next)
        self.assertEqual(node.next.val, 2)
        self.assertIsNotNone(node.next.next)
        self.assertEqual(node.next.next.val, 3)
        
    def test_init_with_default_values(self):
        # 测试使用默认参数初始化链表节点
        node = ListNode()
        self.assertEqual(node.val, 0)
        self.assertIsNone(node.next)

    def test_init_with_next_none(self):
        # 测试初始化时next为None的链表节点
        node = ListNode(1, None)
        self.assertEqual(node.val, 1)
        self.assertIsNone(node.next)

    def test_init_with_single_node(self):
        # 测试初始化只有一个节点的链表
        node = ListNode(1)
        self.assertEqual(node.val, 1)
        self.assertIsNone(node.next)

if __name__ == '__main__':
    unittest.main()