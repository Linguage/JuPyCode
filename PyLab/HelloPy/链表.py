# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

curr = ListNode()

l1 = ListNode([1,2,4])

l2 = [1,3,4]

print(l1.val)

print(curr.next)
curr.next = l1
print(curr.next)
curr.next = l2
print(curr.next)
curr.val = 10
print(curr.val)
print(curr)
# a = l1.next
# print(a)