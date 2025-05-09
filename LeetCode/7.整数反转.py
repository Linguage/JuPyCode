#
# @lc app=leetcode.cn id=7 lang=python3
#
# [7] 整数反转
#

# @lc code=start
class Solution:
    def reverse(self, x: int) -> int:
        x = int(x)
        ans = 0

        while x != 0:
            digit = x % 10  # 个位

            # Python3 的取模运算在 x 为负数时也会返回 [0, 9) 以内的结果，因此这里需要进行特殊判断
            if x < 0 and digit > 0:
                digit -= 10

            # 同理，Python3 的整数除法在 x 为负数时会向下（更小的负数）取整，因此不能写成 x //= 10
            x = (x - digit) // 10

            ans = ans * 10 + digit

        # 溢出则返回 0
        if ans < -2**31 or ans > 2**31 - 1:
            return 0
        return ans

# @lc code=end

