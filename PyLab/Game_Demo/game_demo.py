# 导入pygame模块，用于绘制图形界面
import pygame
# 导入random模块，用于生成随机数
import random
# 初始化pygame
pygame.init()
# 设置窗口的大小，单位是像素
screen_width = 600
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
# 设置窗口的标题
pygame.display.set_caption('贪吃蛇')
# 设置游戏的帧率，即每秒刷新的次数
fps = 5
# 创建一个时钟对象，用于控制游戏的速度
clock = pygame.time.Clock()
# 定义一些颜色，用于绘制不同的元素
black = (0, 0, 0) # 背景色
white = (255, 255, 255) # 蛇的颜色
red = (255, 0, 0) # 食物的颜色
green = (0, 255, 0) # 得分的颜色
# 定义蛇的初始位置，用一个列表表示，列表中的每个元素是一个坐标元组
snake = [(300, 300), (280, 300), (260, 300)]
# 定义蛇的初始方向，用一个字符串表示，可以是'up'，'down'，'left'，'right'
direction = 'right'
# 定义蛇的移动速度，即每次移动的像素数
speed = 10
# 定义食物的初始位置，用一个坐标元组表示
food = (random.randrange(0, screen_width, speed), random.randrange(0, screen_height, speed))
# 定义游戏的状态，用一个布尔值表示，True表示游戏进行中，False表示游戏结束
game_over = False
# 定义游戏的得分，用一个整数表示，初始为0
score = 0
# 进入游戏的主循环
while True:
    # 处理游戏的事件，例如按键，鼠标点击等
    for event in pygame.event.get():
        # 如果事件类型是退出，即点击了窗口的关闭按钮，那么退出游戏
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        # 如果事件类型是按键按下，那么根据按键改变蛇的方向
        if event.type == pygame.KEYDOWN:
            # 如果按下了上方向键，并且蛇的方向不是向下，那么蛇的方向改为向上
            if event.key == pygame.K_UP and direction != 'down':
                direction = 'up'
            # 如果按下了下方向键，并且蛇的方向不是向上，那么蛇的方向改为向下
            if event.key == pygame.K_DOWN and direction != 'up':
                direction = 'down'
            # 如果按下了左方向键，并且蛇的方向不是向右，那么蛇的方向改为向左
            if event.key == pygame.K_LEFT and direction != 'right':
                direction = 'left'
            # 如果按下了右方向键，并且蛇的方向不是向左，那么蛇的方向改为向右
            if event.key == pygame.K_RIGHT and direction != 'left':
                direction = 'right'
    # 如果游戏没有结束，那么更新游戏的逻辑
    if not game_over:
        # 根据蛇的方向，计算蛇头的新位置
        if direction == 'up':
            new_head = (snake[0][0], snake[0][1] - speed)
        if direction == 'down':
            new_head = (snake[0][0], snake[0][1] + speed)
        if direction == 'left':
            new_head = (snake[0][0] - speed, snake[0][1])
        if direction == 'right':
            new_head = (snake[0][0] + speed, snake[0][1])
        # 将新的蛇头插入到蛇的列表的最前面
        snake.insert(0, new_head)
        # 判断蛇是否吃到了食物
        if snake[0] == food:
            # 如果吃到了食物，那么得分加一，食物的位置重新随机生成
            score += 1
            food = (random.randrange(0, screen_width, speed), random.randrange(0, screen_height, speed))
        else:
            # 如果没有吃到食物，那么删除蛇的列表的最后一个元素，即蛇尾
            snake.pop()
        # 判断蛇是否撞到了自己的身体或者边界
        if snake[0][0] < 0 or snake[0][0] >= screen_width or snake[0][1] < 0 or snake[0][1] >= screen_height or snake[0] in snake[1:]:
            # 如果撞到了，那么游戏结束
            game_over = True
    # 填充屏幕的背景色
    screen.fill(black)
    # 绘制蛇的每个部分，用白色的矩形表示
    for x, y in snake:
        pygame.draw.rect(screen, white, (x, y, speed, speed))
    # 绘制食物，用红色的矩形表示
    pygame.draw.rect(screen, red, (food[0], food[1], speed, speed))
    # 绘制得分，用绿色的文字表示，显示在屏幕的左上角
    font = pygame.font.SysFont('arial', 32)
    text = font.render('Score: ' + str(score), True, green)
    screen.blit(text, (0, 0))
    # 更新屏幕的显示
    pygame.display.flip()
    # 控制游戏的速度，使每秒刷新fps次
    clock.tick(fps)
