# 定义微分方程 y' = f(x, y)
function f(x, y)
    return x + y
end

# 欧拉法实现
function euler(f, x0, y0, h, n)
    x = x0
    y = y0
    xs = [x0]
    ys = [y0]
    for i in 1:n
        y += h * f(x, y)
        x += h
        push!(xs, x)
        push!(ys, y)
    end
    return xs, ys
end

# 参数设置
x0 = 0.0      # 初始x
y0 = 1.0      # 初始y
h = 0.1       # 步长
n = 20        # 步数

# 计算
xs, ys = euler(f, x0, y0, h, n)

# 打印结果
for (xi, yi) in zip(xs, ys)
    println("x = $xi, y = $yi")
end

