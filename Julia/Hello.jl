# 分支
x = 10
if x > 5
    println("x大于5")
else
    println("x不大于5")
end

# 循环
for i in 1:5
    println("当前值为：", i)
end

# 文件输入输出
# 写入文件
open("output.txt", "w") do file
    println(file, "这是一行文本。")
end

# 读取文件
open("output.txt", "r") do file
    content = read(file, String)
    println("文件内容：", content)
end

# 文本的显示
println("Hello, World!")

# 使用函数
function square(x)
    return x * x
end

println("5的平方是：", square(5))
