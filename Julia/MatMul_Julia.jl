using BenchmarkTools

# 定义矩阵大小
N = 2000

# 初始化矩阵 a 和 b
a = zeros(Float64, N, N)
b = zeros(Float64, N, N)
for i in 1:N
    for j in 1:N
        a[i, j] = i + j + 0.5
        b[i, j] = i + j - 0.5
    end
end

# 使用 @btime 宏来计算矩阵相乘的时间
@btime c = a * b

# 输出矩阵相乘的结果
println("Matrix multiplication completed.")
