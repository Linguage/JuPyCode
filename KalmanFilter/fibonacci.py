def fibonacci(n):
    """Generate Fibonacci sequence up to n terms"""
    a, b = 0, 1
    for _ in range(n):
        print(a, end=' ')
        a, b = b, a + b
    print()

# Example usage
if __name__ == "__main__":
    fibonacci(10)  # Prints first 10 Fibonacci numbers
 