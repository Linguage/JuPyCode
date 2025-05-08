import random
import string

def generate_password(length=12):
    """
    Generate a strong random password.
    
    Args:
        length (int): Length of the password. Defaults to 12.
    
    Returns:
        str: A randomly generated password.
    """
    # Define character sets
    lowercase = string.ascii_lowercase
    uppercase = string.ascii_uppercase
    digits = string.digits
    special_chars = string.punctuation
    
    # Combine all character sets
    all_chars = lowercase + uppercase + digits + special_chars
    
    # Ensure the password contains at least one character from each set
    password = [
        random.choice(lowercase),
        random.choice(uppercase),
        random.choice(digits),
        random.choice(special_chars)
    ]
    
    # Fill the rest of the password with random characters
    password.extend(random.choice(all_chars) for _ in range(length - 4))
    
    # Shuffle the password characters
    random.shuffle(password)
    
    # Convert list to string
    return ''.join(password)

# Example usage
if __name__ == "__main__":
    # Generate and print 5 random passwords
    for i in range(5):
        print(f"Password {i+1}: {generate_password()}")
