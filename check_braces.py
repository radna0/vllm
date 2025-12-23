
def check_braces(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    stack = []
    for i, line in enumerate(lines):
        line = line.split('//')[0]  # Ignore comments
        for j, char in enumerate(line):
            if char == '{':
                stack.append((i + 1, j + 1))
            elif char == '}':
                if not stack:
                    print(f"Error: Extra '}}' at line {i + 1}, col {j + 1}")
                    return
                stack.pop()

    if stack:
        print(f"Error: Unclosed '{{' at line {stack[-1][0]}, col {stack[-1][1]}")
        print(f"Total unclosed braces: {len(stack)}")
        # Print the last few unclosed to assist debugging
        for item in stack[-5:]:
           print(f"Unclosed open brace at line {item[0]}")
    else:
        print("Braces are balanced!")

check_braces('csrc/spec_decode_kernels.cu')
