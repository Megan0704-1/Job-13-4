import re
import argparse
import os
import pdb

# pdb.set_trace()

# Regular expression to match operation definitions in TableGen files
# OP_DEF_PATTERN = re.compile(r'def\s+.+Op.*<\s*"([^"]+)"\s*.*')

OP_DEF_PATTERN = re.compile(
    r'def\s+(\w+)Op\s*:\s*(?:Op<\s*(\w+)_Dialect\s*,\s*"([^"]+)"|(\w+)<\s*"([^"]+)")'
)


def condition(ops: tuple):
    for op in ops:
        if op.islower():
            return op
    return None


def extract_op_names_from_file(name, file_path):
    """Extract operation names from a TableGen file.""" ""
    with open(file_path) as f:
        content = f.read()
        # Remove comments to avoid false positives
        content = re.sub(r"//.*?\n", "", content)  # Remove single-line comments
        content = re.sub(
            r"/\*.*?\*/", "", content, flags=re.DOTALL
        )  # Remove multi-line comments
        # Search for operation definitions
        matches = OP_DEF_PATTERN.findall(content)

    loaded_ops = []
    for match in matches:
        if condition(match) is None:
            continue
        else:
            loaded_ops.append(condition(match))

    return loaded_ops


def save_op_names_to_file(op_names, folder_name, dialect_name):
    """Save operation names to a file in the specified folder.""" ""
    # Create the folder if it doesn't exist
    dialect_name = folder_name
    folder_name = "./opspec/" + folder_name
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Define the output file path
    output_file = os.path.join(folder_name, f"{dialect_name}_ops.txt")

    existing_ops = set()
    if os.path.exists(output_file):
        with open(output_file) as fh:
            existing_ops = {line.strip() for line in fh.readlines()}

    new_ops = {dialect_name + "." + op_name for op_name in op_names}
    unique_ops = existing_ops.union(new_ops)

    # Write the operation names to the file
    with open(output_file, "w") as fh:
        for op_name in sorted(unique_ops):
            fh.write(f"{op_name}\n")


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Extract MLIR operation names from TableGen files."
    )
    parser.add_argument(
        "-td", "--tablegen-file", required=True, help="Path to the TableGen file."
    )
    parser.add_argument(
        "-n",
        "--dialect-name",
        required=True,
        help="Name of the dialect (e.g., 'affine').",
    )
    args = parser.parse_args()

    # Extract operation names
    op_names = extract_op_names_from_file(args.dialect_name, args.tablegen_file)

    print("Operation names found:")
    for op_name in sorted(op_names):
        print(op_name)

    save_op_names_to_file(op_names, args.dialect_name, args.dialect_name)
    print(f"Operation names saved to ./{args.dialect_name}/{args.dialect_name}_ops.txt")


if __name__ == "__main__":
    main()
