#!/usr/bin/env python3
import subprocess
import sys


def main():
    files = sys.argv[1:]
    for file in files:
        subprocess.run(["clang-format", "-i", file], check=True)


if __name__ == "__main__":
    main()
