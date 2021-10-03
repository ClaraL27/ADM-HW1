
# Say "Hello, World!" With Python
if __name__ == '__main__':
    print("Hello, World!")


# Python If-Else
if __name__ == '__main__':
    n = int(input().strip())
    if n % 2 == 0:
        if (n >= 2 and n <= 5) or n > 20:
            print("Not Weird")
        else:
            print("Weird")
    else:
        print("Weird")


# Arithmetic Operators
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a + b)
    print(a - b)
    print(a * b)


# Python: Division
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a//b)
    print(a/b)


# Loops
if __name__ == '__main__':
    n = int(input())
    i = 0
    while (i < n):
        print(i * i)
        i += 1


# Write a function
def is_leap(year):
    leap = False

    # Write your logic here
    if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0:
        leap = True
    return leap

year = int(input())
print(is_leap(year))


# Print Function
if __name__ == '__main__':
    n = int(input())
    for i in range(1, n+ 1):
        print(i, end="")


# List Comprehensions
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    list = []
    for i in range(x + 1):
        for j in range(y + 1):
            for k in range(z + 1):
                if (i + j + k != n):
                    list.append([i, j, k])
    print(list)


# Find the Runner-Up Score!
if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())

    array = list(arr)
    i = max(array)
    while (i == max(array)):
        array.remove(max(array))
    print(max(array))


# Nested Lists
if __name__ == '__main__':
    d = {}
    for _ in range(int(input())):
        name = input()
        score = float(input())
        d[name] = score
    v = d.values()
    second = sorted(list(set(v)))[1]
    second_list = []
    for key, value in d.items():
        if (value == second):
            second_list.append(key)
    second_list.sort()
    for i in second_list:
        print(i)


# Finding the percentage
from decimal import Decimal

if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    scores_list = student_marks[query_name]
    sums = sum(scores_list);
    avg = Decimal(sums / 3)
    print(round(avg, 2))


# Lists
if __name__ == '__main__':
    N = int(input())

    l = []
    for _ in range(N):
        s = input().split()
        cmd = s[0]
        args = s[1:]
        if cmd != "print":
            cmd += "(" + ",".join(args) + ")"
            eval("l." + cmd)
        else:
            print(l)


# Tuples
if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    t = tuple(integer_list)
    print(hash(t))


# sWAP cASE
def swap_case(s):
    ret = ""
    for l in s:
        if (l.islower()):
            ret += l.upper()
        else:
            ret += l.lower()
    return ret


if __name__ == '__main__':
    s = input()
    result = swap_case(s)
    print(result)


# String Split and Join
def split_and_join(line):
    res = ""
    res_arr = line.split(" ")
    res = "-".join(res_arr)
    return res


if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)


# What's Your Name?
def print_full_name(a, b):
    print("Hello " + a + " " + b + "! You just delved into python.")


if __name__ == '__main__':
    first_name = input()
    last_name = input()
    print_full_name(first_name, last_name)


# Mutations
def mutate_string(string, position, character):
    res = string[:position] + character + string[position + 1:]
    return res


if __name__ == '__main__':
    s = input()
    i, c = input().split()
    s_new = mutate_string(s, int(i), c)
    print(s_new)


# Find a string
def count_substring(string, sub_string):
    res = 0
    for i in range(len(string)):
        if (string[i:i + len(sub_string)] == sub_string):
            res += 1
    return res


if __name__ == '__main__':
    string = input().strip()
    sub_string = input().strip()

    count = count_substring(string, sub_string)
    print(count)


# String Validators
if __name__ == '__main__':
    s = input()
    a = "False"
    b = "False"
    c = "False"
    d = "False"
    e = "False"
    for car in s:
        if (car.isalnum()):
            a = "True"
        if (car.isalpha()):
            b = "True"
        if (car.isdigit()):
            c = "True"
        if (car.islower()):
            d = "True"
        if (car.isupper()):
            e = "True"
    print(a + "\n" + b + "\n" + c + "\n" + d + "\n" + e)


# Text Alignment
thickness = int(input())
c = 'H'

# top arrow
for i in range(thickness):
    print((c * i).rjust(thickness - 1) + c + (c * i).ljust(thickness - 1))

for i in range(thickness + 1):
    print((c * thickness).center(thickness * 2) + (c * thickness).center(thickness * 6))

for i in range((thickness + 1) // 2):
    print((c * thickness * 5).center(thickness * 6))

for i in range(thickness + 1):
    print((c * thickness).center(thickness * 2) + (c * thickness).center(thickness * 6))

# bottom arrow
for i in range(thickness):
    print(((c * (thickness - i - 1)).rjust(thickness) + c + (c * (thickness - i - 1)).ljust(thickness)).rjust(
        thickness * 6))


# Text Wrap
import textwrap

def wrap(string, max_width):
    sol = textwrap.fill(string, max_width)
    return sol

if __name__ == '__main__':
    string, max_width = input(), int(input())
    result = wrap(string, max_width)
    print(result)


# Designer Door Mat
rows, cols = map(int, input().split())

middle = rows // 2 + 1

for i in range(1, middle):
    cen = (i * 2 - 1) * ".|."
    print(cen.center(cols, "-"))

print("WELCOME".center(cols, "-"))

for i in reversed(range(1, middle)):
    cen = (i * 2 - 1) * ".|."
    print(cen.center(cols, "-"))


# String Formatting
def print_formatted(number):
    width = len("{0:b}".format(n))
    for i in range(1, number + 1):
        print("{0:{width_}d} {0:{width_}o} {0:{width_}X} {0:{width_}b}".format(i, width_=width))


if __name__ == '__main__':
    n = int(input())
    print_formatted(n)


# Alphabet Rangoli
import string


def print_rangoli(size):
    arr_string = string.ascii_lowercase
    R = []
    for i in range(size):
        cen = "-".join(arr_string[i:size])
        R.append((cen[::-1] + cen[1:]).center(4 * size - 3, "-"))
    print('\n'.join(R[:0:-1] + R))


if __name__ == '__main__':
    n = int(input())
    print_rangoli(n)


# Capitalize!
# !/bin/python3

import math
import os
import random
import re
import sys


# Complete the solve function below.
def solve(s):
    for word in s.split():
        s = s.replace(word, word.capitalize())
    return s


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    s = input()

    result = solve(s)

    fptr.write(result + '\n')

    fptr.close()


# The Minion Game
def minion_game(string):
    vow = "AEIOU"
    kev = 0
    stu = 0
    for i in range(len(string)):
        if string[i] in vow:
            kev += (len(string) - i)
        else:
            stu += (len(string) - i)
    if kev > stu:
        print("Kevin", kev)
    elif kev < stu:
        print("Stuart", stu)
    else:
        print("Draw")


if __name__ == '__main__':
    s = input()
    minion_game(s)


# Merge the Tools!
import textwrap


def merge_the_tools(string, k):
    u = []
    len_u = 0
    for c in string:
        len_u += 1
        if c not in u:
            u.append(c)
        if len_u == k:
            print("".join(u))
            u = []
            len_u = 0


if __name__ == '__main__':
    string, k = input(), int(input())
    merge_the_tools(string, k)


# Introduction to Sets
def average(array):
    s = set(array)
    avg = (sum(s) / len(s))
    return avg


if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    result = average(arr)
    print(result)


# Symmetric Difference
m, M = (int(input()), input().split())
n, N = (int(input()), input().split())
x = set(M)
y = set(N)
diff_yx = y.difference(x)
diff_xy = x.difference(y)
un = diff_yx.union(diff_xy)
print('\n'.join(sorted(un, key=int)))


# No Idea!
nm = list(map(int, input().split()))
arr = list(map(int, input().split()))
a = list(map(int, input().split()))
b = list(map(int, input().split()))

A = set(a)
B = set(b)

happy = 0

for num in arr:
    if num in A:
        happy += 1
    if num in B:
        happy -= 1
print(happy)


# Set .add()
N = int(input())
s = set()
for i in range(N):
    s.add(input())
print(len(s))


# Set .discard(), .remove() & .pop()
n = int(input())
s = set(map(int, input().split()))
N = int(input())
for i in range(N):
    cmd = input().split()
    if cmd[0] == "pop" and len(s) != 0:
        s.pop()
    elif cmd[0] == "remove" and int(cmd[1]) in s:
        s.remove(int(cmd[1]))
    elif cmd[0] == "discard":
        s.discard(int(cmd[1]))

print(sum(s))


# Set .union() Operation
_, a = input(), set(input().split())
_, b = input(), set(input().split())
print(len(a.union(b)))


# Set .intersection() Operation
_, a = input(), set(input().split())
_, b = input(), set(input().split())
print(len(a.intersection(b)))


# Set .difference() Operation
_, a = input(), set(input().split())
_, b = input(), set(input().split())
print(len(a.difference(b)))


# Set .symmetric_difference() Operation
_, a = input(), set(input().split())
_, b = input(), set(input().split())
print(len(a.symmetric_difference(b)))


# Set Mutations
_, A = int(input()), set(map(int, input().split()))
B = int(input())
for _ in range(B):
    command, newSet = input().split()[0], set(map(int, input().split()))
    getattr(A, command)(newSet)

print(sum(A))


# The Captain's Room
n, arr = int(input()), list(map(int, input().split()))
myset = set(arr)
sum1 = sum(myset) * n
sum2 = sum(arr)
print((sum1 - sum2) // (n - 1))


# Check Subset
n = int(input())
for _ in range(n):
    x, a, z, b = input(), set(input().split()), input(), set(input().split())
    print(a.issubset(b))


# Check Strict Superset
a = set(input().split())
count = 0
n = int(input())
for i in range(n):
    b = set(input().split())
    if a.issuperset(b):
        count += 1
print(count == n)


# collections.Counter()
from collections import Counter

numShoes = int(input())
shoes = Counter(map(int, input().split()))
numCust = int(input())
income = 0

for i in range(numCust):
    size, price = map(int, input().split())
    if shoes[size]:
        income += price
        shoes[size] -= 1

print(income)


# DefaultDict Tutorial
from collections import defaultdict

d = defaultdict(list)
list1 = []

n, m = map(int, input().split())

for i in range(0, n):
    d[input()].append(i + 1)

for i in range(0, m):
    list1 = list1 + [input()]

for i in list1:
    if i in d:
        print(" ".join(map(str, d[i])))
    else:
        print(-1)


# Collections.namedtuple()
from collections import namedtuple

n = int(input())
a = input()
total = 0
Student = namedtuple('Student', a)
for _ in range(n):
    student = Student(*input().split())
    total += int(student.MARKS)
print('{:.2f}'.format(total / n))


# Collections.OrderedDict()
from collections import OrderedDict

d = OrderedDict()
N = int(input())
for _ in range(N):
    item, space, quantity = input().rpartition(' ')
    d[item] = d.get(item, 0) + int(quantity)  # add quantity to the item
for item, quantity in d.items():
    print(item, quantity)


# Word Order
from collections import OrderedDict

d = OrderedDict()
n = int(input())

for _ in range(n):
    word = input()
    d[word] = d.get(word, 0) + 1
print(len(d))
print(*d.values())


# Collections.deque()
from collections import deque

d = deque()

for _ in range(int(input())):
    method, *n = input().split()
    getattr(d, method)(*n)

print(*d)


# Company Logo
from collections import Counter

string = Counter(sorted(input()))
for c in string.most_common(3):
    print(*c)


# Piling Up!
from collections import deque

T = int(input())
for _ in range(T):
    _, queue = input(), deque(map(int, input().split()))

    for cube in reversed(sorted(queue)):
        if queue[-1] == cube:
            queue.pop()
        elif queue[0] == cube:
            queue.popleft()
        else:
            print('No')
            break
    else:
        print('Yes')


# Calendar Module
import calendar

m, d, y = map(int, input().split())
days = {0: 'MONDAY', 1: 'TUESDAY', 2: 'WEDNESDAY', 3: 'THURSDAY', 4: 'FRIDAY', 5: 'SATURDAY', 6: 'SUNDAY'}
print(days[calendar.weekday(y, m, d)])


# Time Delta
from datetime import datetime as dt

format_ = '%a %d %b %Y %H:%M:%S %z'
T = int(input())
for i in range(T):
    t1 = dt.strptime(input(), format_)
    t2 = dt.strptime(input(), format_)
    print(int(abs((t1 - t2).total_seconds())))


# Exceptions
T = int(input())
for i in range(T):
    try:
        a, b = map(int, input().split())
        print(a // b)
    except Exception as e:
        print("Error Code:", e)


# Zipped!
N, X = map(int, input().split())

sheet = []
for _ in range(X):
    marks = map(float, input().split())
    sheet.append(marks)

for i in zip(*sheet):
    print(sum(i) / len(i))


# Athlete Sort
N, M = map(int, input().split())

nums = []
for i in range(N):
    list_ = list(map(int, input().split()))
    nums.append(list_)

K = int(input())
nums.sort(key=lambda x: x[K])

for line in nums:
    print(*line, sep=' ')


# ginortS
low = []
up = []
odd = []
ev = []
S = input()
for i in sorted(S):
    if i.isalpha():
        if i.isupper():
            x = up
        else:
            x = low
    else:
        if (int(i) % 2):
            x = odd
        else:
            x = ev
    x.append(i)

print("".join(low + up + odd + ev))


# Map and Lambda Function
cube = lambda x: pow(x, 3)


def fibonacci(n):
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i - 2] + fib[i - 1])
    return fib[0:n]


if __name__ == '__main__':
    n = int(input())
    print(list(map(cube, fibonacci(n))))


# Detect Floating Point Number
import re

n = int(input())
for i in range(n):
    isnum = input()
    print(bool(re.match(r'^[-+]?[0-9]*\.[0-9]+$', isnum)))


# Re.split()
regex_pattern = r"[.,]+"
import re

print("\n".join(re.split(regex_pattern, input())))


# Group(), Groups() & Groupdict()
import re

s = input().strip()
m = re.search(r'([a-zA-Z0-9])\1+', s)
if m:
    print(m.group(1))
else:
    print(-1)


# Re.findall() & Re.finditer()
import re

v = "aeiou"
c = "qwrtypsdfghjklzxcvbnm"
m = re.findall(r"(?<=[%s])([%s]{2,})[%s]" % (c, v, c), input(), flags=re.I)
if m:
    print("\n".join(m))
else:
    print("\n".join(["-1"]))


# Re.start() & Re.end()
import re

s = input()
k = input()
pattern = re.compile(k)
r = pattern.search(s)
if not r:
    print("(-1, -1)")
while r:
    print("({0}, {1})".format(r.start(), r.end() - 1))
    r = pattern.search(s, r.start() + 1)


# Regex Substitution
n = int(input())
for _ in range(n):
    line = input()
    while " && " in line or " || " in line:
        line = line.replace(" && ", " and ").replace(" || ", " or ")
    print(line)


# Validating Roman Numerals
regex_pattern = r"M{0,3}(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[VX]|V?I{0,3})$"  # Do not delete 'r'.

import re

print(str(bool(re.match(regex_pattern, input()))))


# Validating phone numbers
import re

N = int(input())
for i in range(N):
    line = input()
    if re.match(r"[789]\d{9}$", line):
        print("YES")
    else:
        print("NO")


# Validating and Parsing Email Addresses
import re

n = int(input())
for _ in range(n):
    name, email = input().split(' ')
    m = re.match(r"<[A-Za-z](\w|-|\.|_)+@[A-Za-z]+\.[A-Za-z]{1,3}>", email)
    if m:
        print(name, email)


# Hex Color Code
import re

regex = r":?.(#[0-9a-fA-F]{6}|#[0-9a-fA-F]{3})"
N = int(input())
for _ in range(N):
    line = input()
    match = re.findall(regex, line)
    if match:
        print(*match, sep='\n')


# HTML Parser - Part 1
from html.parser import HTMLParser


class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print('Start :', tag)
        for ele in attrs:
            print('->', ele[0], '>', ele[1])

    def handle_endtag(self, tag):
        print('End   :', tag)

    def handle_startendtag(self, tag, attrs):
        print('Empty :', tag)
        for ele in attrs:
            print('->', ele[0], '>', ele[1])


N = int(input())

parser = MyHTMLParser()
for _ in range(N):
    line = input()
    parser.feed(line)


# HTML Parser - Part 2
from html.parser import HTMLParser


class MyHTMLParser(HTMLParser):
    def handle_comment(self, comment):
        if '\n' in comment:
            print('>>> Multi-line Comment')
        else:
            print('>>> Single-line Comment')

        print(comment)

    def handle_data(self, data):
        if data == '\n': return
        print('>>> Data')
        print(data)


parser = MyHTMLParser()

N = int(input())

html_string = ""
for i in range(N):
    html_string += input().rstrip() + '\n'

parser.feed(html_string)
parser.close()


# Detect HTML Tags, Attributes and Attribute Values
from html.parser import HTMLParser


class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        [print('-> {} > {}'.format(*attr)) for attr in attrs]


parser = MyHTMLParser()
N = int(input())
for i in range(N):
    line = input()
    html = '\n'.join([line])
    parser.feed(html)
parser.close()


# Validating UID
import re

T = int(input())
for _ in range(T):
    uid = ''.join(sorted(input()))
    try:
        assert re.search(r'[A-Z]{2}', uid)
        assert re.search(r'\d\d\d', uid)
        assert not re.search(r'[^a-zA-Z0-9]', uid)
        assert not re.search(r'(.)\1', uid)
        assert len(uid) == 10
    except:
        print('Invalid')
    else:
        print('Valid')


# Validating Credit Card Numbers
import re

N = int(input())
for _ in range(N):
    line = input()
    if re.match(r"^[456]([\d]{15}|[\d]{3}(-[\d]{4}){3})$", line) and not re.search(r"([\d])\1\1\1",
                                                                                   line.replace("-", "")):
        print("Valid")
    else:
        print("Invalid")


# Validating Postal Codes
regex_integer_in_range = r"^[1-9][\d]{5}$"  # Do not delete 'r'.
regex_alternating_repetitive_digit_pair = r"(\d)(?=\d\1)"  # Do not delete 'r'.

import re

P = input()

print(bool(re.match(regex_integer_in_range, P))
      and len(re.findall(regex_alternating_repetitive_digit_pair, P)) < 2)


# Matrix Script
# !/bin/python3

import math
import os
import random
import re
import sys

first_multiple_input = input().rstrip().split()

n = int(first_multiple_input[0])

m = int(first_multiple_input[1])

matrix = []
b = ""
for _ in range(n):
    matrix_item = input()
    matrix.append(matrix_item)

for z in zip(*matrix):
    b += "".join(z)
regex = re.sub(r"(?<=\w)([^\w]+)(?=\w)", " ", b)
print(regex)


# XML 1 - Find the Score
import sys
import xml.etree.ElementTree as etree


def get_attr_number(node):
    sum_ = 0
    for child in node.iter():
        sum_ += (len(child.attrib))
    return sum_


if __name__ == '__main__':
    sys.stdin.readline()
    xml = sys.stdin.read()
    tree = etree.ElementTree(etree.fromstring(xml))
    root = tree.getroot()
    print(get_attr_number(root))


# XML2 - Find the Maximum Depth
import xml.etree.ElementTree as etree

maxdepth = 0


def depth(elem, level):
    global maxdepth
    level += 1
    if (level >= maxdepth):
        maxdepth = level

    for child in elem:
        depth(child, level)


if __name__ == '__main__':
    n = int(input())
    xml = ""
    for i in range(n):
        xml = xml + input() + "\n"
    tree = etree.ElementTree(etree.fromstring(xml))
    depth(tree.getroot(), -1)
    print(maxdepth)


# Standardize Mobile Number Using Decorators
def wrapper(f):
    def fun(l):
        phone = []
        for n in l:
            phone.append('+91 {} {}'.format(n[-10:-5], n[-5:]))
        f(phone)

    return fun


@wrapper
def sort_phone(l):
    print(*sorted(l), sep='\n')


if __name__ == '__main__':
    l = [input() for _ in range(int(input()))]
    sort_phone(l)


# Decorators 2 - Name Directory
import operator


def person_lister(f):
    def inner(people):
        fun = lambda x: int(x[2])
        return map(f, sorted(people, key=fun))

    return inner


@person_lister
def name_format(person):
    return ("Mr. " if person[3] == "M" else "Ms. ") + person[0] + " " + person[1]


if __name__ == '__main__':
    people = [input().split() for i in range(int(input()))]
    print(*name_format(people), sep='\n')


# Arrays
import numpy


def arrays(arr):
    array_ = arr[::-1]
    return numpy.array(array_, float)


arr = input().strip().split(' ')
result = arrays(arr)
print(result)


# Shape and Reshape
import numpy

arr = input().split()
array_ = numpy.array(arr, int).reshape(3, 3)
print(array_)


# Transpose and Flatten
import numpy

n, m = map(int, input().split())
arr = []
for _ in range(n):
    arr.append(input().strip().split())

array = numpy.array(arr, int)
print(array.transpose())
print(array.flatten())


# Concatenate
import numpy

a, b, c = map(int, input().split())
arrA_ = []
for _ in range(a):
    arrA_.append(input().split())
arrB_ = []
for _ in range(b):
    arrB_.append(input().split())
arrA = numpy.array(arrA_, int)
arrB = numpy.array(arrB_, int)

print(numpy.concatenate((arrA, arrB), axis=0))


# Zeros and Ones
import numpy

tup = map(int, input().split())
nums = tuple(tup)
zero = numpy.zeros(nums, dtype=numpy.int)
one = numpy.ones(nums, dtype=numpy.int)
print(zero)
print(one)


# Eye and Identity
import numpy

n, m = (map(int, input().split()))
print(str(numpy.eye(n, m, k=0)).replace('0', ' 0').replace('1', ' 1'))


# Array Mathematics
import numpy as np

n, m = map(int, input().split())

a = np.zeros((n, m), int)
b = np.zeros((n, m), int)
for i in range(n):
    a[i] = np.array(input().split(), int)
for i in range(n):
    b[i] = np.array(input().split(), int)

print(a + b)
print(a - b)
print(a * b)
print(np.array(a / b, int))
print(a % b)
print(a ** b)


# Floor, Ceil and Rint
import numpy

numpy.set_printoptions(sign=' ')
arr = input().split()
a = numpy.array(arr, float)

print(numpy.floor(a))
print(numpy.ceil(a))
print(numpy.rint(a))


# Sum and Prod
import numpy as np

n, m = map(int, input().split())
array = []
for i in range(n):
    array.append(input().split())
ar = np.array(array, int)
summ = np.sum(ar, axis=0)

print(np.prod(summ))


# Min and Max
import numpy as np

n, m = list(map(int, (input().split())))
array = []
for _ in range(int(n)):
    array.append(list(map(int, (input().split()))))
arr = np.array(array)
mini = np.min(arr, axis=1)
maxi = np.max(mini)
print(maxi)


# Mean, Var, and Std
import numpy

n, m = map(int, input().split())
numpy.set_printoptions(legacy='1.13')
array = []
for i in range(n):
    array.append(list(map(int, input().split())))
arr = numpy.array(array)

print(numpy.mean(arr, axis=1))
print(numpy.var(arr, axis=0))
print(numpy.std(arr, axis=None))


# Dot and Cross
import numpy

n = int(input())
array1 = []
for _ in range(n):
    array1.append(list(map(int, input().split())))
arr1 = numpy.array(array1)

array2 = []
for _ in range(n):
    array2.append(list(map(int, input().split())))
arr2 = numpy.array(array2)

print(numpy.dot(arr1, arr2))


# Inner and Outer
import numpy

A = numpy.array(input().split(), int)
B = numpy.array(input().split(), int)
inn = numpy.inner(A, B)
out = numpy.outer(A, B)
print(inn, out, sep='\n')


# Polynomials
import numpy

array = list(map(float, input().split()))
x = float(input())
print(numpy.polyval(array, x))


# Linear Algebra
import numpy

n = int(input())
array = []
for _ in range(n):
    array.append(input().split())

arr = numpy.array(array, float)

numpy.set_printoptions(legacy='1.13')
print(numpy.linalg.det(arr))


# Birthday Cake Candles
# !/bin/python3

import math
import os
import random
import re
import sys


#
# Complete the 'birthdayCakeCandles' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY candles as parameter.
#

def birthdayCakeCandles(candles):
    return candles.count(max(candles))


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()


# Number Line Jumps
def kangaroo(x1, v1, x2, v2):
    if (v1 > v2) and ((x2 - x1) % (v2 - v1)) == 0:
        return 'YES'
    else:
        return 'NO'


x1, v1, x2, v2 = map(int, input().split())
print(kangaroo(x1, v1, x2, v2))


# Viral Advertising
m = 2
tot = 2
n = int(input())
for _ in range(1, n):
    m += m >> 1
    tot += m
print(tot)


# Recursive Digit Sum
# !/bin/python3

import math
import os
import random
import re
import sys


# Complete the superDigit function below.
def superDigit(n, k):
    if len(n) == 1:
        return int(n)
    arr = []
    for i in n:
        arr.append(int(i))
    x = sum(arr) * k
    return superDigit(str(x), 1)


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    nk = input().split()

    n = nk[0]

    k = int(nk[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()


# Insertion Sort - Part 1
import sys


def insertionSort(n, arr):
    target = arr[-1]
    i = n - 2

    while (target < arr[i]) and (i >= 0):
        arr[i + 1] = arr[i]
        print(' '.join(map(str, arr)))
        i -= 1

    arr[i + 1] = target
    print(' '.join(map(str, arr)))


if __name__ == "__main__":
    n = int(input().strip())
    arr = list(map(int, input().strip().split(' ')))
    insertionSort(n, arr)


# Insertion Sort - Part 2
n = int(input())
arr = list(map(int, input().strip().split(' ')))

for i in range(1, n):
    key = arr[i]
    j = i - 1
    while j >= 0 and arr[j] > key:
        arr[j + 1] = arr[j]
        j = j - 1
    arr[j + 1] = key
    print(*arr)
