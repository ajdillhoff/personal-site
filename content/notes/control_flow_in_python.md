+++
title = "Control Flow in Python"
authors = ["Alex Dillhoff"]
date = 2023-08-22T00:00:00-05:00
draft = false
tags = ["python"]
+++

<a target="_blank" href="https://colab.research.google.com/github/ajdillhoff/python-examples/blob/main/basics/control_flow.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# Control Flow

Control flow allows us to build programs that react to some pre-determined condition. For example, what happens when a user logs in with the correct credentials? What if they don't give valid credentials?

This notebook covers the basic tools to writing conditional statements in Python. It follows Chapter 3 in [Python for Everyone](https://www.py4e.com/html3/03-conditional) by Charles Severance along with my own examples.

## Boolean Expressions

A boolean expression evaluates to either `True` or `False`. This type of expression would be used to check status codes or to check if a user has entered the correct password, for example.


```python
print("1==1 is {}".format(1==1))
print("1==2 is {}".format(1==2))
```

    1==1 is True
    1==2 is False


Operators like `==` are called *relational operators* and they compare two operands and return either `True` or `False`. Other relational operators include:

```python
x != y # x is not equal to y
x > y # x is greater than y
x < y # x is less than y
x >= y # x is greater than or equal to y
x <= y # x is less than or equal to y
x is y # x is the same as y
x is not y # x is not the same as y
```

The last two operators, `is` and `is not`, are used to check if two variables are referencing the same object. The fact that the operands must be *objects* is important here. You should avoid comparing a value with a variable. Python will let you do this, but it will also output a warning.

```python
x = 5
y = 5
x is y # True
x is 5 # True, but not recommended
x is not 5 # False, but not recommended
```


```python
x = 5
y = 10
x is 5
```

    <>:3: SyntaxWarning: "is" with a literal. Did you mean "=="?
    <>:3: SyntaxWarning: "is" with a literal. Did you mean "=="?
    /var/folders/vd/wbzsx0g538nfr96xq81fp7k40000gn/T/ipykernel_24914/759655086.py:3: SyntaxWarning: "is" with a literal. Did you mean "=="?
      x is 5





    True



Python includes three logical operators that are verbose compared to other languages.

```python
x and y # True if both x and y are True
x or y # True if either x or y are True
not x # True if x is False
```


```python
# Example: FizzBuzz
# Consider two possible solutions to the FizzBuzz problem

# Solution 1
n = 15
if not n % 3 and not n % 5:
    print("FizzBuzz")

# Solution 2
n = 15
if n % 3 == 0:
    print("Fizz")
if n % 5 == 0:
    print("Buzz")
```

    FizzBuzz
    Fizz
    Buzz


In the second solution, the output was separated to two separate lines since the `print` function automatically adds a newline. We can change this behavior by adding a second argument to the `print` function.


```python
n = 15
if n % 3 == 0:
    print("Fizz", end="")
if n % 5 == 0:
    print("Buzz")
```

    FizzBuzz


## Conditional Execution

We have already used a key conditional execution tool: the `if` statement. The `if` statement allows us to execute a block of code if a condition is met. The general syntax is:

```python
if condition:
    # code to execute if condition is True
```

Also note that Python is particular about indentation. The code that is executed if the condition is met must be indented. The standard is to use four spaces for each level of indentation.

We can also chain conditional statements together using `elif` and `else`. The `elif` statement is short for "else if" and allows us to check another condition if the previous condition was not met. The `else` statement is used to execute code if none of the previous conditions were met. The general syntax is:

```python
if condition:
    # code to execute if condition is True
elif condition:
    # code to execute if the first condition is False and this condition is True
else:
    # code to execute if all other conditions are False
```

### Switch Statements

Until version 3.10, Python did not have a `switch` statement. This is a conditional statement that allows us to check a variable against a series of values.

With version 3.10 comes the `match` statement. This statement is similar to the `switch` statement in other languages. The general syntax is:

```python
match variable:
    case value1:
        # code to execute if variable == value1
    case value2:
        # code to execute if variable == value2
    case value3:
        # code to execute if variable == value3
    case _:
        # code to execute if none of the previous conditions were met
```


```python
language = input("What is your favorite programming language? ")

match language:
    case "Python":
        print("You're in the right place.")
    case "Java":
        print("Do you despise C++ as much as the creator of Java?")
    case "C++":
        print("You probably like game development.")
    case "C":
        print("Speed is your thing.")
    case _:
        print("You like something else!")
```

    You like something else!


Unlike other languages that implement a `switch` statement, Python's `match` statement does not have a `break` statement. We can still utilize fall-through behavior by including multiple values in a single case separated by `|`.

```python
match variable:
    case value1 | value2:
        # code to execute if variable == value1 or variable == value2
    case value3:
        # code to execute if variable == value3
    case _:
        # code to execute if none of the previous conditions were met
```


```python
language = input("What is your favorite programming language? ")

match language:
    case "Python" | "python":
        print("You're in the right place.")
    case "Java":
        print("Do you despise C++ as much as the creator of Java?")
    case "C++":
        print("You probably like game development.")
    case "C":
        print("Speed is your thing.")
    case _:
        print("You like something else!")
```

    You like something else!


## Iterations

Iterations allow us to execute a block of code multiple times. This is useful for iterating over a list of items or for executing a block of code until a condition is met.

Python supports both a `while` loop and a `for` loop. The `while` loop will execute a block of code until a condition is met. The `for` loop will iterate over a sequence of items.

### For Loops

As opposed to something like C, Python's `for` loop is more like a `foreach` loop. The `for` loop will iterate over a sequence of items. The general syntax is:

```python
for item in sequence:
    # code to execute for each item in the sequence
```

It is commonly used with the `range` function to iterate over a sequence of numbers. The `range` function takes three arguments: `start`, `stop`, and `step`. The `start` argument is the first number in the sequence. The `stop` argument is the last number in the sequence. The `step` argument is the amount to increment the sequence by. The `step` argument is optional and defaults to `1`. The `stop` argument is required. The `start` argument is optional and defaults to `0`.

```python
for i in range(5):
    print(i)
```

### While Loops

The `while` loop will execute a block of code until a condition is met. The general syntax is:

```python
while condition:
    # code to execute while condition is True
```


## Lists

Lists are a sequence of values. They are similar to arrays in other languages. The values in a list are called *elements* or *items*. Lists are mutable, meaning that we can change the values in a list. Lists are also ordered, meaning that the order of the elements in a list is important.

We can create a list by separating the elements with commas and surrounding the list with square brackets.

```python
numbers = [1, 2, 3, 4, 5]
```


```python
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
names = ["Naomi", "Bobbie", "James", "Amos", "Chrisjen", "Alex", "Clarissa"]
names_and_numbers = ["Naomi", 5, "Bobbie", 7, "James", 9, "Amos", 11, "Chrisjen", 13, "Alex", 15, "Clarissa", 17]

# We can even include lists in our lists
nested_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
```

### Iterating Over Lists

We can iterate over a list using a `for` loop. The `for` loop will iterate over each element in the list. The general syntax is:

```python
for item in list:
    # code to execute for each item in the list
```

If the list contains tuples, we can use tuple unpacking to assign the values in the tuple to multiple variables.

```python
numbers = [(1, 2), (3, 4), (5, 6)]
for x, y in numbers:
    print(x, y)
```

### Combining Lists with `zip`

The `zip` function allows us to combine two lists into a single list of tuples. The first element in the first list will be paired with the first element in the second list, the second element in the first list will be paired with the second element in the second list, and so on. The general syntax is:

```python
user_ids = [1, 2, 3]
usernames = ['alice', 'bob', 'charlie']
users = zip(user_ids, usernames)
```


```python
user_ids = [1, 2, 3, 4, 5]
user_names = ["Naomi", "Bobbie", "James", "Amos", "Chrisjen"]

# We can combine these lists into a single list of tuples
user_ids_and_names = zip(user_ids, user_names)

# We can also convert the zip object into a list
user_ids_and_names = list(user_ids_and_names)

for users in user_ids_and_names:
    print(users)
```

    (1, 'Naomi')
    (2, 'Bobbie')
    (3, 'James')
    (4, 'Amos')
    (5, 'Chrisjen')

