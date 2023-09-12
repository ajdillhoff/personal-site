+++
title = "Python Review Questions"
authors = ["Alex Dillhoff"]
date = 2023-09-12T00:00:00-05:00
draft = false
tags = ["python"]
+++

# Python Review Questions

The following questions are meant to help you review introductory concepts in Python. They are based on the [Python 3 Tutorial](https://docs.python.org/3/tutorial/index.html) and [Python 3 Documentation](https://docs.python.org/3/index.html) and were written to accompany a 5 lecture series on Python.

There are three types of questions:
- **Verify the Code:** Determine the output of a code snippet.
- **Fill in the Code:** Fill in the code to complete a code snippet.
- **Create the Function:** Create a function that satisfies the given requirements.

## Verify the Code

1. **Operations on Strings**
   ```python
   print("Hello" + " " + "World" * 2)
   ```
   What will the output be?

   <details>
    <summary>Solution</summary>
    
    ```bash
    Hello World World
    ```

    **Explanation:** The `+` operator concatenates strings, and the `*` operator repeats strings.
    </details>

2. **Looping Over Lists**
   ```python
   for i in [1, 2, 3]:
       print(i * 2)
   ```
   What will the output be?

   <details>
    <summary>Solution</summary>
    
    ```bash
    2
    4
    6
    ```

    **Explanation:** The `for` loop iterates over the elements of the list.
    </details>

3. **Mutability of Strings**
   ```python
   s = "hello"
   s[0] = "H"
   ```
   Is this code valid? If not, why?

   <details>
    <summary>Solution</summary>
    
    This code is not valid. Strings are immutable, so you cannot change their elements.
    </details>

4. **Static Methods vs. Instance Methods**
   ```python
   class MyClass:
       def instance_method(self):
           return 'instance method'
       
       @staticmethod
       def static_method():
           return 'static method'
   
   obj = MyClass()
   print(obj.instance_method())
   print(MyClass.static_method())
   ```
   What will the output be?

   <details>
    <summary>Solution</summary>
    
    ```bash
    instance method
    static method
    ```

    **Explanation:** Instance methods are called on an instance of a class, whereas static methods are called on the class itself.
    </details>

5. **`yield` Keyword Example**
   ```python
   def my_gen():
       yield "A"
       yield "B"
   
   gen = my_gen()
   print(next(gen))
   print(next(gen))
   print(next(gen))
   ```
   What will the output be?

    <details>
     <summary>Solution</summary>
     
     ```bash
     A
     B
     Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
     StopIteration
     ```
    
     **Explanation:** The `yield` keyword is used to create generators. Generators are iterators that can be iterated over only once.
     </details>

6. **Type Validation**
   ```python
   x = 42
   print(isinstance(x, int))
   ```
   What will the output be?

   <details>
    <summary>Solution</summary>
    
    ```bash
    True
    ```

    **Explanation:** The `isinstance` function checks if an object is of a certain type.
    </details>

7. **`input` Always Returns a `str`**
   ```python
   x = input("Enter a number: ")
   print(type(x) is int)
   ```
   If the user enters `42`, what will the output be?

    <details>
     <summary>Solution</summary>
     
     ```bash
     False
     ```
    
     **Explanation:** The `input` function always returns a `str`, even if the user enters a number.
     </details>

8. **Search in List Example**
   ```python
   my_list = [1, 2, 3, 4, 5]
   print(3 in my_list)
   ```
   What will the output be?

    <details>
     <summary>Solution</summary>
     
     ```bash
     True
     ```
    
     **Explanation:** The `in` operator checks if an element is in a list.
     </details>

9. **Formatted Printing**
   ```python
   name = "Alice"
   age = 30
   print(f"{name} is {age} years old.")
   ```
   What will the output be?

    <details>
     <summary>Solution</summary>
     
     ```bash
     Alice is 30 years old.
     ```
    
     **Explanation:** The `f` prefix allows you to use formatted strings.
     </details>

10. **Equality Versus `is`**
    ```python
    x = [1, 2, 3]
    y = [1, 2, 3]
    print(x == y)
    print(x is y)
    ```
    What will the output be?

    <details>
     <summary>Solution</summary>
     
     ```bash
     True
     False
     ```
    
     **Explanation:** The `==` operator checks if two objects are equal, whereas the `is` operator checks if two objects are the same object.
     </details>

11. **CSV Line to List Using `split`**
    ```python
    line = "apple,banana,cherry"
    fruits = line.split(',')
    print(fruits)
    ```
    What will the output be?

    <details>
     <summary>Solution</summary>
     
     ```bash
     ['apple', 'banana', 'cherry']
     ```
    
     **Explanation:** The `split` method splits a string into a list of strings using a delimiter.
     </details>

12. **Deep vs. Shallow Copy**
    ```python
    import copy
    a = [1, [2, 3]]
    b = copy.copy(a)
    c = copy.deepcopy(a)
    a[1][0] = 99
    print(b)
    print(c)
    ```
    What will the output be?

    <details>
     <summary>Solution</summary>
     
     ```bash
     [1, [99, 3]]
     [1, [2, 3]]
     ```
    
     **Explanation:** The `copy` function creates a shallow copy, whereas the `deepcopy` function creates a deep copy.
     </details>

13. **Access Class Variable vs. Instance Variable**
    ```python
    class Dog:
        kind = 'canine'
        
        def __init__(self, name):
            self.name = name
    
    d = Dog('Fido')
    print(d.kind)
    print(d.name)
    ```
    What will the output be?

    <details>
     <summary>Solution</summary>
     
     ```bash
     canine
     Fido
     ```
    
     **Explanation:** The `kind` variable is a class variable, whereas the `name` variable is an instance variable.
     </details>

14. **Accessing `global` Variable**
    ```python
    x = 10

    def foo():
        global x
        x += 5
        print(x)

    foo()
    ```
    What will the output be?

    <details>
     <summary>Solution</summary>
     
     ```bash
     15
     ```
    
     **Explanation:** The `global` keyword allows you to access a global variable inside a function.
     </details>

---

## Fill in the Code

15. **Looping Over 2D Lists**
    ```python
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    # Fill in the code to print each element in the 2D list.
    ```

    <details>
    <summary>Solution</summary>
    
    ```python
    for row in matrix:
        for element in row:
            print(element)
    ```
    
    **Explanation:** The outer loop iterates over the rows, and the inner loop iterates over the elements in each row.
    </details>
    
16. **Static Methods vs. Instance Methods**
    ```python
    class Calculator:
        @staticmethod
        def add(a, b):
            return a + b
        
        # Fill in the code to create an instance method that multiplies two numbers.
    ```

    <details>
    <summary>Solution</summary>
    
    ```python
    class Calculator:
        @staticmethod
        def add(a, b):
            return a + b
        
        def multiply(self, a, b):
            return a * b
    ```

    **Explanation:** Instance methods take `self` as the first argument.
    </details>

17. **List Comprehension for 2D List**
    ```python
    # Fill in the code to create a 2D list with list comprehension.
    # The 2D list should contain rows from 0 to 4 and columns from 0 to 4, 
    # where each element is the sum of its row and column index.
    ```

    <details>
    <summary>Solution</summary>
    
    ```python
    matrix = [[row + col for col in range(5)] for row in range(5)]
    ```

    **Explanation:** The outer loop iterates over the rows, and the inner loop iterates over the columns.
    </details>

18. **Dictionary Add, Iterating Over Keys and Values**
    ```python
    my_dict = {'apple': 1, 'banana': 2}
    # Fill in the code to add a key-value pair ('cherry', 3) to my_dict and print all keys and values.
    ```

    <details>
    <summary>Solution</summary>
    
    ```python
    my_dict['cherry'] = 3
    for key in my_dict:
        print(key, my_dict[key])
    ```

    **Explanation:** The `for` loop iterates over the keys of the dictionary.
    </details>

19. **Add List to Existing List (Zip vs. Extend)**
    ```python
    list1 = [1, 2, 3]
    list2 = [4, 5, 6]
    # Fill in the code to append the elements of list2 to list1.
    ```

    <details>
    <summary>Solution</summary>
    
    ```python
    list1.extend(list2)
    ```

    **Explanation:** The `extend` method appends the elements of a list to another list.
    </details>

20. **Override Special Method So Class Can Be Sorted**
    ```python
    class Person:
        def __init__(self, name, age):
            self.name = name
            self.age = age
        
        # Fill in the code to make instances of this class sortable by age.
    ```

    <details>
    <summary>Solution</summary>
    
    ```python
    class Person:
        def __init__(self, name, age):
            self.name = name
            self.age = age
        
        def __lt__(self, other):
            return self.age < other.age
    ```

    **Explanation:** The `__lt__` method overrides the `<` operator. Either this method or the `__gt__` method must be defined for the class to be sortable.
    </details>

---

## Create the Function

21. **Reading Input From the User**
    ```python
    numbers = read_numbers_from_user(5)
    print("The sum is:", sum(numbers))
    ```
    Create the function `read_numbers_from_user` that takes an integer \( n \) and reads \( n \) numbers from the user.

    <details>
    <summary>Solution</summary>
    
    ```python
    def read_numbers_from_user(n):
        numbers = []
        for i in range(n):
            numbers.append(int(input("Enter a number: ")))
        return numbers
    ```

    **Explanation:** The `input` function reads a string from the user. The `int` function converts a string to an integer.
    </details>

22. **Match Statement vs. If-Elif-Else**
    ```python
    print(match_fruit_color("apple"))
    ```
    Create the function `match_fruit_color` that takes a fruit name and returns its color using a match statement.

    <details>
    <summary>Solution</summary>
    
    ```python
    def match_fruit_color(fruit):
        match fruit:
            case "apple":
                return "red"
            case "banana":
                return "yellow"
            case "cherry":
                return "red"
            case _:
                return "unknown"
    ```

    **Explanation:** The `match` statement is used to compare a value against a number of patterns. It is similar to the `switch` statement in other languages.

    <details>

23. **Formatted Printing**
    ```python
    print_formatted_string("John", 25)
    ```
    Create the function `print_formatted_string` that takes a name and an age and prints them in a formatted string.

    <details>
    <summary>Solution</summary>
    
    ```python
    def print_formatted_string(name, age):
        print(f"{name} is {age} years old.")
    ```

    **Explanation:** The `f` prefix allows you to use formatted strings.
    </details>

24. **Type Validation**
    ```python
    print(is_valid_number("42"))
    ```
    Create the function `is_valid_number` that takes a string and returns True if it can be converted to an integer or a float, otherwise False.

    <details>
    <summary>Solution</summary>
    
    ```python
    def is_valid_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False
    ```

    **Explanation:** The `try` statement allows you to handle exceptions. The `float` function converts a string to a float. If a string could be converted to a `float`, it can also be converted to an `int`.
    </details>

25. **Accessing `global` Variable**
    ```python
    increment_global_x()
    print(x)
    ```
    Create the function `increment_global_x` that increments the global variable \( x \) by 1.

    <details>
    <summary>Solution</summary>
    
    ```python
    def increment_global_x():
        global x
        x += 1
    ```

    **Explanation:** The `global` keyword allows you to access a global variable inside a function.
    </details>
