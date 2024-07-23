+++
title = "Debugging in C"
authors = ["Alex Dillhoff"]
date = 2024-06-22T00:00:00-05:00
tags = ["programming"]
draft = false
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Types of Errors](#types-of-errors)
- [Debugging Techniques](#debugging-techniques)
- [Examples](#examples)

</div>
<!--endtoc-->

> With great power comes great responsibility. -Ben Parker

C is a powerful language, but it is also considered _unsafe_. Mistakes stemming from pointer arithmetic, memory allocation, and other low-level operations can lead to segmentation faults, memory leaks, and other undefined behavior. This is one of the reasons why advocates are pushing for the use of safer languages like [Rust](https://www.rust-lang.org/).

These notes will cover some of the tools and techniques that can be used to debug C programs, from simple `printf` statements to more advanced tools like `gdb`.


## Types of Errors {#types-of-errors}

Classifying the different types of errors is the first step towards becoming an expert debugger. If you can quickly identify what type of problem you have, you will be able to move towards a solution faster.


### Syntax Errors {#syntax-errors}

A **syntax error** is one in which a minor syntactical mistake was made in the code. For example, C requires that all statements end with a semicolon. If you forget to add one, the compiler will fail to compile your code and produce a relevent error message. These are the simplest types of errors to fix because they are typically accompanied by a helpful compiler message.

Consider the following example:

```c
#include <stdio.h>

int main(void) {
    printf("Hello, World!\n")
    return 0;
}
```

This code will produce the following error message:

```text
hello.c: In function ‘main’:
hello.c:4:30: error: expected ‘;’ before ‘return’
    4 |     printf("Hello, World!\n")
      |                              ^
      |                              ;
    5 |     return 0;
      |     ~~~~~~
```

Your compiler may vary slightly, but the message is clear: you forgot to add a semicolon at the end of the `printf` statement. You may read that message and find it a tad misleading. It says that the semicolon was expected before `return`. You might expect it mention that it is missing at the end of the `printf` statement. This is because the compiler would not detect the error until it reaches the `return` statement. It is important to read the error message carefully and look at the line numbers to determine where the error actually is.


### Semantic Errors {#semantic-errors}

**Semantic errors** do not produce a compiler error, so the program will still compile and run. These errors range from easy to extremely difficult depending on the size of the program and how often the error occurs. Consider the following working example:

```c
#include <stdio.h>

int main(void) {
    double subtotal = 52.75;
    int num_purchases = 10;

    if (num_purchases >= 10)
        subtotal = subtotal - (subtotal * 0.10);  // 10% discount

    printf("Total: $%.2f\n", subtotal);

    return 0;
}
```

This program checks if a customer loyalty discount should be applied. If they have purchased 10 or more items, they receive a 10% discount. The program compiles and runs without any errors and works correctly. Let's look at a slightly modified version of this code.

```c
#include <stdio.h>

int main(void) {
    double subtotal = 52.75;
    int num_purchases = 10;

    if (num_purchases >= 10)
        printf("Congratulations! You've unlocked a 10% discount.\n");
        subtotal = subtotal - (subtotal * 0.10);  // 10% discount

    printf("Total: $%.2f\n", subtotal);

    return 0;
}
```

This code compiles and runs but no longer matches our original intent. Can you spot the error?

The `if` statement only applies to the first line of code after it. This is a common mistake that can be difficult to spot in larger programs. In this case, a discount is applied regardless of the number of purchases because only the `printf` statement following the `if` statement is considered within the scope of the `if`, even though the next line is indented. **The fix is simple: add curly braces to the `if` statement.**


### Logical Errors {#logical-errors}

Errors in logic can be hard to detect, especially as the scope of our problem grows. These errors are not necessarily syntax errors, and they may not even produce a compiler error. They are usually detected by testing the program and verifying that the output matches the expected output.

Consider the [Collatz conjecture](https://en.wikipedia.org/wiki/Collatz_conjecture) problem, as seen on [Project Euler](https://projecteuler.net/problem=14):

> The following iterative sequence is defined for the set of positive integers:
>
> n → n/2 (n is even)
> n → 3n + 1 (n is odd)
>
> Using the rule above and starting with 13, we generate the following sequence:
>
> 13 → 40 → 20 → 10 → 5 → 16 → 8 → 4 → 2 → 1
>
> It can be seen that this sequence (starting at 13 and finishing at 1) contains 10 terms. Although it has not been proved yet (Collatz Problem), it is thought that all starting numbers finish at 1.
>
> Which starting number, under one million, produces the longest chain?

An incorrect solution to the problem is shown below.

```c
#include <stdio.h>

int collatz_steps(long long n) {
    int steps = 0;

    while (n != 1) {
        if (!n % 2 == 0) {
            n = n / 2;
        } else {
            n = 3 * n + 1;
        }
        printf("%lld", n);
        if (n != 1) {
            printf(" -> ");
        }
        steps++;
    }

    printf("\n");

    return steps;
}

int main() {
    long long n = 12;
    int result = collatz_steps(n);
    printf("Number of steps for %lld to reach 1: %d\n", n, result);
    return 0;
}
```

This solution is _almost_ correct. The issue lies in the `if` statement in the `collatz_steps` function. The condition `!n % 2 == 0` is incorrect. The correct condition should be `n % 2 == 0`. If you organize your code such that your functions are relatively small and modular, you can easily test each function individually to ensure that it works as expected.


### Runtime Errors {#runtime-errors}


## Debugging Techniques {#debugging-techniques}


### Simple Debugging {#simple-debugging}

The simplest way to debug a C program is to litter it with `printf` statements until you narrow down exactly where the problem is. It works, sure, but it's not very efficient. Depending on the size of your program, it can be very time consuming to add and remove `printf` statements. To this point, I would always recommend adding a `debug` target to your `Makefile` that compiles your program with debugging symbols. When used with a macro specifically for debugging, you can easily add and remove debugging statements without needing to manually remove them.


### Debugging with `gdb` {#debugging-with-gdb}

`GDB` stands for GNU Debugger. It is a command-line debugger that can be used to step through a program line-by-line, set breakpoints, and inspect memory. It is a very powerful tool, but it can be a bit intimidating to use at first. This section will cover some of the basic commands that can be used to debug a program. Another variant of this for ARM-based processors like the new Macbooks is called `lldb`. It is very similar to `gdb` and can be used in the same way. I will include both versions of the commands as we go along.

`GDB` typically comes pre-installed on most Linux distributions. If you are using a Mac, you can install it with `brew`. It is available for Windows via [MSYS2](https://www.msys2.org/). I trust you know how to install it on your own system. Once you have it installed, you can run it by typing `gdb` in your terminal. You will be greeted with a prompt that looks like this:

```text
GNU gdb (Ubuntu 12.1-0ubuntu1~22.04) 12.1
Copyright (C) 2022 Free Software Foundation, Inc.
License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.
Type "show copying" and "show warranty" for details.
This GDB was configured as "x86_64-linux-gnu".
Type "show configuration" for configuration details.
For bug reporting instructions, please see:
<https://www.gnu.org/software/gdb/bugs/>.
Find the GDB manual and other documentation resources online at:
    <http://www.gnu.org/software/gdb/documentation/>.

For help, type "help".
Type "apropos word" to search for commands related to "word".
(gdb)
```

In order for `gdb` to be useful, you need to compile your program with debugging symbols. This can be done by adding the `-g` flag to your `gcc` command. It is also recommended to disable compiler optimizations with the `-O0` flag. This will ensure that the code you are debugging is as close to the source code as possible. For example, if you have a program called `hello.c`, you can compile it with debugging symbols like this:

```bash
gcc -g -O0 hello.c -o hello
```

Once you have compiled your program with debugging symbols, you can run it with `gdb` like this:

```bash
gdb ./hello
```

This will not actually begin running the program itself. Instead, it loads `gdb` which initializes the debugger. You can run the program with `run` to start execution. Let's make a simple program to debug a memory issue.

```c
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int *ptr;
    *ptr = 42;
    printf("%d\n", *ptr);
    return 0;
}
```

Running this on my machine produces the following output:

```bash
(gdb) run
Starting program: /home/alex/dev/teaching/cse1320/CSE1320-Examples/debugging/a.out
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".

Program received signal SIGSEGV, Segmentation fault.
0x0000555555555159 in main () at forgot_alloc.c:6
6	    *ptr = 42;
```

We already get a lot of information that would have taken us much longer to acquire through the simple `printf` method. Without adding any debugging statements, we know the exact line which caused the segmentation fault. We also know that it was caused by a `SIGSEGV` signal. This is a signal that is sent to a process when it tries to access memory that it does not have access to. In this case, we are trying to write to a memory address that we have not allocated. This is a very common mistake in C programs.


#### Using breakpoints {#using-breakpoints}

We can use breakpoints to stop execution at a specific line. This is useful if we want to inspect the state of the program at a specific point in time. We can set a breakpoint at a specific line with the `break` command. For example, if we want to stop execution at line 6, we can do this:

**GDB**

```bash
(gdb) break 6
Breakpoint 1 at 0x555555555159: file forgot_alloc.c, line 6.
```

**LLDB**

```bash
(gdb) break 6
Breakpoint 1 at 0x555555555159: file forgot_alloc.c, line 6.
```

Running the program again from the beginning will stop execution at line 6. We can then inspect the value of `ptr` with the `print` command.

```bash
(gdb) run
The program being debugged has been started already.
Start it from the beginning? (y or n) y
Starting program: /home/alex/dev/teaching/cse1320/CSE1320-Examples/debugging/a.out
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".

Breakpoint 1, main () at forgot_alloc.c:6
6	    *ptr = 42;
(gdb) print ptr
$1 = (int *) 0x555555555060 <_start>
```

I wanted to start with this example, because you should _never_ let this scenario happen. I'm not saying that you should always write bugless code, but this example highlights a very common mistake: forgetting to initialize your variables. In C, every single pointer variable should always be initialized to `NULL`. Let's make that change, recompile the program, and debug again. Again, set a breakpoint at line 6 so we can see the value of `ptr`.

```c
(gdb) print ptr
$1 = (int *) 0x0
```

Ensuring that our pointers are always initialized to `NULL` saves us a LOT of headaches.


#### Inspecting Memory {#inspecting-memory}

We can inspect the memory of our program with the `x` command. This command takes two arguments: the number of units to print and the format. For example, if we want to print the first 10 bytes of memory, we can do this:

```bash
(gdb) x/10b ptr
0x0:	0	0	0	0	0	0	0	0
0x8:	0	0
```


## Examples {#examples}

1.  Allocate memory in a function that will inevitably lead to a segmentation fault.
2.  Discover the original source of the allocation.
