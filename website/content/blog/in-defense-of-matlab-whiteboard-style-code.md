---
title: "In Defense of MATLAB: Why Engineers Still Need Whiteboard-Style Code"
description: "MATLAB still shines for math-heavy work. Its whiteboard-style syntax makes code easy to read, review, and run on modern tools like RunMat and fast GPUs."
date: "2025-12-11"
author: "Nabeel Allana"
readTime: "8 min read"
slug: "in-defense-of-matlab-whiteboard-style-code"
tags: ["matlab", "runmat", "whiteboard-code", "scientific-computing", "engineering-math", "llm-codegen"]
keywords: "MATLAB, MATLAB syntax, whiteboard-style code, RunMat, scientific computing, engineering math, code review, LLM code generation, MATLAB alternative"
excerpt: "People joke about MATLAB, but the syntax is not the problem. Whiteboard-style code is still one of the safest ways for engineers to write and review math-heavy logic. This article explains why readable math matters in an AI coding world and how RunMat keeps the syntax while modernizing the engine."
image: "https://web.runmatstatic.com/whiteboard-math-vs-matlab-syntax-sm.png"
imageAlt: "Engineer copying matrix equations from a whiteboard into MATLAB-style code on a laptop."
ogType: "article"
ogTitle: "In Defense of MATLAB: Why Engineers Still Need Whiteboard-Style Code"
ogDescription: "MATLAB still shines for math-heavy work. Its whiteboard-style syntax makes code easy to read, review, and run on modern tools like RunMat and fast GPUs."
twitterCard: "summary_large_image"
twitterTitle: "In Defense of MATLAB: Why Engineers Still Need Whiteboard-Style Code"
twitterDescription: "Whiteboard-style MATLAB code is still a superpower for engineers. Here’s why readable math matters for safety, code review, and AI-assisted workflows."
canonical: "https://runmat.org/blog/in-defense-of-matlab-whiteboard-style-code"
---


*The problem was never the syntax—it was the runtime. Here is why readable math still matters in an LLM-assisted code gen world.*

## Introduction

If you look at the most preferred language list on any Stack Overflow developer survey, you will usually find MATLAB hovering near the bottom. It sits there alongside VBA and COBOL, often dismissed by modern software engineers as a dinosaur. You have probably seen the memes: complaints about license manager errors, the massive install size, or the feeling that it is a language strictly for "old-school academics.”

The world has moved toward open source, containerization, and agile cloud deployments. In that context, a closed ecosystem feels restrictive.

But if you walk into the R&D departments of top aerospace, automotive, or medical device companies, MATLAB is still everywhere. It isn't there because these engineers don't know better. It is there because, for a specific type of work—linear algebra, signal processing, and control theory—MATLAB did one thing better than almost anyone else:

**It made the code look exactly like the math on the whiteboard.**

We need to separate the **language syntax** (which is excellent) from the **runtime and business model** (which are dated).

## What is "Whiteboard-Style Code"?

![Engineer copying matrix equations from a whiteboard into MATLAB-style code on a laptop.](https://web.runmatstatic.com/whiteboard-math-vs-matlab-syntax-sm.png)

When I say "whiteboard-style code," I am referring to a specific level of abstraction.
In engineering, the "truth" is derived on a whiteboard or a notepad. That is where the physics is worked out. You draw matrices, you transpose vectors, and you define relationships ($F = ma$, $V = IR$, $Y = Hx$).
The goal of engineering software is to translate that whiteboard truth into executable logic with as little "translation loss" as possible.
"Whiteboard-style code" means:
1. **High Density:** One line of math equals one line of code.
2. **Visual Similarity:** The code visually resembles the equation.
3. **Low Boilerplate:** No memory allocation logic, no type declarations, and minimal imports.
For vectors, matrices, and arrays, MATLAB’s syntax is often the shortest distance between the board and the running code

## The Translation Test: Board vs. Code

Let’s look at a concrete example. Imagine you are sketching a simple linear algebra operation during a lecture or a design review.

![Whiteboard translation placeholder](/placeholder-image.png)

**In MATLAB:**
The code is almost a direct transcription of the board:

```matlab
X = [1, 2, 3];
Y = [1, 2, 3; ...
     4, 5, 6; ...
     7, 8, 9];

Z = Y * X';      % The ' operator is transpose
W = [Z, Z];      % Brackets imply concatenation
```

**In Python (NumPy):**
Python is an incredible language, and NumPy is a powerhouse. But notice the cognitive load required to handle the shapes explicitly:

```python
import numpy as np

X = np.array([1, 2, 3])
Y = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# We must reshape X to be a column vector (3,1)
# or rely on broadcasting rules carefully.
Z = Y @ X.reshape(3, 1)

# Concatenation requires a function call and an axis definition
W = np.concatenate([Z, Z], axis=1)
```

The math is identical. But in the Python version, the engineer is thinking about computer science concepts: `imports`, `methods`, `tuples`, and `axes`. In the MATLAB version, the engineer is thinking about linear algebra: *rows, columns, and multiplication.*

## Why Readable Math is a Safety Feature

Why does this subtle difference matter?

In mission-critical fields, **code review is safety review.**

Senior engineers and Principal Investigators perform these reviews. These are brilliant people who understand the physics deeply, but they may not be experts in modern software design patterns. They don't want to parse decorators or understand the nuances of an object-oriented class hierarchy.

When they review code, they are holding their derivation notes in one hand and looking at the screen with the other. They want to verify that step A leads to step B.

- "Here is the rotation matrix."
- "Here we apply the filter."
- "Here we calculate the error."

MATLAB’s syntax allows them to verify the *math* without getting bogged down in the *implementation*. When the code looks like the math, bugs have fewer places to hide. The syntax itself reduces the cognitive load, allowing the reviewer to focus on the physics rather than the programming.

![MATLAB code review](https://web.runmatstatic.com/matlab-code-review-sm.png)

## It’s Not Just for Humans (Compilers Love It Too)

There is a misconception that "simple syntax" means "slow interpretation." Actually, high-level array syntax gives runtimes and compilers excellent signals for optimization.

When you write `C = A * B` in a vectorized language, you are giving the runtime a very high-level instruction: "Perform a matrix multiplication on these two objects."

Because the language constraints are strict (matrices have defined shapes, types are usually consistent), a modern runtime can:

- Immediately infer the shapes of the result.
- Fuse multiple element-wise operations into a single pass through memory.
- Offload the entire chunk of work to a GPU without the user writing a single line of CUDA kernel code.

The structure that makes it readable for humans also makes it predictable for machines.

## The Honest Truth: Why the Hate Exists

If the syntax is so good, why is the sentiment so mixed?

To be fair, the backlash against MATLAB is largely justified. The frustration usually stems from three areas, none of which have to do with the math syntax itself:

1. **The "Black Box" Runtime:** The engine is closed source. You cannot see how `fft` or `ode45` are implemented under the hood. For high-stakes engineering, not being able to audit your tools is a risk.
2. **Licensing Pain:** Everyone has a story about a simulation crashing because the license server timed out, or not being able to run a script because a colleague "checked out" the toolbox token.
3. **The Cloud Gap:** Modern engineering happens in CI/CD pipelines, Docker containers, and cloud clusters. Integrating a heavy, licensed desktop application into these lightweight, automated workflows is painful.

This friction pushed a generation of engineers toward Python, not because they preferred writing `np.concatenate`, but because they needed tools that played nice with the modern stack.

![MATLAB meme](https://web.runmatstatic.com/matlab-meme-sm.png)

## A Vision for a Modern "Whiteboard" Runtime

The solution isn't to abandon the syntax that engineers love. The solution is to build a new, modern engine to run it.

We need a runtime that preserves the dense, array-oriented notation but operates like a modern piece of software infrastructure. It should be:

- **Open and inspectable:** No black boxes.
- **Hardware agnostic:** It should run on CPUs or GPUs without changing the code.
- **Portable:** It should run in a Docker container or a web browser.

We should keep the language surface that mimics the whiteboard, but swap out the engine for something designed for the era of cloud computing and massive datasets.  

## Keeping the Math, Changing the Engine

This is exactly why we are building **RunMat**.

Our team realized that the problem wasn't the `.m` files—it was how they were being executed. RunMat is a new, high-performance runtime designed to execute MATLAB-style syntax. It targets modern hardware (CPUs and GPUs) and integrates seamlessly into cloud and CI workflows, all without the traditional licensing headaches.

It allows teams to keep their "whiteboard code" while gaining the performance and portability of a modern software stack.

## Conclusion

Technology trends come and go, but the laws of physics and mathematics don't change.

Engineers working on the next generation of renewable energy grids, autonomous vehicles, and medical robotics need tools that respect the complexity of their math. They need code that can be written, read, and verified by experts who care more about differential equations than software dependencies.

The future doesn't need to copy the business models of the past. But it should absolutely keep the best part of the legacy: code that looks like the math on the board.

