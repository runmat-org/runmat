---
title: "Choosing Rust for LLM-Generated Code"
description: "Why the training distribution of programming languages matters for LLMs, and how Rust improves reliability in model-written code."
date: "2025-09-05"
author: "Nabeel Allana"
readTime: "8 min read"
slug: "rust-llm-training-distribution"
tags: ["Rust", "LLM", "MATLAB", "RunMat", "Programming Languages"]
keywords: "Rust LLM code generation, training distribution, programming languages, RunMat, scientific computing"
excerpt: "Training data quality shapes LLM outputs. Rust’s tight distribution and strong compiler feedback make it uniquely suited for reliable model-assisted coding."
image: "/website/public/rust-blog-image.png"
imageAlt: "Why Rust for LLM code gen"
ogType: "article"
ogTitle: "Why Rust Improves LLM-Written Code"
ogDescription: "Rust offers tighter training distribution, stronger compiler checks, and better linting feedback — making it ideal for LLM-assisted development."
twitterCard: "summary_large_image"
twitterTitle: "Rust + LLMs: Why Distribution Matters"
twitterDescription: "Rust narrows the training distribution and provides strong compiler feedback, enabling more reliable model-written code."
canonical: "https://runmat.org/blog/rust-llm-training-distribution"
---

![Why Rust](/website/public/rust-blog-image.png)



## 📚 Training Distribution: Why the Corpus Matters

I’ve been programming for decades and have learned 15-20 different languages, from C/C++ to Verilog/VHDL, Terraform, and TypeScript. Despite their surface differences, most languages share a common set of building blocks: loops, functions, variables, and conditionals. Once you’ve learned a few, the patterns are recognizable across all of them. 

Large language models learn in much the same way. They encode the underlying concepts within programming well. A for loop looks different in Python than in Rust, but the underlying idea is consistent, and models encode that reliably. 

Where models struggle is at the last mile of programming, where languages diverge. Style, quality, and correctness vary widely. Models don’t distinguish between good and bad practice; they only reflect the distribution they’ve seen. If that distribution is noisy, as it often is in ecosystems with a lot of inconsistent code, the outputs will be noisy as well. If the distribution is tighter, the outputs are much more reliable. \

A couple of weeks ago, I started the most ambitious coding experiment of my career so far. I wanted to push these models to their edge and see what they are really capable of today. Building RunMat — a MATLAB-compatible runtime — meant building a brand new compiler and runtime from scratch. For a project of this kind, the practical choices were Go, C/C++, or Rust. I chose Rust not only because it was well-suited to building a runtime, but also because I wanted to test a hypothesis: given that much of our work at Dystr had recently been in TypeScript, would LLMs prove more productive in Rust than in TypeScript? And because a compiler/runtime has such a large implementation surface, it offered an unusually good way to evaluate that productivity delta.

---

## 🔑 Three Reasons Rust Improves Model-Written Code

Three factors made Rust stand out for building RunMat in an “LLM-as-the-primary-coder” workflow: the quality of its training distribution, the strength of its type system, and the resulting strength of its linting system.



### Training Distribution

TypeScript is one of the most popular languages taught in coding bootcamps, which has made it a dominant presence in public repositories. But this volume comes at the cost of quality. Inconsistent style and beginner-level mistakes are common, and models trained on that data inherit the noise. Rust’s public corpus is narrower and generally higher in quality, shaped by a community that prizes correctness. Rust also has well-defined conventions for project organization and testing, areas where other ecosystems often diverge. That lack of variability further narrows the distribution, leaving models with cleaner, more consistent examples to learn from. The result is a tighter distribution, which is reflected in more reliable, idiomatic output. 

### Type system and feedback. 

Rust’s compiler adds another layer of advantage. Its strong typing and borrow checker produce detailed, structured errors at compile time, catching problems before code ever runs. For humans, this is valuable; for LLMs, it’s an immediate feedback loop. Each generated snippet is validated against strict rules, which helps models converge faster on usable solutions. By contrast, TypeScript’s permissive type system leaves more room for ambiguous or deferred errors, giving models less guidance during generation. 

### Linting system and tooling. 

Rust’s typing also powers a strong linting ecosystem that provides immediate feedback after each edit, often before the code is even run. Problems that are notoriously hard to debug in other languages, such as tricky memory patterns, appear instantly as red underlines. In practice, using Cursor made this especially effective: the linter’s signals showed up straight away, allowing the model to fix issues within seconds before the broader task context was lost. Tools like Clippy add further refinements, making the feedback loop even tighter. 

Together, these three factors: higher-quality training data, stronger compiler checks, and immediate linting feedback made Rust uniquely well-suited for testing how far LLMs can be pushed in a serious project. 



## 📊 Productivity Over Popularity

 A frequent reason organizations choose TypeScript over Rust is the availability of a larger hiring pool. TypeScript is one of the most widely taught and adopted languages, while Rust expertise is less common. On paper, the larger pool appears to be the safer choice. 
 
 But in the context of LLM-assisted development, the calculation shifts. If models generate more reliable Rust code because of the quality of its training distribution and the strictness of its compiler, the barrier to finding Rust specialists is lower than it once was. Developers can lean on models for a greater share of the routine work, and the compiler provides strong guardrails for learning the rest. 
 
 This does not remove the difficulty of Rust as a language, but it does change the tradeoff. The choice is no longer just about how many developers already know a language, but also how well that language aligns with model capabilities. In that light, the “smaller pool” argument against Rust carries less weight than it did even a few years ago. 

---

## 🏗️ Engineering Benefits Beyond the LLM Fit

 Beyond its advantages in training distribution and compiler feedback, Rust offered practical benefits that supported the development of RunMat. 
 
 - Cross-platform support. With Cranelift as a backend, Rust made it straightforward to target multiple architectures, from x86-64 and AArch64 servers to WebAssembly environments. 
 
- Memory management. Rust’s ownership model eliminated many of the subtle crashes common in C and C++ runtimes, while still allowing fine-grained control needed for numerical workloads. 
 - Simple + powerful GPU abstraction. Through WGPU, RunMat gained a clean abstraction over vendor-specific graphics APIs, enabling GPU acceleration and plotting without fragmented, device-specific code. 
 - Modular design. Rust’s crate ecosystem encouraged a clean separation between the lexer, parser, interpreter, JIT, and runtime components, which improved testability and lowered the barrier for contributors.

These were not the primary reasons for choosing Rust, but they made the decision easier. Once the project was underway, they reinforced the sense that Rust was not only a better fit for LLM-assisted coding but also a strong foundation for building a modern scientific runtime. 

---

## 🚀 Performance at a Glance

Direct performance comparisons to MATLAB are not possible, as benchmarking is explicitly prohibited under their license terms: a restriction that seems at odds with the principles of open science and engineering practice. Independent sources suggest startup times in the range of 2–10 seconds, driven by factors like JVM loading and large runtime initialization, but those numbers can only be cited second-hand. 

Instead, we benchmarked RunMat against GNU Octave, the open-source alternative. Octave is primarily written in C++, with parts in C and Fortran, which makes it a closer architectural peer to MATLAB than most modern languages and therefore a helpful benchmark. 

The results: 
- Startup: 914 ms in Octave vs. 5 ms in RunMat (183× faster) 
- Matrix operations: 822 ms in Octave vs. 5 ms in RunMat (164× faster) 
- Control flow: 876 ms in Octave vs. 5.7 ms in RunMat (154× faster) 
- Mathematical functions: 868 ms in Octave vs. 5.3 ms in RunMat (163× faster) 

These benchmarks highlight both the efficiency of the runtime and the advantage of a clean, modern architecture. All benchmark scripts are open and can be reviewed or reproduced here: https://github.com/runmat-org/runmat/tree/main/benchmarks 

What may be even more significant than the performance is how quickly the system came together. In three weeks, I conducted over 20,000 inference requests, supervised roughly 250 hours of LLM output, and assembled a codebase of more than three million characters, with over one million characters of tests and 250,000 characters of documentation. 

By traditional estimates, achieving 100% MATLAB language grammar and semantics coverage would have required three to five senior engineers working for two or more years, or five years of work for a single developer. RunMat reached that milestone in weeks. 

![Runmat vs GNU Octave](/website/public/runmat-octave-benchmark-data.png)


---

## 🔤 Language Matters Less; Orchestration Matters More 

The biggest surprise was how the workflow itself changed. I wasn’t writing code in the way I had for the last 25 years. I was designing architectures, shaping test cases, curating context, and reviewing thousands of small changes generated by the model. The role shifted from writing every line to steering, supervising, and iterating — a fundamentally different way of building software. 

Language choice is no longer solely about runtime speed or the availability of developers. It is also about how well a language’s ecosystem aligns with the strengths and limitations of large language models. For me, Rust provided that alignment, and it turned what would once have been a multi-year effort into a matter of weeks. 

We have a few more updates coming for RunMat over the next few months, including a world-class plotting system, a clean core standard library, and a modern package system and builder. Each is an opportunity to continue experimenting with the edge of what model-driven development can achieve. 

**Subscribe to our newsletter in the footer to receive updates on Runmat's progress**


