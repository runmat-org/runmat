---
title: "Why We Built RunMat"
description: "We started with Dystr, an AI-powered engineering workbook. Here's why we pivoted to RunMat, a fast, open-source MATLAB runtime that meets engineers where they are."
date: "2026-02-02"
authors:
  - name: "Nabeel Allana"
    url: "https://x.com/nabeelallana"
  - name: "Julie Ruiz"
    url: "https://www.linkedin.com/in/julie-ruiz-64b24328/"
readTime: "6 min read"
slug: "why-we-built-runmat"
tags: ["startup", "pivot", "MATLAB", "RunMat", "Dystr", "founder-story"]
keywords: "RunMat origin story, Dystr pivot, MATLAB runtime, startup lessons, engineering tools"
excerpt: "We started with Dystr, an AI-powered engineering workbook. Here's why we pivoted to RunMat, a fast, open-source MATLAB runtime that meets engineers where they are."
image: "https://web.runmatstatic.com/julie-nabeel-group-about-page.png"
imageAlt: "RunMat co-founders Nabeel Allana and Julie Ruiz"
ogType: "article"
ogTitle: "Why We Built RunMat"
ogDescription: "We started with Dystr, an AI-powered engineering workbook. Here's why we pivoted to RunMat, a fast, open-source MATLAB runtime that meets engineers where they are."
twitterCard: "summary_large_image"
twitterTitle: "Why We Built RunMat"
twitterDescription: "We started with Dystr. Here's why we pivoted to RunMat, a fast, open-source MATLAB runtime that meets engineers where they are."
canonical: "https://runmat.com/blog/why-we-built-runmat"
jsonLd:
  "@context": "https://schema.org"
  "@graph":
    - "@type": "BreadcrumbList"
      itemListElement:
        - "@type": "ListItem"
          position: 1
          name: "RunMat"
          item: "https://runmat.com"
        - "@type": "ListItem"
          position: 2
          name: "Blog"
          item: "https://runmat.com/blog"
        - "@type": "ListItem"
          position: 3
          name: "Why We Built RunMat"
          item: "https://runmat.com/blog/why-we-built-runmat"

    - "@type": "BlogPosting"
      "@id": "https://runmat.com/blog/why-we-built-runmat#article"
      headline: "Why We Built RunMat"
      description: "We started with Dystr, an AI-powered engineering workbook. Here's why we pivoted to RunMat, a fast, open-source MATLAB runtime that meets engineers where they are."
      image: "https://web.runmatstatic.com/julie-nabeel-group-about-page.png"
      datePublished: "2026-02-02T00:00:00Z"
      dateModified: "2026-02-02T00:00:00Z"
      author:
        - "@type": "Person"
          name: "Nabeel Allana"
          url: "https://x.com/nabeelallana"
        - "@type": "Person"
          name: "Julie Ruiz"
          url: "https://www.linkedin.com/in/julie-ruiz-64b24328/"
      publisher:
        "@type": "Organization"
        name: "RunMat by Dystr"
        logo:
          "@type": "ImageObject"
          url: "https://runmat.com/runmat-logo.svg"
      about:
        - "@type": "SoftwareApplication"
          name: "RunMat"
          sameAs: "https://runmat.com"
          applicationCategory: "ScientificApplication"
          operatingSystem: ["Windows", "macOS", "Linux", "Browser"]
          offers:
            "@type": "Offer"
            price: "0"
            priceCurrency: "USD"
        - "@type": "Organization"
          name: "Dystr"
          sameAs: "https://dystr.com"
---

We started Dystr in 2022. Nabeel had gone from programmer to electromechanical engineer, worked at Apple and Toyota. He'd seen how much time engineers lose to outdated tools. We were betting those engineers would want AI to write their calculations for them. Describe the physics, get the code.

![RunMat co-founders Nabeel Allana and Julie Ruiz](https://web.runmatstatic.com/julie-nabeel-group-about-page.png)

We built what we thought would be the future: a cloud-native engineering workbook with an LLM at the center. Not a plugin or an assistant, but a complete rethinking of how engineers do calculations.

Early adopters got it immediately. We launched with 12 companies; 83% were paying after three months, with a third of sessions running two-plus hours. One CTO said we'd replaced $100K/year of labor.

We shut it down anyway.

The engineers who needed it most wouldn't touch it. And after two years, we finally understood why.

---

## What We Built

Dystr was a cloud-based engineering workbook. You'd describe a calculation (thermal analysis, control system model, whatever) and an LLM would generate the code. But you could see every line, edit it, save it, version it. The AI was the starting point, not a black box. Visual outputs, collaboration, the works. We thought we were building the way engineers would work in five years.

![Dystr's AI Engineering Platform](https://web.runmatstatic.com/dystr-ai-eng-platform.png)


This was before code generation was everywhere. Before Copilot was in every IDE, before ChatGPT could write Python. We were early, and for a while that felt like an advantage.

The engineers who adopted Dystr loved it. They'd spend hours in the product. They'd tell us it was saving them weeks. Some teams built their entire workflow around it.

But these were early adopters, people excited to try something different. When we tried to expand beyond them, we hit resistance we didn't expect.

---

## Why It Didn't Scale

Three things worked against us, and they happened at the same time.

Dystr wasn't a tool you could drop into an existing process. It required learning to prompt effectively, trusting code you didn't write, and changing how you organized your work. Early adopters embraced this. They wanted something different. But most engineers aren't early adopters. They have deadlines. They have scripts that already work. They're not looking for a new platform; they want their current tools to suck less.

![Dystr's AI Engineering Report Generation](https://web.runmatstatic.com/dystr-eng-report.png)



We built collaboration in from day one, thinking teams would naturally start sharing work with each other. They didn't. People used Dystr solo, even when their teammates had accounts.

On top of that, the engineers who most needed better tools (aerospace, defense, medical devices) couldn't use a cloud product with their IP. Security reviews, procurement cycles, institutional risk aversion. Even if they wanted to try Dystr, they couldn't get approval. We kept hearing the same thing: "I'd love to use this, but I can't get it past security." Or worse: "I can't even ask, because then I have to explain why I was looking at outside tools."

And then the ground moved underneath us. When we started building Dystr, ChatGPT didn't exist yet. By the time we had real traction, it could write Python scripts and Copilot was in every IDE. The thing that made Dystr different, AI that writes your code, was suddenly everywhere, often for free.

We still added value. Our workflow was built specifically for engineers doing physics and math, not general-purpose coding. But the reason to adopt a whole new platform got weaker every month. Engineers could get 80% of the benefit by pasting their problem into ChatGPT and copying the output into their existing tools.

We were asking them to change everything for a differentiation that was disappearing.

---

## What Changed

We stopped trying to change how engineers work and started meeting them where they already were. Instead of asking them to learn something new, we'd make what they already do faster.

Most engineers we talked to already had a language: MATLAB. They'd learned it in school, used it for years, had folders full of scripts. The syntax was fine. What wasn't fine was the runtime: slow startup, expensive licenses, no GPU support, stuck on 1990s architecture.

So we built a new runtime. Same MATLAB syntax they already know. Way faster. Runs locally on their hardware. Uses their GPU automatically. Costs nothing.

Bring your existing MATLAB scripts and run them. That's it.

We went from "here's a new way to work" to "here's your old way, but 100x faster."

---

## What RunMat Is Today

RunMat is a single binary you download and run. No license server, no account required. It also runs in the browser, and your code executes locally in WebAssembly, not on our servers.

Under the hood: Rust, a JIT compiler, and automatic GPU acceleration with kernel fusion. Works on NVIDIA, AMD, Intel, and Apple chips without changing your code.

On GPU-accelerated workloads (Monte Carlo simulations, elementwise math, image processing) we're seeing 10-130x speedups over NumPy and 2-5x over PyTorch. The benchmarks are open source if you want to run them yourself.

![RunMat performance for elementwise math performance](https://web.runmatstatic.com/elementwise-math_speedup-b.svg)





The whole runtime is on GitHub. MIT licensed. If you want to read how the JIT works or why we made a particular tradeoff, you can. CLI-first, transparent telemetry you can audit, and even the browser sandbox keeps your code local. It runs in your browser, not on our servers.



Performance is the moat. When we watched AI code generation get commoditized in real time, we learned that the differentiation has to be something harder to copy. Speed compounds. Every optimization we ship makes RunMat more valuable relative to alternatives.

---

## Where We're Going

RunMat already runs in the browser. No installation, no login, just open and start writing. Live syntax validation, execution tracing, async support that MATLAB can't match.

![RunMat Sandbox Environment](https://web.runmatstatic.com/runmat-sandbox-dark.png)


But the real goal is what we originally wanted to build with Dystr: an AI that actually helps engineers do math and physics. Not a copilot that writes a first draft and leaves. An agent that can iterate, run code, check plots, and work through problems step-by-step.

We just had to build the foundation first. A fast, local-first runtime that engineers can adopt without procurement headaches or workflow changes. Once people trust RunMat with their real work, the AI becomes something that genuinely helps rather than another thing to evaluate and get approved.

---

## Why This Matters

We were early to LLM code generation, but we built it into the wrong product. We tried to change how engineers work when we should have been accelerating how they already work.

Dystr taught us that you can't ask engineers to adopt a new platform *and* trust a startup's cloud at the same time, especially when the differentiation is eroding underneath you. You have to earn trust first, with something that runs locally, works offline, and doesn't require anyone to change their habits.

That's RunMat. Same syntax engineers have used for 40 years, modern runtime underneath, and eventually an agent that actually understands what you're trying to do.

We're still working on the same problem we started with. We just know a lot more about how to solve it now.

---

**Ready to try it?** [Open the RunMat sandbox](https://runmat.com/sandbox) and run your first script in seconds. No install, no signup.

Have questions or want to share feedback? Reach us at [team@runmat.com](mailto:team@runmat.com).
