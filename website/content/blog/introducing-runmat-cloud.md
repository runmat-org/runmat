---
title: "RunMat Cloud: From CLI to Browser to Persistent Projects"
description: "RunMat launched last August as a CLI. In November we added GPU acceleration with Accelerate. Now the sandbox puts the full runtime in your browser, and RunMat Cloud makes it persistent: projects, run history, version snapshots, and collaboration."
date: "2026-03-10"
authors:
  - name: "Nabeel Allana"
    url: "https://x.com/nabeelallana"
readTime: "7 min read"
slug: "introducing-runmat-cloud"
visibility: unlisted
tags: ["RunMat Cloud", "sandbox", "scientific computing", "GPU computing", "WebAssembly", "WebGPU", "collaboration"]
keywords: "RunMat Cloud, RunMat sandbox, MATLAB online alternative, browser GPU computing, WebAssembly scientific computing, WebGPU math, engineering collaboration tool, version control scientific computing, MATLAB cloud alternative"
excerpt: "Last August we shipped a fast CLI runtime. In November we added GPU acceleration. Now you can run the same engine in your browser, save your work across sessions, and collaborate. No install required."
image: "https://web.runmatstatic.com/blog-images/introducing-runmat-cloud.png"
imageAlt: "RunMat Cloud project environment with run history"
ogType: "article"
ogTitle: "RunMat Cloud: From CLI to Browser to Persistent Projects"
ogDescription: "The RunMat sandbox runs GPU-accelerated math in your browser. RunMat Cloud makes it persistent: projects, run history, version snapshots, and team collaboration."
twitterCard: "summary_large_image"
twitterTitle: "RunMat Cloud: From CLI to Browser to Persistent Projects"
twitterDescription: "GPU-accelerated math in your browser. Persistent projects, automatic run history, and team collaboration. Free tier available."
canonical: "https://runmat.com/blog/introducing-runmat-cloud"
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
          name: "Introducing RunMat Cloud"
          item: "https://runmat.com/blog/introducing-runmat-cloud"

    - "@type": "BlogPosting"
      "@id": "https://runmat.com/blog/introducing-runmat-cloud#article"
      headline: "RunMat Cloud: From CLI to Browser to Persistent Projects"
      alternativeHeadline: "GPU-Accelerated Math in Your Browser, Now with Persistent Projects"
      description: "RunMat launched as a CLI last August. Accelerate added GPU acceleration in November. Now the sandbox runs in your browser, and RunMat Cloud adds persistent projects, run history, and collaboration."
      image: "https://web.runmatstatic.com/blog-images/introducing-runmat-cloud.png"
      datePublished: "2026-03-10T00:00:00Z"
      dateModified: "2026-03-10T00:00:00Z"
      author:
        "@type": "Person"
        name: "Nabeel Allana"
        url: "https://x.com/nabeelallana"
      publisher:
        "@type": "Organization"
        name: "RunMat by Dystr"
        logo:
          "@type": "ImageObject"
          url: "/runmat-logo.svg"
      about:
        - "@type": "SoftwareApplication"
          name: "RunMat"
          sameAs: "https://runmat.com"
          applicationCategory: "ScientificApplication"
          operatingSystem: ["Windows", "macOS", "Linux"]
          offers:
            "@type": "Offer"
            price: "0"
            priceCurrency: "USD"

    - "@type": "FAQPage"
      mainEntity:
        - "@type": "Question"
          name: "Does RunMat Cloud run on my GPU?"
          acceptedAnswer:
            "@type": "Answer"
            text: "Yes. The browser sandbox uses WebGPU to access your local GPU for acceleration. Your code and data stay on your machine. RunMat Cloud stores project files and run metadata, not your GPU computations."
---

When we [launched RunMat](/blog/introducing-runmat) last August, you downloaded a binary and ran `.m` files from your terminal. It was fast (150x-180x faster than Octave on the same workloads) but you had to install it. In November we shipped [Accelerate](/blog/runmat-accelerate-fastest-runtime-for-your-math), which added automatic GPU acceleration through a fusion engine that plans your math across CPU and GPU without any code changes. Still a CLI. Still a download.

The runtime was fast, but a CLI is not for everyone. Engineers who live in MATLAB or Python often do not want to open a terminal just to try something new. That limited who could actually experience RunMat.

Between November and now, we also shipped a lot of runtime work that did not get its own announcement: 3D plotting with interactive camera controls, async execution so long-running computations do not block the UI, shape inference for better autocomplete and error detection as you type, linting in the editor so you see problems before you hit Run, and logging and tracing so we can diagnose issues quickly. These all shipped incrementally, and they all carry over into the sandbox and cloud.

Since then we have shipped two things that change how people actually get to RunMat. The sandbox puts the full runtime and its fusion engine in your browser: same GPU acceleration, nothing to install. And now RunMat Cloud makes that browser experience persistent: projects that persist across sessions, automatic run history, version snapshots, and team collaboration.

## The sandbox: RunMat in your browser

The RunMat sandbox at [runmat.com/sandbox](https://runmat.com/sandbox) runs the same runtime that powers the CLI, compiled to run in your browser. Open a tab, write MATLAB-style code, hit Run. If your browser supports WebGPU (Chrome, Edge, Safari 18+, Firefox 139+), the fusion engine routes work to your local GPU automatically. The [4K image pipeline](/blog/runmat-accelerate-fastest-runtime-for-your-math) from the Accelerate blog runs in the sandbox. This is not a stripped-down demo.

You also get things the CLI does not have: a built-in editor with linting and autocomplete, and interactive 2D and 3D plotting.

One caveat: browsers throttle GPU access. Your browser limits how much GPU time and memory a single tab can use, so the sandbox will not match the full throughput of the native CLI on large workloads. For most scripts and visualizations you will not notice. For large simulations where you need every bit of GPU performance, the CLI will be faster. The sandbox trades peak GPU performance for zero-install convenience. The upcoming desktop app will remove this tradeoff entirely: the same editor, linting, plotting, and cloud features as the sandbox, but running natively on your machine with full, unrestricted GPU access.

All computation happens locally in your browser. When you sign into Cloud, your project files and run metadata are stored on our servers for persistence, but the computation still happens on your machine.

## RunMat Cloud: your work persists

RunMat Cloud is what you get when you sign in. Log in with Google, Apple, Microsoft, a passkey, or your email, and the sandbox becomes a persistent project. Your files save to cloud storage. Every run records automatically. Come back tomorrow, next week, or from a different machine, and everything is where you left it.

Every time you hit Run, we capture the full state: your code, workspace variables, and figures. Open the history dropdown, pick a previous run, and the workspace reconstructs — exact values, exact plot state, exact camera angle on a 3D surface. You do not configure this. There are no checkpoint files to manage. We wrote about [how run state restoration works](/blog/restoring-historical-run-state-scientific-numerical-calculations) in detail.

The storage is smart about duplicates. If 90% of your workspace is unchanged between runs, that 90% is stored once, so your storage usage reflects actual changes, not redundant copies. The [technical details are here](/blog/ad-hoc-checkpoints-to-large-data-persistence) for those who want them.

Run history is automatic. Snapshots are deliberate — you take one before a milestone, a test campaign, or a major parameter sweep. One click freezes the full project state. Snapshots form a linear chain with an audit trail: who, what, when. The chain [exports as git-compatible history](/blog/version-control-for-engineers-who-dont-use-git) when compliance or reproducibility requires it.

Invite teammates to your project. Everyone sees the same files, the same run history, the same storage usage. When someone saves a file, teammates see the update immediately. No more emailing scripts back and forth or wondering which version someone is running.

The dashboard shows what is using space and where. You can delete individual runs to reclaim storage. Pro and Team plans include on-demand overage if you exceed your included storage. We will notify you as you approach your limits, and you can set a usage cap so you never spend more than you are comfortable with. Billing is transparent: you see exactly what you are using and what it costs.

## Plans and pricing

RunMat Cloud launches with three tiers:

**Hobby (free)** — 100 MB of storage. Unlimited projects. Sign in and start working.

**Pro ($30/month per user)** — 10 GB of storage with on-demand overage. For individuals and small teams shipping real work.

**Team ($100/month per user)** — 100 GB of storage, SSO/SCIM, and admin controls. For organizations that need governance.

For teams in aerospace, defense, or any environment where data cannot leave the building: **RunMat Enterprise** deploys the entire platform on your own infrastructure. Same collaboration, same versioning, same AI integration, fully self-hosted inside your air-gapped network. No license server, no phone-home requirement. The open-source runtime is a single binary you carry in on approved media. We wrote a [detailed post on how RunMat works in air-gapped environments](/blog/mission-critical-math-airgap) for those who need it.

The full breakdown is on the [pricing page](/pricing).

## What has not changed

The sandbox is free, anonymous, and instant. The CLI is available for local execution and CI/CD. The fusion engine, the automatic CPU/GPU acceleration, and the interactive 2D/3D plotting are the same across all three surfaces: CLI, sandbox, and cloud.

One runtime. Three ways to use it. All of them fast.

## Open source

The [full runtime is open source](https://github.com/runmat-org/runmat) and MIT licensed. You can install it from Homebrew (`brew install runmat-org/tap/runmat`), NPM (`npm install runmat`), or crates.io (`cargo install runmat`). The NPM package ships the complete runtime — execution, GPU acceleration, and plotting — so you can embed RunMat inside your own web tools. There is also a Jupyter kernel if notebooks are your workflow. Everything that powers the sandbox and RunMat Cloud starts in this repo.

## What is next

The native desktop application mentioned above is in active development, alongside an integrated AI assistant for the cloud environment. We will write about each when they are ready to use.

For now: open [runmat.com/sandbox](https://runmat.com/sandbox), sign in, and save your first project. It will be there when you come back.

## FAQ

**Does RunMat Cloud run on my GPU?**
Yes. The browser uses WebGPU to access your local GPU for acceleration. All computation happens locally in your browser. RunMat Cloud stores your project files and run metadata on our servers for persistence, but your math is never executed on our infrastructure.

**How is this different from MATLAB Online?**
MATLAB Online runs your code on MathWorks' servers and requires a paid license. RunMat executes all computation locally in your browser via WebAssembly. Your math runs on your own hardware, not ours. It also uses your local GPU for acceleration (any vendor, no CUDA required), works offline after the initial page load, and requires no account to start. Cloud stores your project files for persistence and adds automatic versioning, but the computation never leaves your machine. We wrote a [full comparison](/matlab-online) for those evaluating the switch.

**What browsers support WebGPU?**
Chrome, Edge, Safari 18+, and Firefox 139+ all support WebGPU. If WebGPU is not available, the runtime falls back to CPU execution in the browser.

**Can I use RunMat Cloud for large simulations?**
Yes. The storage system is designed for large numerical outputs and deduplicates efficiently across runs. Storage limits depend on your plan. The Hobby tier includes 100 MB, and paid plans offer 10 GB to 100 GB with on-demand overage available.
