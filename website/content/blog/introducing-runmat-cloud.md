---
title: "More Math Per Hour: Introducing RunMat Cloud"
description: "Every jump in computation per engineer has unlocked new categories of engineering. RunMat brings automatic GPU acceleration to a browser-based runtime, and RunMat Cloud makes it persistent: projects, run history, version snapshots, and collaboration."
date: "2026-03-10"
authors:
  - name: "Nabeel Allana"
    url: "https://x.com/nabeelallana"
readTime: "8 min read"
slug: "introducing-runmat-cloud"
visibility: unlisted
tags: ["RunMat Cloud", "sandbox", "scientific computing", "GPU computing", "WebAssembly", "WebGPU", "collaboration"]
keywords: "RunMat Cloud, RunMat sandbox, MATLAB online alternative, browser GPU computing, WebAssembly scientific computing, WebGPU math, engineering collaboration tool, version control scientific computing, MATLAB cloud alternative"
excerpt: "Every jump in computation per engineer has unlocked new categories of engineering. RunMat brings automatic GPU acceleration to a browser-based runtime. RunMat Cloud makes it persistent. No install required."
image: "https://web.runmatstatic.com/blog-images/introducing-runmat-cloud.png"
imageAlt: "RunMat Cloud project environment with run history"
ogType: "article"
ogTitle: "More Math Per Hour: Introducing RunMat Cloud"
ogDescription: "Automatic GPU acceleration in a browser-based runtime. RunMat Cloud makes it persistent: projects, run history, version snapshots, and team collaboration."
twitterCard: "summary_large_image"
twitterTitle: "More Math Per Hour: Introducing RunMat Cloud"
twitterDescription: "Automatic GPU acceleration in your browser. Persistent projects, run history, and team collaboration. Open source, free tier available."
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
      headline: "More Math Per Hour: Introducing RunMat Cloud"
      alternativeHeadline: "GPU-Accelerated Math in Your Browser, Now with Persistent Projects"
      description: "Every jump in computation per engineer has unlocked new categories of engineering. RunMat brings automatic GPU acceleration to a browser-based runtime, and RunMat Cloud adds persistent projects, run history, and collaboration."
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

Engineering progress follows a recursive pattern. Better math leads to better computation, which produces better models, which enables better engineering, which builds better machines, which demand even more computation. The loop has been running for centuries. And at each major jump in what a single engineer can compute per hour, the result has been new categories of engineering that the previous era could not have attempted.

```
  better math
      ↓
  better computation
      ↓
  better models
      ↓
  better engineering
      ↓
  better machines
      ↓
  even more computation
      ↓
  (loop repeats)
```

When [ENIAC](https://en.wikipedia.org/wiki/ENIAC) came online in 1945, it could do roughly 5,000 additions per second, about 10,000 times what a room of human computers could manage. Stanislaw Ulam later wrote that the thermonuclear weapon design it enabled was impossible to do by hand, at any speed. That same machine made [numerical weather prediction](https://en.wikipedia.org/wiki/Numerical_weather_prediction#History) and early finite element analysis feasible for the first time.

![Glen Beck and Betty Snyder programming ENIAC at the Ballistic Research Laboratory, circa 1947. U.S. Army photo, public domain.](https://upload.wikimedia.org/wikipedia/commons/d/d3/Glen_Beck_and_Betty_Snyder_program_the_ENIAC_in_building_328_at_the_Ballistic_Research_Laboratory.jpg)

In the late 1970s, Cleve Moler saw a version of the same barrier at the University of New Mexico. His students needed LINPACK and EISPACK for serious linear algebra, but both required FORTRAN. So he [wrote MATLAB](https://www.mathworks.com/company/newsletters/articles/the-origins-of-matlab.html) to let them work with matrices directly, without a systems language in the way. He removed a barrier between engineers and computation, and it worked. MATLAB became the standard for over 40 years.

Several things have changed since MATLAB was designed. GPUs are now in every laptop, but MATLAB gates GPU access behind the Parallel Computing Toolbox and requires CUDA, making CUDA the new FORTRAN: an unnecessary language between the engineer and the computation. Browsers can run compiled code at near-native speed through WebAssembly and access GPUs through WebGPU, neither of which existed when MATLAB was architected. Open source has become the norm: Python with NumPy and SciPy has already taken a large share of MATLAB's market, proving the demand for free, open tools. And AI models can now write, run, and iterate on mathematical code, which means engineering tools need to be built for models to use, not just humans. MATLAB remains proprietary and expensive.

RunMat is built for these current constraints. The runtime [automatically routes math across CPU and GPU](/blog/runmat-accelerate-fastest-runtime-for-your-math) without code changes or paid add-ons. It compiles to WebAssembly and runs in a browser, so there is nothing to install. It is [open source](https://github.com/runmat-org/runmat) and MIT licensed. We [launched the CLI](/blog/introducing-runmat) last August at 150–180x faster than Octave on the same workloads. In November we shipped [Accelerate](/blog/runmat-accelerate-fastest-runtime-for-your-math) for automatic GPU acceleration. Now the sandbox puts that same engine in your browser, and RunMat Cloud makes it persistent.

## The sandbox: RunMat in your browser

The sandbox at [runmat.com/sandbox](https://runmat.com/sandbox) runs the full RunMat runtime in your browser, the same engine and [GPU acceleration](/blog/runmat-accelerate-fastest-runtime-for-your-math) as the CLI. If your browser supports WebGPU (Chrome, Edge, Safari 18+, Firefox 139+), the fusion engine routes work to your local GPU automatically. You get a built-in editor with linting and autocomplete, interactive 2D and 3D plotting, and all computation happens locally on your machine. Browsers do throttle GPU time, so peak throughput on large workloads will be lower than the native CLI. The upcoming desktop app removes this tradeoff: native GPU access with the same editor and cloud features.

## RunMat Cloud: your work persists

Sign in with Google, Apple, Microsoft, a passkey, or your email, and the sandbox becomes a persistent project. Your files save to cloud storage. Every run records automatically — code, workspace variables, figures, exact camera angle on a 3D surface. Come back tomorrow or from a different machine and everything is where you left it. No checkpoint files to manage. We wrote about [how run state restoration works](/blog/restoring-historical-run-state-scientific-numerical-calculations), [how the storage deduplicates across runs](/blog/ad-hoc-checkpoints-to-large-data-persistence), and [how snapshots export as git-compatible history](/blog/version-control-for-engineers-who-dont-use-git) for those who want the technical details.

Invite teammates to your project. Everyone sees the same files, the same run history, the same storage usage. When someone saves a file, teammates see the update immediately.

## What else shipped

Between the Accelerate launch in November and now, we shipped 3D plotting with interactive camera controls, async execution so long-running computations do not block the UI, shape inference for better autocomplete and error detection, linting in the editor, and logging and tracing for diagnostics. All of it carries over into the sandbox and cloud.

## Plans and pricing

RunMat Cloud launches with three tiers:

**Hobby (free)** — 100 MB of storage. Unlimited projects. Sign in and start working.

**Pro ($30/month per user)** — 10 GB of storage with on-demand overage. For individuals and small teams shipping real work.

**Team ($100/month per user)** — 100 GB of storage, SSO/SCIM, and admin controls. For organizations that need governance.

For teams in aerospace, defense, or any environment where data cannot leave the building: **RunMat Enterprise** deploys the entire platform on your own infrastructure. Same collaboration, same versioning, same AI integration, fully self-hosted inside your air-gapped network. No license server, no phone-home requirement. The open-source runtime is a single binary you carry in on approved media. We wrote a [detailed post on how RunMat works in air-gapped environments](/blog/mission-critical-math-airgap) for those who need it.

The full breakdown is on the [pricing page](/pricing).

## What has not changed

The sandbox is free, anonymous, and instant. The CLI is available for local execution and CI/CD. The fusion engine, the automatic CPU/GPU acceleration, and the interactive 2D/3D plotting are the same across all three surfaces: CLI, sandbox, and cloud. Anyone with a browser can use the same runtime as a team with an enterprise license.

One runtime, three ways to use it, all of them fast.

## Open source

The [full runtime is open source](https://github.com/runmat-org/runmat) and MIT licensed. You can install it from Homebrew (`brew install runmat-org/tap/runmat`), NPM (`npm install runmat`), or crates.io (`cargo install runmat`). The NPM package ships the complete runtime — execution, GPU acceleration, and plotting — so you can embed RunMat inside your own web tools. There is also a Jupyter kernel if notebooks are your workflow. Everything that powers the sandbox and RunMat Cloud starts in this repo.

## What is next

The native desktop application mentioned above is in active development, alongside an integrated AI assistant for the cloud environment. Each step continues the same work: more math per hour, for more engineers. We will write about each when they are ready to use.

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
