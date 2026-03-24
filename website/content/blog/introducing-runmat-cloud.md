---
title: "Durable Project State: Introducing RunMat Cloud"
description: "RunMat launched last August as a CLI. In November we added GPU acceleration with Accelerate. Now the sandbox puts the full runtime in your browser, and RunMat Cloud makes it persistent: projects, run history, version snapshots, and collaboration."
date: "2026-03-10"
authors:
  - name: "Nabeel Allana"
    url: "https://x.com/nabeelallana"
readTime: "9 min read"
slug: "introducing-runmat-cloud"
visibility: unlisted
tags: ["RunMat Cloud", "sandbox", "scientific computing", "GPU computing", "WebAssembly", "WebGPU", "collaboration"]
keywords: "RunMat Cloud, RunMat sandbox, MATLAB online alternative, browser GPU computing, WebAssembly scientific computing, WebGPU math, engineering collaboration tool, version control scientific computing, MATLAB cloud alternative, durable project state, run history, version snapshots"
excerpt: "RunMat launched as a CLI last August. Accelerate added GPU acceleration in November. Now the sandbox runs the same engine in your browser, and RunMat Cloud turns each run into durable project state you can keep, inspect, compare, and share."
image: "https://web.runmatstatic.com/blog-images/introducing-runmat-cloud.png"
imageAlt: "RunMat Cloud project environment with run history"
ogType: "article"
ogTitle: "Durable Project State: Introducing RunMat Cloud"
ogDescription: "The RunMat sandbox runs GPU-accelerated math in your browser. RunMat Cloud makes it persistent: durable project state with run history, version snapshots, and team collaboration."
twitterCard: "summary_large_image"
twitterTitle: "Durable Project State: Introducing RunMat Cloud"
twitterDescription: "GPU-accelerated math in your browser. Durable project state with automatic run history, version snapshots, and team collaboration. Free tier available."
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
      headline: "Durable Project State: Introducing RunMat Cloud"
      alternativeHeadline: "GPU-Accelerated Math in Your Browser, Now with Persistent Projects"
      description: "RunMat launched as a CLI last August. Accelerate added GPU acceleration in November. Now the sandbox runs in your browser, and RunMat Cloud turns each run into durable project state: run history, version snapshots, and collaboration."
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
        - "@type": "Question"
          name: "How does RunMat Cloud make large datasets practical?"
          acceptedAnswer:
            "@type": "Answer"
            text: "RunMat Cloud uses sharded storage and manifest-based versioning, so large datasets do not require full-file rewrites or whole-project duplication to preserve history. This makes it practical to keep long experiment sequences, restore historical state, and inspect results without treating every checkpoint like a brand-new full copy."
        - "@type": "Question"
          name: "Can I restore a run someone else did?"
          acceptedAnswer:
            "@type": "Answer"
            text: "Yes. In a shared project, historical runs, snapshots, and project state are not tied to one machine. Teammates can inspect prior results directly instead of asking the original author to rerun the work."
---

Engineering progress follows a recursive pattern: better math leads to better computation, which produces better models, which enables better engineering, which builds better machines, which demand even more computation.

At each major jump in what one engineer can compute per hour — and in how easily they can access that computation — the result has been new categories of engineering that the previous era could not have attempted.

```text
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

When ENIAC came online in 1945, it made categories of computation practical that had been effectively impossible by hand. In the late 1970s, Cleve Moler saw a different version of the same barrier: his students needed serious linear algebra tools, but they were trapped behind FORTRAN. So he wrote MATLAB to make that computation accessible to more engineers. It worked, and MATLAB became the standard for decades.

Several things have changed since then. GPUs are now in every laptop, but MATLAB still [gates GPU access behind a paid toolbox](/blog/how-to-use-gpu-in-matlab) and CUDA. Browsers can now run compiled code at near-native speed through WebAssembly and access GPUs through WebGPU. And open source should be the expectation for engineering software, not a bonus: if you cannot fully inspect the system producing a result, you leave proof gaps in the reasoning built on top of it.

We are building RunMat for the world as it exists now: GPUs in every laptop, browsers as a real application runtime, and an expectation that engineering tools should be performant, open, inspectable, and easy to use.

We wanted mathematical computing to use the hardware engineers already have, without code changes, paid add-ons, or extra layers between the engineer and the machine. RunMat automatically routes math across CPU and GPU, compiles to any processor architecture (or WebAssembly), and can run inside the browser's built-in sandbox, so there is nothing to install and isolation comes by default. It is open source and MIT licensed. Last August we [launched the CLI](/blog/introducing-runmat) at 150–180x faster than Octave on the same workloads. In November we shipped [Accelerate](/blog/runmat-accelerate-fastest-runtime-for-your-math) for automatic GPU acceleration and benchmarked at 82x faster than PyTorch (the Python GPU acceleration framework) at elementwise matrix math on 1B points on a commodity Apple MacBook Pro M2 ([exact benchmark here](/blog/runmat-accelerate-fastest-runtime-for-your-math)).

Now we are taking the next step: putting that same engine in your browser and turning each run into durable project state you can keep, inspect, compare, and share.

![RunMat Cloud project environment with run history](https://web.runmatstatic.com/blog-images/introducing-runmat-cloud.png)

## The sandbox: immediate compute, zero setup

[runmat.com/sandbox](https://runmat.com/sandbox)

Open a URL, write code, run it, and get the full RunMat runtime immediately. No local install, environment management, driver setup, or paid acceleration tier to unlock the fast path. If your browser supports WebGPU (Chrome, Edge, Safari 18+, Firefox 139+), RunMat routes work to your local GPU automatically. If not, it falls back to CPU execution and still runs.

You still get the things a real working environment needs: a built-in editor with linting and autocomplete, interactive 2D and 3D plotting, and local execution on your own machine.

Browsers do still impose limits. On very large workloads, peak throughput will be lower than the native CLI because browsers throttle GPU access and operate inside a sandbox. An upcoming native desktop application removes that tradeoff: the same environment and project model, but with full use of your machine's hardware and native-performance I/O.

## RunMat Cloud: from ephemeral work to durable engineering state

The default pattern when doing an engineering computation is usually: run something, get a result, maybe save a few outputs, maybe keep a few values as csv's or screenshots, and hope enough context survives to revisit it later. The work is real, but the state around it is fragile.

RunMat Cloud changes that. Code, datasets, run history, workspace variables, figures, and snapshots become durable project state tied to a shared project instead of one machine, one local directory, or one person's memory.

The hard part of serious numerical work is often not producing one result once. It is keeping the result, revisiting it later, comparing it to the next run, handing it to someone else, and preserving enough context that nobody has to reconstruct what happened from screenshots, copied files, or memory.

RunMat Cloud is designed for that reality. You can point to a specific run, restore a project to any prior state, and let a dataset evolve without forcing a full rewrite every time it changes.

<video
  autoPlay
  controls
  loop
  muted
  playsInline
  preload="metadata"
  style={{ width: "100%", maxWidth: "520px", aspectRatio: "1 / 1", margin: "0 auto", display: "block" }}
>
  <source src="https://web.runmatstatic.com/video/runmat-run-history.mp4" type="video/mp4" />
  Your browser does not support the video tag.
</video>



## Why this matters for large datasets

This becomes especially important once outputs stop fitting comfortably in one engineer's working memory or one machine's local workflow.

If each experiment produces 10 GB or 100 GB of data, a team cannot realistically work by fully materializing every version, keeping full copies on individual laptops, and duplicating the entire dataset every time one region changes. That is not just expensive — in many workflows it makes persistence and comparison practically impossible.

RunMat Cloud avoids that trap.

The RunMat filesystem is built around sharded datasets and manifest-based versioning. That means large artifacts do not have to be treated like monolithic blobs that must be loaded, copied, and rewritten in full every time you want to save, version, restore, or compare them.

Instead:

* the system can materialize the subset you need,
* rewrite only the touched regions,
* preserve history without duplicating the entire dataset on every change,
* and keep long experiment sequences economical enough to actually retain.

A single large run is one problem. A sequence of large runs is the real workflow: baseline, experiment, comparison, rollback, review, handoff.

The useful question is not "can we store one 100 GB result?" It is "can we keep the chain of twenty of them without turning history into a storage bill or a manual archive process?"

Here is the difference in concrete terms:

| Scenario | Naive full-copy storage | Sharded + manifest versioning |
| -- | -- | -- |
| Baseline dataset | 100 GB | 100 GB |
| 10 follow-on runs, each changing 5 GB | 1,000 GB additional | 50 GB additional + small manifests |
| Total after 11 versions | 1.1 TB | ~150 GB + small manifests |

The exact numbers depend on how much changes between runs, but the point is general: if most of the dataset is unchanged, most of the storage should not be duplicated. That is what makes long experiment chains practical to keep instead of constantly pruning, exporting, or pretending history does not matter.

If you want the deeper storage details (sharding, manifests, and historical restore for large numerical artifacts), we break it down in [From Ad-Hoc Checkpoints to Reliable Large Data Persistence](/blog/ad-hoc-checkpoints-to-large-data-persistence).

## Review without rerun

Without durable project state, review and debugging usually degrade into some combination of:

* rerun it and see if it happens again,
* send me the file,
* send me a screenshot,
* I think I was looking at a different version.

That works for small scripts. It breaks down for expensive numerical work.

With RunMat Cloud, a teammate does not need to ask you to rerun the job just to inspect the output. They can open the historical run directly, inspect the same variables, the same figures, and the same project state you saw. They do not need to reconstruct context from exported checkpoint files or hope that rerunning today reproduces what happened yesterday.

That changes review, handoff, and debugging in the same direction.

Reviews are exact — you inspect the actual run, not a summary of it. Baselines persist across sessions instead of living in someone's local directory. And the full history of a project is there to browse, not something you reconstruct after the fact.

For an individual, this means clear history and state inspection. For a team, it means calculation reviews become practical, experiment sequences become feasible to retain and revert, and historical results become [directly inspectable instead of expensive to recreate](/blog/restoring-historical-run-state-scientific-numerical-calculations).




## Snapshots, baselines, and project history

RunMat Cloud also gives teams a natural way to mark important states in a project.

Before a risky migration, before a model change, before a test campaign, or at a milestone, you can snapshot the project. That snapshot gives you a durable project state for rollback, review, comparison, or audit without requiring a full manual export or duplicate copy of every large artifact.

That matters because large numerical work is not just about code. It is about code + datasets + outputs + interpretation over time.

A useful history system for this kind of work has to preserve the project as a whole, not just the script that produced one result.

Snapshots, historical runs, and shared access are what make this practical — the project state is preserved, browseable, and available to the whole team. If your team does not use git today, this is [version control designed for how engineers actually work](/blog/version-control-for-engineers-who-dont-use-git).

<video
  autoPlay
  controls
  loop
  muted
  playsInline
  preload="metadata"
  style={{ width: "100%", maxWidth: "520px", aspectRatio: "1 / 1", margin: "0 auto", display: "block" }}
>
  <source src="https://web.runmatstatic.com/video/runmat-versioning.mp4" type="video/mp4" />
  Your browser does not support the video tag.
</video>

## What else shipped into the sandbox and cloud

Between the Accelerate launch in November and now, we also shipped:

* 3D plotting with interactive camera controls,
* async execution so long-running computations do not block the UI,
* shape inference for better autocomplete and earlier error detection,
* editor linting,
* clear compiler error and stack traces, logging and tracing for diagnostics.

## Plans and pricing

RunMat Cloud launches with three tiers:

**Hobby (free)** — 100 MB of storage. Unlimited projects. Sign in and start working.

**Pro ($30/month per user)** — 10 GB of storage. Additional storage available from your account. For individuals and small teams doing real work.

**Team ($100/month per user)** — 100 GB of storage, SSO/SCIM, and admin controls. For organizations that need governance.

For teams in aerospace, defense, or any environment where data cannot leave the building, **RunMat Enterprise** deploys the entire platform on your own infrastructure: same collaboration, same versioning, same AI integration, fully self-hosted inside your air-gapped network. No license server, no phone-home requirement. The open-source runtime is a single binary you carry in on approved media.

We wrote a [detailed post on how RunMat works in air-gapped environments](/blog/mission-critical-math-airgap) for teams that need that deployment model.

The full breakdown is on the [pricing page](/pricing).

## What has not changed

The sandbox is still free, anonymous, and instant. The CLI is still available for local execution and CI/CD. The fusion engine, automatic CPU/GPU acceleration, and interactive plotting are the same across all three surfaces: CLI, sandbox, and cloud.

Anyone with a browser can use the same runtime as a team with an enterprise deployment.

One runtime, three ways to use it, all of them fast.

## Open source

The [full runtime is open source](https://github.com/runmat-org/runmat) and MIT licensed. You can install it from Homebrew (`brew install runmat-org/tap/runmat`), NPM (`npm install runmat`), or crates.io (`cargo install runmat`), and see the source at [github.com/runmat-org/runmat](https://github.com/runmat-org/runmat).

The NPM package ships the complete runtime — execution, GPU acceleration, and plotting — so you can embed RunMat inside your own web tools. There is also a Jupyter kernel if notebooks are your workflow.

Everything that powers the sandbox and RunMat Cloud starts in this repo.

## What is next

The native desktop application mentioned above is in active development, alongside a few other things we think you'll love. That desktop app will unlock the full use of your machine's hardware, native-performance local I/O, and no browser-imposed GPU limits.

Each step continues the same work: more math per hour, for more engineers.

For now: open [runmat.com/sandbox](https://runmat.com/sandbox), sign in, and save your first project. We're excited to hear your feedback.

## FAQ

**Does RunMat Cloud run on my GPU?**
Yes. The browser uses WebGPU to access your local GPU for acceleration. All computation happens locally in your browser. RunMat Cloud stores project files and run metadata for persistence, but your math is not executed on our infrastructure.

**How is this different from MATLAB Online?**
MATLAB Online runs your code on MathWorks' servers and requires a paid license. RunMat executes computation locally in your browser via WebAssembly, uses your local GPU for acceleration when available, and requires no account to start. RunMat Cloud adds persistence, versioning, run history, and collaboration without moving the math execution itself onto our servers. We wrote a [full comparison](/matlab-online) for those evaluating the switch.

**How does RunMat Cloud make large datasets practical?**
Because it does not require full-file rewrites or whole-project duplication to preserve history. Large datasets are handled through sharded storage and manifest-based versioning, which makes it practical to keep long experiment sequences, restore historical state, and inspect results without treating every checkpoint like a brand-new full copy.

**Can I use RunMat Cloud for large simulations?**
Yes. The storage model is designed so large numerical outputs can be retained, versioned, and revisited efficiently. The exact limits depend on your plan, but the core idea is the same across tiers: preserve useful history without forcing whole-dataset duplication every time a result changes.

**What browsers support WebGPU?**
Chrome, Edge, Safari 18+, and Firefox 139+ all support WebGPU. If WebGPU is not available, the runtime falls back to CPU execution in the browser.

**Can I restore a run someone else did?**
Yes. In a shared project, historical runs, snapshots, and project state are not tied to one machine. Teammates can inspect prior results directly instead of asking the original author to rerun the work.
