import type { Metadata } from "next";
import React from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { SandboxCta } from "@/components/SandboxCta";
import { FAQAccordion, type FAQItem } from "@/components/FAQAccordion";
import { TryInBrowserButton } from "@/components/TryInBrowserButton";
import LazyVideo from "@/components/LazyVideo";
import {
  Cpu,
  FlaskConical,
  FileDiff,
  Wand2,
  FolderSearch,
  Eye,
  FileCode2,
  Sparkles,
  ArrowRight,
  KeyRound,
  Gift,
  Bug,
  Lightbulb,
  ScanLine,
  Undo2,
  History,
  LineChart,
  Monitor,
} from "lucide-react";

const heroVideoSrc = "https://web.runmatstatic.com/video/3D-wave-surface-runmat.mp4";
const heroPosterSrc = "https://web.runmatstatic.com/video/posters/3D-wave-surface-runmat.webp";
const agentDiffImageSrc = "https://web.runmatstatic.com/runmat-agent-diff-rain.webp";

// Open question per plan: /sandbox does not yet handle a `?prompt=` query.
// Cards link to plain /sandbox and display the prompt so users can copy it.
// When the query handler ships, switch href to `/sandbox?prompt=encodeURIComponent(prompt)`.
const tryPrompts: { id: string; title: string; prompt: string }[] = [
  {
    id: "sweep",
    title: "Sweep a parameter range",
    prompt:
      "Plot temperature decay for a coffee cooling in a 20°C room. Try 5 different cup materials and overlay them.",
  },
  {
    id: "vibration",
    title: "Animate a plate vibration",
    prompt:
      "Animate a clamped square-plate vibration with two strikes and show the combined wave pattern.",
  },
  {
    id: "fit",
    title: "Fit a model to lab data",
    prompt:
      "Load measurements.csv. Try a linear, exponential, and power-law fit. Plot residuals for each and tell me which fits best.",
  },
  {
    id: "thermal",
    title: "Model from physics",
    prompt:
      "I'm modelling thermal resistance through a 3-layer wall. Write the code and plot the temperature gradient.",
  },
  {
    id: "efield",
    title: "Visualize an electric field",
    prompt:
      "Plot the electric field and equipotential contours around two point charges of opposite sign. Show both 2D and 3D views.",
  },
  {
    id: "control",
    title: "Tune a PID controller",
    prompt:
      "Design a PID controller for a 1 kg mass on a spring. Plot the step response and find gains that settle in under 0.5 seconds without overshoot.",
  },
];

const faqItems: FAQItem[] = [
  {
    id: "ag-what-is",
    question: "What is RunMat's agent?",
    answer:
      "RunMat's agent is an AI built into the RunMat runtime. It runs your MATLAB-syntax code in RunMat with GPU acceleration when available, reads workspace variables and 3D plot data, and proposes edits as reviewable diffs. The agent lives inside the runtime, so it sees the same workspace and plots you do.",
  },
  // Reviewer tag: legal/product. MATLAB Copilot claim should stay grounded in MathWorks' documented behaviour.
  {
    id: "ag-vs-matlab-copilot",
    question: "How is RunMat's agent different from MATLAB Copilot?",
    answer:
      "MATLAB Copilot is documented around chat, editor assistance, code generation, and explanations. RunMat's agent is built around the runtime loop: variables, figures, diagnostics, and reviewable diffs. RunMat also requires no MATLAB license.",
  },
  {
    id: "ag-vs-chatgpt",
    question: "How is RunMat's agent different from ChatGPT?",
    answer:
      "ChatGPT can generate useful MATLAB-syntax code, but every iteration still depends on moving code, errors, data, and plots between separate tools. RunMat's agent does the iteration loop inside the runtime where the math actually executes.",
  },
  {
    id: "ag-vs-mcp",
    question: "Why not just use ChatGPT with MCP, or Claude with MATLAB?",
    answer:
      "MCP lets you hand a model tools. RunMat gives it an engineering environment for math. The agent reads lints before the code runs, inspects workspace variables after it runs, sees the figures it produces, and lands every change as a reviewable diff. That's the difference between a chat model that generates code and one that runs and verifies it alongside you.",
  },
  {
    id: "ag-runs-matlab",
    question: "Can the agent run MATLAB-syntax code?",
    answer:
      "Yes. The agent uses the same RunMat runtime that powers the sandbox, with GPU acceleration when available. It writes a script, runs it, and reads the result before deciding what to change next.",
  },
  {
    id: "ag-extend-capabilities",
    question: "Can the agent extend what my MATLAB-syntax scripts can do?",
    answer:
      "Yes. The agent can refactor scripts into supported runtime patterns, add analysis steps, build visualizations, and turn one-off code into a repeatable workflow. The change comes back as a diff you can accept or reject.",
  },
  {
    id: "ag-models",
    question: "What models can the agent use?",
    answer:
      "The agent is provider-agnostic. RunMat's managed route works with supported model providers, and the Enterprise tier supports bringing your own API key.",
    answerContent: (
      <>
        The agent is provider-agnostic. RunMat&apos;s managed route works with supported model providers, and the Enterprise tier supports bringing your own API key. See <Link href="/pricing" className="underline hover:text-foreground">pricing</Link> for tier details.
      </>
    ),
  },
  {
    id: "ag-free",
    question: "Is the agent free to use?",
    answer:
      "Yes, with usage limits. The sandbox includes a small free allowance per session. Signed-in Hobby accounts get a free monthly credit. Paid tiers add more credit and team features.",
    answerContent: (
      <>
        Yes, with usage limits. The sandbox includes a small free allowance per session. Signed-in Hobby accounts get a free monthly credit. Paid tiers add more credit and team features. See <Link href="/pricing" className="underline hover:text-foreground">pricing</Link>.
      </>
    ),
  },
  {
    id: "ag-offline",
    question: "Does the agent work offline?",
    answer:
      "The desktop app runs the full RunMat runtime offline — open code, run scripts, inspect variables, view figures, all without a network. The agent itself talks to your configured model provider, so each prompt round trip still needs internet. In the browser sandbox, the runtime works offline once the page is loaded.",
  },
  {
    id: "ag-desktop",
    question: "Is there a desktop version, or only browser?",
    answer:
      "Both. The browser sandbox is the fastest way to try the agent — no install, no account. The desktop app runs the same runtime on your machine, keeps your project on disk, and works offline. Download links are on the download page.",
    answerContent: (
      <>
        Both. The browser sandbox is the fastest way to try the agent — no install, no account. The desktop app runs the same runtime on your machine, keeps your project on disk, and works offline. <Link href="/download" className="underline hover:text-foreground">Download links</Link> are on the download page.
      </>
    ),
  },
  {
    id: "ag-what-sees",
    question: "What can the agent see when it is running my code?",
    answer:
      "Workspace variables it materializes, the code in your project, snapshots of the figures you produce, and the trace from each run. It can also read RunMat's open-source reference docs. Its tools do not have shell or filesystem access outside the project sandbox; model reasoning runs over the network to your chosen provider.",
  },
  {
    id: "ag-vs-github-copilot",
    question: "How is RunMat different from GitHub Copilot for MATLAB?",
    answer:
      "GitHub Copilot writes code in your editor. RunMat's agent lives in the runtime loop: it writes MATLAB-syntax code, runs it in RunMat with GPU acceleration when available, reads the variables and plot data, and adjusts when the result is not right.",
  },
];

const jsonLd = {
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "Organization",
      "@id": "https://runmat.com/#organization",
      name: "RunMat",
      alternateName: ["RunMat by Dystr", "Dystr"],
      legalName: "Dystr Inc.",
      url: "https://runmat.com",
      logo: {
        "@type": "ImageObject",
        url: "https://runmat.com/runmat-logo.svg",
        caption: "RunMat",
      },
      sameAs: [
        "https://github.com/runmat-org/runmat",
        "https://x.com/runmat_com",
        "https://dystr.com",
      ],
    },
    {
      "@type": "WebPage",
      "@id": "https://runmat.com/matlab-ai-agent#webpage",
      url: "https://runmat.com/matlab-ai-agent",
      name: "MATLAB Copilot Alternative — AI Agent for Engineering Math | RunMat",
      description:
        "RunMat's built-in AI agent runs MATLAB-syntax code on GPU, reads your workspace and figures, and proposes the next iteration. The Copilot alternative engineers run instead of stitching ChatGPT, MCP, and a MATLAB license together.",
      inLanguage: "en",
      datePublished: "2026-05-18T00:00:00Z",
      dateModified: "2026-05-18T00:00:00Z",
      breadcrumb: { "@id": "https://runmat.com/matlab-ai-agent#breadcrumb" },
      author: { "@id": "https://runmat.com/#organization" },
      publisher: { "@id": "https://runmat.com/#organization" },
      video: { "@id": "https://runmat.com/matlab-ai-agent#hero-video" },
      mainEntity: [
        { "@id": "https://runmat.com/matlab-ai-agent#faq" },
        { "@id": "https://runmat.com/matlab-ai-agent#software" },
      ],
    },
    {
      "@type": "VideoObject",
      "@id": "https://runmat.com/matlab-ai-agent#hero-video",
      name: "RunMat AI agent for MATLAB — writes and runs code in the browser",
      description:
        "RunMat's built-in AI agent writes MATLAB-syntax code, runs it on GPU, and reads the figures it produces. A MATLAB Copilot alternative that works in the browser with no license required.",
      thumbnailUrl: heroPosterSrc,
      contentUrl: heroVideoSrc,
      uploadDate: "2026-05-14T00:00:00Z",
      duration: "PT47S",
    },
    {
      "@type": "BreadcrumbList",
      "@id": "https://runmat.com/matlab-ai-agent#breadcrumb",
      itemListElement: [
        { "@type": "ListItem", position: 1, name: "Home", item: "https://runmat.com" },
        {
          "@type": "ListItem",
          position: 2,
          name: "Agent",
          item: "https://runmat.com/matlab-ai-agent",
        },
      ],
    },
    {
      "@type": "SoftwareApplication",
      "@id": "https://runmat.com/matlab-ai-agent#software",
      name: "RunMat Agent",
      description:
        "AI agent built into the RunMat runtime. Runs MATLAB-syntax code on the GPU, reads lints and editor diagnostics before running, materializes workspace variables, captures figure snapshots, and lands every change as a reviewable diff. Model-agnostic. Runs in the browser or as a desktop app.",
      applicationCategory: "DeveloperApplication",
      applicationSubCategory: "EngineeringApplication",
      operatingSystem: ["Browser", "Windows", "macOS", "Linux"],
      featureList: [
        "Runs MATLAB-syntax code in the RunMat runtime (GPU-accelerated via WebGPU/wgpu when available)",
        "Reads lints, shape mismatches, and editor diagnostics before re-running",
        "Materializes workspace variables for inspection",
        "Captures snapshots of figures via the runtime's figure monitor",
        "Extends MATLAB-syntax scripts into repeatable analysis workflows",
        "Reads, edits, and searches files across the project",
        "Reversible edits: every change lands as a reviewable diff with one-click revert",
        "Automatic file history and snapshots without git setup",
        "Model-agnostic: works with OpenAI, Anthropic, and Gemini via RunMat's managed model route",
        "Runs in the browser sandbox or as a desktop app",
      ],
      offers: { "@type": "Offer", price: "0", priceCurrency: "USD" },
      url: "https://runmat.com/sandbox",
      author: { "@id": "https://runmat.com/#organization" },
      publisher: { "@id": "https://runmat.com/#organization" },
      mainEntityOfPage: { "@id": "https://runmat.com/matlab-ai-agent#webpage" },
    },
    {
      "@type": "FAQPage",
      "@id": "https://runmat.com/matlab-ai-agent#faq",
      mainEntityOfPage: { "@id": "https://runmat.com/matlab-ai-agent#webpage" },
      mainEntity: faqItems.map(item => ({
        "@type": "Question",
        name: item.question,
        acceptedAnswer: {
          "@type": "Answer",
          text: item.answer,
        },
      })),
    },
  ],
};

export const metadata: Metadata = {
  title: "MATLAB Copilot Alternative — AI Agent for Engineering Math | RunMat",
  description:
    "RunMat's built-in AI agent runs MATLAB-syntax code with GPU acceleration when available, reads your workspace and figures, and proposes the next iteration. The Copilot alternative engineers run instead of stitching ChatGPT, MCP, and a MATLAB license together.",
  alternates: { canonical: "https://runmat.com/matlab-ai-agent" },
  keywords: [
    "matlab copilot",
    "matlab copilot alternative",
    "ai for matlab",
    "matlab ai",
    "matlab ai agent",
    "matlab ai assistant",
    "ai agent for engineering math",
    "chatgpt for matlab",
    "claude for matlab",
    "matlab chatgpt",
    "ai matlab code",
    "ai that runs matlab code",
    "matlab mcp",
    "mcp for matlab",
    "rewrite matlab code with ai",
    "runmat agent",
    "ai for engineering math",
    "matlab debugging ai",
    "ai coding for engineers",
    "github copilot for matlab",
  ],
  openGraph: {
    title: "MATLAB Copilot Alternative — AI Agent for Engineering Math | RunMat",
    description:
      "RunMat's built-in AI agent runs MATLAB-syntax code with GPU acceleration when available, reads your workspace and figures, and proposes the next iteration. Run it instead of stitching ChatGPT, MCP, and a MATLAB license together.",
    url: "/matlab-ai-agent",
    siteName: "RunMat",
    type: "website",
    videos: [
      {
        url: heroVideoSrc,
        type: "video/mp4",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "MATLAB Copilot Alternative — AI Agent for Engineering Math | RunMat",
    description:
      "RunMat's built-in AI agent runs MATLAB-syntax code with GPU acceleration when available, reads your workspace and figures, and proposes the next iteration. Run it instead of stitching ChatGPT, MCP, and a MATLAB license together.",
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
};

export default function AgentPage() {
  return (
    <div className="min-h-screen bg-background">
      <link
        rel="preload"
        as="image"
        href={heroPosterSrc}
        fetchPriority="high"
      />
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify(jsonLd).replace(/<\//g, "<\\/"),
        }}
      />

      {/* 1. Hero */}
      <section className="w-full py-16 md:py-24 lg:py-32" id="hero">
        <div className="container mx-auto px-4 md:px-6 lg:px-8">
          <div className="grid grid-cols-1 gap-10 lg:grid-cols-2 lg:items-center">
            <div className="flex flex-col space-y-6 text-left items-start">
              <div className="mb-2 p-0 text-sm font-semibold uppercase tracking-wider text-foreground">
                AI agent for MATLAB
              </div>
              <h1 className="font-bold text-left leading-tight tracking-tight text-3xl sm:text-4xl md:text-5xl">
                Write high-performance MATLAB code using AI agents.
              </h1>
              <p className="max-w-[42rem] leading-relaxed text-foreground text-[0.938rem]">
                Start from an idea, a model, or an existing MATLAB script. Work with the agent to run sweeps, compare designs, inspect results, and apply reviewable diffs as you iterate through days of engineering alternatives in one session.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 w-full sm:w-auto">
                <Button
                  size="lg"
                  asChild
                  className="h-12 px-8 text-base font-semibold rounded-none bg-[hsl(var(--brand))] text-white hover:bg-[hsl(var(--brand))]/90 border-0 shadow-none"
                >
                  <Link
                    href="/sandbox"
                    data-ph-capture-attribute-destination="sandbox"
                    data-ph-capture-attribute-source="agent-page-hero"
                    data-ph-capture-attribute-cta="open-sandbox"
                  >
                    Open sandbox
                  </Link>
                </Button>
                <Button
                  variant="outline"
                  size="lg"
                  asChild
                  className="h-12 px-8 text-base rounded-none bg-card border-border text-foreground"
                >
                  <Link
                    href="/download"
                    data-ph-capture-attribute-destination="download"
                    data-ph-capture-attribute-source="agent-page-hero"
                    data-ph-capture-attribute-cta="download"
                  >
                    Download
                  </Link>
                </Button>
              </div>
            </div>
            <div className="rounded-xl border border-border bg-card p-2 elevated-panel">
              <Link
                href="/sandbox"
                className="block rounded-lg overflow-hidden focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                data-ph-capture-attribute-destination="sandbox"
                data-ph-capture-attribute-source="agent-page-hero-video"
                data-ph-capture-attribute-cta="try-runmat-agent"
                aria-label="Open the RunMat sandbox"
              >
                <LazyVideo
                  className="w-full h-auto rounded-lg"
                  muted
                  loop
                  playsInline
                  initialPosterVariant="poster"
                  poster={heroPosterSrc}
                  aria-label="RunMat 3D wave surface simulation"
                >
                  <source src={heroVideoSrc} type="video/mp4" />
                </LazyVideo>
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* 4. Use it to */}
      <section className="w-full py-16 md:py-24 lg:py-32">
        <div className="container mx-auto px-4 md:px-6 lg:px-8">
          <div className="mx-auto max-w-[58rem] text-center mb-12">
            <h2 className="font-bold text-3xl leading-[1.1] sm:text-3xl md:text-5xl text-foreground">
              Describe the scenario. Compare the outcomes.
            </h2>
          </div>
          <div className="mx-auto max-w-5xl grid gap-6 md:grid-cols-2">
            <div className="rounded-lg border border-border bg-muted/40 p-8 flex flex-col">
              <h3 className="text-xl md:text-2xl font-semibold text-foreground mb-3 leading-tight">
                Build a 3D simulation from a paragraph of physics
              </h3>
              <p className="text-[0.938rem] text-foreground leading-relaxed mb-5 flex-1">
                Describe the system. The agent can draft the script, run it, and help refine the figure. Sandbox sessions are full of wave propagation, electric fields, and thermal diffusion plots.
              </p>
              <div className="rounded-md border border-border/60 bg-foreground/5 px-4 py-3 font-mono text-[0.8125rem] text-foreground/80 leading-relaxed">
                <p>
                  <span className="select-none text-foreground/40 mr-2">›</span>
                  Visualize the propagation of two interfering waves on a 1m × 1m membrane over 5 seconds.
                </p>
                <div className="mt-3">
                  <TryInBrowserButton
                    code="% Visualize the propagation of two interfering waves on a 1m × 1m membrane over 5 seconds."
                    agentPrompt="Visualize the propagation of two interfering waves on a 1m × 1m membrane over 5 seconds."
                    source="agent-page-use-case-wave"
                    exampleId="wave-interference"
                    size="sm"
                    className="font-sans"
                  >
                    Run prompt
                  </TryInBrowserButton>
                </div>
              </div>
            </div>
            <div className="rounded-lg border border-border bg-muted/40 p-8 flex flex-col">
              <h3 className="text-xl md:text-2xl font-semibold text-foreground mb-3 leading-tight">
                Sweep a parameter and find the boundary
              </h3>
              <p className="text-[0.938rem] text-foreground leading-relaxed mb-5 flex-1">
                Name the parameter and the range. The agent can run the sweep and overlay the traces, so the transition point becomes visible.
              </p>
              <div className="rounded-md border border-border/60 bg-foreground/5 px-4 py-3 font-mono text-[0.8125rem] text-foreground/80 leading-relaxed">
                <p>
                  <span className="select-none text-foreground/40 mr-2">›</span>
                  Sweep the damping coefficient from 0.01 to 1.0 in 20 steps and overlay the step responses.
                </p>
                <div className="mt-3">
                  <TryInBrowserButton
                    code="% Sweep the damping coefficient from 0.01 to 1.0 in 20 steps and overlay the step responses."
                    agentPrompt="Sweep the damping coefficient from 0.01 to 1.0 in 20 steps and overlay the step responses."
                    source="agent-page-use-case-damping"
                    exampleId="damping-sweep"
                    size="sm"
                    className="font-sans"
                  >
                    Run prompt
                  </TryInBrowserButton>
                </div>
              </div>
            </div>
            <div className="rounded-lg border border-border bg-muted/40 p-8 flex flex-col">
              <h3 className="text-xl md:text-2xl font-semibold text-foreground mb-3 leading-tight">
                Compare two designs and pick the winner
              </h3>
              <p className="text-[0.938rem] text-foreground leading-relaxed mb-5 flex-1">
                Describe both designs and choose the metric. The agent can run the comparison and show the tradeoff.
              </p>
              <div className="rounded-md border border-border/60 bg-foreground/5 px-4 py-3 font-mono text-[0.8125rem] text-foreground/80 leading-relaxed">
                <p>
                  <span className="select-none text-foreground/40 mr-2">›</span>
                  Compare a 2nd-order Butterworth and a 4th-order Bessel filter at the same cutoff. Which has less phase distortion?
                </p>
                <div className="mt-3">
                  <TryInBrowserButton
                    code="% Compare a 2nd-order Butterworth and a 4th-order Bessel filter at the same cutoff. Which has less phase distortion?"
                    agentPrompt="Compare a 2nd-order Butterworth and a 4th-order Bessel filter at the same cutoff. Which has less phase distortion?"
                    source="agent-page-use-case-filter"
                    exampleId="filter-comparison"
                    size="sm"
                    className="font-sans"
                  >
                    Run prompt
                  </TryInBrowserButton>
                </div>
              </div>
            </div>
            <div className="rounded-lg border border-border bg-muted/40 p-8 flex flex-col">
              <h3 className="text-xl md:text-2xl font-semibold text-foreground mb-3 leading-tight">
                Explore design ideas and find the best performer
              </h3>
              <p className="text-[0.938rem] text-foreground leading-relaxed mb-5 flex-1">
                List the constraints and the metric that matters. The agent can test parameter sets, compare the results, and return the code behind the strongest candidate.
              </p>
              <div className="rounded-md border border-border/60 bg-foreground/5 px-4 py-3 font-mono text-[0.8125rem] text-foreground/80 leading-relaxed">
                <p>
                  <span className="select-none text-foreground/40 mr-2">›</span>
                  Use a simple beam-stress formula to compare steel beam thicknesses for a 1 kN/m load over a 2 m span. Plot stress versus weight and return the code behind the best option.
                </p>
                <div className="mt-3">
                  <TryInBrowserButton
                    code="% Use a simple beam-stress formula to compare steel beam thicknesses for a 1 kN/m load over a 2 m span. Plot stress versus weight and return the code behind the best option."
                    agentPrompt="Use a simple beam-stress formula to compare steel beam thicknesses for a 1 kN/m load over a 2 m span. Plot stress versus weight and return the code behind the best option."
                    source="agent-page-use-case-beam"
                    exampleId="beam-stress"
                    size="sm"
                    className="font-sans"
                  >
                    Run prompt
                  </TryInBrowserButton>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 5. What the agent does */}
      <section className="w-full py-16 md:py-24 lg:py-32">
        <div className="container mx-auto px-4 md:px-6 lg:px-8">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-8 text-center mb-12">
            <h2 className="font-bold text-3xl leading-[1.1] sm:text-3xl md:text-5xl text-foreground">
              An agent that works with your full project context.
            </h2>
          </div>
          <div className="mx-auto mt-12 max-w-5xl">
            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
              <div className="rounded-lg border border-border bg-muted/40 p-6">
                <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-foreground/10 text-foreground mb-3">
                  <Eye className="h-5 w-5" />
                </span>
                <p className="text-base font-medium text-foreground">Reads visual results</p>
                <p className="text-[0.938rem] text-foreground mt-1">
                  Plot and figure snapshots give the agent visual context, so it can help reason about the output, not only the code that produced it.
                </p>
              </div>
              <div className="rounded-lg border border-border bg-muted/40 p-6">
                <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-foreground/10 text-foreground mb-3">
                  <FlaskConical className="h-5 w-5" />
                </span>
                <p className="text-base font-medium text-foreground">Inspects live variables</p>
                <p className="text-[0.938rem] text-foreground mt-1">
                  Pull values from the workspace when you need them. The agent can use the same previews to debug shapes, ranges, and intermediate results.
                </p>
              </div>
              <div className="rounded-lg border border-border bg-muted/40 p-6">
                <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-foreground/10 text-foreground mb-3">
                  <Cpu className="h-5 w-5" />
                </span>
                <p className="text-base font-medium text-foreground">Keeps context in one place</p>
                <p className="text-[0.938rem] text-foreground mt-1">
                  You and the agent share the same files, variables, figures, and run history, so each suggestion is grounded in the current project.
                </p>
              </div>
              <div className="rounded-lg border border-border bg-muted/40 p-6">
                <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-foreground/10 text-foreground mb-3">
                  <ScanLine className="h-5 w-5" />
                </span>
                <p className="text-base font-medium text-foreground">Uses static and runtime feedback</p>
                <p className="text-[0.938rem] text-foreground mt-1">
                  RunMat can surface lints, shape issues, and semantic feedback before execution, then use errors, variables, figures, and runtime output after the script runs.
                </p>
              </div>
              <div className="rounded-lg border border-border bg-muted/40 p-6">
                <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-foreground/10 text-foreground mb-3">
                  <Wand2 className="h-5 w-5" />
                </span>
                <p className="text-base font-medium text-foreground">Builds from your script</p>
                <p className="text-[0.938rem] text-foreground mt-1">
                  Ask for a sweep, comparison, plot, or report. The agent proposes the next step as code you can review and edit.
                </p>
              </div>
              <div className="rounded-lg border border-border bg-muted/40 p-6">
                <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-foreground/10 text-foreground mb-3">
                  <FolderSearch className="h-5 w-5" />
                </span>
                <p className="text-base font-medium text-foreground">Works across the project</p>
                <p className="text-[0.938rem] text-foreground mt-1">
                  It can read and edit project files, not just the tab in front of you. Larger changes still land as reviewable diffs.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 6. Start from anywhere */}
      <section className="w-full py-16 md:py-24 lg:py-32">
        <div className="container mx-auto px-4 md:px-6 lg:px-8">
          <div className="mx-auto max-w-[58rem] text-center mb-12">
            <h2 className="font-bold text-3xl leading-[1.1] sm:text-3xl md:text-5xl text-foreground">
              Start from anywhere
            </h2>
          </div>
          <div className="mx-auto max-w-5xl grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="rounded-lg border border-border bg-card p-6 flex flex-col">
              <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-foreground/10 text-foreground mb-3">
                <FileCode2 className="h-5 w-5" />
              </span>
              <h3 className="text-lg font-semibold text-foreground">An existing MATLAB script</h3>
              <p className="text-[0.938rem] text-foreground mt-1 flex-1">
                Open a script you already use. Ask the agent to explain it, debug a run, add a plot, or turn a single run into a parameter sweep.
              </p>
              <Link
                href="/sandbox"
                className="text-sm text-[hsl(var(--brand))] hover:text-[hsl(var(--brand))]/80 mt-4 inline-flex items-center gap-1"
                data-ph-capture-attribute-destination="sandbox"
                data-ph-capture-attribute-source="agent-page-start-matlab"
                data-ph-capture-attribute-cta="try-from-matlab"
              >
                Try it <ArrowRight className="h-3.5 w-3.5" />
              </Link>
            </div>
            <div className="rounded-lg border border-border bg-card p-6 flex flex-col">
              <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-foreground/10 text-foreground mb-3">
                <LineChart className="h-5 w-5" />
              </span>
              <h3 className="text-lg font-semibold text-foreground">Raw data</h3>
              <p className="text-[0.938rem] text-foreground mt-1 flex-1">
                Add a CSV or table. Ask the agent to clean it, plot it, fit a model, or compare residuals.
              </p>
              <Link
                href="/sandbox"
                className="text-sm text-[hsl(var(--brand))] hover:text-[hsl(var(--brand))]/80 mt-4 inline-flex items-center gap-1"
                data-ph-capture-attribute-destination="sandbox"
                data-ph-capture-attribute-source="agent-page-start-csv"
                data-ph-capture-attribute-cta="try-from-csv"
              >
                Try it <ArrowRight className="h-3.5 w-3.5" />
              </Link>
            </div>
            <div className="rounded-lg border border-border bg-card p-6 flex flex-col">
              <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-foreground/10 text-foreground mb-3">
                <Bug className="h-5 w-5" />
              </span>
              <h3 className="text-lg font-semibold text-foreground">A script that&apos;s broken</h3>
              <p className="text-[0.938rem] text-foreground mt-1 flex-1">
                Run it in the workspace and ask the agent to debug it. It can use the error, variables, and output to propose a fix.
              </p>
              <Link
                href="/sandbox"
                className="text-sm text-[hsl(var(--brand))] hover:text-[hsl(var(--brand))]/80 mt-4 inline-flex items-center gap-1"
                data-ph-capture-attribute-destination="sandbox"
                data-ph-capture-attribute-source="agent-page-start-broken"
                data-ph-capture-attribute-cta="try-from-broken"
              >
                Try it <ArrowRight className="h-3.5 w-3.5" />
              </Link>
            </div>
            <div className="rounded-lg border border-border bg-card p-6 flex flex-col">
              <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-foreground/10 text-foreground mb-3">
                <Lightbulb className="h-5 w-5" />
              </span>
              <h3 className="text-lg font-semibold text-foreground">An engineering problem</h3>
              <p className="text-[0.938rem] text-foreground mt-1 flex-1">
                Describe the system, constraints, and result you need. The agent can help write the first script.
              </p>
              <Link
                href="/sandbox"
                className="text-sm text-[hsl(var(--brand))] hover:text-[hsl(var(--brand))]/80 mt-4 inline-flex items-center gap-1"
                data-ph-capture-attribute-destination="sandbox"
                data-ph-capture-attribute-source="agent-page-start-idea"
                data-ph-capture-attribute-cta="try-from-idea"
              >
                Try it <ArrowRight className="h-3.5 w-3.5" />
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* 7. Every change is reviewable */}
      <section className="w-full py-16 md:py-24 lg:py-32" id="version-control">
        <div className="container mx-auto px-4 md:px-6 lg:px-8">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-6 text-center mb-12">
            <h2 className="font-bold text-3xl leading-[1.1] sm:text-3xl md:text-5xl text-foreground">
              Run more experiments without losing project state.
            </h2>
            <p className="max-w-[42rem] leading-relaxed text-[0.938rem] text-foreground">
              The agent can help try more branches in a session. Every change lands as a diff you can read, accept, or revert, so the project stays legible as the exploration widens.
            </p>
          </div>
          <div className="mx-auto max-w-5xl grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="rounded-lg border border-border bg-card p-6 flex flex-col">
              <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-foreground/10 text-foreground mb-3">
                <FileDiff className="h-5 w-5" />
              </span>
              <h3 className="text-lg font-semibold text-foreground">Reviewable diffs</h3>
              <p className="text-[0.938rem] text-foreground mt-1 flex-1">
                See exactly what the agent changed, line by line, before it becomes part of the project.
              </p>
            </div>
            <div className="rounded-lg border border-border bg-card p-6 flex flex-col">
              <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-foreground/10 text-foreground mb-3">
                <Undo2 className="h-5 w-5" />
              </span>
              <h3 className="text-lg font-semibold text-foreground">One-click revert</h3>
              <p className="text-[0.938rem] text-foreground mt-1 flex-1">
                Reject a change to roll it back. Keep the promising experiments and discard the dead ends.
              </p>
            </div>
            <div className="rounded-lg border border-border bg-card p-6 flex flex-col">
              <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-foreground/10 text-foreground mb-3">
                <History className="h-5 w-5" />
              </span>
              <h3 className="text-lg font-semibold text-foreground">History is automatic</h3>
              <p className="text-[0.938rem] text-foreground mt-1 flex-1">
                Files snapshot as the agent works. Roll back to an earlier state without setting up git.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* 7. What an iteration looks like */}
      <section className="w-full py-16 md:py-24 lg:py-32">
        <div className="container mx-auto px-4 md:px-6 lg:px-8">
          <div className="mx-auto max-w-[58rem] text-center mb-12">
            <h2 className="font-bold text-3xl leading-[1.1] sm:text-3xl md:text-5xl text-foreground">
              Designed for agent-in-the-loop engineering
            </h2>
            <p className="mx-auto mt-6 max-w-[42rem] leading-relaxed text-[0.938rem] text-foreground">
              Ask RunMat to change a simulation and the agent works inside the runtime: it edits MATLAB-syntax code, runs it, reads plots and variables, then returns the exact diff behind the result. Keep the change, revert it, or continue iterating without losing the path that got you there.
            </p>
          </div>
          <div className="mx-auto max-w-5xl">
            <div className="overflow-hidden rounded-lg border border-border elevated-panel">
              <Link
                href="/sandbox"
                className="block focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 rounded-b-none rounded-t-lg overflow-hidden"
                data-ph-capture-attribute-destination="sandbox"
                data-ph-capture-attribute-source="agent-page-agent-diff"
                data-ph-capture-attribute-cta="try-runmat-agent"
                aria-label="Open the RunMat sandbox"
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={agentDiffImageSrc}
                  alt="RunMat agent showing a reviewable code diff beside runtime context"
                  loading="lazy"
                  decoding="async"
                  className="w-full"
                />
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* 8. Prompts to try */}
      <section className="w-full py-16 md:py-24 lg:py-32" id="prompts">
        <div className="container mx-auto px-4 md:px-6 lg:px-8">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-8 text-center mb-12">
            <h2 className="font-bold text-3xl leading-[1.1] sm:text-3xl md:text-5xl text-foreground">
              Runnable prompts for the sandbox
            </h2>
            <p className="max-w-[42rem] leading-relaxed text-[0.938rem] text-foreground">
              These are small, current examples: open one in the sandbox and watch the agent create, run, inspect, and revise.
            </p>
          </div>
          <div className="mx-auto max-w-5xl grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {tryPrompts.map(p => (
              <div key={p.id} className="rounded-lg border border-border bg-card p-6 flex flex-col">
                <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-foreground/10 text-foreground mb-3">
                  <Sparkles className="h-5 w-5" />
                </span>
                <h3 className="text-lg font-semibold text-foreground">{p.title}</h3>
                <p className="text-[0.875rem] text-foreground mt-2 font-mono leading-relaxed flex-1">
                  {p.prompt}
                </p>
                <TryInBrowserButton
                  code={`% ${p.prompt}`}
                  agentPrompt={p.prompt}
                  source={`agent-page-prompt-${p.id}`}
                  exampleId={p.id}
                  size="sm"
                  className="mt-4 w-fit"
                >
                  Run prompt
                </TryInBrowserButton>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* 9. Where other tools stop short */}
      <section className="w-full py-16 md:py-24 lg:py-32">
        <div className="container mx-auto px-4 md:px-6 lg:px-8">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-8 text-center mb-12">
            <h2 className="font-bold text-3xl leading-[1.1] sm:text-3xl md:text-5xl text-foreground">
              Where other tools stop short
            </h2>
            <p className="max-w-[42rem] leading-relaxed text-[0.938rem] text-foreground">
              Most AI tools can suggest code. RunMat gives the agent structured feedback from the engineering loop: static checks before execution, runtime state while it runs, figures and variables after.
            </p>
          </div>
          <div className="mx-auto grid max-w-5xl gap-6 md:grid-cols-2">
            {/* Reviewer tag: legal/product. MATLAB Copilot claims must stay grounded in MathWorks' documented behaviour. */}
            <Card className="border border-border bg-card shadow-sm">
              <CardContent className="p-6 space-y-4">
                <h3 className="text-lg font-semibold text-foreground">MATLAB Copilot</h3>
                <div className="space-y-3 text-sm">
                  <p className="text-muted-foreground"><span className="font-medium text-emerald-300/80">Best at:</span> MATLAB-aware chat, code generation, and explanations inside MATLAB.</p>
                  <p className="text-muted-foreground"><span className="font-medium text-rose-300/80">Missing:</span> RunMat&apos;s reviewable project diffs, open runtime, and no-license workflow.</p>
                </div>
              </CardContent>
            </Card>
            <Card className="border border-border bg-card shadow-sm">
              <CardContent className="p-6 space-y-4">
                <h3 className="text-lg font-semibold text-foreground">ChatGPT, Claude, Gemini</h3>
                <div className="space-y-3 text-sm">
                  <p className="text-muted-foreground"><span className="font-medium text-emerald-300/80">Best at:</span> strong reasoning and useful MATLAB-syntax code suggestions.</p>
                  <p className="text-muted-foreground"><span className="font-medium text-rose-300/80">Missing:</span> running the code, seeing variables and plots, and keeping edits tied to the project.</p>
                </div>
              </CardContent>
            </Card>
            <Card className="border border-border bg-card shadow-sm">
              <CardContent className="p-6 space-y-4">
                <h3 className="text-lg font-semibold text-foreground">Cursor and other AI IDEs</h3>
                <div className="space-y-3 text-sm">
                  <p className="text-muted-foreground"><span className="font-medium text-emerald-300/80">Best at:</span> editing files, refactoring code, and working through software projects.</p>
                  <p className="text-muted-foreground"><span className="font-medium text-rose-300/80">Missing:</span> treating MATLAB-syntax execution, tensors, figures, and runtime output as first-class context.</p>
                </div>
              </CardContent>
            </Card>
            <Card className="border border-border bg-card shadow-sm">
              <CardContent className="p-6 space-y-4">
                <h3 className="text-lg font-semibold text-foreground">Spreadsheet + ChatGPT stitching</h3>
                <div className="space-y-3 text-sm">
                  <p className="text-muted-foreground"><span className="font-medium text-emerald-300/80">Best at:</span> quick models with familiar tools you already have open.</p>
                  <p className="text-muted-foreground"><span className="font-medium text-rose-300/80">Missing:</span> a continuous loop between data, code, plots, runtime output, and reviewable history.</p>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* 9. Open runtime without MATLAB license friction */}
      <section className="w-full py-16 md:py-24 lg:py-32">
        <div className="container mx-auto px-4 md:px-6 lg:px-8">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-8 text-center mb-12">
            <h2 className="font-bold text-3xl leading-[1.1] sm:text-3xl md:text-5xl text-foreground">
              An open runtime without MATLAB license friction.
            </h2>
            <p className="max-w-[42rem] leading-relaxed text-[0.938rem] text-foreground">
              The runtime is open source. The browser sandbox is free to try; the desktop app keeps everything on your machine. Either way, the .m files you already have just run.
            </p>
          </div>
          <div className="mx-auto max-w-5xl grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="rounded-lg border border-border bg-card p-6 flex flex-col">
              <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-foreground/10 text-foreground mb-3">
                <KeyRound className="h-5 w-5" />
              </span>
              <h3 className="text-lg font-semibold text-foreground">No MATLAB license required</h3>
              <p className="text-[0.938rem] text-foreground mt-1 flex-1">
                The runtime is an MIT-licensed reimplementation. It runs MATLAB-syntax code without a paid MathWorks license — for coursework, prototypes, and engineering workflows.
              </p>
            </div>
            <div className="rounded-lg border border-border bg-card p-6 flex flex-col">
              <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-foreground/10 text-foreground mb-3">
                <Monitor className="h-5 w-5" />
              </span>
              <h3 className="text-lg font-semibold text-foreground">Browser or desktop</h3>
              <p className="text-[0.938rem] text-foreground mt-1 flex-1">
                Open the sandbox URL for a zero-install session. Or <Link href="/download" className="underline hover:text-foreground/80">download the desktop app</Link> to keep your project and the runtime on your machine.
              </p>
            </div>
            <div className="rounded-lg border border-border bg-card p-6 flex flex-col">
              <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-foreground/10 text-foreground mb-3">
                <Gift className="h-5 w-5" />
              </span>
              <h3 className="text-lg font-semibold text-foreground">Free to start</h3>
              <p className="text-[0.938rem] text-foreground mt-1 flex-1">
                The sandbox includes a free allowance per session. Signed-in Hobby accounts get a monthly credit on top. <Link href="/pricing" className="underline hover:text-foreground/80">See pricing</Link>.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* 10. FAQ */}
      <section className="w-full py-16 md:py-24 lg:py-32">
        <div className="container mx-auto px-4 md:px-6 lg:px-8">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-8 text-center mb-12">
            <h2 className="font-bold text-3xl leading-[1.1] sm:text-3xl md:text-5xl text-foreground">
              Frequently asked questions
            </h2>
            <p className="max-w-[42rem] leading-relaxed text-[0.938rem] text-foreground">
              Common questions about RunMat&apos;s agent, model choice, and what it can and can&apos;t do.
            </p>
          </div>
          <FAQAccordion items={faqItems} />
        </div>
      </section>

      {/* 11. Final CTA */}
      <section className="w-full py-16 md:py-24 lg:py-32">
        <div className="container mx-auto px-4 md:px-6 lg:px-8">
          <div className="mx-auto max-w-3xl">
            <SandboxCta
              source="agent-page-bottom-cta"
              secondaryLabel="Download desktop app"
              secondaryHref="/download"
            />
          </div>
        </div>
      </section>
    </div>
  );
}
