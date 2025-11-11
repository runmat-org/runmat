# Title Case Audit

This document tracks Title Case consistency across all headers in the website.

## Title Case Rules
- Capitalize major words: nouns, verbs, adjectives, adverbs
- Lowercase minor words: articles (a, an, the), prepositions (in, on, at, for, etc.), conjunctions (and, or, but)
- Always capitalize the first and last word
- Capitalize words that are 4+ letters (some style guides)
- Capitalize technical terms and proper nouns

## Issues Found

### Homepage (`website/app/page.tsx`)

#### ✅ Correct Title Case:
- "The Fastest Runtime for Math" - Correct
- "Full MATLAB Language Semantics" - Correct (MATLAB is proper noun)
- "Faster by Design (Fusion + Residency)" - Correct
- "Slim Core + Packages + IDE LSP" - Correct (acronyms)
- "Portable + Lightweight" - Correct
- "Free and Open Source, Forever" - Correct

#### ❌ Needs Fixing:
- "GPU Accelerated Math" - Should be "GPU-Accelerated Math" (hyphenated compound)
- "Real workloads, Reproducible Results" - Should be "Real Workloads, Reproducible Results" (workloads should be capitalized)

### Docs Page (`website/app/docs/page.tsx`)

#### ✅ Correct Title Case:
- "RunMat Documentation" - Correct
- "Getting Started" - Correct
- "How It Works" - Correct
- "CLI Reference" - Correct
- "Configuration" - Correct
- "Built-in Functions" - Correct
- "Architecture" - Correct
- "Language Coverage" - Correct
- "Roadmap" - Correct
- "License" - Correct
- "Quick Links" - Correct

### Benchmarks Page (`website/app/benchmarks/page.tsx`)

#### ✅ Correct Title Case:
- "RunMat Benchmarks" - Correct
- "Reproduce the Benchmarks" - Correct

### Blog Page (`website/app/blog/page.tsx`)

#### ✅ Correct Title Case:
- "RunMat Blog" - Correct
- "Stay Updated" - Correct

### Download Page (`website/app/download/page.tsx`)

#### ✅ Correct Title Case:
- "Download RunMat" - Correct
- "Next Steps" - Correct
- "Start Coding" - Correct
- "Set up Jupyter Kernel" - Correct (phrasal verb "Set up")
- "Alternative Installation Methods" - Correct
- "Package Managers" - Correct
- "Cargo (Source Build)" - Correct

#### ❌ Needs Fixing:
- "Install from crates.io" - Should be "Install From crates.io" (From should be capitalized)
- "From source (latest)" - Should be "From Source (Latest)" (Source and Latest should be capitalized)

### License Page (`website/app/license/page.tsx`)

#### ✅ Correct Title Case:
- "RunMat License" - Correct
- "Free for Most Uses" - Correct
- "Attribution Required" - Correct
- "Special Rules" - Correct
- "Frequently Asked Questions" - Correct
- "Can I use RunMat for free?" - Correct (question format)
- "What does "attribution required" mean?" - Correct (question format)

### MATLAB Function Reference (`website/app/docs/matlab-function-reference/page.tsx`)

#### ✅ Correct Title Case:
- "MATLAB Function Reference" - Correct

### Builtin Function Reference (`website/app/docs/reference/builtins/page.tsx`)

#### ✅ Correct Title Case:
- "Builtin Function Reference" - Correct (or could be "Built-in Function Reference")

## Summary

**Total Issues Found: 4**
1. ✅ FIXED: "GPU Accelerated Math" → "GPU-Accelerated Math" (in `website/app/page.tsx`)
2. ✅ FIXED: "Real workloads, Reproducible Results" → "Real Workloads, Reproducible Results" (in `website/app/page.tsx`)
3. ✅ FIXED: "Install from crates.io" → "Install From crates.io" (in `website/app/download/page.tsx`)
4. ✅ FIXED: "From source (latest)" → "From Source (Latest)" (in `website/app/download/page.tsx`)

All Title Case issues have been fixed. The website now uses consistent Title Case throughout all headers.

