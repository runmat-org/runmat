---
title: "The FFT Explained Visually: An Interactive Guide to the Fast Fourier Transform"
description: "An interactive guide to FFT windowing, spectral leakage, and frequency resolution. Compare Hann, Hamming, and Blackman-Harris windows in real time, then run fft() at scale in RunMat."
date: "2026-03-13"
readTime: "12 min read"
authors:
  - name: "Fin Watterson"
    url: "https://www.linkedin.com/in/finbarrwatterson/"
slug: "fft-explained"
tags: ["FFT", "signal processing", "RunMat", "scientific computing", "spectral analysis"]
keywords: "FFT explained, FFT window function comparison, Hann vs Hamming vs Blackman-Harris, spectral leakage fix, FFT frequency resolution, how many FFT samples, interactive FFT, RunMat"
excerpt: "An interactive guide to FFT windowing, frequency resolution, and spectral leakage. Compare window functions in real time, then run fft() at scale in RunMat."
ogType: "article"
ogTitle: "The FFT Explained Visually: Interactive Guide"
ogDescription: "Compare Hann, Hamming, and Blackman-Harris windows interactively. Understand spectral leakage, frequency resolution, and when 4,096 samples isn't enough."
twitterCard: "summary_large_image"
twitterTitle: "The FFT Explained Visually: Interactive Guide"
twitterDescription: "Compare Hann, Hamming, and Blackman-Harris windows interactively. Understand spectral leakage, frequency resolution, and when 4,096 samples isn't enough."
canonical: "https://runmat.com/blog/fft-explained"
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
          name: "The FFT Explained Visually"
          item: "https://runmat.com/blog/fft-explained"
    - "@type": "TechArticle"
      "@id": "https://runmat.com/blog/fft-explained#article"
      headline: "The FFT Explained Visually: An Interactive Guide to the Fast Fourier Transform"
      description: "An interactive, visual guide to the Fast Fourier Transform covering signal composition, window functions, and spectral analysis."
      datePublished: "2026-03-13T00:00:00Z"
      author:
        "@type": "Person"
        name: "Fin Watterson"
        url: "https://www.linkedin.com/in/finbarrwatterson/"
      publisher:
        "@id": "https://runmat.com/#organization"
      mainEntityOfPage:
        "@id": "https://runmat.com/blog/fft-explained"
      isPartOf:
        "@id": "https://runmat.com/#website"
---

You have accelerometer data from a gearbox test rig sampled at 48 kHz, and there's a new peak at 847 Hz that wasn't there six months ago. Is it a bearing defect? A resonance excited by a recent housing change? Before you can answer, you need to decide: how many samples to analyze, which window function to apply, and whether your frequency resolution is fine enough to distinguish this tone from the 840 Hz gear-mesh harmonic next to it.

The FFT itself is one function call. The decisions around it -- windowing, sample count, resolution trade-offs -- are where engineers actually spend time and where mistakes cost you. The interactive tool below lets you experiment with those decisions in real time: build a signal, change the window, and watch the spectrum shift.

<FFTVisualizer />

## Choosing a window function

Load the **Two-tone** preset above: two sinusoids at 100 Hz and 112 Hz. Under a Rectangular window, you get two distinct peaks -- the 12 Hz separation is clearly resolved. Switch to Blackman-Harris and they merge into a single broad hump. You traded frequency resolution for 79 dB of additional sidelobe suppression.

That trade-off is quantifiable. Each window has specific main-lobe width and sidelobe characteristics that determine what it can and cannot resolve:

| Window | First sidelobe | Roll-off | Main lobe width | When to use |
|---|---|---|---|---|
| Rectangular | -13 dB | -6 dB/oct | 1 bin | Transients shorter than the window; coherent sampling where the signal is exactly periodic |
| Hann | -31 dB | -18 dB/oct | 2 bins | General-purpose spectral analysis; the default in most commercial vibration analyzers |
| Hamming | -41 dB | -6 dB/oct | 2 bins | Narrowband signals where first-sidelobe suppression matters more than roll-off rate |
| Blackman-Harris | -92 dB | -6 dB/oct | 4 bins | High dynamic range measurement; finding a -60 dB harmonic next to a dominant tone |

Toggle between these in the widget to see the numbers play out. The **Noisy signal** preset makes the sidelobe differences especially visible: under Rectangular, leakage smears energy across the spectrum; under Blackman-Harris, the 100 Hz tone stands cleanly above a much lower noise floor.

### Which window for which domain

Vibration analysis on rotating machinery almost always uses Hann. It's the default in Bruel & Kjaer and National Instruments analyzers for a reason: -31 dB first sidelobe with -18 dB/octave roll-off gives you enough leakage suppression to find bearing tones without blurring closely-spaced gear-mesh harmonics.

EMI compliance testing needs Blackman-Harris or a similar high-dynamic-range window. Regulatory limits are specified in dBuV/m, and you need to detect harmonics 60-80 dB below the fundamental. A Hann window's -31 dB sidelobes would bury those harmonics in leakage from the carrier.

Radar and sonar pulse processing often uses Kaiser windows (adjustable via the beta parameter), but for general spectral analysis of radar IF signals, Hamming's -41 dB first sidelobe with a narrow main lobe is a common starting point.

All four windows in the table above ship as built-in functions in RunMat -- `hann`, `hamming`, `blackman`. In MATLAB, these require the Signal Processing Toolbox, an additional license on top of the base seat.

## Frequency resolution and sample count

The widget runs on 4,096 samples. Whether that's enough depends on what you're trying to resolve.

Frequency resolution is `Δf = fs / N`, where `fs` is the sample rate and `N` is the FFT length. The numbers are concrete:

At **48 kHz** (vibration data from an accelerometer), 4,096 samples gives you 11.7 Hz resolution across 85 ms of signal. Fine for identifying a bearing tone at 847 Hz, but if you need to separate it from an 840 Hz gear-mesh harmonic 7 Hz away, you need sub-Hz resolution -- at least 480,000 samples, covering 10 seconds of data.

At **44.1 kHz** (audio), 4,096 samples gives you 10.8 Hz resolution. Enough to separate musical notes a semitone apart at concert pitch (440 Hz vs 466 Hz is a 26 Hz gap). To resolve vibrato modulation at 1-2 Hz, you need 44,100+ samples -- a full second of audio.

At **1 MHz** (radar IF), 4,096 samples gives you 244 Hz resolution over 4 ms. A target moving at 100 m/s with a 10 GHz carrier produces ~6.67 kHz Doppler shift, easily resolved. But separating two targets with a 10 Hz Doppler difference requires 100,000 samples.

### Zero-padding is not a shortcut

Zero-padding with `fft(x, N)` where `N > length(x)` adds interpolated points between existing frequency bins. The spectrum plot looks smoother, and peak locations become easier to read visually. But you cannot resolve two tones that were unresolvable before -- the information content of the original signal hasn't changed. Genuine resolution improvement requires more real samples, which means longer observation time.

### Where the widget stops

The interactive tool is useful for building intuition about windowing and leakage on synthetic signals. Real spectral analysis involves hundreds of thousands of samples, multichannel sensor arrays, and iterating over parameter sweeps -- varying window type, FFT length, and overlap percentage to find the combination that reveals the feature you're looking for.

## RunMat's FFT at scale

RunMat's `fft` uses the same MATLAB syntax and handles the scale that the browser widget cannot. Here is the exact scenario from the opening paragraph -- an 847 Hz bearing tone sitting 7 Hz from an 840 Hz gear-mesh harmonic, resolved with a Hann window at 48 kHz:

```matlab:runnable
fs = 48000;
t = 0:1/fs:0.1-1/fs;
x = sin(2*pi*847*t) + 0.3*sin(2*pi*840*t);
N = length(x);
w = hann(N)';
Y = fft(x .* w);
f = (0:N-1) * (fs / N);
plot(f(1:N/2), 20*log10(abs(Y(1:N/2))));
xlabel('Frequency (Hz)'); ylabel('Magnitude (dB)');
```

That script runs unmodified in MATLAB and in RunMat. Click **Run in browser** above to try it. The difference is what happens under the hood.

RunMat's Rust-native FFT runs 150-180x faster than GNU Octave on the same input. For GPU-resident arrays, the runtime automatically dispatches to your GPU via native APIs -- Metal on macOS, Vulkan on Linux, DirectX 12 on Windows. No CUDA toolkit installation, no `gpuArray` wrapper calls, no code changes. It works on NVIDIA, AMD, Intel, and Apple Silicon hardware.

The full spectral toolkit ships built-in: `fft`, `ifft`, `fftshift`, `abs`, `angle`, and all the window functions from the comparison table above. In MATLAB, window functions and several spectral utilities require the Signal Processing Toolbox -- roughly $1,000 per seat on top of the base license. In RunMat, they're part of the runtime and always available.

Click **Run in RunMat** in the visualizer above to open a sandbox with equivalent code pre-loaded, or [read the full fft documentation](https://runmat.com/docs/reference/builtins/fft) for syntax details and more examples.
