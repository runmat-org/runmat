---
title: "MATLAB FFT in RunMat: Frequency Analysis with GPU-Accelerated Plotting"
description: "Learn MATLAB FFT online with RunMat. Compute and plot FFT spectra with MATLAB-compatible syntax, including single-sided spectra, windowing, and 2D FFT in the browser."
date: "2026-04-21"
readTime: "18 min read"
authors:
  - name: "Fin Watterson"
    url: "https://www.linkedin.com/in/finbarrwatterson/"
slug: "matlab-fft-guide"
tags: ["MATLAB", "RunMat", "FFT", "signal processing", "GPU", "frequency analysis"]
collections: ["guides"]
keywords: "matlab fft, matlab fft online, fft matlab, how to use fft in matlab, matlab frequency analysis, fft plot matlab, matlab fft example, matlab spectrum analysis, matlab dft, fast fourier transform matlab, RunMat fft, GPU fft"
excerpt: "A hands-on FFT guide with runnable code examples covering single-sided spectra, windowing, 2D FFT, and GPU-accelerated spectral visualization in RunMat."
image: "https://web.runmatstatic.com/blog-images/runmat-fft-guide.webp"
imageAlt: "500 Hz square wave and its FFT showing odd harmonics at 500, 1500, 2500, 3500 Hz, rendered in the RunMat browser sandbox"
ogType: "article"
ogTitle: "MATLAB FFT in RunMat: Frequency Analysis with GPU-Accelerated Plotting"
ogDescription: "Learn MATLAB FFT online in RunMat. Build single-sided spectra, apply windows, and plot 2D FFT results with MATLAB syntax in your browser."
twitterCard: "summary_large_image"
twitterTitle: "MATLAB FFT in RunMat"
twitterDescription: "Compute FFTs with MATLAB syntax, visualize spectra on the GPU, and run everything directly in your browser."
canonical: "https://runmat.com/blog/matlab-fft-guide"
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
          name: "MATLAB FFT Guide"
          item: "https://runmat.com/blog/matlab-fft-guide"

    - "@type": "TechArticle"
      "@id": "https://runmat.com/blog/matlab-fft-guide#article"
      headline: "MATLAB FFT in RunMat: Frequency Analysis with GPU-Accelerated Plotting"
      alternativeHeadline: "How to compute and plot FFT spectra with MATLAB syntax in the browser"
      description: "Learn MATLAB FFT online with RunMat. Compute and plot FFT spectra with MATLAB-compatible syntax, including single-sided spectra, windowing, and 2D FFT in the browser."
      url: "https://runmat.com/blog/matlab-fft-guide"
      mainEntityOfPage: "https://runmat.com/blog/matlab-fft-guide"
      image: "https://web.runmatstatic.com/blog-images/runmat-fft-guide.webp"
      datePublished: "2026-04-21T00:00:00Z"
      dateModified: "2026-04-21T00:00:00Z"
      inLanguage: "en-US"
      keywords: "matlab fft, matlab fft online, fft matlab, matlab frequency analysis, fft plot matlab, RunMat fft"
      proficiencyLevel: "Intermediate"
      hasPart:
        "@id": "https://runmat.com/blog/matlab-fft-guide#faq"
      author:
        - "@type": "Person"
          name: "Fin Watterson"
          url: "https://www.linkedin.com/in/finbarrwatterson/"
      publisher:
        "@id": "https://runmat.com/#organization"
      about:
        - "@type": "SoftwareApplication"
          name: "RunMat"
          applicationCategory: "ScientificApplication"
          operatingSystem: "Browser, Windows, Linux, macOS"
          offers:
            "@type": "Offer"
            price: "0"
            priceCurrency: "USD"
        - "@type": "SoftwareApplication"
          name: "MATLAB"
          applicationCategory: "ScientificApplication"
          operatingSystem: "Windows, Linux, macOS"

    - "@type": "FAQPage"
      "@id": "https://runmat.com/blog/matlab-fft-guide#faq"
      mainEntityOfPage:
        "@id": "https://runmat.com/blog/matlab-fft-guide"
      mainEntity:
        - "@type": "Question"
          name: "How do I compute an FFT in RunMat?"
          acceptedAnswer:
            "@type": "Answer"
            text: "Use Y = fft(x) where x is a time-domain signal vector. The output Y is a complex array of frequency-domain coefficients. To get the magnitude spectrum, use abs(Y). RunMat uses the same fft syntax as MATLAB."
        - "@type": "Question"
          name: "How do I plot a single-sided FFT spectrum?"
          acceptedAnswer:
            "@type": "Answer"
            text: "Compute Y = fft(x), take the first N/2+1 elements, compute the magnitude with abs(), scale by 2/N (doubling to account for the discarded negative frequencies), and plot against a frequency axis computed as f = Fs*(0:N/2)/N where Fs is the sampling frequency."
        - "@type": "Question"
          name: "What window functions does RunMat support?"
          acceptedAnswer:
            "@type": "Answer"
            text: "RunMat supports hann, hamming, and blackman window functions with GPU-accelerated generation. Each supports symmetric and periodic modes."
        - "@type": "Question"
          name: "Can I run FFT in the browser without installing anything?"
          acceptedAnswer:
            "@type": "Answer"
            text: "Yes. RunMat's browser sandbox runs FFT computations client-side via WebAssembly. The plotting system renders spectra through WebGPU. No install, no account, no server round-trip."
        - "@type": "Question"
          name: "Does RunMat support 2D FFT?"
          acceptedAnswer:
            "@type": "Answer"
            text: "Yes. RunMat supports fft2, ifft2, fftn, and ifftn for multi-dimensional Fourier transforms. The 2D FFT is useful for image processing and spatial frequency analysis."
        - "@type": "Question"
          name: "How does RunMat FFT compare to MATLAB's FFT?"
          acceptedAnswer:
            "@type": "Answer"
            text: "RunMat supports the same fft, ifft, fft2, ifft2, fftshift, and ifftshift functions with identical syntax. MATLAB uses FFTW internally, which is heavily optimized for CPU. RunMat uses RustFFT for CPU paths with GPU shader support for power-of-two and mixed-radix lengths. For most signal processing workflows, the syntax and results are identical."
---

**TL;DR:** RunMat supports `fft`, `ifft`, `fft2`, `ifft2`, `fftshift`, `ifftshift`, and window functions (`hann`, `hamming`, `blackman`) with MATLAB-compatible syntax. Every code example on this page is runnable in the [sandbox](https://runmat.com/sandbox) with no install. Scroll to [basic FFT](#your-first-fft), [windowing](#windowing-why-your-spectrum-has-skirts), [2D FFT](#two-dimensional-fft), or [the FAQ](#frequently-asked-questions).


## FFT is where math meets measurement

The Fast Fourier Transform converts a time-domain signal into its frequency components. It sits between time-domain inspection and frequency-domain analysis, and it is the single most used operation in signal processing, vibration analysis, communications, audio engineering, and radar.

The math is well understood. The workflow around it is where most of the pain lives. MATLAB's FFT itself is fast (FFTW is excellent), but the surrounding workflow is slow: license checkout, launching the desktop app, server-rendered figures in MATLAB Online with 20-hour monthly caps. RunMat runs the same syntax in the browser with GPU-accelerated plots.


## Your first FFT

Generate a 50 Hz sine wave sampled at 1 kHz, compute the FFT, and plot the single-sided magnitude spectrum:

```matlab:runnable
Fs = 1000;
t = 0:1/Fs:0.5-1/Fs;
x = sin(2*pi*50*t);

N = length(x);
Y = fft(x);
P = abs(Y(1:N/2+1)) * 2/N;
f = Fs*(0:N/2)/N;

plot(f, P, 'o-', 'Color', [0 0.85 1], 'LineWidth', 10, 'MarkerSize', 10);
title('Single-sided magnitude spectrum');
xlabel('Frequency (Hz)');
ylabel('|X(f)|');
grid on;
xlim([0 120]);
ylim([0 1.2]);
```

The peak sits at 50 Hz. The spectrum is one-sided because a real signal's FFT is symmetric — the negative frequencies mirror the positive ones, so you only need the first `N/2+1` points. The `2/N` scaling normalizes the amplitude to match the original signal's peak value (the factor of 2 accounts for the energy in the discarded negative half).


## Multi-tone signal: resolving components

Real signals contain multiple frequencies. Generate a signal with three tones at 120 Hz, 350 Hz, and 780 Hz, add noise, and extract the components:

```matlab:runnable
Fs = 4000;
t = 0:1/Fs:0.25-1/Fs;
x = 0.8*sin(2*pi*120*t) + 0.5*sin(2*pi*350*t) + 0.3*sin(2*pi*780*t);
x = x + 0.4*randn(size(t));

N = length(x);
Y = fft(x);
P = abs(Y(1:N/2+1)) * 2/N;
f = Fs*(0:N/2)/N;

subplot(2, 1, 1);
plot(t*1000, x);
title('Noisy three-tone signal');
xlabel('Time (ms)');
ylabel('Amplitude');
grid on;

subplot(2, 1, 2);
plot(f, P);
title('Magnitude spectrum');
xlabel('Frequency (Hz)');
ylabel('|X(f)|');
grid on;
xlim([0 1000]);
```

Three peaks emerge from the noise floor at 120, 350, and 780 Hz. The amplitude ratios (0.8, 0.5, 0.3) are visible in the peak heights. The noise lifts the baseline; more noise means a higher floor, which is why signal-to-noise ratio matters in spectral analysis.


## Square wave: the iconic Fourier demo

A square wave decomposes into a sum of odd harmonics, each with amplitude `1/n`. This is the demonstration every signal processing textbook opens with, and it is the cleanest way to see what an FFT actually does:

```matlab:runnable
Fs = 10000;
t = 0:1/Fs:0.02-1/Fs;
x = sign(sin(2*pi*500*t));

N = length(x);
Y = fft(x);
P = abs(Y(1:N/2+1)) * 2/N;
f = Fs*(0:N/2)/N;

subplot(2, 1, 1);
plot(t*1000, x);
title('500 Hz square wave');
xlabel('Time (ms)'); ylabel('Amplitude');
grid on;

subplot(2, 1, 2);
plot(f, P);
title('Odd harmonics: 500, 1500, 2500, 3500...');
xlabel('Frequency (Hz)'); ylabel('|X(f)|');
grid on; xlim([0 5000]);
```

![500 Hz square wave and its FFT showing odd harmonics at 500, 1500, 2500, 3500 Hz](https://web.runmatstatic.com/blog-images/runmat-fft-guide.webp)

The time-domain plot is the square wave itself. The spectrum shows sharp peaks at 500 Hz, 1500 Hz, 2500 Hz, 3500 Hz — and nothing at the even harmonics (1000, 2000, 3000, 4000). The `1/n` amplitude rolloff is visible as the decreasing peak heights. Fourier's original result, in ten lines.


## Windowing: why your spectrum has skirts

If you FFT a finite-length signal without windowing, each frequency peak spreads into neighboring bins. This is spectral leakage: the FFT assumes the signal is periodic, so any discontinuity at the edges of the window creates artificial high-frequency content.

Window functions (Hann, Hamming, Blackman) taper the signal toward zero at both ends, reducing this leakage. The tradeoff: windows widen the main lobe (reducing frequency resolution) but push the sidelobes much lower.

```matlab:runnable
Fs = 1000;
t = 0:1/Fs:0.1-1/Fs;
x = sin(2*pi*50*t) + 0.5*sin(2*pi*55*t);
N = length(x);

w_rect = ones(1, N);
w_hann = hann(N)';
w_hamm = hamming(N)';
w_black = blackman(N)';

f = Fs*(0:N/2)/N;

subplot(2, 2, 1);
P = abs(fft(x .* w_rect));
plot(f, P(1:N/2+1) * 2/N);
title('Rectangular (no window)');
xlabel('Hz'); ylabel('|X(f)|');
grid on; xlim([30 80]);

subplot(2, 2, 2);
P = abs(fft(x .* w_hann));
plot(f, P(1:N/2+1) * 2/N);
title('Hann');
xlabel('Hz'); ylabel('|X(f)|');
grid on; xlim([30 80]);

subplot(2, 2, 3);
P = abs(fft(x .* w_hamm));
plot(f, P(1:N/2+1) * 2/N);
title('Hamming');
xlabel('Hz'); ylabel('|X(f)|');
grid on; xlim([30 80]);

subplot(2, 2, 4);
P = abs(fft(x .* w_black));
plot(f, P(1:N/2+1) * 2/N);
title('Blackman');
xlabel('Hz'); ylabel('|X(f)|');
grid on; xlim([30 80]);
```

The two tones at 50 Hz and 55 Hz are only 5 Hz apart — close enough that spectral leakage from the rectangular window blurs them together. The Hann window separates them. Blackman pushes the sidelobes even lower at the cost of a wider main lobe.

Choosing a window depends on what you need: Hann is a good general-purpose default. Hamming has a slightly narrower main lobe. Blackman has the best sidelobe suppression. If frequency resolution is critical and sidelobes are acceptable, use a rectangular window (or a longer signal).


## Zero-padding for interpolation

Zero-padding a signal before FFT does not add new frequency information. It interpolates between existing bins, producing a smoother-looking spectrum. Use `nextpow2` to find a good FFT length:

```matlab:runnable
Fs = 1000;
t = 0:1/Fs:0.05-1/Fs;
x = sin(2*pi*120*t) + 0.6*sin(2*pi*200*t);

N_orig = length(x);
N_padded = 2^nextpow2(4*N_orig);

Y_orig = fft(x, N_orig);
Y_padded = fft(x, N_padded);

f_orig = Fs*(0:N_orig/2)/N_orig;
f_padded = Fs*(0:N_padded/2)/N_padded;

P_orig = abs(Y_orig(1:N_orig/2+1)) * 2/N_orig;
P_padded = abs(Y_padded(1:N_padded/2+1)) * 2/N_orig;

subplot(2, 1, 1);
plot(f_orig, P_orig, 'o-', 'LineWidth', 2, 'MarkerSize', 5);
title('No zero-padding (50 samples)');
xlabel('Hz'); ylabel('|X(f)|');
grid on; xlim([0 400]);

subplot(2, 1, 2);
plot(f_padded, P_padded);
title('Zero-padded to 256 samples');
xlabel('Hz'); ylabel('|X(f)|');
grid on; xlim([0 400]);
```

The top plot uses `stem` because there are few frequency bins — the discrete samples are the honest representation. The bottom plot interpolates smoothly between those same bins. Both contain the same frequency information; the padded version is easier to read visually.


## Phase spectrum

The FFT output is complex. `abs` gives magnitude; `angle` gives phase. Phase tells you the time offset of each frequency component, which matters in system identification, filter design, and beamforming.

```matlab:runnable
Fs = 1000;
t = 0:1/Fs:0.1-1/Fs;
x = sin(2*pi*50*t + pi/4) + 0.7*cos(2*pi*120*t);
N = length(x);

Y = fft(x);
f = Fs*(0:N/2)/N;
mag = abs(Y(1:N/2+1)) * 2/N;
phase = angle(Y(1:N/2+1));

subplot(2, 1, 1);
plot(f, mag);
title('Magnitude');
xlabel('Hz'); ylabel('|X(f)|');
grid on; xlim([0 200]);

subplot(2, 1, 2);
plot(f, phase);
title('Phase');
xlabel('Hz'); ylabel('Angle (rad)');
grid on; xlim([0 200]);
```

The 50 Hz component has a phase offset of pi/4 radians (45 degrees), visible as a non-zero phase at that frequency. The 120 Hz component (a cosine) shows a phase near zero. Phase values at frequencies with low magnitude are dominated by noise and can be ignored.


## Inverse FFT: round-trip verification

`ifft` converts frequency-domain data back to the time domain. A clean round-trip (`ifft(fft(x))` recovers the original signal) is a basic sanity check for any FFT workflow:

```matlab:runnable
Fs = 1000;
t = 0:1/Fs:0.05-1/Fs;
x = sin(2*pi*100*t) + 0.5*sin(2*pi*250*t);
N = length(x);

Y = fft(x);
x_recovered = ifft(Y);
Y_roundtrip = fft(x_recovered);
err = abs(Y_roundtrip - Y);

subplot(2, 1, 1);
plot(t*1000, x);
title('Original');
xlabel('ms'); ylabel('Amplitude');
grid on;

subplot(2, 1, 2);
plot(0:N-1, err, 'Color', [0 0.85 1], 'LineWidth', 2);
title('Round-trip FFT error (near zero)');
xlabel('Bin index'); ylabel('Absolute error');
grid on;
```

The two plots should be identical. If they differ, something went wrong with your frequency-domain manipulation (asymmetric modification, wrong conjugate handling, or length mismatch).


## Two-dimensional FFT

`fft2` computes the 2D Fourier transform — used in image processing, antenna design, and spatial frequency analysis. This version keeps the construction simple for maximum runtime compatibility:

```matlab:runnable
N = 64;
n = 0:N-1;

rowPattern = sin(2*pi*5*n/N);
colPattern = sin(2*pi*12*n/N);
Z = ones(N, 1)*rowPattern + colPattern.'*ones(1, N);

F = abs(fft2(Z));

subplot(1, 2, 1);
imagesc(Z);
colorbar;
title('Spatial pattern');

subplot(1, 2, 2);
imagesc(log(1 + F));
colorbar;
title('2D FFT magnitude (unshifted log scale)');
```

The log scale compresses dynamic range so dominant frequency components stand out clearly. This example keeps the spectrum unshifted to avoid compatibility issues while still showing the main 2D frequency peaks.


## Practical pattern: frequency-domain filtering

Remove a specific frequency from a signal by zeroing out its FFT bins. This version focuses on before/after spectra so it stays robust across runtimes:

```matlab:runnable
Fs = 1000;
t = 0:1/Fs:0.5-1/Fs;
x = sin(2*pi*50*t) + sin(2*pi*200*t);
N = length(x);

Y_before = fft(x);
Y_after = Y_before;
f = Fs*(0:N/2)/N;

freqBins = Fs*(0:N-1)/N;
Y_after(abs(freqBins - 200) < 5 | abs(freqBins - (Fs-200)) < 5) = 0;

P_before = abs(Y_before(1:N/2+1)) * 2/N;
P_after = abs(Y_after(1:N/2+1)) * 2/N;

subplot(2, 1, 1);
plot(t*1000, x);
title('Original: 50 Hz + 200 Hz');
xlabel('ms'); grid on;

subplot(2, 1, 2);
plot(f, P_before, 'Color', [0.6 0.6 0.6], 'LineWidth', 2); hold on;
plot(f, P_after, 'Color', [0 0.85 1], 'LineWidth', 3);
hold off;
title('Spectrum before/after removing 200 Hz');
xlabel('Hz'); ylabel('|X(f)|');
grid on; xlim([0 400]);
```

Zeroing FFT bins is a brick-wall filter, which can cause ringing in the time domain (Gibbs phenomenon) if you transform back with `ifft`. For production filtering, taper the transition band or use a proper FIR/IIR filter. But this FFT-domain view is useful for quick exploration when you want to verify that a component was removed.


## How RunMat FFT compares

RunMat uses the same `fft` / `ifft` / `fft2` / `ifft2` / `fftshift` / `ifftshift` syntax as MATLAB. The implementation details differ:

| Dimension | RunMat | MATLAB |
|-----------|--------|--------|
| FFT engine | RustFFT (CPU), GPU shaders for power-of-two and mixed-radix lengths | FFTW (heavily optimized CPU library) |
| Window functions | `hann`, `hamming`, `blackman` with GPU-accelerated generation | Full Signal Processing Toolbox (dozens of windows) |
| Plotting | GPU-rendered in browser via WebGPU | CPU-rendered Java figure system (desktop), server-rendered (MATLAB Online) |
| Browser execution | Client-side via WebAssembly, no server | Server-side, 20 hr/mo free tier |
| License | Free, MIT licensed | $2,000+/seat + toolbox licenses |

MATLAB's FFTW is faster than RustFFT for raw FFT computation on large transforms. RunMat's advantage is the workflow around the FFT: generate a signal, compute the spectrum, and see the GPU-rendered plot in the same browser tab, with no license server, no install, and no figure round-trip to a remote host. For interactive exploration and teaching, the bottleneck is rarely the FFT itself. It is the time from wanting to look at a spectrum to actually seeing one, and that gap is mostly tool friction.

RunMat does not include MATLAB's Signal Processing Toolbox. Functions like `spectrogram`, `pwelch`, `butter`, `filter`, `designfilt`, and `detrend` are not available. If your workflow depends on these, MATLAB or Python (SciPy) is the better choice today. RunMat covers the core FFT primitives and the most common windowing functions.


## Try it now

Paste this into the [sandbox](https://runmat.com/sandbox) and experiment. Change the frequencies, adjust the window, add noise, switch between `hann` and `blackman`:

```matlab:runnable
Fs = 2000;
t = 0:1/Fs:0.2-1/Fs;
x = sin(2*pi*100*t) + 0.6*sin(2*pi*340*t) + 0.3*sin(2*pi*720*t);
x = x + 0.2*randn(size(t));

N = length(x);
w = hann(N)';
Y = fft(x .* w, N);
P = abs(Y(1:N/2+1)) * 2/N;
f = Fs*(0:N/2)/N;

subplot(2, 1, 1);
plot(t*1000, x);
title('Time domain');
xlabel('ms'); ylabel('Amplitude');
grid on;

subplot(2, 1, 2);
plot(f, P);
title('Hann-windowed spectrum');
xlabel('Hz'); ylabel('|X(f)|');
grid on; xlim([0 1000]);
```

<a href="https://runmat.com/sandbox" data-ph-capture-attribute-button-type="cta-sandbox" data-ph-capture-attribute-page="matlab-fft-guide">Open the RunMat sandbox</a> and start analyzing — it runs in the browser without an account.

For plotting fundamentals, read the [MATLAB plotting guide](/blog/matlab-plotting-guide). For GPU acceleration details, see the [GPU guide](/blog/how-to-use-gpu-in-matlab). For a broader comparison of tools, see [MATLAB alternatives](/blog/free-matlab-alternatives).


## Frequently asked questions

**How do I compute an FFT in RunMat?**
Use `Y = fft(x)` where `x` is a time-domain signal vector. The output `Y` is a complex array of frequency-domain coefficients. To get the magnitude spectrum, use `abs(Y)`. RunMat uses the same `fft` syntax as MATLAB.

**How do I plot a single-sided FFT spectrum?**
Compute `Y = fft(x)`, take the first `N/2+1` elements, compute the magnitude with `abs()`, scale by `2/N`, and plot against a frequency axis `f = Fs*(0:N/2)/N` where `Fs` is the sampling frequency.

**What window functions does RunMat support?**
RunMat supports `hann`, `hamming`, and `blackman` with GPU-accelerated generation. Each supports `symmetric` and `periodic` modes.

**Can I run FFT in the browser without installing anything?**
Yes. RunMat's sandbox runs FFT computations client-side via WebAssembly, and the plotting system renders spectra through WebGPU. Nothing is shipped to a server.

**Does RunMat support 2D FFT?**
Yes. `fft2`, `ifft2`, `fftn`, and `ifftn` are implemented for multi-dimensional transforms.

**How does RunMat FFT compare to MATLAB's FFT?**
Same syntax: `fft`, `ifft`, `fft2`, `ifft2`, `fftshift`, `ifftshift`. MATLAB uses FFTW internally (heavily optimized for CPU). RunMat uses RustFFT for CPU paths with GPU shader support for power-of-two and mixed-radix lengths. For most workflows, the syntax and results are identical.

**Does RunMat have the Signal Processing Toolbox?**
No. RunMat covers the core FFT family and common window functions. Functions like `spectrogram`, `pwelch`, `butter`, `filter`, and `designfilt` are not available. If your workflow requires these, MATLAB or Python (SciPy) is a better fit today.
