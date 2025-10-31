## Legend

| Symbol / Abbreviation         | Meaning                                                                                |
|------------------------------|----------------------------------------------------------------------------------------|
| **GPU**                      | ✓ = Supported (device kernels via provider hooks or offload) <br> — = Host only <br> N/A = Not applicable |
| **Fusion**                   | E = Elementwise <br> R = Reduction <br> S = Stencil / Convolution <br> M = MatMul <br> T = Transpose / Permutation <br> P = Pipeline / Fuse-friendly <br> — = Not applicable |
| **BLAS/LAPACK/FFTs**         | BLAS, LAPACK, FFT, — (not applicable)                                                  |
| **Names**                    | Many rows list multiple names (comma-separated) for closely related families; each name is a distinct builtin.|


## Acceleration / GPU

| Completed | Path                | Name(s)                                     | Purpose                       | GPU | Fusion | BLAS/LAPACK/FFTs | Notes                                         |
|-----------|---------------------|---------------------------------------------|-------------------------------|-----|--------|------------------|-----------------------------------------------|
|     ✅     | acceleration/gpu    | gpuArray, gather, gpuDevice, gpuInfo        | Move to/from device; query device | ✓   | P      | —                | Residency-aware; gpuArray/gather. |
|     ✅     | acceleration/gpu    | arrayfun                                    | Elementwise map on GPU        | ✓   | E      | —                | Provider hook (unary_map) + fusion inline constants. |
|     ✅     | acceleration/gpu    | pagefun                                     | Batched page-wise ops         | ✓   | P      | —                | Maps batched BLAS/FFT when present. |

## Arrays — creation, shape, indexing

| Completed | Path                | Name(s)                                                                                              | Purpose                         | GPU | Fusion | BLAS/LAPACK/FFTs               | Notes                                                                                   |
|-----------|---------------------|-----------------------------------------------------------------------------------------------------|---------------------------------|-----|--------|--------------------|-----------------------------------------------------------------------------------------|
|     ✅     | array/creation      | zeros, ones, eye, diag, linspace, logspace, randi, rand, randn, randperm, fill, range, colon         | Constructors & ranges           | ✓   | E      | —                  | RNG integrates rng streams; colon(a,b,…) implements :.                                   |
|      ✅    | array/creation      |  meshgrid                                                                                           | Grid generators                 | ✓   | T      | —                  | X/Y grid; device index kernels.                                                             |
|     ✅     | array/shape         | reshape, squeeze, permute, ipermute, repmat, kron, cat, horzcat, vertcat, flip, fliplr, flipud, rot90, circshift, tril, triu, diag | Layout transforms               | ✓   | T      | BLAS (for kron via GEMM blocks) | Device-side index kernels; fuse where possible.                                          |
|     ✅     | array/indexing      | sub2ind, ind2sub, find                                                                               | Index conversions & queries     | ✓   | P      | —                  | find supports dim/k variants; device kernels for masks.                                 |
|     ✅     | array/sorting-sets  | sort, sortrows, argsort (internal), unique, union, intersect, setdiff, ismember, issorted            | Ordering & set ops              | ✓   | P      | —                  | GPU sort; host Timsort/ radix; full semantics.                                         |
|     ✅     | array/introspection | size, length, numel, ndims, isempty, isscalar, isvector, ismatrix                                   | Metadata                        | N/A | —      | —                  |                                                          |

## Math — elementwise & reductions

| Completed | Path                | Name(s)                                                                         | Purpose                     | GPU | Fusion | BLAS/LAPACK/FFTs | Notes                        |
|-----------|---------------------|----------------------------------------------------------------------------------|-----------------------------|-----|--------|------------------|------------------------------|
|     ✅     | math/elementwise    | abs, sign, real, imag, conj, angle                                              | Complex helpers             | ✓   | E      | —                | Complex tensors supported.   |
|     ✅     | math/elementwise    | exp, expm1, log, log1p, log10, log2, sqrt, hypot, pow2                          | Exponentials & roots        | ✓   | E      | —                | Provider hooks unary_*.      |
|      ✅    | math/elementwise    |  single, double                                                                | Numeric casting (array/scalar) | ✓   | E      | —                | Type conversion to f32/f64. |
|      ✅    | math/elementwise    |  times, rdivide, ldivide, power, gamma, factorial                              | Elementwise ops & specials  | ✓   | E      | —                | GPU-aware; broadcast semantics. |
|     ✅     | math/trigonometry   | sin, cos, tan, asin, acos, atan, atan2, sinh, cosh, tanh, asinh, acosh, atanh   | Trig & hyperbolic           | ✓   | E      | —                | sin, asinh, acosh, atanh implemented; others mirror. |
|     ✅     | math/rounding       | round, floor, ceil, fix, mod, rem                                               | Rounding & modulo           | ✓   | E      | —                | mod/rem MATLAB semantics.    |
|     ✅     | math/reduction      | sum, prod, mean, median, min, max, any, all, std, var, cumsum, cumprod, cummin, cummax, diff, nnz | Reductions & cumulatives    | ✓   | R      | —                | omitnan host fallback initially; GPU reductions via hooks. |

## Linear algebra

| Completed | Path                   | Name(s)                                               | Purpose                           | GPU | Fusion | BLAS/LAPACK/FFTs | Notes                                               |
|-----------|------------------------|------------------------------------------------------|-----------------------------------|-----|--------|------------------|-----------------------------------------------------|
|     ✅     | math/linalg/ops        | mtimes (*), mrdivide (/), mldivide (\), transpose (.'), ctranspose ('), trace | Core ops            | ✓   | M/T    | BLAS              | GEMM/GEMV; fall back to host BLAS; GPU matmul hooks.|
|      ✅    | math/linalg/ops        |  dot, mpower                                                | Dot product; matrix power      | ✓   | M      | BLAS              | Device/offload support via provider hooks.             |
|     ✅     | math/linalg/factor     | lu, qr, chol, svd, eig                               | Factorizations & eigensolvers     | ✓   | —      | LAPACK           | GPU optional via provider; host uses LAPACK.        |
|     ✅     | math/linalg/solve      | linsolve, pinv, inv, det, rank, rcond, cond, norm    | Solves & metrics                  | ✓   | —      | LAPACK/BLAS      | Encourage \ instead of inv(A)*b.                    |
|     ✅     | math/linalg/structure  | bandwidth, issymmetric, ishermitian, symrcm| Structure queries                 | —   | —      | —                | Useful diagnostics.                                 |

## FFTs & signal / image primitives

| Completed | Path           | Name(s)                                                    | Purpose                   | GPU | Fusion | BLAS/LAPACK/FFTs | Notes                                                                                   |
|-----------|----------------|------------------------------------------------------------|---------------------------|-----|--------|------------------|-----------------------------------------------------------------------------------------|
|     ✅     | math/fft       | fft, ifft, fft2, ifft2, fftshift, ifftshift                | Fourier transforms        | ✓   | P      | FFT              | Provider hook to device FFTs; host uses pocketfft.                                      |
|     ✅     | math/signal    | conv, conv2, deconv, filter                                | 1D/2D conv & filtering   | ✓   | S      | —                | Small kernels fuse; large can call FFT‑based conv.                                      |
|     ✅     | image/filters  | fspecial                                                   | Generate filter kernels   | —   | —      | —                | Included: gaussian, average, laplacian, log, motion, sobel, prewitt, unsharp, disk.     |
|     ✅     | image/filters  | imfilter, filter2                                          | Apply linear filters      | ✓   | S      | —                | Padding modes: replicate, circular, symmetric.                                          |

## Polynomials & fitting

| Completed | Path       | Name(s)                                         | Purpose           | GPU | Fusion | BLAS/LAPACK/FFTs | Notes                                  |
|-----------|------------|--------------------------------------------------|-------------------|-----|--------|------------------|----------------------------------------|
|     ✅    | math/poly  | polyval, polyfit, roots, polyder, polyint        | Poly utilities    | —   | P      | LAPACK (QR)      | polyfit via least squares (\/QR).      |

## Statistics (core)

| Completed | Path           | Name(s)                | Purpose                    | GPU | Fusion | BLAS/LAPACK/FFTs | Notes                                 |
|-----------|----------------|-----------------------|----------------------------|-----|--------|------------------|---------------------------------------|
|     ✅     | stats/summary  | corrcoef, cov         | Correlation / covariance   | ✓   | R      | BLAS             | GPU optional; host via BLAS reductions. |
|     ✅     | stats/hist     | histcounts, histcounts2| Binning                   | ✓   | P      | —                | Includes GPU radix/binning.           |
|     ✅     | stats/random   | rng                   | RNG control                | N/A | —      | —                | Seeds, streams (host + device).       |

## Logical & comparisons

| Completed | Path           | Name(s)                                | Purpose            | GPU | Fusion | BLAS/LAPACK/FFTs | Notes                                |
|-----------|----------------|----------------------------------------|--------------------|-----|--------|------------------|--------------------------------------|
|     ✅     | logical/rel    | eq, ne, gt, ge, lt, le                 | Relational ops     | ✓   | E      | —                | Implemented for scalars/tensors/strings. |
|     ✅     | logical/bit    | and, or, xor, not                      | Logical ops        | ✓   | E      | —                | Elementwise semantics.               |
|     ✅     | logical/tests  | isnan, isinf, isfinite, isreal, islogical, isnumeric | Predicates       | ✓   | E      | —                | Device kernels for masks.            |
|      ✅    | logical         |  logical                             | Type conversion    | ✓   | E      | —                | Produces logical arrays/scalars.     |

## Strings & text

| Completed | Path              | Name(s)                                                        | Purpose                | GPU | Fusion | BLAS/LAPACK/FFTs | Notes                                     |
|-----------|-------------------|---------------------------------------------------------------|------------------------|-----|--------|------------------|--------------------------------------------|
|     ✅     | strings/core      | string, char, strlength, num2str, str2double, compose, sprintf| Conversions & formatting| —   | —      | —                | sprintf returns string; fprintf in I/O.     |
|     ✅     | strings/core      |  strcmp, strcmpi, strncmp, strings, string.empty             | Comparisons & constructors| —   | —      | —                |                                            |
|     ✅     | strings/search    | contains, startsWith, endsWith, strfind                       | Search                 | —   | —      | —                |                                            |
|     ✅     | strings/transform | lower, upper, strip, strtrim, replace, split, join            | Transformations        | —   | —      | —                |                                            |
|     ✅     | strings/transform |  strrep, strcat, extractBetween, erase, eraseBetween, pad    | Transformations        | —   | —      | —                |                                            |
|     ✅     | strings/regex     | regexp, regexpi, regexprep                                    | Regex utilities        | —   | —      | —                | Backed by regex crate.                     |

## Structs, cells, containers

| Completed | Path            | Name(s)                                          | Purpose               | GPU | Fusion | BLAS/LAPACK/FFTs | Notes                                            |
|-----------|-----------------|--------------------------------------------------|-----------------------|-----|--------|------------------|--------------------------------------------------|
|     ✅      | structs/core    | struct, fieldnames, isfield, getfield, setfield, rmfield, orderfields | Struct manipulation | —   | —      | —                |                                                  |
|     ✅     | cells/core      | cell, cell2mat, mat2cell, cellfun                 | Cell arrays & ops     | —   | P      | —                | cellfun maps to fusion where trivial.            |
|     ✅     | cells/core      |  cellstr                                      | Cell/string conversion | —   | —      | —                |                                                  |
|     ✅     | containers/map  | containers.Map (constructor)                      | String→value map      | —   | —      | —                | Handy; small footprint.             |

## Introspection, environment, diagnostics

| Completed | Path             | Name(s)                                                                         | Purpose                    | GPU | Fusion | BLAS/LAPACK/FFTs | Notes                                    |
|-----------|------------------|---------------------------------------------------------------------------------|----------------------------|-----|--------|------------------|------------------------------------------|
|     ✅     | introspection    | class, isa, which, whos, who                                                    | Type & workspace queries   | N/A | —      | —                | which resolves builtins/files.            |
|     ✅     | introspection    |  ischar, isstring                                                              | Type predicates            | N/A | —      | —                |                                           |
|     ✅     | diagnostics      | error, warning, assert                                                           | Errors & checks            | N/A | —      | —                | Integrate with MException.                |
|     ✅     | timing           | tic, toc, timeit, pause                                                          | Timing utilities           | N/A | —      | —                | timeit microbenchmark helper.             |

## I/O — filesystem and files

| Completed | Path         | Name(s)                                                                                                                                    | Purpose                | GPU | Fusion | BLAS/LAPACK/FFTs | Notes                                     |
|-----------|--------------|--------------------------------------------------------------------------------------------------------------------------------------------|------------------------|-----|--------|------------------|--------------------------------------------|
|     ✅     | io/repl-fs   | cd, pwd, ls/dir, mkdir, rmdir, movefile, copyfile, delete, exist, which, path, addpath, rmpath, genpath, savepath, tempdir, tempname, getenv, setenv | REPL ergonomics & FS  | —   | —      | —                | Covers cd and friends.                     |
|     ✅     | io/filetext  | fileread, filewrite, fopen, fclose, fread, fwrite, feof, fgets, fprintf                              | Text/binary I/O        | —   | —      | —                |                                            |
|     ✅     | io/core       |  disp                                                                                               | Display to stdout      | —   | —      | —                | sink builtin                             |
|     ✅     | io/tabular   | readmatrix, writematrix, csvread, csvwrite, dlmread, dlmwrite                                        | Simple tabular I/O     | —   | —      | —                | CSV/TSV focus; tables package later.        |
|     ✅     | io/mat       | save, load                                                                                            | MAT‑like persistence  | —   | —      | —                | Must completely support all semantics. |

## I/O — JSON & Networking

| Completed | Path         | Name(s)                                           | Purpose                    | GPU | Fusion | BLAS/LAPACK/FFTs | Notes                                  |
|-----------|--------------|---------------------------------------------------|----------------------------|-----|--------|------------------|----------------------------------------|
|     ✅     | io/json      | jsonencode, jsondecode                            | JSON serialize/parse       | —   | —      | —                | UTF‑8.                                |
|     ✅     | io/net       | tcpserver, accept, tcpclient, read, readline, write, close | TCP servers/clients       | —   | —      | —                | Complete, supports existing MATLAB semantics; includes blocking, unblocking, TLS. Unix socket (UDS) supported (beyond MATLAB). |
|     ✅     | io/http | webread, webwrite, weboptions               | HTTP client                | —   | —      | —                | Built on tcpclient + TLS.              |

## I/O — images (core)

| Completed | Path              | Name(s)        | Purpose      | GPU | Fusion | BLAS/LAPACK/FFTs | Notes                       |
|-----------|-------------------|--------------- |------------- |-----|--------|------------------|-----------------------------|
|     ✅     | io/image          | imread, imwrite, imfinfo | Image I/O     | —   | —      | —                | All image formats in `imread` / `imwrite` / `imfinfo` on MATLAB are supported, including PNG/JPEG.          |
|     ✅     | plot/images (gui) | imshow, imagesc         | Display       | —   | —      | —                | Behind gui/jupyter features.|

## Plotting & visualization (feature‑gated)

| Completed | Path        | Name(s)                                                                                          | Purpose         | GPU | Fusion | BLAS/LAPACK/FFTs | Notes                                          |
|-----------|-------------|--------------------------------------------------------------------------------------------------|-----------------|-----|--------|------------------|------------------------------------------------|
|           | plot/core   | plot, scatter, semilogx, semilogy, loglog, bar, histogram, stairs, stem, subplot, figure, hold, title, xlabel, ylabel, legend, xlim, ylim, axis | 2D plotting    | —   | —      | —                | Provided by runmat-plot (gui/jupyter).         |
|           | plot/3d     | surf, mesh, contour, contourf                                                                    | 3D/surfaces     | —   | —      | —                |                                                |

## Constants

| Completed | Path      | Name(s)                           | Purpose              | GPU | Fusion | BLAS/LAPACK/FFTs | Notes                                   |
|-----------|-----------|-----------------------------------|----------------------|-----|--------|------------------|-----------------------------------------|
|           | constants | pi, e, eps, inf, Inf, nan, NaN, true, false | Fundamental constants | N/A | —      | —                | |
