---
title: "Builtins"
category: "Builtins"
section: "6.1"
last_updated: "May 28, 2026"
---

# RunMat Runtime Builtin Library

This matrix tracks the public builtin surface implemented by `runmat-runtime`.

## Legend

| Column | Meaning |
| --- | --- |
| `Status` | `done` means implemented in the runtime; `planned` means documented as expected but not complete in the current library table. |
| `GPU` | `yes` means provider hooks or device offload exist; `host` means host only; `N/A` means acceleration does not apply. |
| `Fusion` | `E` elementwise, `R` reduction, `S` stencil/convolution, `M` matrix multiply, `T` transpose/permutation, `P` pipeline/fuse-friendly, `-` not applicable. |
| `Backend` | External numerical backend category: BLAS, LAPACK, FFT, or `-` when not applicable. |

For GPU execution details, see [GPU Acceleration & Fusion Engine](/docs/runtime/gpu). For JIT interaction with builtin and semantic calls, see [JIT Compilation Pipeline](/docs/runtime/jit/pipeline).

## Acceleration / GPU

| Status | Path | Names | GPU | Fusion | Backend | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| done | `acceleration/gpu` | `gpuArray`, `gather`, `gpuDevice`, `gpuInfo` | yes | P | - | Moves values to and from device memory and queries device state. |
| done | `acceleration/gpu` | `arrayfun` | yes | E | - | Elementwise GPU map through provider hooks and fusion inline constants. |
| done | `acceleration/gpu` | `pagefun` | yes | P | - | Batched page-wise operations, mapped to batched BLAS/FFT when available. |

## Arrays

| Status | Path | Names | GPU | Fusion | Backend | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| done | `array/creation` | `zeros`, `ones`, `eye`, `diag`, `linspace`, `logspace`, `randi`, `rand`, `randn`, `randperm`, `fill`, `range`, `colon` | yes | E | - | Constructors, ranges, and RNG-backed array creation. |
| done | `array/creation` | `meshgrid` | yes | T | - | Grid generation with device index kernels. |
| done | `array/shape` | `reshape`, `squeeze`, `permute`, `ipermute`, `repmat`, `repelem`, `kron`, `cat`, `horzcat`, `vertcat`, `flip`, `fliplr`, `flipud`, `rot90`, `circshift`, `tril`, `triu`, `diag` | yes | T | BLAS | Layout transforms; `kron` can use GEMM blocks. |
| done | `array/indexing` | `sub2ind`, `ind2sub`, `find` | yes | P | - | Index conversions and mask queries. |
| done | `array/sorting-sets` | `sort`, `sortrows`, `argsort`, `unique`, `union`, `intersect`, `setdiff`, `ismember`, `issorted` | yes | P | - | Ordering and set operations. |
| done | `array/introspection` | `size`, `length`, `numel`, `ndims`, `isempty`, `isscalar`, `isvector`, `ismatrix` | N/A | - | - | Shape and metadata queries. |

## Math

| Status | Path | Names | GPU | Fusion | Backend | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| done | `math/elementwise` | `abs`, `sign`, `real`, `imag`, `conj`, `angle` | yes | E | - | Complex helpers. |
| done | `math/elementwise` | `exp`, `expm1`, `log`, `log1p`, `log10`, `log2`, `sqrt`, `hypot`, `pow2` | yes | E | - | Exponentials and roots. |
| done | `math/elementwise` | `single`, `double`, `uint16` | yes | E | - | Numeric casting for scalars and arrays. |
| done | `math/elementwise` | `plus`, `minus`, `times`, `rdivide`, `ldivide`, `power`, `gamma`, `factorial` | yes | E | - | Elementwise arithmetic and specials. |
| done | `math/trigonometry` | `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh` | yes | E | - | Trigonometric and hyperbolic functions. |
| done | `math/rounding` | `round`, `floor`, `ceil`, `fix`, `mod`, `rem` | yes | E | - | MATLAB-style rounding and modulo behavior. |
| done | `math/reduction` | `sum`, `prod`, `mean`, `median`, `min`, `max`, `any`, `all`, `std`, `var`, `cumsum`, `cumprod`, `cummin`, `cummax`, `diff`, `nnz` | yes | R | - | Reductions and cumulative operations. |

## Linear Algebra

| Status | Path | Names | GPU | Fusion | Backend | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| done | `math/linalg/ops` | `mtimes`, `mrdivide`, `mldivide`, `transpose`, `ctranspose`, `trace` | yes | M/T | BLAS | Core matrix operations. |
| done | `math/linalg/ops` | `dot`, `mpower` | yes | M | BLAS | Dot product and matrix power. |
| done | `math/linalg/factor` | `lu`, `qr`, `chol`, `svd`, `eig` | yes | - | LAPACK | Factorizations and eigensolvers. |
| done | `math/linalg/solve` | `linsolve`, `pinv`, `inv`, `det`, `rank`, `rref`, `rcond`, `cond`, `norm` | yes | - | LAPACK/BLAS/- | Solves, row reduction, and matrix metrics. |
| done | `math/linalg/structure` | `bandwidth`, `issymmetric`, `ishermitian`, `symrcm` | host | - | - | Structure queries and diagnostics. |

## FFT, Signal, and Image Processing

| Status | Path | Names | GPU | Fusion | Backend | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| done | `math/fft` | `fft`, `ifft`, `fft2`, `ifft2`, `fftshift`, `ifftshift` | yes | P | FFT | Fourier transforms through host or device FFT providers. |
| done | `math/signal` | `conv`, `conv2`, `deconv`, `filter`, `butter` | yes | S | - | Convolution, Butterworth design, and filtering; large convolution may use FFT paths. |
| done | `image/filters` | `fspecial` | host | - | - | Filter kernel generation. |
| done | `image/filters` | `imfilter`, `filter2` | yes | S | - | Linear image filters and padding modes. |
| done | `image/color` | `rgb2gray`, `gray2rgb`, `ind2rgb`, `im2double`, `im2uint8`, `im2uint16`, `rgb2hsv`, `hsv2rgb`, `rgb2lab`, `lab2rgb` | host | - | - | Image color and class conversions. |

## Polynomials and Fitting

| Status | Path | Names | GPU | Fusion | Backend | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| done | `math/poly` | `polyval`, `polyfit`, `roots`, `polyder`, `polyint` | host | P | LAPACK | Polynomial utilities; `polyfit` uses least squares. |

## Control Systems

| Status | Path | Names | GPU | Fusion | Backend | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| done | `control` | `tf` | host | - | - | Transfer-function model construction. |
| done | `control` | `step`, `impulse`, `nyquist` | host | - | - | Time and frequency response evaluation. |

## Statistics

| Status | Path | Names | GPU | Fusion | Backend | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| done | `stats/summary` | `corrcoef`, `cov` | yes | R | BLAS | Correlation and covariance. |
| done | `stats/hist` | `histcounts`, `histcounts2` | yes | P | - | Binning and histogram counts. |
| done | `stats/random` | `rng` | N/A | - | - | Host and device RNG state. |

## Dates and Times

| Status | Path | Names | GPU | Fusion | Backend | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| done | `datetime` | `datetime`, `dateshift`, `year`, `month`, `day`, `hour`, `minute`, `second` | host | - | - | Date/time construction, formatting, component extraction, arithmetic, and calendar boundary shifting. |
| done | `duration` | `duration`, `seconds`, `minutes`, `hours`, `days`, `milliseconds` | host | - | - | Duration construction, display, arithmetic, and datetime interop. |

## Logical and Comparisons

| Status | Path | Names | GPU | Fusion | Backend | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| done | `logical/rel` | `eq`, `ne`, `gt`, `ge`, `lt`, `le` | yes | E | - | Relational operations for scalars, tensors, and strings. |
| done | `logical/bit` | `and`, `or`, `xor`, `not` | yes | E | - | Elementwise logical operations. |
| done | `logical/tests` | `isnan`, `isinf`, `isfinite`, `isreal`, `islogical`, `isnumeric` | yes | E | - | Predicates and mask generation. |
| done | `logical` | `logical` | yes | E | - | Logical conversion. |

## Strings and Text

| Status | Path | Names | GPU | Fusion | Backend | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| done | `strings/core` | `string`, `char`, `strlength`, `num2str`, `str2double`, `compose`, `sprintf` | host | - | - | Conversion and formatting. |
| done | `strings/core` | `strcmp`, `strcmpi`, `strncmp`, `strings`, `string.empty` | host | - | - | String comparisons and constructors. |
| done | `strings/search` | `contains`, `startsWith`, `endsWith`, `strfind` | host | - | - | Search operations. |
| done | `strings/transform` | `lower`, `upper`, `strip`, `strtrim`, `replace`, `split`, `join` | host | - | - | Basic transformations. |
| done | `strings/transform` | `strrep`, `strcat`, `extractBetween`, `erase`, `eraseBetween`, `pad` | host | - | - | Additional string transformations. |
| done | `strings/regex` | `regexp`, `regexpi`, `regexprep` | host | - | - | Regex-backed operations. |

## Structs, Cells, and Containers

| Status | Path | Names | GPU | Fusion | Backend | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| done | `structs/core` | `struct`, `fieldnames`, `isfield`, `getfield`, `setfield`, `rmfield`, `orderfields` | host | - | - | Struct manipulation. |
| done | `cells/core` | `cell`, `cell2mat`, `mat2cell`, `cellfun` | host | P | - | Cell arrays and mapping helpers. |
| done | `cells/core` | `cellstr` | host | - | - | Cell/string conversion. |
| done | `containers/map` | `containers.Map` | host | - | - | String-to-value map support. |
| done | `table` | `table`, `height`, `width`, `groupsummary` | host | - | - | Table containers, properties, dot/paren/brace indexing, grouping summaries, display, and row sorting through `sortrows`. |

## Introspection, Environment, and Diagnostics

| Status | Path | Names | GPU | Fusion | Backend | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| done | `introspection` | `class`, `isa`, `which`, `whos`, `who` | N/A | - | - | Type and workspace queries. |
| done | `introspection` | `ischar`, `isstring` | N/A | - | - | Type predicates. |
| done | `diagnostics` | `error`, `warning`, `assert` | N/A | - | - | Runtime diagnostics. |
| done | `timing` | `tic`, `toc`, `timeit`, `pause` | N/A | - | - | Timing and benchmarking helpers. |

## I/O - Filesystem and Files

| Status | Path | Names | GPU | Fusion | Backend | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| done | `io/repl-fs` | `cd`, `pwd`, `ls`, `dir`, `mkdir`, `rmdir`, `movefile`, `copyfile`, `delete`, `exist`, `which`, `path`, `addpath`, `rmpath`, `genpath`, `savepath`, `tempdir`, `tempname`, `getenv`, `setenv` | host | - | - | REPL filesystem and environment helpers. |
| done | `io/filetext` | `fileread`, `filewrite`, `fopen`, `fclose`, `fread`, `fwrite`, `feof`, `fgetl`, `fgets`, `fprintf` | host | - | - | Text and binary file I/O. |
| done | `io/core` | `disp` | host | - | - | Display output sink. |
| done | `io/interactive` | `input` | host | - | - | Prompted input, including text mode. |
| done | `io/tabular` | `readtable`, `readmatrix`, `writematrix`, `csvread`, `csvwrite`, `dlmread`, `dlmwrite` | host | - | - | Tabular I/O; `readtable` imports delimited text and spreadsheet files as table variables with sheet/range/name options. |
| done | `io/mat` | `save`, `load` | host | - | - | MAT-like persistence. |

## I/O - JSON, Networking, and Images

| Status | Path | Names | GPU | Fusion | Backend | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| done | `io/json` | `jsonencode`, `jsondecode` | host | - | - | UTF-8 JSON serialization and parsing. |
| done | `io/net` | `tcpserver`, `accept`, `tcpclient`, `read`, `readline`, `write`, `close` | host | - | - | TCP client/server operations, TLS, and Unix sockets. |
| done | `io/http` | `webread`, `webwrite`, `weboptions` | host | - | - | HTTP client operations. |
| done | `io/image` | `imread`, `imwrite`, `imfinfo` | host | - | - | Image file I/O. |
| done | `image` | `imhist` | host | - | - | Grayscale and indexed-image intensity histograms, with statement-form plotting. |
| done | `plot/images` | `image`, `imshow`, `imagesc` | host | - | - | Image display through plotting features. |

## Plotting and Visualization

| Status | Path | Names | GPU | Fusion | Backend | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| done | `plot/core` | `plot`, `scatter`, `scatterplot`, `semilogx`, `semilogy`, `loglog`, `bar`, `area`, `histogram`, `hist`, `stairs`, `stem`, `errorbar`, `quiver`, `patch`, `pie`, `subplot`, `figure`, `gcf`, `gca`, `hold`, `clf`, `cla`, `drawnow`, `print`, `get`, `set`, `isgraphics`, `ishandle`, `title`, `sgtitle`, `suptitle`, `xlabel`, `ylabel`, `zlabel`, `legend`, `xlim`, `ylim`, `zlim`, `axis`, `caxis`, `clim`, `xline`, `yline`, `text`, `grid`, `box`, `colormap`, `shading`, `colorbar` | host | - | - | Feature-gated through `plot-core` and backed by `runmat-plot`. |
| done | `plot/3d` | `plot3`, `scatter3`, `surf`, `surfc`, `mesh`, `meshc`, `contour`, `contour3`, `contourf`, `fill3`, `heatmap`, `view` | host | - | - | Surface, contour, 3D, and heatmap plotting through `plot-core`. |

## Constants

| Status | Path | Names | GPU | Fusion | Backend | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| done | `constants` | `pi`, `e`, `eps`, `inf`, `Inf`, `nan`, `NaN`, `true`, `false` | N/A | - | - | Fundamental constants. |
| done | `constants` | `i`, `j` | N/A | - | - | Imaginary units. |
