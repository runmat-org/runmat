# Binary64 trigonometry translation manifest

Algorithm source: locked, cached rust-lang/libm **0.2.16**, registry checksum `b6d2cec3eae94f9f509c767b45932f1ada8350c4bdb85af2fcab4a3c14807981`. Packaged `.cargo_vcs_info.json` pins git revision `dfd2203a4d6110820ad7bb65cafe1bf331a03a3d`, path `libm`. All eight Rust source files were read completely before translation. No upstream test suites were copied.

| Translation | Upstream `src/math/` file | SHA256 of upstream source |
| --- | --- | --- |
| `trigonometry.wgsl` | `sin.rs` | `b48cebd120fc2ac93a6563059cabda28a72c8680304eb8335b92b412235cc232` |
| `trigonometry.wgsl` | `cos.rs` | `6e81752899c471bd42d04bdf08e0b22bc10a29f92fa2a343281b995fd342f035` |
| `trigonometry.wgsl` | `tan.rs` | `2ea87ca18526753db6d149635ed6c895c8fbafd1e0fa7a9c04f7dc44b4eae986` |
| `trig_kernels.wgsl` | `k_sin.rs` | `dfe6524cb2d530ec51eb1a2ab233aeb65b3f12177c1fb2f50f044ea41170f5d8` |
| `trig_kernels.wgsl` | `k_cos.rs` | `d4867527877c846aa8e2a0b43cf06f03e2d94c321a548acaf8d5171c09ff4057` |
| `trig_kernels.wgsl` | `k_tan.rs` | `dc3f8ece9187bbd65792a7d1068e676092e9c2a0886f28c80579f9bd94d744b7` |
| `range_reduction.wgsl` | `rem_pio2.rs` | `5a040ab09dc356bca6d57712327151ed43003e2bac156553e3eb87d1cdefd2d1` |
| `range_reduction.wgsl` | `rem_pio2_large.rs` | `986b5b9ad2c8fba828aee47bf198eab97fab243359ab03eb09783d98ca5847b5` |

The origins are FreeBSD msun `s_sin.c`, `s_cos.c`, `s_tan.c`, `k_sin.c`, `k_cos.c`, `k_tan.c`, `e_rem_pio2.c`, and `k_rem_pio2.c`. Their Sun permission notices remain in translated shader source. Bruce D. Evans' reducer optimization attribution is retained.

Modifications: WGSL syntax, explicit binary64 literals, fixed-size addressable arrays, result struct instead of output slices, and specialization of the large reducer to binary64 precision (`prec=1`, `jk=jp=4`). The 66-entry base-2^24 table is the upstream binary64-capable table, not the 690-entry quad-precision table. The upstream 20-entry work arrays and cancellation-driven recomputation remain. The medium reducer retains its three-stage cancellation corrections and the large reducer returns both high and low remainder components. Signed small-reduction branches are factored into a signed multiplier without changing the arithmetic operations. Floating exception flags are not exposed; tiny-input return values, NaN/infinity results and signed zeros remain.

`conversions.wgsl` is independent original support, not a libm translation. Radians/degrees multiply by the rounded native binary64 conversion constants. Normal-range multiplication uses binary64 arithmetic. For magnitudes below 2^-767, a base-2^32 integer convolution retains the full 106-bit product and rounds once to nearest/even, including subnormal input/output and both signed zeros. This avoids native subnormal flushing and the double-rounding hazard of scaling an already-rounded product down. Its private helper is intentionally limited to the two positive normal angle constants.

Dependencies: `runmat_f64_high`, `runmat_f64_low`, `runmat_f64_from_words` (low word first), and `runmat_math64_scalbn` from the shared word/scaling support. Dynamic indices use i32 and nonnegative values; no signed remainder operation is used. The three trig WGSL files form a coherent dependency group plus shared word/scaling support. Independent `conversions.wgsl` requires `runmat_f64_high` and SHADER_INT64 but no trig kernels or scaling. Public scalar functions: `runmat_math64_sin`, `runmat_math64_cos`, `runmat_math64_tan`, `runmat_math64_radians`, `runmat_math64_degrees`. Vector wrappers belong to typed legalization.

These near-rounded algorithms require correctly rounded binary64 primitive arithmetic with original grouping, no reassociation/contraction, and qualified denormal behavior. Their presence does not establish correctly-rounded results for all inputs or hardware qualification. Compiler and hardware verification are separate acceptance work.

## libm MIT permission

rust-lang/libm as a whole is available for use under the MIT license. Its retained copyright notices include:

Copyright (c) 2018 Jorge Aparicio

Copyright © 2005-2020 Rich Felker, et al.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
