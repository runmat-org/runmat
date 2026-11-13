# Elementary binary64 WGSL source provenance

These files translate the already locked rust-lang/libm **0.2.16** registry
source. Cargo registry archive checksum:
`b6d2cec3eae94f9f509c767b45932f1ada8350c4bdb85af2fcab4a3c14807981`.
Upstream project: <https://github.com/rust-lang/libm>.
All source paths below are under that version's `src/math/`.

| Source | SHA-256 of original source bytes |
| --- | --- |
| `exp2.rs` | `e7270f15a682b69a49e042c80503c206d37c6c895f2c63da59a5e2431fa5fedc` |
| `log2.rs` | `1ec0cdd94bf4435ff0402d209089b1bdbcb1ba9a4cd5f77807f9c84cf534f980` |
| `expm1.rs` | `6cc612ca7cf1384dc15f541b301a1f176efbc79536aff1293408b3121cc30ad3` |
| `log1p.rs` | `bb8ce741e54d8f001d9c12450a14a663bd413014da2ade9ba1f5fbc9ff0c9a88` |
| `pow.rs` | `c5566d0f48682d107e8803b0ed204ff3521503348f18a4f26cbe3f47fef15192` |
| `asin.rs` | `9693a7cb61674ec6fe22be9db8fe20276729c48b1562fd86d5421e372a0a12e4` |
| `acos.rs` | `efd83a6138061209867f06421c6a756b8ca3aab5c6c0a85c4aba31bedb299da5` |
| `atan.rs` | `54181dbddf4f77b90b32b9f6819ac989da570dc02b791f3685d20a23d51324be` |
| `atan2.rs` | `7d028d4d397f7484b8feed7b228eda65b84acac94d27729d236d5df95e38d8f0` |
| `sinh.rs` | `95cb1019a76c82312a5a65b3dd5f8a6a8b9b77aed1934c7254eafb94604b94a0` |
| `cosh.rs` | `71aed40077a04e659c15c2d2cc6e4faba89a89957e61ecacb7f6f5516bed1753` |
| `tanh.rs` | `c81ff9f4aa902d8fc63922f17d7d9e395983a6582c4587cbbeb194b613dd90c6` |
| `asinh.rs` | `8fd9431978e06904ed810d873f3de5bc097130d4b681a727644aa82d4f0fc627` |
| `acosh.rs` | `c3d9de791399f7d6c946bee5642ac1357783eb3d1a9d226e2f514979c764bfdc` |
| `atanh.rs` | `84d3fe8b2674a9681f553b7afd950bea47f498c38f81af9d58115d52e833178e` |
| `expo2.rs` | `abd80ec291bebe24d3e0401f5600469f6248052be97f06885c32451fae57d3a8` |
| `k_expo2.rs` | `0e08b00b774731da6f16fd3ac8282a727a493e47b7a7292172b6136c04bd5dd4` |

## Translation map and numerical changes

- `exp2.wgsl`: exp2.rs, including all 512 interleaved table encodings
  and its Gal/Bachelis reduction and minimax polynomial. Table f64 values are
  stored as exact low/high u32 words. Final scaling uses the shared integer
  round-to-nearest-even `runmat_math64_scalbn`.
- `log2.wgsl`: log2.rs, split reconstruction unchanged; subnormal input
  normalization uses encoded scaling rather than native denormal multiplication.
- `expm1.wgsl`, `log1p.wgsl`: respective upstream cancellation-sensitive
  algorithms, split constants and reconstruction unchanged.
- `pow.wgsl`: pow.rs split logarithm, simulated multiprecision exponent
  product and power reconstruction. This is not exp(y*log(x)). Special square,
  reciprocal and positive square-root paths normalize input and pack subnormal
  output through encoded scaling. Tiny finite exponents below 2^-65 return one
  after domain/special classification: even |log(x)| for finite nonzero binary64
  x cannot move that result across a rounding boundary. The existing RunMat
  wrapper retains ownership of its NaN-first and complex-domain policy; this
  kernel itself has the upstream pow exceptional-case order.
- `inverse_trig.wgsl`: asin.rs/acos.rs rational kernels and atan.rs regional
  reduction/polynomial; atan2.rs quadrant and exceptional-case logic. atan2's
  ratio is formed from normal mantissas plus an exponent difference, with
  integer scaling/encoded sign for tiny ratios. Tiny asin/atan return the input
  encoding directly. Native sqrt is a permitted instruction, whose precision
  belongs to the qualified provider arithmetic contract.
- `hyperbolic.wgsl`: sinh/cosh/tanh/asinh/acosh/atanh regional formulas,
  together with expo2.rs/k_expo2.rs split exponential scaling. Tiny odd
  functions return their original input encoding directly.
  Acosh additionally admits its real domain by encoded classification before
  region arithmetic: NaNs propagate by class, negative inputs and inputs below
  one return canonical NaN, one returns +0 and positive infinity returns itself.
  This intentionally strengthens upstream's assumption that called functions
  reject out-of-domain inputs: actual T4/MPFR qualification found 16 negative
  inputs where intermediate cancellation instead produced finite values or
  -infinity. Formulas for finite x > 1 are unchanged.

All files replace host-only dispatch/macros with explicit WGSL, rename globals,
and omit floating exception-flag forcing (shaders expose no such flags).
Exceptional outputs are encoded where a zero-divide expression would otherwise
be used. NaN payload/sign preservation is not promised. Signed zero is preserved
for the odd functions and atan2 quadrant cases. No algorithm relies on true FMA.
Parenthesization alone is not a compiler-order guarantee: the production
preparation boundary must emit NoContraction plus the supported RTE and
SignedZeroInfNanPreserve binary64 modes and qualify native arithmetic.
These ports intentionally do not require a DenormPreserve execution mode:
the T4 witness does not support that mode for f64.

This manifest records source provenance and implementation assumptions, **not**
a claim of device numerical acceptance or universal correct rounding. Upstream
libm includes documented errors above one ULP in some inverse-hyperbolic
intervals (up to two ULP in tanh/acosh); existing RunMat acceptance tolerances
must not be weakened. Independent oracle and actual-device tests remain required.

## Licenses and notices

The pinned libm LICENSE.txt states the library as a whole is MIT, and preserves
per-file notices from compatible third-party sources. Original notices relevant
to these translations are retained below and in corresponding WGSL files.

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

Sun-derived log2, expm1, log1p, asin, acos, atan and atan2:

Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
Developed at SunPro, a Sun Microsystems, Inc. business.
Permission to use, copy, modify, and distribute this software is freely
granted, provided that this notice is preserved.

Sun-derived pow:

Copyright (C) 2004 by Sun Microsystems, Inc. All rights reserved.
Permission to use, copy, modify, and distribute this software is freely
granted, provided that this notice is preserved.

FreeBSD-derived exp2:

Copyright (c) 2005 David Schultz <das@FreeBSD.ORG>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
SUCH DAMAGE.
