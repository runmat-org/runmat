# Binary64 elementary-function sources

The algorithm source is the existing locked `libm` 0.2.16 dependency, registry checksum `b6d2cec3eae94f9f509c767b45932f1ada8350c4bdb85af2fcab4a3c14807981`. Translations retain the source algorithms' per-file permission notices. Floating-point exception flags are outside the shader interface; encoded special values, arithmetic ordering and numerical output remain relevant.

| Translation | Upstream file | Upstream SHA256 |
| --- | --- | --- |
| `exp.wgsl` | `src/math/exp.rs` | `1d486e847e1752fb9ce74a714096a11adcac935dacd4f17625ff63819103a80d` |
| `log.wgsl` | `src/math/log.rs` | `df5f4d1325e89621bfcd9cf7788d0d7e20a6ef7f429110fe35b370b3c9362615` |

`scaling.wgsl` packs IEEE binary64 values with integer arithmetic, including ties-to-even subnormal rounding. It does not depend on a floating-point subnormal intermediate.

The algorithms require a qualified binary64 arithmetic contract; their presence alone does not establish that a backend preserves evaluation order or required primitive precision. Near-rounded libm algorithms are not a blanket correctly-rounded guarantee. Keep the upstream notices in distributed shader source and include this permission with distributions containing these translations.

## libm permission

rust-lang/libm as a whole is available for use under the MIT license:

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
