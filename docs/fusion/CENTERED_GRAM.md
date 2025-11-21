# Centered Gram / Covariance Fusion

This fusion targets covariance or Gram-matrix construction where a tall matrix is mean-centered, multiplied by its transpose, and scaled by `n` or `n-1`.

## What Qualifies

- **Pattern:** `mean(X, 1)` → `X - mean` → `(X - mean)' * (X - mean)` → division by a scalar (`n-1` or `n`).
- **Transpose + mul pairing.** One operand of the multiplication must be a transpose node whose input matches the centred matrix.
- **Scalar divisor.** The denominator must be a compile-time scalar constant so the planner can identify biased vs unbiased covariance.
- **Exclusive ownership.** All nodes involved (mean, subtraction, transpose, multiplication, division) must not already belong to another fusion group.

## Why Centered Gram?

- Centered Gram is a common pattern in linear algebra and statistics. It shows up in covariance matrix construction, principal component analysis, and other linear algebra operations
- By fusing this pattern, we can keep the Gram matrix resident on the GPU, avoiding the overhead of uploading and downloading it to the CPU.

## Benefits

- **Zero host traffic.** Neither the centred matrix nor the Gram matrix round-trips through the CPU.
- **Fewer GPU launches.** We can consolidate multiple covariance matrix constructions into a single kernel, reducing the number of GPU launches.

## Known Limitations

- Only the canonical layout (mean → subtraction → transpose → matmul → scalar divide) is detected today.
- Weighted covariance or per-column scaling is not yet fused.
- Very large matrices may still trigger size guardrails in `provider_impl.rs`, forcing a fallback to the CPU implementation.

If your program follows the pattern above and still falls back, confirm that the divisor literal matches either `size(X,1)` or `size(X,1)-1` exactly; otherwise the planner cannot classify the normalization.