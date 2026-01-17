//! MATLAB-compatible `linsolve` builtin with structural hints and GPU-aware fallbacks.

use nalgebra::{linalg::SVD, DMatrix};
use num_complex::Complex64;
use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, HostTensorView, ProviderLinsolveOptions, ProviderLinsolveResult,
};
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{
    gpu_helpers,
    linalg::{diagonal_rcond, singular_value_rcond},
    tensor,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeControlFlow};

const NAME: &str = "linsolve";

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "linsolve",
        builtin_path = "crate::builtins::math::linalg::solve::linsolve"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "linsolve"
category: "math/linalg/solve"
keywords: ["linsolve", "linear solve", "triangular system", "posdef", "gpu"]
summary: "Solve linear systems A * X = B with optional structural hints (triangular, symmetric, positive-definite, or transposed)."
references: ["https://www.mathworks.com/help/matlab/ref/linsolve.html"]
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Prefers the accel provider's linsolve hook; the current WGPU backend downloads operands to the host, runs the shared solver, then re-uploads the result to preserve GPU residency."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 2
  constants: "uniform"
requires_feature: 'wgpu'
tested:
  unit: "builtins::math::linalg::solve::linsolve::tests"
  gpu: "builtins::math::linalg::solve::linsolve::tests::gpu_round_trip_matches_cpu"
  wgpu: "builtins::math::linalg::solve::linsolve::tests::wgpu_round_trip_matches_cpu"
  doc: "builtins::math::linalg::solve::linsolve::tests::doc_examples_present"
---

# What does the `linsolve` function do in MATLAB / RunMat?
`X = linsolve(A, B)` solves the linear system `A * X = B`. The optional `opts` structure lets you
declare that `A` is lower- or upper-triangular, symmetric, positive-definite, rectangular, or that
the transposed system should be solved instead. These hints mirror MATLAB and allow the runtime to
skip unnecessary factorizations.

## How does the `linsolve` function behave in MATLAB / RunMat?
- Inputs must behave like 2-D matrices (trailing singleton dimensions are accepted). `size(A, 1)` must
  match `size(B, 1)` after accounting for `opts.TRANSA`.
- When `opts.LT` or `opts.UT` are supplied, `linsolve` performs forward/back substitution instead of a
  full factorization. Singular pivots trigger the MATLAB error `"linsolve: matrix is singular to working precision."`
- `opts.TRANSA = 'T'` or `'C'` solves `Aᵀ * X = B` (conjugate transpose for complex matrices).
- `opts.POSDEF` and `opts.SYM` are accepted for compatibility; the current implementation still falls
  back to the SVD-based dense solver when a specialised route is not yet wired in.
- The optional second output `[X, rcond_est] = linsolve(...)` (exposed via the VM multi-output path)
  returns the estimated reciprocal condition number used to honour `opts.RCOND`.
- Logical and integer inputs are promoted to double precision. Complex inputs are handled in complex
  arithmetic.

## `linsolve` GPU execution behaviour
When a gpuArray provider is active, RunMat offers the solve to its `linsolve` hook. The current WGPU
backend downloads the operands to the host, executes the shared CPU solver, and uploads the result
back to the device so downstream kernels retain their residency. If no provider is registered—or a
provider declines the hook—RunMat gathers inputs to the host and returns a host tensor.

## Examples of using the `linsolve` function in MATLAB / RunMat

### Solving a 2×2 linear system
```matlab
A = [4 -2; 1 3];
b = [6; 7];
x = linsolve(A, b);
```
Expected output:
```matlab
x =
     2
     1
```

### Using a lower-triangular hint
```matlab
L = [3 0 0; -1 2 0; 4 1 5];
b = [9; 1; 12];
opts.LT = true;
x = linsolve(L, b, opts);
```
Expected output:
```matlab
x =
     3
     2
     1
```

### Solving the transposed system
```matlab
A = [2 1 0; 0 3 4; 0 0 5];
b = [3; 11; 5];
opts.UT = true;
opts.TRANSA = 'T';
x = linsolve(A, b, opts);
```
Expected output:
```matlab
x =
     1
     2
     1
```

### Complex triangular solve
```matlab
U = [2+1i  -1i; 0  4-2i];
b = [3+2i; 7];
opts.UT = true;
x = linsolve(U, b, opts);
```
Expected output:
```matlab
x =
   2.0000 + 0.0000i
   1.7500 + 0.8750i
```

### Estimating the reciprocal condition number
```matlab
A = [1 1; 1 1+1e-12];
b = [2; 2+1e-12];
[x, rcond_est] = linsolve(A, b);
```
Expected output (up to small round-off):
```matlab
x =
     1
     1

rcond_est =
    4.4409e-12
```

## GPU residency in RunMat (Do I need `gpuArray`?)
No additional residency management is required. When both operands already reside on the GPU,
RunMat executes the provider's `linsolve` hook. The current WGPU backend gathers the data to the
host, runs the shared solver, and re-uploads the output automatically, so downstream GPU work keeps
its residency. Providers that implement an on-device kernel can execute entirely on the GPU without
any MATLAB-level changes.

## FAQ

### What happens if I pass both `opts.LT` and `opts.UT`?
RunMat raises the MATLAB error `"linsolve: LT and UT are mutually exclusive."`—a matrix cannot be
simultaneously strictly lower- and upper-triangular.

### Does `opts.TRANSA` accept lowercase characters?
Yes. `opts.TRANSA` is case-insensitive and accepts `'N'`, `'T'`, `'C'`, or their lowercase variants.
`'C'` and `'T'` are equivalent for real matrices; `'C'` takes the conjugate transpose for complex
matrices (mirroring MATLAB).

### How is `opts.RCOND` used?
`opts.RCOND` provides a lower bound on the acceptable reciprocal condition number. If the estimated
`rcond` falls below the requested threshold the builtin raises
`"linsolve: matrix is singular to working precision."`

### Do `opts.SYM` or `opts.POSDEF` change the algorithm today?
They are accepted for MATLAB compatibility. The current implementation still uses the dense SVD
solver when no specialised routine is wired in; future work will route positive-definite systems to
Cholesky-based kernels.

### Can I use higher-dimensional arrays?
Inputs must behave like matrices. Trailing singleton dimensions are permitted, but other higher-rank
arrays should be reshaped before calling `linsolve`, just like in MATLAB.

## See Also
[mldivide](./mldivide), [mrdivide](./mrdivide), [lu](./lu), [chol](./chol), [gpuArray](./gpuarray), [gather](./gather)
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::solve::linsolve")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "linsolve",
    op_kind: GpuOpKind::Custom("solve"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("linsolve")],
    constant_strategy: ConstantStrategy::UniformBuffer,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Prefers the provider linsolve hook; WGPU currently gathers to the host solver and re-uploads the result.",
};

fn builtin_error(message: impl Into<String>) -> RuntimeControlFlow {
    build_runtime_error(message).with_builtin(NAME).build().into()
}

fn map_control_flow(flow: RuntimeControlFlow) -> RuntimeControlFlow {
    match flow {
        RuntimeControlFlow::Suspend(pending) => RuntimeControlFlow::Suspend(pending),
        RuntimeControlFlow::Error(err) => {
            let mut builder = build_runtime_error(err.message()).with_builtin(NAME);
            if let Some(identifier) = err.identifier() {
                builder = builder.with_identifier(identifier.to_string());
            }
            if let Some(task_id) = err.context.task_id.clone() {
                builder = builder.with_task_id(task_id);
            }
            if !err.context.call_stack.is_empty() {
                builder = builder.with_call_stack(err.context.call_stack.clone());
            }
            if let Some(phase) = err.context.phase.clone() {
                builder = builder.with_phase(phase);
            }
            builder.with_source(err).build().into()
        }
    }
}

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::linalg::solve::linsolve"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "linsolve",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::UniformBuffer,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Linear solves are terminal operations and do not fuse with surrounding kernels.",
};

#[runtime_builtin(
    name = "linsolve",
    category = "math/linalg/solve",
    summary = "Solve A * X = B with structural hints such as LT, UT, POSDEF, or TRANSA.",
    keywords = "linsolve,linear system,triangular,gpu",
    accel = "linsolve",
    builtin_path = "crate::builtins::math::linalg::solve::linsolve"
)]
fn linsolve_builtin(lhs: Value, rhs: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let eval = evaluate_args(lhs, rhs, &rest)?;
    Ok(eval.solution())
}

/// Evaluate `linsolve`, returning both the solution and the estimated reciprocal condition number.
pub fn evaluate(lhs: Value, rhs: Value, options: SolveOptions) -> BuiltinResult<LinsolveEval> {
    if let Some(eval) = try_gpu_linsolve(&lhs, &rhs, &options)? {
        return Ok(eval);
    }

    let lhs_host = crate::dispatcher::gather_if_needed(&lhs).map_err(map_control_flow)?;
    let rhs_host = crate::dispatcher::gather_if_needed(&rhs).map_err(map_control_flow)?;
    let pair = coerce_numeric_pair(lhs_host, rhs_host)?;
    match pair {
        NumericPair::Real(lhs_r, rhs_r) => {
            let (solution, rcond) = solve_real(lhs_r, rhs_r, &options)?;
            Ok(LinsolveEval::new(
                tensor::tensor_into_value(solution),
                Some(rcond),
            ))
        }
        NumericPair::Complex(lhs_c, rhs_c) => {
            let (solution, rcond) = solve_complex(lhs_c, rhs_c, &options)?;
            Ok(LinsolveEval::new(
                Value::ComplexTensor(solution),
                Some(rcond),
            ))
        }
    }
}

/// Host implementation shared with acceleration providers that fall back to CPU execution.
pub fn linsolve_host_real_for_provider(
    lhs: &Tensor,
    rhs: &Tensor,
    options: &ProviderLinsolveOptions,
) -> BuiltinResult<(Tensor, f64)> {
    let opts = SolveOptions::from(options);
    solve_real(lhs.clone(), rhs.clone(), &opts)
}

/// Result wrapper that exposes both primary and secondary outputs.
#[derive(Clone)]
pub struct LinsolveEval {
    solution: Value,
    rcond: Option<f64>,
}

impl LinsolveEval {
    fn new(solution: Value, rcond: Option<f64>) -> Self {
        Self { solution, rcond }
    }

    /// Primary solution output.
    pub fn solution(&self) -> Value {
        self.solution.clone()
    }

    /// Estimated reciprocal condition number (second output).
    pub fn reciprocal_condition(&self) -> Value {
        match self.rcond {
            Some(r) => Value::Num(r),
            None => Value::Num(f64::NAN),
        }
    }
}

#[derive(Clone, Default)]
pub struct SolveOptions {
    lower: bool,
    upper: bool,
    rectangular: bool,
    transposed: bool,
    conjugate: bool,
    symmetric: bool,
    posdef: bool,
    rcond: Option<f64>,
}

impl From<&SolveOptions> for ProviderLinsolveOptions {
    fn from(opts: &SolveOptions) -> Self {
        Self {
            lower: opts.lower,
            upper: opts.upper,
            rectangular: opts.rectangular,
            transposed: opts.transposed,
            conjugate: opts.conjugate,
            symmetric: opts.symmetric,
            posdef: opts.posdef,
            rcond: opts.rcond,
        }
    }
}

impl From<&ProviderLinsolveOptions> for SolveOptions {
    fn from(opts: &ProviderLinsolveOptions) -> Self {
        Self {
            lower: opts.lower,
            upper: opts.upper,
            rectangular: opts.rectangular,
            transposed: opts.transposed,
            conjugate: opts.conjugate,
            symmetric: opts.symmetric,
            posdef: opts.posdef,
            rcond: opts.rcond,
        }
    }
}

fn options_from_rest(rest: &[Value]) -> BuiltinResult<SolveOptions> {
    match rest.len() {
        0 => Ok(SolveOptions::default()),
        1 => parse_options(&rest[0]),
        _ => Err(builtin_error("linsolve: too many input arguments")),
    }
}

/// Public helper for the VM multi-output surface.
pub fn evaluate_args(lhs: Value, rhs: Value, rest: &[Value]) -> BuiltinResult<LinsolveEval> {
    let options = options_from_rest(rest)?;
    evaluate(lhs, rhs, options)
}

fn try_gpu_linsolve(
    lhs: &Value,
    rhs: &Value,
    options: &SolveOptions,
) -> BuiltinResult<Option<LinsolveEval>> {
    let provider = match runmat_accelerate_api::provider() {
        Some(p) => p,
        None => return Ok(None),
    };

    if contains_complex(lhs) || contains_complex(rhs) {
        return Ok(None);
    }

    let mut lhs_operand = match prepare_gpu_operand(lhs, provider)? {
        Some(op) => op,
        None => return Ok(None),
    };
    let mut rhs_operand = match prepare_gpu_operand(rhs, provider)? {
        Some(op) => op,
        None => {
            release_operand(provider, &mut lhs_operand);
            return Ok(None);
        }
    };

    if is_scalar_handle(lhs_operand.handle()) || is_scalar_handle(rhs_operand.handle()) {
        release_operand(provider, &mut lhs_operand);
        release_operand(provider, &mut rhs_operand);
        return Ok(None);
    }

    let provider_opts: ProviderLinsolveOptions = options.into();
    let result = provider
        .linsolve(lhs_operand.handle(), rhs_operand.handle(), &provider_opts)
        .ok();

    release_operand(provider, &mut lhs_operand);
    release_operand(provider, &mut rhs_operand);

    if let Some(ProviderLinsolveResult {
        solution,
        reciprocal_condition,
    }) = result
    {
        let eval = LinsolveEval::new(Value::GpuTensor(solution), Some(reciprocal_condition));
        return Ok(Some(eval));
    }

    Ok(None)
}

fn parse_options(value: &Value) -> BuiltinResult<SolveOptions> {
    let struct_val = match value {
        Value::Struct(s) => s,
        other => return Err(builtin_error(format!("linsolve: opts must be a struct, got {other:?}"))),
    };
    let mut opts = SolveOptions::default();
    for (key, raw_value) in &struct_val.fields {
        let name = key.to_ascii_uppercase();
        match name.as_str() {
            "LT" => opts.lower = parse_bool_field("LT", raw_value)?,
            "UT" => opts.upper = parse_bool_field("UT", raw_value)?,
            "RECT" => opts.rectangular = parse_bool_field("RECT", raw_value)?,
            "SYM" => opts.symmetric = parse_bool_field("SYM", raw_value)?,
            "POSDEF" => opts.posdef = parse_bool_field("POSDEF", raw_value)?,
            "TRANSA" => {
                let transa = parse_transa(raw_value)?;
                opts.transposed = transa != TransposeMode::None;
                opts.conjugate = transa == TransposeMode::Conjugate;
            }
            "RCOND" => {
                let threshold = parse_scalar_f64("RCOND", raw_value)?;
                if threshold < 0.0 {
                    return Err(builtin_error("linsolve: RCOND must be non-negative"));
                }
                opts.rcond = Some(threshold);
            }
            other => return Err(builtin_error(format!("linsolve: unknown option '{other}'"))),
        }
    }
    if opts.lower && opts.upper {
        return Err(builtin_error("linsolve: LT and UT are mutually exclusive."));
    }
    Ok(opts)
}

fn parse_bool_field(name: &str, value: &Value) -> BuiltinResult<bool> {
    match value {
        Value::Bool(b) => Ok(*b),
        Value::Int(i) => Ok(!i.is_zero()),
        Value::Num(n) => Ok(*n != 0.0),
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => Ok(t.data[0] != 0.0),
        Value::LogicalArray(arr) if arr.len() == 1 => Ok(arr.data[0] != 0),
        other => Err(builtin_error(format!(
            "linsolve: option '{name}' must be logical or numeric, got {other:?}"
        ))),
    }
}

fn parse_scalar_f64(name: &str, value: &Value) -> BuiltinResult<f64> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => Ok(t.data[0]),
        other => Err(builtin_error(format!(
            "linsolve: option '{name}' must be a scalar numeric value, got {other:?}"
        ))),
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum TransposeMode {
    None,
    Transpose,
    Conjugate,
}

fn parse_transa(value: &Value) -> BuiltinResult<TransposeMode> {
    let text = tensor::value_to_string(value).ok_or_else(|| {
        builtin_error("linsolve: TRANSA must be a character vector or string scalar")
    })?;
    if text.is_empty() {
        return Err(builtin_error("linsolve: TRANSA cannot be empty"));
    }
    match text.trim().to_ascii_uppercase().as_str() {
        "N" => Ok(TransposeMode::None),
        "T" => Ok(TransposeMode::Transpose),
        "C" => Ok(TransposeMode::Conjugate),
        other => Err(builtin_error(format!(
            "linsolve: TRANSA must be 'N', 'T', or 'C', got '{other}'"
        ))),
    }
}

enum NumericInput {
    Real(Tensor),
    Complex(ComplexTensor),
}

enum NumericPair {
    Real(Tensor, Tensor),
    Complex(ComplexTensor, ComplexTensor),
}

fn coerce_numeric_pair(lhs: Value, rhs: Value) -> BuiltinResult<NumericPair> {
    let lhs_num = coerce_numeric(lhs)?;
    let rhs_num = coerce_numeric(rhs)?;
    match (lhs_num, rhs_num) {
        (NumericInput::Real(lhs_r), NumericInput::Real(rhs_r)) => {
            Ok(NumericPair::Real(lhs_r, rhs_r))
        }
        (NumericInput::Complex(lhs_c), NumericInput::Complex(rhs_c)) => {
            Ok(NumericPair::Complex(lhs_c, rhs_c))
        }
        (NumericInput::Complex(lhs_c), NumericInput::Real(rhs_r)) => {
            let rhs_c = promote_real_tensor(&rhs_r)?;
            Ok(NumericPair::Complex(lhs_c, rhs_c))
        }
        (NumericInput::Real(lhs_r), NumericInput::Complex(rhs_c)) => {
            let lhs_c = promote_real_tensor(&lhs_r)?;
            Ok(NumericPair::Complex(lhs_c, rhs_c))
        }
    }
}

fn coerce_numeric(value: Value) -> BuiltinResult<NumericInput> {
    match value {
        Value::Tensor(tensor) => {
            ensure_matrix_shape(NAME, &tensor.shape)?;
            Ok(NumericInput::Real(tensor))
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical).map_err(builtin_error)?;
            ensure_matrix_shape(NAME, &tensor.shape)?;
            Ok(NumericInput::Real(tensor))
        }
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1]).map_err(builtin_error)?;
            Ok(NumericInput::Real(tensor))
        }
        Value::Int(i) => {
            let tensor = Tensor::new(vec![i.to_f64()], vec![1, 1]).map_err(builtin_error)?;
            Ok(NumericInput::Real(tensor))
        }
        Value::Bool(b) => {
            let tensor = Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
                .map_err(builtin_error)?;
            Ok(NumericInput::Real(tensor))
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(builtin_error)?;
            Ok(NumericInput::Complex(tensor))
        }
        Value::ComplexTensor(ct) => {
            ensure_matrix_shape(NAME, &ct.shape)?;
            Ok(NumericInput::Complex(ct))
        }
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor(&handle).map_err(map_control_flow)?;
            ensure_matrix_shape(NAME, &tensor.shape)?;
            Ok(NumericInput::Real(tensor))
        }
        other => Err(builtin_error(format!(
            "{NAME}: unsupported input type {:?}; convert to numeric values first",
            other
        ))),
    }
}

fn contains_complex(value: &Value) -> bool {
    matches!(value, Value::Complex(_, _) | Value::ComplexTensor(_))
}

fn is_scalar_handle(handle: &GpuTensorHandle) -> bool {
    handle.shape.iter().copied().product::<usize>() == 1
}

struct PreparedOperand {
    handle: GpuTensorHandle,
    owned: bool,
}

impl PreparedOperand {
    fn borrowed(handle: &GpuTensorHandle) -> Self {
        Self {
            handle: handle.clone(),
            owned: false,
        }
    }

    fn owned(handle: GpuTensorHandle) -> Self {
        Self {
            handle,
            owned: true,
        }
    }

    fn handle(&self) -> &GpuTensorHandle {
        &self.handle
    }
}

fn prepare_gpu_operand(
    value: &Value,
    provider: &'static dyn AccelProvider,
) -> BuiltinResult<Option<PreparedOperand>> {
    match value {
        Value::GpuTensor(handle) => {
            if is_scalar_handle(handle) {
                Ok(None)
            } else {
                Ok(Some(PreparedOperand::borrowed(handle)))
            }
        }
        Value::Tensor(tensor) => {
            if tensor::is_scalar_tensor(tensor) {
                Ok(None)
            } else {
                let uploaded = upload_tensor(provider, tensor)?;
                Ok(Some(PreparedOperand::owned(uploaded)))
            }
        }
        Value::LogicalArray(logical) => {
            if logical.data.len() == 1 {
                Ok(None)
            } else {
                let tensor = tensor::logical_to_tensor(logical).map_err(builtin_error)?;
                let uploaded = upload_tensor(provider, &tensor)?;
                Ok(Some(PreparedOperand::owned(uploaded)))
            }
        }
        _ => Ok(None),
    }
}

fn upload_tensor(
    provider: &'static dyn AccelProvider,
    tensor: &Tensor,
) -> BuiltinResult<GpuTensorHandle> {
    let view = HostTensorView {
        data: &tensor.data,
        shape: &tensor.shape,
    };
    provider
        .upload(&view)
        .map_err(|e| builtin_error(format!("{NAME}: {e}")))
}

fn release_operand(provider: &'static dyn AccelProvider, operand: &mut PreparedOperand) {
    if operand.owned {
        let _ = provider.free(&operand.handle);
        operand.owned = false;
    }
}

fn solve_real(lhs: Tensor, rhs: Tensor, options: &SolveOptions) -> BuiltinResult<(Tensor, f64)> {
    let mut lhs_effective = lhs;
    let mut rhs_effective = rhs;
    let mut lower = options.lower;
    let mut upper = options.upper;

    if options.transposed {
        lhs_effective = transpose_tensor(&lhs_effective);
        if options.conjugate {
            conjugate_in_place(&mut lhs_effective);
        }
        if lower || upper {
            std::mem::swap(&mut lower, &mut upper);
        }
    }

    rhs_effective = normalize_rhs_tensor(rhs_effective, lhs_effective.rows())?;

    if lower {
        ensure_square(lhs_effective.rows(), lhs_effective.cols())?;
        let (solution, rcond) = forward_substitution_real(&lhs_effective, &rhs_effective)?;
        enforce_rcond(options, rcond)?;
        return Ok((solution, rcond));
    }

    if upper {
        ensure_square(lhs_effective.rows(), lhs_effective.cols())?;
        let (solution, rcond) = backward_substitution_real(&lhs_effective, &rhs_effective)?;
        enforce_rcond(options, rcond)?;
        return Ok((solution, rcond));
    }

    let (solution, rcond) = solve_general_real(&lhs_effective, &rhs_effective)?;
    enforce_rcond(options, rcond)?;
    Ok((solution, rcond))
}

fn solve_complex(
    lhs: ComplexTensor,
    rhs: ComplexTensor,
    options: &SolveOptions,
) -> BuiltinResult<(ComplexTensor, f64)> {
    let mut lhs_effective = lhs;
    let mut rhs_effective = rhs;
    let mut lower = options.lower;
    let mut upper = options.upper;

    if options.transposed {
        lhs_effective = transpose_complex(&lhs_effective);
        if options.conjugate {
            conjugate_complex_in_place(&mut lhs_effective);
        }
        if lower || upper {
            std::mem::swap(&mut lower, &mut upper);
        }
    }

    rhs_effective = normalize_rhs_complex(rhs_effective, lhs_effective.rows)?;

    if lower {
        ensure_square(lhs_effective.rows, lhs_effective.cols)?;
        let (solution, rcond) = forward_substitution_complex(&lhs_effective, &rhs_effective)?;
        enforce_rcond(options, rcond)?;
        return Ok((solution, rcond));
    }

    if upper {
        ensure_square(lhs_effective.rows, lhs_effective.cols)?;
        let (solution, rcond) = backward_substitution_complex(&lhs_effective, &rhs_effective)?;
        enforce_rcond(options, rcond)?;
        return Ok((solution, rcond));
    }

    let (solution, rcond) = solve_general_complex(&lhs_effective, &rhs_effective)?;
    enforce_rcond(options, rcond)?;
    Ok((solution, rcond))
}

fn forward_substitution_real(lhs: &Tensor, rhs: &Tensor) -> BuiltinResult<(Tensor, f64)> {
    let n = lhs.rows();
    let nrhs = rhs.data.len() / n;
    let mut solution = rhs.data.clone();
    let mut min_diag = f64::INFINITY;
    let mut max_diag = 0.0_f64;

    for col in 0..nrhs {
        for i in 0..n {
            let diag = lhs.data[i + i * n];
            let diag_abs = diag.abs();
            min_diag = min_diag.min(diag_abs);
            max_diag = max_diag.max(diag_abs);
            if diag_abs == 0.0 {
                return Err(builtin_error("linsolve: matrix is singular to working precision."));
            }
            let mut accum = 0.0;
            for j in 0..i {
                accum += lhs.data[i + j * n] * solution[j + col * n];
            }
            let rhs_value = solution[i + col * n] - accum;
            solution[i + col * n] = rhs_value / diag;
        }
    }

    let rcond = diagonal_rcond(min_diag, max_diag);
    let tensor = Tensor::new(solution, rhs.shape.clone()).map_err(|e| builtin_error(format!("{NAME}: {e}")))?;
    Ok((tensor, rcond))
}

fn backward_substitution_real(lhs: &Tensor, rhs: &Tensor) -> BuiltinResult<(Tensor, f64)> {
    let n = lhs.rows();
    let nrhs = rhs.data.len() / n;
    let mut solution = rhs.data.clone();
    let mut min_diag = f64::INFINITY;
    let mut max_diag = 0.0_f64;

    for col in 0..nrhs {
        for row_rev in 0..n {
            let i = n - 1 - row_rev;
            let diag = lhs.data[i + i * n];
            let diag_abs = diag.abs();
            min_diag = min_diag.min(diag_abs);
            max_diag = max_diag.max(diag_abs);
            if diag_abs == 0.0 {
                return Err(builtin_error("linsolve: matrix is singular to working precision."));
            }
            let mut accum = 0.0;
            for j in (i + 1)..n {
                accum += lhs.data[i + j * n] * solution[j + col * n];
            }
            let rhs_value = solution[i + col * n] - accum;
            solution[i + col * n] = rhs_value / diag;
        }
    }

    let rcond = diagonal_rcond(min_diag, max_diag);
    let tensor = Tensor::new(solution, rhs.shape.clone()).map_err(|e| builtin_error(format!("{NAME}: {e}")))?;
    Ok((tensor, rcond))
}

fn forward_substitution_complex(
    lhs: &ComplexTensor,
    rhs: &ComplexTensor,
) -> BuiltinResult<(ComplexTensor, f64)> {
    let n = lhs.rows;
    let nrhs = rhs.data.len() / n;
    let lhs_data: Vec<Complex64> = lhs
        .data
        .iter()
        .map(|&(re, im)| Complex64::new(re, im))
        .collect();
    let mut solution: Vec<Complex64> = rhs
        .data
        .iter()
        .map(|&(re, im)| Complex64::new(re, im))
        .collect();
    let mut min_diag = f64::INFINITY;
    let mut max_diag = 0.0_f64;

    for col in 0..nrhs {
        for i in 0..n {
            let diag = lhs_data[i + i * n];
            let diag_abs = diag.norm();
            min_diag = min_diag.min(diag_abs);
            max_diag = max_diag.max(diag_abs);
            if diag_abs == 0.0 {
                return Err(builtin_error("linsolve: matrix is singular to working precision."));
            }
            let mut accum = Complex64::new(0.0, 0.0);
            for j in 0..i {
                accum += lhs_data[i + j * n] * solution[j + col * n];
            }
            let rhs_value = solution[i + col * n] - accum;
            solution[i + col * n] = rhs_value / diag;
        }
    }

    let rcond = diagonal_rcond(min_diag, max_diag);
    let tensor = ComplexTensor::new(
        solution.iter().map(|c| (c.re, c.im)).collect(),
        rhs.shape.clone(),
    )
    .map_err(|e| builtin_error(format!("{NAME}: {e}")))?;
    Ok((tensor, rcond))
}

fn backward_substitution_complex(
    lhs: &ComplexTensor,
    rhs: &ComplexTensor,
) -> BuiltinResult<(ComplexTensor, f64)> {
    let n = lhs.rows;
    let nrhs = rhs.data.len() / n;
    let lhs_data: Vec<Complex64> = lhs
        .data
        .iter()
        .map(|&(re, im)| Complex64::new(re, im))
        .collect();
    let mut solution: Vec<Complex64> = rhs
        .data
        .iter()
        .map(|&(re, im)| Complex64::new(re, im))
        .collect();
    let mut min_diag = f64::INFINITY;
    let mut max_diag = 0.0_f64;

    for col in 0..nrhs {
        for row_rev in 0..n {
            let i = n - 1 - row_rev;
            let diag = lhs_data[i + i * n];
            let diag_abs = diag.norm();
            min_diag = min_diag.min(diag_abs);
            max_diag = max_diag.max(diag_abs);
            if diag_abs == 0.0 {
                return Err(builtin_error("linsolve: matrix is singular to working precision."));
            }
            let mut accum = Complex64::new(0.0, 0.0);
            for j in (i + 1)..n {
                accum += lhs_data[i + j * n] * solution[j + col * n];
            }
            let rhs_value = solution[i + col * n] - accum;
            solution[i + col * n] = rhs_value / diag;
        }
    }

    let rcond = diagonal_rcond(min_diag, max_diag);
    let tensor = ComplexTensor::new(
        solution.iter().map(|c| (c.re, c.im)).collect(),
        rhs.shape.clone(),
    )
    .map_err(|e| builtin_error(format!("{NAME}: {e}")))?;
    Ok((tensor, rcond))
}

fn solve_general_real(lhs: &Tensor, rhs: &Tensor) -> BuiltinResult<(Tensor, f64)> {
    let a = DMatrix::from_column_slice(lhs.rows(), lhs.cols(), &lhs.data);
    let b = DMatrix::from_column_slice(rhs.rows(), rhs.cols(), &rhs.data);
    let svd = SVD::new(a.clone(), true, true);
    let rcond = singular_value_rcond(svd.singular_values.as_slice());
    let tol = compute_svd_tolerance(svd.singular_values.as_slice(), lhs.rows(), lhs.cols());
    let solution = svd.solve(&b, tol).map_err(|e| builtin_error(format!("{NAME}: {e}")))?;
    let tensor = matrix_real_to_tensor(solution)?;
    Ok((tensor, rcond))
}

fn solve_general_complex(
    lhs: &ComplexTensor,
    rhs: &ComplexTensor,
) -> BuiltinResult<(ComplexTensor, f64)> {
    let a_data: Vec<Complex64> = lhs
        .data
        .iter()
        .map(|&(re, im)| Complex64::new(re, im))
        .collect();
    let b_data: Vec<Complex64> = rhs
        .data
        .iter()
        .map(|&(re, im)| Complex64::new(re, im))
        .collect();
    let a = DMatrix::from_column_slice(lhs.rows, lhs.cols, &a_data);
    let b = DMatrix::from_column_slice(rhs.rows, rhs.cols, &b_data);
    let svd = SVD::new(a.clone(), true, true);
    let rcond = singular_value_rcond(svd.singular_values.as_slice());
    let tol = compute_svd_tolerance(svd.singular_values.as_slice(), lhs.rows, lhs.cols);
    let solution = svd.solve(&b, tol).map_err(|e| builtin_error(format!("{NAME}: {e}")))?;
    let tensor = matrix_complex_to_tensor(solution)?;
    Ok((tensor, rcond))
}

fn normalize_rhs_tensor(rhs: Tensor, expected_rows: usize) -> BuiltinResult<Tensor> {
    if rhs.rows() == expected_rows {
        return Ok(rhs);
    }
    if rhs.shape.len() == 1 && rhs.shape[0] == expected_rows {
        return Tensor::new(rhs.data, vec![expected_rows, 1]).map_err(|e| builtin_error(format!("{NAME}: {e}")));
    }
    if rhs.data.is_empty() && expected_rows == 0 {
        return Ok(rhs);
    }
    Err(builtin_error("Matrix dimensions must agree."))
}

fn normalize_rhs_complex(
    rhs: ComplexTensor,
    expected_rows: usize,
) -> BuiltinResult<ComplexTensor> {
    if rhs.rows == expected_rows {
        return Ok(rhs);
    }
    if rhs.shape.len() == 1 && rhs.shape[0] == expected_rows {
        return ComplexTensor::new(rhs.data, vec![expected_rows, 1])
            .map_err(|e| builtin_error(format!("{NAME}: {e}")));
    }
    if rhs.data.is_empty() && expected_rows == 0 {
        return Ok(rhs);
    }
    Err(builtin_error("Matrix dimensions must agree."))
}

fn enforce_rcond(options: &SolveOptions, rcond: f64) -> BuiltinResult<()> {
    if let Some(threshold) = options.rcond {
        if rcond < threshold {
            return Err(builtin_error("linsolve: matrix is singular to working precision."));
        }
    }
    Ok(())
}

fn compute_svd_tolerance(singular_values: &[f64], rows: usize, cols: usize) -> f64 {
    let max_sv = singular_values
        .iter()
        .copied()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let max_dim = rows.max(cols) as f64;
    f64::EPSILON * max_dim * max_sv.max(1.0)
}

fn matrix_real_to_tensor(matrix: DMatrix<f64>) -> BuiltinResult<Tensor> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    Tensor::new(matrix.as_slice().to_vec(), vec![rows, cols]).map_err(|e| builtin_error(format!("{NAME}: {e}")))
}

fn matrix_complex_to_tensor(matrix: DMatrix<Complex64>) -> BuiltinResult<ComplexTensor> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    let data: Vec<(f64, f64)> = matrix.as_slice().iter().map(|c| (c.re, c.im)).collect();
    ComplexTensor::new(data, vec![rows, cols]).map_err(|e| builtin_error(format!("{NAME}: {e}")))
}

fn promote_real_tensor(tensor: &Tensor) -> BuiltinResult<ComplexTensor> {
    let data: Vec<(f64, f64)> = tensor.data.iter().map(|&re| (re, 0.0)).collect();
    ComplexTensor::new(data, tensor.shape.clone()).map_err(|e| builtin_error(format!("{NAME}: {e}")))
}

fn ensure_matrix_shape(name: &str, shape: &[usize]) -> BuiltinResult<()> {
    if is_effectively_matrix(shape) {
        Ok(())
    } else {
        Err(builtin_error(format!("{name}: inputs must be 2-D matrices or vectors")))
    }
}

fn is_effectively_matrix(shape: &[usize]) -> bool {
    match shape.len() {
        0..=2 => true,
        _ => shape.iter().skip(2).all(|&dim| dim == 1),
    }
}

fn ensure_square(rows: usize, cols: usize) -> BuiltinResult<()> {
    if rows == cols {
        Ok(())
    } else {
        Err(builtin_error("linsolve: triangular solves require a square coefficient matrix."))
    }
}

fn transpose_tensor(tensor: &Tensor) -> Tensor {
    let rows = tensor.rows();
    let cols = tensor.cols();
    let mut data = vec![0.0; tensor.data.len()];
    for r in 0..rows {
        for c in 0..cols {
            data[c + r * cols] = tensor.data[r + c * rows];
        }
    }
    Tensor::new(data, vec![cols, rows]).expect("transpose_tensor valid")
}

fn transpose_complex(tensor: &ComplexTensor) -> ComplexTensor {
    let rows = tensor.rows;
    let cols = tensor.cols;
    let mut data = vec![(0.0, 0.0); tensor.data.len()];
    for r in 0..rows {
        for c in 0..cols {
            data[c + r * cols] = tensor.data[r + c * rows];
        }
    }
    ComplexTensor::new(data, vec![cols, rows]).expect("transpose_complex valid")
}

fn conjugate_in_place(_tensor: &mut Tensor) {
    // Real-valued matrices are unaffected by conjugation.
}

fn conjugate_complex_in_place(tensor: &mut ComplexTensor) {
    for value in &mut tensor.data {
        value.1 = -value.1;
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use runmat_accelerate_api::HostTensorView;
    use crate::RuntimeControlFlow;
    use runmat_builtins::{CharArray, StructValue, Tensor};

    fn unwrap_error(flow: crate::RuntimeControlFlow) -> crate::RuntimeError {
        match flow {
            RuntimeControlFlow::Error(err) => err,
            RuntimeControlFlow::Suspend(_) => panic!("unexpected suspend"),
        }
    }

    fn approx_eq(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-12);
    }

    use crate::builtins::common::test_support;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linsolve_basic_square() {
        let a = Tensor::new(vec![2.0, 1.0, 1.0, 2.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![4.0, 5.0], vec![2, 1]).unwrap();
        let result =
            linsolve_builtin(Value::Tensor(a), Value::Tensor(b), Vec::new()).expect("linsolve");
        let t = test_support::gather(result).expect("gather");
        assert_eq!(t.shape, vec![2, 1]);
        approx_eq(t.data[0], 1.0);
        approx_eq(t.data[1], 2.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linsolve_lower_triangular_hint() {
        let a = Tensor::new(
            vec![3.0, -1.0, 4.0, 0.0, 2.0, 1.0, 0.0, 0.0, 5.0],
            vec![3, 3],
        )
        .unwrap();
        let b = Tensor::new(vec![9.0, 1.0, 19.0], vec![3, 1]).unwrap();
        let mut opts = StructValue::new();
        opts.fields.insert("LT".to_string(), Value::Bool(true));
        let result = linsolve_builtin(
            Value::Tensor(a),
            Value::Tensor(b),
            vec![Value::Struct(opts)],
        )
        .expect("linsolve");
        let tensor = test_support::gather(result).expect("gather");
        assert_eq!(tensor.shape, vec![3, 1]);
        approx_eq(tensor.data[0], 3.0);
        approx_eq(tensor.data[1], 2.0);
        approx_eq(tensor.data[2], 1.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linsolve_transposed_triangular_hint() {
        let a = Tensor::new(
            vec![3.0, 1.0, 0.0, 0.0, 4.0, 2.0, 0.0, 0.0, 5.0],
            vec![3, 3],
        )
        .unwrap();
        let b = Tensor::new(vec![5.0, 14.0, 23.0], vec![3, 1]).unwrap();
        let mut opts = StructValue::new();
        opts.fields.insert("LT".to_string(), Value::Bool(true));
        opts.fields.insert(
            "TRANSA".to_string(),
            Value::CharArray(CharArray::new_row("T")),
        );

        let result = linsolve_builtin(
            Value::Tensor(a.clone()),
            Value::Tensor(b.clone()),
            vec![Value::Struct(opts)],
        )
        .expect("linsolve");
        let tensor = test_support::gather(result).expect("gather");
        assert_eq!(tensor.shape, vec![3, 1]);

        let a_transposed = transpose_tensor(&a);
        let reference = super::evaluate(
            Value::Tensor(a_transposed.clone()),
            Value::Tensor(b.clone()),
            SolveOptions::default(),
        )
        .expect("reference");
        let expected_tensor = test_support::gather(reference.solution()).expect("gather ref");

        for (actual, expected) in tensor.data.iter().zip(expected_tensor.data.iter()) {
            approx_eq(*actual, *expected);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linsolve_rcond_enforced() {
        let a = Tensor::new(vec![1.0, 1.0, 1.0, 1.0 + 1e-12], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![2.0, 2.0 + 1e-12], vec![2, 1]).unwrap();
        let mut opts = StructValue::new();
        opts.fields.insert("RCOND".to_string(), Value::Num(1e-3));
        let err = unwrap_error(
            linsolve_builtin(
                Value::Tensor(a),
                Value::Tensor(b),
                vec![Value::Struct(opts)],
            )
            .expect_err("singular matrix must fail"),
        );
        assert!(
            err.message().contains("singular to working precision"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linsolve_recovers_rcond_output() {
        let a = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let eval = super::evaluate_args(Value::Tensor(a.clone()), Value::Tensor(b.clone()), &[])
            .expect("evaluate");
        let solution_tensor = match eval.solution() {
            Value::Tensor(sol) => sol.clone(),
            Value::GpuTensor(handle) => {
                test_support::gather(Value::GpuTensor(handle.clone())).expect("gather solution")
            }
            other => panic!("unexpected solution value {other:?}"),
        };
        assert_eq!(solution_tensor.shape, vec![2, 1]);
        approx_eq(solution_tensor.data[0], 1.0);
        approx_eq(solution_tensor.data[1], 2.0);

        let rcond_value = match eval.reciprocal_condition() {
            Value::Num(r) => r,
            Value::GpuTensor(handle) => {
                let gathered =
                    test_support::gather(Value::GpuTensor(handle.clone())).expect("gather rcond");
                gathered.data[0]
            }
            other => panic!("unexpected rcond value {other:?}"),
        };
        approx_eq(rcond_value, 1.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_round_trip_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let a = Tensor::new(vec![2.0, 1.0, 1.0, 3.0], vec![2, 2]).unwrap();
            let b = Tensor::new(vec![4.0, 5.0], vec![2, 1]).unwrap();

            let cpu = linsolve_builtin(
                Value::Tensor(a.clone()),
                Value::Tensor(b.clone()),
                Vec::new(),
            )
            .expect("cpu linsolve");
            let cpu_tensor = test_support::gather(cpu).expect("cpu gather");

            let view_a = HostTensorView {
                data: &a.data,
                shape: &a.shape,
            };
            let view_b = HostTensorView {
                data: &b.data,
                shape: &b.shape,
            };
            let ha = provider.upload(&view_a).expect("upload A");
            let hb = provider.upload(&view_b).expect("upload B");

            let gpu_value = linsolve_builtin(
                Value::GpuTensor(ha.clone()),
                Value::GpuTensor(hb.clone()),
                Vec::new(),
            )
            .expect("gpu linsolve");
            let gathered = test_support::gather(gpu_value).expect("gather");
            let _ = provider.free(&ha);
            let _ = provider.free(&hb);

            assert_eq!(gathered.shape, cpu_tensor.shape);
            for (gpu, cpu) in gathered.data.iter().zip(cpu_tensor.data.iter()) {
                assert!((gpu - cpu).abs() < 1e-12);
            }
        });
    }

    #[cfg(feature = "wgpu")]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn wgpu_round_trip_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };

        let a = Tensor::new(vec![3.0, 1.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![7.0, 8.0], vec![2, 1]).unwrap();

        let cpu = linsolve_builtin(
            Value::Tensor(a.clone()),
            Value::Tensor(b.clone()),
            Vec::new(),
        )
        .expect("cpu linsolve");
        let cpu_tensor = test_support::gather(cpu).expect("cpu gather");

        let view_a = HostTensorView {
            data: &a.data,
            shape: &a.shape,
        };
        let view_b = HostTensorView {
            data: &b.data,
            shape: &b.shape,
        };
        let ha = provider.upload(&view_a).expect("upload A");
        let hb = provider.upload(&view_b).expect("upload B");
        let gpu_value = linsolve_builtin(
            Value::GpuTensor(ha.clone()),
            Value::GpuTensor(hb.clone()),
            Vec::new(),
        )
        .expect("gpu linsolve");
        let gathered = test_support::gather(gpu_value).expect("gather");
        let _ = provider.free(&ha);
        let _ = provider.free(&hb);

        assert_eq!(gathered.shape, cpu_tensor.shape);
        for (gpu, cpu) in gathered.data.iter().zip(cpu_tensor.data.iter()) {
            assert!((gpu - cpu).abs() < tol);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
