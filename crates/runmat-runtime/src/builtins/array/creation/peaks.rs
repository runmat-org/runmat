//! MATLAB-compatible `peaks` builtin for RunMat.
//!
//! `peaks` is a sample-data function that evaluates a well-known 3-D test surface over an n×n
//! grid spanning [-3, 3] × [-3, 3].  The formula is:
//!
//! ```text
//! Z = 3*(1-x)^2 * exp(-x^2 - (y+1)^2)
//!     - 10*(x/5 - x^3 - y^5) * exp(-x^2 - y^2)
//!     - 1/3 * exp(-(x+1)^2 - y^2)
//! ```
//!
//! Call forms
//! ----------
//! * `peaks`        – 49×49 Z matrix (MATLAB default)
//! * `peaks(n)`     – n×n Z matrix over the standard [-3,3] grid
//! * `peaks(X, Y)`  – evaluate at caller-supplied coordinate matrices
//! * `[X,Y,Z] = peaks(…)` – also return the coordinate matrices
//!
//! GPU acceleration
//! ----------------
//! When a GPU provider is active the `peaks(n)` and `peaks(X_gpu, Y_gpu)` forms
//! dispatch to dedicated WGSL compute shaders that evaluate the formula directly
//! on the device.  The `[X, Y, Z] = peaks(n)` multi-output form constructs X and
//! Y via the existing meshgrid GPU path and obtains Z from the peaks shader.

use runmat_builtins::{ResolveContext, Tensor, Type, Value};
use runmat_macros::runtime_builtin;

use crate::build_runtime_error;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};

const DEFAULT_N: usize = 49;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::creation::peaks")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "peaks",
    op_kind: GpuOpKind::Custom("array_construct"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("peaks"), ProviderHook::Custom("peaks_xy")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "The peaks(n) and peaks(X,Y) forms dispatch to dedicated WGSL shaders. The host path is used as fallback and for multi-output [X,Y,Z] coordinate grids.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::creation::peaks")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "peaks",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "peaks materialises a dense matrix and is not fused with other operations.",
};

fn builtin_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message).with_builtin("peaks").build()
}

fn peaks_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::tensor()
}

#[runtime_builtin(
    name = "peaks",
    category = "array/creation",
    summary = "Sample data: 3-D test surface on an n-by-n grid.",
    keywords = "peaks,sample,surface,test,demo",
    accel = "array_construct",
    type_resolver(peaks_type),
    builtin_path = "crate::builtins::array::creation::peaks"
)]
async fn peaks_builtin(rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let out_count = crate::output_count::current_output_count();

    if matches!(out_count, Some(n) if n > 3) {
        return Err(builtin_error(
            "peaks: too many output arguments; maximum is 3",
        ));
    }

    match rest.len() {
        // peaks  or  peaks()
        0 => peaks_from_n(DEFAULT_N, out_count).await,

        // peaks(n)
        1 => {
            let n = parse_scalar_n(&rest[0]).await?;
            peaks_from_n(n, out_count).await
        }

        // peaks(X, Y)
        2 => peaks_from_xy(&rest[0], &rest[1], out_count).await,

        _ => Err(builtin_error("peaks: expected 0, 1, or 2 input arguments")),
    }
}

/// Generate the peaks surface for an n×n standard grid, trying the GPU provider first.
async fn peaks_from_n(n: usize, out_count: Option<usize>) -> crate::BuiltinResult<Value> {
    let wants_multi = matches!(out_count, Some(2) | Some(3));

    // Single-Z case: try the GPU peaks shader directly.
    if !wants_multi {
        if let Some(provider) = runmat_accelerate_api::provider() {
            if let Ok(handle) = provider.peaks(n) {
                return Ok(Value::GpuTensor(handle));
            }
        }
    }

    // Host computation (also used for multi-output [X, Y, Z]).
    let (x_flat, y_flat) = make_axis(n);
    let (x_mat, y_mat) = make_grids(&x_flat, &y_flat, n, n);
    let z_mat = compute_z(&x_mat, &y_mat, n, n);

    // For multi-output, try to upload all three matrices to the GPU.
    if wants_multi {
        if let Some(provider) = runmat_accelerate_api::provider() {
            let shape = [n, n];
            let x_view = runmat_accelerate_api::HostTensorView {
                data: &x_mat,
                shape: &shape,
            };
            let y_view = runmat_accelerate_api::HostTensorView {
                data: &y_mat,
                shape: &shape,
            };
            let z_view = runmat_accelerate_api::HostTensorView {
                data: &z_mat,
                shape: &shape,
            };
            if let (Ok(xh), Ok(yh), Ok(zh)) = (
                provider.upload(&x_view),
                provider.upload(&y_view),
                provider.upload(&z_view),
            ) {
                return Ok(match out_count {
                    Some(3) => Value::OutputList(vec![
                        Value::GpuTensor(xh),
                        Value::GpuTensor(yh),
                        Value::GpuTensor(zh),
                    ]),
                    _ => Value::OutputList(vec![Value::GpuTensor(xh), Value::GpuTensor(yh)]),
                });
            }
        }
    }

    build_output(x_mat, y_mat, z_mat, n, n, out_count)
}

/// Evaluate the peaks formula at caller-supplied X and Y values.
async fn peaks_from_xy(
    x_val: &Value,
    y_val: &Value,
    out_count: Option<usize>,
) -> crate::BuiltinResult<Value> {
    // GPU fast path: both inputs already on device.
    if let (Value::GpuTensor(x_handle), Value::GpuTensor(y_handle)) = (x_val, y_val) {
        if let Some(provider) = runmat_accelerate_api::provider() {
            if let Ok(z_handle) = provider.peaks_xy(x_handle, y_handle) {
                return match out_count {
                    Some(3) => Ok(Value::OutputList(vec![
                        Value::GpuTensor(x_handle.clone()),
                        Value::GpuTensor(y_handle.clone()),
                        Value::GpuTensor(z_handle),
                    ])),
                    Some(2) => Ok(Value::OutputList(vec![
                        Value::GpuTensor(x_handle.clone()),
                        Value::GpuTensor(y_handle.clone()),
                    ])),
                    _ => Ok(Value::GpuTensor(z_handle)),
                };
            }
        }
        // Provider absent or failed: gather to host and continue.
        let x_tensor = gpu_helpers::gather_tensor_async(x_handle).await?;
        let y_tensor = gpu_helpers::gather_tensor_async(y_handle).await?;
        validate_xy_shapes(&x_tensor, &y_tensor)?;
        let (rows, cols) = matrix_shape(&x_tensor)?;
        let z_mat = compute_z(&x_tensor.data, &y_tensor.data, rows, cols);
        return build_output(x_tensor.data, y_tensor.data, z_mat, rows, cols, out_count);
    }

    // Host path.
    let x_tensor = gather_tensor(x_val).await?;
    let y_tensor = gather_tensor(y_val).await?;
    validate_xy_shapes(&x_tensor, &y_tensor)?;
    let (rows, cols) = matrix_shape(&x_tensor)?;
    let z_mat = compute_z(&x_tensor.data, &y_tensor.data, rows, cols);
    build_output(x_tensor.data, y_tensor.data, z_mat, rows, cols, out_count)
}

// ---------------------------------------------------------------------------
// Grid construction
// ---------------------------------------------------------------------------

/// Linearly-spaced values from -3 to 3 (n points) for both axes.
fn make_axis(n: usize) -> (Vec<f64>, Vec<f64>) {
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    if n == 1 {
        return (vec![3.0], vec![3.0]);
    }
    let axis: Vec<f64> = (0..n)
        .map(|i| -3.0 + 6.0 * (i as f64) / ((n - 1) as f64))
        .collect();
    (axis.clone(), axis)
}

/// Build flat X and Y coordinate matrices stored column-major.
///
/// meshgrid(x_axis, y_axis): X[row,col] = x_axis[col], Y[row,col] = y_axis[row].
/// Column-major: element (row, col) lives at index `row + col * rows`.
fn make_grids(x_axis: &[f64], y_axis: &[f64], rows: usize, cols: usize) -> (Vec<f64>, Vec<f64>) {
    let size = rows * cols;
    let mut x_mat = vec![0.0f64; size];
    let mut y_mat = vec![0.0f64; size];
    for col in 0..cols {
        for row in 0..rows {
            x_mat[row + col * rows] = *x_axis.get(col).unwrap_or(&0.0);
            y_mat[row + col * rows] = *y_axis.get(row).unwrap_or(&0.0);
        }
    }
    (x_mat, y_mat)
}

// ---------------------------------------------------------------------------
// Surface formula
// ---------------------------------------------------------------------------

#[inline]
fn peaks_at(x: f64, y: f64) -> f64 {
    3.0 * (1.0 - x).powi(2) * (-(x.powi(2)) - (y + 1.0).powi(2)).exp()
        - 10.0 * (x / 5.0 - x.powi(3) - y.powi(5)) * (-(x.powi(2)) - y.powi(2)).exp()
        - 1.0 / 3.0 * (-(x + 1.0).powi(2) - y.powi(2)).exp()
}

fn compute_z(x_mat: &[f64], y_mat: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let _ = (rows, cols);
    x_mat
        .iter()
        .zip(y_mat.iter())
        .map(|(&x, &y)| peaks_at(x, y))
        .collect()
}

// ---------------------------------------------------------------------------
// Output helpers
// ---------------------------------------------------------------------------

fn build_output(
    x_flat: Vec<f64>,
    y_flat: Vec<f64>,
    z_flat: Vec<f64>,
    rows: usize,
    cols: usize,
    out_count: Option<usize>,
) -> crate::BuiltinResult<Value> {
    let shape = vec![rows, cols];

    match out_count {
        // Caller explicitly requested 3 outputs: [X, Y, Z] = peaks(…)
        Some(3) => {
            let x_val = make_tensor(x_flat, shape.clone())?;
            let y_val = make_tensor(y_flat, shape.clone())?;
            let z_val = make_tensor(z_flat, shape)?;
            Ok(Value::OutputList(vec![x_val, y_val, z_val]))
        }
        // Caller requested 2 outputs: [X, Y] = peaks(…) — unusual but allowed
        Some(2) => {
            let x_val = make_tensor(x_flat, shape.clone())?;
            let y_val = make_tensor(y_flat, shape)?;
            Ok(Value::OutputList(vec![x_val, y_val]))
        }
        // Default single output: Z = peaks(…)
        _ => make_tensor(z_flat, shape),
    }
}

fn make_tensor(data: Vec<f64>, shape: Vec<usize>) -> crate::BuiltinResult<Value> {
    if shape.contains(&0) {
        return Tensor::new(Vec::new(), shape)
            .map(tensor::tensor_into_value)
            .map_err(|e| builtin_error(format!("peaks: {e}")));
    }
    Tensor::new(data, shape)
        .map(tensor::tensor_into_value)
        .map_err(|e| builtin_error(format!("peaks: {e}")))
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

async fn parse_scalar_n(value: &Value) -> crate::BuiltinResult<usize> {
    let Some(raw) = tensor::scalar_f64_from_value_async(value)
        .await
        .map_err(|e| builtin_error(format!("peaks: {e}")))?
    else {
        return Err(builtin_error("peaks: n must be a numeric scalar"));
    };
    if !raw.is_finite() {
        return Err(builtin_error("peaks: n must be finite"));
    }
    let rounded = raw.round();
    if (rounded - raw).abs() > 1e-6 {
        return Err(builtin_error("peaks: n must be an integer"));
    }
    if rounded < 0.0 {
        return Err(builtin_error("peaks: n must be non-negative"));
    }
    if rounded > usize::MAX as f64 {
        return Err(builtin_error("peaks: n is too large for this platform"));
    }
    Ok(rounded as usize)
}

async fn gather_tensor(value: &Value) -> crate::BuiltinResult<Tensor> {
    match value {
        Value::Tensor(t) => Ok(t.clone()),
        Value::Num(v) => {
            Tensor::new(vec![*v], vec![1, 1]).map_err(|e| builtin_error(format!("peaks: {e}")))
        }
        _ => Err(builtin_error("peaks: X and Y must be numeric matrices")),
    }
}

fn matrix_shape(tensor: &Tensor) -> crate::BuiltinResult<(usize, usize)> {
    match tensor.shape.as_slice() {
        [rows, cols] => Ok((*rows, *cols)),
        [n] => Ok((1, *n)),
        _ => Err(builtin_error("peaks: X and Y must be 2-D matrices")),
    }
}

fn validate_xy_shapes(x: &Tensor, y: &Tensor) -> crate::BuiltinResult<()> {
    if x.shape != y.shape {
        return Err(builtin_error("peaks: X and Y must have the same size"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    fn peaks_builtin(rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(super::peaks_builtin(rest))
    }

    #[test]
    fn peaks_default_shape() {
        let value = peaks_builtin(vec![]).expect("peaks");
        match value {
            Value::Tensor(t) => assert_eq!(t.shape, vec![49, 49]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn peaks_n_shape() {
        let value = peaks_builtin(vec![Value::Num(20.0)]).expect("peaks");
        match value {
            Value::Tensor(t) => assert_eq!(t.shape, vec![20, 20]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn peaks_zero_is_empty() {
        let value = peaks_builtin(vec![Value::Num(0.0)]).expect("peaks");
        match value {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![0, 0]);
                assert!(t.data.is_empty());
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[test]
    fn peaks_one_is_scalar() {
        // At n=1 the single grid point maps to the stop endpoint (x=3, y=3).
        // tensor_into_value may collapse a 1×1 tensor to Value::Num.
        let expected = peaks_at(3.0, 3.0);
        let value = peaks_builtin(vec![Value::Num(1.0)]).expect("peaks");
        let got = match value {
            Value::Num(v) => v,
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 1]);
                t.data[0]
            }
            other => panic!("expected scalar or 1×1 tensor, got {other:?}"),
        };
        assert!((got - expected).abs() < 1e-12);
    }

    #[test]
    fn peaks_formula_known_value() {
        // The origin (x=0, y=0) is a well-known point.
        // Z = 3*(1)^2*exp(0 - 1) - 10*(0 - 0 - 0)*exp(0) - 1/3*exp(-1 - 0)
        //   = 3*exp(-1) - 0 - 1/3*exp(-1)
        //   = exp(-1) * (3 - 1/3)
        //   = exp(-1) * 8/3
        let expected = std::f64::consts::E.recip() * 8.0 / 3.0;
        let got = peaks_at(0.0, 0.0);
        assert!(
            (got - expected).abs() < 1e-12,
            "got {got}, expected {expected}"
        );
    }

    #[test]
    fn peaks_xy_form() {
        use runmat_builtins::Tensor;
        let x = Tensor::new(vec![0.0, 1.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let y = Tensor::new(vec![0.0, 0.0, 1.0, 1.0], vec![2, 2]).unwrap();
        let value =
            peaks_builtin(vec![Value::Tensor(x.clone()), Value::Tensor(y.clone())]).expect("peaks");
        match value {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                for i in 0..4 {
                    let expected = peaks_at(x.data[i], y.data[i]);
                    assert!((t.data[i] - expected).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn peaks_too_many_args_errors() {
        let err =
            peaks_builtin(vec![Value::Num(1.0), Value::Num(2.0), Value::Num(3.0)]).unwrap_err();
        assert!(err.to_string().contains("0, 1, or 2"));
    }

    #[test]
    fn peaks_too_many_outputs_errors() {
        // Simulate [a,b,c,d] = peaks() — out_count = 4 must be rejected.
        let _guard = crate::output_count::push_output_count(Some(4));
        let err = peaks_builtin(vec![]).unwrap_err();
        assert!(
            err.to_string().contains("too many output arguments"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn peaks_non_integer_n_errors() {
        let err = peaks_builtin(vec![Value::Num(3.7)]).unwrap_err();
        assert!(err.to_string().contains("integer"));
    }

    #[test]
    fn peaks_negative_n_errors() {
        let err = peaks_builtin(vec![Value::Num(-1.0)]).unwrap_err();
        assert!(err.to_string().contains("non-negative"));
    }

    #[test]
    fn peaks_n_too_large_errors() {
        // 2e19 exceeds usize::MAX on all common platforms; must error cleanly.
        let err = peaks_builtin(vec![Value::Num(2e19)]).unwrap_err();
        assert!(
            err.to_string().contains("too large"),
            "unexpected error: {err}"
        );
    }

    // -----------------------------------------------------------------------
    // GPU parity tests
    //
    // The simple_provider used by with_test_provider has no GPU compute; it
    // stores tensors in a host HashMap.  Calling provider.peaks(n) returns Err
    // from that provider, so peaks_from_n falls back to the host path — meaning
    // with_test_provider tests would silently compare CPU against CPU.
    //
    // The wgpu-gated tests below call provider.peaks / provider.peaks_xy
    // *directly*, so we know for certain we are exercising the WGSL shader.
    // CPU reference values come from the host peaks_at / compute_z functions.
    // -----------------------------------------------------------------------

    #[cfg(feature = "wgpu")]
    mod wgpu_parity {
        use super::*;
        use crate::builtins::common::test_support;
        use runmat_accelerate::backend::wgpu::provider::{
            register_wgpu_provider, WgpuProviderOptions,
        };
        use runmat_accelerate_api::{AccelProvider, HostTensorView, ProviderPrecision};

        fn wgpu_provider() -> &'static dyn AccelProvider {
            register_wgpu_provider(WgpuProviderOptions::default()).expect("wgpu provider");
            runmat_accelerate_api::provider().expect("provider registered")
        }

        fn tol(provider: &dyn AccelProvider) -> f64 {
            match provider.precision() {
                ProviderPrecision::F64 => 1e-10,
                ProviderPrecision::F32 => 1e-4,
            }
        }

        fn gather_handle(
            handle: runmat_accelerate_api::GpuTensorHandle,
        ) -> runmat_builtins::Tensor {
            test_support::gather(Value::GpuTensor(handle)).expect("gather")
        }

        /// peaks(n) shader matches host for n ∈ {1, 3, 7, 10, 20, 49}.
        /// Calls provider.peaks() directly so there is no silent fallback.
        #[test]
        fn peaks_wgpu_parity_n() {
            let provider = wgpu_provider();
            let tol = tol(provider);

            for &n in &[1usize, 3, 7, 10, 20, 49] {
                let (x_flat, y_flat) = make_axis(n);
                let (x_mat, y_mat) = make_grids(&x_flat, &y_flat, n, n);
                let z_ref = compute_z(&x_mat, &y_mat, n, n);

                let handle = provider
                    .peaks(n)
                    .unwrap_or_else(|e| panic!("provider.peaks({n}) failed: {e}"));
                let t = gather_handle(handle);

                assert_eq!(t.shape, vec![n, n], "shape mismatch for n={n}");
                for (i, (&gv, &cv)) in t.data.iter().zip(z_ref.iter()).enumerate() {
                    let err = (gv - cv).abs();
                    assert!(
                        err <= tol,
                        "n={n} row={} col={}: gpu={gv:.8e} cpu={cv:.8e} err={err:.2e}",
                        i % n,
                        i / n,
                    );
                }
            }
        }

        /// peaks(0) produces an empty tensor without panicking.
        #[test]
        fn peaks_wgpu_empty() {
            let provider = wgpu_provider();
            let handle = provider.peaks(0).expect("peaks(0)");
            let t = gather_handle(handle);
            assert_eq!(t.shape, vec![0, 0]);
            assert!(t.data.is_empty());
        }

        /// peaks_xy shader matches host at coordinates spanning the interesting
        /// region: corners, origin, saddle points, and near the main peak.
        /// Calls provider.peaks_xy() directly — no silent fallback possible.
        #[test]
        fn peaks_wgpu_xy_parity() {
            let provider = wgpu_provider();
            let tol = tol(provider);

            let x_data: Vec<f64> = vec![
                -3.0, 0.0, 3.0, -3.0, 0.0, 3.0, -1.0, 0.5, 1.5, -2.5, 2.5, 0.0,
            ];
            let y_data: Vec<f64> = vec![
                -3.0, -3.0, -3.0, 3.0, 3.0, 3.0, -1.0, -0.5, 1.0, 0.0, 0.0, 0.5,
            ];
            let n = x_data.len();
            let shape = [1usize, n];

            let z_ref: Vec<f64> = x_data
                .iter()
                .zip(y_data.iter())
                .map(|(&x, &y)| peaks_at(x, y))
                .collect();

            let x_handle = provider
                .upload(&HostTensorView {
                    data: &x_data,
                    shape: &shape,
                })
                .expect("upload x");
            let y_handle = provider
                .upload(&HostTensorView {
                    data: &y_data,
                    shape: &shape,
                })
                .expect("upload y");

            let z_handle = provider
                .peaks_xy(&x_handle, &y_handle)
                .expect("provider.peaks_xy");
            let t = gather_handle(z_handle);

            assert_eq!(t.shape, shape.to_vec());
            for (i, (&gv, &cv)) in t.data.iter().zip(z_ref.iter()).enumerate() {
                let err = (gv - cv).abs();
                assert!(
                    err <= tol,
                    "peaks_xy idx={i} x={} y={}: gpu={gv:.8e} cpu={cv:.8e} err={err:.2e}",
                    x_data[i],
                    y_data[i],
                );
            }
        }

        /// peaks_xy with empty tensors produces an empty result without panicking.
        #[test]
        fn peaks_wgpu_xy_empty() {
            let provider = wgpu_provider();
            let empty: &[f64] = &[];
            let shape = [0usize, 0];
            let x_handle = provider
                .upload(&HostTensorView {
                    data: empty,
                    shape: &shape,
                })
                .expect("upload x");
            let y_handle = provider
                .upload(&HostTensorView {
                    data: empty,
                    shape: &shape,
                })
                .expect("upload y");
            let z_handle = provider
                .peaks_xy(&x_handle, &y_handle)
                .expect("peaks_xy empty");
            let t = gather_handle(z_handle);
            assert!(t.data.is_empty());
        }

        /// End-to-end: peaks_builtin(n) dispatches to GPU and returns a GpuTensor
        /// — confirms there is no silent host fallback.
        #[test]
        fn peaks_builtin_returns_gpu_tensor() {
            register_wgpu_provider(WgpuProviderOptions::default()).expect("wgpu provider");
            let value = peaks_builtin(vec![Value::Num(10.0)]).expect("peaks");
            assert!(
                matches!(value, Value::GpuTensor(_)),
                "expected GpuTensor from peaks(10), got {value:?}"
            );
        }

        /// End-to-end: peaks_builtin(X_gpu, Y_gpu) dispatches to GPU.
        #[test]
        fn peaks_builtin_xy_returns_gpu_tensor() {
            let provider = wgpu_provider();
            let x_data = [0.0f64, 1.0, -1.0];
            let y_data = [0.0f64, 0.5, -0.5];
            let shape = [1usize, 3];
            let x_handle = provider
                .upload(&HostTensorView {
                    data: &x_data,
                    shape: &shape,
                })
                .expect("upload x");
            let y_handle = provider
                .upload(&HostTensorView {
                    data: &y_data,
                    shape: &shape,
                })
                .expect("upload y");

            let value = peaks_builtin(vec![Value::GpuTensor(x_handle), Value::GpuTensor(y_handle)])
                .expect("peaks xy");
            assert!(
                matches!(value, Value::GpuTensor(_)),
                "expected GpuTensor from peaks(X_gpu, Y_gpu), got {value:?}"
            );
        }
    }
}
