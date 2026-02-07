//! MATLAB-compatible `eig` builtin with host and GPU-aware semantics.
//!
//! Implements the dense eigenvalue decomposition for real and complex matrices,
//! including the vector-only form, the `[V,D]` factorisation, and the
//! three-output `[V,D,W]` variant that returns left eigenvectors. GPU inputs are
//! currently gathered back to the host unless a provider implements the
//! reserved `eig` hook; see the documentation string for full details.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::linalg::type_resolvers::eig_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use nalgebra::linalg::Schur;
use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

const BUILTIN_NAME: &str = "eig";

const REAL_EPS: f64 = 1e-12;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::factor::eig")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "eig",
    op_kind: GpuOpKind::Custom("eig-factor"),
    supported_precisions: &[ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("eig")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Prefers the provider `eig` hook (WGPU reuploads host-computed results for real spectra) and falls back to the CPU implementation for complex spectra or unsupported options.",
};

fn eig_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

fn with_eig_context(mut error: RuntimeError) -> RuntimeError {
    if error.message() == "interaction pending..." {
        return build_runtime_error("interaction pending...")
            .with_builtin(BUILTIN_NAME)
            .build();
    }
    if error.context.builtin.is_none() {
        error.context = error.context.with_builtin(BUILTIN_NAME);
    }
    error
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::linalg::factor::eig")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "eig",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Eigenvalue decomposition executes eagerly and never participates in fusion.",
};

#[runtime_builtin(
    name = "eig",
    category = "math/linalg/factor",
    summary = "Eigenvalue decomposition with MATLAB-compatible multi-output forms.",
    keywords = "eig,eigenvalues,eigenvectors,linalg",
    accel = "sink",
    sink = true,
    type_resolver(eig_type),
    builtin_path = "crate::builtins::math::linalg::factor::eig"
)]
async fn eig_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if let Some(out_count) = crate::output_count::current_output_count() {
        let require_left = out_count >= 3;
        let eval = evaluate(value, &rest, require_left).await?;
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::OutputList(vec![eval.eigenvalues()]));
        }
        if out_count == 2 {
            return Ok(Value::OutputList(vec![eval.right(), eval.diagonal()]));
        }
        if out_count == 3 {
            let left = eval.left()?;
            return Ok(Value::OutputList(vec![
                eval.right(),
                eval.diagonal(),
                left,
            ]));
        }
        return Err(eig_error("eig currently supports at most three outputs"));
    }
    let eval = evaluate(value, &rest, false).await?;
    Ok(eval.eigenvalues())
}

#[derive(Clone, Debug)]
pub struct EigEval {
    eigenvalues: Value,
    diagonal_matrix: Value,
    diagonal_output: Value,
    right: Value,
    left: Option<Value>,
}

impl EigEval {
    pub fn eigenvalues(&self) -> Value {
        self.eigenvalues.clone()
    }

    pub fn diagonal(&self) -> Value {
        self.diagonal_output.clone()
    }

    pub fn diagonal_matrix(&self) -> Value {
        self.diagonal_matrix.clone()
    }

    pub fn right(&self) -> Value {
        self.right.clone()
    }

    pub fn left(&self) -> BuiltinResult<Value> {
        self.left.clone().ok_or_else(|| {
            eig_error(
                "eig: left eigenvectors are not available from the active acceleration provider",
            )
        })
    }

    fn from_provider(result: runmat_accelerate_api::ProviderEigResult) -> Self {
        let runmat_accelerate_api::ProviderEigResult {
            eigenvalues,
            diagonal,
            right,
            left,
        } = result;
        EigEval {
            eigenvalues: Value::GpuTensor(eigenvalues),
            diagonal_matrix: Value::GpuTensor(diagonal.clone()),
            diagonal_output: Value::GpuTensor(diagonal),
            right: Value::GpuTensor(right),
            left: left.map(Value::GpuTensor),
        }
    }
}

#[derive(Clone, Copy)]
struct EigOptions {
    balance: bool,
    vector_output: bool,
}

impl Default for EigOptions {
    fn default() -> Self {
        Self {
            balance: true,
            vector_output: false,
        }
    }
}

pub async fn evaluate(value: Value, args: &[Value], require_left: bool) -> BuiltinResult<EigEval> {
    let options = parse_options(args)?;
    match value {
        Value::GpuTensor(handle) => {
            if let Some(eval) = evaluate_gpu(&handle, options, require_left).await? {
                return Ok(eval);
            }
            let tensor = gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(with_eig_context)?;
            evaluate_host(Value::Tensor(tensor), options, require_left).await
        }
        other => evaluate_host(other, options, require_left).await,
    }
}

async fn evaluate_gpu(
    handle: &GpuTensorHandle,
    options: EigOptions,
    require_left: bool,
) -> BuiltinResult<Option<EigEval>> {
    if options.vector_output {
        return Ok(None);
    }
    if !options.balance {
        return Ok(None);
    }
    let provider = match runmat_accelerate_api::provider() {
        Some(p) => p,
        None => return Ok(None),
    };
    match provider.eig(handle, require_left).await {
        Ok(result) => {
            if require_left && result.left.is_none() {
                Ok(None)
            } else {
                Ok(Some(EigEval::from_provider(result)))
            }
        }
        Err(_) => Ok(None),
    }
}

async fn evaluate_host(
    value: Value,
    options: EigOptions,
    require_left: bool,
) -> BuiltinResult<EigEval> {
    let matrix = value_to_complex_matrix(value).await?;
    compute_eigen(matrix, options, require_left)
}

fn parse_options(args: &[Value]) -> BuiltinResult<EigOptions> {
    let mut opts = EigOptions::default();
    for (idx, arg) in args.iter().enumerate() {
        if let Some(text) = tensor::value_to_string(arg) {
            match text.trim().to_ascii_lowercase().as_str() {
                "balance" => opts.balance = true,
                "nobalance" => opts.balance = false,
                "vector" => opts.vector_output = true,
                "matrix" => opts.vector_output = false,
                other => {
                    return Err(eig_error(format!("eig: unknown option '{other}'")));
                }
            }
        } else if idx == 0 {
            return Err(eig_error(
                "eig: generalized eigenvalue decomposition (eig(A,B)) is not implemented",
            ));
        } else {
            return Err(eig_error(
                "eig: option arguments must be character vectors or string scalars",
            ));
        }
    }
    Ok(opts)
}

fn compute_eigen(
    matrix: DMatrix<Complex64>,
    options: EigOptions,
    require_left: bool,
) -> BuiltinResult<EigEval> {
    if matrix.nrows() != matrix.ncols() {
        return Err(eig_error("eig: input matrix must be square"));
    }
    let n = matrix.nrows();
    if n == 0 {
        let empty_vals =
            Tensor::new(Vec::new(), vec![0, 0]).map_err(|e| eig_error(format!("eig: {e}")))?;
        let empty_mat =
            Tensor::new(Vec::new(), vec![0, 0]).map_err(|e| eig_error(format!("eig: {e}")))?;
        let eigenvalues_value = Value::Tensor(empty_vals.clone());
        let diagonal_matrix_value = Value::Tensor(empty_mat.clone());
        let diagonal_output = if options.vector_output {
            eigenvalues_value.clone()
        } else {
            diagonal_matrix_value.clone()
        };
        return Ok(EigEval {
            eigenvalues: eigenvalues_value,
            diagonal_matrix: diagonal_matrix_value,
            diagonal_output,
            right: Value::Tensor(empty_mat.clone()),
            left: if require_left {
                Some(Value::Tensor(empty_mat))
            } else {
                None
            },
        });
    }

    let balanced = maybe_balance(&matrix, options.balance);
    let (eigenvalues, right) = schur_eigendecompose(&balanced)?;

    let eigenvalue_value = vector_to_value(&eigenvalues)?;
    let diag_value = diag_matrix_value(&eigenvalues)?;
    let right_value = matrix_to_value(&right)?;

    let left_value = if require_left {
        let left_matrix =
            compute_left_vectors(&balanced, &right, &eigenvalues).ok_or_else(|| {
                eig_error("eig: unable to compute left eigenvectors for the requested matrix")
            })?;
        Some(matrix_to_value(&left_matrix)?)
    } else {
        None
    };

    let spectral_output = if options.vector_output {
        eigenvalue_value.clone()
    } else {
        diag_value.clone()
    };

    Ok(EigEval {
        eigenvalues: eigenvalue_value,
        diagonal_matrix: diag_value,
        diagonal_output: spectral_output,
        right: right_value,
        left: left_value,
    })
}

fn maybe_balance(matrix: &DMatrix<Complex64>, _balance: bool) -> DMatrix<Complex64> {
    matrix.clone()
}

fn schur_eigendecompose(
    matrix: &DMatrix<Complex64>,
) -> BuiltinResult<(DVector<Complex64>, DMatrix<Complex64>)> {
    let n = matrix.nrows();
    if n == 0 {
        return Ok((DVector::from(vec![]), DMatrix::zeros(0, 0)));
    }
    let schur = Schur::new(matrix.clone());
    let (q, t) = schur.unpack();
    let diag = t.diagonal().clone_owned();
    let mut eigenvectors = DMatrix::<Complex64>::zeros(n, n);

    for idx in 0..n {
        let lambda = diag[idx];
        let mut z = DVector::<Complex64>::zeros(n);
        let mut success = false;

        for attempt in 0..2 {
            for k in (0..n).rev() {
                let mut sum = Complex64::new(0.0, 0.0);
                for j in (k + 1)..n {
                    sum += t[(k, j)] * z[j];
                }
                let coeff = t[(k, k)] - lambda;
                if coeff.norm() <= REAL_EPS {
                    if sum.norm() <= REAL_EPS {
                        if attempt == 0 && k == idx {
                            z[k] = Complex64::new(1.0, 0.0);
                        }
                    } else {
                        z[k] = -sum / Complex64::new(REAL_EPS, 0.0);
                    }
                } else {
                    z[k] = -sum / coeff;
                }
            }
            let norm = z.norm();
            if norm > REAL_EPS {
                let scale = Complex64::new(norm, 0.0);
                z /= scale;
                success = true;
                break;
            } else {
                z.fill(Complex64::new(0.0, 0.0));
                z[idx] = Complex64::new(1.0, 0.0);
            }
        }
        if !success {
            z.fill(Complex64::new(0.0, 0.0));
            z[idx] = Complex64::new(1.0, 0.0);
        }
        let vec = q.clone() * z;
        let norm = vec.norm();
        let final_vec = if norm > REAL_EPS {
            vec / Complex64::new(norm, 0.0)
        } else {
            vec
        };
        eigenvectors.set_column(idx, &final_vec);
    }

    Ok((diag, eigenvectors))
}

fn compute_left_vectors(
    matrix: &DMatrix<Complex64>,
    right: &DMatrix<Complex64>,
    eigenvalues: &DVector<Complex64>,
) -> Option<DMatrix<Complex64>> {
    if let Some(inv) = right.clone().try_inverse() {
        let mut left = inv.adjoint();
        normalize_left(&mut left, right);
        return Some(left);
    }

    let (left_vals, left_vecs) = schur_eigendecompose(&matrix.adjoint()).ok()?;
    let n = eigenvalues.len();
    let mut left = DMatrix::<Complex64>::zeros(n, n);
    let mut used = vec![false; n];

    for i in 0..n {
        let target = eigenvalues[i].conj();
        let mut best_idx = None;
        let mut best_err = f64::MAX;
        for j in 0..n {
            if used[j] {
                continue;
            }
            let err = (left_vals[j] - target).norm();
            if err < best_err {
                best_err = err;
                best_idx = Some(j);
            }
        }
        let idx = best_idx.unwrap_or(i);
        used[idx] = true;
        left.set_column(i, &left_vecs.column(idx));
    }

    normalize_left(&mut left, right);
    Some(left)
}

fn normalize_left(left: &mut DMatrix<Complex64>, right: &DMatrix<Complex64>) {
    for i in 0..right.ncols() {
        let dot = left.column(i).dot(&right.column(i));
        let finite = dot.re.is_finite() && dot.im.is_finite();
        if dot.norm() > REAL_EPS && finite {
            let scale = dot.conj();
            for r in 0..left.nrows() {
                left[(r, i)] /= scale;
            }
        }
    }
}

async fn value_to_complex_matrix(value: Value) -> BuiltinResult<DMatrix<Complex64>> {
    match value {
        Value::Tensor(tensor) => tensor_to_matrix(&tensor),
        Value::ComplexTensor(ct) => complex_tensor_to_matrix(&ct),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|err| eig_error(format!("eig: {err}")))?;
            tensor_to_matrix(&tensor)
        }
        Value::Num(n) => Ok(DMatrix::from_element(1, 1, Complex64::new(n, 0.0))),
        Value::Int(i) => Ok(DMatrix::from_element(1, 1, Complex64::new(i.to_f64(), 0.0))),
        Value::Bool(b) => Ok(DMatrix::from_element(
            1,
            1,
            Complex64::new(if b { 1.0 } else { 0.0 }, 0.0),
        )),
        Value::Complex(re, im) => Ok(DMatrix::from_element(1, 1, Complex64::new(re, im))),
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(with_eig_context)?;
            tensor_to_matrix(&tensor)
        }
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => Err(eig_error(
            "eig: input must be numeric or logical; convert character data with double() first",
        )),
        other => Err(eig_error(format!(
            "eig: unsupported input type {other:?}; expected numeric or logical values"
        ))),
    }
}

fn tensor_to_matrix(tensor: &Tensor) -> BuiltinResult<DMatrix<Complex64>> {
    if tensor.shape.len() > 2 {
        return Err(eig_error("eig: input must be 2-D"));
    }
    let rows = tensor.rows();
    let cols = tensor.cols();
    let mut data = Vec::with_capacity(tensor.data.len());
    for &value in &tensor.data {
        data.push(Complex64::new(value, 0.0));
    }
    Ok(DMatrix::from_column_slice(rows, cols, &data))
}

fn complex_tensor_to_matrix(tensor: &ComplexTensor) -> BuiltinResult<DMatrix<Complex64>> {
    if tensor.shape.len() > 2 {
        return Err(eig_error("eig: input must be 2-D"));
    }
    let rows = tensor.rows;
    let cols = tensor.cols;
    let mut data = Vec::with_capacity(tensor.data.len());
    for &(re, im) in &tensor.data {
        data.push(Complex64::new(re, im));
    }
    Ok(DMatrix::from_column_slice(rows, cols, &data))
}

fn vector_to_value(values: &DVector<Complex64>) -> BuiltinResult<Value> {
    if values.is_empty() {
        let tensor =
            Tensor::new(Vec::new(), vec![0, 0]).map_err(|e| eig_error(format!("eig: {e}")))?;
        return Ok(Value::Tensor(tensor));
    }
    if is_all_real(values.iter().copied()) {
        let mut data = Vec::with_capacity(values.len());
        for value in values.iter() {
            data.push(value.re);
        }
        let tensor =
            Tensor::new(data, vec![values.len(), 1]).map_err(|e| eig_error(format!("eig: {e}")))?;
        Ok(tensor::tensor_into_value(tensor))
    } else {
        let mut data = Vec::with_capacity(values.len());
        for value in values.iter() {
            data.push((value.re, value.im));
        }
        let tensor = ComplexTensor::new(data, vec![values.len(), 1])
            .map_err(|e| eig_error(format!("eig: {e}")))?;
        Ok(Value::ComplexTensor(tensor))
    }
}

fn diag_matrix_value(values: &DVector<Complex64>) -> BuiltinResult<Value> {
    if values.is_empty() {
        let tensor =
            Tensor::new(Vec::new(), vec![0, 0]).map_err(|e| eig_error(format!("eig: {e}")))?;
        return Ok(Value::Tensor(tensor));
    }
    let size = values.len();
    if is_all_real(values.iter().copied()) {
        let mut data = vec![0.0f64; size * size];
        for i in 0..size {
            data[i + i * size] = values[i].re;
        }
        let tensor =
            Tensor::new(data, vec![size, size]).map_err(|e| eig_error(format!("eig: {e}")))?;
        Ok(Value::Tensor(tensor))
    } else {
        let mut data = vec![(0.0f64, 0.0f64); size * size];
        for i in 0..size {
            data[i + i * size] = (values[i].re, values[i].im);
        }
        let tensor = ComplexTensor::new(data, vec![size, size])
            .map_err(|e| eig_error(format!("eig: {e}")))?;
        Ok(Value::ComplexTensor(tensor))
    }
}

fn matrix_to_value(matrix: &DMatrix<Complex64>) -> BuiltinResult<Value> {
    if is_all_real(matrix.iter().copied()) {
        let mut data = Vec::with_capacity(matrix.len());
        for value in matrix.iter() {
            data.push(value.re);
        }
        let tensor = Tensor::new(data, vec![matrix.nrows(), matrix.ncols()])
            .map_err(|e| eig_error(format!("eig: {e}")))?;
        Ok(Value::Tensor(tensor))
    } else {
        let mut data = Vec::with_capacity(matrix.len());
        for value in matrix.iter() {
            data.push((value.re, value.im));
        }
        let tensor = ComplexTensor::new(data, vec![matrix.nrows(), matrix.ncols()])
            .map_err(|e| eig_error(format!("eig: {e}")))?;
        Ok(Value::ComplexTensor(tensor))
    }
}

fn is_all_real<I>(iter: I) -> bool
where
    I: IntoIterator<Item = Complex64>,
{
    iter.into_iter()
        .all(|value| value.im.is_nan() || value.im.abs() <= REAL_EPS)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, ResolveContext, Type};

    fn error_message(err: RuntimeError) -> String {
        err.message().to_string()
    }

    fn matrix_from_value(value: Value) -> DMatrix<Complex64> {
        match value {
            Value::Tensor(t) => tensor_to_matrix(&t).expect("matrix"),
            Value::ComplexTensor(ct) => complex_tensor_to_matrix(&ct).expect("matrix"),
            other => panic!("expected matrix value, got {other:?}"),
        }
    }

    #[test]
    fn eig_type_returns_eigenvalue_vector() {
        let out = eig_type(
            &[Type::Tensor {
                shape: Some(vec![Some(3), Some(3)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(3), Some(1)])
            }
        );
    }

    fn column_vector_from_value(value: Value) -> Vec<Complex64> {
        match value {
            Value::Tensor(t) => t
                .data
                .iter()
                .map(|&v| Complex64::new(v, 0.0))
                .collect::<Vec<_>>(),
            Value::ComplexTensor(ct) => ct
                .data
                .iter()
                .map(|&(re, im)| Complex64::new(re, im))
                .collect::<Vec<_>>(),
            other => panic!("expected tensor for eigenvalues, got {other:?}"),
        }
    }

    fn assert_matrix_close(a: &DMatrix<Complex64>, b: &DMatrix<Complex64>, tol: f64) {
        assert_eq!(a.nrows(), b.nrows(), "row mismatch");
        assert_eq!(a.ncols(), b.ncols(), "column mismatch");
        for r in 0..a.nrows() {
            for c in 0..a.ncols() {
                let diff = (a[(r, c)] - b[(r, c)]).norm();
                assert!(
                    diff <= tol,
                    "matrix mismatch at ({r},{c}): {} vs {} (diff {diff})",
                    a[(r, c)],
                    b[(r, c)]
                );
            }
        }
    }

    #[cfg(feature = "wgpu")]
    fn assert_tensor_close(a: &Tensor, b: &Tensor, tol: f64) {
        assert_eq!(
            a.shape, b.shape,
            "shape mismatch: {:?} vs {:?}",
            a.shape, b.shape
        );
        for (idx, (lhs, rhs)) in a.data.iter().zip(b.data.iter()).enumerate() {
            let diff = (lhs - rhs).abs();
            assert!(
                diff <= tol,
                "tensor mismatch at index {}: {} vs {} (diff {diff})",
                idx,
                lhs,
                rhs
            );
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_scalar_real() {
        let result = eig_builtin(Value::Num(5.0), Vec::new()).expect("eig");
        match result {
            Value::Num(v) => assert!((v - 5.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_two_outputs_reconstruct() {
        let tensor = Tensor::new(vec![0.0, -2.0, 1.0, -3.0], vec![2, 2]).unwrap();
        let eval = evaluate(Value::Tensor(tensor.clone()), &[], false).expect("evaluate");
        let v = matrix_from_value(eval.right());
        let d = matrix_from_value(eval.diagonal_matrix());
        let a = matrix_from_value(Value::Tensor(tensor));
        let recon = &v * &d * v.try_inverse().unwrap();
        assert_matrix_close(&a, &recon, 1e-10);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_three_outputs_biorthogonality() {
        let tensor = Tensor::new(vec![4.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap();
        let eval = evaluate(Value::Tensor(tensor), &[], true).expect("evaluate");
        let v = matrix_from_value(eval.right());
        let d = matrix_from_value(eval.diagonal_matrix());
        let w = matrix_from_value(eval.left().expect("left eigenvectors"));
        let vw = w.adjoint() * &v;
        let identity = DMatrix::<Complex64>::identity(v.ncols(), v.ncols());
        assert_matrix_close(&vw, &identity, 1e-10);
        let tensor = Tensor::new(vec![4.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap();
        let a = matrix_from_value(Value::Tensor(tensor));
        let lhs = w.adjoint() * &a;
        let rhs = &d * w.adjoint();
        assert_matrix_close(&lhs, &rhs, 1e-10);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_complex_matrix() {
        let data = vec![(1.0, 2.0), (0.0, 0.0), (2.0, -1.0), (0.0, -3.0)];
        let tensor = ComplexTensor::new(data, vec![2, 2]).unwrap();
        let result = eig_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("eig");
        let values = column_vector_from_value(result);
        assert_eq!(values.len(), 2);
        assert!((values[0] - Complex64::new(1.0, 2.0)).norm() < 1e-10);
        assert!((values[1] - Complex64::new(0.0, -3.0)).norm() < 1e-10);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_errors_on_non_square() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = error_message(eig_builtin(Value::Tensor(tensor), Vec::new()).unwrap_err());
        assert!(err.contains("square"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_accepts_nobalance_option() {
        let tensor = Tensor::new(vec![1.0, 1.0, 0.0, 2.0], vec![2, 2]).unwrap();
        let base = eig_builtin(Value::Tensor(tensor.clone()), Vec::new()).expect("eig");
        let opt = eig_builtin(
            Value::Tensor(tensor),
            vec![Value::String("nobalance".into())],
        )
        .expect("eig with option");
        let base_vals = column_vector_from_value(base);
        let opt_vals = column_vector_from_value(opt);
        for (a, b) in base_vals.iter().zip(opt_vals.iter()) {
            assert!((a - b).norm() < 1e-10);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_vector_option_returns_column_vector() {
        let tensor = Tensor::new(vec![0.0, 1.0, -2.0, -3.0], vec![2, 2]).unwrap();
        let eval =
            evaluate(Value::Tensor(tensor), &[Value::from("vector")], false).expect("evaluate");
        match eval.diagonal() {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
            }
            other => panic!("expected vector second output, got {other:?}"),
        }
        // Matrix form stays available for reconstruction.
        let _ = matrix_from_value(eval.diagonal_matrix());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_vector_option_allows_left_eigenvectors() {
        let tensor = Tensor::new(vec![4.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap();
        let eval =
            evaluate(Value::Tensor(tensor), &[Value::from("vector")], true).expect("evaluate");
        match eval.diagonal() {
            Value::Tensor(t) => assert_eq!(t.shape, vec![2, 1]),
            other => panic!("expected vector second output, got {other:?}"),
        }
        let left = eval.left().expect("left eigenvectors");
        match left {
            Value::Tensor(t) => assert_eq!(t.shape, vec![2, 2]),
            Value::ComplexTensor(ct) => assert_eq!(ct.shape, vec![2, 2]),
            other => panic!("unexpected type for left eigenvectors: {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_vector_option_gpu_falls_back_to_host() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![2.0, 1.0, 0.0, 3.0], vec![2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let eval = evaluate(Value::GpuTensor(handle), &[Value::from("vector")], false)
                .expect("evaluate");
            match eval.diagonal() {
                Value::Tensor(t) => assert_eq!(t.shape, vec![2, 1]),
                Value::ComplexTensor(ct) => assert_eq!(ct.shape, vec![2, 1]),
                Value::GpuTensor(_) => panic!("expected host fallback for 'vector' option"),
                other => panic!("unexpected eigenvalue output: {other:?}"),
            }
            if let Value::GpuTensor(_) = eval.right() {
                panic!("expected right eigenvectors on host after fallback");
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_gpu_provider_roundtrip_gathers_to_host() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![2.0, 0.0, 0.0, 3.0], vec![2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result =
                eig_builtin(Value::GpuTensor(handle), vec![Value::from("nobalance")]).expect("eig");
            match result {
                Value::Tensor(t) => assert_eq!(t.data, vec![2.0, 3.0]),
                Value::ComplexTensor(ct) => {
                    assert_eq!(ct.data[0], (2.0, 0.0));
                    assert_eq!(ct.data[1], (3.0, 0.0));
                }
                other => panic!("expected tensor result, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_handles_single_numeric_argument() {
        let args = vec![Value::Int(IntValue::I32(3))];
        let err = error_message(evaluate(Value::Num(4.0), &args, false).unwrap_err());
        assert!(
            err.contains("generalized")
                || err.contains("option arguments must be")
                || err.contains("unknown option"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn eig_wgpu_matches_cpu_for_real_spectrum() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        let tensor = Tensor::new(vec![4.0, 2.0, 1.0, 3.0], vec![2, 2]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");

        let gpu_eval = evaluate(Value::GpuTensor(handle), &[], true).expect("gpu evaluate");

        let eig_gpu_value = gpu_eval.eigenvalues();
        assert!(matches!(eig_gpu_value, Value::GpuTensor(_)));
        let diag_gpu_value = gpu_eval.diagonal_matrix();
        assert!(matches!(diag_gpu_value, Value::GpuTensor(_)));
        let right_gpu_value = gpu_eval.right();
        assert!(matches!(right_gpu_value, Value::GpuTensor(_)));
        let left_gpu_value = gpu_eval.left().expect("left eigenvectors");
        assert!(matches!(left_gpu_value, Value::GpuTensor(_)));

        let eig_gpu = test_support::gather(eig_gpu_value).expect("gather eigenvalues");
        let diag_gpu = test_support::gather(diag_gpu_value).expect("gather diagonal");
        let right_gpu = test_support::gather(right_gpu_value).expect("gather right vectors");
        let left_gpu = test_support::gather(left_gpu_value).expect("gather left vectors");

        let host_eval = evaluate(Value::Tensor(tensor.clone()), &[], true).expect("host evaluate");

        let eig_host = match host_eval.eigenvalues() {
            Value::Tensor(t) => t,
            other => panic!("expected tensor eigenvalues, got {other:?}"),
        };
        let diag_host = match host_eval.diagonal_matrix() {
            Value::Tensor(t) => t,
            other => panic!("expected tensor diagonal, got {other:?}"),
        };
        let right_host = match host_eval.right() {
            Value::Tensor(t) => t,
            other => panic!("expected tensor right eigenvectors, got {other:?}"),
        };
        let left_host = match host_eval.left().expect("host left eigenvectors") {
            Value::Tensor(t) => t,
            other => panic!("expected tensor left eigenvectors, got {other:?}"),
        };

        assert_tensor_close(&eig_gpu, &eig_host, tol);
        assert_tensor_close(&diag_gpu, &diag_host, tol);
        assert_tensor_close(&right_gpu, &right_host, tol);
        assert_tensor_close(&left_gpu, &left_host, tol);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn eig_wgpu_nobalance_falls_back_to_host() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let tensor = Tensor::new(vec![1.0, 1.0, 0.0, 2.0], vec![2, 2]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let eval = evaluate(Value::GpuTensor(handle), &[Value::from("nobalance")], false)
            .expect("evaluate");
        if let Value::GpuTensor(_) = eval.eigenvalues() {
            panic!("expected host fallback for 'nobalance' option");
        }
    }

    fn eig_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::eig_builtin(value, rest))
    }

    fn evaluate(value: Value, args: &[Value], require_left: bool) -> BuiltinResult<EigEval> {
        block_on(super::evaluate(value, args, require_left))
    }
}
