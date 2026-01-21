//! MATLAB-compatible `lu` builtin with CPU-backed semantics.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};

use num_complex::Complex64;
use runmat_accelerate_api::{GpuTensorHandle, ProviderLuResult};
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::factor::lu")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "lu",
    op_kind: GpuOpKind::Custom("lu-factor"),
    supported_precisions: &[ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("lu")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Prefers the provider `lu` hook; automatically gathers and falls back to the CPU implementation when no provider support is registered.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::linalg::factor::lu")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "lu",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "LU decomposition is not part of expression fusion; calls execute eagerly on the CPU.",
};

#[runtime_builtin(
    name = "lu",
    category = "math/linalg/factor",
    summary = "LU decomposition with partial pivoting.",
    keywords = "lu,factorization,decomposition,permutation",
    accel = "sink",
    sink = true,
    builtin_path = "crate::builtins::math::linalg::factor::lu"
)]
fn lu_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    let eval = evaluate(value, &rest)?;
    Ok(eval.combined())
}

/// Output form for `lu`, reused by both the builtin wrapper and the VM multi-output path.
#[derive(Clone)]
pub struct LuEval {
    combined: Value,
    lower: Value,
    upper: Value,
    perm_matrix: Value,
    perm_vector: Value,
    pivot_mode: PivotMode,
}

impl LuEval {
    /// Combined LU factor (single-output form).
    pub fn combined(&self) -> Value {
        self.combined.clone()
    }

    /// Lower-triangular factor.
    pub fn lower(&self) -> Value {
        self.lower.clone()
    }

    /// Upper-triangular factor.
    pub fn upper(&self) -> Value {
        self.upper.clone()
    }

    /// Permutation value respecting the selected pivot mode.
    pub fn permutation(&self) -> Value {
        match self.pivot_mode {
            PivotMode::Matrix => self.perm_matrix.clone(),
            PivotMode::Vector => self.perm_vector.clone(),
        }
    }

    /// Permutation matrix (always available, useful for tests).
    pub fn permutation_matrix(&self) -> Value {
        self.perm_matrix.clone()
    }

    /// Pivot vector (always available, useful for tests).
    pub fn pivot_vector(&self) -> Value {
        self.perm_vector.clone()
    }

    /// The pivot mode that was requested.
    pub fn pivot_mode(&self) -> PivotMode {
        self.pivot_mode
    }

    fn from_components(components: LuComponents, pivot_mode: PivotMode) -> Result<Self, String> {
        let combined = matrix_to_value(&components.combined)?;
        let lower = matrix_to_value(&components.lower)?;
        let upper = matrix_to_value(&components.upper)?;
        let perm_matrix = matrix_to_value(&components.permutation)?;
        let perm_vector = pivot_vector_to_value(&components.pivot_vector)?;
        Ok(Self {
            combined,
            lower,
            upper,
            perm_matrix,
            perm_vector,
            pivot_mode,
        })
    }

    fn from_provider(result: ProviderLuResult, pivot_mode: PivotMode) -> Self {
        Self {
            combined: Value::GpuTensor(result.combined),
            lower: Value::GpuTensor(result.lower),
            upper: Value::GpuTensor(result.upper),
            perm_matrix: Value::GpuTensor(result.perm_matrix),
            perm_vector: Value::GpuTensor(result.perm_vector),
            pivot_mode,
        }
    }
}

/// Permutation output mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PivotMode {
    Matrix,
    Vector,
}

impl Default for PivotMode {
    fn default() -> Self {
        Self::Matrix
    }
}

/// Evaluate `lu` while preserving all output forms for later extraction.
pub fn evaluate(value: Value, args: &[Value]) -> Result<LuEval, String> {
    let pivot_mode = parse_pivot_mode(args)?;
    match value {
        Value::GpuTensor(handle) => {
            if let Some(eval) = evaluate_gpu(&handle, pivot_mode)? {
                return Ok(eval);
            }
            let tensor = gpu_helpers::gather_tensor(&handle)?;
            evaluate_host_value(Value::Tensor(tensor), pivot_mode)
        }
        other => evaluate_host_value(other, pivot_mode),
    }
}

fn evaluate_host_value(value: Value, pivot_mode: PivotMode) -> Result<LuEval, String> {
    let matrix = extract_matrix(value)?;
    let components = lu_factor(matrix)?;
    LuEval::from_components(components, pivot_mode)
}

fn evaluate_gpu(handle: &GpuTensorHandle, pivot_mode: PivotMode) -> Result<Option<LuEval>, String> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Ok(result) = provider.lu(handle) {
            return Ok(Some(LuEval::from_provider(result, pivot_mode)));
        }
    }
    Ok(None)
}

fn parse_pivot_mode(args: &[Value]) -> Result<PivotMode, String> {
    if args.is_empty() {
        return Ok(PivotMode::Matrix);
    }
    if args.len() > 1 {
        return Err("lu: too many option arguments".to_string());
    }
    let Some(option) = tensor::value_to_string(&args[0]) else {
        return Err("lu: option must be a string or character vector".to_string());
    };
    match option.trim().to_ascii_lowercase().as_str() {
        "matrix" => Ok(PivotMode::Matrix),
        "vector" => Ok(PivotMode::Vector),
        other => Err(format!("lu: unknown option '{other}'")),
    }
}

fn extract_matrix(value: Value) -> Result<RowMajorMatrix, String> {
    match value {
        Value::Tensor(t) => RowMajorMatrix::from_tensor(&t),
        Value::ComplexTensor(ct) => RowMajorMatrix::from_complex_tensor(&ct),
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor(&handle)?;
            RowMajorMatrix::from_tensor(&tensor)
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)?;
            RowMajorMatrix::from_tensor(&tensor)
        }
        Value::Num(n) => Ok(RowMajorMatrix::from_scalar(Complex64::new(n, 0.0))),
        Value::Int(i) => Ok(RowMajorMatrix::from_scalar(Complex64::new(i.to_f64(), 0.0))),
        Value::Bool(b) => Ok(RowMajorMatrix::from_scalar(Complex64::new(
            if b { 1.0 } else { 0.0 },
            0.0,
        ))),
        Value::Complex(re, im) => Ok(RowMajorMatrix::from_scalar(Complex64::new(re, im))),
        Value::CharArray(_) | Value::String(_) | Value::StringArray(_) => {
            Err("lu: character data is not supported; convert to numeric values first".to_string())
        }
        other => Err(format!("lu: unsupported input type {:?}", other)),
    }
}

struct LuComponents {
    combined: RowMajorMatrix,
    lower: RowMajorMatrix,
    upper: RowMajorMatrix,
    permutation: RowMajorMatrix,
    pivot_vector: Vec<f64>,
}

fn lu_factor(mut matrix: RowMajorMatrix) -> Result<LuComponents, String> {
    let rows = matrix.rows;
    let cols = matrix.cols;
    let min_dim = rows.min(cols);
    let mut perm: Vec<usize> = (0..rows).collect();

    for k in 0..min_dim {
        // Select pivot row with maximal absolute value in column k.
        let mut pivot_row = k;
        let mut pivot_abs = 0.0;
        for r in k..rows {
            let val = matrix.get(r, k);
            let abs = val.norm();
            if abs > pivot_abs {
                pivot_abs = abs;
                pivot_row = r;
            }
        }

        if pivot_row != k {
            matrix.swap_rows(pivot_row, k);
            perm.swap(pivot_row, k);
        }

        if pivot_abs <= EPS {
            // Entire column is effectively zero; set multipliers to zero and continue.
            for r in (k + 1)..rows {
                matrix.set(r, k, Complex64::new(0.0, 0.0));
            }
            continue;
        }

        let pivot_value = matrix.get(k, k);
        for r in (k + 1)..rows {
            let factor = matrix.get(r, k) / pivot_value;
            matrix.set(r, k, factor);
            for c in (k + 1)..cols {
                let updated = matrix.get(r, c) - factor * matrix.get(k, c);
                matrix.set(r, c, updated);
            }
        }
    }

    let combined = matrix.clone();
    let lower = build_lower(&matrix);
    let upper = build_upper(&matrix);
    let permutation = build_permutation(rows, &perm);
    let pivot_vector: Vec<f64> = perm.iter().map(|idx| (*idx + 1) as f64).collect();

    Ok(LuComponents {
        combined,
        lower,
        upper,
        permutation,
        pivot_vector,
    })
}

fn build_lower(matrix: &RowMajorMatrix) -> RowMajorMatrix {
    let rows = matrix.rows;
    let cols = matrix.cols;
    let min_dim = rows.min(cols);
    let mut lower = RowMajorMatrix::identity(rows);
    for i in 0..rows {
        for j in 0..min_dim {
            if i > j {
                lower.set(i, j, matrix.get(i, j));
            }
        }
    }
    lower
}

fn build_upper(matrix: &RowMajorMatrix) -> RowMajorMatrix {
    let rows = matrix.rows;
    let cols = matrix.cols;
    let mut upper = RowMajorMatrix::zeros(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            if i <= j {
                upper.set(i, j, matrix.get(i, j));
            }
        }
    }
    upper
}

fn build_permutation(rows: usize, perm: &[usize]) -> RowMajorMatrix {
    let mut matrix = RowMajorMatrix::zeros(rows, rows);
    for (i, &col) in perm.iter().enumerate() {
        if col < rows {
            matrix.set(i, col, Complex64::new(1.0, 0.0));
        }
    }
    matrix
}

const EPS: f64 = 1.0e-12;

fn matrix_to_value(matrix: &RowMajorMatrix) -> Result<Value, String> {
    let mut has_imag = false;
    for val in &matrix.data {
        if val.im.abs() > EPS {
            has_imag = true;
            break;
        }
    }
    if has_imag {
        let mut data = Vec::with_capacity(matrix.rows * matrix.cols);
        for col in 0..matrix.cols {
            for row in 0..matrix.rows {
                let idx = row * matrix.cols + col;
                let v = matrix.data[idx];
                data.push((v.re, v.im));
            }
        }
        let tensor = ComplexTensor::new(data, vec![matrix.rows, matrix.cols])
            .map_err(|e| format!("lu: {e}"))?;
        Ok(Value::ComplexTensor(tensor))
    } else {
        let mut data = Vec::with_capacity(matrix.rows * matrix.cols);
        for col in 0..matrix.cols {
            for row in 0..matrix.rows {
                let idx = row * matrix.cols + col;
                data.push(matrix.data[idx].re);
            }
        }
        let tensor =
            Tensor::new(data, vec![matrix.rows, matrix.cols]).map_err(|e| format!("lu: {e}"))?;
        Ok(Value::Tensor(tensor))
    }
}

fn pivot_vector_to_value(pivot: &[f64]) -> Result<Value, String> {
    let rows = pivot.len();
    let tensor = Tensor::new(pivot.to_vec(), vec![rows, 1]).map_err(|e| format!("lu: {e}"))?;
    Ok(Value::Tensor(tensor))
}

#[derive(Clone)]
struct RowMajorMatrix {
    rows: usize,
    cols: usize,
    data: Vec<Complex64>,
}

impl RowMajorMatrix {
    fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![Complex64::new(0.0, 0.0); rows.saturating_mul(cols)],
        }
    }

    fn identity(size: usize) -> Self {
        let mut matrix = Self::zeros(size, size);
        for i in 0..size {
            matrix.set(i, i, Complex64::new(1.0, 0.0));
        }
        matrix
    }

    fn from_scalar(value: Complex64) -> Self {
        Self {
            rows: 1,
            cols: 1,
            data: vec![value],
        }
    }

    fn from_tensor(tensor: &Tensor) -> Result<Self, String> {
        if tensor.shape.len() > 2 {
            return Err("lu: input must be 2-D".to_string());
        }
        let rows = tensor.rows();
        let cols = tensor.cols();
        let mut data = vec![Complex64::new(0.0, 0.0); rows.saturating_mul(cols)];
        for col in 0..cols {
            for row in 0..rows {
                let idx_col_major = row + col * rows;
                let idx_row_major = row * cols + col;
                data[idx_row_major] = Complex64::new(tensor.data[idx_col_major], 0.0);
            }
        }
        Ok(Self { rows, cols, data })
    }

    fn from_complex_tensor(tensor: &ComplexTensor) -> Result<Self, String> {
        if tensor.shape.len() > 2 {
            return Err("lu: input must be 2-D".to_string());
        }
        let rows = tensor.rows;
        let cols = tensor.cols;
        let mut data = vec![Complex64::new(0.0, 0.0); rows.saturating_mul(cols)];
        for col in 0..cols {
            for row in 0..rows {
                let idx_col_major = row + col * rows;
                let idx_row_major = row * cols + col;
                let (re, im) = tensor.data[idx_col_major];
                data[idx_row_major] = Complex64::new(re, im);
            }
        }
        Ok(Self { rows, cols, data })
    }

    fn get(&self, row: usize, col: usize) -> Complex64 {
        self.data[row * self.cols + col]
    }

    fn set(&mut self, row: usize, col: usize, value: Complex64) {
        self.data[row * self.cols + col] = value;
    }

    fn swap_rows(&mut self, r1: usize, r2: usize) {
        if r1 == r2 {
            return;
        }
        for col in 0..self.cols {
            self.data.swap(r1 * self.cols + col, r2 * self.cols + col);
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{ComplexTensor as CMatrix, Tensor as Matrix};

    fn tensor_from_value(value: Value) -> Matrix {
        match value {
            Value::Tensor(t) => t,
            other => panic!("expected dense tensor, got {other:?}"),
        }
    }

    fn row_major_from_value(value: Value) -> RowMajorMatrix {
        match value {
            Value::Tensor(t) => RowMajorMatrix::from_tensor(&t).expect("row-major tensor"),
            Value::ComplexTensor(ct) => {
                RowMajorMatrix::from_complex_tensor(&ct).expect("row-major complex tensor")
            }
            other => panic!("expected tensor value, got {other:?}"),
        }
    }

    fn row_major_matmul(a: &RowMajorMatrix, b: &RowMajorMatrix) -> RowMajorMatrix {
        assert_eq!(a.cols, b.rows, "incompatible shapes for matmul");
        let mut out = RowMajorMatrix::zeros(a.rows, b.cols);
        for i in 0..a.rows {
            for k in 0..a.cols {
                let aik = a.get(i, k);
                for j in 0..b.cols {
                    let acc = out.get(i, j) + aik * b.get(k, j);
                    out.set(i, j, acc);
                }
            }
        }
        out
    }

    fn assert_tensor_close(a: &Matrix, b: &Matrix, tol: f64) {
        assert_eq!(a.shape, b.shape);
        for (lhs, rhs) in a.data.iter().zip(&b.data) {
            assert!(
                (lhs - rhs).abs() <= tol,
                "mismatch: lhs={lhs}, rhs={rhs}, tol={tol}"
            );
        }
    }

    fn assert_row_major_close(a: &RowMajorMatrix, b: &RowMajorMatrix, tol: f64) {
        assert_eq!(a.rows, b.rows, "row mismatch");
        assert_eq!(a.cols, b.cols, "col mismatch");
        for row in 0..a.rows {
            for col in 0..a.cols {
                let lhs = a.get(row, col);
                let rhs = b.get(row, col);
                let diff = (lhs - rhs).norm();
                assert!(
                    diff <= tol,
                    "mismatch at ({row}, {col}): lhs={lhs:?}, rhs={rhs:?}, diff={diff}, tol={tol}"
                );
            }
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lu_single_output_produces_combined_matrix() {
        let a = Matrix::new(
            vec![2.0, 4.0, -2.0, 1.0, -6.0, 7.0, 1.0, 0.0, 2.0],
            vec![3, 3],
        )
        .unwrap();
        let result = lu_builtin(Value::Tensor(a.clone()), Vec::new()).expect("lu");
        let lu = tensor_from_value(result);
        let eval = evaluate(Value::Tensor(a), &[]).expect("evaluate");
        let expected = tensor_from_value(eval.combined());
        assert_tensor_close(&lu, &expected, 1e-12);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lu_three_outputs_matches_factorization() {
        let data = vec![2.0, 4.0, -2.0, 1.0, -6.0, 7.0, 1.0, 0.0, 2.0];
        let a = Matrix::new(data.clone(), vec![3, 3]).unwrap();
        let eval = evaluate(Value::Tensor(a.clone()), &[]).expect("evaluate");
        let l = tensor_from_value(eval.lower());
        let u = tensor_from_value(eval.upper());
        let p = tensor_from_value(eval.permutation_matrix());

        let pa = crate::matrix::matrix_mul(&p, &a).expect("P*A");
        let lu_product = crate::matrix::matrix_mul(&l, &u).expect("L*U");
        assert_tensor_close(&pa, &lu_product, 1e-9);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lu_complex_matrix_factorization() {
        let data = vec![(1.0, 2.0), (3.0, -1.0), (2.0, -1.0), (4.0, 2.0)];
        let a = CMatrix::new(data.clone(), vec![2, 2]).expect("complex tensor");
        let eval = evaluate(Value::ComplexTensor(a.clone()), &[]).expect("evaluate complex");

        let l = row_major_from_value(eval.lower());
        let u = row_major_from_value(eval.upper());
        let p = row_major_from_value(eval.permutation_matrix());
        let input = RowMajorMatrix::from_complex_tensor(&a).expect("row-major input");

        let pa = row_major_matmul(&p, &input);
        let lu = row_major_matmul(&l, &u);
        assert_row_major_close(&pa, &lu, 1e-9);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lu_handles_singular_matrix() {
        let a = Matrix::new(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]).unwrap();
        let eval = evaluate(Value::Tensor(a.clone()), &[]).expect("evaluate singular");
        let l = tensor_from_value(eval.lower());
        let u = tensor_from_value(eval.upper());
        let p = tensor_from_value(eval.permutation_matrix());

        assert!(u.data.iter().any(|&v| v.abs() <= 1e-12));

        let pa = crate::matrix::matrix_mul(&p, &a).expect("P*A");
        let lu_product = crate::matrix::matrix_mul(&l, &u).expect("L*U");
        assert_tensor_close(&pa, &lu_product, 1e-9);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lu_vector_option_returns_pivot_vector() {
        let a = Matrix::new(vec![4.0, 6.0, 3.0, 3.0], vec![2, 2]).unwrap();
        let eval =
            evaluate(Value::Tensor(a), &[Value::from("vector")]).expect("evaluate vector mode");
        assert_eq!(eval.pivot_mode(), PivotMode::Vector);
        let pivot = tensor_from_value(eval.pivot_vector());
        assert_eq!(pivot.shape, vec![2, 1]);
        assert_eq!(pivot.data, vec![2.0, 1.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lu_vector_option_case_insensitive() {
        let a = Matrix::new(vec![4.0, 6.0, 3.0, 3.0], vec![2, 2]).unwrap();
        let eval =
            evaluate(Value::Tensor(a), &[Value::from("VECTOR")]).expect("evaluate vector option");
        assert_eq!(eval.pivot_mode(), PivotMode::Vector);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lu_matrix_option_returns_permutation_matrix() {
        let a = Matrix::new(vec![2.0, 1.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let eval =
            evaluate(Value::Tensor(a), &[Value::from("matrix")]).expect("evaluate matrix option");
        assert_eq!(eval.pivot_mode(), PivotMode::Matrix);
        let perm_selected = tensor_from_value(eval.permutation());
        let perm_matrix = tensor_from_value(eval.permutation_matrix());
        assert_eq!(perm_selected.shape, perm_matrix.shape);
        assert_tensor_close(&perm_selected, &perm_matrix, 1e-12);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lu_handles_rectangular_matrices() {
        let a = Matrix::new(vec![3.0, 6.0, 1.0, 3.0, 2.0, 4.0], vec![2, 3]).unwrap();
        let eval = evaluate(Value::Tensor(a.clone()), &[]).expect("evaluate rectangular");
        let l = tensor_from_value(eval.lower());
        let u = tensor_from_value(eval.upper());
        let p = tensor_from_value(eval.permutation_matrix());
        assert_eq!(l.shape, vec![2, 2]);
        assert_eq!(u.shape, vec![2, 3]);
        assert_eq!(p.shape, vec![2, 2]);

        let pa = crate::matrix::matrix_mul(&p, &a).expect("P*A");
        let lu_product = crate::matrix::matrix_mul(&l, &u).expect("L*U");
        assert_tensor_close(&pa, &lu_product, 1e-9);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lu_rejects_unknown_option() {
        let a = Matrix::new(vec![1.0], vec![1, 1]).unwrap();
        let err = match evaluate(Value::Tensor(a), &[Value::from("invalid")]) {
            Ok(_) => panic!("expected option parse failure"),
            Err(err) => err,
        };
        assert!(err.contains("unknown option"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lu_rejects_non_string_option() {
        let a = Matrix::new(vec![1.0], vec![1, 1]).unwrap();
        let err = match evaluate(Value::Tensor(a), &[Value::Num(2.0)]) {
            Ok(_) => panic!("expected option parse failure"),
            Err(err) => err,
        };
        assert!(err.contains("unknown option"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lu_rejects_multiple_options() {
        let a = Matrix::new(vec![1.0], vec![1, 1]).unwrap();
        let err = match evaluate(
            Value::Tensor(a),
            &[Value::from("matrix"), Value::from("vector")],
        ) {
            Ok(_) => panic!("expected option arity failure"),
            Err(err) => err,
        };
        assert!(err.contains("too many option arguments"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lu_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let host = Matrix::new(vec![10.0, 3.0, 7.0, 2.0], vec![2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &host.data,
                shape: &host.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let eval = evaluate(Value::GpuTensor(handle.clone()), &[]).expect("evaluate gpu input");
            let lower_val = eval.lower();
            let upper_val = eval.upper();
            let perm_val = eval.permutation_matrix();
            assert!(matches!(lower_val, Value::GpuTensor(_)));
            assert!(matches!(upper_val, Value::GpuTensor(_)));
            assert!(matches!(perm_val, Value::GpuTensor(_)));
            let l = test_support::gather(lower_val).expect("gather lower");
            let u = test_support::gather(upper_val).expect("gather upper");
            let p = test_support::gather(perm_val).expect("gather permutation");
            let pa = crate::matrix::matrix_mul(&p, &host).expect("P*A");
            let lu_product = crate::matrix::matrix_mul(&l, &u).expect("L*U");
            assert_tensor_close(&pa, &lu_product, 1e-9);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lu_gpu_vector_option_roundtrip() {
        test_support::with_test_provider(|provider| {
            let host = Matrix::new(vec![4.0, 6.0, 3.0, 3.0], vec![2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &host.data,
                shape: &host.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let eval =
                evaluate(Value::GpuTensor(handle), &[Value::from("vector")]).expect("gpu vector");
            let pivot_val = eval.permutation();
            assert!(matches!(pivot_val, Value::GpuTensor(_)));
            let pivot = test_support::gather(pivot_val).expect("gather pivot");
            assert_eq!(pivot.shape, vec![2, 1]);
            let expected = Matrix::new(vec![2.0, 1.0], vec![2, 1]).unwrap();
            assert_tensor_close(&pivot, &expected, 1e-12);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lu_accepts_scalar_inputs() {
        let eval = evaluate(Value::Num(5.0), &[]).expect("evaluate scalar");
        let l = tensor_from_value(eval.lower());
        let u = tensor_from_value(eval.upper());
        let p = tensor_from_value(eval.permutation_matrix());
        assert_eq!(l.data, vec![1.0]);
        assert_eq!(u.data, vec![5.0]);
        assert_eq!(p.data, vec![1.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn lu_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let host = Matrix::new(
            vec![2.0, 4.0, -2.0, 1.0, -6.0, 7.0, 1.0, 0.0, 2.0],
            vec![3, 3],
        )
        .unwrap();
        let cpu_eval = evaluate(Value::Tensor(host.clone()), &[]).expect("cpu evaluate");
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let view = runmat_accelerate_api::HostTensorView {
            data: &host.data,
            shape: &host.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_eval = evaluate(Value::GpuTensor(handle), &[]).expect("gpu evaluate");

        let l_cpu = tensor_from_value(cpu_eval.lower());
        let u_cpu = tensor_from_value(cpu_eval.upper());
        let p_cpu = tensor_from_value(cpu_eval.permutation_matrix());
        let lu_cpu = tensor_from_value(cpu_eval.combined());

        let l_gpu = test_support::gather(gpu_eval.lower()).expect("gather L");
        let u_gpu = test_support::gather(gpu_eval.upper()).expect("gather U");
        let p_gpu = test_support::gather(gpu_eval.permutation_matrix()).expect("gather P");
        let lu_gpu = test_support::gather(gpu_eval.combined()).expect("gather LU");

        assert_tensor_close(&l_cpu, &l_gpu, 1e-12);
        assert_tensor_close(&u_cpu, &u_gpu, 1e-12);
        assert_tensor_close(&p_cpu, &p_gpu, 1e-12);
        assert_tensor_close(&lu_cpu, &lu_gpu, 1e-12);

        let pivot_cpu = tensor_from_value(cpu_eval.pivot_vector());
        let pivot_gpu = test_support::gather(gpu_eval.pivot_vector()).expect("gather pivot vector");
        assert_tensor_close(&pivot_cpu, &pivot_gpu, 1e-12);

        let handle_vector = provider.upload(&view).expect("upload vector option");
        let gpu_vector_eval = evaluate(Value::GpuTensor(handle_vector), &[Value::from("vector")])
            .expect("gpu vector evaluate");
        let pivot_vector =
            test_support::gather(gpu_vector_eval.permutation()).expect("gather vector pivot");
        assert_tensor_close(&pivot_cpu, &pivot_vector, 1e-12);
    }
}
