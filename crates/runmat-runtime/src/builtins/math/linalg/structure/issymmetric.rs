//! MATLAB-compatible `issymmetric` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{GpuTensorHandle, ProviderSymmetryKind};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, LogicalArray, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::linalg::type_resolvers::logical_scalar_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "issymmetric";

const ISSYMMETRIC_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when the matrix satisfies the selected symmetry predicate.",
}];

const ISSYMMETRIC_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input matrix.",
}];

const ISSYMMETRIC_INPUTS_OPTION: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input matrix.",
    },
    BuiltinParamDescriptor {
        name: "option",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Symmetry flag or tolerance scalar.",
    },
];

const ISSYMMETRIC_INPUTS_FLAG_TOL: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input matrix.",
    },
    BuiltinParamDescriptor {
        name: "flag",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Symmetry flag (for example \"skew\" or \"symmetric\").",
    },
    BuiltinParamDescriptor {
        name: "tol",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Tolerance for element-wise symmetry checks.",
    },
];

const ISSYMMETRIC_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "tf = issymmetric(A)",
        inputs: &ISSYMMETRIC_INPUTS,
        outputs: &ISSYMMETRIC_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "tf = issymmetric(A, option)",
        inputs: &ISSYMMETRIC_INPUTS_OPTION,
        outputs: &ISSYMMETRIC_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "tf = issymmetric(A, flag, tol)",
        inputs: &ISSYMMETRIC_INPUTS_FLAG_TOL,
        outputs: &ISSYMMETRIC_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "tf = issymmetric(A, tol, flag)",
        inputs: &ISSYMMETRIC_INPUTS_FLAG_TOL,
        outputs: &ISSYMMETRIC_OUTPUT,
    },
];

const ISSYMMETRIC_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISSYMMETRIC.INVALID_ARGUMENT",
    identifier: Some("RunMat:issymmetric:InvalidArgument"),
    when: "Flag/tolerance arguments are malformed, duplicated, or unsupported.",
    message: "issymmetric: invalid argument",
};

const ISSYMMETRIC_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISSYMMETRIC.INVALID_INPUT",
    identifier: Some("RunMat:issymmetric:InvalidInput"),
    when: "Input shape/type cannot be interpreted as a numeric or logical 2-D matrix.",
    message: "issymmetric: invalid input",
};

const ISSYMMETRIC_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISSYMMETRIC.INTERNAL",
    identifier: Some("RunMat:issymmetric:Internal"),
    when: "Runtime fails while creating internal tensors or coercing values.",
    message: "issymmetric: internal runtime failure",
};

const ISSYMMETRIC_ERRORS: [BuiltinErrorDescriptor; 3] = [
    ISSYMMETRIC_ERROR_INVALID_ARGUMENT,
    ISSYMMETRIC_ERROR_INVALID_INPUT,
    ISSYMMETRIC_ERROR_INTERNAL,
];

pub const ISSYMMETRIC_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ISSYMMETRIC_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ISSYMMETRIC_ERRORS,
};

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::math::linalg::structure::issymmetric"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "issymmetric",
    op_kind: GpuOpKind::Custom("structure_analysis"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("issymmetric")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may supply a symmetry predicate hook; otherwise the runtime gathers the tensor and evaluates on the host.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::linalg::structure::issymmetric"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "issymmetric",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Returns a host logical scalar and acts as a fusion sink.",
};

fn issymmetric_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn issymmetric_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    issymmetric_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

#[runtime_builtin(
    name = "issymmetric",
    category = "math/linalg/structure",
    summary = "Test whether a matrix is symmetric or skew-symmetric.",
    keywords = "issymmetric,symmetric,skew-symmetric,matrix structure,gpu",
    accel = "metadata",
    type_resolver(logical_scalar_type),
    descriptor(crate::builtins::math::linalg::structure::issymmetric::ISSYMMETRIC_DESCRIPTOR),
    builtin_path = "crate::builtins::math::linalg::structure::issymmetric"
)]
async fn issymmetric_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let (mode, tol) = parse_optional_args(&rest)?;
    match value {
        Value::GpuTensor(handle) => issymmetric_gpu(handle, mode, tol).await,
        other => {
            let matrix = MatrixInput::from_value(other).await?;
            let result = evaluate_matrix(matrix, mode, tol);
            Ok(Value::Bool(result))
        }
    }
}

async fn issymmetric_gpu(
    handle: GpuTensorHandle,
    mode: SymmetryMode,
    tol: f64,
) -> BuiltinResult<Value> {
    #[cfg(all(test, feature = "wgpu"))]
    {
        if handle.device_id != 0 {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }

    if let Some(provider) = runmat_accelerate_api::provider() {
        let kind = match mode {
            SymmetryMode::Symmetric => ProviderSymmetryKind::Symmetric,
            SymmetryMode::Skew => ProviderSymmetryKind::Skew,
        };
        let tried = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            provider.issymmetric(&handle, kind, tol)
        }));
        if let Ok(Ok(flag)) = tried {
            return Ok(Value::Bool(flag));
        }
        log::debug!("issymmetric: provider path failed or panicked; falling back to host");
    }

    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    let matrix = MatrixInput::from_value(Value::Tensor(tensor)).await?;
    let result = evaluate_matrix(matrix, mode, tol);
    Ok(Value::Bool(result))
}

#[derive(Clone, Copy)]
enum SymmetryMode {
    Symmetric,
    Skew,
}

struct MatrixInput {
    data: MatrixData,
    rows: usize,
    cols: usize,
}

enum MatrixData {
    Real(Tensor),
    Complex(ComplexTensor),
}

impl MatrixData {
    fn shape(&self) -> &[usize] {
        match self {
            MatrixData::Real(t) => &t.shape,
            MatrixData::Complex(t) => &t.shape,
        }
    }
}

impl MatrixInput {
    async fn from_value(value: Value) -> BuiltinResult<Self> {
        let data = match value {
            Value::Tensor(tensor) => MatrixData::Real(tensor),
            Value::LogicalArray(logical) => {
                let tensor = logical_to_tensor(&logical)?;
                MatrixData::Real(tensor)
            }
            Value::GpuTensor(handle) => {
                let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
                MatrixData::Real(tensor)
            }
            Value::ComplexTensor(tensor) => MatrixData::Complex(tensor),
            Value::Complex(re, im) => {
                let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                    .map_err(|e| issymmetric_error_with_detail(&ISSYMMETRIC_ERROR_INTERNAL, e))?;
                MatrixData::Complex(tensor)
            }
            Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
                let tensor = value_into_tensor_for(value)?;
                MatrixData::Real(tensor)
            }
            other => {
                return Err(issymmetric_error_with_detail(
                    &ISSYMMETRIC_ERROR_INVALID_INPUT,
                    format!(
                        "unsupported input type {:?}; expected numeric or logical matrix",
                        other
                    ),
                ));
            }
        };

        let shape = data.shape();
        let (rows, cols) = matrix_dimensions_for(shape)?;
        Ok(Self { data, rows, cols })
    }
}

fn evaluate_matrix(matrix: MatrixInput, mode: SymmetryMode, tol: f64) -> bool {
    if matrix.rows != matrix.cols {
        return false;
    }
    match matrix.data {
        MatrixData::Real(tensor) => is_symmetric_real(&tensor, mode, tol),
        MatrixData::Complex(tensor) => is_symmetric_complex(&tensor, mode, tol),
    }
}

fn parse_optional_args(args: &[Value]) -> BuiltinResult<(SymmetryMode, f64)> {
    if args.len() > 2 {
        return Err(issymmetric_error_with_detail(
            &ISSYMMETRIC_ERROR_INVALID_ARGUMENT,
            "too many input arguments",
        ));
    }

    let mut mode = SymmetryMode::Symmetric;
    let mut mode_set = false;
    let mut tol: Option<f64> = None;

    for arg in args {
        if let Some(flag) = parse_mode_flag(arg)? {
            if mode_set {
                return Err(issymmetric_error_with_detail(
                    &ISSYMMETRIC_ERROR_INVALID_ARGUMENT,
                    "duplicate symmetry flag",
                ));
            }
            mode = flag;
            mode_set = true;
            continue;
        }

        if tol.is_some() {
            return Err(issymmetric_error_with_detail(
                &ISSYMMETRIC_ERROR_INVALID_ARGUMENT,
                "tolerance specified more than once",
            ));
        }

        let local = parse_single_tolerance(arg)?;
        tol = Some(local);
    }

    Ok((mode, tol.unwrap_or(0.0)))
}

fn parse_mode_flag(value: &Value) -> BuiltinResult<Option<SymmetryMode>> {
    let text = match value {
        Value::String(s) => Some(s.clone()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].clone()),
        Value::CharArray(ca) if ca.rows == 1 => Some(ca.data.iter().collect()),
        _ => None,
    };

    let Some(raw) = text else {
        return Ok(None);
    };

    let lowered = raw.trim().to_ascii_lowercase();
    match lowered.as_str() {
        "skew" => Ok(Some(SymmetryMode::Skew)),
        "nonskew" | "symmetric" => Ok(Some(SymmetryMode::Symmetric)),
        other => Err(issymmetric_error_with_detail(
            &ISSYMMETRIC_ERROR_INVALID_ARGUMENT,
            format!("unknown flag '{other}'"),
        )),
    }
}

fn parse_single_tolerance(arg: &Value) -> BuiltinResult<f64> {
    let value = parse_tolerance_value(arg)?;
    Ok(value)
}

fn parse_tolerance_value(value: &Value) -> BuiltinResult<f64> {
    let raw = match value {
        Value::Num(n) => *n,
        Value::Int(i) => i.to_f64(),
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => t.data[0],
        Value::Bool(b) => {
            if *b {
                1.0
            } else {
                0.0
            }
        }
        Value::LogicalArray(l) if l.len() == 1 => {
            if l.data[0] != 0 {
                1.0
            } else {
                0.0
            }
        }
        other => {
            return Err(issymmetric_error_with_detail(
                &ISSYMMETRIC_ERROR_INVALID_ARGUMENT,
                format!("tolerance must be a real scalar, got {other:?}"),
            ))
        }
    };
    if !raw.is_finite() {
        return Err(issymmetric_error_with_detail(
            &ISSYMMETRIC_ERROR_INVALID_ARGUMENT,
            "tolerance must be finite",
        ));
    }
    if raw < 0.0 {
        return Err(issymmetric_error_with_detail(
            &ISSYMMETRIC_ERROR_INVALID_ARGUMENT,
            "tolerance must be >= 0",
        ));
    }
    Ok(raw)
}

fn matrix_dimensions_for(shape: &[usize]) -> BuiltinResult<(usize, usize)> {
    match shape.len() {
        0 => Ok((1, 1)),
        1 => Ok((shape[0], 1)),
        _ => {
            if shape.len() > 2 && shape.iter().skip(2).any(|&dim| dim != 1) {
                Err(issymmetric_error_with_detail(
                    &ISSYMMETRIC_ERROR_INVALID_INPUT,
                    "inputs must be 2-D matrices or vectors",
                ))
            } else {
                Ok((shape[0], shape[1]))
            }
        }
    }
}

fn value_into_tensor_for(value: Value) -> BuiltinResult<Tensor> {
    match value {
        Value::Tensor(t) => Ok(t),
        Value::LogicalArray(logical) => logical_to_tensor(&logical),
        Value::Num(n) => Tensor::new(vec![n], vec![1, 1])
            .map_err(|e| issymmetric_error_with_detail(&ISSYMMETRIC_ERROR_INTERNAL, e)),
        Value::Int(i) => Tensor::new(vec![i.to_f64()], vec![1, 1])
            .map_err(|e| issymmetric_error_with_detail(&ISSYMMETRIC_ERROR_INTERNAL, e)),
        Value::Bool(b) => Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
            .map_err(|e| issymmetric_error_with_detail(&ISSYMMETRIC_ERROR_INTERNAL, e)),
        other => Err(issymmetric_error_with_detail(
            &ISSYMMETRIC_ERROR_INVALID_INPUT,
            format!(
                "unsupported input type {:?}; expected numeric or logical values",
                other
            ),
        )),
    }
}

fn logical_to_tensor(logical: &LogicalArray) -> BuiltinResult<Tensor> {
    let data: Vec<f64> = logical
        .data
        .iter()
        .map(|&b| if b != 0 { 1.0 } else { 0.0 })
        .collect();
    Tensor::new(data, logical.shape.clone())
        .map_err(|e| issymmetric_error_with_detail(&ISSYMMETRIC_ERROR_INTERNAL, e))
}

fn is_symmetric_real(tensor: &Tensor, mode: SymmetryMode, tol: f64) -> bool {
    let rows = tensor.rows();
    let cols = tensor.cols();
    debug_assert_eq!(rows, cols, "is_symmetric_real requires a square matrix");
    let data = &tensor.data;

    for col in 0..cols {
        if matches!(mode, SymmetryMode::Skew) {
            let diag = data[col + col * rows];
            if !real_within(diag, 0.0, tol) {
                return false;
            }
        }
        for row in 0..col {
            let a = data[row + col * rows];
            let b = data[col + row * rows];
            let (diff, reference) = match mode {
                SymmetryMode::Symmetric => (a, b),
                SymmetryMode::Skew => (a, -b),
            };
            if !real_within(diff, reference, tol) {
                return false;
            }
        }
    }
    true
}

fn is_symmetric_complex(tensor: &ComplexTensor, mode: SymmetryMode, tol: f64) -> bool {
    let rows = tensor.rows;
    let cols = tensor.cols;
    debug_assert_eq!(rows, cols, "is_symmetric_complex requires a square matrix");
    let data = &tensor.data;

    for col in 0..cols {
        if matches!(mode, SymmetryMode::Skew) {
            let (re, im) = data[col + col * rows];
            if !complex_within(re, im, 0.0, 0.0, tol) {
                return false;
            }
        }
        for row in 0..col {
            let (ar, ai) = data[row + col * rows];
            let (br, bi) = data[col + row * rows];
            let (target_r, target_i) = match mode {
                SymmetryMode::Symmetric => (br, bi),
                SymmetryMode::Skew => (-br, -bi),
            };
            if !complex_within(ar, ai, target_r, target_i, tol) {
                return false;
            }
        }
    }
    true
}

fn real_within(value: f64, reference: f64, tol: f64) -> bool {
    if value == reference {
        return true;
    }
    if !value.is_finite() || !reference.is_finite() {
        return false;
    }
    let diff = (value - reference).abs();
    diff <= tol
}

fn complex_within(re: f64, im: f64, ref_re: f64, ref_im: f64, tol: f64) -> bool {
    if re == ref_re && im == ref_im {
        return true;
    }
    if !re.is_finite() || !im.is_finite() || !ref_re.is_finite() || !ref_im.is_finite() {
        return false;
    }
    let diff_r = re - ref_re;
    let diff_i = im - ref_im;
    diff_r.hypot(diff_i) <= tol
}

pub fn ensure_matrix_shape(shape: &[usize]) -> BuiltinResult<(usize, usize)> {
    matrix_dimensions_for(shape)
}

pub fn issymmetric_host_real_tensor(tensor: &Tensor, skew: bool, tol: f64) -> BuiltinResult<bool> {
    let (rows, cols) = matrix_dimensions_for(&tensor.shape)?;
    if rows != cols {
        return Ok(false);
    }
    let mode = if skew {
        SymmetryMode::Skew
    } else {
        SymmetryMode::Symmetric
    };
    Ok(is_symmetric_real(tensor, mode, tol))
}

pub fn issymmetric_host_complex_tensor(
    tensor: &ComplexTensor,
    skew: bool,
    tol: f64,
) -> BuiltinResult<bool> {
    let (rows, cols) = matrix_dimensions_for(&tensor.shape)?;
    if rows != cols {
        return Ok(false);
    }
    let mode = if skew {
        SymmetryMode::Skew
    } else {
        SymmetryMode::Symmetric
    };
    Ok(is_symmetric_complex(tensor, mode, tol))
}

pub fn issymmetric_host_real_data(
    shape: &[usize],
    data: &[f64],
    skew: bool,
    tol: f64,
) -> BuiltinResult<bool> {
    let (rows, cols) = matrix_dimensions_for(shape)?;
    if rows != cols {
        return Ok(false);
    }
    let tensor = Tensor::new(data.to_vec(), shape.to_vec())
        .map_err(|e| issymmetric_error_with_detail(&ISSYMMETRIC_ERROR_INTERNAL, e))?;
    issymmetric_host_real_tensor(&tensor, skew, tol)
}

pub fn issymmetric_host_complex_data(
    shape: &[usize],
    data: &[(f64, f64)],
    skew: bool,
    tol: f64,
) -> BuiltinResult<bool> {
    let (rows, cols) = matrix_dimensions_for(shape)?;
    if rows != cols {
        return Ok(false);
    }
    let tensor = ComplexTensor::new(data.to_vec(), shape.to_vec())
        .map_err(|e| issymmetric_error_with_detail(&ISSYMMETRIC_ERROR_INTERNAL, e))?;
    issymmetric_host_complex_tensor(&tensor, skew, tol)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider as wgpu_provider;
    use runmat_builtins::{IntValue, LogicalArray, ResolveContext, Type};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn symmetric_matrix_returns_true() {
        let tensor = Tensor::new(
            vec![2.0, 1.0, 1.0, 1.0, 3.0, 4.0, 1.0, 4.0, 5.0],
            vec![3, 3],
        )
        .unwrap();
        let result = issymmetric_builtin(Value::Tensor(tensor), Vec::new()).expect("issymmetric");
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn issymmetric_type_returns_bool() {
        let out = logical_scalar_type(
            &[Type::Tensor {
                shape: Some(vec![Some(2), Some(2)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(out, Type::Bool);
    }

    #[test]
    fn issymmetric_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = ISSYMMETRIC_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert!(labels.contains(&"tf = issymmetric(A)"));
        assert!(labels.contains(&"tf = issymmetric(A, option)"));
        assert!(labels.contains(&"tf = issymmetric(A, flag, tol)"));
        assert!(labels.contains(&"tf = issymmetric(A, tol, flag)"));
    }

    #[test]
    fn issymmetric_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = ISSYMMETRIC_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert!(codes.contains(&"RM.ISSYMMETRIC.INVALID_ARGUMENT"));
        assert!(codes.contains(&"RM.ISSYMMETRIC.INVALID_INPUT"));
        assert!(codes.contains(&"RM.ISSYMMETRIC.INTERNAL"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn nonsymmetric_matrix_returns_false() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = issymmetric_builtin(Value::Tensor(tensor), Vec::new()).expect("issymmetric");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn skew_symmetric_flag_requires_zero_diagonal() {
        let tensor = Tensor::new(vec![0.0, 2.0, -2.0, 0.0], vec![2, 2]).unwrap();
        let result =
            issymmetric_builtin(Value::Tensor(tensor), vec![Value::from("skew")]).expect("skew");
        assert_eq!(result, Value::Bool(true));

        let tensor = Tensor::new(vec![1.0, -2.0, 2.0, 1.0], vec![2, 2]).unwrap();
        let result =
            issymmetric_builtin(Value::Tensor(tensor), vec![Value::from("skew")]).expect("skew");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tolerance_allows_small_deviations() {
        let tensor = Tensor::new(vec![1.0, 1.0 + 1e-12, 1.0 - 1e-12, 1.0], vec![2, 2]).unwrap();
        let result = issymmetric_builtin(Value::Tensor(tensor), vec![Value::Num(1e-9)])
            .expect("issymmetric");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_matrix_symmetry() {
        let tensor = ComplexTensor::new(
            vec![(1.0, 2.0), (3.0, -4.0), (3.0, -4.0), (5.0, 6.0)],
            vec![2, 2],
        )
        .unwrap();
        let result =
            issymmetric_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("issymmetric");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_matrix_promoted() {
        let logical = LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).unwrap();
        let result =
            issymmetric_builtin(Value::LogicalArray(logical), Vec::new()).expect("issymmetric");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn non_square_returns_false() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let result = issymmetric_builtin(Value::Tensor(tensor), Vec::new()).expect("issymmetric");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn higher_dimensional_inputs_error() {
        let tensor = Tensor::new(vec![1.0; 8], vec![2, 2, 2]).unwrap();
        let err = issymmetric_builtin(Value::Tensor(tensor), Vec::new()).unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("inputs must be 2-D matrices"),
            "unexpected error message: {message}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tolerance_and_flag_in_any_order() {
        let tensor = Tensor::new(vec![0.0, 1.0, -1.0000000001, 0.0], vec![2, 2]).unwrap();
        let result = issymmetric_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::from("skew"), Value::Num(1e-9)],
        )
        .expect("issymmetric");
        assert_eq!(result, Value::Bool(true));

        let result = issymmetric_builtin(
            Value::Tensor(tensor),
            vec![Value::Num(1e-9), Value::from("skew")],
        )
        .expect("issymmetric");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn invalid_flag_errors() {
        let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let err =
            issymmetric_builtin(Value::Tensor(tensor), vec![Value::from("diagonal")]).unwrap_err();
        assert_eq!(
            err.identifier(),
            ISSYMMETRIC_ERROR_INVALID_ARGUMENT.identifier
        );
        let message = err.to_string();
        assert!(
            message.contains("unknown flag"),
            "unexpected error message: {message}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn duplicate_tolerance_errors() {
        let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let err = issymmetric_builtin(
            Value::Tensor(tensor),
            vec![Value::Num(1e-9), Value::Num(1e-6)],
        )
        .unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("tolerance specified more than once"),
            "unexpected error message: {message}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn negative_tolerance_errors() {
        let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let err = issymmetric_builtin(Value::Tensor(tensor), vec![Value::Num(-1.0)]).unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("tolerance must be >= 0"),
            "unexpected error message: {message}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn scalar_inputs_follow_rules() {
        let result = issymmetric_builtin(Value::Num(5.0), Vec::new()).expect("issymmetric scalar");
        assert_eq!(result, Value::Bool(true));

        let result = issymmetric_builtin(Value::Int(IntValue::I32(0)), vec![Value::from("skew")])
            .expect("issymmetric skew scalar");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn issymmetric_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![2.0, 1.0, 1.0, 3.0], vec![2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result =
                issymmetric_builtin(Value::GpuTensor(handle), Vec::new()).expect("issymmetric");
            assert_eq!(result, Value::Bool(true));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn issymmetric_wgpu_matches_cpu() {
        let _ =
            wgpu_provider::register_wgpu_provider(wgpu_provider::WgpuProviderOptions::default());

        let tensor = Tensor::new(vec![1.0, 2.0, 2.0, 3.0], vec![2, 2]).unwrap();
        let cpu = issymmetric_builtin(Value::Tensor(tensor.clone()), Vec::new()).unwrap();

        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let provider = runmat_accelerate_api::provider().unwrap();
        let handle = provider.upload(&view).unwrap();
        let gpu = issymmetric_builtin(Value::GpuTensor(handle.clone()), Vec::new()).unwrap();
        assert_eq!(cpu, gpu);

        let skew = Tensor::new(vec![0.0, 1.0, -1.0 - 1.0e-9, 0.0], vec![2, 2]).unwrap();
        let cpu_skew = issymmetric_builtin(
            Value::Tensor(skew.clone()),
            vec![Value::from("skew"), Value::Num(1.0e-6)],
        )
        .unwrap();
        let view_skew = runmat_accelerate_api::HostTensorView {
            data: &skew.data,
            shape: &skew.shape,
        };
        let handle_skew = provider.upload(&view_skew).unwrap();
        let gpu_skew = issymmetric_builtin(
            Value::GpuTensor(handle_skew.clone()),
            vec![Value::from("skew"), Value::Num(1.0e-6)],
        )
        .unwrap();
        assert_eq!(cpu_skew, gpu_skew);

        let _ = provider.free(&handle);
        let _ = provider.free(&handle_skew);
    }

    fn issymmetric_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::issymmetric_builtin(value, rest))
    }
}
