//! MATLAB-compatible `conv2` builtin with GPU-aware semantics for RunMat.

use num_complex::Complex;
use runmat_accelerate_api::ProviderConvMode;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::builtins::math::signal::type_resolvers::conv2_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const EPS: f64 = 1e-12;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::signal::conv2")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "conv2",
    op_kind: GpuOpKind::Custom("conv2d"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("conv2d")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers can keep results on-device by implementing a conv2d custom hook; absent that, the builtin gathers to the host for CPU execution.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::signal::conv2")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "conv2",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Currently implemented as a standalone op; future work may add FFT-backed or fused variants.",
};

const BUILTIN_NAME: &str = "conv2";

const CONV2_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "2-D convolution result.",
}];

const CONV2_SIG_AB_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First matrix input.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second matrix input.",
    },
];

const CONV2_SIG_AB_SHAPE_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First matrix input.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second matrix input.",
    },
    BuiltinParamDescriptor {
        name: "shape",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"full\""),
        description: "Output shape: \"full\", \"same\", or \"valid\".",
    },
];

const CONV2_SIG_SEPARABLE_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "hcol",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Column vector kernel component.",
    },
    BuiltinParamDescriptor {
        name: "hrow",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Row vector kernel component.",
    },
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input matrix.",
    },
];

const CONV2_SIG_SEPARABLE_SHAPE_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "hcol",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Column vector kernel component.",
    },
    BuiltinParamDescriptor {
        name: "hrow",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Row vector kernel component.",
    },
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input matrix.",
    },
    BuiltinParamDescriptor {
        name: "shape",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"full\""),
        description: "Output shape: \"full\", \"same\", or \"valid\".",
    },
];

const CONV2_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "C = conv2(A, B)",
        inputs: &CONV2_SIG_AB_INPUTS,
        outputs: &CONV2_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = conv2(A, B, shape)",
        inputs: &CONV2_SIG_AB_SHAPE_INPUTS,
        outputs: &CONV2_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = conv2(hcol, hrow, A)",
        inputs: &CONV2_SIG_SEPARABLE_INPUTS,
        outputs: &CONV2_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = conv2(hcol, hrow, A, shape)",
        inputs: &CONV2_SIG_SEPARABLE_SHAPE_INPUTS,
        outputs: &CONV2_OUTPUT,
    },
];

const CONV2_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONV2.ARG_COUNT",
    identifier: Some("RunMat:conv2:ArgCount"),
    when: "More than four input arguments are provided.",
    message: "conv2: expected at most four input arguments",
};

const CONV2_ERROR_SHAPE_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONV2.SHAPE_INVALID",
    identifier: Some("RunMat:conv2:ShapeInvalid"),
    when: "Shape argument is not one of full/same/valid.",
    message: "conv2: shape argument must be the string 'full', 'same', or 'valid'",
};

const CONV2_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONV2.INVALID_INPUT",
    identifier: Some("RunMat:conv2:InvalidInput"),
    when: "An operand is not numeric/logical scalar/vector/matrix compatible.",
    message: "conv2: unsupported input type",
};

const CONV2_ERROR_VECTOR_REQUIRED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONV2.VECTOR_REQUIRED",
    identifier: Some("RunMat:conv2:VectorRequired"),
    when: "Separable hcol/hrow inputs are not vectors.",
    message: "conv2: vector input required",
};

const CONV2_ERROR_MATRIX_REQUIRED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONV2.MATRIX_REQUIRED",
    identifier: Some("RunMat:conv2:MatrixRequired"),
    when: "Input matrix has non-singleton dimensions beyond 2-D.",
    message: "conv2: input must be 2-D",
};

const CONV2_ERROR_CONVERSION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONV2.CONVERSION",
    identifier: Some("RunMat:conv2:Conversion"),
    when: "Input conversion from logical/gpu tensor into host matrix domain fails.",
    message: "conv2: input conversion failed",
};

const CONV2_ERROR_GATHER_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONV2.GATHER_FAILED",
    identifier: Some("RunMat:conv2:GatherFailed"),
    when: "GPU input cannot be gathered for host fallback normalization.",
    message: "conv2: failed to gather GPU input",
};

const CONV2_ERROR_BUILD_OUTPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONV2.BUILD_OUTPUT",
    identifier: Some("RunMat:conv2:BuildOutput"),
    when: "Output tensor allocation fails.",
    message: "conv2: failed to build tensor",
};

const CONV2_ERROR_BUILD_COMPLEX_OUTPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONV2.BUILD_COMPLEX_OUTPUT",
    identifier: Some("RunMat:conv2:BuildComplexOutput"),
    when: "Output complex tensor allocation fails.",
    message: "conv2: failed to build complex tensor",
};

const CONV2_ERRORS: [BuiltinErrorDescriptor; 9] = [
    CONV2_ERROR_ARG_COUNT,
    CONV2_ERROR_SHAPE_INVALID,
    CONV2_ERROR_INVALID_INPUT,
    CONV2_ERROR_VECTOR_REQUIRED,
    CONV2_ERROR_MATRIX_REQUIRED,
    CONV2_ERROR_CONVERSION,
    CONV2_ERROR_GATHER_FAILED,
    CONV2_ERROR_BUILD_OUTPUT,
    CONV2_ERROR_BUILD_COMPLEX_OUTPUT,
];

pub const CONV2_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CONV2_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CONV2_ERRORS,
};

fn conv2_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    conv2_error_with_message(error.message, error)
}

fn conv2_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    conv2_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn conv2_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn conv2_error_with_source(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
    source: RuntimeError,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "conv2",
    category = "math/signal",
    summary = "Two-dimensional convolution with MATLAB-compatible padding modes.",
    keywords = "conv2,2d convolution,image filtering,gpu",
    accel = "custom",
    type_resolver(conv2_type),
    descriptor(crate::builtins::math::signal::conv2::CONV2_DESCRIPTOR),
    builtin_path = "crate::builtins::math::signal::conv2"
)]
async fn conv2_builtin(a: Value, b: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let mut extras = rest;
    let mode = extract_mode(&mut extras)?;

    match extras.len() {
        0 => {
            if let Some(device_value) = try_conv2_gpu(&a, &b, mode)? {
                return Ok(device_value);
            }
            let left = convert_matrix(a, "conv2", "A").await?;
            let right = convert_matrix(b, "conv2", "B").await?;
            if mode == Conv2Mode::Full && left.cols == 1 && right.rows == 1 {
                let result = outer_product(&left.data, &right.data);
                return matrix_to_value(result);
            }
            let result = conv2_matrices(&left, &right, mode);
            matrix_to_value(result)
        }
        1 => {
            let signal = convert_matrix(extras.remove(0), "conv2", "A").await?;
            let column = convert_vector(a, "conv2", "H column").await?;
            let row = convert_vector(b, "conv2", "H row").await?;
            let kernel = outer_product(&column, &row);
            let result = conv2_matrices(&kernel, &signal, mode);
            matrix_to_value(result)
        }
        _ => Err(conv2_error(&CONV2_ERROR_ARG_COUNT)),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Conv2Mode {
    Full,
    Same,
    Valid,
}

fn try_conv2_gpu(a: &Value, b: &Value, mode: Conv2Mode) -> BuiltinResult<Option<Value>> {
    let provider = match runmat_accelerate_api::provider() {
        Some(p) => p,
        None => return Ok(None),
    };

    let (lhs, rhs) = match (a, b) {
        (Value::GpuTensor(lhs), Value::GpuTensor(rhs)) => (lhs, rhs),
        _ => return Ok(None),
    };

    #[cfg(all(test, feature = "wgpu"))]
    {
        if lhs.device_id != 0 || rhs.device_id != 0 {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }

    let lhs_dims = match conv2_dimensions(&lhs.shape) {
        Some(dims) => dims,
        None => return Ok(None),
    };
    let rhs_dims = match conv2_dimensions(&rhs.shape) {
        Some(dims) => dims,
        None => return Ok(None),
    };

    // Keep the explicit full-mode vector kernel path on the host so it can
    // return an outer-product kernel that matches MATLAB's separable identity.
    if mode == Conv2Mode::Full && lhs_dims.1 == 1 && rhs_dims.0 == 1 {
        return Ok(None);
    }

    // If either operand is effectively empty we can still defer to the provider, which will
    // honour MATLAB's shape rules. No additional guarding is required here.
    let provider_mode = match mode {
        Conv2Mode::Full => ProviderConvMode::Full,
        Conv2Mode::Same => ProviderConvMode::Same,
        Conv2Mode::Valid => ProviderConvMode::Valid,
    };

    match provider.conv2d(lhs, rhs, provider_mode) {
        Ok(handle) => Ok(Some(Value::GpuTensor(handle))),
        Err(err) => {
            log::trace!("conv2: provider conv2d unavailable, falling back to host: {err}");
            Ok(None)
        }
    }
}

fn conv2_dimensions(shape: &[usize]) -> Option<(usize, usize)> {
    match shape.len() {
        0 => Some((1, 1)),
        1 => Some((shape[0], 1)),
        2 => Some((shape[0], shape[1])),
        _ => {
            if shape.iter().skip(2).all(|&dim| dim == 1) {
                Some((shape[0], shape[1]))
            } else {
                None
            }
        }
    }
}

#[derive(Clone)]
struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<Complex<f64>>,
}

impl Matrix {
    fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![Complex::new(0.0, 0.0); rows.saturating_mul(cols)],
        }
    }

    fn is_empty(&self) -> bool {
        self.rows == 0 || self.cols == 0
    }

    #[inline]
    fn index(&self, row: usize, col: usize) -> usize {
        col * self.rows + row
    }

    #[inline]
    fn get(&self, row: usize, col: usize) -> Complex<f64> {
        self.data[self.index(row, col)]
    }

    #[inline]
    fn add_assign(&mut self, row: usize, col: usize, value: Complex<f64>) {
        let idx = self.index(row, col);
        self.data[idx] += value;
    }

    fn slice(&self, row_start: usize, row_end: usize, col_start: usize, col_end: usize) -> Self {
        let row_end = row_end.min(self.rows);
        let col_end = col_end.min(self.cols);
        if row_start >= row_end || col_start >= col_end {
            return Self::zeros(
                row_end.saturating_sub(row_start),
                col_end.saturating_sub(col_start),
            );
        }
        let rows = row_end - row_start;
        let cols = col_end - col_start;
        let mut data = vec![Complex::new(0.0, 0.0); rows * cols];
        for c in 0..cols {
            for r in 0..rows {
                let value = self.get(row_start + r, col_start + c);
                data[c * rows + r] = value;
            }
        }
        Self { rows, cols, data }
    }
}

fn extract_mode(extras: &mut Vec<Value>) -> BuiltinResult<Conv2Mode> {
    if let Some(mode) = extras
        .last()
        .and_then(|last| parse_mode_value(last).transpose())
        .transpose()?
    {
        extras.pop();
        return Ok(mode);
    }
    Ok(Conv2Mode::Full)
}

fn parse_mode_value(value: &Value) -> BuiltinResult<Option<Conv2Mode>> {
    let Some(text) = tensor::value_to_string(value) else {
        return Ok(None);
    };
    let lowered = text.trim().to_ascii_lowercase();
    let mode = match lowered.as_str() {
        "full" => Conv2Mode::Full,
        "same" => Conv2Mode::Same,
        "valid" => Conv2Mode::Valid,
        _ => {
            return Err(conv2_error(&CONV2_ERROR_SHAPE_INVALID));
        }
    };
    Ok(Some(mode))
}

async fn convert_matrix(value: Value, name: &str, arg: &str) -> BuiltinResult<Matrix> {
    match value {
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(|flow| {
                    let message = flow.message().to_owned();
                    conv2_error_with_source(
                        &CONV2_ERROR_GATHER_FAILED,
                        message,
                        map_control_flow_with_builtin(flow, BUILTIN_NAME),
                    )
                })?;
            tensor_to_matrix(tensor, name, arg)
        }
        Value::Tensor(tensor) => tensor_to_matrix(tensor, name, arg),
        Value::ComplexTensor(tensor) => complex_tensor_to_matrix(tensor, name, arg),
        Value::LogicalArray(logical) => tensor::logical_to_tensor(&logical)
            .map_err(|err| {
                conv2_error_with_detail(&CONV2_ERROR_CONVERSION, format!("{name}: {err}"))
            })
            .and_then(|tensor| tensor_to_matrix(tensor, name, arg)),
        Value::Num(n) => Ok(Matrix {
            rows: 1,
            cols: 1,
            data: vec![Complex::new(n, 0.0)],
        }),
        Value::Int(i) => Ok(Matrix {
            rows: 1,
            cols: 1,
            data: vec![Complex::new(i.to_f64(), 0.0)],
        }),
        Value::Bool(b) => Ok(Matrix {
            rows: 1,
            cols: 1,
            data: vec![Complex::new(if b { 1.0 } else { 0.0 }, 0.0)],
        }),
        Value::Complex(re, im) => Ok(Matrix {
            rows: 1,
            cols: 1,
            data: vec![Complex::new(re, im)],
        }),
        other => Err(conv2_error_with_detail(
            &CONV2_ERROR_INVALID_INPUT,
            format!("{name}/{arg}: expected numeric or logical values, got {other:?}"),
        )),
    }
}

async fn convert_vector(value: Value, name: &str, arg: &str) -> BuiltinResult<Vec<Complex<f64>>> {
    let matrix = convert_matrix(value, name, arg).await?;
    if matrix.rows > 1 && matrix.cols > 1 {
        return Err(conv2_error_with_detail(
            &CONV2_ERROR_VECTOR_REQUIRED,
            format!("{name}: {arg} got {}x{}", matrix.rows, matrix.cols),
        ));
    }
    Ok(matrix.data)
}

fn tensor_to_matrix(tensor: Tensor, name: &str, arg: &str) -> BuiltinResult<Matrix> {
    if tensor.shape.iter().skip(2).any(|&dim| dim > 1) {
        return Err(conv2_error_with_detail(
            &CONV2_ERROR_MATRIX_REQUIRED,
            format!("{name}: {arg} received shape {:?}", tensor.shape),
        ));
    }
    Ok(Matrix {
        rows: tensor.rows,
        cols: tensor.cols,
        data: tensor
            .data
            .into_iter()
            .map(|re| Complex::new(re, 0.0))
            .collect(),
    })
}

fn complex_tensor_to_matrix(tensor: ComplexTensor, name: &str, arg: &str) -> BuiltinResult<Matrix> {
    if tensor.shape.iter().skip(2).any(|&dim| dim > 1) {
        return Err(conv2_error_with_detail(
            &CONV2_ERROR_MATRIX_REQUIRED,
            format!("{name}: {arg} received shape {:?}", tensor.shape),
        ));
    }
    Ok(Matrix {
        rows: tensor.rows,
        cols: tensor.cols,
        data: tensor
            .data
            .into_iter()
            .map(|(re, im)| Complex::new(re, im))
            .collect(),
    })
}

fn outer_product(column: &[Complex<f64>], row: &[Complex<f64>]) -> Matrix {
    let rows = column.len();
    let cols = row.len();
    let mut data = vec![Complex::new(0.0, 0.0); rows.saturating_mul(cols)];
    for c in 0..cols {
        for r in 0..rows {
            data[c * rows + r] = column[r] * row[c];
        }
    }
    Matrix { rows, cols, data }
}

fn conv2_matrices(a: &Matrix, b: &Matrix, mode: Conv2Mode) -> Matrix {
    if a.is_empty() || b.is_empty() {
        return empty_result(a, b, mode);
    }

    let rows = a.rows + b.rows - 1;
    let cols = a.cols + b.cols - 1;
    let mut full = Matrix::zeros(rows, cols);
    for ac in 0..a.cols {
        for ar in 0..a.rows {
            let aval = a.get(ar, ac);
            for bc in 0..b.cols {
                let out_c = ac + bc;
                let bcol = b.cols - 1 - bc;
                for br in 0..b.rows {
                    let out_r = ar + br;
                    let brow = b.rows - 1 - br;
                    let bval = b.get(brow, bcol);
                    full.add_assign(out_r, out_c, aval * bval);
                }
            }
        }
    }

    match mode {
        Conv2Mode::Full => full,
        Conv2Mode::Same => {
            if a.is_empty() {
                return Matrix::zeros(a.rows, a.cols);
            }
            let row_start = (b.rows - 1) / 2;
            let col_start = (b.cols - 1) / 2;
            full.slice(row_start, row_start + a.rows, col_start, col_start + a.cols)
        }
        Conv2Mode::Valid => {
            if a.rows < b.rows || a.cols < b.cols {
                return Matrix::zeros(0, 0);
            }
            let rows = a.rows - b.rows + 1;
            let cols = a.cols - b.cols + 1;
            let row_start = b.rows - 1;
            let col_start = b.cols - 1;
            full.slice(row_start, row_start + rows, col_start, col_start + cols)
        }
    }
}

fn empty_result(a: &Matrix, _b: &Matrix, mode: Conv2Mode) -> Matrix {
    match mode {
        Conv2Mode::Full | Conv2Mode::Valid => Matrix::zeros(0, 0),
        Conv2Mode::Same => Matrix::zeros(a.rows, a.cols),
    }
}

fn matrix_to_value(matrix: Matrix) -> BuiltinResult<Value> {
    let rows = matrix.rows;
    let cols = matrix.cols;
    let all_real = matrix.data.iter().all(|c| c.im.abs() <= EPS);

    if all_real {
        let real_data: Vec<f64> = matrix.data.into_iter().map(|c| c.re).collect();
        let tensor = Tensor::new(real_data, vec![rows, cols])
            .map_err(|e| conv2_error_with_detail(&CONV2_ERROR_BUILD_OUTPUT, &e))?;
        return Ok(tensor::tensor_into_value(tensor));
    }

    let complex_data: Vec<(f64, f64)> = matrix.data.into_iter().map(|c| (c.re, c.im)).collect();
    let tensor = ComplexTensor::new(complex_data, vec![rows, cols])
        .map_err(|e| conv2_error_with_detail(&CONV2_ERROR_BUILD_COMPLEX_OUTPUT, &e))?;
    if tensor.data.len() == 1 {
        let (re, im) = tensor.data[0];
        if im.abs() <= EPS {
            return Ok(Value::Num(re));
        }
        return Ok(Value::Complex(re, im));
    }
    Ok(Value::ComplexTensor(tensor))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::{tensor, test_support};
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{builtin_function_by_name, LogicalArray, ResolveContext, Type};

    fn error_message(error: RuntimeError) -> String {
        error.message().to_string()
    }

    fn tensor_from_rows(rows: usize, cols: usize, data: &[f64]) -> Tensor {
        assert_eq!(rows * cols, data.len());
        // Convert from row-major (provided for readability) to column-major.
        let mut col_major = vec![0.0; data.len()];
        for r in 0..rows {
            for c in 0..cols {
                col_major[c * rows + r] = data[r * cols + c];
            }
        }
        Tensor::new(col_major, vec![rows, cols]).unwrap()
    }

    #[test]
    fn conv2_type_full_uses_dimensions() {
        let out = conv2_type(
            &[
                Type::Tensor {
                    shape: Some(vec![Some(3), Some(2)]),
                },
                Type::Tensor {
                    shape: Some(vec![Some(2), Some(1)]),
                },
            ],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(4), Some(2)])
            }
        );
    }

    #[test]
    fn conv2_descriptor_signatures_and_errors() {
        let builtin = builtin_function_by_name(BUILTIN_NAME).expect("conv2 builtin");
        let descriptor = builtin.descriptor.expect("conv2 descriptor");
        let labels: Vec<&str> = descriptor.signatures.iter().map(|sig| sig.label).collect();
        assert!(labels.contains(&"C = conv2(A, B)"));
        assert!(labels.contains(&"C = conv2(A, B, shape)"));
        assert!(labels.contains(&"C = conv2(hcol, hrow, A)"));
        assert!(labels.contains(&"C = conv2(hcol, hrow, A, shape)"));
        assert!(descriptor
            .errors
            .iter()
            .any(|err| err.code == "RM.CONV2.SHAPE_INVALID"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv2_full_basic() {
        let a = tensor_from_rows(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let b = tensor_from_rows(2, 2, &[1.0, 1.0, 1.0, 1.0]);
        let result = conv2_builtin(Value::Tensor(a), Value::Tensor(b), Vec::new()).expect("conv2");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 3]);
                let expected =
                    tensor_from_rows(3, 3, &[1.0, 3.0, 2.0, 4.0, 10.0, 6.0, 3.0, 7.0, 4.0]);
                assert_eq!(t.data, expected.data);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv2_same_matches_reference() {
        let a = tensor_from_rows(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let b = tensor_from_rows(3, 3, &[1.0; 9]);
        let result = conv2_builtin(
            Value::Tensor(a),
            Value::Tensor(b),
            vec![Value::from("same")],
        )
        .expect("conv2 same");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 3]);
                let expected = tensor_from_rows(
                    3,
                    3,
                    &[12.0, 21.0, 16.0, 27.0, 45.0, 33.0, 24.0, 39.0, 28.0],
                );
                assert_eq!(t.data, expected.data);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv2_same_flips_kernel() {
        let a = tensor_from_rows(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let b = tensor_from_rows(3, 3, &[1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0]);
        let result = conv2_builtin(
            Value::Tensor(a),
            Value::Tensor(b),
            vec![Value::from("same")],
        )
        .expect("conv2 same");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 3]);
                let expected = tensor_from_rows(
                    3,
                    3,
                    &[-7.0, -4.0, 7.0, -15.0, -6.0, 15.0, -13.0, -4.0, 13.0],
                );
                assert_eq!(t.data, expected.data);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv2_valid_returns_expected_sum() {
        let a = tensor_from_rows(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let b = tensor_from_rows(3, 3, &[1.0; 9]);
        let result = conv2_builtin(
            Value::Tensor(a),
            Value::Tensor(b),
            vec![Value::from("valid")],
        )
        .expect("conv2 valid");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 1]);
                assert_eq!(t.data, vec![45.0]);
            }
            Value::Num(n) => assert!((n - 45.0).abs() <= EPS),
            other => panic!("expected scalar 45, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv2_separable_matches_explicit_kernel() {
        let hcol = tensor_from_rows(3, 1, &[1.0, 2.0, 1.0]);
        let hrow = tensor_from_rows(1, 3, &[1.0, 0.0, -1.0]);
        let signal = tensor_from_rows(3, 3, &[3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
        let separable = conv2_builtin(
            Value::Tensor(hcol.clone()),
            Value::Tensor(hrow.clone()),
            vec![Value::Tensor(signal.clone()), Value::from("same")],
        )
        .expect("conv2 separable");

        // Build full kernel explicitly and compare.
        let kernel = {
            let h_matrix =
                conv2_builtin(Value::Tensor(hcol), Value::Tensor(hrow), Vec::new()).unwrap();
            match h_matrix {
                Value::Tensor(t) => Value::Tensor(t),
                other => panic!("expected tensor kernel, got {other:?}"),
            }
        };
        let explicit =
            conv2_builtin(kernel, Value::Tensor(signal), vec![Value::from("same")]).unwrap();
        assert_eq!(separable, explicit);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv2_complex_scaling() {
        let tensor = tensor_from_rows(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let expected_data = tensor.data.clone();
        let result =
            conv2_builtin(Value::Tensor(tensor), Value::Complex(0.0, 2.0), Vec::new()).unwrap();
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                for (idx, &(re, im)) in t.data.iter().enumerate() {
                    assert!(re.abs() <= EPS);
                    assert!((im - 2.0 * expected_data[idx]).abs() <= 10.0 * EPS);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv2_empty_inputs_follow_shape_rules() {
        let empty = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let kernel = tensor_from_rows(2, 2, &[1.0, 1.0, 1.0, 1.0]);
        let result = conv2_builtin(
            Value::Tensor(empty.clone()),
            Value::Tensor(kernel.clone()),
            Vec::new(),
        )
        .unwrap();
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![0, 0]);
                assert!(t.data.is_empty());
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }

        let same = conv2_builtin(
            Value::Tensor(empty.clone()),
            Value::Tensor(kernel),
            vec![Value::from("same")],
        )
        .unwrap();
        match same {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![0, 3]);
                assert!(t.data.is_empty());
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv2_rejects_invalid_shape_keyword() {
        let a = tensor_from_rows(1, 1, &[1.0]);
        let b = tensor_from_rows(1, 1, &[1.0]);
        let err = error_message(
            conv2_builtin(
                Value::Tensor(a),
                Value::Tensor(b),
                vec![Value::from("diagonal")],
            )
            .unwrap_err(),
        );
        assert!(err.contains("shape argument"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv2_promotes_logical_inputs() {
        let logical = LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).unwrap();
        let kernel = tensor_from_rows(1, 2, &[1.0, -1.0]);

        let logical_result = conv2_builtin(
            Value::LogicalArray(logical.clone()),
            Value::Tensor(kernel.clone()),
            Vec::new(),
        )
        .expect("conv2 logical");

        let numeric_tensor = tensor::logical_to_tensor(&logical).unwrap();
        let numeric_result = conv2_builtin(
            Value::Tensor(numeric_tensor),
            Value::Tensor(kernel),
            Vec::new(),
        )
        .expect("conv2 numeric");

        let logical_tensor = test_support::gather(logical_result).expect("gather logical");
        let numeric_tensor = test_support::gather(numeric_result).expect("gather numeric");

        assert_eq!(logical_tensor.shape, numeric_tensor.shape);
        assert_eq!(logical_tensor.data, numeric_tensor.data);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv2_same_even_kernel_alignment() {
        let a = tensor_from_rows(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let b = tensor_from_rows(2, 2, &[1.0, 2.0, 3.0, 4.0]);

        let result = conv2_builtin(
            Value::Tensor(a),
            Value::Tensor(b),
            vec![Value::from("same")],
        )
        .expect("conv2 same even");

        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 3]);
                let expected = tensor_from_rows(
                    3,
                    3,
                    &[
                        4.0, 11.0, 18.0, //
                        18.0, 37.0, 47.0, //
                        36.0, 67.0, 77.0,
                    ],
                );
                assert_eq!(t.data, expected.data);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv2_gpu_roundtrip_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let a = tensor_from_rows(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
            let b = tensor_from_rows(2, 2, &[1.0, 0.0, 0.0, -1.0]);

            let a_view = HostTensorView {
                data: &a.data,
                shape: &a.shape,
            };
            let b_view = HostTensorView {
                data: &b.data,
                shape: &b.shape,
            };

            let ah = provider.upload(&a_view).unwrap();
            let bh = provider.upload(&b_view).unwrap();

            let gpu_result = conv2_builtin(
                Value::GpuTensor(ah),
                Value::GpuTensor(bh),
                vec![Value::from("same")],
            )
            .unwrap();
            let gathered = test_support::gather(gpu_result).unwrap();

            let cpu_result = conv2_builtin(
                Value::Tensor(a),
                Value::Tensor(b),
                vec![Value::from("same")],
            )
            .unwrap();
            let cpu_tensor = test_support::gather(cpu_result).unwrap();

            assert_eq!(gathered.shape, cpu_tensor.shape);
            assert_eq!(gathered.data, cpu_tensor.data);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn conv2_wgpu_fallback_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");

        let a = tensor_from_rows(
            4,
            4,
            &[
                1.0, 2.0, 3.0, 4.0, //
                5.0, 6.0, 7.0, 8.0, //
                9.0, 10.0, 11.0, 12.0, //
                13.0, 14.0, 15.0, 16.0,
            ],
        );
        let b = tensor_from_rows(
            3,
            3,
            &[
                1.0, 0.0, -1.0, //
                2.0, 0.0, -2.0, //
                1.0, 0.0, -1.0,
            ],
        );

        let a_view = HostTensorView {
            data: &a.data,
            shape: &a.shape,
        };
        let b_view = HostTensorView {
            data: &b.data,
            shape: &b.shape,
        };

        let a_handle = provider.upload(&a_view).expect("upload A");
        let b_handle = provider.upload(&b_view).expect("upload B");

        let gpu_value = conv2_builtin(
            Value::GpuTensor(a_handle),
            Value::GpuTensor(b_handle),
            vec![Value::from("valid")],
        )
        .expect("conv2 gpu");
        let gpu_tensor = test_support::gather(gpu_value).expect("gather gpu");

        let cpu_value = conv2_builtin(
            Value::Tensor(a.clone()),
            Value::Tensor(b.clone()),
            vec![Value::from("valid")],
        )
        .expect("conv2 cpu");
        let cpu_tensor = test_support::gather(cpu_value).expect("gather cpu");

        assert_eq!(gpu_tensor.shape, cpu_tensor.shape);
        assert_eq!(gpu_tensor.data, cpu_tensor.data);
    }

    fn conv2_builtin(a: Value, b: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::conv2_builtin(a, b, rest))
    }
}
