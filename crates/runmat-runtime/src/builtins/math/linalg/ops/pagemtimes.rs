//! MATLAB-compatible `pagemtimes` builtin.

use runmat_accelerate_api::{AccelProvider, HostTensorView, ProviderPrecision};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, NumericDType, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::random_args::{complex_tensor_into_value, keyword_of};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const NAME: &str = "pagemtimes";

const PAGEMTIMES_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Z",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Page-wise matrix product.",
}];

const PAGEMTIMES_INPUTS_BASIC: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Left dense single, double, or complex array.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Right dense single, double, or complex array.",
    },
];

const PAGEMTIMES_INPUTS_TRANSPOSE: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Left dense single, double, or complex array.",
    },
    BuiltinParamDescriptor {
        name: "transpX",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "`none`, `transpose`, or `ctranspose` for X.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Right dense single, double, or complex array.",
    },
    BuiltinParamDescriptor {
        name: "transpY",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "`none`, `transpose`, or `ctranspose` for Y.",
    },
];

const PAGEMTIMES_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Z = pagemtimes(X, Y)",
        inputs: &PAGEMTIMES_INPUTS_BASIC,
        outputs: &PAGEMTIMES_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Z = pagemtimes(X, transpX, Y, transpY)",
        inputs: &PAGEMTIMES_INPUTS_TRANSPOSE,
        outputs: &PAGEMTIMES_OUTPUT,
    },
];

const PAGEMTIMES_ERROR_INVALID_ARITY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAGEMTIMES.INVALID_ARITY",
    identifier: Some("RunMat:pagemtimes:InvalidArity"),
    when: "The call does not use the two-argument or four-argument form.",
    message: "pagemtimes: expected X,Y or X,transpX,Y,transpY",
};

const PAGEMTIMES_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAGEMTIMES.INVALID_INPUT",
    identifier: Some("RunMat:pagemtimes:InvalidInput"),
    when: "Inputs are not dense single, double, or complex arrays.",
    message: "pagemtimes: inputs must be dense single, double, or complex arrays",
};

const PAGEMTIMES_ERROR_INVALID_TRANSPOSE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAGEMTIMES.INVALID_TRANSPOSE",
    identifier: Some("RunMat:pagemtimes:InvalidTranspose"),
    when: "Transpose flags are not one of none, transpose, or ctranspose.",
    message: "pagemtimes: transpose options must be 'none', 'transpose', or 'ctranspose'",
};

const PAGEMTIMES_ERROR_PAGE_DIM_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAGEMTIMES.PAGE_DIM_MISMATCH",
    identifier: Some("RunMat:pagemtimes:PageDimensionMismatch"),
    when: "Trailing page dimensions cannot be implicitly expanded.",
    message: "pagemtimes: page dimensions are not compatible",
};

const PAGEMTIMES_ERROR_MATRIX_DIM_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAGEMTIMES.MATRIX_DIM_MISMATCH",
    identifier: Some("RunMat:pagemtimes:MatrixDimensionMismatch"),
    when: "Matrix dimensions are not valid for page-wise multiplication.",
    message: "pagemtimes: inner matrix dimensions must agree",
};

const PAGEMTIMES_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAGEMTIMES.INTERNAL",
    identifier: Some("RunMat:pagemtimes:InternalError"),
    when: "Runtime cannot materialize the page-wise result.",
    message: "pagemtimes: internal runtime failure",
};

const PAGEMTIMES_ERRORS: [BuiltinErrorDescriptor; 6] = [
    PAGEMTIMES_ERROR_INVALID_ARITY,
    PAGEMTIMES_ERROR_INVALID_INPUT,
    PAGEMTIMES_ERROR_INVALID_TRANSPOSE,
    PAGEMTIMES_ERROR_PAGE_DIM_MISMATCH,
    PAGEMTIMES_ERROR_MATRIX_DIM_MISMATCH,
    PAGEMTIMES_ERROR_INTERNAL,
];

pub const PAGEMTIMES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PAGEMTIMES_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &PAGEMTIMES_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::ops::pagemtimes")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "pagemtimes",
    op_kind: GpuOpKind::Custom("pagemtimes"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("pagefun")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Uses host-complete semantics for transpose/scalar forms and re-uploads GPU results; provider pagefun hooks may accelerate plain page products.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::linalg::ops::pagemtimes"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "pagemtimes",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "Acts as a fusion barrier because it performs batched matrix products over page dimensions.",
};

fn pagemtimes_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    pagemtimes_error_with_message(error, error.message)
}

fn pagemtimes_error_with_message(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn pagemtimes_invalid_input(message: impl Into<String>) -> RuntimeError {
    pagemtimes_error_with_message(&PAGEMTIMES_ERROR_INVALID_INPUT, message)
}

fn pagemtimes_internal(message: impl Into<String>) -> RuntimeError {
    pagemtimes_error_with_message(&PAGEMTIMES_ERROR_INTERNAL, message)
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
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
    builder.with_source(err).build()
}

fn pagemtimes_type(args: &[Type], _ctx: &runmat_builtins::ResolveContext) -> Type {
    let _ = args;
    Type::Tensor { shape: None }
}

#[runtime_builtin(
    name = "pagemtimes",
    category = "math/linalg/ops",
    summary = "Multiply corresponding matrix pages in dense arrays.",
    keywords = "pagemtimes,page-wise matrix multiplication,batched matrix multiply,linear algebra,gpu",
    accel = "custom",
    type_resolver(pagemtimes_type),
    descriptor(crate::builtins::math::linalg::ops::pagemtimes::PAGEMTIMES_DESCRIPTOR),
    builtin_path = "crate::builtins::math::linalg::ops::pagemtimes"
)]
async fn pagemtimes_builtin(first: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let call = PagemtimesCall::parse(first, rest)?;
    crate::builtins::common::validation::reject_typed_complex_integer(&call.lhs, NAME)?;
    crate::builtins::common::validation::reject_typed_complex_integer(&call.rhs, NAME)?;
    pagemtimes_eval(call).await
}

async fn pagemtimes_eval(call: PagemtimesCall) -> BuiltinResult<Value> {
    let wants_gpu = call.contains_gpu();
    let provider = gpu_provider_for_call(&call);

    let lhs = gather_if_needed_async(&call.lhs)
        .await
        .map_err(map_control_flow)?;
    let rhs = gather_if_needed_async(&call.rhs)
        .await
        .map_err(map_control_flow)?;

    let lhs = PageInput::from_value(lhs)?;
    let rhs = PageInput::from_value(rhs)?;
    let output = pagemtimes_host(&lhs, call.transp_lhs, &rhs, call.transp_rhs)?;
    output.into_value(wants_gpu, provider)
}

fn gpu_provider_for_call(call: &PagemtimesCall) -> Option<&'static dyn AccelProvider> {
    [&call.lhs, &call.rhs]
        .iter()
        .find_map(|value| match value {
            Value::GpuTensor(handle) => runmat_accelerate_api::provider_for_handle(handle),
            _ => None,
        })
        .or_else(runmat_accelerate_api::provider)
}

#[derive(Clone)]
struct PagemtimesCall {
    lhs: Value,
    rhs: Value,
    transp_lhs: PageTranspose,
    transp_rhs: PageTranspose,
}

impl PagemtimesCall {
    fn parse(first: Value, rest: Vec<Value>) -> BuiltinResult<Self> {
        match rest.len() {
            1 => Ok(Self {
                lhs: first,
                rhs: rest.into_iter().next().expect("one rhs"),
                transp_lhs: PageTranspose::None,
                transp_rhs: PageTranspose::None,
            }),
            3 => {
                let mut iter = rest.into_iter();
                let transp_lhs = PageTranspose::parse(&iter.next().expect("transpose lhs"))?;
                let rhs = iter.next().expect("rhs");
                let transp_rhs = PageTranspose::parse(&iter.next().expect("transpose rhs"))?;
                Ok(Self {
                    lhs: first,
                    rhs,
                    transp_lhs,
                    transp_rhs,
                })
            }
            _ => Err(pagemtimes_error(&PAGEMTIMES_ERROR_INVALID_ARITY)),
        }
    }

    fn contains_gpu(&self) -> bool {
        matches!(self.lhs, Value::GpuTensor(_)) || matches!(self.rhs, Value::GpuTensor(_))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PageTranspose {
    None,
    Transpose,
    CTranspose,
}

impl PageTranspose {
    fn parse(value: &Value) -> BuiltinResult<Self> {
        match keyword_of(value).as_deref() {
            Some("none") => Ok(Self::None),
            Some("transpose") => Ok(Self::Transpose),
            Some("ctranspose") => Ok(Self::CTranspose),
            _ => Err(pagemtimes_error(&PAGEMTIMES_ERROR_INVALID_TRANSPOSE)),
        }
    }

    fn swaps(self) -> bool {
        !matches!(self, Self::None)
    }

    fn conjugates(self) -> bool {
        matches!(self, Self::CTranspose)
    }
}

#[derive(Clone)]
struct PageInput {
    rows: usize,
    cols: usize,
    page_dims: Vec<usize>,
    dtype: NumericDType,
    data: PageData,
}

#[derive(Clone)]
enum PageData {
    Real(Vec<f64>),
    Complex(Vec<(f64, f64)>),
}

impl PageInput {
    fn from_value(value: Value) -> BuiltinResult<Self> {
        match value {
            Value::Num(n) => {
                Self::from_tensor(Tensor::new(vec![n], vec![1, 1]).map_err(|err| {
                    pagemtimes_internal(format!("pagemtimes: failed to build scalar ({err})"))
                })?)
            }
            Value::Complex(re, im) => Self::from_complex_tensor(
                ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(|err| {
                    pagemtimes_internal(format!("pagemtimes: failed to build scalar ({err})"))
                })?,
            ),
            Value::Tensor(tensor) => Self::from_tensor(tensor),
            Value::ComplexTensor(tensor) => Self::from_complex_tensor(tensor),
            other => Err(pagemtimes_invalid_input(format!(
                "pagemtimes: unsupported input type {}",
                type_name(&other)
            ))),
        }
    }

    fn from_tensor(tensor: Tensor) -> BuiltinResult<Self> {
        let dtype = tensor.numeric_dtype();
        let output_dtype = if matches!(dtype, NumericDType::F32) {
            NumericDType::F32
        } else {
            NumericDType::F64
        };
        let shape = canonical_matrix_shape(&tensor.shape);
        let expected = checked_product(&shape)?;
        let data = tensor::tensor_into_values_f64(tensor);
        if data.len() != expected {
            return Err(pagemtimes_internal(
                "pagemtimes: tensor data length does not match shape",
            ));
        }
        Ok(Self {
            rows: shape[0],
            cols: shape[1],
            page_dims: shape.get(2..).unwrap_or(&[]).to_vec(),
            dtype: output_dtype,
            data: PageData::Real(data),
        })
    }

    fn from_complex_tensor(tensor: ComplexTensor) -> BuiltinResult<Self> {
        let shape = canonical_matrix_shape(&tensor.shape);
        let expected = checked_product(&shape)?;
        if tensor.data.len() != expected {
            return Err(pagemtimes_internal(
                "pagemtimes: complex tensor data length does not match shape",
            ));
        }
        Ok(Self {
            rows: shape[0],
            cols: shape[1],
            page_dims: shape.get(2..).unwrap_or(&[]).to_vec(),
            dtype: NumericDType::F64,
            data: PageData::Complex(tensor.data),
        })
    }

    fn transformed_rows(&self, transpose: PageTranspose) -> usize {
        if transpose.swaps() {
            self.cols
        } else {
            self.rows
        }
    }

    fn transformed_cols(&self, transpose: PageTranspose) -> usize {
        if transpose.swaps() {
            self.rows
        } else {
            self.cols
        }
    }

    fn is_complex(&self) -> bool {
        matches!(self.data, PageData::Complex(_))
    }

    fn page_size(&self) -> BuiltinResult<usize> {
        self.rows
            .checked_mul(self.cols)
            .ok_or_else(|| pagemtimes_internal("pagemtimes: page size overflow"))
    }

    fn value_at(
        &self,
        row: usize,
        col: usize,
        transpose: PageTranspose,
        coords: &[usize],
    ) -> BuiltinResult<(f64, f64)> {
        let (source_row, source_col) = if transpose.swaps() {
            (col, row)
        } else {
            (row, col)
        };
        let page = source_page_index(&self.page_dims, coords)?;
        let offset = page
            .checked_mul(self.page_size()?)
            .and_then(|base| {
                source_col
                    .checked_mul(self.rows)
                    .and_then(|c| base.checked_add(c))
            })
            .and_then(|base| base.checked_add(source_row))
            .ok_or_else(|| pagemtimes_internal("pagemtimes: source index overflow"))?;
        match &self.data {
            PageData::Real(data) => data
                .get(offset)
                .map(|value| (*value, 0.0))
                .ok_or_else(|| pagemtimes_internal("pagemtimes: source index out of bounds")),
            PageData::Complex(data) => data
                .get(offset)
                .copied()
                .map(|(re, im)| {
                    if transpose.conjugates() {
                        (re, -im)
                    } else {
                        (re, im)
                    }
                })
                .ok_or_else(|| pagemtimes_internal("pagemtimes: source index out of bounds")),
        }
    }
}

enum PageOutput {
    Real(Tensor),
    Complex(ComplexTensor),
}

impl PageOutput {
    fn into_value(
        self,
        wants_gpu: bool,
        provider: Option<&'static dyn AccelProvider>,
    ) -> BuiltinResult<Value> {
        match self {
            Self::Real(tensor) => {
                if wants_gpu {
                    if let Some(provider) = provider {
                        let dtype = tensor.numeric_dtype();
                        let values = tensor::tensor_values_f64_cow(&tensor);
                        let view = HostTensorView {
                            data: values.as_ref(),
                            shape: &tensor.shape,
                        };
                        let handle = provider.upload(&view).map_err(|err| {
                            pagemtimes_internal(format!("pagemtimes: gpu upload failed ({err})"))
                        })?;
                        runmat_accelerate_api::set_handle_precision(
                            &handle,
                            precision_for_dtype(dtype),
                        );
                        return Ok(gpu_helpers::resident_gpu_value(handle));
                    }
                }
                Ok(tensor::tensor_into_value(tensor))
            }
            Self::Complex(tensor) => {
                if wants_gpu {
                    if let Some(provider) = provider {
                        let handle = gpu_helpers::upload_complex_tensor(provider, &tensor)?;
                        return Ok(gpu_helpers::complex_gpu_value(handle));
                    }
                }
                Ok(complex_tensor_into_value(tensor))
            }
        }
    }
}

fn pagemtimes_host(
    lhs: &PageInput,
    lhs_transpose: PageTranspose,
    rhs: &PageInput,
    rhs_transpose: PageTranspose,
) -> BuiltinResult<PageOutput> {
    let lhs_rows = lhs.transformed_rows(lhs_transpose);
    let lhs_cols = lhs.transformed_cols(lhs_transpose);
    let rhs_rows = rhs.transformed_rows(rhs_transpose);
    let rhs_cols = rhs.transformed_cols(rhs_transpose);
    let lhs_scalar = lhs_rows == 1 && lhs_cols == 1;
    let rhs_scalar = rhs_rows == 1 && rhs_cols == 1;

    let (out_rows, out_cols) = if lhs_scalar && rhs_scalar {
        (1, 1)
    } else if lhs_scalar {
        (rhs_rows, rhs_cols)
    } else if rhs_scalar {
        (lhs_rows, lhs_cols)
    } else {
        if lhs_cols != rhs_rows {
            return Err(pagemtimes_error_with_message(
                &PAGEMTIMES_ERROR_MATRIX_DIM_MISMATCH,
                format!(
                    "pagemtimes: inner matrix dimensions must agree ({}x{} * {}x{})",
                    lhs_rows, lhs_cols, rhs_rows, rhs_cols
                ),
            ));
        }
        (lhs_rows, rhs_cols)
    };

    let page_dims = broadcast_page_dims(&lhs.page_dims, &rhs.page_dims)?;
    let page_volume = checked_product_or_one(&page_dims)?;
    let element_count = out_rows
        .checked_mul(out_cols)
        .and_then(|n| n.checked_mul(page_volume))
        .ok_or_else(|| pagemtimes_internal("pagemtimes: result size overflow"))?;
    let mut shape = vec![out_rows, out_cols];
    shape.extend_from_slice(&page_dims);

    let complex_output = lhs.is_complex() || rhs.is_complex();
    let mut coords = vec![0usize; page_dims.len()];

    if complex_output {
        let mut data = Vec::with_capacity(element_count);
        for page_index in 0..page_volume {
            if page_index > 0 {
                increment_coords(&mut coords, &page_dims);
            }
            write_complex_page(
                lhs,
                lhs_transpose,
                rhs,
                rhs_transpose,
                lhs_scalar,
                rhs_scalar,
                out_rows,
                out_cols,
                lhs_cols,
                &coords,
                &mut data,
            )?;
        }
        let tensor = ComplexTensor::new(data, shape).map_err(|err| {
            pagemtimes_internal(format!(
                "pagemtimes: failed to build complex result ({err})"
            ))
        })?;
        Ok(PageOutput::Complex(tensor))
    } else {
        let mut data = Vec::with_capacity(element_count);
        for page_index in 0..page_volume {
            if page_index > 0 {
                increment_coords(&mut coords, &page_dims);
            }
            write_real_page(
                lhs,
                lhs_transpose,
                rhs,
                rhs_transpose,
                lhs_scalar,
                rhs_scalar,
                out_rows,
                out_cols,
                lhs_cols,
                &coords,
                &mut data,
            )?;
        }
        let dtype = if lhs.dtype == NumericDType::F32 && rhs.dtype == NumericDType::F32 {
            NumericDType::F32
        } else {
            NumericDType::F64
        };
        if dtype == NumericDType::F32 {
            for value in &mut data {
                *value = (*value as f32) as f64;
            }
        }
        let tensor = Tensor::new_with_dtype(data, shape, dtype).map_err(|err| {
            pagemtimes_internal(format!("pagemtimes: failed to build result ({err})"))
        })?;
        Ok(PageOutput::Real(tensor))
    }
}

#[allow(clippy::too_many_arguments)]
fn write_real_page(
    lhs: &PageInput,
    lhs_transpose: PageTranspose,
    rhs: &PageInput,
    rhs_transpose: PageTranspose,
    lhs_scalar: bool,
    rhs_scalar: bool,
    out_rows: usize,
    out_cols: usize,
    kdim: usize,
    coords: &[usize],
    out: &mut Vec<f64>,
) -> BuiltinResult<()> {
    for col in 0..out_cols {
        for row in 0..out_rows {
            let value = if lhs_scalar {
                let scalar = lhs.value_at(0, 0, lhs_transpose, coords)?.0;
                scalar * rhs.value_at(row, col, rhs_transpose, coords)?.0
            } else if rhs_scalar {
                let scalar = rhs.value_at(0, 0, rhs_transpose, coords)?.0;
                lhs.value_at(row, col, lhs_transpose, coords)?.0 * scalar
            } else {
                let mut acc = 0.0;
                for k in 0..kdim {
                    acc += lhs.value_at(row, k, lhs_transpose, coords)?.0
                        * rhs.value_at(k, col, rhs_transpose, coords)?.0;
                }
                acc
            };
            out.push(value);
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_complex_page(
    lhs: &PageInput,
    lhs_transpose: PageTranspose,
    rhs: &PageInput,
    rhs_transpose: PageTranspose,
    lhs_scalar: bool,
    rhs_scalar: bool,
    out_rows: usize,
    out_cols: usize,
    kdim: usize,
    coords: &[usize],
    out: &mut Vec<(f64, f64)>,
) -> BuiltinResult<()> {
    for col in 0..out_cols {
        for row in 0..out_rows {
            let value = if lhs_scalar {
                let scalar = lhs.value_at(0, 0, lhs_transpose, coords)?;
                complex_mul(scalar, rhs.value_at(row, col, rhs_transpose, coords)?)
            } else if rhs_scalar {
                let scalar = rhs.value_at(0, 0, rhs_transpose, coords)?;
                complex_mul(lhs.value_at(row, col, lhs_transpose, coords)?, scalar)
            } else {
                let mut acc = (0.0, 0.0);
                for k in 0..kdim {
                    acc = complex_add(
                        acc,
                        complex_mul(
                            lhs.value_at(row, k, lhs_transpose, coords)?,
                            rhs.value_at(k, col, rhs_transpose, coords)?,
                        ),
                    );
                }
                acc
            };
            out.push(value);
        }
    }
    Ok(())
}

fn complex_mul((ar, ai): (f64, f64), (br, bi): (f64, f64)) -> (f64, f64) {
    (ar * br - ai * bi, ar * bi + ai * br)
}

fn complex_add((ar, ai): (f64, f64), (br, bi): (f64, f64)) -> (f64, f64) {
    (ar + br, ai + bi)
}

fn broadcast_page_dims(lhs: &[usize], rhs: &[usize]) -> BuiltinResult<Vec<usize>> {
    let rank = lhs.len().max(rhs.len());
    let mut out = vec![1usize; rank];
    for dim in 0..rank {
        let lhs_extent = lhs.get(dim).copied().unwrap_or(1);
        let rhs_extent = rhs.get(dim).copied().unwrap_or(1);
        out[dim] = match (lhs_extent, rhs_extent) {
            (a, b) if a == b => a,
            (1, value) => value,
            (value, 1) => value,
            (a, b) => {
                return Err(pagemtimes_error_with_message(
                    &PAGEMTIMES_ERROR_PAGE_DIM_MISMATCH,
                    format!(
                        "pagemtimes: page dimension {} mismatch ({} vs {})",
                        dim + 3,
                        a,
                        b
                    ),
                ))
            }
        };
    }
    Ok(out)
}

fn precision_for_dtype(dtype: NumericDType) -> ProviderPrecision {
    match dtype {
        NumericDType::F32 => ProviderPrecision::F32,
        NumericDType::F64
        | NumericDType::I8
        | NumericDType::I16
        | NumericDType::I32
        | NumericDType::I64
        | NumericDType::U8
        | NumericDType::U16
        | NumericDType::U32
        | NumericDType::U64 => ProviderPrecision::F64,
    }
}

fn source_page_index(page_dims: &[usize], coords: &[usize]) -> BuiltinResult<usize> {
    let mut stride = 1usize;
    let mut index = 0usize;
    for dim in 0..coords.len() {
        let extent = page_dims.get(dim).copied().unwrap_or(1);
        if extent == 0 {
            return Err(pagemtimes_internal(
                "pagemtimes: cannot index an empty page extent",
            ));
        }
        let coord = if extent == 1 { 0 } else { coords[dim] };
        if coord >= extent {
            return Err(pagemtimes_internal("pagemtimes: page index out of bounds"));
        }
        index = index
            .checked_add(coord.checked_mul(stride).ok_or_else(|| {
                pagemtimes_internal("pagemtimes: page index multiplication overflow")
            })?)
            .ok_or_else(|| pagemtimes_internal("pagemtimes: page index overflow"))?;
        stride = stride
            .checked_mul(extent)
            .ok_or_else(|| pagemtimes_internal("pagemtimes: page stride overflow"))?;
    }
    Ok(index)
}

fn increment_coords(coords: &mut [usize], dims: &[usize]) {
    for dim in 0..coords.len() {
        coords[dim] += 1;
        if coords[dim] < dims[dim] {
            return;
        }
        coords[dim] = 0;
    }
}

fn checked_product(shape: &[usize]) -> BuiltinResult<usize> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| pagemtimes_internal("pagemtimes: shape size overflow"))
    })
}

fn checked_product_or_one(shape: &[usize]) -> BuiltinResult<usize> {
    if shape.is_empty() {
        Ok(1)
    } else {
        checked_product(shape)
    }
}

fn canonical_matrix_shape(shape: &[usize]) -> Vec<usize> {
    match shape.len() {
        0 => vec![1, 1],
        1 => vec![1, shape[0]],
        _ => shape.to_vec(),
    }
}

fn type_name(value: &Value) -> &'static str {
    match value {
        Value::Int(_) => "integer",
        Value::Num(_) => "double",
        Value::Complex(_, _) => "complex double",
        Value::Bool(_) => "logical",
        Value::LogicalArray(_) => "logical array",
        Value::String(_) => "string",
        Value::StringArray(_) => "string array",
        Value::CharArray(_) => "char array",
        Value::Symbolic(_) => "sym",
        Value::Tensor(_) => "numeric array",
        Value::SparseTensor(_) => "sparse array",
        Value::ComplexTensor(_) => "complex array",
        Value::Cell(_) => "cell array",
        Value::Struct(_) => "struct",
        Value::GpuTensor(_) => "gpuArray",
        Value::Object(_) => "object",
        Value::HandleObject(_) => "handle object",
        Value::Listener(_) => "listener",
        Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. } => "function handle",
        Value::Closure(_) => "closure",
        Value::ClassRef(_) => "class reference",
        Value::MException(_) => "MException",
        Value::OutputList(_) => "output list",
    }
}

#[cfg(test)]
fn handle_is_complex(value: &Value) -> bool {
    matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_storage(handle) == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::{GpuTensorStorage, HostTensorView, ProviderPrecision};
    use runmat_builtins::{CharArray, IntValue, IntegerStorage};

    fn call(first: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(pagemtimes_builtin(first, rest))
    }

    fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new(data, shape).unwrap())
    }

    fn expect_tensor(value: Value) -> Tensor {
        match value {
            Value::Tensor(tensor) => tensor,
            Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).unwrap(),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn descriptor_has_documented_forms() {
        let labels: Vec<&str> = PAGEMTIMES_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert!(labels.contains(&"Z = pagemtimes(X, Y)"));
        assert!(labels.contains(&"Z = pagemtimes(X, transpX, Y, transpY)"));
    }

    #[test]
    fn multiplies_matching_3d_pages() {
        let lhs = tensor(
            vec![
                1.0, 2.0, 3.0, 4.0, // page 1
                5.0, 6.0, 7.0, 8.0, // page 2
            ],
            vec![2, 2, 2],
        );
        let rhs = tensor(
            vec![
                10.0, 20.0, 30.0, 40.0, // page 1
                50.0, 60.0, 70.0, 80.0, // page 2
            ],
            vec![2, 2, 2],
        );
        let out = expect_tensor(call(lhs, vec![rhs]).unwrap());
        assert_eq!(out.shape, vec![2, 2, 2]);
        assert_eq!(
            out.materialize_f64(),
            vec![70.0, 100.0, 150.0, 220.0, 670.0, 780.0, 910.0, 1060.0]
        );
    }

    #[test]
    fn typed_integer_pages_read_exact_storage_and_return_double() {
        let lhs = Tensor::new_integer(
            IntegerStorage::I16(vec![
                1, 2, 3, 4, // page 1
                5, 6, 7, 8, // page 2
            ]),
            vec![2, 2, 2],
        )
        .unwrap();
        let rhs = Tensor::new_integer(
            IntegerStorage::U16(vec![
                10, 20, 30, 40, // page 1
                50, 60, 70, 80, // page 2
            ]),
            vec![2, 2, 2],
        )
        .unwrap();
        let out = expect_tensor(call(Value::Tensor(lhs), vec![Value::Tensor(rhs)]).unwrap());
        assert_eq!(out.shape, vec![2, 2, 2]);
        assert_eq!(out.numeric_dtype(), NumericDType::F64);
        assert!(out.integer_storage().is_none());
        assert_eq!(
            out.materialize_f64(),
            vec![70.0, 100.0, 150.0, 220.0, 670.0, 780.0, 910.0, 1060.0]
        );
    }

    #[test]
    fn mixed_single_and_integer_pages_return_double_from_exact_storage() {
        let lhs = Tensor::new_with_dtype(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], NumericDType::F32)
            .unwrap();
        let rhs = Tensor::new_integer(IntegerStorage::U16(vec![5, 7, 6, 8]), vec![2, 2]).unwrap();

        let out = expect_tensor(call(Value::Tensor(lhs), vec![Value::Tensor(rhs)]).unwrap());
        assert_eq!(out.shape, vec![2, 2]);
        assert_eq!(out.numeric_dtype(), NumericDType::F64);
        assert_eq!(out.materialize_f64(), vec![26.0, 38.0, 30.0, 44.0]);
    }

    #[test]
    fn broadcasts_matrix_across_pages() {
        let lhs = tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let rhs = tensor(
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            vec![2, 2, 2],
        );
        let out = expect_tensor(call(lhs, vec![rhs]).unwrap());
        assert_eq!(out.shape, vec![2, 2, 2]);
        assert_eq!(
            out.materialize_f64(),
            vec![70.0, 100.0, 150.0, 220.0, 230.0, 340.0, 310.0, 460.0]
        );
    }

    #[test]
    fn expands_multiple_page_dimensions_by_position() {
        let lhs = Value::Tensor(
            Tensor::new((1..=12).map(|v| v as f64).collect(), vec![2, 2, 1, 3]).unwrap(),
        );
        let rhs = Value::Tensor(
            Tensor::new((1..=16).map(|v| v as f64).collect(), vec![2, 2, 4, 1]).unwrap(),
        );
        let out = expect_tensor(call(lhs, vec![rhs]).unwrap());
        assert_eq!(out.shape, vec![2, 2, 4, 3]);
        let values = out.materialize_f64();
        assert_eq!(values.len(), 2 * 2 * 4 * 3);
        assert_eq!(&values[0..4], &[7.0, 10.0, 15.0, 22.0]);
        assert_eq!(&values[44..48], &[271.0, 298.0, 311.0, 342.0]);
    }

    #[test]
    fn scalar_left_scales_each_page() {
        let rhs = tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]);
        let out = expect_tensor(call(Value::Num(2.5), vec![rhs]).unwrap());
        assert_eq!(out.shape, vec![2, 2, 2]);
        assert_eq!(
            out.materialize_f64(),
            vec![2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0]
        );
    }

    #[test]
    fn scalar_right_scales_each_page() {
        let lhs = tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]);
        let out = expect_tensor(call(lhs, vec![Value::Num(-2.0)]).unwrap());
        assert_eq!(out.shape, vec![2, 2, 2]);
        assert_eq!(
            out.materialize_f64(),
            vec![-2.0, -4.0, -6.0, -8.0, -10.0, -12.0, -14.0, -16.0]
        );
    }

    #[test]
    fn supports_transpose_flags() {
        let lhs = tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let rhs = tensor(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![2, 3]);
        let out = expect_tensor(
            call(
                lhs,
                vec![
                    Value::CharArray(CharArray::new_row("transpose")),
                    rhs,
                    Value::String("none".to_string()),
                ],
            )
            .unwrap(),
        );
        assert_eq!(out.shape, vec![3, 3]);
        assert_eq!(
            out.materialize_f64(),
            vec![23.0, 53.0, 83.0, 29.0, 67.0, 105.0, 35.0, 81.0, 127.0]
        );
    }

    #[test]
    fn ctranspose_conjugates_complex_pages() {
        let lhs = ComplexTensor::new(vec![(1.0, 1.0), (2.0, -1.0)], vec![2, 1]).unwrap();
        let rhs = ComplexTensor::new(vec![(3.0, 2.0), (4.0, -3.0)], vec![2, 1]).unwrap();
        let out = call(
            Value::ComplexTensor(lhs),
            vec![
                Value::String("ctranspose".to_string()),
                Value::ComplexTensor(rhs),
                Value::String("none".to_string()),
            ],
        )
        .unwrap();
        match out {
            Value::Complex(re, im) => {
                assert!((re - 16.0).abs() < 1e-12);
                assert!((im + 3.0).abs() < 1e-12);
            }
            other => panic!("expected complex scalar, got {other:?}"),
        }
    }

    #[test]
    fn rejects_unpaired_or_bad_transpose_options() {
        let lhs = tensor(vec![1.0], vec![1, 1]);
        let rhs = tensor(vec![2.0], vec![1, 1]);
        let err = call(
            lhs.clone(),
            vec![Value::String("transpose".to_string()), rhs.clone()],
        )
        .unwrap_err();
        assert_eq!(err.identifier(), PAGEMTIMES_ERROR_INVALID_ARITY.identifier);

        let err = call(
            lhs,
            vec![
                Value::String("bad".to_string()),
                rhs,
                Value::String("none".to_string()),
            ],
        )
        .unwrap_err();
        assert_eq!(
            err.identifier(),
            PAGEMTIMES_ERROR_INVALID_TRANSPOSE.identifier
        );
    }

    #[test]
    fn rejects_incompatible_page_dimensions() {
        let lhs = tensor(vec![1.0; 8], vec![2, 2, 2]);
        let rhs = tensor(vec![1.0; 12], vec![2, 2, 3]);
        let err = call(lhs, vec![rhs]).unwrap_err();
        assert_eq!(
            err.identifier(),
            PAGEMTIMES_ERROR_PAGE_DIM_MISMATCH.identifier
        );
    }

    #[test]
    fn zero_page_dimensions_only_expand_with_singleton_or_zero() {
        let empty_pages = tensor(Vec::new(), vec![2, 2, 0]);
        let singleton_pages = tensor(vec![1.0; 4], vec![2, 2, 1]);
        let out = expect_tensor(call(empty_pages.clone(), vec![singleton_pages]).unwrap());
        assert_eq!(out.shape, vec![2, 2, 0]);
        assert!(out.is_empty());

        let three_pages = tensor(vec![1.0; 12], vec![2, 2, 3]);
        let err = call(empty_pages, vec![three_pages]).unwrap_err();
        assert_eq!(
            err.identifier(),
            PAGEMTIMES_ERROR_PAGE_DIM_MISMATCH.identifier
        );
    }

    #[test]
    fn rejects_matrix_dimension_mismatch() {
        let lhs = tensor(vec![1.0; 6], vec![2, 3]);
        let rhs = tensor(vec![1.0; 8], vec![4, 2]);
        let err = call(lhs, vec![rhs]).unwrap_err();
        assert_eq!(
            err.identifier(),
            PAGEMTIMES_ERROR_MATRIX_DIM_MISMATCH.identifier
        );
    }

    #[test]
    fn rejects_integer_scalar_inputs() {
        let err = call(Value::Int(IntValue::I32(2)), vec![Value::Num(3.0)]).unwrap_err();
        assert_eq!(err.identifier(), PAGEMTIMES_ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn preserves_single_for_single_inputs() {
        let lhs = Value::Tensor(
            Tensor::new_with_dtype(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], NumericDType::F32)
                .unwrap(),
        );
        let rhs = Value::Tensor(
            Tensor::new_with_dtype(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2], NumericDType::F32)
                .unwrap(),
        );
        let out = expect_tensor(call(lhs, vec![rhs]).unwrap());
        assert_eq!(out.numeric_dtype(), NumericDType::F32);
        assert_eq!(out.materialize_f64(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn gpu_input_roundtrips_to_gpu_result() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let data = tensor.as_f64_slice().expect("double tensor");
            let view = HostTensorView {
                data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).unwrap();
            let out = call(Value::GpuTensor(handle), vec![Value::Num(2.0)]).unwrap();
            assert!(matches!(out, Value::GpuTensor(_)));
            let gathered = test_support::gather(out).unwrap();
            assert_eq!(gathered.shape, vec![2, 2]);
            assert_eq!(gathered.materialize_f64(), vec![2.0, 4.0, 6.0, 8.0]);
        });
    }

    #[test]
    fn gpu_single_input_preserves_single_precision_metadata() {
        test_support::with_test_provider(|provider| {
            let lhs =
                Tensor::new_with_dtype(vec![1.25, 2.5, 3.75, 4.5], vec![2, 2], NumericDType::F32)
                    .unwrap();
            let upload_values = lhs.materialize_f64();
            let view = HostTensorView {
                data: &upload_values,
                shape: &lhs.shape,
            };
            let handle = provider.upload(&view).unwrap();
            runmat_accelerate_api::set_handle_precision(&handle, ProviderPrecision::F32);
            let rhs = Value::Tensor(
                Tensor::new_with_dtype(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2], NumericDType::F32)
                    .unwrap(),
            );
            let out = call(Value::GpuTensor(handle), vec![rhs]).unwrap();
            match &out {
                Value::GpuTensor(handle) => assert_eq!(
                    runmat_accelerate_api::handle_precision(handle),
                    Some(ProviderPrecision::F32)
                ),
                other => panic!("expected gpu tensor, got {other:?}"),
            }
            let gathered = test_support::gather(out).unwrap();
            assert_eq!(gathered.numeric_dtype(), NumericDType::F32);
            assert_eq!(gathered.materialize_f64(), lhs.materialize_f64());
        });
    }

    #[test]
    fn complex_gpu_input_uploads_complex_result() {
        test_support::with_test_provider(|provider| {
            let tensor = ComplexTensor::new(vec![(1.0, 2.0)], vec![1, 1]).unwrap();
            let handle = gpu_helpers::upload_complex_tensor(provider, &tensor).unwrap();
            let out = call(Value::GpuTensor(handle), vec![Value::Complex(3.0, -1.0)]).unwrap();
            match &out {
                Value::GpuTensor(handle) => {
                    assert_eq!(
                        runmat_accelerate_api::handle_storage(handle),
                        GpuTensorStorage::ComplexInterleaved
                    );
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }
            let gathered = block_on(gather_if_needed_async(&out)).unwrap();
            match gathered {
                Value::Complex(re, im) => {
                    assert!((re - 5.0).abs() < 1e-12);
                    assert!((im - 5.0).abs() < 1e-12);
                }
                Value::ComplexTensor(tensor) => {
                    assert_eq!(tensor.shape, vec![1, 1]);
                    let (re, im) = tensor.data[0];
                    assert!((re - 5.0).abs() < 1e-12);
                    assert!((im - 5.0).abs() < 1e-12);
                }
                other => panic!("expected complex scalar, got {other:?}"),
            }
        });
    }

    #[test]
    fn helper_detects_complex_gpu_handles() {
        test_support::with_test_provider(|provider| {
            let tensor = ComplexTensor::new(vec![(1.0, 0.0)], vec![1, 1]).unwrap();
            let handle = gpu_helpers::upload_complex_tensor(provider, &tensor).unwrap();
            assert!(handle_is_complex(&Value::GpuTensor(handle)));
        });
    }
}
