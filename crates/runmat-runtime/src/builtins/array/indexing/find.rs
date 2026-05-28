//! MATLAB-compatible `find` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{HostTensorView, ProviderFindResult};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, ResolveContext, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::array::type_resolvers::column_vector_type;
use crate::builtins::common::arg_tokens::ArgToken;
use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::indexing::find")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "find",
    op_kind: GpuOpKind::Custom("find"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("find")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "WGPU provider executes find directly on device; other providers fall back to host and re-upload results to preserve residency.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::indexing::find")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "find",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Find drives control flow and currently bypasses fusion; metadata is present for completeness only.",
};

fn find_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    column_vector_type()
}

const BUILTIN_NAME: &str = "find";

const FIND_OUTPUT_LINEAR: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "idx",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Linear indices of non-zero elements.",
}];

const FIND_OUTPUT_ROW_COL: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "row",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Row subscripts of non-zero elements.",
    },
    BuiltinParamDescriptor {
        name: "col",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Column subscripts of non-zero elements.",
    },
];

const FIND_OUTPUT_ROW_COL_VAL: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "row",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Row subscripts of non-zero elements.",
    },
    BuiltinParamDescriptor {
        name: "col",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Column subscripts of non-zero elements.",
    },
    BuiltinParamDescriptor {
        name: "v",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Values at the reported row/column locations.",
    },
];

const FIND_INPUTS_BASE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array to search.",
}];

const FIND_INPUTS_LIMIT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array to search.",
    },
    BuiltinParamDescriptor {
        name: "K",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Maximum number of indices to return.",
    },
];

const FIND_INPUTS_LIMIT_DIR: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array to search.",
    },
    BuiltinParamDescriptor {
        name: "K",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Maximum number of indices to return.",
    },
    BuiltinParamDescriptor {
        name: "direction",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"first\""),
        description: "Direction selector: `\"first\"` or `\"last\"`.",
    },
];

const FIND_SIGNATURES: [BuiltinSignatureDescriptor; 7] = [
    BuiltinSignatureDescriptor {
        label: "idx = find(X)",
        inputs: &FIND_INPUTS_BASE,
        outputs: &FIND_OUTPUT_LINEAR,
    },
    BuiltinSignatureDescriptor {
        label: "idx = find(X, K)",
        inputs: &FIND_INPUTS_LIMIT,
        outputs: &FIND_OUTPUT_LINEAR,
    },
    BuiltinSignatureDescriptor {
        label: "idx = find(X, K, direction)",
        inputs: &FIND_INPUTS_LIMIT_DIR,
        outputs: &FIND_OUTPUT_LINEAR,
    },
    BuiltinSignatureDescriptor {
        label: "[row, col] = find(X)",
        inputs: &FIND_INPUTS_BASE,
        outputs: &FIND_OUTPUT_ROW_COL,
    },
    BuiltinSignatureDescriptor {
        label: "[row, col] = find(X, K, direction)",
        inputs: &FIND_INPUTS_LIMIT_DIR,
        outputs: &FIND_OUTPUT_ROW_COL,
    },
    BuiltinSignatureDescriptor {
        label: "[row, col, v] = find(X)",
        inputs: &FIND_INPUTS_BASE,
        outputs: &FIND_OUTPUT_ROW_COL_VAL,
    },
    BuiltinSignatureDescriptor {
        label: "[row, col, v] = find(X, K, direction)",
        inputs: &FIND_INPUTS_LIMIT_DIR,
        outputs: &FIND_OUTPUT_ROW_COL_VAL,
    },
];

const FIND_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIND.INVALID_INPUT",
    identifier: Some("RunMat:find:InvalidInput"),
    when: "Input type or option arguments are not valid for find.",
    message: "find: invalid input arguments",
};

const FIND_ERROR_PROVIDER_OUTPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIND.PROVIDER_OUTPUT",
    identifier: Some("RunMat:find:ProviderOutput"),
    when: "GPU provider does not return expected output buffers for requested nargout.",
    message: "find: provider output buffer mismatch",
};

const FIND_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIND.INTERNAL",
    identifier: Some("RunMat:find:InternalError"),
    when: "Internal tensor conversion/materialization fails while building outputs.",
    message: "find: internal error",
};

const FIND_ERRORS: [BuiltinErrorDescriptor; 3] = [
    FIND_ERROR_INVALID_INPUT,
    FIND_ERROR_PROVIDER_OUTPUT,
    FIND_ERROR_INTERNAL,
];

pub const FIND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FIND_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FIND_ERRORS,
};

fn find_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    find_error_with_message(error.message, error)
}

fn find_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn parse_find_tokens(tokens: &[ArgToken]) -> crate::BuiltinResult<FindOptions> {
    match tokens.len() {
        0 => Ok(FindOptions::default()),
        1 => {
            if let Some(direction) = token_to_direction(&tokens[0])? {
                let limit = if matches!(direction, FindDirection::Last) {
                    Some(1)
                } else {
                    None
                };
                Ok(FindOptions { limit, direction })
            } else {
                let limit = token_to_limit(&tokens[0])?;
                Ok(FindOptions {
                    limit: Some(limit),
                    direction: FindDirection::First,
                })
            }
        }
        2 => {
            let limit = token_to_limit(&tokens[0])?;
            let direction = token_to_direction(&tokens[1])?.ok_or_else(|| {
                find_error_with_message(
                    "find: third argument must be 'first' or 'last'",
                    &FIND_ERROR_INVALID_INPUT,
                )
            })?;
            Ok(FindOptions {
                limit: Some(limit),
                direction,
            })
        }
        _ => Err(find_error_with_message(
            "find: too many input arguments",
            &FIND_ERROR_INVALID_INPUT,
        )),
    }
}

fn token_to_direction(token: &ArgToken) -> crate::BuiltinResult<Option<FindDirection>> {
    match token {
        ArgToken::String(text) => match text.as_str() {
            "first" => Ok(Some(FindDirection::First)),
            "last" => Ok(Some(FindDirection::Last)),
            _ => Err(find_error_with_message(
                "find: direction must be 'first' or 'last'",
                &FIND_ERROR_INVALID_INPUT,
            )),
        },
        _ => Ok(None),
    }
}

fn token_to_limit(token: &ArgToken) -> crate::BuiltinResult<usize> {
    match token {
        ArgToken::Number(value) => parse_limit_scalar(*value),
        _ => Err(find_error_with_message(
            "find: second argument must be a scalar",
            &FIND_ERROR_INVALID_INPUT,
        )),
    }
}

#[runtime_builtin(
    name = "find",
    category = "array/indexing",
    summary = "Locate indices and values of nonzero elements.",
    keywords = "find,nonzero,indices,row,column,gpu",
    accel = "custom",
    type_resolver(find_type),
    descriptor(crate::builtins::array::indexing::find::FIND_DESCRIPTOR),
    builtin_path = "crate::builtins::array::indexing::find"
)]
async fn find_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let eval = evaluate(value, &rest).await?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        if out_count <= 1 {
            let linear = eval.linear_value()?;
            return Ok(crate::output_count::output_list_with_padding(
                out_count,
                vec![linear],
            ));
        }
        let rows = eval.row_value()?;
        let cols = eval.column_value()?;
        let mut outputs = vec![rows, cols];
        if out_count >= 3 {
            outputs.push(eval.values_value()?);
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count, outputs,
        ));
    }
    eval.linear_value()
}

/// Evaluate `find` and return an object that can materialise the various outputs.
pub async fn evaluate(value: Value, args: &[Value]) -> crate::BuiltinResult<FindEval> {
    let options = parse_options(args).await?;
    match value {
        Value::GpuTensor(handle) => {
            if let Some(result) = try_provider_find(&handle, &options) {
                return Ok(FindEval::from_gpu(result));
            }
            let (storage, _) = materialize_input(Value::GpuTensor(handle)).await?;
            let result = compute_find(&storage, &options);
            Ok(FindEval::from_host(result, true))
        }
        other => {
            let (storage, input_was_gpu) = materialize_input(other).await?;
            let result = compute_find(&storage, &options);
            Ok(FindEval::from_host(result, input_was_gpu))
        }
    }
}

fn try_provider_find(
    handle: &runmat_accelerate_api::GpuTensorHandle,
    options: &FindOptions,
) -> Option<ProviderFindResult> {
    #[cfg(all(test, feature = "wgpu"))]
    {
        if handle.device_id != 0 {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }
    let provider = runmat_accelerate_api::provider()?;
    let direction = match options.direction {
        FindDirection::First => runmat_accelerate_api::FindDirection::First,
        FindDirection::Last => runmat_accelerate_api::FindDirection::Last,
    };
    let limit = options.effective_limit();
    provider.find(handle, limit, direction).ok()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FindDirection {
    First,
    Last,
}

#[derive(Debug, Clone)]
struct FindOptions {
    limit: Option<usize>,
    direction: FindDirection,
}

impl Default for FindOptions {
    fn default() -> Self {
        Self {
            limit: None,
            direction: FindDirection::First,
        }
    }
}

impl FindOptions {
    fn effective_limit(&self) -> Option<usize> {
        match self.direction {
            FindDirection::Last => self.limit.or(Some(1)),
            FindDirection::First => self.limit,
        }
    }
}

#[derive(Clone)]
enum DataStorage {
    Real(Tensor),
    Complex(ComplexTensor),
}

impl DataStorage {
    fn shape(&self) -> &[usize] {
        match self {
            DataStorage::Real(t) => &t.shape,
            DataStorage::Complex(t) => &t.shape,
        }
    }
}

#[derive(Clone)]
struct FindResult {
    shape: Vec<usize>,
    indices: Vec<usize>,
    values: FindValues,
}

#[derive(Clone)]
enum FindValues {
    Real(Vec<f64>),
    Complex(Vec<(f64, f64)>),
}

pub struct FindEval {
    inner: FindEvalInner,
}

enum FindEvalInner {
    Host {
        result: FindResult,
        prefer_gpu: bool,
    },
    Gpu {
        result: ProviderFindResult,
    },
}

impl FindEval {
    fn from_host(result: FindResult, prefer_gpu: bool) -> Self {
        Self {
            inner: FindEvalInner::Host { result, prefer_gpu },
        }
    }

    fn from_gpu(result: ProviderFindResult) -> Self {
        Self {
            inner: FindEvalInner::Gpu { result },
        }
    }

    pub fn linear_value(&self) -> crate::BuiltinResult<Value> {
        match &self.inner {
            FindEvalInner::Host { result, prefer_gpu } => {
                let tensor = result.linear_tensor()?;
                Ok(tensor_to_value(tensor, *prefer_gpu))
            }
            FindEvalInner::Gpu { result } => Ok(Value::GpuTensor(result.linear.clone())),
        }
    }

    pub fn row_value(&self) -> crate::BuiltinResult<Value> {
        match &self.inner {
            FindEvalInner::Host { result, prefer_gpu } => {
                let tensor = result.row_tensor()?;
                Ok(tensor_to_value(tensor, *prefer_gpu))
            }
            FindEvalInner::Gpu { result } => Ok(Value::GpuTensor(result.rows.clone())),
        }
    }

    pub fn column_value(&self) -> crate::BuiltinResult<Value> {
        match &self.inner {
            FindEvalInner::Host { result, prefer_gpu } => {
                let tensor = result.column_tensor()?;
                Ok(tensor_to_value(tensor, *prefer_gpu))
            }
            FindEvalInner::Gpu { result } => Ok(Value::GpuTensor(result.cols.clone())),
        }
    }

    pub fn values_value(&self) -> crate::BuiltinResult<Value> {
        match &self.inner {
            FindEvalInner::Host { result, prefer_gpu } => result.values_value(*prefer_gpu),
            FindEvalInner::Gpu { result } => result
                .values
                .as_ref()
                .map(|handle| Value::GpuTensor(handle.clone()))
                .ok_or_else(|| find_error(&FIND_ERROR_PROVIDER_OUTPUT)),
        }
    }
}

async fn parse_options(args: &[Value]) -> crate::BuiltinResult<FindOptions> {
    parse_find_tokens(&crate::builtins::common::arg_tokens::tokens_from_values(
        args,
    ))
}

fn parse_limit_scalar(value: f64) -> crate::BuiltinResult<usize> {
    if !value.is_finite() {
        return Err(find_error_with_message(
            "find: K must be a finite, non-negative integer",
            &FIND_ERROR_INVALID_INPUT,
        ));
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(find_error_with_message(
            "find: K must be a finite, non-negative integer",
            &FIND_ERROR_INVALID_INPUT,
        ));
    }
    if rounded < 0.0 {
        return Err(find_error_with_message(
            "find: K must be >= 0",
            &FIND_ERROR_INVALID_INPUT,
        ));
    }
    Ok(rounded as usize)
}

async fn materialize_input(value: Value) -> crate::BuiltinResult<(DataStorage, bool)> {
    match value {
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
            Ok((DataStorage::Real(tensor), true))
        }
        Value::Tensor(tensor) => Ok((DataStorage::Real(tensor), false)),
        Value::SparseTensor(sparse) => Ok((DataStorage::Real(sparse.to_dense()), false)),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|message| find_error_with_message(message, &FIND_ERROR_INTERNAL))?;
            Ok((DataStorage::Real(tensor), false))
        }
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1])
                .map_err(|e| find_error_with_message(format!("find: {e}"), &FIND_ERROR_INTERNAL))?;
            Ok((DataStorage::Real(tensor), false))
        }
        Value::Int(i) => {
            let tensor = Tensor::new(vec![i.to_f64()], vec![1, 1])
                .map_err(|e| find_error_with_message(format!("find: {e}"), &FIND_ERROR_INTERNAL))?;
            Ok((DataStorage::Real(tensor), false))
        }
        Value::Bool(b) => {
            let tensor = Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
                .map_err(|e| find_error_with_message(format!("find: {e}"), &FIND_ERROR_INTERNAL))?;
            Ok((DataStorage::Real(tensor), false))
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| find_error_with_message(format!("find: {e}"), &FIND_ERROR_INTERNAL))?;
            Ok((DataStorage::Complex(tensor), false))
        }
        Value::ComplexTensor(tensor) => Ok((DataStorage::Complex(tensor), false)),
        Value::CharArray(chars) => {
            let mut data = Vec::with_capacity(chars.data.len());
            for c in 0..chars.cols {
                for r in 0..chars.rows {
                    let ch = chars.data[r * chars.cols + c] as u32;
                    data.push(ch as f64);
                }
            }
            let tensor = Tensor::new(data, vec![chars.rows, chars.cols])
                .map_err(|e| find_error_with_message(format!("find: {e}"), &FIND_ERROR_INTERNAL))?;
            Ok((DataStorage::Real(tensor), false))
        }
        other => Err(find_error_with_message(
            format!(
                "find: unsupported input type {:?}; expected numeric, logical, or char data",
                other
            ),
            &FIND_ERROR_INVALID_INPUT,
        )),
    }
}

fn compute_find(storage: &DataStorage, options: &FindOptions) -> FindResult {
    let shape = storage.shape().to_vec();
    let limit = options.effective_limit();

    match storage {
        DataStorage::Real(tensor) => {
            let mut indices = Vec::new();
            let mut values = Vec::new();

            if matches!(limit, Some(0)) {
                return FindResult::new(shape, indices, FindValues::Real(values));
            }

            let len = tensor.data.len();
            match options.direction {
                FindDirection::First => {
                    for idx in 0..len {
                        let value = tensor.data[idx];
                        if value != 0.0 {
                            indices.push(idx + 1);
                            values.push(value);
                            if limit.is_some_and(|k| indices.len() >= k) {
                                break;
                            }
                        }
                    }
                }
                FindDirection::Last => {
                    for idx in (0..len).rev() {
                        let value = tensor.data[idx];
                        if value != 0.0 {
                            indices.push(idx + 1);
                            values.push(value);
                            if limit.is_some_and(|k| indices.len() >= k) {
                                break;
                            }
                        }
                    }
                }
            }

            FindResult::new(shape, indices, FindValues::Real(values))
        }
        DataStorage::Complex(tensor) => {
            let mut indices = Vec::new();
            let mut values = Vec::new();

            if matches!(limit, Some(0)) {
                return FindResult::new(shape, indices, FindValues::Complex(values));
            }

            let len = tensor.data.len();
            match options.direction {
                FindDirection::First => {
                    for idx in 0..len {
                        let (re, im) = tensor.data[idx];
                        if re != 0.0 || im != 0.0 {
                            indices.push(idx + 1);
                            values.push((re, im));
                            if limit.is_some_and(|k| indices.len() >= k) {
                                break;
                            }
                        }
                    }
                }
                FindDirection::Last => {
                    for idx in (0..len).rev() {
                        let (re, im) = tensor.data[idx];
                        if re != 0.0 || im != 0.0 {
                            indices.push(idx + 1);
                            values.push((re, im));
                            if limit.is_some_and(|k| indices.len() >= k) {
                                break;
                            }
                        }
                    }
                }
            }

            FindResult::new(shape, indices, FindValues::Complex(values))
        }
    }
}

impl FindResult {
    fn new(shape: Vec<usize>, indices: Vec<usize>, values: FindValues) -> Self {
        Self {
            shape,
            indices,
            values,
        }
    }

    fn linear_tensor(&self) -> crate::BuiltinResult<Tensor> {
        let data: Vec<f64> = self.indices.iter().map(|&idx| idx as f64).collect();
        let rows = data.len();
        Tensor::new(data, vec![rows, 1])
            .map_err(|e| find_error_with_message(format!("find: {e}"), &FIND_ERROR_INTERNAL))
    }

    fn row_tensor(&self) -> crate::BuiltinResult<Tensor> {
        let mut data = Vec::with_capacity(self.indices.len());
        let rows = self.shape.first().copied().unwrap_or(1).max(1);
        for &idx in &self.indices {
            let zero_based = idx - 1;
            let row = (zero_based % rows) + 1;
            data.push(row as f64);
        }
        Tensor::new(data, vec![self.indices.len(), 1])
            .map_err(|e| find_error_with_message(format!("find: {e}"), &FIND_ERROR_INTERNAL))
    }

    fn column_tensor(&self) -> crate::BuiltinResult<Tensor> {
        let mut data = Vec::with_capacity(self.indices.len());
        let rows = self.shape.first().copied().unwrap_or(1).max(1);
        for &idx in &self.indices {
            let zero_based = idx - 1;
            let col = (zero_based / rows) + 1;
            data.push(col as f64);
        }
        Tensor::new(data, vec![self.indices.len(), 1])
            .map_err(|e| find_error_with_message(format!("find: {e}"), &FIND_ERROR_INTERNAL))
    }

    fn values_value(&self, prefer_gpu: bool) -> crate::BuiltinResult<Value> {
        match &self.values {
            FindValues::Real(values) => {
                let tensor = Tensor::new(values.clone(), vec![values.len(), 1]).map_err(|e| {
                    find_error_with_message(format!("find: {e}"), &FIND_ERROR_INTERNAL)
                })?;
                Ok(tensor_to_value(tensor, prefer_gpu))
            }
            FindValues::Complex(values) => {
                let tensor =
                    ComplexTensor::new(values.clone(), vec![values.len(), 1]).map_err(|e| {
                        find_error_with_message(format!("find: {e}"), &FIND_ERROR_INTERNAL)
                    })?;
                Ok(complex_tensor_into_value(tensor))
            }
        }
    }
}

fn tensor_to_value(tensor: Tensor, prefer_gpu: bool) -> Value {
    if prefer_gpu {
        if let Some(provider) = runmat_accelerate_api::provider() {
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            if let Ok(handle) = provider.upload(&view) {
                return Value::GpuTensor(handle);
            }
        }
    }
    tensor::tensor_into_value(tensor)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{CharArray, IntValue, Type};

    fn find_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(super::find_builtin(value, rest))
    }

    fn evaluate(value: Value, rest: &[Value]) -> crate::BuiltinResult<FindEval> {
        block_on(super::evaluate(value, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn find_linear_indices_basic() {
        let tensor = Tensor::new(vec![0.0, 4.0, 0.0, 7.0, 0.0, 9.0], vec![2, 3]).unwrap();
        let value = find_builtin(Value::Tensor(tensor), Vec::new()).expect("find");
        match value {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 1]);
                assert_eq!(t.data, vec![2.0, 4.0, 6.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn find_type_is_column_vector() {
        assert_eq!(
            find_type(
                &[Type::Tensor { shape: None }],
                &ResolveContext::new(Vec::new()),
            ),
            Type::Tensor {
                shape: Some(vec![None, Some(1)])
            }
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn find_limited_first() {
        let tensor = Tensor::new(vec![0.0, 3.0, 5.0, 0.0, 8.0], vec![1, 5]).unwrap();
        let result =
            find_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(2))]).expect("find");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![2.0, 3.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn find_last_single() {
        let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 6.0, 0.0, 2.0], vec![1, 6]).unwrap();
        let result = find_builtin(Value::Tensor(tensor), vec![Value::from("last")]).expect("find");
        match result {
            Value::Num(n) => assert_eq!(n, 6.0),
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![6.0]);
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn find_complex_values() {
        let tensor =
            ComplexTensor::new(vec![(0.0, 0.0), (1.0, 2.0), (0.0, 0.0)], vec![3, 1]).unwrap();
        let eval = evaluate(Value::ComplexTensor(tensor), &[]).expect("find compute");
        let values = eval.values_value().expect("values");
        match values {
            Value::Complex(re, im) => {
                assert_eq!(re, 1.0);
                assert_eq!(im, 2.0);
            }
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![1, 1]);
                assert_eq!(ct.data, vec![(1.0, 2.0)]);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn find_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 4.0, 0.0, 7.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = find_builtin(Value::GpuTensor(handle), Vec::new()).expect("find");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 1]);
            assert_eq!(gathered.data, vec![2.0, 4.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn find_direction_error() {
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let err = find_builtin(
            Value::Tensor(tensor),
            vec![Value::Int(IntValue::I32(1)), Value::from("invalid")],
        )
        .expect_err("expected error");
        assert!(err.to_string().contains("direction"));
        assert_eq!(err.identifier(), super::FIND_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn find_multi_output_rows_cols_values() {
        let tensor = Tensor::new(vec![0.0, 2.0, 3.0, 0.0, 0.0, 6.0], vec![2, 3]).unwrap();
        let eval = evaluate(Value::Tensor(tensor), &[]).expect("evaluate");

        let rows = test_support::gather(eval.row_value().expect("rows")).expect("gather rows");
        assert_eq!(rows.shape, vec![3, 1]);
        assert_eq!(rows.data, vec![2.0, 1.0, 2.0]);

        let cols = test_support::gather(eval.column_value().expect("cols")).expect("gather cols");
        assert_eq!(cols.shape, vec![3, 1]);
        assert_eq!(cols.data, vec![1.0, 2.0, 3.0]);

        let vals = test_support::gather(eval.values_value().expect("vals")).expect("gather vals");
        assert_eq!(vals.shape, vec![3, 1]);
        assert_eq!(vals.data, vec![2.0, 3.0, 6.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn find_last_order_descending() {
        let tensor = Tensor::new(vec![1.0, 0.0, 2.0, 3.0, 0.0], vec![1, 5]).unwrap();
        let result = find_builtin(
            Value::Tensor(tensor),
            vec![Value::Int(IntValue::I32(2)), Value::from("last")],
        )
        .expect("find");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                assert_eq!(t.data, vec![4.0, 3.0]);
            }
            Value::Num(_) => panic!("expected column vector"),
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn find_limit_zero_returns_empty() {
        let tensor = Tensor::new(vec![1.0, 0.0, 3.0], vec![3, 1]).unwrap();
        let result = find_builtin(Value::Tensor(tensor), vec![Value::Num(0.0)]).expect("find");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![0, 1]);
                assert!(t.data.is_empty());
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn find_char_array_supports_nonzero_codes() {
        let chars = CharArray::new(vec!['\0', 'A', '\0'], 1, 3).unwrap();
        let result = find_builtin(Value::CharArray(chars), Vec::new()).expect("find");
        match result {
            Value::Num(n) => assert_eq!(n, 2.0),
            Value::Tensor(t) => assert_eq!(t.data, vec![2.0]),
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn find_gpu_multi_outputs_return_gpu_handles() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 4.0, 5.0, 0.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let eval = evaluate(Value::GpuTensor(handle), &[]).expect("evaluate");

            let rows = eval.row_value().expect("rows");
            assert!(matches!(rows, Value::GpuTensor(_)));
            let rows_host = test_support::gather(rows).expect("gather rows");
            assert_eq!(rows_host.data, vec![2.0, 1.0]);

            let cols = eval.column_value().expect("cols");
            assert!(matches!(cols, Value::GpuTensor(_)));
            let cols_host = test_support::gather(cols).expect("gather cols");
            assert_eq!(cols_host.data, vec![1.0, 2.0]);

            let vals = eval.values_value().expect("vals");
            assert!(matches!(vals, Value::GpuTensor(_)));
            let vals_host = test_support::gather(vals).expect("gather vals");
            assert_eq!(vals_host.data, vec![4.0, 5.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn find_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![0.0, 2.0, 0.0, 3.0, 4.0, 0.0], vec![3, 2]).unwrap();
        let cpu_eval = evaluate(Value::Tensor(tensor.clone()), &[]).expect("cpu evaluate");
        let cpu_linear =
            test_support::gather(cpu_eval.linear_value().expect("cpu linear")).expect("cpu gather");
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_eval = evaluate(Value::GpuTensor(handle), &[]).expect("gpu evaluate");
        let gpu_linear =
            test_support::gather(gpu_eval.linear_value().expect("gpu linear")).expect("gpu gather");
        assert_eq!(gpu_linear.data, cpu_linear.data);
    }
}
