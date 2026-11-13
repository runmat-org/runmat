//! MATLAB-compatible `find` builtin with GPU-aware semantics for RunMat.
mod publication;

use runmat_accelerate_api::ProviderFindResult;
use runmat_builtins::{
    BuiltinErrorDescriptor, FIND_DIRECTION_ONLY_EXTENSION, FIND_ERROR_INTERNAL,
    FIND_ERROR_INVALID_INPUT, FIND_ERROR_PROVIDER, FIND_ERROR_PROVIDER_OUTPUT,
    FIND_INTEGER_SPARSE_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{
    ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage, LogicalArray, Tensor, Value,
};

use super::common::fits_positive_platform_index;
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
    supported_precisions: &[ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("find")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers execute find directly only when they can return exact f64 indices; f32, logical, and integer cases use a correctness-first host fallback and restore resident outputs.",
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

const BUILTIN_NAME: &str = "find";

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
        ArgToken::Integer(value) => parse_limit_integer(value),
        _ => Err(find_error_with_message(
            "find: second argument must be a scalar",
            &FIND_ERROR_INVALID_INPUT,
        )),
    }
}

#[runtime_builtin(
    name = "find",
    binding_variant = "default",
    builtin_path = "crate::builtins::array::indexing::find"
)]
async fn find_builtin(
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    evaluate(value, rest.view())
        .await?
        .into_publication(crate::output_count::current_output_count())
}

/// Evaluate `find` and return an object that can materialise the various outputs.
pub async fn evaluate(
    value: Value,
    args: crate::call::FunctionArgumentView<'_>,
) -> crate::BuiltinResult<FindEval> {
    if args.len() == 1
        && matches!(
            crate::builtins::common::arg_tokens::tokens_from_values(args).first(),
            Some(ArgToken::String(_))
        )
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FIND_DIRECTION_ONLY_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if matches!(&value, Value::SparseTensor(sparse) if sparse.integer_storage().is_some()) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FIND_INTEGER_SPARSE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let options = parse_options(args).await?;
    match value {
        Value::GpuTensor(handle) => {
            let owner = runmat_accelerate_api::provider_for_handle(&handle).ok_or_else(|| {
                find_error_with_message(
                    "find: no acceleration provider owns the input handle",
                    &FIND_ERROR_INTERNAL,
                )
            })?;
            let provider_has_exact_double_indices = matches!(
                runmat_accelerate_api::handle_precision(&handle),
                Some(runmat_accelerate_api::ProviderPrecision::F64)
            );
            let provider_indices_are_exact_double = provider_has_exact_double_indices
                && !runmat_accelerate_api::handle_is_logical(&handle)
                && runmat_accelerate_api::handle_integer_type(&handle).is_none();
            if provider_indices_are_exact_double {
                if let Some(result) = try_provider_find(owner, &handle, &options)? {
                    return Ok(FindEval::from_gpu(result));
                }
            }
            let (storage, _) = materialize_input(Value::GpuTensor(handle.clone())).await?;
            let result = compute_find(&storage, &options);
            let mut evaluation = FindEval::from_host(result);
            evaluation.source = Some(handle);
            Ok(evaluation)
        }
        Value::SparseTensor(sparse) => {
            let result = compute_find_sparse(&sparse, &options);
            Ok(FindEval::from_host(result))
        }
        other => {
            let (storage, _) = materialize_input(other).await?;
            let result = compute_find(&storage, &options);
            Ok(FindEval::from_host(result))
        }
    }
}

fn try_provider_find(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    handle: &runmat_accelerate_api::GpuTensorHandle,
    options: &FindOptions,
) -> crate::BuiltinResult<Option<ProviderFindResult>> {
    if matches!(options.direction, FindDirection::Last) {
        return Ok(None);
    }
    let direction = match options.direction {
        FindDirection::First => runmat_accelerate_api::FindDirection::First,
        FindDirection::Last => runmat_accelerate_api::FindDirection::Last,
    };
    let limit = options.effective_limit();
    if !provider.supports_find(handle, limit, direction) {
        return Ok(None);
    }
    let mut result = provider.find(handle, limit, direction).map_err(|error| {
        build_runtime_error(format!("find: provider execution failed: {error}"))
            .with_builtin(BUILTIN_NAME)
            .with_identifier(FIND_ERROR_PROVIDER.identifier.expect("provider identifier"))
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .with_source(
                runmat_accelerate_api::ProviderExecutionError::from(error).with_unproven_effects(),
            )
            .build()
    })?;
    if !valid_provider_find_result(provider, handle, limit, &result) {
        return Err(reject_find_result(
            result,
            find_provider_output_error("find: provider returned invalid output handles"),
        ));
    }
    if is_row_vector_shape(handle.shape()) {
        let matches = result.linear.shape().first().copied().unwrap_or(0);
        let shape = if handle.shape() == [1, 1] && matches == 0 {
            vec![0, 0]
        } else {
            vec![1, matches]
        };
        let view = result.linear.view().clone().with_shape(shape);
        let reshaped = result.linear.with_view(view).map_err(|error| {
            build_runtime_error(format!("find: provider row result is invalid: {error}"))
                .with_builtin(BUILTIN_NAME)
                .with_identifier(
                    FIND_ERROR_PROVIDER_OUTPUT
                        .identifier
                        .expect("provider output identifier"),
                )
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .with_source(error)
                .build()
        });
        match reshaped {
            Ok(linear) => {
                let original = std::mem::replace(&mut result.linear, linear);
                if let Err(error) = original.release() {
                    return Err(reject_find_result(
                        result,
                        find_provider_output_error(format!(
                            "find: provider row view retirement failed: {error}"
                        )),
                    ));
                }
            }
            Err(error) => return Err(reject_find_result(result, error)),
        }
    }
    Ok(Some(result))
}

fn find_provider_output_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_identifier(
            FIND_ERROR_PROVIDER_OUTPUT
                .identifier
                .expect("provider output identifier"),
        )
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}

fn reject_find_result(mut result: ProviderFindResult, mut error: RuntimeError) -> RuntimeError {
    error = gpu_helpers::reject_gpu_output(result.linear, error);
    error = gpu_helpers::reject_gpu_output(result.rows, error);
    error = gpu_helpers::reject_gpu_output(result.cols, error);
    if let Some(values) = result.values.take() {
        error = gpu_helpers::reject_gpu_output(values, error);
    }
    error
}

fn valid_provider_find_result(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    input: &runmat_accelerate_api::GpuTensorHandle,
    limit: Option<usize>,
    result: &ProviderFindResult,
) -> bool {
    let Some(values) = result.values.as_ref() else {
        return false;
    };
    let count = result.linear.shape().first().copied().unwrap_or(usize::MAX);
    let max_input = input
        .shape()
        .iter()
        .try_fold(1usize, |total, extent| total.checked_mul(*extent));
    if max_input.is_none_or(|maximum| count > maximum)
        || limit.is_some_and(|maximum| count > maximum)
    {
        return false;
    }
    let outputs = [&result.linear, &result.rows, &result.cols, values];
    outputs.iter().enumerate().all(|(index, output)| {
        output.allocation_lease().validate_live().is_ok()
            && output.shape() == [count, 1]
            && output.device_id() == input.device_id()
            && gpu_helpers::exact_provider_for_handle(output)
                .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
            && runmat_accelerate_api::handle_storage(output)
                == runmat_accelerate_api::GpuTensorStorage::Real
            && runmat_accelerate_api::handle_precision(output)
                == Some(runmat_accelerate_api::ProviderPrecision::F64)
            && runmat_accelerate_api::handle_integer_type(output).is_none()
            && !runmat_accelerate_api::handle_is_logical(output)
            && gpu_helpers::gpu_class_metadata_matches(
                output,
                Some(runmat_accelerate_api::ProviderPrecision::F64),
                None,
                false,
            )
            && !output.shares_allocation_with(input)
            && outputs[..index]
                .iter()
                .all(|previous| !output.shares_allocation_with(previous))
    })
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
    Logical(LogicalArray),
    Complex(ComplexTensor),
}

impl DataStorage {
    fn shape(&self) -> &[usize] {
        match self {
            DataStorage::Real(t) => t.shape(),
            DataStorage::Logical(t) => &t.shape,
            DataStorage::Complex(t) => t.shape(),
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
    F32(Vec<f32>),
    Logical(Vec<u8>),
    Integer(IntegerStorage),
    Complex(Vec<(f64, f64)>),
    ComplexF32(Vec<(f32, f32)>),
    IntegerComplex(IntegerComplexStorage),
}

pub struct FindEval {
    inner: FindEvalInner,
    // Retain the original allocation until deferred output uploads finish.
    source: Option<runmat_accelerate_api::GpuTensorHandle>,
}

enum FindEvalInner {
    Host { result: FindResult },
    Gpu { result: ProviderFindResult },
}

impl FindEval {
    fn output_values(&self, out_count: usize) -> crate::BuiltinResult<Vec<Value>> {
        let rows = self.row_value()?;
        let cols = match self.column_value() {
            Ok(cols) => cols,
            Err(error) => return Err(retire_unpublished_find_values(vec![rows], error)),
        };
        let mut outputs = vec![rows, cols];
        if find_outputs_alias(&outputs) {
            return Err(retire_unpublished_find_values(
                outputs,
                find_provider_output_error("find: provider aliased distinct outputs"),
            ));
        }
        if out_count >= 3 {
            match self.values_value() {
                Ok(values) => outputs.push(values),
                Err(error) => return Err(retire_unpublished_find_values(outputs, error)),
            }
            if find_outputs_alias(&outputs) {
                return Err(retire_unpublished_find_values(
                    outputs,
                    find_provider_output_error("find: provider aliased distinct outputs"),
                ));
            }
        }
        Ok(outputs)
    }

    fn from_host(result: FindResult) -> Self {
        Self {
            source: None,
            inner: FindEvalInner::Host { result },
        }
    }

    fn from_gpu(result: ProviderFindResult) -> Self {
        Self {
            source: None,
            inner: FindEvalInner::Gpu { result },
        }
    }

    pub fn linear_value(&self) -> crate::BuiltinResult<Value> {
        match &self.inner {
            FindEvalInner::Host { result } => {
                let tensor = result.linear_tensor()?;
                tensor_to_value(tensor, self.source.as_ref())
            }
            FindEvalInner::Gpu { result } => Ok(Value::GpuTensor(result.linear.clone())),
        }
    }

    pub fn row_value(&self) -> crate::BuiltinResult<Value> {
        match &self.inner {
            FindEvalInner::Host { result } => {
                let tensor = result.row_tensor()?;
                tensor_to_value(tensor, self.source.as_ref())
            }
            FindEvalInner::Gpu { result } => Ok(Value::GpuTensor(result.rows.clone())),
        }
    }

    pub fn column_value(&self) -> crate::BuiltinResult<Value> {
        match &self.inner {
            FindEvalInner::Host { result } => {
                let tensor = result.column_tensor()?;
                tensor_to_value(tensor, self.source.as_ref())
            }
            FindEvalInner::Gpu { result } => Ok(Value::GpuTensor(result.cols.clone())),
        }
    }

    pub fn values_value(&self) -> crate::BuiltinResult<Value> {
        match &self.inner {
            FindEvalInner::Host { result } => result.values_value(self.source.as_ref()),
            FindEvalInner::Gpu { result } => result
                .values
                .as_ref()
                .map(|handle| Value::GpuTensor(handle.clone()))
                .ok_or_else(|| find_error(&FIND_ERROR_PROVIDER_OUTPUT)),
        }
    }
}

async fn parse_options(
    args: crate::call::FunctionArgumentView<'_>,
) -> crate::BuiltinResult<FindOptions> {
    parse_find_tokens(&crate::builtins::common::arg_tokens::tokens_from_values(
        args,
    ))
}

fn parse_limit_integer(value: &IntValue) -> crate::BuiltinResult<usize> {
    let value = value.try_to_usize().ok_or_else(|| {
        find_error_with_message(
            "find: K must be a positive integer within the supported range",
            &FIND_ERROR_INVALID_INPUT,
        )
    })?;
    if value == 0 {
        return Err(find_error_with_message(
            "find: K must be a positive integer",
            &FIND_ERROR_INVALID_INPUT,
        ));
    }
    Ok(value)
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
    if rounded <= 0.0 {
        return Err(find_error_with_message(
            "find: K must be a positive integer",
            &FIND_ERROR_INVALID_INPUT,
        ));
    }
    if !fits_positive_platform_index(rounded) {
        return Err(find_error_with_message(
            "find: K exceeds the maximum supported index range",
            &FIND_ERROR_INVALID_INPUT,
        ));
    }
    Ok(rounded as usize)
}

async fn materialize_input(value: Value) -> crate::BuiltinResult<(DataStorage, bool)> {
    match value {
        Value::GpuTensor(handle) => {
            let is_logical = runmat_accelerate_api::handle_is_logical(&handle);
            let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
            if is_logical {
                let data = (0..tensor::tensor_element_len(&tensor))
                    .map(|index| u8::from(tensor::tensor_value_f64(&tensor, index) != 0.0))
                    .collect();
                let shape = tensor.shape().to_vec();
                return LogicalArray::new(data, shape)
                    .map(|logical| (DataStorage::Logical(logical), true))
                    .map_err(|e| {
                        find_error_with_message(format!("find: {e}"), &FIND_ERROR_INTERNAL)
                    });
            }
            Ok((DataStorage::Real(tensor), true))
        }
        Value::Tensor(tensor) => Ok((DataStorage::Real(tensor), false)),
        Value::SparseTensor(sparse) => {
            let dense = if sparse.is_logical() {
                tensor::logical_to_tensor(&sparse.to_dense_logical().map_err(|e| {
                    find_error_with_message(format!("find: {e}"), &FIND_ERROR_INTERNAL)
                })?)
                .map_err(|message| find_error_with_message(message, &FIND_ERROR_INTERNAL))?
            } else {
                sparse.to_dense().map_err(|e| {
                    find_error_with_message(format!("find: {e}"), &FIND_ERROR_INTERNAL)
                })?
            };
            Ok((DataStorage::Real(dense), false))
        }
        Value::LogicalArray(logical) => Ok((DataStorage::Logical(logical), false)),
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1])
                .map_err(|e| find_error_with_message(format!("find: {e}"), &FIND_ERROR_INTERNAL))?;
            Ok((DataStorage::Real(tensor), false))
        }
        Value::Int(i) => {
            let tensor = Tensor::new_integer(integer_storage_from_scalar(&i), vec![1, 1])
                .map_err(|e| find_error_with_message(format!("find: {e}"), &FIND_ERROR_INTERNAL))?;
            Ok((DataStorage::Real(tensor), false))
        }
        Value::Bool(b) => LogicalArray::new(vec![u8::from(b)], vec![1, 1])
            .map(|logical| (DataStorage::Logical(logical), false))
            .map_err(|e| find_error_with_message(format!("find: {e}"), &FIND_ERROR_INTERNAL)),
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| find_error_with_message(format!("find: {e}"), &FIND_ERROR_INTERNAL))?;
            Ok((DataStorage::Complex(tensor), false))
        }
        Value::ComplexTensor(tensor) => Ok((DataStorage::Complex(tensor), false)),
        Value::CharArray(chars) => {
            let mut data = Vec::with_capacity(chars.data().len());
            for c in 0..chars.cols {
                for r in 0..chars.rows {
                    let ch = chars.data()[r * chars.cols + c].raw();
                    data.push(f64::from(ch));
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
            let typed_storage = tensor.integer_storage();

            if matches!(limit, Some(0)) {
                return FindResult::new(shape, indices, find_values_for_tensor(tensor, &[]));
            }

            let len = typed_storage
                .map(|storage| storage.len())
                .unwrap_or_else(|| tensor::tensor_element_len(tensor));
            match options.direction {
                FindDirection::First => {
                    for idx in 0..len {
                        let nonzero = typed_storage.map_or_else(
                            || tensor::tensor_value_f64(tensor, idx) != 0.0,
                            |storage| {
                                storage
                                    .value_at(idx)
                                    .map(|value| !value.is_zero())
                                    .expect("typed integer storage is structurally valid")
                            },
                        );
                        if nonzero {
                            indices.push(idx + 1);
                            if limit.is_some_and(|k| indices.len() >= k) {
                                break;
                            }
                        }
                    }
                }
                FindDirection::Last => {
                    for idx in (0..len).rev() {
                        let nonzero = typed_storage.map_or_else(
                            || tensor::tensor_value_f64(tensor, idx) != 0.0,
                            |storage| {
                                storage
                                    .value_at(idx)
                                    .map(|value| !value.is_zero())
                                    .expect("typed integer storage is structurally valid")
                            },
                        );
                        if nonzero {
                            indices.push(idx + 1);
                            if limit.is_some_and(|k| indices.len() >= k) {
                                break;
                            }
                        }
                    }
                }
            }

            if matches!(options.direction, FindDirection::Last) {
                indices.reverse();
            }
            let values = find_values_for_tensor(tensor, &indices);
            FindResult::new(shape, indices, values)
        }
        DataStorage::Logical(logical) => {
            let mut indices = Vec::new();
            if !matches!(options.effective_limit(), Some(0)) {
                let iter: Box<dyn Iterator<Item = usize>> = match options.direction {
                    FindDirection::First => Box::new(0..logical.data.len()),
                    FindDirection::Last => Box::new((0..logical.data.len()).rev()),
                };
                for idx in iter {
                    if logical.data[idx] != 0 {
                        indices.push(idx + 1);
                        if options
                            .effective_limit()
                            .is_some_and(|limit| indices.len() >= limit)
                        {
                            break;
                        }
                    }
                }
            }
            if matches!(options.direction, FindDirection::Last) {
                indices.reverse();
            }
            let values = FindValues::Logical(vec![1; indices.len()]);
            FindResult::new(shape, indices, values)
        }
        DataStorage::Complex(tensor) => {
            let mut indices = Vec::new();

            if matches!(limit, Some(0)) {
                let values = find_values_for_complex_tensor(tensor, &indices);
                return FindResult::new(shape, indices, values);
            }

            let len = tensor.len();
            let nonzero = |index| {
                let (real, imaginary) = tensor
                    .numeric_value_at(index)
                    .expect("validated complex find storage");
                !real.is_zero() || !imaginary.is_zero()
            };
            match options.direction {
                FindDirection::First => {
                    for idx in 0..len {
                        if nonzero(idx) {
                            indices.push(idx + 1);
                            if limit.is_some_and(|k| indices.len() >= k) {
                                break;
                            }
                        }
                    }
                }
                FindDirection::Last => {
                    for idx in (0..len).rev() {
                        if nonzero(idx) {
                            indices.push(idx + 1);
                            if limit.is_some_and(|k| indices.len() >= k) {
                                break;
                            }
                        }
                    }
                }
            }

            if matches!(options.direction, FindDirection::Last) {
                indices.reverse();
            }
            let values = find_values_for_complex_tensor(tensor, &indices);
            FindResult::new(shape, indices, values)
        }
    }
}

fn sparse_find_values(
    sparse: &runmat_value::SparseTensor,
    real_values: Vec<f64>,
    single_values: Vec<f32>,
    complex_values: Vec<(f64, f64)>,
    logical_values: Vec<u8>,
    integer_value_indices: &[usize],
) -> FindValues {
    if sparse.is_logical() {
        FindValues::Logical(logical_values)
    } else if sparse.is_complex() {
        if sparse.numeric_dtype() == Some(runmat_value::NumericDType::F32) {
            FindValues::ComplexF32(
                complex_values
                    .into_iter()
                    .map(|(real, imaginary)| (real as f32, imaginary as f32))
                    .collect(),
            )
        } else {
            FindValues::Complex(complex_values)
        }
    } else if let Some(storage) = sparse.integer_storage() {
        FindValues::Integer(select_integer_values(storage, integer_value_indices))
    } else if sparse.as_f32_slice().is_some() {
        FindValues::F32(single_values)
    } else {
        FindValues::Real(real_values)
    }
}

fn sparse_stored_value_is_nonzero(sparse: &runmat_value::SparseTensor, index: usize) -> bool {
    if let Some(value) = sparse.complex_value_at(index) {
        return value.0 != 0.0 || value.1 != 0.0;
    }
    !sparse
        .numeric_value_at(index)
        .expect("SparseTensor value storage is consistent")
        .is_zero()
}

fn compute_find_sparse(sparse: &runmat_value::SparseTensor, options: &FindOptions) -> FindResult {
    let shape = vec![sparse.rows, sparse.cols];
    let limit = options.effective_limit();

    let mut indices = Vec::new();
    let mut values = Vec::new();
    let mut single_values = Vec::new();
    let mut complex_values = Vec::new();
    let mut logical_values = Vec::new();
    let integer_storage = sparse.integer_storage();
    let floating_values = sparse.as_f64_slice();
    let native_single_values = sparse.as_f32_slice();
    let mut integer_value_indices = Vec::new();

    if matches!(limit, Some(0)) {
        let values = sparse_find_values(
            sparse,
            values,
            single_values,
            complex_values,
            logical_values,
            &integer_value_indices,
        );
        return FindResult::new(shape, indices, values);
    }

    match options.direction {
        FindDirection::First => {
            for col in 0..sparse.cols {
                let col_start = sparse.col_ptrs[col];
                let col_end = sparse.col_ptrs[col + 1];
                for idx in col_start..col_end {
                    let row = sparse.row_indices[idx];
                    if sparse_stored_value_is_nonzero(sparse, idx) {
                        let linear_idx = row + col * sparse.rows;
                        indices.push(linear_idx + 1);
                        if sparse.is_logical() {
                            logical_values.push(1);
                        } else if let Some(value) = sparse.complex_value_at(idx) {
                            complex_values.push((value.0, value.1));
                        } else if integer_storage.is_some() {
                            integer_value_indices.push(idx);
                        } else if let Some(native_single_values) = native_single_values {
                            single_values.push(native_single_values[idx]);
                        } else {
                            values.push(floating_values.expect("double sparse storage")[idx]);
                        }
                        if limit.is_some_and(|k| indices.len() >= k) {
                            let values = sparse_find_values(
                                sparse,
                                values,
                                single_values,
                                complex_values,
                                logical_values,
                                &integer_value_indices,
                            );
                            return FindResult::new(shape, indices, values);
                        }
                    }
                }
            }
        }
        FindDirection::Last => {
            for col in (0..sparse.cols).rev() {
                let col_start = sparse.col_ptrs[col];
                let col_end = sparse.col_ptrs[col + 1];
                for idx in (col_start..col_end).rev() {
                    let row = sparse.row_indices[idx];
                    if sparse_stored_value_is_nonzero(sparse, idx) {
                        let linear_idx = row + col * sparse.rows;
                        indices.push(linear_idx + 1);
                        if sparse.is_logical() {
                            logical_values.push(1);
                        } else if let Some(value) = sparse.complex_value_at(idx) {
                            complex_values.push((value.0, value.1));
                        } else if integer_storage.is_some() {
                            integer_value_indices.push(idx);
                        } else if let Some(native_single_values) = native_single_values {
                            single_values.push(native_single_values[idx]);
                        } else {
                            values.push(floating_values.expect("double sparse storage")[idx]);
                        }
                        if limit.is_some_and(|k| indices.len() >= k) {
                            indices.reverse();
                            values.reverse();
                            single_values.reverse();
                            complex_values.reverse();
                            logical_values.reverse();
                            integer_value_indices.reverse();
                            let values = sparse_find_values(
                                sparse,
                                values,
                                single_values,
                                complex_values,
                                logical_values,
                                &integer_value_indices,
                            );
                            return FindResult::new(shape, indices, values);
                        }
                    }
                }
            }
        }
    }

    if matches!(options.direction, FindDirection::Last) {
        indices.reverse();
        values.reverse();
        single_values.reverse();
        complex_values.reverse();
        logical_values.reverse();
        integer_value_indices.reverse();
    }
    let values = sparse_find_values(
        sparse,
        values,
        single_values,
        complex_values,
        logical_values,
        &integer_value_indices,
    );
    FindResult::new(shape, indices, values)
}

fn is_row_vector_shape(shape: &[usize]) -> bool {
    shape.len() == 2 && shape.first() == Some(&1)
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
        let data = self
            .indices
            .iter()
            .map(|&idx| exact_index_as_f64(idx))
            .collect::<crate::BuiltinResult<Vec<_>>>()?;
        let shape = if data.is_empty() && matches!(self.shape.as_slice(), [0, 0] | [1, 1]) {
            vec![0, 0]
        } else if is_row_vector_shape(&self.shape) {
            vec![1, data.len()]
        } else {
            vec![data.len(), 1]
        };
        Tensor::new(data, shape)
            .map_err(|e| find_error_with_message(format!("find: {e}"), &FIND_ERROR_INTERNAL))
    }

    fn row_tensor(&self) -> crate::BuiltinResult<Tensor> {
        let mut data = Vec::with_capacity(self.indices.len());
        let rows = self.shape.first().copied().unwrap_or(1).max(1);
        for &idx in &self.indices {
            let zero_based = idx - 1;
            let row = (zero_based % rows) + 1;
            data.push(exact_index_as_f64(row)?);
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
            data.push(exact_index_as_f64(col)?);
        }
        Tensor::new(data, vec![self.indices.len(), 1])
            .map_err(|e| find_error_with_message(format!("find: {e}"), &FIND_ERROR_INTERNAL))
    }

    fn values_value(
        &self,
        source: Option<&runmat_accelerate_api::GpuTensorHandle>,
    ) -> crate::BuiltinResult<Value> {
        match &self.values {
            FindValues::Real(values) => {
                let tensor = Tensor::new(values.clone(), vec![values.len(), 1]).map_err(|e| {
                    find_error_with_message(format!("find: {e}"), &FIND_ERROR_INTERNAL)
                })?;
                tensor_to_value(tensor, source)
            }
            FindValues::F32(values) => {
                let tensor =
                    Tensor::from_f32(values.clone(), vec![values.len(), 1]).map_err(|e| {
                        find_error_with_message(format!("find: {e}"), &FIND_ERROR_INTERNAL)
                    })?;
                tensor_to_value(tensor, source)
            }
            FindValues::Logical(values) => {
                let logical =
                    LogicalArray::new(values.clone(), vec![values.len(), 1]).map_err(|e| {
                        find_error_with_message(format!("find: {e}"), &FIND_ERROR_INTERNAL)
                    })?;
                restore_find_value(Value::LogicalArray(logical), source)
            }
            FindValues::Integer(values) => integer_values_to_value(values.clone(), source),
            FindValues::Complex(values) => {
                let tensor =
                    ComplexTensor::new(values.clone(), vec![values.len(), 1]).map_err(|e| {
                        find_error_with_message(format!("find: {e}"), &FIND_ERROR_INTERNAL)
                    })?;
                complex_tensor_to_value(tensor, source)
            }
            FindValues::ComplexF32(values) => {
                let tensor = ComplexTensor::from_f32(values.clone(), vec![values.len(), 1])
                    .map_err(|e| {
                        find_error_with_message(format!("find: {e}"), &FIND_ERROR_INTERNAL)
                    })?;
                complex_tensor_to_value(tensor, source)
            }
            FindValues::IntegerComplex(storage) => {
                let tensor = ComplexTensor::new_integer(storage.clone(), vec![storage.len(), 1])
                    .map_err(|e| {
                        find_error_with_message(format!("find: {e}"), &FIND_ERROR_INTERNAL)
                    })?;
                complex_tensor_to_value(tensor, source)
            }
        }
    }
}

fn exact_index_as_f64(index: usize) -> crate::BuiltinResult<f64> {
    const MAX_EXACT_BINARY64_INTEGER: u128 = 1_u128 << 53;
    if (index as u128) > MAX_EXACT_BINARY64_INTEGER {
        return Err(find_error_with_message(
            "find: index exceeds the exact binary64 index range",
            &FIND_ERROR_INVALID_INPUT,
        ));
    }
    Ok(index as f64)
}

fn find_values_for_tensor(tensor: &Tensor, indices: &[usize]) -> FindValues {
    let Some(storage) = tensor.integer_storage() else {
        if let Some(values) = tensor.as_f32_slice() {
            return FindValues::F32(indices.iter().map(|index| values[index - 1]).collect());
        }
        return FindValues::Real(
            indices
                .iter()
                .map(|index| tensor::tensor_value_f64(tensor, index - 1))
                .collect(),
        );
    };
    let selected: Vec<usize> = indices.iter().map(|index| index - 1).collect();
    FindValues::Integer(select_integer_values(storage, &selected))
}

fn find_values_for_complex_tensor(tensor: &ComplexTensor, indices: &[usize]) -> FindValues {
    if let Some(values) = tensor.as_f32_slice() {
        return FindValues::ComplexF32(
            indices
                .iter()
                .map(|index| values[index - 1].into())
                .collect(),
        );
    }
    if let Some(values) = tensor.as_f64_slice() {
        return FindValues::Complex(
            indices
                .iter()
                .map(|index| values[index - 1].into())
                .collect(),
        );
    }
    let storage = tensor
        .integer_storage()
        .expect("complex find has native floating or integer storage");
    let selected: Vec<usize> = indices.iter().map(|index| index - 1).collect();
    let real = select_integer_values(storage.real(), &selected);
    let imag = select_integer_values(storage.imag(), &selected);
    let storage = IntegerComplexStorage::new(real, imag)
        .expect("paired typed complex storage preserves class and length through find");
    FindValues::IntegerComplex(storage)
}

fn select_integer_values(storage: &IntegerStorage, indices: &[usize]) -> IntegerStorage {
    macro_rules! select {
        ($values:expr, $variant:ident) => {
            IntegerStorage::$variant(indices.iter().map(|&index| $values[index]).collect())
        };
    }
    match storage {
        IntegerStorage::I8(values) => select!(values, I8),
        IntegerStorage::I16(values) => select!(values, I16),
        IntegerStorage::I32(values) => select!(values, I32),
        IntegerStorage::I64(values) => select!(values, I64),
        IntegerStorage::U8(values) => select!(values, U8),
        IntegerStorage::U16(values) => select!(values, U16),
        IntegerStorage::U32(values) => select!(values, U32),
        IntegerStorage::U64(values) => select!(values, U64),
    }
}

fn integer_values_to_value(
    storage: IntegerStorage,
    source: Option<&runmat_accelerate_api::GpuTensorHandle>,
) -> crate::BuiltinResult<Value> {
    if storage.len() == 1 && source.is_none() {
        return Ok(Value::Int(integer_storage_value(&storage, 0)));
    }
    let shape = vec![storage.len(), 1];
    let tensor = Tensor::new_integer(storage, shape)
        .map_err(|e| find_error_with_message(format!("find: {e}"), &FIND_ERROR_INTERNAL))?;
    tensor_to_value(tensor, source)
}

fn integer_storage_value(storage: &IntegerStorage, index: usize) -> IntValue {
    match storage {
        IntegerStorage::I8(values) => IntValue::I8(values[index]),
        IntegerStorage::I16(values) => IntValue::I16(values[index]),
        IntegerStorage::I32(values) => IntValue::I32(values[index]),
        IntegerStorage::I64(values) => IntValue::I64(values[index]),
        IntegerStorage::U8(values) => IntValue::U8(values[index]),
        IntegerStorage::U16(values) => IntValue::U16(values[index]),
        IntegerStorage::U32(values) => IntValue::U32(values[index]),
        IntegerStorage::U64(values) => IntValue::U64(values[index]),
    }
}

fn integer_storage_from_scalar(value: &IntValue) -> IntegerStorage {
    match value {
        IntValue::I8(value) => IntegerStorage::I8(vec![*value]),
        IntValue::I16(value) => IntegerStorage::I16(vec![*value]),
        IntValue::I32(value) => IntegerStorage::I32(vec![*value]),
        IntValue::I64(value) => IntegerStorage::I64(vec![*value]),
        IntValue::U8(value) => IntegerStorage::U8(vec![*value]),
        IntValue::U16(value) => IntegerStorage::U16(vec![*value]),
        IntValue::U32(value) => IntegerStorage::U32(vec![*value]),
        IntValue::U64(value) => IntegerStorage::U64(vec![*value]),
    }
}

fn tensor_to_value(
    tensor: Tensor,
    source: Option<&runmat_accelerate_api::GpuTensorHandle>,
) -> crate::BuiltinResult<Value> {
    // The shared restore boundary takes a native integer tensor, not an Int
    // scalar. Other host carriers retain the public scalarization convention.
    let value = if source.is_some() && tensor.integer_storage().is_some() {
        Value::Tensor(tensor)
    } else {
        tensor::tensor_into_value(tensor)
    };
    restore_find_value(value, source)
}

fn complex_tensor_to_value(
    tensor: ComplexTensor,
    source: Option<&runmat_accelerate_api::GpuTensorHandle>,
) -> crate::BuiltinResult<Value> {
    restore_find_value(complex_tensor_into_value(tensor), source)
}

fn restore_find_value(
    value: Value,
    source: Option<&runmat_accelerate_api::GpuTensorHandle>,
) -> crate::BuiltinResult<Value> {
    match source {
        Some(source) => gpu_helpers::restore_class_preserving_value(source, value, BUILTIN_NAME),
        None => Ok(value),
    }
}

fn retire_unpublished_find_values(outputs: Vec<Value>, mut error: RuntimeError) -> RuntimeError {
    for output in outputs {
        if let Value::GpuTensor(handle) = output {
            error = gpu_helpers::reject_gpu_output(handle, error);
        }
    }
    error
}

fn find_outputs_alias(outputs: &[Value]) -> bool {
    outputs.iter().enumerate().any(|(index, output)| {
        let Value::GpuTensor(handle) = output else {
            return false;
        };
        outputs[..index].iter().any(|previous| {
            matches!(previous, Value::GpuTensor(other) if handle.shares_allocation_with(other))
        })
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::{
        allocation::WeakAllocationLease,
        test_support::{RetirementProbe, TestAllocationOwner},
        AccelProvider, GpuTensorHandle, GpuTensorStorage, HostLogicalTensorView,
        HostNumericTensorView, HostTensorView, NumericElementType, ProviderExecutionError,
        ProviderExecutionResult,
    };
    use runmat_builtins::{FIND_EXTENSIONS, FIND_INTEGER_CAPABILITIES};
    use runmat_value::{CharArray, IntValue};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct FailingFindProvider {
        inner: runmat_accelerate::simple_provider::InProcessProvider,
        foreign: TestAllocationOwner,
        fault: FindFault,
        calls: AtomicUsize,
        downloads: AtomicUsize,
        output_leases: std::sync::Mutex<Vec<WeakAllocationLease>>,
        foreign_probes: std::sync::Mutex<Vec<RetirementProbe>>,
        restore_fault: Option<RestoreFault>,
        restore_calls: AtomicUsize,
        restore_leases: std::sync::Mutex<Vec<WeakAllocationLease>>,
        restore_source: std::sync::Mutex<Option<GpuTensorHandle>>,
    }

    #[derive(Clone, Copy, Debug)]
    enum FindFault {
        Execution,
        MissingValues,
        AliasedInput,
        WrongShape,
        WrongClass,
        ForeignOwner,
        CrossOutputAlias,
        ExpiredOutput,
    }

    #[derive(Clone, Copy, Debug)]
    enum RestoreFault {
        Success,
        SourceAlias,
        Unproven,
        AfterEffect,
        Unsupported,
        SecondFailure,
    }

    impl runmat_accelerate_api::AccelProvider for FailingFindProvider {
        fn upload_numeric_raw(
            &self,
            host: &HostNumericTensorView,
        ) -> ProviderExecutionResult<GpuTensorHandle> {
            let call = self.restore_calls.fetch_add(1, Ordering::SeqCst) + 1;
            let Some(fault) = self.restore_fault else {
                return self.inner.upload_numeric_raw(host);
            };
            match fault {
                RestoreFault::Success => {
                    let output = self.inner.upload_numeric_raw(host)?;
                    self.restore_leases
                        .lock()
                        .unwrap()
                        .push(output.allocation_lease().downgrade());
                    Ok(output)
                }
                RestoreFault::SecondFailure if call == 1 => {
                    let output = self.inner.upload_numeric_raw(host)?;
                    self.restore_leases
                        .lock()
                        .unwrap()
                        .push(output.allocation_lease().downgrade());
                    Ok(output)
                }
                RestoreFault::SourceAlias => Ok(self
                    .restore_source
                    .lock()
                    .unwrap()
                    .as_ref()
                    .expect("protected source")
                    .clone()),
                RestoreFault::Unproven => Err(ProviderExecutionError::from(anyhow::anyhow!(
                    "unproven upload failure"
                ))
                .with_unproven_effects()),
                RestoreFault::AfterEffect | RestoreFault::SecondFailure => {
                    let allocation = self.foreign.allocate_zeroed(
                        host.shape.to_vec(),
                        host.data.element_type(),
                        host.storage,
                    )?;
                    self.foreign_probes.lock().unwrap().push(allocation.probe);
                    allocation
                        .handle
                        .release()
                        .map_err(ProviderExecutionError::from_source)?;
                    Err(ProviderExecutionError::from(anyhow::anyhow!(
                        "post-allocation upload failure"
                    ))
                    .after_effect())
                }
                RestoreFault::Unsupported => {
                    Err(runmat_accelerate_api::unsupported_provider_operation(
                        "pre-effect representation unavailable",
                    )
                    .into())
                }
            }
        }

        fn upload(
            &self,
            host: &HostTensorView,
        ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
            self.inner.upload(host)
        }

        fn download<'a>(
            &'a self,
            handle: &'a runmat_accelerate_api::GpuTensorHandle,
        ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
            self.downloads.fetch_add(1, Ordering::SeqCst);
            self.inner.download(handle)
        }

        fn download_numeric<'a>(
            &'a self,
            handle: &'a runmat_accelerate_api::GpuTensorHandle,
        ) -> runmat_accelerate_api::AccelNumericDownloadFuture<'a> {
            self.downloads.fetch_add(1, Ordering::SeqCst);
            self.inner.download_numeric(handle)
        }

        fn device_id(&self) -> u32 {
            self.inner.device_id()
        }

        fn device_info(&self) -> String {
            self.inner.device_info()
        }

        fn precision(&self) -> runmat_accelerate_api::ProviderPrecision {
            self.inner.precision()
        }

        fn supports_find(
            &self,
            _input: &runmat_accelerate_api::GpuTensorHandle,
            _limit: Option<usize>,
            _direction: runmat_accelerate_api::FindDirection,
        ) -> bool {
            true
        }

        fn find(
            &self,
            input: &runmat_accelerate_api::GpuTensorHandle,
            limit: Option<usize>,
            direction: runmat_accelerate_api::FindDirection,
        ) -> anyhow::Result<runmat_accelerate_api::ProviderFindResult> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if matches!(self.fault, FindFault::Execution) {
                return Err(anyhow::anyhow!("deterministic admitted find failure"));
            }
            let mut result = self.inner.find(input, limit, direction)?;
            {
                let mut leases = self.output_leases.lock().unwrap();
                leases.extend(
                    [&result.linear, &result.rows, &result.cols]
                        .into_iter()
                        .chain(result.values.iter())
                        .map(|handle| handle.allocation_lease().downgrade()),
                );
            }
            match self.fault {
                FindFault::Execution => unreachable!(),
                FindFault::MissingValues => {
                    if let Some(values) = result.values.take() {
                        values.release().expect("release omitted fixture values");
                    }
                }
                FindFault::AliasedInput => {
                    let alias = input.with_view(input.view().clone())?;
                    assert_ne!(alias.view_identity(), input.view_identity());
                    let discarded = std::mem::replace(&mut result.rows, alias);
                    discarded.release().expect("release replaced fixture rows");
                }
                FindFault::WrongShape => {
                    let count = result.rows.shape()[0];
                    result.rows = result
                        .rows
                        .with_view(result.rows.view().clone().with_shape(vec![1, count]))?;
                }
                FindFault::WrongClass => {
                    let count = result.rows.shape()[0];
                    let logical = self
                        .inner
                        .upload_logical(&HostLogicalTensorView {
                            data: &vec![1; count],
                            shape: &[count, 1],
                        })
                        .expect("fixture logical output allocation");
                    let discarded = std::mem::replace(&mut result.rows, logical);
                    discarded.release().expect("release replaced fixture rows");
                }
                FindFault::ForeignOwner | FindFault::ExpiredOutput => {
                    let count = result.rows.shape()[0];
                    let allocation = self.foreign.allocate_zeroed(
                        vec![count, 1],
                        NumericElementType::F64,
                        GpuTensorStorage::Real,
                    )?;
                    if matches!(self.fault, FindFault::ExpiredOutput) {
                        allocation.probe.mark_unavailable();
                    }
                    self.foreign_probes.lock().unwrap().push(allocation.probe);
                    let discarded = std::mem::replace(&mut result.rows, allocation.handle);
                    discarded.release().expect("release replaced fixture rows");
                }
                FindFault::CrossOutputAlias => {
                    let alias = result.linear.with_view(result.linear.view().clone())?;
                    assert_ne!(alias.view_identity(), result.linear.view_identity());
                    let discarded = std::mem::replace(&mut result.rows, alias);
                    discarded.release().expect("release replaced fixture rows");
                }
            }
            Ok(result)
        }
    }

    fn find_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(super::find_builtin(
            value,
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
    }

    fn evaluate(value: Value, rest: &[Value]) -> crate::BuiltinResult<FindEval> {
        block_on(super::evaluate(
            value,
            crate::call::FunctionArgumentView::from_slice(rest),
        ))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn find_linear_indices_basic() {
        let tensor = Tensor::new(vec![0.0, 4.0, 0.0, 7.0, 0.0, 9.0], vec![2, 3]).unwrap();
        let value = find_builtin(Value::Tensor(tensor), Vec::new()).expect("find");
        match value {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), [3, 1]);
                assert_eq!(t.materialize_f64(), vec![2.0, 4.0, 6.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn find_integer_tokens_parse_exact_limits() {
        let options =
            parse_find_tokens(&[ArgToken::Integer(IntValue::U64(2))]).expect("uint64 limit");
        assert_eq!(options.limit, Some(2));
        assert_eq!(options.direction, FindDirection::First);

        let options = parse_find_tokens(&[
            ArgToken::Integer(IntValue::U16(3)),
            ArgToken::String("last".to_string()),
        ])
        .expect("integer limit with direction");
        assert_eq!(options.limit, Some(3));
        assert_eq!(options.direction, FindDirection::Last);

        let err = parse_find_tokens(&[ArgToken::Integer(IntValue::I64(-1))])
            .expect_err("negative integer limit must reject");
        assert_eq!(err.identifier(), FIND_ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn find_float_limits_reject_oversized_values_before_casting() {
        assert!(parse_find_tokens(&[ArgToken::Number(1.0e300)]).is_err());
        assert!(parse_find_tokens(&[ArgToken::Number(usize::MAX as f64)]).is_err());
        assert!(parse_find_tokens(&[ArgToken::Number(0.0)]).is_err());
        assert!(parse_find_tokens(&[ArgToken::Integer(IntValue::U8(0))]).is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn find_limited_first() {
        let tensor = Tensor::new(vec![0.0, 3.0, 5.0, 0.0, 8.0], vec![1, 5]).unwrap();
        let result =
            find_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(2))]).expect("find");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), [1, 2]);
                assert_eq!(t.materialize_f64(), vec![2.0, 3.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn find_last_single() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 6.0, 0.0, 2.0], vec![1, 6]).unwrap();
        let result = find_builtin(Value::Tensor(tensor), vec![Value::from("last")]).expect("find");
        match result {
            Value::Num(n) => assert_eq!(n, 6.0),
            Value::Tensor(t) => {
                assert_eq!(t.materialize_f64(), vec![6.0]);
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
            other => panic!("expected complex result, got {other:?}"),
        }
        let single =
            ComplexTensor::from_f32(vec![(0.0, 0.0), (1.0, 2.0), (0.0, 0.0)], vec![3, 1]).unwrap();
        let eval = evaluate(Value::ComplexTensor(single), &[]).expect("single complex find");
        let Value::ComplexTensor(values) = eval.values_value().expect("single values") else {
            panic!("single complex values must retain a tensor carrier");
        };
        assert_eq!(values.shape(), [1, 1]);
        assert_eq!(values.materialize_f64(), vec![(1.0, 2.0)]);

        // Selection copies only chosen native components, preserving even
        // subnormals, signed zero and NaN payload bits in either precision.
        let single = [
            (0.0, -0.0),
            (f32::from_bits(1), -0.0),
            (f32::from_bits(0x7fc0_1234), 3.0),
            (f32::INFINITY, f32::NEG_INFINITY),
        ];
        let double = [
            (0.0, -0.0),
            (f64::from_bits(1), -0.0),
            (f64::from_bits(0x7ff8_0000_0000_1234), 3.0),
            (f64::INFINITY, f64::NEG_INFINITY),
        ];
        for (limit, direction, expected) in [
            (Some(0), FindDirection::First, &[][..]),
            (Some(1), FindDirection::First, &[2][..]),
            (Some(2), FindDirection::Last, &[3, 4][..]),
            (None, FindDirection::First, &[2, 3, 4][..]),
            (None, FindDirection::Last, &[4][..]),
        ] {
            let options = FindOptions { limit, direction };
            let input =
                DataStorage::Complex(ComplexTensor::from_f32(single.to_vec(), vec![2, 2]).unwrap());
            let result = compute_find(&input, &options);
            assert_eq!(result.indices, expected);
            let FindValues::ComplexF32(values) = result.values else {
                panic!("native single selection must retain its component class");
            };
            assert_eq!(values.len(), expected.len());
            for ((real, imaginary), &index) in values.iter().zip(expected) {
                let (expected_real, expected_imaginary) = single[index - 1];
                assert_eq!(real.to_bits(), expected_real.to_bits());
                assert_eq!(imaginary.to_bits(), expected_imaginary.to_bits());
            }
            let input =
                DataStorage::Complex(ComplexTensor::new(double.to_vec(), vec![2, 2]).unwrap());
            let result = compute_find(&input, &options);
            assert_eq!(result.indices, expected);
            let FindValues::Complex(values) = result.values else {
                panic!("native double selection must retain its component class");
            };
            assert_eq!(values.len(), expected.len());
            for ((real, imaginary), &index) in values.iter().zip(expected) {
                let (expected_real, expected_imaginary) = double[index - 1];
                assert_eq!(real.to_bits(), expected_real.to_bits());
                assert_eq!(imaginary.to_bits(), expected_imaginary.to_bits());
            }
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn find_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 4.0, 0.0, 7.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let result = find_builtin(Value::GpuTensor(handle), Vec::new()).expect("find");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape(), [2, 1]);
            assert_eq!(gathered.materialize_f64(), vec![2.0, 4.0]);
        });
    }

    #[test]
    fn find_f32_resident_fallback_returns_host_double_indices_when_owner_cannot_store_double() {
        test_support::with_f32_test_provider(|provider| {
            let values = [0.0, 4.0, 0.0, 7.0];
            let handle = provider
                .upload(&HostTensorView {
                    data: &values,
                    shape: &[2, 2],
                })
                .expect("upload f32-owner input")
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
                .expect("known automatic intent");

            let eval = evaluate(Value::GpuTensor(handle), &[]).expect("find fallback");
            let Value::Tensor(indices) = eval.linear_value().expect("linear indices") else {
                panic!("expected host double indices");
            };
            assert_eq!(indices.numeric_dtype(), runmat_value::NumericDType::F64);
            assert_eq!(indices.materialize_f64(), vec![2.0, 4.0]);
        });
    }

    #[test]
    fn find_resident_logical_value_output_stays_logical_and_resident() {
        test_support::with_test_provider(|provider| {
            let handle = provider
                .upload_logical(&runmat_accelerate_api::HostLogicalTensorView {
                    data: &[0, 1, 0, 1],
                    shape: &[2, 2],
                })
                .expect("upload logical input");
            let input = gpu_helpers::logical_gpu_value(handle).expect("adopt logical producer");

            let eval = evaluate(input, &[]).expect("find logical fallback");
            let Value::GpuTensor(values_handle) = eval.values_value().expect("selected values")
            else {
                panic!("expected resident logical selected values");
            };
            assert!(runmat_accelerate_api::handle_is_logical(&values_handle));
            let values =
                test_support::gather(Value::GpuTensor(values_handle)).expect("gather logical");
            assert_eq!(values.shape(), [2, 1]);
            assert_eq!(values.materialize_f64(), vec![1.0, 1.0]);
        });
    }

    #[test]
    fn find_routes_native_and_fallback_outputs_to_the_input_owner() {
        let _lock = test_support::accel_test_lock();
        let owner: &'static dyn runmat_accelerate_api::AccelProvider = Box::leak(Box::new(
            runmat_accelerate::simple_provider::InProcessProvider::new(),
        ));
        let active: &'static dyn runmat_accelerate_api::AccelProvider = Box::leak(Box::new(
            runmat_accelerate::simple_provider::InProcessProvider::new(),
        ));
        unsafe {
            runmat_accelerate_api::register_provider(owner);
            runmat_accelerate_api::register_provider(active);
        }
        let _active = runmat_accelerate_api::ThreadProviderGuard::set(Some(active));
        assert_ne!(owner.device_id(), active.device_id());

        let native_input = owner
            .upload(&HostTensorView {
                data: &[0.0, 4.0, 0.0, 7.0],
                shape: &[2, 2],
            })
            .expect("upload native input");
        let native = evaluate(Value::GpuTensor(native_input), &[]).expect("native find");
        let Value::GpuTensor(native_indices) = native.linear_value().expect("native indices")
        else {
            panic!("expected native resident indices");
        };
        assert_eq!(native_indices.device_id(), owner.device_id());

        let fallback_input = owner
            .upload(&HostTensorView {
                data: &[0.0, 4.0, 0.0, 7.0],
                shape: &[2, 2],
            })
            .expect("upload fallback input");
        let fallback = evaluate(Value::GpuTensor(fallback_input), &[]).expect("fallback find");
        let Value::GpuTensor(fallback_indices) = fallback.linear_value().expect("fallback indices")
        else {
            panic!("expected fallback resident indices");
        };
        assert_eq!(fallback_indices.device_id(), owner.device_id());
        assert_eq!(
            test_support::gather(Value::GpuTensor(fallback_indices))
                .expect("gather fallback indices")
                .materialize_f64(),
            vec![2.0, 4.0]
        );

        let integer_input = owner
            .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
                data: runmat_accelerate_api::HostIntegerDataView::U64(&[0, 9_007_199_254_740_993]),
                shape: &[1, 2],
            })
            .expect("upload integer input");
        let integer = evaluate(Value::GpuTensor(integer_input), &[]).expect("integer find");
        let Value::GpuTensor(integer_values) = integer.values_value().expect("integer values")
        else {
            panic!("expected resident integer values");
        };
        assert_eq!(integer_values.device_id(), owner.device_id());
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&integer_values),
            Some(runmat_accelerate_api::IntegerElementType::U64)
        );
        assert_eq!(
            test_support::gather(Value::GpuTensor(integer_values))
                .expect("gather integer values")
                .integer_storage(),
            Some(&IntegerStorage::U64(vec![9_007_199_254_740_993]))
        );

        let logical_input = owner
            .upload_logical(&runmat_accelerate_api::HostLogicalTensorView {
                data: &[0, 1],
                shape: &[1, 2],
            })
            .expect("upload logical input");
        let logical = evaluate(
            gpu_helpers::logical_gpu_value(logical_input).expect("adopt logical producer"),
            &[],
        )
        .expect("logical find");
        let Value::GpuTensor(logical_values) = logical.values_value().expect("logical values")
        else {
            panic!("expected resident logical values");
        };
        assert_eq!(logical_values.device_id(), owner.device_id());
        assert!(runmat_accelerate_api::handle_is_logical(&logical_values));
    }

    #[test]
    fn admitted_provider_failure_never_replays_find_on_the_host() {
        let _lock = test_support::accel_test_lock();
        let provider = Box::leak(Box::new(FailingFindProvider {
            inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
            foreign: TestAllocationOwner::new(),
            fault: FindFault::Execution,
            calls: AtomicUsize::new(0),
            downloads: AtomicUsize::new(0),
            output_leases: std::sync::Mutex::new(Vec::new()),
            foreign_probes: std::sync::Mutex::new(Vec::new()),
            restore_fault: None,
            restore_calls: AtomicUsize::new(0),
            restore_leases: std::sync::Mutex::new(Vec::new()),
            restore_source: std::sync::Mutex::new(None),
        }));
        unsafe { runmat_accelerate_api::register_provider(provider) };
        let _active = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let input = provider
            .upload(&HostTensorView {
                data: &[0.0, 4.0, 0.0, 7.0],
                shape: &[2, 2],
            })
            .expect("resident input");
        let error = match evaluate(Value::GpuTensor(input), &[]) {
            Ok(_) => panic!("native failure must propagate"),
            Err(error) => error,
        };
        assert_eq!(error.identifier(), FIND_ERROR_PROVIDER.identifier);
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn malformed_admitted_find_outputs_are_rejected_without_replay() {
        let _lock = test_support::accel_test_lock();
        for fault in [
            FindFault::MissingValues,
            FindFault::AliasedInput,
            FindFault::WrongShape,
            FindFault::WrongClass,
            FindFault::ForeignOwner,
            FindFault::CrossOutputAlias,
            FindFault::ExpiredOutput,
        ] {
            let provider = Box::leak(Box::new(FailingFindProvider {
                inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
                foreign: TestAllocationOwner::new(),
                fault,
                calls: AtomicUsize::new(0),
                downloads: AtomicUsize::new(0),
                output_leases: std::sync::Mutex::new(Vec::new()),
                foreign_probes: std::sync::Mutex::new(Vec::new()),
                restore_fault: None,
                restore_calls: AtomicUsize::new(0),
                restore_leases: std::sync::Mutex::new(Vec::new()),
                restore_source: std::sync::Mutex::new(None),
            }));
            unsafe { runmat_accelerate_api::register_provider(provider) };
            let _active = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
            let input = provider
                .upload(&HostTensorView {
                    data: &[0.0, 4.0, 0.0, 7.0],
                    shape: &[2, 2],
                })
                .expect("resident input");
            let error = match evaluate(Value::GpuTensor(input.clone()), &[]) {
                Ok(_) => panic!("{fault:?}: malformed output must fail"),
                Err(error) => error,
            };
            assert_eq!(error.identifier(), FIND_ERROR_PROVIDER_OUTPUT.identifier);
            assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
            assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
            assert!(
                provider
                    .output_leases
                    .lock()
                    .unwrap()
                    .iter()
                    .all(|lease| lease.acquire().is_none()),
                "{fault:?}: provider outputs not retired"
            );
            for probe in provider.foreign_probes.lock().unwrap().iter() {
                assert_eq!(
                    probe.retirements(),
                    1,
                    "{fault:?}: foreign output not retired"
                );
            }
            assert!(input.allocation_lease().validate_live().is_ok());
            input.release().expect("borrowed input remains releasable");
        }
    }

    #[test]
    fn find_restoration_faults_preserve_source_and_never_mask_admitted_failures() {
        let _lock = test_support::accel_test_lock();
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        for fault in [
            RestoreFault::Success,
            RestoreFault::SourceAlias,
            RestoreFault::Unproven,
            RestoreFault::AfterEffect,
            RestoreFault::Unsupported,
            RestoreFault::SecondFailure,
        ] {
            let provider = Box::leak(Box::new(FailingFindProvider {
                inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
                foreign: TestAllocationOwner::new(),
                fault: FindFault::Execution,
                calls: AtomicUsize::new(0),
                downloads: AtomicUsize::new(0),
                output_leases: std::sync::Mutex::new(Vec::new()),
                foreign_probes: std::sync::Mutex::new(Vec::new()),
                restore_fault: Some(fault),
                restore_calls: AtomicUsize::new(0),
                restore_leases: std::sync::Mutex::new(Vec::new()),
                restore_source: std::sync::Mutex::new(None),
            }));
            unsafe { runmat_accelerate_api::register_provider(provider) };
            let _active = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
            let source = provider
                .upload(&HostTensorView {
                    data: &[0.0, 4.0, 0.0, 7.0],
                    shape: &[2, 2],
                })
                .expect("resident input")
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
                .expect("known automatic intent");
            *provider.restore_source.lock().unwrap() = Some(source.clone());
            let eval = evaluate(Value::GpuTensor(source.clone()), &[Value::from("last")])
                .expect("host find fallback");
            assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
            assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
            let result = if matches!(fault, RestoreFault::SecondFailure) {
                eval.output_values(3).map(|_| ())
            } else {
                eval.linear_value().map(|value| match fault {
                    RestoreFault::Success => assert!(matches!(value, Value::GpuTensor(_))),
                    RestoreFault::Unsupported => assert!(matches!(value, Value::Num(_))),
                    _ => panic!("{fault:?}: restoration must fail"),
                })
            };
            if matches!(
                fault,
                RestoreFault::SourceAlias
                    | RestoreFault::Unproven
                    | RestoreFault::AfterEffect
                    | RestoreFault::SecondFailure
            ) {
                assert!(result.is_err(), "{fault:?}: failure was masked");
            } else {
                result.expect("successful or certified unsupported restoration");
            }
            let expected_uploads = if matches!(fault, RestoreFault::SecondFailure) {
                2
            } else {
                1
            };
            assert_eq!(
                provider.restore_calls.load(Ordering::SeqCst),
                expected_uploads
            );
            assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
            if matches!(fault, RestoreFault::SecondFailure) {
                assert!(
                    provider
                        .restore_leases
                        .lock()
                        .unwrap()
                        .iter()
                        .all(|lease| lease.acquire().is_none()),
                    "first unpublished output survived"
                );
            }
            for probe in provider.foreign_probes.lock().unwrap().iter() {
                assert_eq!(probe.retirements(), 1, "post-effect owner was not retired");
            }
            assert!(source.allocation_lease().validate_live().is_ok());
            drop(eval);
            provider.restore_source.lock().unwrap().take();
            source.release().expect("borrowed input remains releasable");
        }
    }

    #[test]
    fn admitted_provider_zero_scalar_uses_host_empty_matrix_convention() {
        let _lock = test_support::accel_test_lock();
        let provider = Box::leak(Box::new(
            runmat_accelerate::simple_provider::InProcessProvider::new(),
        ));
        unsafe { runmat_accelerate_api::register_provider(provider) };
        let _active = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let input = provider
            .upload(&HostTensorView {
                data: &[0.0],
                shape: &[1, 1],
            })
            .expect("resident scalar");
        let result = evaluate(Value::GpuTensor(input), &[]).expect("resident find");
        let Value::GpuTensor(indices) = result.linear_value().expect("linear output") else {
            panic!("admitted provider must retain a resident output");
        };
        assert_eq!(indices.shape(), [0, 0]);
        assert!(runmat_accelerate_api::same_provider_instance(
            provider,
            runmat_accelerate_api::provider_for_handle(&indices).expect("output owner")
        ));
    }

    #[test]
    fn inferred_find_shapes_admit_runtime_scalar_and_orientation_cases() {
        use runmat_types::{
            CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
            OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact,
            ValueKindFact,
        };

        let entry = runmat_builtins::builtin_catalog_entry_by_name("find")
            .expect("canonical find catalog entry");
        let numeric = |shape| {
            ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                }),
                shape,
                StorageFact::Dense,
            )
        };
        let cases = [
            (vec![1, 1], vec![0.0], Some(LiteralValue::Number(0.0))),
            (vec![1, 1], vec![7.0], Some(LiteralValue::Number(7.0))),
            (vec![1, 3], vec![0.0, 2.0, 0.0], None),
            (vec![3, 1], vec![0.0, 2.0, 0.0], None),
        ];
        for (shape, data, literal) in cases {
            let runtime = compute_find(
                &DataStorage::Real(Tensor::new(data, shape.clone()).expect("input")),
                &FindOptions::default(),
            )
            .linear_tensor()
            .expect("linear runtime result");
            for requested in [
                RequestedOutputCount::One,
                RequestedOutputCount::CurrentFunctionNargout,
            ] {
                let inferred = runmat_builtins::infer_catalog_call(
                    entry,
                    &CallRequest {
                        arguments: vec![numeric(ShapeFact::from(
                            shape.iter().copied().map(Some).collect::<Vec<_>>(),
                        ))],
                        literals: LiteralContext::new(literal.clone().into_iter().collect()),
                        outputs: OutputSelection::new(requested),
                    },
                );
                let dims = inferred.outputs[0]
                    .shape
                    .known_dims()
                    .expect("ranked result");
                assert_eq!(dims.len(), runtime.shape().len(), "input {shape:?}");
                assert!(
                    dims.iter()
                        .zip(runtime.shape())
                        .all(|(known, actual)| known.is_none_or(|known| known == *actual)),
                    "input {shape:?}, output mode {requested:?}, inferred {dims:?}, runtime {:?}",
                    runtime.shape()
                );
            }
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn find_gpu_row_vector_preserves_linear_index_orientation() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 4.0, 5.0, 7.0], vec![1, 4]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let result = find_builtin(Value::GpuTensor(handle), Vec::new()).expect("find");
            assert!(matches!(result, Value::GpuTensor(_)));
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape(), [1, 3]);
            assert_eq!(gathered.materialize_f64(), vec![2.0, 3.0, 4.0]);
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
        assert_eq!(rows.shape(), [3, 1]);
        assert_eq!(rows.materialize_f64(), vec![2.0, 1.0, 2.0]);

        let cols = test_support::gather(eval.column_value().expect("cols")).expect("gather cols");
        assert_eq!(cols.shape(), [3, 1]);
        assert_eq!(cols.materialize_f64(), vec![1.0, 2.0, 3.0]);

        let vals = test_support::gather(eval.values_value().expect("vals")).expect("gather vals");
        assert_eq!(vals.shape(), [3, 1]);
        assert_eq!(vals.materialize_f64(), vec![2.0, 3.0, 6.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn find_values_preserve_exact_uint64_storage() {
        let input = Tensor::new_integer(
            IntegerStorage::U64(vec![0, u64::MAX, 1_u64 << 63, 0]),
            vec![2, 2],
        )
        .expect("integer tensor");
        let eval = evaluate(Value::Tensor(input), &[]).expect("evaluate");
        let values = eval.values_value().expect("values");
        let Value::Tensor(values) = values else {
            panic!("expected typed tensor values");
        };
        assert_eq!(values.shape(), [2, 1]);
        assert_eq!(
            values.integer_storage(),
            Some(&IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63]))
        );
    }

    #[test]
    fn find_indices_read_typed_integer_storage_exactly() {
        let input = Tensor::new_integer(IntegerStorage::I16(vec![0, -7, 0, 9]), vec![2, 2])
            .expect("integer tensor");

        let value = find_builtin(Value::Tensor(input), Vec::new()).expect("find");

        match value {
            Value::Tensor(indices) => {
                assert_eq!(indices.shape(), [2, 1]);
                assert_eq!(indices.materialize_f64(), vec![2.0, 4.0]);
            }
            other => panic!("expected index tensor, got {other:?}"),
        }
    }

    #[test]
    fn find_last_indices_read_typed_integer_storage_exactly() {
        let input = Tensor::new_integer(IntegerStorage::U16(vec![5, 0, 3, 0]), vec![2, 2])
            .expect("integer tensor");

        let value = find_builtin(
            Value::Tensor(input),
            vec![Value::Int(IntValue::I32(1)), Value::from("last")],
        )
        .expect("find");

        assert_eq!(value, Value::Num(3.0));
    }

    #[test]
    fn find_reads_mirrorless_typed_complex_integer_storage() {
        let storage = IntegerComplexStorage::new(
            IntegerStorage::I16(vec![0, -7, 0, 9]),
            IntegerStorage::I16(vec![0, 0, 5, 0]),
        )
        .expect("complex integer storage");
        let input = ComplexTensor::new_integer(storage, vec![2, 2]).expect("complex tensor");

        let eval = evaluate(Value::ComplexTensor(input), &[]).expect("find");
        let linear = tensor::value_into_tensor_for("find", eval.linear_value().expect("linear"))
            .expect("linear tensor");
        assert_eq!(linear.materialize_f64(), vec![2.0, 3.0, 4.0]);
        let values = eval.values_value().expect("values");
        let Value::ComplexTensor(values) = values else {
            panic!("expected typed complex tensor values");
        };
        let storage = values.integer_storage().expect("typed complex values");
        assert_eq!(storage.real(), &IntegerStorage::I16(vec![-7, 0, 9]));
        assert_eq!(storage.imag(), &IntegerStorage::I16(vec![0, 5, 0]));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn find_sparse_values_preserve_exact_storage_and_traversal_order() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let sparse = runmat_value::SparseTensor::new_integer(
            3,
            2,
            vec![0, 2, 3],
            vec![0, 2, 1],
            IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63, 7]),
        )
        .expect("typed sparse");

        let all = evaluate(Value::SparseTensor(sparse.clone()), &[]).expect("find sparse");
        let Value::Tensor(all_values) = all.values_value().expect("all values") else {
            panic!("expected typed sparse find values");
        };
        assert_eq!(
            all_values.integer_storage(),
            Some(&IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63, 7]))
        );

        let first = evaluate(
            Value::SparseTensor(sparse.clone()),
            &[Value::Int(IntValue::I32(2))],
        )
        .expect("find sparse first");
        let Value::Tensor(first_values) = first.values_value().expect("first values") else {
            panic!("expected typed sparse first values");
        };
        assert_eq!(
            first_values.integer_storage(),
            Some(&IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63]))
        );

        let last = evaluate(
            Value::SparseTensor(sparse),
            &[Value::Int(IntValue::I32(2)), Value::from("last")],
        )
        .expect("find sparse last");
        let Value::Tensor(last_values) = last.values_value().expect("last values") else {
            panic!("expected typed sparse last values");
        };
        assert_eq!(
            last_values.integer_storage(),
            Some(&IntegerStorage::U64(vec![1_u64 << 63, 7]))
        );
    }

    #[test]
    fn find_sparse_values_preserve_native_single_class_and_order() {
        let sparse = runmat_value::SparseTensor::new_f32(
            3,
            2,
            vec![0, 2, 3],
            vec![0, 2, 1],
            vec![1.25, 3.5, 7.75],
        )
        .expect("single sparse");
        let eval = evaluate(Value::SparseTensor(sparse), &[]).expect("find sparse");
        let Value::Tensor(values) = eval.values_value().expect("values") else {
            panic!("expected native-single find values");
        };
        assert_eq!(values.numeric_dtype(), runmat_value::NumericDType::F32);
        assert_eq!(values.as_f32_slice(), Some(&[1.25, 3.5, 7.75][..]));
    }

    #[test]
    fn find_sparse_values_preserve_logical_class_and_order() {
        let sparse = runmat_value::SparseTensor::new_logical(3, 2, vec![0, 2, 3], vec![0, 2, 1])
            .expect("logical sparse");
        let eval = evaluate(Value::SparseTensor(sparse), &[]).expect("find sparse");
        let Value::LogicalArray(values) = eval.values_value().expect("values") else {
            panic!("expected logical sparse find values");
        };
        assert_eq!(values.shape, vec![3, 1]);
        assert_eq!(values.data, vec![1, 1, 1]);
    }

    #[test]
    fn find_sparse_complex_uses_both_components_and_preserves_values() {
        let sparse = runmat_value::SparseTensor::new_complex(
            3,
            2,
            vec![0, 3, 4],
            vec![0, 1, 2, 1],
            vec![(0.0, 2.0), (0.0, 0.0), (3.0, 0.0), (-4.0, 5.0)],
        )
        .expect("complex sparse");
        let eval = evaluate(Value::SparseTensor(sparse), &[]).expect("find sparse");
        let linear = tensor::value_into_tensor_for("find", eval.linear_value().expect("linear"))
            .expect("linear tensor");
        assert_eq!(linear.materialize_f64(), vec![1.0, 3.0, 5.0]);
        let Value::ComplexTensor(values) = eval.values_value().expect("values") else {
            panic!("expected complex sparse find values");
        };
        assert_eq!(
            values.materialize_f64(),
            vec![(0.0, 2.0), (3.0, 0.0), (-4.0, 5.0)]
        );
    }

    #[test]
    fn find_sparse_complex_single_preserves_component_class() {
        let sparse = runmat_value::SparseTensor::new_complex_f32(
            2,
            1,
            vec![0, 2],
            vec![0, 1],
            vec![(0.0, 2.0), (3.0, -4.0)],
        )
        .expect("complex single sparse");
        let eval = evaluate(Value::SparseTensor(sparse), &[]).expect("find sparse");
        let Value::ComplexTensor(values) = eval.values_value().expect("values") else {
            panic!("expected complex sparse find values");
        };
        assert_eq!(values.numeric_dtype(), runmat_value::NumericDType::F32);
        assert_eq!(
            values.as_f32_slice(),
            Some(
                [
                    runmat_value::ComplexElement(0.0, 2.0),
                    runmat_value::ComplexElement(3.0, -4.0),
                ]
                .as_slice()
            )
        );
    }

    #[test]
    fn find_integer_selection_preserves_every_integer_class() {
        let cases = [
            (
                IntegerStorage::I8(vec![-2, 0, 3]),
                IntegerStorage::I8(vec![3, -2]),
            ),
            (
                IntegerStorage::I16(vec![-2, 0, 3]),
                IntegerStorage::I16(vec![3, -2]),
            ),
            (
                IntegerStorage::I32(vec![-2, 0, 3]),
                IntegerStorage::I32(vec![3, -2]),
            ),
            (
                IntegerStorage::I64(vec![-2, 0, 3]),
                IntegerStorage::I64(vec![3, -2]),
            ),
            (
                IntegerStorage::U8(vec![2, 0, 3]),
                IntegerStorage::U8(vec![3, 2]),
            ),
            (
                IntegerStorage::U16(vec![2, 0, 3]),
                IntegerStorage::U16(vec![3, 2]),
            ),
            (
                IntegerStorage::U32(vec![2, 0, 3]),
                IntegerStorage::U32(vec![3, 2]),
            ),
            (
                IntegerStorage::U64(vec![2, 0, 3]),
                IntegerStorage::U64(vec![3, 2]),
            ),
        ];
        for (storage, expected) in cases {
            assert_eq!(select_integer_values(&storage, &[2, 0]), expected);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn find_single_integer_value_preserves_scalar_class() {
        let input = Tensor::new_integer(IntegerStorage::I64(vec![0, i64::MIN]), vec![2, 1])
            .expect("integer tensor");
        let eval = evaluate(Value::Tensor(input), &[]).expect("evaluate");
        assert_eq!(
            eval.values_value().expect("values"),
            Value::Int(IntValue::I64(i64::MIN))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn find_integer_scalar_preserves_exact_value_output() {
        let eval = evaluate(Value::Int(IntValue::U64(u64::MAX)), &[]).expect("evaluate");
        assert_eq!(
            eval.values_value().expect("values"),
            Value::Int(IntValue::U64(u64::MAX))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn find_last_returns_selected_indices_in_ascending_order() {
        let tensor = Tensor::new(vec![1.0, 0.0, 2.0, 3.0, 0.0], vec![1, 5]).unwrap();
        let result = find_builtin(
            Value::Tensor(tensor),
            vec![Value::Int(IntValue::I32(2)), Value::from("last")],
        )
        .expect("find");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), [1, 2]);
                assert_eq!(t.materialize_f64(), vec![3.0, 4.0]);
            }
            Value::Num(_) => panic!("expected column vector"),
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn find_limit_zero_rejects() {
        let tensor = Tensor::new(vec![1.0, 0.0, 3.0], vec![3, 1]).unwrap();
        find_builtin(Value::Tensor(tensor), vec![Value::Num(0.0)])
            .expect_err("zero is not a positive count");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn find_empty_orientation_follows_input_vector_shape() {
        for (shape, expected_shape) in [
            (vec![1, 0], vec![1, 0]),
            (vec![0, 1], vec![0, 1]),
            (vec![0, 3], vec![0, 1]),
        ] {
            let input =
                Tensor::new_integer(IntegerStorage::U32(Vec::new()), shape).expect("empty input");
            let Value::Tensor(indices) =
                find_builtin(Value::Tensor(input), Vec::new()).expect("find")
            else {
                panic!("expected empty tensor");
            };
            assert_eq!(indices.shape(), expected_shape);
            assert!(indices.materialize_f64().is_empty());
        }
    }

    #[test]
    fn find_scalar_zero_and_empty_matrix_use_empty_matrix_convention() {
        for input in [
            Value::Num(0.0),
            Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
        ] {
            let Value::Tensor(indices) = find_builtin(input, Vec::new()).expect("find") else {
                panic!("expected empty tensor");
            };
            assert_eq!(indices.shape(), [0, 0]);
        }
    }

    #[test]
    fn find_dense_logical_value_output_preserves_logical_class() {
        let input = LogicalArray::new(vec![0, 1, 1, 0], vec![2, 2]).unwrap();
        let eval = evaluate(Value::LogicalArray(input), &[]).expect("find");
        let Value::LogicalArray(values) = eval.values_value().expect("values") else {
            panic!("expected logical values");
        };
        assert_eq!(values.shape, vec![2, 1]);
        assert_eq!(values.data, vec![1, 1]);
    }

    #[test]
    fn find_runmat_only_forms_gate_before_evaluation() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let input = Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap());
        let err = evaluate(input, &[Value::from("last")])
            .err()
            .expect("direction-only form must gate");
        assert_eq!(
            err.identifier(),
            FIND_DIRECTION_ONLY_EXTENSION.error_identifier
        );

        let sparse = runmat_value::SparseTensor::new_integer(
            1,
            1,
            vec![0, 1],
            vec![0],
            IntegerStorage::U64(vec![u64::MAX]),
        )
        .unwrap();
        let err = evaluate(Value::SparseTensor(sparse), &[])
            .err()
            .expect("integer sparse form must gate");
        assert_eq!(
            err.identifier(),
            FIND_INTEGER_SPARSE_EXTENSION.error_identifier
        );
    }

    #[test]
    fn find_integer_metadata_covers_values_counts_and_sparse_extension() {
        assert_eq!(FIND_INTEGER_CAPABILITIES.len(), 4);
        assert_eq!(FIND_EXTENSIONS.len(), 2);
        for capability in FIND_INTEGER_CAPABILITIES {
            for input in capability.inputs {
                assert_eq!(
                    input.classes,
                    &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES
                );
            }
        }
        if let Some(largest_exact_index) = 1_usize.checked_shl(53) {
            assert_eq!(
                exact_index_as_f64(largest_exact_index).expect("largest exact index"),
                9_007_199_254_740_992.0
            );
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn find_integer_gpu_last_preserves_order_orientation_class_and_residency() {
        test_support::with_f32_test_provider(|provider| {
            let handle = provider
                .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
                    data: runmat_accelerate_api::HostIntegerDataView::U64(&[
                        0,
                        1_u64 << 63,
                        7,
                        u64::MAX,
                    ]),
                    shape: &[1, 4],
                })
                .expect("upload integer row vector")
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
                .expect("known automatic intent permits host double indices");
            let eval = evaluate(
                Value::GpuTensor(handle),
                &[Value::Int(IntValue::I32(2)), Value::from("last")],
            )
            .expect("find last");

            let linear = eval.linear_value().expect("linear indices");
            let Value::Tensor(linear) = linear else {
                panic!("double indices must fall back to host storage");
            };
            assert_eq!(linear.shape(), [1, 2]);
            assert_eq!(linear.materialize_f64(), vec![3.0, 4.0]);

            let values = eval.values_value().expect("selected values");
            let Value::GpuTensor(values_handle) = &values else {
                panic!("expected resident selected values, got {values:?}");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(values_handle),
                Some(runmat_accelerate_api::IntegerElementType::U64)
            );
            let values = test_support::gather(values).expect("gather selected values");
            assert_eq!(values.shape(), [2, 1]);
            assert_eq!(
                values.integer_storage(),
                Some(&IntegerStorage::U64(vec![7, u64::MAX]))
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn find_char_array_supports_nonzero_codes() {
        let chars = CharArray::from_unicode_scalars(vec!['\0', 'A', '\0'], 1, 3).unwrap();
        let result = find_builtin(Value::CharArray(chars), Vec::new()).expect("find");
        match result {
            Value::Num(n) => assert_eq!(n, 2.0),
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![2.0]),
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn find_gpu_multi_outputs_return_gpu_handles() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 4.0, 5.0, 0.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let eval = evaluate(Value::GpuTensor(handle), &[]).expect("evaluate");

            let rows = eval.row_value().expect("rows");
            assert!(matches!(rows, Value::GpuTensor(_)));
            let rows_host = test_support::gather(rows).expect("gather rows");
            assert_eq!(rows_host.materialize_f64(), vec![2.0, 1.0]);

            let cols = eval.column_value().expect("cols");
            assert!(matches!(cols, Value::GpuTensor(_)));
            let cols_host = test_support::gather(cols).expect("gather cols");
            assert_eq!(cols_host.materialize_f64(), vec![1.0, 2.0]);

            let vals = eval.values_value().expect("vals");
            assert!(matches!(vals, Value::GpuTensor(_)));
            let vals_host = test_support::gather(vals).expect("gather vals");
            assert_eq!(vals_host.materialize_f64(), vec![4.0, 5.0]);
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
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_eval = evaluate(Value::GpuTensor(handle), &[]).expect("gpu evaluate");
        let gpu_linear =
            test_support::gather(gpu_eval.linear_value().expect("gpu linear")).expect("gpu gather");
        assert_eq!(gpu_linear.materialize_f64(), cpu_linear.materialize_f64());
    }
}
