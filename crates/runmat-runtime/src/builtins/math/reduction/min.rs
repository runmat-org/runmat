//! MATLAB-compatible `min` builtin with GPU-aware semantics for RunMat.

use std::cmp::Ordering;
use std::collections::BTreeSet;

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinErrorDescriptor, MinComparisonMethod as ComparisonMethod, MinControl,
    MIN_ERROR_INTERNAL, MIN_ERROR_INVALID_ARGUMENT, MIN_ERROR_INVALID_INPUT,
    MIN_ERROR_SIZE_MISMATCH,
};
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexStorage, ComplexTensor, NumericDType, NumericStorage, Tensor, Value};

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "min";

use crate::builtins::common::broadcast::BroadcastPlan;
use crate::builtins::common::random_args::{complex_tensor_into_value, keyword_of};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType,
    ShapeRequirements,
};
use crate::builtins::common::{
    shape::{is_scalar_shape, normalize_scalar_shape},
    tensor,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::reduction::min")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "min",
    op_kind: GpuOpKind::Reduction,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        ProviderHook::Reduction {
            name: "reduce_min_dim",
        },
        ProviderHook::Reduction {
            name: "reduce_min",
        },
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: Some(256),
    workgroup_size: Some(256),
    accepts_nan_mode: false,
    notes:
        "Providers should implement reduce_min_dim / reduce_min. Requests that require omitnan, comparisonmethod overrides, or complex inputs fall back to the host implementation.",
};

fn min_descriptor_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn min_descriptor_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    min_descriptor_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn min_invalid_argument(detail: impl AsRef<str>) -> RuntimeError {
    min_descriptor_error_with_detail(&MIN_ERROR_INVALID_ARGUMENT, detail)
}

fn min_invalid_input(detail: impl AsRef<str>) -> RuntimeError {
    min_descriptor_error_with_detail(&MIN_ERROR_INVALID_INPUT, detail)
}

fn min_size_mismatch(detail: impl AsRef<str>) -> RuntimeError {
    min_descriptor_error_with_detail(&MIN_ERROR_SIZE_MISMATCH, detail)
}

fn min_internal_error(detail: impl AsRef<str>) -> RuntimeError {
    min_descriptor_error_with_detail(&MIN_ERROR_INTERNAL, detail)
}

#[path = "min/fusion_contract.rs"]
mod fusion_contract;

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::reduction::min", epilogue = crate::builtins::common::spec::FusionEpilogueOperation::Minimum, reduction = fusion_contract::DECLARATION)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "min",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: crate::builtins::common::spec::FusionExprBuilder::new(|ctx, writer| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            writer.write_formatted(format_args!("accumulator = min(accumulator, {input});"))
        }),
    }),
    emits_nan: true,
    notes: "Fusion planner emits canonical reduction kernels; providers may substitute custom WGSL via reduce_min_dim hooks.",
};

/// Evaluation artifact returned by `min` that carries both values and indices.
#[derive(Debug, Clone)]
pub struct MinEvaluation {
    values: Value,
    indices: Value,
}

impl MinEvaluation {
    /// Consume the evaluation and return only the minimum values (single-output call).
    pub fn into_value(self) -> Value {
        self.values
    }

    /// Consume the evaluation and return both minima and indices.
    pub fn into_pair(self) -> (Value, Value) {
        (self.values, self.indices)
    }

    /// Peek at the indices without consuming.
    pub fn indices_value(&self) -> Value {
        self.indices.clone()
    }
}

#[runtime_builtin(
    name = "min",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::reduction::min"
)]
pub(crate) async fn min_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if let Some(eval) = crate::builtins::table::categorical_min_evaluate(&value, rest.view()).await
    {
        return crate::builtins::table::categorical_extrema_to_value(eval?);
    }
    if crate::output_count::current_output_count().unwrap_or(1) > 1
        && rest
            .first()
            .is_some_and(|value| !is_empty_placeholder(value))
    {
        return Err(min_invalid_argument(
            "min: pairwise element-wise form has exactly one output",
        ));
    }
    let demand = if crate::output_count::current_output_count().unwrap_or(1) > 1 {
        super::extrema_provider::ExtremaDemand::ValuesAndIndices
    } else {
        super::extrema_provider::ExtremaDemand::ValuesOnly
    };
    let eval = evaluate_for_demand(runtime.as_ref(), value, rest.view(), demand).await?;
    min_output_from_eval(eval)
}

/// Apply the owning reduction grammar with an alias-imposed missing-data policy.
/// The alias can select reduction without constructing a placeholder argument.
pub(crate) async fn min_reduction_with_nan_policy(
    value: Value,
    rest: crate::call::BuiltinArguments,
    imposed_nan_mode: ReductionNaN,
) -> BuiltinResult<Value> {
    if crate::builtins::common::validation::is_typed_complex_integer(&value) {
        return Err(min_invalid_input(
            "operations involving complex numbers with integer types are not supported",
        ));
    }
    let controls = if rest.first().is_some_and(is_empty_placeholder) {
        rest.view().range(1..)?
    } else {
        rest.view()
    };
    let mut parsed = ReductionArgs::default();
    parse_reduction_options(&mut parsed, controls).await?;
    parsed.nan_mode = imposed_nan_mode;
    let demand = if crate::output_count::current_output_count().unwrap_or(1) > 1 {
        super::extrema_provider::ExtremaDemand::ValuesAndIndices
    } else {
        super::extrema_provider::ExtremaDemand::ValuesOnly
    };
    let eval = reduction_min(None, value, parsed, demand).await?;
    min_output_from_eval(eval)
}

fn min_output_from_eval(eval: super::extrema_provider::ExtremaResult) -> BuiltinResult<Value> {
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::output_list(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::output_list(vec![eval.into_value()]));
        }
        let (values, indices) = eval.into_pair(super::extrema_provider::ExtremaKind::Min)?;
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![values, indices],
        ));
    }
    Ok(eval.into_value())
}

/// Evaluate the builtin once and expose both outputs (value + indices).
pub async fn evaluate(
    value: Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<MinEvaluation> {
    let evaluated = evaluate_for_demand(
        None,
        value,
        rest,
        super::extrema_provider::ExtremaDemand::ValuesAndIndices,
    )
    .await?;
    let (values, indices) = evaluated.into_pair(super::extrema_provider::ExtremaKind::Min)?;
    Ok(MinEvaluation { values, indices })
}

pub(super) async fn evaluate_for_demand(
    runtime: Option<&crate::context::RuntimeContext>,
    value: Value,
    rest: crate::call::FunctionArgumentView<'_>,
    demand: super::extrema_provider::ExtremaDemand,
) -> BuiltinResult<super::extrema_provider::ExtremaResult> {
    if crate::builtins::common::validation::is_typed_complex_integer(&value) {
        return Err(min_invalid_input(
            "operations involving complex numbers with integer types are not supported",
        ));
    }
    match parse_call(rest).await? {
        ParsedCall::Elementwise(args) => elementwise_min(value, args, demand).await,
        ParsedCall::Reduction(args) => reduction_min(runtime, value, args, demand).await,
    }
}

/// Validate a reduction's controls without materializing its data operand.
/// L07 will move this parser together with the extrema engine into its domain
/// service; bounds must not duplicate option or exact-dimension authority.
pub(super) async fn validate_reduction_arguments(
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    match parse_call(rest).await? {
        ParsedCall::Reduction(_) => Ok(()),
        ParsedCall::Elementwise(_) => Err(min_invalid_argument("expected reduction controls")),
    }
}

#[derive(Debug, Clone)]
enum ParsedCall {
    Reduction(ReductionArgs),
    Elementwise(ElementwiseArgs),
}

#[derive(Debug, Clone)]
struct ReductionArgs {
    selection: DimSelection,
    nan_mode: ReductionNaN,
    comparison: ComparisonMethod,
    linear_index: bool,
}

impl Default for ReductionArgs {
    fn default() -> Self {
        Self {
            selection: DimSelection::Auto,
            nan_mode: ReductionNaN::Include,
            comparison: ComparisonMethod::Auto,
            linear_index: false,
        }
    }
}

#[derive(Debug, Clone)]
enum DimSelection {
    Auto,
    Dim(usize),
    Vec(Vec<usize>),
    All,
}

#[derive(Debug, Clone)]
struct ElementwiseArgs {
    other: Value,
    comparison: ComparisonMethod,
}

async fn parse_call(rest: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<ParsedCall> {
    if rest.is_empty() {
        return Ok(ParsedCall::Reduction(ReductionArgs::default()));
    }

    let first = &rest[0];
    if !is_empty_placeholder(first) {
        let comparison = parse_elementwise_options(rest.range(1..)?)?;
        return Ok(ParsedCall::Elementwise(ElementwiseArgs {
            other: first.clone(),
            comparison,
        }));
    }

    let mut args = ReductionArgs::default();
    parse_reduction_options(&mut args, rest.range(1..)?).await?;
    Ok(ParsedCall::Reduction(args))
}

fn is_empty_placeholder(value: &Value) -> bool {
    match value {
        Value::Tensor(t) => tensor_len(t) == 0,
        Value::LogicalArray(l) => l.data.is_empty(),
        Value::StringArray(sa) => sa.data().is_empty(),
        Value::CharArray(ca) => ca.data().is_empty(),
        Value::Cell(cell) => cell.data().is_empty(),
        Value::String(runmat_value::StringElement::Text(s)) => s.is_empty(),
        _ => false,
    }
}

fn tensor_len(tensor: &Tensor) -> usize {
    tensor::tensor_element_len(tensor)
}

async fn parse_reduction_options(
    args: &mut ReductionArgs,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    let mut idx = 0usize;
    let mut selection_set = !matches!(args.selection, DimSelection::Auto);
    let mut comparison_set = matches!(args.comparison, ComparisonMethod::Auto);
    while idx < rest.len() {
        if let Some(keyword) = keyword_of(&rest[idx])? {
            match MinControl::from_public_option(&keyword) {
                Some(MinControl::Missing(mode)) => {
                    args.nan_mode = mode;
                    idx += 1;
                    continue;
                }
                Some(MinControl::All) => {
                    if selection_set {
                        return Err(min_invalid_argument(
                            "min: 'all' cannot be combined with an explicit dimension",
                        ));
                    }
                    args.selection = DimSelection::All;
                    selection_set = true;
                    idx += 1;
                    continue;
                }
                Some(MinControl::Linear) => {
                    if selection_set {
                        return Err(min_invalid_argument(
                            "min: 'linear' cannot be combined with an explicit dimension",
                        ));
                    }
                    args.selection = DimSelection::All;
                    args.linear_index = true;
                    selection_set = true;
                    idx += 1;
                    continue;
                }
                Some(MinControl::ComparisonMethod) => {
                    let Some(value) = rest.get(idx + 1) else {
                        return Err(min_invalid_argument(
                            "min: expected a value after 'ComparisonMethod'",
                        ));
                    };
                    args.comparison = parse_comparison_method(value)?;
                    comparison_set = true;
                    idx += 2;
                    continue;
                }
                None => {}
            }
        }

        if !selection_set {
            if let Some(selection) = parse_dimension_value(&rest[idx]).await? {
                args.selection = selection;
                selection_set = true;
                idx += 1;
                continue;
            }
        }

        return Err(min_invalid_argument(format!(
            "min: unrecognised argument {:?}",
            rest[idx]
        )));
    }

    if !comparison_set {
        args.comparison = ComparisonMethod::Auto;
    }

    Ok(())
}

fn parse_elementwise_options(
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<ComparisonMethod> {
    let mut comparison = ComparisonMethod::Auto;
    let mut comparison_set = false;
    let mut idx = 0usize;
    while idx < rest.len() {
        if let Some(keyword) = keyword_of(&rest[idx])? {
            match MinControl::from_public_option(&keyword) {
                Some(MinControl::ComparisonMethod) => {
                    let Some(value) = rest.get(idx + 1) else {
                        return Err(min_invalid_argument(
                            "min: expected a value after 'ComparisonMethod'",
                        ));
                    };
                    comparison = parse_comparison_method(value)?;
                    comparison_set = true;
                    idx += 2;
                    continue;
                }
                Some(MinControl::Missing(_) | MinControl::All | MinControl::Linear) => {
                    return Err(min_invalid_argument(format!(
                        "min: '{}' is only supported for reduction calls",
                        keyword
                    )));
                }
                None => {}
            }
        }
        return Err(min_invalid_argument(format!(
            "min: unrecognised argument {:?}",
            rest[idx]
        )));
    }
    if !comparison_set {
        comparison = ComparisonMethod::Auto;
    }
    Ok(comparison)
}

fn parse_comparison_method(value: &Value) -> BuiltinResult<ComparisonMethod> {
    let Some(keyword) = keyword_of(value)? else {
        return Err(min_invalid_argument(
            "min: 'ComparisonMethod' expects a string value",
        ));
    };
    ComparisonMethod::from_public_option(&keyword).ok_or_else(|| {
        min_invalid_argument(format!("min: unsupported ComparisonMethod '{keyword}'"))
    })
}

async fn parse_dimension_value(value: &Value) -> BuiltinResult<Option<DimSelection>> {
    match value {
        Value::Int(_) | Value::Num(_) => tensor::scalar_dimension_from_value_typed(value, false)
            .map_err(map_scalar_dim_error)
            .map(|dim| dim.map(DimSelection::Dim)),
        Value::Tensor(t) => parse_dimension_tensor(value, t.shape()).await,
        Value::LogicalArray(logical) => parse_dimension_tensor(value, &logical.shape).await,
        Value::GpuTensor(_) => Err(min_invalid_argument(
            "min: dimension arguments must reside on the host",
        )),
        _ => Ok(None),
    }
}

async fn parse_dimension_tensor(
    value: &Value,
    shape: &[usize],
) -> BuiltinResult<Option<DimSelection>> {
    if tensor::element_count(shape) == 0 {
        return Ok(Some(DimSelection::Auto));
    }
    let is_vector = shape.len() == 1
        || shape.get(0).copied().unwrap_or(1) == 1
        || shape.get(1).copied().unwrap_or(1) == 1;
    if !is_vector {
        return Err(min_invalid_argument(
            "min: dimension vector must be a row or column vector",
        ));
    }
    let dims = tensor::dims_from_value_async_typed(value)
        .await
        .map_err(map_vector_dim_error)?;
    let Some(dims) = dims else {
        return Ok(None);
    };
    if dims.is_empty() {
        return Ok(Some(DimSelection::Auto));
    }
    let mut seen = BTreeSet::new();
    let mut uniq = Vec::with_capacity(dims.len());
    for dim in dims {
        if dim < 1 {
            return Err(min_invalid_argument("min: dimension indices must be >= 1"));
        }
        if seen.insert(dim) {
            uniq.push(dim);
        }
    }
    Ok(Some(DimSelection::Vec(uniq)))
}

fn map_scalar_dim_error(error: tensor::DimensionDecodeError) -> RuntimeError {
    use tensor::DimensionDecodeError;
    let message = match &error {
        DimensionDecodeError::Fractional => "min: dimension must be integral".to_string(),
        DimensionDecodeError::NonFinite => "min: dimension must be finite".to_string(),
        DimensionDecodeError::Negative | DimensionDecodeError::BelowMinimum => {
            "min: dimension must be >= 1".to_string()
        }
        DimensionDecodeError::OutOfPlatformRange
        | DimensionDecodeError::IntegerNegative
        | DimensionDecodeError::IntegerOutOfPlatformRange => {
            "min: dimension is outside the supported range".to_string()
        }
        DimensionDecodeError::Gather(_) => error.to_string(),
    };
    error.attach_cause(min_invalid_argument(message))
}

fn map_vector_dim_error(error: tensor::DimensionDecodeError) -> RuntimeError {
    use tensor::DimensionDecodeError;
    let message = match &error {
        DimensionDecodeError::Negative
        | DimensionDecodeError::IntegerNegative
        | DimensionDecodeError::IntegerOutOfPlatformRange
        | DimensionDecodeError::BelowMinimum => "min: dimension indices must be >= 1".to_string(),
        DimensionDecodeError::NonFinite => "min: dimension entries must be finite".to_string(),
        DimensionDecodeError::Fractional => "min: dimension entries must be integers".to_string(),
        DimensionDecodeError::OutOfPlatformRange | DimensionDecodeError::Gather(_) => {
            error.to_string()
        }
    };
    error.attach_cause(min_invalid_argument(message))
}

async fn reduction_min(
    runtime: Option<&crate::context::RuntimeContext>,
    value: Value,
    args: ReductionArgs,
    demand: super::extrema_provider::ExtremaDemand,
) -> BuiltinResult<super::extrema_provider::ExtremaResult> {
    match value {
        Value::GpuTensor(handle) => {
            if let Some(eval) = reduction_min_gpu(runtime, handle.clone(), &args, demand).await? {
                return Ok(eval);
            }
            // Fall back to host if GPU path is unavailable.
            let value = super::extrema_provider::ExtremaKind::Min
                .gather(&handle)
                .await?;
            reduction_min_host(value, &args).map(|eval| {
                super::extrema_provider::ExtremaResult::from_pair(demand, eval.values, eval.indices)
            })
        }
        other => {
            let typed_host_input = matches!(&other, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32);
            let eval = reduction_min_host(other, &args)?;
            if typed_host_input
                && matches!(&eval.values, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
            {
                if let Some(runtime) = runtime {
                    if let Some(route) = runtime.vm_fusion_host_route() {
                        route.confirm_host_branch(
                            runtime,
                            crate::context::VmFusionHostInstruction::Reduction,
                        );
                    }
                }
            }
            Ok(super::extrema_provider::ExtremaResult::from_pair(
                demand,
                eval.values,
                eval.indices,
            ))
        }
    }
}

async fn reduction_min_gpu(
    runtime: Option<&crate::context::RuntimeContext>,
    handle: GpuTensorHandle,
    args: &ReductionArgs,
    demand: super::extrema_provider::ExtremaDemand,
) -> BuiltinResult<Option<super::extrema_provider::ExtremaResult>> {
    if args.nan_mode == ReductionNaN::Omit {
        log::trace!("min: gpu path disabled (nan_mode=omit)");
        return Ok(None);
    }
    if args.comparison != ComparisonMethod::Auto {
        log::trace!("min: gpu path disabled (comparison != auto)");
        return Ok(None);
    }
    if args.linear_index {
        log::trace!("min: gpu path disabled (linear_index=true)");
        return Ok(None);
    }
    let target_dim = match args.selection {
        DimSelection::Auto => default_dimension_from_shape(handle.shape()),
        DimSelection::Dim(dim) => dim,
        DimSelection::Vec(ref dims) if dims.len() == 1 => dims[0],
        DimSelection::All => {
            if handle.shape().len() <= 1 {
                1
            } else {
                return Ok(None);
            }
        }
        _ => return Ok(None),
    };
    if target_dim == 0 {
        return Ok(None);
    }
    // MATLAB dimensions are 1-based; `reduce_min_dim` expects zero-based.
    let zero_based = target_dim.saturating_sub(1);
    if zero_based >= handle.shape().len() {
        return Ok(None);
    }
    log::trace!(
        "min: attempting reduce_min_dim dim={} (zero_based={}) shape={:?} device_id={}",
        target_dim,
        zero_based,
        handle.shape(),
        handle.device_id()
    );
    super::extrema_provider::ExtremaKind::Min
        .reduce_dim(runtime, &handle, zero_based, demand)
        .await
}

fn reduction_min_host(value: Value, args: &ReductionArgs) -> BuiltinResult<MinEvaluation> {
    if let Value::Int(value) = &value {
        let storage = crate::builtins::math::reduction::integer_native::storage_from_scalar(value);
        return reduce_integer_min(&storage, vec![1, 1], args);
    }
    if let Some((storage, shape)) = native_integer_input(&value) {
        return reduce_integer_min(storage, shape, args);
    }
    match materialize_for_min("min", value)? {
        InputData::Real(tensor) => reduce_real_tensor(tensor, args),
        InputData::Complex(tensor) => reduce_complex_tensor(tensor, args),
    }
}

fn native_integer_input(value: &Value) -> Option<(&runmat_value::IntegerStorage, Vec<usize>)> {
    match value {
        Value::Tensor(tensor) => tensor
            .integer_storage()
            .map(|storage| (storage, tensor.shape().to_vec())),
        _ => None,
    }
}

fn reduce_integer_min(
    storage: &runmat_value::IntegerStorage,
    shape: Vec<usize>,
    args: &ReductionArgs,
) -> BuiltinResult<MinEvaluation> {
    if storage.is_empty() {
        // Integer extrema preserve the empty input shape for every reduction
        // selector. A non-empty reduced shape cannot represent an empty typed
        // payload, and MATLAB extrema of an empty array stay empty.
        let output_shape = shape;
        let values = crate::builtins::math::reduction::integer_native::empty_like(
            storage,
            output_shape.clone(),
        )
        .map_err(|error| min_internal_error(format!("min: {error}")))?;
        let indices = Tensor::new(Vec::new(), output_shape)
            .map_err(|error| min_internal_error(format!("min: {error}")))?;
        return Ok(MinEvaluation {
            values,
            indices: tensor::tensor_into_value(indices),
        });
    }

    let resolved = resolve_reduction_dims(&shape, &args.selection)?;
    let comparison = match args.comparison {
        ComparisonMethod::Auto | ComparisonMethod::Real => {
            crate::builtins::math::reduction::integer_native::ExtremaComparison::Natural
        }
        ComparisonMethod::Abs => {
            crate::builtins::math::reduction::integer_native::ExtremaComparison::Absolute
        }
    };
    let extrema = crate::builtins::math::reduction::integer_native::extrema(
        storage,
        &shape,
        resolved.output_shape,
        &resolved.reduced_dims,
        &resolved.dims_mask,
        &resolved.reduce_strides,
        resolved.reduce_all,
        args.linear_index,
        crate::builtins::math::reduction::integer_native::ExtremaDirection::Min,
        comparison,
    )
    .map_err(|error| min_internal_error(format!("min: {error}")))?;
    Ok(MinEvaluation {
        values: extrema.values,
        indices: extrema.indices,
    })
}

enum InputData {
    Real(Tensor),
    Complex(ComplexTensor),
}

fn real_tensor_from_f64(
    values: Vec<f64>,
    shape: Vec<usize>,
    dtype: NumericDType,
) -> Result<Tensor, String> {
    match dtype {
        NumericDType::F64 => Tensor::new(values, shape),
        NumericDType::F32 => Tensor::from_f32(
            values.into_iter().map(|value| value as f32).collect(),
            shape,
        ),
        dtype => Err(format!(
            "min: unexpected {} storage in floating reduction",
            dtype.class_name()
        )),
    }
}

fn materialize_for_min(name: &str, value: Value) -> BuiltinResult<InputData> {
    match value {
        Value::Tensor(t) => Ok(InputData::Real(t)),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical).map_err(min_invalid_input)?;
            Ok(InputData::Real(tensor))
        }
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1])
                .map_err(|e| min_internal_error(format!("{name}: {e}")))?;
            Ok(InputData::Real(tensor))
        }
        Value::Int(i) => {
            let tensor = Tensor::new(vec![i.to_f64()], vec![1, 1])
                .map_err(|e| min_internal_error(format!("{name}: {e}")))?;
            Ok(InputData::Real(tensor))
        }
        Value::Bool(b) => {
            let tensor = Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
                .map_err(|e| min_internal_error(format!("{name}: {e}")))?;
            Ok(InputData::Real(tensor))
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| min_internal_error(format!("{name}: {e}")))?;
            Ok(InputData::Complex(tensor))
        }
        Value::ComplexTensor(ct) => Ok(InputData::Complex(ct)),
        Value::String(_)
        | Value::StringArray(_)
        | Value::CharArray(_)
        | Value::SparseTensor(_)
        | Value::Symbolic(_)
        | Value::SymbolicArray(_)
        | Value::Cell(_) => Err(min_invalid_input(format!(
            "{name}: expected numeric or logical input, received non-numeric value"
        ))),
        Value::GpuTensor(_) => Err(min_internal_error(format!(
            "{name}: internal error – GPU tensors must be gathered before host execution"
        ))),
        Value::ObjectArray(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Struct(_)
        | Value::StructArray(_)
        | Value::Listener(_) => Err(min_invalid_input(format!("{name}: unsupported input type"))),
        Value::ScopedFunctionHandle(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle(_)
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::Future(_)
        | Value::Task(_)
        | Value::Pool(_)
        | Value::Job(_)
        | Value::Distributed(_)
        | Value::Composite(_)
        | Value::Foreign(_)
        | Value::OutputList(_) => Err(min_invalid_input(format!("{name}: unsupported input type"))),
    }
}

fn reduce_real_tensor(tensor: Tensor, args: &ReductionArgs) -> BuiltinResult<MinEvaluation> {
    let shape = tensor.shape().to_vec();
    let dtype = tensor.numeric_dtype();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|e| min_internal_error(format!("min: {e}")))?;
    let data = storage.materialize_f64();
    if data.is_empty() {
        let output_shape = resolve_output_shape(&shape, &args.selection, &[])?;
        let values = real_tensor_from_f64(Vec::new(), output_shape.clone(), dtype)
            .map_err(|e| min_internal_error(format!("min: {e}")))?;
        let indices = Tensor::new(Vec::new(), output_shape)
            .map_err(|e| min_internal_error(format!("min: {e}")))?;
        return Ok(MinEvaluation {
            values: tensor::tensor_into_value(values),
            indices: tensor::tensor_into_value(indices),
        });
    }
    let resolved = resolve_reduction_dims(&shape, &args.selection)?;
    let output_shape = resolved.output_shape.clone();
    let output_len = tensor::element_count(&output_shape);

    if output_len == 0 {
        let values = real_tensor_from_f64(Vec::new(), output_shape.clone(), dtype)
            .map_err(|e| min_internal_error(format!("min: {e}")))?;
        let indices = Tensor::new(Vec::new(), output_shape)
            .map_err(|e| min_internal_error(format!("min: {e}")))?;
        return Ok(MinEvaluation {
            values: tensor::tensor_into_value(values),
            indices: tensor::tensor_into_value(indices),
        });
    }

    let strides = compute_strides(&shape);
    let output_strides = compute_strides(&output_shape);
    let dims_mask = resolved.dims_mask.clone();
    let reduce_strides = resolved.reduce_strides.clone();

    let mut best = vec![BestReal::new(); output_len];
    let mut coords = vec![0usize; shape.len()];
    for &value in &data {
        let out_idx = map_output_index(&coords, &output_strides, &dims_mask);
        let reduce_idx = map_reduce_index(
            &coords,
            &resolved.reduced_dims,
            &reduce_strides,
            resolved.reduce_all,
        );
        let full_idx = map_linear_index(&coords, &strides);

        update_best_real(
            &mut best[out_idx],
            value,
            reduce_idx,
            full_idx,
            args.nan_mode,
            args.comparison,
        );
        increment_coords(&mut coords, &shape);
    }

    let mut values = vec![0.0f64; output_len];
    let mut indices = vec![0.0f64; output_len];

    for (i, entry) in best.iter().enumerate() {
        if entry.nan_fixed {
            values[i] = f64::NAN;
            indices[i] = if args.linear_index || resolved.reduce_all {
                (entry.full_index + 1) as f64
            } else if resolved.reduced_dims.is_empty() {
                1.0
            } else {
                (entry.reduce_index + 1) as f64
            };
            continue;
        }
        if !entry.has_value {
            values[i] = f64::NAN;
            indices[i] = f64::NAN;
            continue;
        }
        values[i] = entry.value;
        indices[i] = if args.linear_index || resolved.reduce_all {
            (entry.full_index + 1) as f64
        } else if resolved.reduced_dims.is_empty() {
            1.0
        } else {
            (entry.reduce_index + 1) as f64
        };
    }

    let value_tensor = real_tensor_from_f64(values, output_shape.clone(), dtype)
        .map_err(|e| min_internal_error(format!("min: {e}")))?;
    let index_tensor =
        Tensor::new(indices, output_shape).map_err(|e| min_internal_error(format!("min: {e}")))?;

    Ok(MinEvaluation {
        values: tensor::tensor_into_value(value_tensor),
        indices: tensor::tensor_into_value(index_tensor),
    })
}

fn reduce_complex_tensor(
    tensor: ComplexTensor,
    args: &ReductionArgs,
) -> BuiltinResult<MinEvaluation> {
    let shape = tensor.shape().to_vec();
    let dtype = tensor.numeric_dtype();
    let data = tensor.materialize_f64();
    if data.is_empty() {
        let output_shape = resolve_output_shape(&shape, &args.selection, &[])?;
        let values =
            ComplexTensor::from_f64_values_with_dtype(Vec::new(), output_shape.clone(), dtype)
                .map_err(|e| min_internal_error(format!("min: {e}")))?;
        let indices = Tensor::new(Vec::new(), output_shape)
            .map_err(|e| min_internal_error(format!("min: {e}")))?;
        return Ok(MinEvaluation {
            values: complex_tensor_into_value(values),
            indices: tensor::tensor_into_value(indices),
        });
    }

    let resolved = resolve_reduction_dims(&shape, &args.selection)?;
    let output_shape = resolved.output_shape.clone();
    let output_len = tensor::element_count(&output_shape);

    if output_len == 0 {
        let values =
            ComplexTensor::from_f64_values_with_dtype(Vec::new(), output_shape.clone(), dtype)
                .map_err(|e| min_internal_error(format!("min: {e}")))?;
        let indices = Tensor::new(Vec::new(), output_shape)
            .map_err(|e| min_internal_error(format!("min: {e}")))?;
        return Ok(MinEvaluation {
            values: complex_tensor_into_value(values),
            indices: tensor::tensor_into_value(indices),
        });
    }

    let strides = compute_strides(&shape);
    let output_strides = compute_strides(&output_shape);
    let dims_mask = resolved.dims_mask.clone();
    let reduce_strides = resolved.reduce_strides.clone();

    let mut best = vec![BestComplex::new(); output_len];
    let mut coords = vec![0usize; shape.len()];

    for &(re, im) in &data {
        let out_idx = map_output_index(&coords, &output_strides, &dims_mask);
        let reduce_idx = map_reduce_index(
            &coords,
            &resolved.reduced_dims,
            &reduce_strides,
            resolved.reduce_all,
        );
        let full_idx = map_linear_index(&coords, &strides);
        update_best_complex(
            &mut best[out_idx],
            (re, im),
            reduce_idx,
            full_idx,
            args.nan_mode,
            args.comparison,
        );
        increment_coords(&mut coords, &shape);
    }

    let mut values = vec![(0.0f64, 0.0f64); output_len];
    let mut indices = vec![0.0f64; output_len];

    for (i, entry) in best.iter().enumerate() {
        if entry.nan_fixed {
            values[i] = (f64::NAN, f64::NAN);
            indices[i] = if args.linear_index || resolved.reduce_all {
                (entry.full_index + 1) as f64
            } else if resolved.reduced_dims.is_empty() {
                1.0
            } else {
                (entry.reduce_index + 1) as f64
            };
            continue;
        }
        if !entry.has_value {
            values[i] = (f64::NAN, f64::NAN);
            indices[i] = f64::NAN;
            continue;
        }
        values[i] = entry.value;
        indices[i] = if args.linear_index || resolved.reduce_all {
            (entry.full_index + 1) as f64
        } else if resolved.reduced_dims.is_empty() {
            1.0
        } else {
            (entry.reduce_index + 1) as f64
        };
    }

    let value_tensor =
        ComplexTensor::from_f64_values_with_dtype(values, output_shape.clone(), dtype)
            .map_err(|e| min_internal_error(format!("min: {e}")))?;
    let index_tensor =
        Tensor::new(indices, output_shape).map_err(|e| min_internal_error(format!("min: {e}")))?;
    Ok(MinEvaluation {
        values: complex_tensor_into_value(value_tensor),
        indices: tensor::tensor_into_value(index_tensor),
    })
}

/// The Single real one-output host path materializes F64 input for extrema,
/// retains a BestReal record and F64 value/index arrays per output, and may
/// briefly retain a converted F32 result alongside them. Include the
/// rank-sized strides, masks, and coordinates at their joint maximum. The
/// caller accounts for the input tensor and producer outputs separately.
pub(crate) fn checked_single_host_scratch(
    input_elements: u64,
    output_elements: u64,
    rank: u64,
) -> Option<u64> {
    let input_copy = input_elements.checked_mul(8)?;
    let output_record = u64::try_from(std::mem::size_of::<BestReal>())
        .ok()?
        .checked_add(8)?
        .checked_add(8)?
        .checked_add(4)?;
    let outputs = output_elements.checked_mul(output_record)?;
    // The real path's owned usize vectors are: source shape; resolved output
    // shape, reduced axes, and reduction strides; local output-shape clone;
    // input and output strides; local reduction-strides clone; coordinates;
    // value-tensor shape; and index-tensor shape. Bound all eleven at once,
    // plus the canonical binding's argument Tensor shape clone. Some do not
    // overlap. The resolved and local dimension masks are two bool vectors.
    // Empty/singleton normalization uses rank >= 2.
    let usize_word = u64::try_from(std::mem::size_of::<usize>()).ok()?;
    let bool_word = u64::try_from(std::mem::size_of::<bool>()).ok()?;
    let dimensions = rank.checked_mul(
        usize_word
            .checked_mul(12)?
            .checked_add(bool_word.checked_mul(2)?)?,
    )?;
    input_copy.checked_add(outputs)?.checked_add(dimensions)
}

#[derive(Debug, Clone)]
struct BestReal {
    value: f64,
    reduce_index: usize,
    full_index: usize,
    has_value: bool,
    nan_fixed: bool,
}

impl BestReal {
    fn new() -> Self {
        Self {
            value: 0.0,
            reduce_index: 0,
            full_index: 0,
            has_value: false,
            nan_fixed: false,
        }
    }
}

#[derive(Debug, Clone)]
struct BestComplex {
    value: (f64, f64),
    reduce_index: usize,
    full_index: usize,
    has_value: bool,
    nan_fixed: bool,
}

impl BestComplex {
    fn new() -> Self {
        Self {
            value: (0.0, 0.0),
            reduce_index: 0,
            full_index: 0,
            has_value: false,
            nan_fixed: false,
        }
    }
}

fn resolve_output_shape(
    shape: &[usize],
    selection: &DimSelection,
    reduced_dims: &[usize],
) -> BuiltinResult<Vec<usize>> {
    if is_scalar_shape(shape) {
        return Ok(normalize_scalar_shape(shape));
    }
    let mut output = shape.to_vec();
    match selection {
        DimSelection::All => {
            output.fill(1);
        }
        _ => {
            for &dim in reduced_dims {
                if dim < output.len() {
                    output[dim] = 1;
                }
            }
        }
    }
    Ok(output)
}

struct ResolvedDims {
    output_shape: Vec<usize>,
    reduced_dims: Vec<usize>,
    reduce_all: bool,
    dims_mask: Vec<bool>,
    reduce_strides: Vec<usize>,
}

fn resolve_reduction_dims(
    shape: &[usize],
    selection: &DimSelection,
) -> BuiltinResult<ResolvedDims> {
    if is_scalar_shape(shape) {
        return Ok(ResolvedDims {
            output_shape: normalize_scalar_shape(shape),
            reduced_dims: Vec::new(),
            reduce_all: true,
            dims_mask: Vec::new(),
            reduce_strides: Vec::new(),
        });
    }

    let mut reduced_dims = match selection {
        DimSelection::Auto => {
            let mut dim = None;
            for (index, &len) in shape.iter().enumerate() {
                if len > 1 {
                    dim = Some(index);
                    break;
                }
            }
            vec![dim.unwrap_or(0)]
        }
        DimSelection::Dim(dim) => {
            if *dim == 0 {
                return Err(min_invalid_argument("min: dimension must be >= 1"));
            }
            let index = dim.saturating_sub(1);
            if index >= shape.len() {
                Vec::new()
            } else {
                vec![index]
            }
        }
        DimSelection::Vec(dims) => {
            if dims.is_empty() {
                Vec::new()
            } else {
                dims.iter()
                    .filter_map(|dim| {
                        if *dim == 0 {
                            None
                        } else {
                            let idx = dim - 1;
                            if idx < shape.len() {
                                Some(idx)
                            } else {
                                None
                            }
                        }
                    })
                    .collect()
            }
        }
        DimSelection::All => (0..shape.len()).collect(),
    };

    reduced_dims.sort_unstable();
    reduced_dims.dedup();

    let reduce_all = !reduced_dims.is_empty()
        && reduced_dims.len() == shape.len()
        && reduced_dims.iter().enumerate().all(|(i, &d)| i == d);

    let output_shape = resolve_output_shape(shape, selection, &reduced_dims)?;
    let mut dims_mask = vec![false; shape.len()];
    for &dim in &reduced_dims {
        if dim < dims_mask.len() {
            dims_mask[dim] = true;
        }
    }
    let reduce_strides = compute_subspace_strides(shape, &reduced_dims);

    Ok(ResolvedDims {
        output_shape,
        reduced_dims,
        reduce_all,
        dims_mask,
        reduce_strides,
    })
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1usize;
    for &len in shape {
        strides.push(stride);
        stride = stride.saturating_mul(len.max(1));
    }
    strides
}

fn compute_subspace_strides(shape: &[usize], dims: &[usize]) -> Vec<usize> {
    if dims.is_empty() {
        return Vec::new();
    }
    let mut strides = Vec::with_capacity(dims.len());
    let mut accum = 1usize;
    for &dim in dims {
        let len = shape.get(dim).copied().unwrap_or(1).max(1);
        strides.push(accum);
        accum = accum.saturating_mul(len);
    }
    strides
}

fn map_output_index(coords: &[usize], output_strides: &[usize], dims_mask: &[bool]) -> usize {
    if coords.is_empty() {
        return 0;
    }
    let mut index = 0usize;
    for (dim, stride) in output_strides.iter().enumerate() {
        let coord = if *dims_mask.get(dim).unwrap_or(&false) {
            0
        } else {
            coords[dim]
        };
        index = index.saturating_add(coord.saturating_mul(*stride));
    }
    index
}

fn map_reduce_index(
    coords: &[usize],
    reduced_dims: &[usize],
    reduce_strides: &[usize],
    reduce_all: bool,
) -> usize {
    if reduced_dims.is_empty() {
        return 0;
    }
    if reduce_all {
        // When all dimensions are reduced, the full index is used separately.
        return 0;
    }
    let mut index = 0usize;
    for (pos, &dim) in reduced_dims.iter().enumerate() {
        if let Some(coord) = coords.get(dim) {
            if let Some(stride) = reduce_strides.get(pos) {
                index = index.saturating_add(coord.saturating_mul(*stride));
            }
        }
    }
    index
}

fn map_linear_index(coords: &[usize], strides: &[usize]) -> usize {
    coords
        .iter()
        .zip(strides.iter())
        .fold(0usize, |acc, (&coord, &stride)| {
            acc.saturating_add(coord.saturating_mul(stride))
        })
}

fn increment_coords(coords: &mut [usize], shape: &[usize]) {
    for dim in 0..coords.len() {
        if shape[dim] == 0 {
            continue;
        }
        coords[dim] += 1;
        if coords[dim] < shape[dim] {
            break;
        }
        coords[dim] = 0;
    }
}

fn update_best_real(
    best: &mut BestReal,
    value: f64,
    reduce_index: usize,
    full_index: usize,
    nan_mode: ReductionNaN,
    comparison: ComparisonMethod,
) {
    if value.is_nan() {
        match nan_mode {
            ReductionNaN::Include => {
                if !best.nan_fixed {
                    best.value = f64::NAN;
                    best.reduce_index = reduce_index;
                    best.full_index = full_index;
                    best.has_value = true;
                    best.nan_fixed = true;
                }
            }
            ReductionNaN::Omit => {}
        }
        return;
    }
    if best.nan_fixed {
        return;
    }

    if !best.has_value {
        best.value = value;
        best.reduce_index = reduce_index;
        best.full_index = full_index;
        best.has_value = true;
        return;
    }

    if should_replace_real(best.value, value, comparison) {
        best.value = value;
        best.reduce_index = reduce_index;
        best.full_index = full_index;
    }
}

fn update_best_complex(
    best: &mut BestComplex,
    value: (f64, f64),
    reduce_index: usize,
    full_index: usize,
    nan_mode: ReductionNaN,
    comparison: ComparisonMethod,
) {
    if value.0.is_nan() || value.1.is_nan() {
        match nan_mode {
            ReductionNaN::Include => {
                if !best.nan_fixed {
                    best.value = (f64::NAN, f64::NAN);
                    best.reduce_index = reduce_index;
                    best.full_index = full_index;
                    best.has_value = true;
                    best.nan_fixed = true;
                }
            }
            ReductionNaN::Omit => {}
        }
        return;
    }
    if best.nan_fixed {
        return;
    }

    if !best.has_value {
        best.value = value;
        best.reduce_index = reduce_index;
        best.full_index = full_index;
        best.has_value = true;
        return;
    }

    if should_replace_complex(best.value, value, comparison) {
        best.value = value;
        best.reduce_index = reduce_index;
        best.full_index = full_index;
    }
}

fn should_replace_real(current: f64, candidate: f64, comparison: ComparisonMethod) -> bool {
    match comparison {
        ComparisonMethod::Auto | ComparisonMethod::Real => {
            if candidate < current {
                return true;
            }
            if candidate > current {
                return false;
            }
            if candidate == 0.0 && current == 0.0 {
                return candidate.is_sign_negative() && !current.is_sign_negative();
            }
            false
        }
        ComparisonMethod::Abs => {
            let curr_abs = current.abs();
            let cand_abs = candidate.abs();
            if cand_abs < curr_abs {
                return true;
            }
            if cand_abs > curr_abs {
                return false;
            }
            if candidate < current {
                return true;
            }
            if candidate > current {
                return false;
            }
            if candidate == 0.0 && current == 0.0 {
                return candidate.is_sign_negative() && !current.is_sign_negative();
            }
            false
        }
    }
}

fn should_replace_complex(
    current: (f64, f64),
    candidate: (f64, f64),
    comparison: ComparisonMethod,
) -> bool {
    match comparison {
        ComparisonMethod::Auto | ComparisonMethod::Abs => {
            compare_complex_auto(current, candidate) == Ordering::Greater
        }
        ComparisonMethod::Real => compare_complex_real(current, candidate) == Ordering::Greater,
    }
}

fn compare_complex_auto(a: (f64, f64), b: (f64, f64)) -> Ordering {
    let a_mag = magnitude_squared(a);
    let b_mag = magnitude_squared(b);
    if a_mag < b_mag {
        return Ordering::Less;
    }
    if a_mag > b_mag {
        return Ordering::Greater;
    }
    // Equal magnitude: tie-break using phase angle.
    let a_angle = a.1.atan2(a.0);
    let b_angle = b.1.atan2(b.0);
    if a_angle < b_angle {
        Ordering::Less
    } else if a_angle > b_angle {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

fn compare_complex_real(a: (f64, f64), b: (f64, f64)) -> Ordering {
    if a.0 < b.0 {
        return Ordering::Less;
    }
    if a.0 > b.0 {
        return Ordering::Greater;
    }
    // Equal real parts: use magnitude and phase tie-breakers.
    compare_complex_auto(a, b)
}

fn magnitude_squared(z: (f64, f64)) -> f64 {
    z.0.mul_add(z.0, z.1 * z.1)
}

fn default_dimension_from_shape(shape: &[usize]) -> usize {
    if is_scalar_shape(shape) {
        return 1;
    }
    for (i, &len) in shape.iter().enumerate() {
        if len > 1 {
            return i + 1;
        }
    }
    1
}

async fn elementwise_min(
    value: Value,
    args: ElementwiseArgs,
    demand: super::extrema_provider::ExtremaDemand,
) -> BuiltinResult<super::extrema_provider::ExtremaResult> {
    use super::extrema_provider::{ExtremaDemand, ExtremaKind, ExtremaResult};
    let ElementwiseArgs { other, comparison } = args;
    let kind = ExtremaKind::Min;
    let resident = if comparison == ComparisonMethod::Auto {
        match (&value, &other) {
            (Value::GpuTensor(left), Value::GpuTensor(right)) => {
                kind.pair_values(left, right).await?
            }
            (Value::GpuTensor(input), other) => match extract_scalar(other) {
                Some(scalar) => kind.scalar_values(input, other, scalar, false).await?,
                None => None,
            },
            (other, Value::GpuTensor(input)) => match extract_scalar(other) {
                Some(scalar) => kind.scalar_values(input, other, scalar, true).await?,
                None => None,
            },
            _ => None,
        }
    } else {
        None
    };
    if let Some(values) = resident {
        if demand == ExtremaDemand::ValuesOnly {
            return Ok(ExtremaResult::Values(Value::GpuTensor(values)));
        }
        // Generic <=/>= masks do not encode NaN or signed-zero extrema origins.
        // The existing typed host selector is the sole origin authority.
        let left = kind.gather_input(value).await?;
        let right = kind.gather_input(other).await?;
        let evaluated = elementwise_real_or_complex(left, right, comparison)?;
        return Ok(ExtremaResult::Pair {
            values: Value::GpuTensor(values),
            indices: evaluated.indices,
        });
    }
    let left = kind.gather_input(value).await?;
    let right = kind.gather_input(other).await?;
    let evaluated = elementwise_real_or_complex(left, right, comparison)?;
    Ok(ExtremaResult::from_pair(
        demand,
        evaluated.values,
        evaluated.indices,
    ))
}

fn extract_scalar(v: &Value) -> Option<f64> {
    match v {
        Value::Num(n) => Some(*n),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(t) if tensor::is_scalar_tensor(t) && t.integer_storage().is_none() => {
            Some(tensor::tensor_value_f64(t, 0))
        }
        Value::LogicalArray(l) if l.data.len() == 1 => Some(if l.data[0] != 0 { 1.0 } else { 0.0 }),
        _ => None,
    }
}

fn elementwise_real_or_complex(
    lhs: Value,
    rhs: Value,
    comparison: ComparisonMethod,
) -> BuiltinResult<MinEvaluation> {
    let integer_comparison = match comparison {
        ComparisonMethod::Auto | ComparisonMethod::Real => {
            crate::builtins::math::reduction::integer_native::ExtremaComparison::Natural
        }
        ComparisonMethod::Abs => {
            crate::builtins::math::reduction::integer_native::ExtremaComparison::Absolute
        }
    };
    if let Some(eval) = crate::builtins::math::reduction::integer_native::elementwise_value_extrema(
        &lhs,
        &rhs,
        crate::builtins::math::reduction::integer_native::ExtremaDirection::Min,
        integer_comparison,
        false,
    )
    .map_err(|error| min_size_mismatch(format!("min: {error}")))?
    {
        return Ok(MinEvaluation {
            values: eval.values,
            indices: eval.indices,
        });
    }
    if crate::builtins::math::reduction::integer_native::value_has_integer_storage(&lhs)
        || crate::builtins::math::reduction::integer_native::value_has_integer_storage(&rhs)
    {
        return Err(min_invalid_input(
            "min: integer pairwise inputs require the same integer class or a scalar double",
        ));
    }
    if let Some(eval) = scalar_elementwise_min(&lhs, &rhs, comparison) {
        return Ok(eval);
    }
    match (
        materialize_for_min("min", lhs)?,
        materialize_for_min("min", rhs)?,
    ) {
        (InputData::Complex(a), InputData::Complex(b)) => elementwise_complex_min(a, b, comparison),
        (InputData::Complex(a), InputData::Real(b)) => {
            let converted = promote_real_tensor_to_complex(b);
            elementwise_complex_min(a, converted, comparison)
        }
        (InputData::Real(a), InputData::Complex(b)) => {
            let converted = promote_real_tensor_to_complex(a);
            elementwise_complex_min(converted, b, comparison)
        }
        (InputData::Real(a), InputData::Real(b)) => elementwise_real_min(a, b, comparison),
    }
}

fn scalar_complex_value(value: &Value) -> Option<(f64, f64)> {
    match value {
        Value::Complex(re, im) => Some((*re, *im)),
        Value::ComplexTensor(ct) if tensor::is_scalar_complex_tensor(ct) => {
            let value = tensor::complex_tensor_value_complex64(ct, 0);
            Some((value.re, value.im))
        }
        _ => None,
    }
}

fn scalar_elementwise_min(
    lhs: &Value,
    rhs: &Value,
    comparison: ComparisonMethod,
) -> Option<MinEvaluation> {
    if super::extrema_provider::needs_typed_scalar_selection(lhs)
        || super::extrema_provider::needs_typed_scalar_selection(rhs)
    {
        return None;
    }
    let left = scalar_complex_value(lhs).or_else(|| extract_scalar(lhs).map(|v| (v, 0.0)))?;
    let right = scalar_complex_value(rhs).or_else(|| extract_scalar(rhs).map(|v| (v, 0.0)))?;
    let (ar, ai) = left;
    let (br, bi) = right;
    if ai != 0.0 || bi != 0.0 {
        let (value, origin) = choose_complex_elementwise((ar, ai), (br, bi), comparison);
        return Some(MinEvaluation {
            values: Value::Complex(value.0, value.1),
            indices: Value::Num(origin),
        });
    }
    let (value, origin) = choose_real_elementwise(ar, br, comparison);
    Some(MinEvaluation {
        values: Value::Num(value),
        indices: Value::Num(origin),
    })
}

fn elementwise_real_min(
    lhs: Tensor,
    rhs: Tensor,
    comparison: ComparisonMethod,
) -> BuiltinResult<MinEvaluation> {
    let plan = BroadcastPlan::new(lhs.shape(), rhs.shape())
        .map_err(|err| min_size_mismatch(format!("min: {err}")))?;
    let mut indices = vec![0.0f64; plan.len()];
    let lhs_storage = lhs
        .into_numeric_storage()
        .map_err(|e| min_internal_error(format!("min: {e}")))?;
    let rhs_storage = rhs
        .into_numeric_storage()
        .map_err(|e| min_internal_error(format!("min: {e}")))?;

    macro_rules! select_same_class {
        ($left:expr, $right:expr, $variant:ident) => {{
            let mut values = Vec::with_capacity(plan.len());
            for (offset, index_a, index_b) in plan.iter() {
                let a = $left[index_a];
                let b = $right[index_b];
                let (_, origin) = choose_real_elementwise(a as f64, b as f64, comparison);
                values.push(if origin == 1.0 { a } else { b });
                indices[offset] = origin;
            }
            NumericStorage::$variant(values)
        }};
    }

    let values = match (lhs_storage, rhs_storage) {
        (NumericStorage::F64(left), NumericStorage::F64(right)) => {
            select_same_class!(left, right, F64)
        }
        (NumericStorage::F32(left), NumericStorage::F32(right)) => {
            select_same_class!(left, right, F32)
        }
        (NumericStorage::I8(left), NumericStorage::I8(right)) => {
            select_same_class!(left, right, I8)
        }
        (NumericStorage::I16(left), NumericStorage::I16(right)) => {
            select_same_class!(left, right, I16)
        }
        (NumericStorage::I32(left), NumericStorage::I32(right)) => {
            select_same_class!(left, right, I32)
        }
        (NumericStorage::I64(left), NumericStorage::I64(right)) => {
            select_same_class!(left, right, I64)
        }
        (NumericStorage::U8(left), NumericStorage::U8(right)) => {
            select_same_class!(left, right, U8)
        }
        (NumericStorage::U16(left), NumericStorage::U16(right)) => {
            select_same_class!(left, right, U16)
        }
        (NumericStorage::U32(left), NumericStorage::U32(right)) => {
            select_same_class!(left, right, U32)
        }
        (NumericStorage::U64(left), NumericStorage::U64(right)) => {
            select_same_class!(left, right, U64)
        }
        (left, right) => {
            let left = left.materialize_f64();
            let right = right.materialize_f64();
            let mut values = Vec::with_capacity(plan.len());
            for (offset, index_a, index_b) in plan.iter() {
                let (value, origin) =
                    choose_real_elementwise(left[index_a], right[index_b], comparison);
                values.push(value);
                indices[offset] = origin;
            }
            NumericStorage::F64(values)
        }
    };

    let value_tensor = Tensor::from_numeric_storage(values, plan.output_shape().to_vec())
        .map_err(|e| min_internal_error(format!("min: {e}")))?;
    let index_tensor = Tensor::new(indices, plan.output_shape().to_vec())
        .map_err(|e| min_internal_error(format!("min: {e}")))?;

    Ok(MinEvaluation {
        values: tensor::tensor_into_value(value_tensor),
        indices: tensor::tensor_into_value(index_tensor),
    })
}

fn elementwise_complex_min(
    lhs: ComplexTensor,
    rhs: ComplexTensor,
    comparison: ComparisonMethod,
) -> BuiltinResult<MinEvaluation> {
    let plan = BroadcastPlan::new(lhs.shape(), rhs.shape())
        .map_err(|err| min_size_mismatch(format!("min: {err}")))?;
    let mut indices = vec![0.0f64; plan.len()];
    let lhs_storage = lhs.into_complex_storage();
    let rhs_storage = rhs.into_complex_storage();

    macro_rules! select_same_class {
        ($left:expr, $right:expr, $variant:ident) => {{
            let mut values = Vec::with_capacity(plan.len());
            for (offset, index_a, index_b) in plan.iter() {
                let a = $left[index_a];
                let b = $right[index_b];
                let (_, origin) = choose_complex_elementwise(
                    (f64::from(a.0), f64::from(a.1)),
                    (f64::from(b.0), f64::from(b.1)),
                    comparison,
                );
                values.push(if origin == 1.0 { a } else { b });
                indices[offset] = origin;
            }
            ComplexStorage::$variant(values.into())
        }};
    }

    let values = match (lhs_storage, rhs_storage) {
        (ComplexStorage::F64(left), ComplexStorage::F64(right)) => {
            select_same_class!(left, right, F64)
        }
        (ComplexStorage::F32(left), ComplexStorage::F32(right)) => {
            select_same_class!(left, right, F32)
        }
        (left, right) => {
            let left = left.materialize_f64();
            let right = right.materialize_f64();
            let mut values = Vec::with_capacity(plan.len());
            for (offset, index_a, index_b) in plan.iter() {
                let (value, origin) =
                    choose_complex_elementwise(left[index_a], right[index_b], comparison);
                values.push(value);
                indices[offset] = origin;
            }
            ComplexStorage::F64(values.into())
        }
    };

    let value_tensor = ComplexTensor::from_complex_storage(values, plan.output_shape().to_vec())
        .map_err(|e| min_internal_error(format!("min: {e}")))?;
    let index_tensor = Tensor::new(indices, plan.output_shape().to_vec())
        .map_err(|e| min_internal_error(format!("min: {e}")))?;

    Ok(MinEvaluation {
        values: complex_tensor_into_value(value_tensor),
        indices: tensor::tensor_into_value(index_tensor),
    })
}

fn promote_real_tensor_to_complex(tensor: Tensor) -> ComplexTensor {
    let shape = tensor.shape().to_vec();
    let storage = match tensor
        .into_numeric_storage()
        .expect("real extrema input has numeric storage")
    {
        NumericStorage::F32(values) => {
            ComplexStorage::F32(values.into_iter().map(|re| (re, 0.0)).collect())
        }
        storage => ComplexStorage::F64(
            storage
                .materialize_f64()
                .into_iter()
                .map(|re| (re, 0.0))
                .collect(),
        ),
    };
    ComplexTensor::from_complex_storage(storage, shape)
        .expect("real tensor shape remains valid after promotion")
}

fn choose_real_elementwise(a: f64, b: f64, comparison: ComparisonMethod) -> (f64, f64) {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => (f64::NAN, 1.0),
        (true, false) => (f64::NAN, 1.0),
        (false, true) => (f64::NAN, 2.0),
        (false, false) => {
            if should_replace_real(a, b, comparison) {
                (b, 2.0)
            } else {
                (a, 1.0)
            }
        }
    }
}

fn choose_complex_elementwise(
    a: (f64, f64),
    b: (f64, f64),
    comparison: ComparisonMethod,
) -> ((f64, f64), f64) {
    let a_nan = a.0.is_nan() || a.1.is_nan();
    let b_nan = b.0.is_nan() || b.1.is_nan();
    match (a_nan, b_nan) {
        (true, true) => ((f64::NAN, f64::NAN), 1.0),
        (true, false) => ((f64::NAN, f64::NAN), 1.0),
        (false, true) => ((f64::NAN, f64::NAN), 2.0),
        (false, false) => {
            if should_replace_complex(a, b, comparison) {
                (b, 2.0)
            } else {
                (a, 1.0)
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    #[cfg(feature = "wgpu")]
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::{
        HostIntegerDataView, HostIntegerTensorView, HostTensorView, IntegerElementType,
    };
    use runmat_value::{
        ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage, Tensor, Value,
    };

    fn min_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let rest = crate::execution::RootedValues::new(rest)
            .map_err(|error| error.into_runtime_error())?;
        block_on(super::min_builtin(None, value, rest.into()))
    }

    #[test]
    fn min_typed_controls_keep_pairwise_restrictions_and_method_alias() {
        let pairwise = min_builtin(
            Value::Num(3.0),
            vec![
                Value::Num(2.0),
                Value::from("ComparisonMethod"),
                Value::from("magnitude"),
            ],
        )
        .expect("comparison method alias");
        assert_eq!(pairwise, Value::Num(2.0));

        let invalid = min_builtin(
            Value::Num(3.0),
            vec![Value::Num(2.0), Value::from("omitnan")],
        )
        .expect_err("missing policy is reduction-only");
        assert_eq!(invalid.identifier(), MIN_ERROR_INVALID_ARGUMENT.identifier);
        assert!(invalid.message().contains("only supported for reduction"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn same_class_single_scalar_pairwise_retains_typed_storage() {
        let left = Tensor::from_f32(vec![3.25], vec![1, 1]).expect("single left");
        let right = Tensor::from_f32(vec![2.5], vec![1, 1]).expect("single right");
        let (value, index) = evaluate(Value::Tensor(left), &[Value::Tensor(right)])
            .expect("single scalar pairwise minimum")
            .into_pair();
        let Value::Tensor(value) = value else {
            panic!("single scalar remains a typed tensor");
        };
        assert_eq!(value.numeric_dtype(), NumericDType::F32);
        assert_eq!(value.shape(), [1, 1]);
        assert_eq!(value.materialize_f64(), vec![2.5]);
        assert_eq!(index, Value::Num(2.0));
    }

    fn evaluate(value: Value, rest: &[Value]) -> BuiltinResult<MinEvaluation> {
        block_on(super::evaluate(
            value,
            crate::call::FunctionArgumentView::from_slice(rest),
        ))
    }

    fn placeholder() -> Value {
        let tensor = Tensor::new(Vec::<f64>::new(), vec![0, 0]).unwrap();
        Value::Tensor(tensor)
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn min_scalar_returns_input() {
        let result = min_builtin(Value::Num(5.0), Vec::new()).expect("min");
        assert_eq!(result, Value::Num(5.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn min_vector_with_indices() {
        let tensor = Tensor::new(vec![3.0, 1.0, 5.0], vec![3, 1]).unwrap();
        let eval = evaluate(Value::Tensor(tensor), &[]).expect("evaluate");
        let (values, indices) = eval.into_pair();
        assert_eq!(values, Value::Num(1.0));
        assert_eq!(indices, Value::Num(2.0));
    }

    #[test]
    fn min_reduction_preserves_native_single_storage() {
        let tensor = Tensor::from_f32(vec![3.0, 1.0, 4.0, 2.0], vec![2, 2]).unwrap();
        let (values, _) = evaluate(Value::Tensor(tensor), &[])
            .expect("min")
            .into_pair();
        let Value::Tensor(values) = values else {
            panic!("expected tensor values");
        };
        assert_eq!(
            values.into_numeric_storage().expect("single storage"),
            runmat_value::NumericStorage::F32(vec![1.0, 2.0])
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn min_native_integer_reduction_preserves_uint64_values_and_indices() {
        let input = Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![u64::MAX - 1, u64::MAX, 3, 2]),
            vec![2, 2],
        )
        .expect("input");
        let (values, indices) = evaluate(Value::Tensor(input), &[])
            .expect("min")
            .into_pair();
        assert_eq!(
            values,
            Value::Tensor(
                Tensor::new_integer(
                    runmat_value::IntegerStorage::U64(vec![u64::MAX - 1, 2]),
                    vec![1, 2],
                )
                .expect("values"),
            )
        );
        assert_eq!(
            indices,
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("indices")),
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn min_native_integer_abs_all_uses_exact_int64_minimum() {
        let input = Tensor::new_integer(
            runmat_value::IntegerStorage::I64(vec![i64::MIN, -3, 3]),
            vec![3, 1],
        )
        .expect("input");
        let args = vec![
            placeholder(),
            Value::from("all"),
            Value::from("ComparisonMethod"),
            Value::from("abs"),
        ];
        let (values, indices) = evaluate(Value::Tensor(input), &args)
            .expect("min")
            .into_pair();
        assert_eq!(values, Value::Int(IntValue::I64(-3)));
        assert_eq!(indices, Value::Num(2.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn min_native_integer_empty_array_retains_its_class() {
        let input = Tensor::new_integer(runmat_value::IntegerStorage::U32(Vec::new()), vec![0, 0])
            .expect("input");
        let (values, indices) = evaluate(Value::Tensor(input), &[])
            .expect("min")
            .into_pair();
        assert_eq!(
            values,
            Value::Tensor(
                Tensor::new_integer(runmat_value::IntegerStorage::U32(Vec::new()), vec![0, 0])
                    .expect("values"),
            )
        );
        assert_eq!(
            indices,
            Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("indices")),
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn min_native_integer_empty_all_reduction_remains_empty() {
        let input = Tensor::new_integer(runmat_value::IntegerStorage::I16(Vec::new()), vec![0, 0])
            .expect("input");
        let values = evaluate(Value::Tensor(input), &[placeholder(), Value::from("all")])
            .expect("min")
            .into_value();
        assert_eq!(
            values,
            Value::Tensor(
                Tensor::new_integer(runmat_value::IntegerStorage::I16(Vec::new()), vec![0, 0])
                    .expect("values"),
            )
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn min_row_vector_reduces_across_columns() {
        let tensor = Tensor::new(vec![3.0, 1.0, 5.0], vec![1, 3]).unwrap();
        let eval = evaluate(Value::Tensor(tensor), &[]).expect("evaluate");
        let (values, indices) = eval.into_pair();
        assert_eq!(values, Value::Num(1.0));
        assert_eq!(indices, Value::Num(2.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn min_single_row_vector_reduces_across_columns() {
        let tensor = Tensor::from_f32(vec![3.0, 1.0, 5.0], vec![1, 3]).unwrap();
        let eval = evaluate(Value::Tensor(tensor), &[]).expect("evaluate");
        let (values, indices) = eval.into_pair();
        assert_eq!(
            values,
            Value::Tensor(Tensor::from_f32(vec![1.0], vec![1, 1]).unwrap())
        );
        assert_eq!(indices, Value::Num(2.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn min_registered_single_row_vector_reduces_across_columns() {
        let tensor = Tensor::from_f32(vec![3.0, 1.0, 5.0], vec![1, 3]).unwrap();
        let value = block_on(crate::call_builtin_async_with_outputs(
            "min",
            &[Value::Tensor(tensor)],
            1,
        ))
        .expect("dispatch min");
        match value.value() {
            Value::OutputList(values) => {
                assert_eq!(values.len(), 1);
                assert_eq!(
                    values[0],
                    Value::Tensor(Tensor::from_f32(vec![1.0], vec![1, 1]).unwrap())
                );
            }
            other => assert_eq!(
                other,
                &Value::Tensor(Tensor::from_f32(vec![1.0], vec![1, 1]).unwrap())
            ),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn min_matrix_default_dimension() {
        let tensor = Tensor::new(vec![3.0, 4.0, 1.0, 2.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let eval = evaluate(Value::Tensor(tensor), &[]).expect("evaluate");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), [1, 3]);
                assert_eq!(t.materialize_f64(), vec![3.0, 1.0, 5.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        match indices {
            Value::Tensor(t) => {
                assert_eq!(t.materialize_f64(), vec![1.0, 1.0, 1.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn min_all_linear_index() {
        let tensor =
            Tensor::new((1..=12).map(|v| v as f64).collect::<Vec<_>>(), vec![3, 4]).unwrap();
        let args = vec![placeholder(), Value::from("all")];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("evaluate");
        let (values, indices) = eval.into_pair();
        assert_eq!(values, Value::Num(1.0));
        assert_eq!(indices, Value::Num(1.0));

        let args_linear = vec![placeholder(), Value::from("linear")];
        let eval = evaluate(
            Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap()),
            &args_linear,
        )
        .expect("evaluate");
        let (values, indices) = eval.into_pair();
        assert_eq!(values, Value::Num(2.0));
        assert_eq!(indices, Value::Num(1.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn min_with_omitnan() {
        let tensor = Tensor::new(vec![f64::NAN, 4.0, 2.0], vec![3, 1]).unwrap();
        let args = vec![placeholder(), Value::from("omitnan")];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("evaluate");
        let (values, indices) = eval.into_pair();
        assert_eq!(values, Value::Num(2.0));
        assert_eq!(indices, Value::Num(3.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn min_omitnan_all_nan_slice() {
        let tensor = Tensor::new(vec![f64::NAN, f64::NAN], vec![2, 1]).unwrap();
        let args = vec![placeholder(), Value::from("omitnan")];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("evaluate");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Num(v) => assert!(v.is_nan()),
            other => panic!("expected scalar NaN, got {other:?}"),
        }
        match indices {
            Value::Num(v) => assert!(v.is_nan()),
            other => panic!("expected scalar NaN index, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn min_reduction_abs_comparison() {
        let tensor = Tensor::new(vec![1.0, -3.0, -2.0, 4.0], vec![2, 2]).unwrap();
        let args = vec![
            placeholder(),
            Value::from("ComparisonMethod"),
            Value::from("abs"),
        ];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("evaluate");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), [1, 2]);
                assert_eq!(t.materialize_f64(), vec![1.0, -2.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
        match indices {
            Value::Tensor(t) => {
                assert_eq!(t.materialize_f64(), vec![1.0, 1.0]);
            }
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn min_reduction_complex_real_comparison() {
        let tensor = ComplexTensor::new(vec![(1.0, 2.0), (0.5, 5.0)], vec![2, 1]).expect("tensor");
        let args = vec![
            placeholder(),
            Value::from("ComparisonMethod"),
            Value::from("real"),
        ];
        let eval = evaluate(Value::ComplexTensor(tensor), &args).expect("evaluate");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Complex(re, im) => {
                assert!((re - 0.5).abs() < 1e-12);
                assert!((im - 5.0).abs() < 1e-12);
            }
            other => panic!("expected complex scalar, got {other:?}"),
        }
        assert_eq!(indices, Value::Num(2.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn min_reduction_preserves_native_complex_single_storage() {
        let tensor =
            ComplexTensor::from_f32(vec![(1.0, 2.0), (0.5, 5.0)], vec![2, 1]).expect("tensor");
        let values = evaluate(Value::ComplexTensor(tensor), &[])
            .expect("evaluate")
            .into_value();
        let Value::ComplexTensor(values) = values else {
            panic!("expected typed complex scalar tensor");
        };
        assert_eq!(
            values.into_complex_storage(),
            ComplexStorage::F32(vec![(1.0, 2.0)].into())
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn min_elementwise_broadcast() {
        let lhs = Tensor::new(vec![1.0, 4.0, 7.0], vec![1, 3]).unwrap();
        let rhs = Tensor::new(vec![2.0, 3.0, 5.0], vec![3, 1]).unwrap();
        let eval = evaluate(Value::Tensor(lhs), &[Value::Tensor(rhs)]).expect("evaluate");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), [3, 3]);
                assert_eq!(
                    [
                        t.materialize_f64()[0],
                        t.materialize_f64()[3],
                        t.materialize_f64()[6]
                    ],
                    [1.0, 2.0, 2.0]
                );
                assert_eq!(
                    [
                        t.materialize_f64()[1],
                        t.materialize_f64()[4],
                        t.materialize_f64()[7]
                    ],
                    [1.0, 3.0, 3.0]
                );
                assert_eq!(
                    [
                        t.materialize_f64()[2],
                        t.materialize_f64()[5],
                        t.materialize_f64()[8]
                    ],
                    [1.0, 4.0, 5.0]
                );
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        match indices {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), [3, 3]);
                assert_eq!(
                    [
                        t.materialize_f64()[0],
                        t.materialize_f64()[3],
                        t.materialize_f64()[6]
                    ],
                    [1.0, 2.0, 2.0]
                );
                assert_eq!(
                    [
                        t.materialize_f64()[1],
                        t.materialize_f64()[4],
                        t.materialize_f64()[7]
                    ],
                    [1.0, 2.0, 2.0]
                );
                assert_eq!(
                    [
                        t.materialize_f64()[2],
                        t.materialize_f64()[5],
                        t.materialize_f64()[8]
                    ],
                    [1.0, 1.0, 2.0]
                );
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn min_elementwise_preserves_native_single_storage() {
        let lhs = Tensor::from_f32(vec![3.5, -2.0], vec![2, 1]).expect("lhs");
        let rhs = Tensor::from_f32(vec![1.25, 4.0], vec![2, 1]).expect("rhs");
        let (values, indices) = evaluate(Value::Tensor(lhs), &[Value::Tensor(rhs)])
            .expect("evaluate")
            .into_pair();
        let Value::Tensor(values) = values else {
            panic!("expected tensor values");
        };
        assert_eq!(
            values.into_numeric_storage().expect("storage"),
            NumericStorage::F32(vec![1.25, -2.0])
        );
        assert_eq!(
            indices,
            Value::Tensor(Tensor::new(vec![2.0, 1.0], vec![2, 1]).expect("indices"))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn min_elementwise_preserves_native_complex_single_storage() {
        let lhs = ComplexTensor::from_f32(vec![(3.0, 4.0), (1.0, 0.0)], vec![2, 1]).expect("lhs");
        let rhs = ComplexTensor::from_f32(vec![(4.0, 0.0), (0.0, 2.0)], vec![2, 1]).expect("rhs");
        let values = evaluate(Value::ComplexTensor(lhs), &[Value::ComplexTensor(rhs)])
            .expect("evaluate")
            .into_value();
        let Value::ComplexTensor(values) = values else {
            panic!("expected complex tensor values");
        };
        assert_eq!(
            values.into_complex_storage(),
            ComplexStorage::F32(vec![(4.0, 0.0), (1.0, 0.0)].into())
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn min_pairwise_public_form_rejects_second_output() {
        let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let error = min_builtin(Value::Num(1.0), vec![Value::Num(2.0)])
            .expect_err("pairwise min has one output");
        assert_eq!(error.identifier(), MIN_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn min_elementwise_abs_keeps_uint64_comparison_exact() {
        let lhs = Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1])
            .expect("lhs");
        let rhs = Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_992]), vec![1, 1])
            .expect("rhs");
        let args = vec![
            Value::Tensor(rhs),
            Value::from("ComparisonMethod"),
            Value::from("abs"),
        ];
        let (values, indices) = evaluate(Value::Tensor(lhs), &args)
            .expect("evaluate")
            .into_pair();
        assert_eq!(values, Value::Int(IntValue::U64(9_007_199_254_740_992)));
        assert_eq!(indices, Value::Num(2.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn min_integer_scalar_double_compares_before_integer_conversion() {
        let lhs = Tensor::new_integer(IntegerStorage::U8(vec![2, 4]), vec![1, 2]).expect("lhs");
        let (values, indices) = evaluate(Value::Tensor(lhs), &[Value::Num(2.5)])
            .expect("evaluate")
            .into_pair();
        assert_eq!(
            values,
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U8(vec![2, 3]), vec![1, 2]).expect("values")
            )
        );
        assert_eq!(
            indices,
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("indices"))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn min_integer_scalar_double_is_exact_above_flintmax() {
        let lhs = Tensor::new_integer(
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
            vec![1, 2],
        )
        .expect("lhs");
        let values = evaluate(Value::Tensor(lhs), &[Value::Num(9_007_199_254_740_992.0)])
            .expect("evaluate")
            .into_value();
        assert_eq!(
            values,
            Value::Tensor(
                Tensor::new_integer(
                    IntegerStorage::U64(vec![9_007_199_254_740_992, 9_007_199_254_740_992]),
                    vec![1, 2],
                )
                .expect("values")
            )
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn min_integer_pairwise_rejects_incompatible_classes_and_float_shapes() {
        let integer =
            Tensor::new_integer(IntegerStorage::I16(vec![1, 2]), vec![1, 2]).expect("integer");
        let mixed =
            Tensor::new_integer(IntegerStorage::U16(vec![1, 2]), vec![1, 2]).expect("mixed");
        let error = evaluate(Value::Tensor(integer.clone()), &[Value::Tensor(mixed)])
            .expect_err("mixed integer classes reject");
        assert_eq!(error.identifier(), MIN_ERROR_INVALID_INPUT.identifier);

        let single = Tensor::from_f32(vec![1.0], vec![1, 1]).expect("single");
        let error = evaluate(Value::Tensor(integer.clone()), &[Value::Tensor(single)])
            .expect_err("scalar single rejects");
        assert_eq!(error.identifier(), MIN_ERROR_INVALID_INPUT.identifier);

        let doubles = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("doubles");
        let error = evaluate(Value::Tensor(integer), &[Value::Tensor(doubles)])
            .expect_err("nonscalar double rejects");
        assert_eq!(error.identifier(), MIN_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn min_elementwise_abs_comparison() {
        let lhs = Tensor::new(vec![-2.0, 1.0], vec![2, 1]).unwrap();
        let rhs = Tensor::new(vec![1.5, -3.0], vec![2, 1]).unwrap();
        let args = vec![
            Value::Tensor(rhs),
            Value::from("ComparisonMethod"),
            Value::from("abs"),
        ];
        let eval = evaluate(Value::Tensor(lhs), &args).expect("evaluate");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(t) => {
                assert_eq!(t.materialize_f64(), vec![1.5, 1.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        match indices {
            Value::Tensor(t) => {
                assert_eq!(t.materialize_f64(), vec![2.0, 1.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn min_elementwise_rejects_reduction_only_keywords() {
        let lhs = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let rhs = Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap();
        let err = evaluate(
            Value::Tensor(lhs),
            &[Value::Tensor(rhs), Value::from("omitnan")],
        )
        .expect_err("expected error");
        assert_eq!(err.identifier(), MIN_ERROR_INVALID_ARGUMENT.identifier);
        assert!(err.message().contains("only supported for reduction"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn min_elementwise_size_mismatch_identifier() {
        let lhs = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let rhs = Tensor::new(vec![3.0, 4.0, 5.0], vec![3, 1]).unwrap();
        let err = evaluate(Value::Tensor(lhs), &[Value::Tensor(rhs)]).expect_err("expected error");
        assert_eq!(err.identifier(), MIN_ERROR_SIZE_MISMATCH.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn min_complex_real_comparison() {
        let lhs = ComplexTensor::new(vec![(1.0, 2.0)], vec![1, 1]).unwrap();
        let rhs = ComplexTensor::new(vec![(0.5, 5.0)], vec![1, 1]).unwrap();
        let args = vec![
            Value::ComplexTensor(rhs),
            Value::from("ComparisonMethod"),
            Value::from("real"),
        ];
        let eval = evaluate(Value::ComplexTensor(lhs), &args).expect("evaluate");
        let (values, indices) = eval.into_pair();
        assert_eq!(values, Value::Complex(0.5, 5.0));
        assert_eq!(indices, Value::Num(2.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn min_dimension_argument_parsing() {
        let tensor = Tensor::new(vec![3.0, 4.0, 1.0, 2.0], vec![2, 2]).unwrap();
        let dims = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let args = vec![placeholder(), Value::Tensor(dims)];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("evaluate");
        let (values, indices) = eval.into_pair();
        assert_eq!(values, Value::Num(1.0));
        assert_eq!(indices, Value::Num(3.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn min_vecdim_duplicate_entries() {
        let tensor = Tensor::new(vec![5.0, 2.0, 7.0, 1.0], vec![2, 2]).unwrap();
        let dims = Tensor::new(vec![1.0, 1.0, 2.0], vec![3, 1]).unwrap();
        let args = vec![placeholder(), Value::Tensor(dims)];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("evaluate");
        let (values, indices) = eval.into_pair();
        assert_eq!(values, Value::Num(1.0));
        assert_eq!(indices, Value::Num(4.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn min_dimension_gpu_argument_errors() {
        let tensor = Tensor::new(vec![3.0, 1.0], vec![2, 1]).unwrap();
        let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("resident reduction admission fixture");
        let dim_handle = Value::GpuTensor(allocation.handle.clone());
        let err = evaluate(Value::Tensor(tensor), &[placeholder(), dim_handle])
            .expect_err("expected error");
        assert_eq!(err.identifier(), MIN_ERROR_INVALID_ARGUMENT.identifier);
        assert!(err
            .message()
            .contains("dimension arguments must reside on the host"));
        assert_eq!(allocation.probe.retirements(), 0);
        allocation
            .handle
            .allocation_lease()
            .validate_live()
            .unwrap();
        drop(err);
        drop(allocation.handle);
        assert_eq!(allocation.probe.retirements(), 1);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn min_invalid_comparison_method_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let args = vec![
            placeholder(),
            Value::from("ComparisonMethod"),
            Value::from("chebyshev"),
        ];
        let err = evaluate(Value::Tensor(tensor), &args).expect_err("expected error");
        assert_eq!(err.identifier(), MIN_ERROR_INVALID_ARGUMENT.identifier);
        assert!(err.message().contains("unsupported ComparisonMethod"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn min_gpu_dim1_matches_cpu() {
        let tensor = Tensor::new(vec![3.0, 1.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let eval_cpu = evaluate(Value::Tensor(tensor.clone()), &[]).expect("cpu");
        let (values_cpu, indices_cpu) = eval_cpu.into_pair();

        test_support::with_test_provider(|provider| {
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let eval_gpu = evaluate(Value::GpuTensor(handle), &[]).expect("gpu");
            let (values_gpu, indices_gpu) = eval_gpu.into_pair();
            match (&values_gpu, &indices_gpu) {
                (Value::GpuTensor(_), Value::GpuTensor(_)) => {}
                other => panic!("expected GPU tensors, got {other:?}"),
            }
            let gathered_vals = test_support::gather(values_gpu).expect("gather values");
            let gathered_idx = test_support::gather(indices_gpu).expect("gather indices");
            let expected_vals = match values_cpu {
                Value::Tensor(t) => t,
                other => panic!("expected tensor values from cpu eval, got {other:?}"),
            };
            let expected_idx = match indices_cpu {
                Value::Tensor(t) => t,
                other => panic!("expected tensor indices from cpu eval, got {other:?}"),
            };
            assert_eq!(gathered_vals.shape(), expected_vals.shape());
            assert_eq!(
                gathered_vals.materialize_f64(),
                expected_vals.materialize_f64()
            );
            assert_eq!(gathered_idx.shape(), expected_idx.shape());
            assert_eq!(
                gathered_idx.materialize_f64(),
                expected_idx.materialize_f64()
            );
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn min_gpu_uint64_reduction_matches_cpu_and_preserves_residency() {
        let data = [1_u64 << 63, 1_u64 << 63, 7, 7];
        let tensor = Tensor::new_integer(IntegerStorage::U64(data.to_vec()), vec![2, 2])
            .expect("uint64 tensor");
        let (values_cpu, indices_cpu) = evaluate(Value::Tensor(tensor), &[])
            .expect("cpu min")
            .into_pair();
        assert_eq!(
            values_cpu,
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U64(vec![1_u64 << 63, 7]), vec![1, 2],)
                    .expect("expected values")
            )
        );
        assert_eq!(
            indices_cpu,
            Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).expect("expected indices"))
        );

        test_support::with_test_provider(|provider| {
            let handle = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&data),
                    shape: &[2, 2],
                })
                .expect("upload exact uint64");
            let (values_gpu, indices_gpu) = evaluate(Value::GpuTensor(handle), &[])
                .expect("gpu min")
                .into_pair();
            let Value::GpuTensor(values_handle) = &values_gpu else {
                panic!("expected resident uint64 values, got {values_gpu:?}");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(values_handle),
                Some(IntegerElementType::U64)
            );
            assert!(matches!(indices_gpu, Value::GpuTensor(_)));
            assert_eq!(
                test_support::gather(values_gpu).expect("gather values"),
                match values_cpu {
                    Value::Tensor(values) => values,
                    other => panic!("expected CPU tensor values, got {other:?}"),
                }
            );
            assert_eq!(
                test_support::gather(indices_gpu).expect("gather indices"),
                match indices_cpu {
                    Value::Tensor(indices) => indices,
                    other => panic!("expected CPU tensor indices, got {other:?}"),
                }
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn min_dimension_numeric_argument() {
        let tensor = Tensor::new(vec![3.0, 4.0, 1.0, 2.0], vec![2, 2]).unwrap();
        let args = vec![placeholder(), Value::Num(2.0)];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("evaluate");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), [2, 1]);
                assert_eq!(t.materialize_f64(), vec![1.0, 2.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        match indices {
            Value::Tensor(t) => {
                assert_eq!(t.materialize_f64(), vec![2.0, 2.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn min_complex_auto_comparison() {
        let lhs = ComplexTensor::new(vec![(1.0, 2.0)], vec![1, 1]).unwrap();
        let rhs = ComplexTensor::new(vec![(2.0, 1.0)], vec![1, 1]).unwrap();
        let eval =
            evaluate(Value::ComplexTensor(lhs), &[Value::ComplexTensor(rhs)]).expect("evaluate");
        let (values, indices) = eval.into_pair();
        assert_eq!(values, Value::Complex(2.0, 1.0));
        assert_eq!(indices, Value::Num(2.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn min_scalar_pair_arguments() {
        let args = vec![Value::Num(2.0)];
        let result = min_builtin(Value::Num(3.0), args).expect("min");
        assert_eq!(result, Value::Num(2.0));
    }

    #[test]
    fn min_scalar_complex_value_reads_typed_integer_complex_storage_without_mirror() {
        let storage =
            IntegerComplexStorage::new(IntegerStorage::I16(vec![5]), IntegerStorage::I16(vec![-1]))
                .expect("complex integer storage");
        let tensor = ComplexTensor::new_integer(storage, vec![1, 1]).expect("complex tensor");

        assert_eq!(
            scalar_complex_value(&Value::ComplexTensor(tensor)),
            Some((5.0, -1.0))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn min_rejects_invalid_dimension() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let args = vec![placeholder(), Value::Int(IntValue::I32(0))];
        let err = evaluate(Value::Tensor(tensor), &args).expect_err("expected error");
        assert!(err.message().contains("dimension must be >= 1"));
    }

    #[test]
    fn min_scalar_dimension_diagnostics_follow_typed_decode_reasons() {
        let invalid = [
            (Value::Num(f64::NAN), "dimension must be finite"),
            (Value::Num(1.25), "dimension must be integral"),
            (Value::Num(0.0), "dimension must be >= 1"),
            (
                Value::Int(IntValue::I64(-1)),
                "dimension is outside the supported range",
            ),
        ];
        for (dimension, expected) in invalid {
            let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
            let error = evaluate(Value::Tensor(input), &[placeholder(), dimension])
                .expect_err("invalid scalar dimension must fail");
            assert_eq!(error.identifier(), MIN_ERROR_INVALID_ARGUMENT.identifier);
            assert!(error.message().contains(expected), "{error}");
        }
    }
}
