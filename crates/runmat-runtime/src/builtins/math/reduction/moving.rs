//! MATLAB-compatible moving-window numeric reductions.

use std::cmp::Ordering;

use runmat_accelerate_api::{
    GpuTensorHandle, GpuTensorStorage, ProviderMovingWindowEndpoints, ProviderMovingWindowOp,
    ProviderMovingWindowRequest, ProviderNanMode, ProviderStdNormalization,
};
use runmat_builtins::{
    BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    MOVING_WINDOW_ERROR_INTERNAL as ERROR_INTERNAL,
    MOVING_WINDOW_ERROR_INVALID_ARGUMENT as ERROR_INVALID_ARGUMENT,
    MOVING_WINDOW_ERROR_INVALID_INPUT as ERROR_INVALID_INPUT,
};
use runmat_macros::runtime_builtin;
use runmat_value::{
    ComplexTensor, IntValue, IntegerStorage, NumericScalar, NumericStorage, Tensor, Value,
};

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

mod admission;
mod arguments;
mod complex;
mod dispatch;
mod keywords;
mod provider;
mod provider_shape;

pub mod movsum {
    use super::*;
    pub use runmat_builtins::{
        MOVSUM_DESCRIPTOR as DESCRIPTOR, MOVSUM_ERROR_TOO_MANY_OUTPUTS as ERROR_TOO_MANY_OUTPUTS,
        MOVSUM_GPU_SAMPLE_POINTS_EXTENSION as GPU_SAMPLE_POINTS_EXTENSION,
    };
    #[runtime_builtin(
        name = "movsum",
        binding_variant = "default",
        context = true,
        builtin_path = "crate::builtins::math::reduction::moving::movsum"
    )]
    pub(crate) async fn movsum_builtin(
        runtime: Option<crate::context::RuntimeContext>,
        args: crate::call::BuiltinArguments,
    ) -> BuiltinResult<Value> {
        crate::output_count::validate_fixed_outputs(
            &DESCRIPTOR,
            &ERROR_TOO_MANY_OUTPUTS,
            "movsum",
        )?;
        super::dispatch::moving_builtin(runtime.as_ref(), "movsum", MovingOp::Sum, args).await
    }
}

pub mod movmean {
    use super::*;
    pub use runmat_builtins::{
        MOVMEAN_DESCRIPTOR as DESCRIPTOR, MOVMEAN_ERROR_TOO_MANY_OUTPUTS as ERROR_TOO_MANY_OUTPUTS,
        MOVMEAN_GPU_SAMPLE_POINTS_EXTENSION as GPU_SAMPLE_POINTS_EXTENSION,
    };
    #[runtime_builtin(
        name = "movmean",
        binding_variant = "default",
        context = true,
        builtin_path = "crate::builtins::math::reduction::moving::movmean"
    )]
    pub(crate) async fn movmean_builtin(
        runtime: Option<crate::context::RuntimeContext>,
        args: crate::call::BuiltinArguments,
    ) -> BuiltinResult<Value> {
        crate::output_count::validate_fixed_outputs(
            &DESCRIPTOR,
            &ERROR_TOO_MANY_OUTPUTS,
            "movmean",
        )?;
        super::dispatch::moving_builtin(runtime.as_ref(), "movmean", MovingOp::Mean, args).await
    }
}

pub mod movprod {
    use super::*;
    pub use runmat_builtins::{
        MOVPROD_DESCRIPTOR as DESCRIPTOR, MOVPROD_ERROR_TOO_MANY_OUTPUTS as ERROR_TOO_MANY_OUTPUTS,
        MOVPROD_GPU_SAMPLE_POINTS_EXTENSION as GPU_SAMPLE_POINTS_EXTENSION,
    };
    #[runtime_builtin(
        name = "movprod",
        binding_variant = "default",
        context = true,
        builtin_path = "crate::builtins::math::reduction::moving::movprod"
    )]
    pub(crate) async fn movprod_builtin(
        runtime: Option<crate::context::RuntimeContext>,
        args: crate::call::BuiltinArguments,
    ) -> BuiltinResult<Value> {
        crate::output_count::validate_fixed_outputs(
            &DESCRIPTOR,
            &ERROR_TOO_MANY_OUTPUTS,
            "movprod",
        )?;
        super::dispatch::moving_builtin(runtime.as_ref(), "movprod", MovingOp::Prod, args).await
    }
}

pub mod movmin {
    use super::*;
    pub use runmat_builtins::{
        MOVMIN_DESCRIPTOR as DESCRIPTOR, MOVMIN_ERROR_TOO_MANY_OUTPUTS as ERROR_TOO_MANY_OUTPUTS,
        MOVMIN_GPU_SAMPLE_POINTS_EXTENSION as GPU_SAMPLE_POINTS_EXTENSION,
    };
    #[runtime_builtin(
        name = "movmin",
        binding_variant = "default",
        context = true,
        builtin_path = "crate::builtins::math::reduction::moving::movmin"
    )]
    pub(crate) async fn movmin_builtin(
        runtime: Option<crate::context::RuntimeContext>,
        args: crate::call::BuiltinArguments,
    ) -> BuiltinResult<Value> {
        crate::output_count::validate_fixed_outputs(
            &DESCRIPTOR,
            &ERROR_TOO_MANY_OUTPUTS,
            "movmin",
        )?;
        super::dispatch::moving_builtin(runtime.as_ref(), "movmin", MovingOp::Min, args).await
    }
}

pub mod movmax {
    use super::*;
    pub use runmat_builtins::{
        MOVMAX_DESCRIPTOR as DESCRIPTOR, MOVMAX_ERROR_TOO_MANY_OUTPUTS as ERROR_TOO_MANY_OUTPUTS,
        MOVMAX_GPU_SAMPLE_POINTS_EXTENSION as GPU_SAMPLE_POINTS_EXTENSION,
    };
    #[runtime_builtin(
        name = "movmax",
        binding_variant = "default",
        context = true,
        builtin_path = "crate::builtins::math::reduction::moving::movmax"
    )]
    pub(crate) async fn movmax_builtin(
        runtime: Option<crate::context::RuntimeContext>,
        args: crate::call::BuiltinArguments,
    ) -> BuiltinResult<Value> {
        crate::output_count::validate_fixed_outputs(
            &DESCRIPTOR,
            &ERROR_TOO_MANY_OUTPUTS,
            "movmax",
        )?;
        super::dispatch::moving_builtin(runtime.as_ref(), "movmax", MovingOp::Max, args).await
    }
}

pub mod movmedian {
    use super::*;
    pub use runmat_builtins::{
        MOVMEDIAN_DESCRIPTOR as DESCRIPTOR,
        MOVMEDIAN_ERROR_TOO_MANY_OUTPUTS as ERROR_TOO_MANY_OUTPUTS,
        MOVMEDIAN_GPU_LARGE_WINDOW_EXTENSION as GPU_LARGE_WINDOW_EXTENSION,
        MOVMEDIAN_GPU_SAMPLE_POINTS_EXTENSION as GPU_SAMPLE_POINTS_EXTENSION,
    };
    #[runtime_builtin(
        name = "movmedian",
        binding_variant = "default",
        context = true,
        builtin_path = "crate::builtins::math::reduction::moving::movmedian"
    )]
    pub(crate) async fn movmedian_builtin(
        runtime: Option<crate::context::RuntimeContext>,
        args: crate::call::BuiltinArguments,
    ) -> BuiltinResult<Value> {
        crate::output_count::validate_fixed_outputs(
            &DESCRIPTOR,
            &ERROR_TOO_MANY_OUTPUTS,
            "movmedian",
        )?;
        super::dispatch::moving_builtin(runtime.as_ref(), "movmedian", MovingOp::Median, args).await
    }
}

pub mod movstd {
    use super::*;
    pub use runmat_builtins::{
        MOVSTD_DESCRIPTOR as DESCRIPTOR, MOVSTD_ERROR_TOO_MANY_OUTPUTS as ERROR_TOO_MANY_OUTPUTS,
        MOVSTD_GPU_SAMPLE_POINTS_EXTENSION as GPU_SAMPLE_POINTS_EXTENSION,
        MOVSTD_TYPED_INTEGER_CONTROL_EXTENSION as TYPED_INTEGER_CONTROL_EXTENSION,
    };
    #[runtime_builtin(
        name = "movstd",
        binding_variant = "default",
        context = true,
        builtin_path = "crate::builtins::math::reduction::moving::movstd"
    )]
    pub(crate) async fn movstd_builtin(
        runtime: Option<crate::context::RuntimeContext>,
        args: crate::call::BuiltinArguments,
    ) -> BuiltinResult<Value> {
        crate::output_count::validate_fixed_outputs(
            &DESCRIPTOR,
            &ERROR_TOO_MANY_OUTPUTS,
            "movstd",
        )?;
        super::dispatch::moving_builtin(runtime.as_ref(), "movstd", MovingOp::Std, args).await
    }
}

pub mod movvar {
    use super::*;
    pub use runmat_builtins::{
        MOVVAR_DESCRIPTOR as DESCRIPTOR, MOVVAR_ERROR_TOO_MANY_OUTPUTS as ERROR_TOO_MANY_OUTPUTS,
        MOVVAR_GPU_SAMPLE_POINTS_EXTENSION as GPU_SAMPLE_POINTS_EXTENSION,
        MOVVAR_TYPED_INTEGER_CONTROL_EXTENSION as TYPED_INTEGER_CONTROL_EXTENSION,
    };
    #[runtime_builtin(
        name = "movvar",
        binding_variant = "default",
        context = true,
        builtin_path = "crate::builtins::math::reduction::moving::movvar"
    )]
    pub(crate) async fn movvar_builtin(
        runtime: Option<crate::context::RuntimeContext>,
        args: crate::call::BuiltinArguments,
    ) -> BuiltinResult<Value> {
        crate::output_count::validate_fixed_outputs(
            &DESCRIPTOR,
            &ERROR_TOO_MANY_OUTPUTS,
            "movvar",
        )?;
        super::dispatch::moving_builtin(runtime.as_ref(), "movvar", MovingOp::Var, args).await
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MovingOp {
    Sum,
    Mean,
    Prod,
    Min,
    Max,
    Median,
    Std,
    Var,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NanMode {
    Include,
    Omit,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Endpoints {
    Shrink,
    Discard,
    FillDefault,
    Fill(NumericScalar),
}

#[derive(Clone, Debug)]
struct ParsedMoving {
    window: WindowSpec,
    dim: Option<usize>,
    nan_mode: NanMode,
    endpoints: Endpoints,
    sample_points: Option<Vec<f64>>,
    normalization: VarianceNormalization,
}

#[cfg(test)]
#[path = "moving/owning_tests.rs"]
mod owning_tests;

#[cfg(test)]
#[path = "moving/admission_tests.rs"]
mod admission_tests;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VarianceNormalization {
    Sample,
    Population,
}

#[derive(Clone, Copy, Debug)]
enum WindowSpec {
    Counts { before: usize, after: usize },
    Span { before: f64, after: f64 },
}

#[derive(Clone, Copy, Debug)]
struct CountWindow {
    start: usize,
    end: usize,
    fill_count: usize,
}

impl WindowSpec {
    fn count_radius(self) -> Option<(usize, usize)> {
        match self {
            WindowSpec::Counts { before, after } => Some((before, after)),
            WindowSpec::Span { .. } => None,
        }
    }

    fn checked_count_len(self, name: &'static str) -> BuiltinResult<Option<usize>> {
        match self {
            WindowSpec::Counts { before, after } => before
                .checked_add(after)
                .and_then(|sum| sum.checked_add(1))
                .map(Some)
                .ok_or_else(|| invalid_argument(name, format!("{name}: window length overflow"))),
            WindowSpec::Span { .. } => Ok(None),
        }
    }
}

impl MovingOp {
    fn default_nan_mode(self) -> NanMode {
        match self {
            MovingOp::Min | MovingOp::Max => NanMode::Omit,
            MovingOp::Sum
            | MovingOp::Mean
            | MovingOp::Prod
            | MovingOp::Median
            | MovingOp::Std
            | MovingOp::Var => NanMode::Include,
        }
    }

    fn default_fill(self) -> f64 {
        match self {
            MovingOp::Min => f64::INFINITY,
            MovingOp::Max => f64::NEG_INFINITY,
            MovingOp::Sum
            | MovingOp::Mean
            | MovingOp::Prod
            | MovingOp::Median
            | MovingOp::Std
            | MovingOp::Var => f64::NAN,
        }
    }

    fn uses_variance_normalization(self) -> bool {
        matches!(self, MovingOp::Std | MovingOp::Var)
    }

    fn provider_op(self) -> Option<ProviderMovingWindowOp> {
        match self {
            MovingOp::Sum => Some(ProviderMovingWindowOp::Sum),
            MovingOp::Mean => Some(ProviderMovingWindowOp::Mean),
            MovingOp::Prod => Some(ProviderMovingWindowOp::Prod),
            MovingOp::Min => Some(ProviderMovingWindowOp::Min),
            MovingOp::Max => Some(ProviderMovingWindowOp::Max),
            MovingOp::Median => Some(ProviderMovingWindowOp::Median),
            MovingOp::Std => Some(ProviderMovingWindowOp::Std),
            MovingOp::Var => Some(ProviderMovingWindowOp::Var),
        }
    }
}

fn is_real_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(
            value,
            Value::GpuTensor(handle)
                if runmat_accelerate_api::handle_integer_type(handle).is_some()
        )
}

fn stdvar_typed_integer_control_extension(op: MovingOp) -> &'static BuiltinExtensionDescriptor {
    match op {
        MovingOp::Std => &movstd::TYPED_INTEGER_CONTROL_EXTENSION,
        MovingOp::Var => &movvar::TYPED_INTEGER_CONTROL_EXTENSION,
        _ => unreachable!("only moving dispersion operations have typed integer controls"),
    }
}

fn gpu_sample_points_extension(op: MovingOp) -> &'static BuiltinExtensionDescriptor {
    match op {
        MovingOp::Sum => &movsum::GPU_SAMPLE_POINTS_EXTENSION,
        MovingOp::Mean => &movmean::GPU_SAMPLE_POINTS_EXTENSION,
        MovingOp::Prod => &movprod::GPU_SAMPLE_POINTS_EXTENSION,
        MovingOp::Min => &movmin::GPU_SAMPLE_POINTS_EXTENSION,
        MovingOp::Max => &movmax::GPU_SAMPLE_POINTS_EXTENSION,
        MovingOp::Median => &movmedian::GPU_SAMPLE_POINTS_EXTENSION,
        MovingOp::Std => &movstd::GPU_SAMPLE_POINTS_EXTENSION,
        MovingOp::Var => &movvar::GPU_SAMPLE_POINTS_EXTENSION,
    }
}

fn provider_endpoints(op: MovingOp, endpoints: Endpoints) -> ProviderMovingWindowEndpoints {
    match endpoints {
        Endpoints::Shrink => ProviderMovingWindowEndpoints::Shrink,
        Endpoints::Discard => ProviderMovingWindowEndpoints::Discard,
        Endpoints::FillDefault => ProviderMovingWindowEndpoints::Fill(op.default_fill()),
        Endpoints::Fill(value) => ProviderMovingWindowEndpoints::Fill(value.materialize_f64()),
    }
}

fn provider_nan_mode(mode: NanMode) -> ProviderNanMode {
    match mode {
        NanMode::Include => ProviderNanMode::Include,
        NanMode::Omit => ProviderNanMode::Omit,
    }
}

fn provider_normalization(normalization: VarianceNormalization) -> ProviderStdNormalization {
    match normalization {
        VarianceNormalization::Sample => ProviderStdNormalization::Sample,
        VarianceNormalization::Population => ProviderStdNormalization::Population,
    }
}

fn moving_real(
    name: &'static str,
    op: MovingOp,
    tensor: Tensor,
    parsed: &ParsedMoving,
) -> BuiltinResult<Tensor> {
    let dim = parsed
        .dim
        .unwrap_or_else(|| default_dimension(tensor.shape()));
    if dim == 0 {
        return Err(invalid_argument(
            name,
            format!("{name}: dimension must be >= 1"),
        ));
    }
    let input_shape = shape_with_trailing_singletons(tensor.shape(), dim);
    let shape = output_shape(name, &input_shape, dim, parsed)?;
    let count = checked_element_count(name, &shape)?;
    let dim_index = dim - 1;
    let axis_len = input_shape[dim_index];
    validate_sample_points_len(name, axis_len, parsed)?;
    let stride_before = checked_product(name, &input_shape[..dim_index])?;
    let stride_after = checked_product(name, &input_shape[dim..])?;
    let out_axis = shape[dim_index];
    let dtype = tensor.numeric_dtype();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|error| internal(name, error))?;
    let (input, output_is_double) = match storage.into_integer_storage() {
        Ok(integer) if matches!(op, MovingOp::Min | MovingOp::Max | MovingOp::Median) => {
            return moving_integer_exact(name, op, integer, input_shape, shape, dim, parsed);
        }
        Ok(integer) => (
            NumericStorage::from_integer_storage(integer).materialize_f64(),
            true,
        ),
        Err(floating) => (floating.materialize_f64(), false),
    };
    let mut output = vec![0.0; count];

    for after in 0..stride_after {
        for out_pos in 0..out_axis {
            let center = source_center(name, out_pos, parsed)?;
            let sample_positions = sample_window_positions(name, center, axis_len, parsed)?;
            let count_window = if sample_positions.is_none() {
                Some(count_window(name, center, axis_len, parsed)?)
            } else {
                None
            };
            let capacity = sample_positions
                .as_ref()
                .map(Vec::len)
                .or_else(|| count_window.map(|window| window.end - window.start))
                .unwrap_or(0);
            let mut values = Vec::with_capacity(capacity);
            for before in 0..stride_before {
                let out_idx = before + out_pos * stride_before + after * stride_before * out_axis;
                values.clear();
                let mut saw_nan = false;
                if let Some(positions) = sample_positions.as_ref() {
                    for &pos in positions {
                        let idx = before + pos * stride_before + after * stride_before * axis_len;
                        let value = input[idx];
                        if value.is_nan() {
                            if parsed.nan_mode == NanMode::Include {
                                saw_nan = true;
                                break;
                            }
                            continue;
                        }
                        values.push(value);
                    }
                    output[out_idx] =
                        reduce_real_window(op, &values, saw_nan, parsed.normalization);
                    continue;
                }

                let window = count_window.expect("count window computed");
                for pos in window.start..window.end {
                    let idx = before + pos * stride_before + after * stride_before * axis_len;
                    let value = input[idx];
                    if value.is_nan() {
                        if parsed.nan_mode == NanMode::Include {
                            saw_nan = true;
                            break;
                        }
                        continue;
                    }
                    values.push(value);
                }
                output[out_idx] = reduce_real_window_with_fill(
                    op,
                    &values,
                    saw_nan,
                    fill_value(op, parsed, window.fill_count),
                    parsed.nan_mode,
                    parsed.normalization,
                );
            }
        }
    }

    if output_is_double {
        Tensor::new(output, shape).map_err(|err| internal(name, err))
    } else {
        Tensor::new_with_dtype(output, shape, dtype).map_err(|err| internal(name, err))
    }
}

fn moving_integer_exact(
    name: &'static str,
    op: MovingOp,
    storage: IntegerStorage,
    input_shape: Vec<usize>,
    shape: Vec<usize>,
    dim: usize,
    parsed: &ParsedMoving,
) -> BuiltinResult<Tensor> {
    debug_assert!(matches!(
        op,
        MovingOp::Min | MovingOp::Max | MovingOp::Median
    ));
    let count = checked_element_count(name, &shape)?;
    let dim_index = dim - 1;
    let axis_len = input_shape[dim_index];
    let stride_before = checked_product(name, &input_shape[..dim_index])?;
    let stride_after = checked_product(name, &input_shape[dim..])?;
    let out_axis = shape[dim_index];
    let mut output = Vec::with_capacity(count);

    for after in 0..stride_after {
        for out_pos in 0..out_axis {
            let center = source_center(name, out_pos, parsed)?;
            let sample_positions = sample_window_positions(name, center, axis_len, parsed)?;
            let count_window = if sample_positions.is_none() {
                Some(count_window(name, center, axis_len, parsed)?)
            } else {
                None
            };
            let capacity = sample_positions
                .as_ref()
                .map(Vec::len)
                .or_else(|| count_window.map(|window| window.end - window.start))
                .unwrap_or(0);
            let fill = count_window
                .and_then(|window| integer_fill_value(&storage, op, parsed, window.fill_count));
            for before in 0..stride_before {
                let mut values = Vec::with_capacity(capacity);
                if let Some(positions) = sample_positions.as_ref() {
                    for &pos in positions {
                        let index = before + pos * stride_before + after * stride_before * axis_len;
                        values.push(
                            storage
                                .value_at(index)
                                .expect("validated integer moving-window index"),
                        );
                    }
                } else {
                    let window = count_window.expect("count window computed");
                    for pos in window.start..window.end {
                        let index = before + pos * stride_before + after * stride_before * axis_len;
                        values.push(
                            storage
                                .value_at(index)
                                .expect("validated integer moving-window index"),
                        );
                    }
                }
                output.push(reduce_integer_window(op, &values, fill.as_ref()));
            }
        }
    }

    Tensor::new_integer(
        storage
            .from_same_class_values(output)
            .map_err(|error| internal(name, error))?,
        shape,
    )
    .map_err(|error| internal(name, error))
}

fn integer_fill_value(
    storage: &IntegerStorage,
    op: MovingOp,
    parsed: &ParsedMoving,
    count: usize,
) -> Option<(IntValue, usize)> {
    if count == 0 {
        return None;
    }
    match parsed.endpoints {
        Endpoints::FillDefault => Some((storage.cast_f64_assignment(op.default_fill()), count)),
        Endpoints::Fill(value) => Some((
            match value.into_int_value() {
                Some(integer) => storage.cast_exact_assignment(&integer),
                None => storage.cast_f64_assignment(value.materialize_f64()),
            },
            count,
        )),
        Endpoints::Shrink | Endpoints::Discard => None,
    }
}

fn reduce_integer_window(
    op: MovingOp,
    values: &[IntValue],
    fill: Option<&(IntValue, usize)>,
) -> IntValue {
    match op {
        MovingOp::Min => values
            .iter()
            .chain(fill.map(|(value, _)| value))
            .min_by(|left, right| super::median::compare_same_class_integer(left, right))
            .expect("integer moving minimum window is nonempty")
            .clone(),
        MovingOp::Max => values
            .iter()
            .chain(fill.map(|(value, _)| value))
            .max_by(|left, right| super::median::compare_same_class_integer(left, right))
            .expect("integer moving maximum window is nonempty")
            .clone(),
        MovingOp::Median => integer_median_with_fill(values, fill),
        _ => unreachable!("exact integer moving path only serves min, max, and median"),
    }
}

fn integer_median_with_fill(values: &[IntValue], fill: Option<&(IntValue, usize)>) -> IntValue {
    let mut sorted = values.to_vec();
    sorted.sort_by(super::median::compare_same_class_integer);
    let Some((fill, fill_count)) = fill else {
        return super::median::integer_median_from_sorted(&sorted);
    };
    if *fill_count == 0 {
        return super::median::integer_median_from_sorted(&sorted);
    }
    let total = sorted.len() + *fill_count;
    let middle = total / 2;
    if total % 2 == 1 {
        return kth_integer_with_repeated_fill(&sorted, fill, *fill_count, middle);
    }
    let lower = kth_integer_with_repeated_fill(&sorted, fill, *fill_count, middle - 1);
    let upper = kth_integer_with_repeated_fill(&sorted, fill, *fill_count, middle);
    super::median::integer_median_from_sorted(&[lower, upper])
}

fn kth_integer_with_repeated_fill(
    sorted: &[IntValue],
    fill: &IntValue,
    fill_count: usize,
    index: usize,
) -> IntValue {
    let less = sorted.partition_point(|value| {
        super::median::compare_same_class_integer(value, fill) == Ordering::Less
    });
    let equal = sorted[less..].partition_point(|value| {
        super::median::compare_same_class_integer(value, fill) == Ordering::Equal
    });
    if index < less {
        sorted[index].clone()
    } else if index < less + equal + fill_count {
        fill.clone()
    } else {
        sorted[index - fill_count].clone()
    }
}

fn complex_tensor_into_value(tensor: ComplexTensor) -> Value {
    if tensor::is_scalar_complex_tensor(&tensor) && tensor.integer_storage().is_none() {
        let value = tensor::complex_tensor_value_complex64(&tensor, 0);
        Value::Complex(value.re, value.im)
    } else {
        Value::ComplexTensor(tensor)
    }
}

fn reduce_real_window(
    op: MovingOp,
    values: &[f64],
    saw_nan: bool,
    normalization: VarianceNormalization,
) -> f64 {
    if saw_nan {
        return f64::NAN;
    }
    if values.is_empty() {
        return match op {
            MovingOp::Sum => 0.0,
            MovingOp::Prod => 1.0,
            MovingOp::Mean
            | MovingOp::Min
            | MovingOp::Max
            | MovingOp::Median
            | MovingOp::Std
            | MovingOp::Var => f64::NAN,
        };
    }
    match op {
        MovingOp::Sum => values.iter().sum(),
        MovingOp::Mean => values.iter().sum::<f64>() / values.len() as f64,
        MovingOp::Prod => values.iter().product(),
        MovingOp::Min => values.iter().copied().fold(f64::INFINITY, f64::min),
        MovingOp::Max => values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        MovingOp::Median => median(values),
        MovingOp::Var => variance_real(values, normalization),
        MovingOp::Std => variance_real(values, normalization).sqrt(),
    }
}

fn reduce_real_window_with_fill(
    op: MovingOp,
    values: &[f64],
    saw_nan: bool,
    fill: Option<(f64, usize)>,
    nan_mode: NanMode,
    normalization: VarianceNormalization,
) -> f64 {
    let Some((fill, fill_count)) = fill else {
        return reduce_real_window(op, values, saw_nan, normalization);
    };
    if fill_count == 0 {
        return reduce_real_window(op, values, saw_nan, normalization);
    }
    if saw_nan || (fill.is_nan() && nan_mode == NanMode::Include) {
        return f64::NAN;
    }
    if fill.is_nan() && nan_mode == NanMode::Omit {
        return reduce_real_window(op, values, false, normalization);
    }
    match op {
        MovingOp::Sum => values.iter().sum::<f64>() + fill * fill_count as f64,
        MovingOp::Mean => {
            (values.iter().sum::<f64>() + fill * fill_count as f64)
                / (values.len() + fill_count) as f64
        }
        MovingOp::Prod => values.iter().product::<f64>() * fill.powf(fill_count as f64),
        MovingOp::Min => values
            .iter()
            .copied()
            .chain(std::iter::once(fill))
            .fold(f64::INFINITY, f64::min),
        MovingOp::Max => values
            .iter()
            .copied()
            .chain(std::iter::once(fill))
            .fold(f64::NEG_INFINITY, f64::max),
        MovingOp::Median => median_with_fill(values, fill, fill_count),
        MovingOp::Var => variance_real_with_fill(values, fill, fill_count, normalization),
        MovingOp::Std => variance_real_with_fill(values, fill, fill_count, normalization).sqrt(),
    }
}

fn variance_denominator(n: usize, normalization: VarianceNormalization) -> f64 {
    match normalization {
        VarianceNormalization::Population => n as f64,
        VarianceNormalization::Sample if n > 1 => (n - 1) as f64,
        VarianceNormalization::Sample => n as f64,
    }
}

fn variance_real(values: &[f64], normalization: VarianceNormalization) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut acc = RealVarianceAccumulator::default();
    for value in values {
        acc.push(*value);
    }
    acc.finish(normalization)
}

fn variance_real_with_fill(
    values: &[f64],
    fill: f64,
    fill_count: usize,
    normalization: VarianceNormalization,
) -> f64 {
    let n = values.len() + fill_count;
    if n == 0 {
        return f64::NAN;
    }
    let mut acc = RealVarianceAccumulator::default();
    for value in values {
        acc.push(*value);
    }
    acc.merge_repeated(fill, fill_count);
    acc.finish(normalization)
}

#[derive(Default)]
struct RealVarianceAccumulator {
    n: usize,
    mean: f64,
    m2: f64,
}

impl RealVarianceAccumulator {
    fn push(&mut self, value: f64) {
        self.n += 1;
        let delta = value - self.mean;
        self.mean += delta / self.n as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    fn merge_repeated(&mut self, value: f64, count: usize) {
        if count == 0 {
            return;
        }
        if self.n == 0 {
            self.n = count;
            self.mean = value;
            self.m2 = 0.0;
            return;
        }
        let total = self.n + count;
        let delta = value - self.mean;
        self.mean += delta * (count as f64 / total as f64);
        self.m2 += delta * delta * (self.n as f64 * count as f64 / total as f64);
        self.n = total;
    }

    fn finish(self, normalization: VarianceNormalization) -> f64 {
        if self.n == 0 {
            return f64::NAN;
        }
        self.m2 / variance_denominator(self.n, normalization)
    }
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    }
}

fn median_with_fill(values: &[f64], fill: f64, fill_count: usize) -> f64 {
    if fill_count == 0 {
        return median(values);
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let total = sorted.len() + fill_count;
    if total == 0 {
        return f64::NAN;
    }
    let mid = total / 2;
    if total % 2 == 1 {
        kth_with_repeated_fill(&sorted, fill, fill_count, mid)
    } else {
        (kth_with_repeated_fill(&sorted, fill, fill_count, mid - 1)
            + kth_with_repeated_fill(&sorted, fill, fill_count, mid))
            / 2.0
    }
}

fn kth_with_repeated_fill(sorted: &[f64], fill: f64, fill_count: usize, k: usize) -> f64 {
    let less = sorted.partition_point(|value| *value < fill);
    let equal = sorted[less..].partition_point(|value| *value == fill);
    if k < less {
        sorted[k]
    } else if k < less + equal + fill_count {
        fill
    } else {
        sorted[k - fill_count]
    }
}

fn source_center(
    name: &'static str,
    out_pos: usize,
    parsed: &ParsedMoving,
) -> BuiltinResult<usize> {
    match parsed.endpoints {
        Endpoints::Discard => {
            let Some((before, _)) = parsed.window.count_radius() else {
                return Ok(out_pos);
            };
            out_pos
                .checked_add(before)
                .ok_or_else(|| invalid_argument(name, format!("{name}: window index overflow")))
        }
        Endpoints::Shrink | Endpoints::FillDefault | Endpoints::Fill(_) => Ok(out_pos),
    }
}

fn fill_value(op: MovingOp, parsed: &ParsedMoving, fill_count: usize) -> Option<(f64, usize)> {
    if fill_count == 0 {
        return None;
    }
    match parsed.endpoints {
        Endpoints::FillDefault => Some((op.default_fill(), fill_count)),
        Endpoints::Fill(value) => Some((value.materialize_f64(), fill_count)),
        Endpoints::Shrink | Endpoints::Discard => None,
    }
}

fn count_window(
    name: &'static str,
    center: usize,
    axis_len: usize,
    parsed: &ParsedMoving,
) -> BuiltinResult<CountWindow> {
    let Some((before, after)) = parsed.window.count_radius() else {
        return Err(invalid_argument(
            name,
            format!("{name}: sample points are required for duration windows"),
        ));
    };
    let start = center as i128 - before as i128;
    let end = center as i128 + after as i128;
    let in_start = start.max(0).min(axis_len as i128) as usize;
    let in_end = end
        .checked_add(1)
        .ok_or_else(|| invalid_argument(name, format!("{name}: window index overflow")))?
        .max(0)
        .min(axis_len as i128) as usize;
    let fill_count = match parsed.endpoints {
        Endpoints::FillDefault | Endpoints::Fill(_) => {
            let before_missing = if start < 0 { (-start) as usize } else { 0 };
            let after_missing = if end >= axis_len as i128 {
                (end - axis_len as i128 + 1) as usize
            } else {
                0
            };
            before_missing
                .checked_add(after_missing)
                .ok_or_else(|| invalid_argument(name, format!("{name}: window length overflow")))?
        }
        Endpoints::Shrink | Endpoints::Discard => 0,
    };
    Ok(CountWindow {
        start: in_start,
        end: in_end,
        fill_count,
    })
}

fn sample_window_positions(
    name: &'static str,
    center: usize,
    axis_len: usize,
    parsed: &ParsedMoving,
) -> BuiltinResult<Option<Vec<usize>>> {
    let Some(sample_points) = parsed.sample_points.as_ref() else {
        return Ok(None);
    };
    if center >= axis_len || center >= sample_points.len() {
        return Err(invalid_argument(
            name,
            format!("{name}: sample point center is outside the selected dimension"),
        ));
    }
    let WindowSpec::Span { before, after } = parsed.window else {
        return Err(invalid_argument(
            name,
            format!("{name}: SamplePoints requires a distance window"),
        ));
    };
    let center_point = sample_points[center];
    let lower = center_point - before;
    let upper = center_point + after;
    Ok(Some(
        sample_points
            .iter()
            .enumerate()
            .filter_map(|(idx, point)| {
                if *point >= lower && *point <= upper {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect(),
    ))
}

fn shape_with_trailing_singletons(shape: &[usize], dim: usize) -> Vec<usize> {
    let mut out = shape.to_vec();
    if dim > out.len() {
        out.resize(dim, 1);
    }
    out
}

fn validate_sample_points_len(
    name: &'static str,
    axis_len: usize,
    parsed: &ParsedMoving,
) -> BuiltinResult<()> {
    if let Some(points) = parsed.sample_points.as_ref() {
        if points.len() != axis_len {
            return Err(invalid_argument(
                name,
                format!("{name}: SamplePoints length must match the selected dimension length"),
            ));
        }
    }
    Ok(())
}

fn output_shape(
    name: &'static str,
    input_shape: &[usize],
    dim: usize,
    parsed: &ParsedMoving,
) -> BuiltinResult<Vec<usize>> {
    let mut shape = input_shape.to_vec();
    if dim == 0 {
        return Ok(shape);
    }
    if matches!(parsed.endpoints, Endpoints::Discard) {
        let Some((before, after)) = parsed.window.count_radius() else {
            return Err(invalid_argument(
                name,
                format!("{name}: Endpoints 'discard' is not supported with SamplePoints"),
            ));
        };
        let axis = shape[dim - 1];
        let trim = before
            .checked_add(after)
            .ok_or_else(|| invalid_argument(name, format!("{name}: window length overflow")))?;
        shape[dim - 1] = axis.saturating_sub(trim);
    }
    checked_element_count(name, &shape)?;
    Ok(shape)
}

fn scalar_numeric(value: &Value) -> Option<NumericScalar> {
    match value {
        Value::Num(n) => Some(NumericScalar::F64(*n)),
        Value::Int(i) => Some(i.clone().into()),
        Value::Bool(b) => Some(NumericScalar::F64(if *b { 1.0 } else { 0.0 })),
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => t.numeric_value_at(0),
        Value::LogicalArray(l) if l.data.len() == 1 => {
            Some(NumericScalar::F64(if l.data[0] != 0 { 1.0 } else { 0.0 }))
        }
        _ => None,
    }
}

fn positive_integer(name: &'static str, raw: f64, what: &str) -> BuiltinResult<usize> {
    let value = nonnegative_integer(name, raw, what)?;
    if value == 0 {
        return Err(invalid_argument(
            name,
            format!("{name}: {what} must be positive"),
        ));
    }
    Ok(value)
}

fn positive_integer_value(
    name: &'static str,
    value: &IntValue,
    what: &str,
) -> BuiltinResult<usize> {
    let value = nonnegative_integer_value(name, value, what)?;
    if value == 0 {
        return Err(invalid_argument(
            name,
            format!("{name}: {what} must be positive"),
        ));
    }
    Ok(value)
}

fn nonnegative_integer_value(
    name: &'static str,
    value: &IntValue,
    what: &str,
) -> BuiltinResult<usize> {
    value.try_to_usize().ok_or_else(|| {
        invalid_argument(
            name,
            format!("{name}: {what} must be a nonnegative integer"),
        )
    })
}

fn nonnegative_integer(name: &'static str, raw: f64, what: &str) -> BuiltinResult<usize> {
    if !raw.is_finite() {
        return Err(invalid_argument(
            name,
            format!("{name}: {what} must be finite"),
        ));
    }
    let rounded = raw.round();
    if (rounded - raw).abs() > 1e-9 {
        return Err(invalid_argument(
            name,
            format!("{name}: {what} must be an integer"),
        ));
    }
    if rounded < 0.0 {
        return Err(invalid_argument(
            name,
            format!("{name}: {what} must be nonnegative"),
        ));
    }
    if rounded > usize::MAX as f64 || (usize::BITS == 64 && rounded == usize::MAX as f64) {
        return Err(invalid_argument(
            name,
            format!("{name}: {what} is too large"),
        ));
    }
    Ok(rounded as usize)
}

fn positive_span(name: &'static str, raw: f64, what: &str) -> BuiltinResult<f64> {
    let value = nonnegative_span(name, raw, what)?;
    if value <= 0.0 {
        return Err(invalid_argument(
            name,
            format!("{name}: {what} must be positive"),
        ));
    }
    Ok(value)
}

fn nonnegative_span(name: &'static str, raw: f64, what: &str) -> BuiltinResult<f64> {
    if !raw.is_finite() {
        return Err(invalid_argument(
            name,
            format!("{name}: {what} must be finite"),
        ));
    }
    if raw < 0.0 {
        return Err(invalid_argument(
            name,
            format!("{name}: {what} must be nonnegative"),
        ));
    }
    Ok(raw)
}

fn value_as_string(name: &'static str, value: &Value) -> BuiltinResult<Option<String>> {
    Ok(match value {
        Value::String(runmat_value::StringElement::Text(s)) => Some(s.as_str().to_owned()),
        Value::StringArray(sa) if sa.data().len() == 1 && !sa.data()[0].is_missing() => {
            sa.data()[0].as_str().map(str::to_owned)
        }
        Value::CharArray(ca) if ca.rows == 1 => Some(
            runmat_value::decode_utf16_code_units(ca.data()).map_err(|source| {
                let mut error = invalid_argument(name, source.to_string());
                error.source = Some(Box::new(source));
                error
            })?,
        ),
        _ => None,
    })
}

fn default_dimension(shape: &[usize]) -> usize {
    shape
        .iter()
        .position(|&dim| dim != 1)
        .map(|idx| idx + 1)
        .unwrap_or(1)
}

fn checked_product(name: &'static str, values: &[usize]) -> BuiltinResult<usize> {
    values.iter().try_fold(1usize, |acc, value| {
        acc.checked_mul(*value)
            .ok_or_else(|| internal(name, format!("{name}: shape product overflow")))
    })
}

fn checked_element_count(name: &'static str, shape: &[usize]) -> BuiltinResult<usize> {
    checked_product(name, shape)
}

fn invalid_argument(name: &'static str, detail: impl Into<String>) -> RuntimeError {
    descriptor_error(name, &ERROR_INVALID_ARGUMENT, "InvalidArgument", detail)
}

fn invalid_input(name: &'static str, detail: impl Into<String>) -> RuntimeError {
    descriptor_error(name, &ERROR_INVALID_INPUT, "InvalidInput", detail)
}

fn internal(name: &'static str, detail: impl Into<String>) -> RuntimeError {
    descriptor_error(name, &ERROR_INTERNAL, "Internal", detail)
}

fn descriptor_error(
    name: &'static str,
    descriptor: &'static BuiltinErrorDescriptor,
    identifier_suffix: &'static str,
    detail: impl Into<String>,
) -> RuntimeError {
    let builder = build_runtime_error(format!("{}: {}", descriptor.message, detail.into()))
        .with_builtin(name)
        .with_identifier(format!("RunMat:{name}:{identifier_suffix}"));
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_value::{IntegerComplexStorage, IntegerStorage, NumericStorage};

    fn call(name: &'static str, op: MovingOp, args: Vec<Value>) -> BuiltinResult<Value> {
        block_on(dispatch::moving_builtin(
            None,
            name,
            op,
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
    }

    fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new(data, shape).unwrap())
    }

    #[test]
    fn moving_operations_reject_typed_complex_integer_inputs() {
        let input = Value::ComplexTensor(
            ComplexTensor::new_integer(
                IntegerComplexStorage::new(
                    IntegerStorage::U64(vec![u64::MAX]),
                    IntegerStorage::U64(vec![1]),
                )
                .expect("storage"),
                vec![1, 1],
            )
            .expect("tensor"),
        );
        let err = call("movmean", MovingOp::Mean, vec![input, Value::Num(1.0)])
            .expect_err("typed complex integer input must reject");
        assert!(err.message().contains("complex numbers with integer types"));
    }

    #[test]
    fn moving_integer_data_contracts_are_operation_specific() {
        let storages = vec![
            IntegerStorage::I8(vec![1, 3, 2]),
            IntegerStorage::I16(vec![1, 3, 2]),
            IntegerStorage::I32(vec![1, 3, 2]),
            IntegerStorage::I64(vec![1, 3, 2]),
            IntegerStorage::U8(vec![1, 3, 2]),
            IntegerStorage::U16(vec![1, 3, 2]),
            IntegerStorage::U32(vec![1, 3, 2]),
            IntegerStorage::U64(vec![1, 3, 2]),
        ];
        for storage in storages {
            let dtype = storage.numeric_dtype();
            for (name, op) in [
                ("movsum", MovingOp::Sum),
                ("movprod", MovingOp::Prod),
                ("movmean", MovingOp::Mean),
            ] {
                let input =
                    Tensor::new_integer(storage.clone(), vec![1, 3]).expect("integer input");
                let output = expect_tensor(
                    call(name, op, vec![Value::Tensor(input), Value::Num(1.0)])
                        .expect("floating moving reduction"),
                );
                assert_eq!(
                    output
                        .into_numeric_storage()
                        .expect("numeric output")
                        .numeric_dtype(),
                    runmat_value::NumericDType::F64,
                    "{name} with {dtype:?}"
                );
            }
            for (name, op) in [
                ("movmin", MovingOp::Min),
                ("movmax", MovingOp::Max),
                ("movmedian", MovingOp::Median),
            ] {
                let input =
                    Tensor::new_integer(storage.clone(), vec![1, 3]).expect("integer input");
                let output = expect_tensor(
                    call(name, op, vec![Value::Tensor(input), Value::Num(3.0)])
                        .expect("exact moving reduction"),
                );
                assert_eq!(output.numeric_dtype(), dtype, "{name} with {dtype:?}");
            }
        }
    }

    #[test]
    fn moving_integer_extrema_and_median_are_exact_above_flintmax() {
        let base = (1_u64 << 63) + 16;
        let input = || {
            Value::Tensor(
                Tensor::new_integer(
                    IntegerStorage::U64(vec![base, base + 2, base + 1]),
                    vec![1, 3],
                )
                .expect("wide input"),
            )
        };
        let expected = [
            ("movmin", MovingOp::Min, vec![base, base, base + 1]),
            ("movmax", MovingOp::Max, vec![base + 2, base + 2, base + 2]),
            (
                "movmedian",
                MovingOp::Median,
                vec![base + 1, base + 1, base + 2],
            ),
        ];
        for (name, op, expected) in expected {
            let output = expect_tensor(
                call(name, op, vec![input(), Value::Num(3.0)]).expect("wide moving reduction"),
            );
            assert_eq!(
                output
                    .integer_storage()
                    .expect("integer output")
                    .exact_values(),
                expected.into_iter().map(IntValue::U64).collect::<Vec<_>>(),
                "{name}"
            );
        }
    }

    #[test]
    fn moving_integer_endpoint_fill_stays_exact() {
        let fill = u64::MAX;
        let input = Tensor::new_integer(IntegerStorage::U64(vec![fill - 2, fill - 3]), vec![1, 2])
            .expect("wide input");
        let output = expect_tensor(
            call(
                "movmax",
                MovingOp::Max,
                vec![
                    Value::Tensor(input),
                    Value::Num(3.0),
                    Value::from("Endpoints"),
                    Value::Int(IntValue::U64(fill)),
                ],
            )
            .expect("exact endpoint fill"),
        );
        assert_eq!(
            output
                .integer_storage()
                .expect("integer output")
                .exact_values(),
            vec![IntValue::U64(fill), IntValue::U64(fill)]
        );
    }

    #[test]
    fn moving_dispersion_rejects_every_integer_data_class() {
        let storages = vec![
            IntegerStorage::I8(vec![1]),
            IntegerStorage::I16(vec![1]),
            IntegerStorage::I32(vec![1]),
            IntegerStorage::I64(vec![1]),
            IntegerStorage::U8(vec![1]),
            IntegerStorage::U16(vec![1]),
            IntegerStorage::U32(vec![1]),
            IntegerStorage::U64(vec![1]),
        ];
        for storage in storages {
            for (name, op) in [("movstd", MovingOp::Std), ("movvar", MovingOp::Var)] {
                let input =
                    Tensor::new_integer(storage.clone(), vec![1, 1]).expect("integer input");
                for extensions_enabled in [false, true] {
                    let _compat =
                        crate::compatibility::push_runmat_extensions_enabled(extensions_enabled);
                    let error = call(
                        name,
                        op,
                        vec![Value::Tensor(input.clone()), Value::Num(1.0)],
                    )
                    .expect_err("integer data must reject");
                    assert_eq!(
                        error.identifier(),
                        Some(if name == "movstd" {
                            "RunMat:movstd:InvalidInput"
                        } else {
                            "RunMat:movvar:InvalidInput"
                        })
                    );
                }
            }
        }
    }

    #[test]
    fn moving_dispersion_rejects_resident_integer_data_before_provider_dispatch() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let input = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
                .expect("integer input");
            let handle = gpu_helpers::upload_tensor(provider, &input).expect("integer upload");
            for (name, op) in [("movstd", MovingOp::Std), ("movvar", MovingOp::Var)] {
                for extensions_enabled in [false, true] {
                    let _compat =
                        crate::compatibility::push_runmat_extensions_enabled(extensions_enabled);
                    let error = call(
                        name,
                        op,
                        vec![Value::GpuTensor(handle.clone()), Value::Num(1.0)],
                    )
                    .expect_err("resident integer data must reject");
                    assert_eq!(
                        error.identifier(),
                        Some(if name == "movstd" {
                            "RunMat:movstd:InvalidInput"
                        } else {
                            "RunMat:movvar:InvalidInput"
                        })
                    );
                }
            }
            handle.release().unwrap();
        });
    }

    #[test]
    fn moving_dispersion_typed_integer_controls_follow_compatibility_mode() {
        for (name, op, identifier) in [
            (
                "movstd",
                MovingOp::Std,
                "RunMat:compatibility:MovstdTypedIntegerControlExtension",
            ),
            (
                "movvar",
                MovingOp::Var,
                "RunMat:compatibility:MovvarTypedIntegerControlExtension",
            ),
        ] {
            let args = || {
                vec![
                    tensor(vec![1.0, 2.0, 3.0], vec![1, 3]),
                    Value::Num(3.0),
                    Value::Int(IntValue::U64(1)),
                ]
            };
            {
                let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
                let error = call(name, op, args()).expect_err("strict mode must reject");
                assert_eq!(error.identifier(), Some(identifier));
            }
            {
                let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
                call(name, op, args()).expect("RunMat mode accepts typed integer control");
            }
        }
    }

    fn expect_tensor(value: Value) -> Tensor {
        match value {
            Value::Tensor(t) => t,
            Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).unwrap(),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn gpu_count_window_reductions_return_resident_results() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let input = provider
                .upload(&HostTensorView {
                    data: &[1.0, 2.0, f64::NAN, 4.0],
                    shape: &[1, 4],
                })
                .expect("upload");
            let result = call(
                "movsum",
                MovingOp::Sum,
                vec![
                    Value::GpuTensor(input.clone()),
                    Value::Num(3.0),
                    Value::from("omitnan"),
                ],
            )
            .expect("movsum gpu");
            let Value::GpuTensor(sum_handle) = result else {
                panic!("expected resident gpu result");
            };
            assert!(runmat_accelerate_api::provider_for_handle(&sum_handle).is_some());
            let sum =
                crate::builtins::common::test_support::gather(Value::GpuTensor(sum_handle.clone()))
                    .expect("gather sum");
            assert_eq!(sum.shape(), [1, 4]);
            assert_eq!(sum.materialize_f64(), vec![3.0, 3.0, 6.0, 4.0]);

            let var_input = provider
                .upload(&HostTensorView {
                    data: &[1.0, 2.0],
                    shape: &[1, 2],
                })
                .expect("upload var");
            let result = call(
                "movvar",
                MovingOp::Var,
                vec![
                    Value::GpuTensor(var_input.clone()),
                    Value::Num(3.0),
                    Value::from("Endpoints"),
                    Value::Num(0.0),
                ],
            )
            .expect("movvar gpu");
            let Value::GpuTensor(var_handle) = result else {
                panic!("expected resident gpu var result");
            };
            assert!(runmat_accelerate_api::provider_for_handle(&var_handle).is_some());
            let var =
                crate::builtins::common::test_support::gather(Value::GpuTensor(var_handle.clone()))
                    .expect("gather var");
            assert_eq!(var.shape(), [1, 2]);
            assert_eq!(var.materialize_f64(), vec![1.0, 1.0]);

            let median_input = provider
                .upload(&HostTensorView {
                    data: &[1.0, f64::NAN, 3.0, 9.0],
                    shape: &[1, 4],
                })
                .expect("upload median");
            let result = call(
                "movmedian",
                MovingOp::Median,
                vec![
                    Value::GpuTensor(median_input.clone()),
                    Value::Num(3.0),
                    Value::from("omitnan"),
                ],
            )
            .expect("movmedian gpu");
            let Value::GpuTensor(median_handle) = result else {
                panic!("expected resident gpu median result");
            };
            assert!(runmat_accelerate_api::provider_for_handle(&median_handle).is_some());
            let median = crate::builtins::common::test_support::gather(Value::GpuTensor(
                median_handle.clone(),
            ))
            .expect("gather median");
            assert_eq!(median.shape(), [1, 4]);
            assert_eq!(median.materialize_f64(), vec![1.0, 2.0, 6.0, 6.0]);

            input.release().unwrap();
            sum_handle.release().unwrap();
            var_input.release().unwrap();
            var_handle.release().unwrap();
            median_input.release().unwrap();
            median_handle.release().unwrap();
        });
    }

    #[test]
    fn movsum_shrink_vector() {
        let result = call(
            "movsum",
            MovingOp::Sum,
            vec![tensor(vec![1., 2., 3., 4.], vec![1, 4]), Value::Num(3.0)],
        )
        .expect("movsum");
        let out = expect_tensor(result);
        assert_eq!(out.shape(), [1, 4]);
        assert_eq!(out.materialize_f64(), vec![3.0, 6.0, 9.0, 7.0]);
    }

    #[test]
    fn moving_real_preserves_native_single_output_class() {
        let input = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).expect("input");
        let result = call(
            "movmean",
            MovingOp::Mean,
            vec![Value::Tensor(input), Value::Num(3.0)],
        )
        .expect("movmean");
        let output = expect_tensor(result);
        assert_eq!(
            output.into_numeric_storage().expect("native storage"),
            NumericStorage::F32(vec![1.5, 2.0, 3.0, 3.5])
        );
    }

    #[test]
    fn movmean_matrix_dimension_two() {
        let input = tensor(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
        let result = call(
            "movmean",
            MovingOp::Mean,
            vec![input, Value::Num(3.0), Value::Num(2.0)],
        )
        .expect("movmean");
        let out = expect_tensor(result);
        assert_eq!(out.shape(), [2, 3]);
        assert_eq!(out.materialize_f64(), vec![2.0, 3.0, 3.0, 4.0, 4.0, 5.0]);
    }

    #[test]
    fn movprod_omitnan_and_asymmetric_window() {
        let input = tensor(vec![2., f64::NAN, 3., 4.], vec![1, 4]);
        let window = tensor(vec![1., 0.], vec![1, 2]);
        let result = call(
            "movprod",
            MovingOp::Prod,
            vec![input, window, Value::from("omitnan")],
        )
        .expect("movprod");
        let out = expect_tensor(result);
        assert_eq!(out.materialize_f64(), vec![2.0, 2.0, 3.0, 12.0]);
    }

    #[test]
    fn moving_window_vector_reads_typed_integer_storage_exactly() {
        for shape in [vec![1, 2], vec![2, 1], vec![1, 2, 1]] {
            let window = Tensor::new_integer(IntegerStorage::I16(vec![1, 0]), shape).unwrap();
            assert!(arguments::parse_window("movsum", &Value::Tensor(window)).is_ok());
        }
        let window = Tensor::new_integer(IntegerStorage::I16(vec![1, 0]), vec![1, 1, 2]).unwrap();
        assert_eq!(
            arguments::parse_window("movsum", &Value::Tensor(window))
                .unwrap_err()
                .identifier(),
            Some("RunMat:movsum:InvalidArgument"),
        );
        let input = tensor(vec![2., f64::NAN, 3., 4.], vec![1, 4]);
        let window =
            Tensor::new_integer(IntegerStorage::I16(vec![1, 0]), vec![1, 2]).expect("window");

        let result = call(
            "movprod",
            MovingOp::Prod,
            vec![input, Value::Tensor(window), Value::from("omitnan")],
        )
        .expect("movprod");
        let out = expect_tensor(result);
        assert_eq!(out.materialize_f64(), vec![2.0, 2.0, 3.0, 12.0]);
    }

    #[test]
    fn moving_window_uses_all_integer_storage_classes_without_mirror() {
        let storages = vec![
            IntegerStorage::I8(vec![3]),
            IntegerStorage::I16(vec![3]),
            IntegerStorage::I32(vec![3]),
            IntegerStorage::I64(vec![3]),
            IntegerStorage::U8(vec![3]),
            IntegerStorage::U16(vec![3]),
            IntegerStorage::U32(vec![3]),
            IntegerStorage::U64(vec![3]),
        ];
        for storage in storages {
            let window = Tensor::new_integer(storage, vec![1, 1]).expect("window");
            assert!(matches!(
                arguments::parse_window("movsum", &Value::Tensor(window)),
                Ok(WindowSpec::Counts {
                    before: 1,
                    after: 1
                })
            ));
        }
    }

    #[test]
    fn movmin_fill_endpoint_includes_fill_value() {
        let result = call(
            "movmin",
            MovingOp::Min,
            vec![
                tensor(vec![4., 2., 8.], vec![1, 3]),
                Value::Num(3.0),
                Value::from("Endpoints"),
                Value::Num(0.0),
            ],
        )
        .expect("movmin");
        let out = expect_tensor(result);
        assert_eq!(out.materialize_f64(), vec![0.0, 2.0, 0.0]);
    }

    #[test]
    fn movmax_discard_endpoint_shortens_dimension() {
        let result = call(
            "movmax",
            MovingOp::Max,
            vec![
                tensor(vec![1., 9., 3., 4.], vec![1, 4]),
                Value::Num(3.0),
                Value::from("Endpoints"),
                Value::from("discard"),
            ],
        )
        .expect("movmax");
        let out = expect_tensor(result);
        assert_eq!(out.shape(), [1, 2]);
        assert_eq!(out.materialize_f64(), vec![9.0, 9.0]);
    }

    #[test]
    fn movmedian_even_window_uses_average_middle() {
        let result = call(
            "movmedian",
            MovingOp::Median,
            vec![tensor(vec![1., 10., 3., 8.], vec![1, 4]), Value::Num(2.0)],
        )
        .expect("movmedian");
        let out = expect_tensor(result);
        assert_eq!(out.materialize_f64(), vec![1.0, 5.5, 6.5, 5.5]);
    }

    #[test]
    fn movstd_defaults_to_sample_normalization_and_accepts_population_weight() {
        let input = tensor(vec![4., 8., 6.], vec![1, 3]);
        let sample = call(
            "movstd",
            MovingOp::Std,
            vec![input.clone(), Value::Num(3.0)],
        )
        .expect("movstd sample");
        let population = call(
            "movstd",
            MovingOp::Std,
            vec![input, Value::Num(3.0), Value::Num(1.0)],
        )
        .expect("movstd population");
        let sample = expect_tensor(sample);
        let population = expect_tensor(population);
        assert!((sample.materialize_f64()[0] - 8.0_f64.sqrt()).abs() < 1e-12);
        assert!((sample.materialize_f64()[1] - 2.0).abs() < 1e-12);
        assert!((sample.materialize_f64()[2] - 2.0_f64.sqrt()).abs() < 1e-12);
        assert!((population.materialize_f64()[0] - 2.0).abs() < 1e-12);
        assert!((population.materialize_f64()[1] - (8.0_f64 / 3.0).sqrt()).abs() < 1e-12);
        assert!((population.materialize_f64()[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn movstd_normalization_reads_typed_integer_tensor_storage_exactly() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let input = tensor(vec![4., 8., 6.], vec![1, 3]);
        let normalization =
            Tensor::new_integer(IntegerStorage::U64(vec![1]), vec![1, 1]).expect("integer tensor");

        let result = call(
            "movstd",
            MovingOp::Std,
            vec![input, Value::Num(3.0), Value::Tensor(normalization)],
        )
        .expect("movstd population");
        let output = expect_tensor(result);
        assert!((output.materialize_f64()[0] - 2.0).abs() < 1e-12);
        assert!((output.materialize_f64()[1] - (8.0_f64 / 3.0).sqrt()).abs() < 1e-12);
        assert!((output.materialize_f64()[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn moving_normalization_uses_all_integer_storage_classes_without_mirror() {
        let storages = vec![
            IntegerStorage::I8(vec![1]),
            IntegerStorage::I16(vec![1]),
            IntegerStorage::I32(vec![1]),
            IntegerStorage::I64(vec![1]),
            IntegerStorage::U8(vec![1]),
            IntegerStorage::U16(vec![1]),
            IntegerStorage::U32(vec![1]),
            IntegerStorage::U64(vec![1]),
        ];
        for storage in storages {
            let normalization = Tensor::new_integer(storage, vec![1, 1]).expect("normalization");
            assert!(matches!(
                arguments::parse_variance_normalization("movstd", &Value::Tensor(normalization)),
                Ok(Some(VarianceNormalization::Population))
            ));
        }
    }

    #[test]
    fn movstd_uses_weight_then_dimension_grammar() {
        let input = tensor(vec![4., -1., -1., 8., -2., 3., 6., -3., 4.], vec![3, 3]);
        let result = call(
            "movstd",
            MovingOp::Std,
            vec![input, Value::Num(3.0), Value::Num(0.0), Value::Num(2.0)],
        )
        .expect("movstd dim");
        let out = expect_tensor(result);
        assert_eq!(out.shape(), [3, 3]);
        assert!((out.materialize_f64()[0] - 8.0_f64.sqrt()).abs() < 1e-12);
        assert!((out.materialize_f64()[1] - (0.5_f64).sqrt()).abs() < 1e-12);
        assert!((out.materialize_f64()[2] - 8.0_f64.sqrt()).abs() < 1e-12);
        assert!((out.materialize_f64()[3] - 2.0).abs() < 1e-12);
        assert!((out.materialize_f64()[4] - 1.0).abs() < 1e-12);
        assert!((out.materialize_f64()[5] - (7.0_f64).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn movstd_combines_weight_dimension_and_nanflag() {
        let input = tensor(
            vec![4., -1., -1., f64::NAN, -2., 3., 6., -3., 4.],
            vec![3, 3],
        );
        let result = call(
            "movstd",
            MovingOp::Std,
            vec![
                input,
                Value::Num(3.0),
                Value::Num(0.0),
                Value::Num(2.0),
                Value::from("omitnan"),
            ],
        )
        .expect("movstd dim omitnan");
        let out = expect_tensor(result);
        assert_eq!(out.shape(), [3, 3]);
        assert_eq!(out.materialize_f64()[0], 0.0);
        assert!((out.materialize_f64()[1] - 0.5_f64.sqrt()).abs() < 1e-12);
        assert!((out.materialize_f64()[2] - 8.0_f64.sqrt()).abs() < 1e-12);
        assert!((out.materialize_f64()[3] - 2.0_f64.sqrt()).abs() < 1e-12);
        assert_eq!(out.materialize_f64()[6], 0.0);
    }

    #[test]
    fn movstd_rejects_dimension_without_explicit_weight() {
        let err = call(
            "movstd",
            MovingOp::Std,
            vec![
                tensor(vec![1., 2., 3.], vec![1, 3]),
                Value::Num(3.0),
                Value::Num(2.0),
            ],
        )
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:movstd:InvalidArgument"));
    }

    #[test]
    fn movvar_supports_omitnan_endpoints_and_samplepoints() {
        let omit = call(
            "movvar",
            MovingOp::Var,
            vec![
                tensor(vec![4., f64::NAN, -1.], vec![1, 3]),
                Value::Num(3.0),
                Value::from("omitnan"),
            ],
        )
        .expect("movvar omitnan");
        assert_eq!(expect_tensor(omit).materialize_f64(), vec![0.0, 12.5, 0.0]);

        let fill = call(
            "movvar",
            MovingOp::Var,
            vec![
                tensor(vec![1., 2.], vec![1, 2]),
                Value::Num(3.0),
                Value::from("Endpoints"),
                Value::Num(0.0),
            ],
        )
        .expect("movvar fill");
        assert_eq!(expect_tensor(fill).materialize_f64(), vec![1.0, 1.0]);

        let sample_points = call(
            "movvar",
            MovingOp::Var,
            vec![
                tensor(vec![1., 2., 4.], vec![1, 3]),
                Value::Num(2.1),
                Value::from("SamplePoints"),
                tensor(vec![0., 1., 3.], vec![1, 3]),
            ],
        )
        .expect("movvar samplepoints");
        assert_eq!(
            expect_tensor(sample_points).materialize_f64(),
            vec![0.5, 0.5, 0.0]
        );
    }

    #[test]
    fn movvar_large_repeated_fill_uses_stable_accumulation() {
        let result = call(
            "movvar",
            MovingOp::Var,
            vec![
                tensor(vec![1.0e308], vec![1, 1]),
                Value::Num(1_000_001.0),
                Value::from("Endpoints"),
                Value::Num(1.0e308),
            ],
        )
        .expect("movvar stable fill");
        let out = expect_tensor(result);
        assert_eq!(out.shape(), [1, 1]);
        assert_eq!(out.materialize_f64(), vec![0.0]);
    }

    #[test]
    fn movstd_discard_endpoint_shortens_dimension() {
        let result = call(
            "movstd",
            MovingOp::Std,
            vec![
                tensor(vec![1., 2., 3., 4.], vec![1, 4]),
                Value::Num(3.0),
                Value::from("Endpoints"),
                Value::from("discard"),
            ],
        )
        .expect("movstd discard");
        let out = expect_tensor(result);
        assert_eq!(out.shape(), [1, 2]);
        assert_eq!(out.materialize_f64(), vec![1.0, 1.0]);
    }

    #[test]
    fn complex_movvar_uses_squared_complex_distance() {
        let input = Value::ComplexTensor(
            ComplexTensor::new(vec![(1., 0.), (3., 0.), (3., 4.)], vec![1, 3]).unwrap(),
        );
        let result =
            call("movvar", MovingOp::Var, vec![input, Value::Num(3.0)]).expect("complex movvar");
        let out = expect_tensor(result);
        assert_eq!(out.shape(), [1, 3]);
        assert!((out.materialize_f64()[0] - 2.0).abs() < 1e-12);
        assert!((out.materialize_f64()[1] - 20.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn moving_include_nan_by_default() {
        let result = call(
            "movsum",
            MovingOp::Sum,
            vec![tensor(vec![1., f64::NAN, 3.], vec![1, 3]), Value::Num(3.0)],
        )
        .expect("movsum");
        let out = expect_tensor(result);
        assert!(out.materialize_f64()[0].is_nan());
        assert!(out.materialize_f64()[1].is_nan());
        assert!(out.materialize_f64()[2].is_nan());
    }

    #[test]
    fn complex_movsum_supported() {
        let input = Value::ComplexTensor(
            ComplexTensor::new(vec![(1., 1.), (2., -1.), (3., 0.)], vec![1, 3]).unwrap(),
        );
        let result = call("movsum", MovingOp::Sum, vec![input, Value::Num(3.0)]).expect("movsum");
        let Value::ComplexTensor(out) = result else {
            panic!("expected complex tensor");
        };
        assert_eq!(
            out.materialize_f64(),
            vec![(3.0, 0.0), (6.0, 0.0), (5.0, -1.0)]
        );
    }

    #[test]
    fn samplepoints_support_irregular_numeric_windows() {
        for shape in [vec![2, 2], vec![1, 1, 4]] {
            let error = call(
                "movsum",
                MovingOp::Sum,
                vec![
                    tensor(vec![1., 2., 3., 4.], vec![1, 4]),
                    Value::Num(2.1),
                    Value::from("SamplePoints"),
                    tensor(vec![0., 1., 3., 4.], shape),
                ],
            )
            .unwrap_err();
            assert_eq!(error.identifier(), Some("RunMat:movsum:InvalidArgument"));
        }
        let result = call(
            "movsum",
            MovingOp::Sum,
            vec![
                tensor(vec![1., 2., 3.], vec![1, 3]),
                Value::Num(2.1),
                Value::from("SamplePoints"),
                tensor(vec![0., 1., 3.], vec![1, 3]),
            ],
        )
        .expect("movsum samplepoints");
        let out = expect_tensor(result);
        assert_eq!(out.materialize_f64(), vec![3.0, 3.0, 3.0]);
    }

    #[test]
    fn moving_gpu_sample_points_follow_compatibility_mode() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            for (name, op, identifier) in [
                (
                    "movsum",
                    MovingOp::Sum,
                    "RunMat:compatibility:MovsumGpuSamplePointsExtension",
                ),
                (
                    "movprod",
                    MovingOp::Prod,
                    "RunMat:compatibility:MovprodGpuSamplePointsExtension",
                ),
                (
                    "movmean",
                    MovingOp::Mean,
                    "RunMat:compatibility:MovmeanGpuSamplePointsExtension",
                ),
                (
                    "movmedian",
                    MovingOp::Median,
                    "RunMat:compatibility:MovmedianGpuSamplePointsExtension",
                ),
                (
                    "movmin",
                    MovingOp::Min,
                    "RunMat:compatibility:MovminGpuSamplePointsExtension",
                ),
                (
                    "movmax",
                    MovingOp::Max,
                    "RunMat:compatibility:MovmaxGpuSamplePointsExtension",
                ),
                (
                    "movstd",
                    MovingOp::Std,
                    "RunMat:compatibility:MovstdGpuSamplePointsExtension",
                ),
                (
                    "movvar",
                    MovingOp::Var,
                    "RunMat:compatibility:MovvarGpuSamplePointsExtension",
                ),
            ] {
                let input = provider
                    .upload(&HostTensorView {
                        data: &[1.0, 2.0, 4.0],
                        shape: &[1, 3],
                    })
                    .expect("upload");
                let args = || {
                    vec![
                        Value::GpuTensor(input.clone()),
                        Value::Num(2.1),
                        Value::from("SamplePoints"),
                        tensor(vec![0.0, 1.0, 3.0], vec![1, 3]),
                    ]
                };
                {
                    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
                    let error = call(name, op, args()).expect_err("strict mode must reject");
                    assert_eq!(error.identifier(), Some(identifier));
                }
                {
                    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
                    let output =
                        call(name, op, args()).expect("RunMat mode accepts GPU SamplePoints");
                    let Value::GpuTensor(output) = output else {
                        panic!("{name} must return a resident result");
                    };
                    output.release().unwrap();
                }
                input.release().unwrap();
            }
        });
    }

    #[test]
    fn movmedian_gpu_large_window_follows_compatibility_mode() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let values = (0..33).map(f64::from).collect::<Vec<_>>();
            let input = provider
                .upload(&HostTensorView {
                    data: &values,
                    shape: &[1, 33],
                })
                .expect("upload");
            {
                let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
                let error = call(
                    "movmedian",
                    MovingOp::Median,
                    vec![Value::GpuTensor(input.clone()), Value::Num(33.0)],
                )
                .expect_err("strict mode must reject");
                assert_eq!(
                    error.identifier(),
                    Some("RunMat:compatibility:MovmedianGpuLargeWindowExtension")
                );
            }
            {
                let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
                let output = call(
                    "movmedian",
                    MovingOp::Median,
                    vec![Value::GpuTensor(input.clone()), Value::Num(33.0)],
                )
                .expect("RunMat mode accepts large GPU window");
                let Value::GpuTensor(output) = output else {
                    panic!("movmedian must return resident result");
                };
                output.release().unwrap();
            }
            input.release().unwrap();
        });
    }

    #[test]
    fn huge_shrink_window_is_clamped_to_input_extent() {
        let result = call(
            "movsum",
            MovingOp::Sum,
            vec![tensor(vec![1., 2.], vec![1, 2]), Value::Num(1.0e12)],
        )
        .expect("movsum huge shrink");
        let out = expect_tensor(result);
        assert_eq!(out.materialize_f64(), vec![3.0, 3.0]);
    }

    #[test]
    fn huge_fill_window_accounts_for_repeated_fill_without_expansion() {
        let result = call(
            "movmean",
            MovingOp::Mean,
            vec![
                tensor(vec![1., 2.], vec![1, 2]),
                Value::Num(1.0e12),
                Value::from("Endpoints"),
                Value::Num(0.0),
            ],
        )
        .expect("movmean huge fill");
        let out = expect_tensor(result);
        assert_eq!(out.materialize_f64(), vec![3.0e-12, 3.0e-12]);
    }

    #[test]
    fn complex_movmin_movmax_and_movmedian_use_magnitude_ordering() {
        let input = Value::ComplexTensor(
            ComplexTensor::new(vec![(3., 0.), (1., 1.), (-2., 0.)], vec![1, 3]).unwrap(),
        );
        let min = call(
            "movmin",
            MovingOp::Min,
            vec![input.clone(), Value::Num(3.0)],
        )
        .expect("movmin");
        let max = call(
            "movmax",
            MovingOp::Max,
            vec![input.clone(), Value::Num(3.0)],
        )
        .expect("movmax");
        let median =
            call("movmedian", MovingOp::Median, vec![input, Value::Num(3.0)]).expect("median");

        let Value::ComplexTensor(min) = min else {
            panic!("expected complex movmin");
        };
        let Value::ComplexTensor(max) = max else {
            panic!("expected complex movmax");
        };
        let Value::ComplexTensor(median) = median else {
            panic!("expected complex movmedian");
        };
        assert_eq!(min.materialize_f64()[1], (1.0, 1.0));
        assert_eq!(max.materialize_f64()[1], (3.0, 0.0));
        assert_eq!(median.materialize_f64()[1], (-2.0, 0.0));
    }

    #[test]
    fn rejects_oversized_window_without_overflowing() {
        let boundary = if usize::BITS == 64 {
            usize::MAX as f64
        } else {
            (usize::MAX as f64) + 1.0
        };
        let err = call(
            "movsum",
            MovingOp::Sum,
            vec![
                tensor(vec![1., 2.], vec![1, 2]),
                Value::Tensor(Tensor::new(vec![boundary, 1.0], vec![1, 2]).unwrap()),
            ],
        )
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:movsum:InvalidArgument"));
    }

    #[test]
    fn dim_beyond_rank_behaves_like_trailing_singleton_dimension() {
        let result = call(
            "movsum",
            MovingOp::Sum,
            vec![
                tensor(vec![2., 3.], vec![1, 2]),
                Value::Num(3.0),
                Value::Num(4.0),
                Value::from("Endpoints"),
                Value::Num(10.0),
            ],
        )
        .expect("movsum trailing dim");
        let out = expect_tensor(result);
        assert_eq!(out.shape(), [1, 2, 1, 1]);
        assert_eq!(out.materialize_f64(), vec![22.0, 23.0]);
    }

    #[test]
    fn omitnan_empty_windows_use_sum_and_product_identities() {
        let nan = tensor(vec![f64::NAN], vec![1, 1]);
        let sum = call(
            "movsum",
            MovingOp::Sum,
            vec![nan.clone(), Value::Num(1.0), Value::from("omitnan")],
        )
        .expect("movsum omitnan");
        let prod = call(
            "movprod",
            MovingOp::Prod,
            vec![nan, Value::Num(1.0), Value::from("omitnan")],
        )
        .expect("movprod omitnan");
        assert_eq!(expect_tensor(sum).materialize_f64(), vec![0.0]);
        assert_eq!(expect_tensor(prod).materialize_f64(), vec![1.0]);
    }

    #[test]
    fn movmin_omits_nan_by_default_and_accepts_missing_alias() {
        let input = tensor(vec![f64::NAN, 2.0, 3.0], vec![1, 3]);
        let defaulted = call(
            "movmin",
            MovingOp::Min,
            vec![input.clone(), Value::Num(3.0)],
        )
        .expect("movmin");
        let explicit = call(
            "movmin",
            MovingOp::Min,
            vec![input, Value::Num(3.0), Value::from("includemissing")],
        )
        .expect("movmin");
        assert_eq!(
            expect_tensor(defaulted).materialize_f64(),
            vec![2.0, 2.0, 2.0]
        );
        assert!(expect_tensor(explicit).materialize_f64()[0].is_nan());
    }
}
