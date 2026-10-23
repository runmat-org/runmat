//! MATLAB-compatible `interp2` builtin for gridded dense numeric data.

#[cfg(test)]
use runmat_builtins::INTERP2_DESCRIPTOR;
use runmat_builtins::{
    BuiltinErrorDescriptor, INTERP2_ERROR_INTERNAL, INTERP2_ERROR_INVALID_ARGUMENT,
    INTERP2_ERROR_INVALID_INPUT, INTERP2_INTEGER_EXTRAPOLATION_EXTENSION,
    INTERP2_INTEGER_QUERY_EXTENSION, INTERP2_INTEGER_SAMPLE_EXTENSION,
    INTERP2_LOGICAL_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexTensor, NumericDType, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::dispatcher;
use crate::{build_runtime_error, RuntimeError};

use super::pp::{
    interval_index, is_vector_shape, out_of_range_value, parse_extrapolation, parse_method,
    query_points, vector_from_value, Extrapolation, InterpMethod,
};

#[path = "interp2/allocation.rs"]
mod allocation;
#[path = "interp2/spline_grid.rs"]
mod spline_grid;

const NAME: &str = "interp2";

fn interp2_matrix_dimensions(tensor: &Tensor) -> (usize, usize) {
    match tensor.shape() {
        [] => (0, 0),
        [columns] => (1, *columns),
        [rows, columns, ..] => (*rows, *columns),
    }
}

fn interp2_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn interp2_invalid_argument(detail: impl AsRef<str>) -> RuntimeError {
    interp2_error_with_message(
        format!(
            "{}: {}",
            INTERP2_ERROR_INVALID_ARGUMENT.message,
            detail.as_ref()
        ),
        &INTERP2_ERROR_INVALID_ARGUMENT,
    )
}

fn interp2_invalid_input(detail: impl AsRef<str>) -> RuntimeError {
    interp2_error_with_message(
        format!(
            "{}: {}",
            INTERP2_ERROR_INVALID_INPUT.message,
            detail.as_ref()
        ),
        &INTERP2_ERROR_INVALID_INPUT,
    )
}

fn interp2_map_error(err: RuntimeError, fallback: &'static BuiltinErrorDescriptor) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        interp2_error_with_message(err.message().to_string(), fallback)
    }
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::interpolation::interp2")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("interpolation-2d"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Initial implementation gathers GPU inputs to the CPU reference path. Bilinear and nearest kernels are good future provider candidates.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::interpolation::interp2"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "interp2 is currently a runtime sink.",
};

#[runtime_builtin(
    name = "interp2",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::interpolation::interp2",
    context = true
)]
async fn interp2_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let allocation = allocation::admit_before_gather(args.view(), runtime.as_ref())?;
    preflight_interp2_inputs(args.view()).await?;
    if let (Some(receipt), Some(runtime)) = (allocation.as_ref(), runtime.as_ref()) {
        receipt
            .checkpoint(runtime)
            .map_err(|error| super::pp::host_allocation_error(NAME, error))?;
    }
    let parsed = ParsedInterp2::parse(args, runtime.as_ref())
        .await
        .map_err(|err| interp2_map_error(err, &INTERP2_ERROR_INVALID_INPUT))?;
    if let (Some(receipt), Some(runtime)) = (allocation.as_ref(), runtime.as_ref()) {
        receipt
            .checkpoint(runtime)
            .map_err(|error| super::pp::host_allocation_error(NAME, error))?;
    }
    if parsed.method == InterpMethod::Spline {
        return spline_grid::evaluate(&parsed, runtime.as_ref(), allocation.as_ref()).await;
    }
    let data =
        evaluate_grid(&parsed).map_err(|err| interp2_map_error(err, &INTERP2_ERROR_INTERNAL))?;
    if data.len() == 1 {
        return Ok(Value::Num(data[0]));
    }
    let tensor = Tensor::new(data, parsed.output_shape).map_err(|err| {
        interp2_error_with_message(format!("{NAME}: {err}"), &INTERP2_ERROR_INTERNAL)
    })?;
    Ok(Value::Tensor(tensor))
}

#[cfg(test)]
async fn interp2_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let args = crate::call::BuiltinArguments::from_borrowed(&args, None)?;
    interp2_builtin_registered(None, args).await
}

async fn preflight_interp2_inputs(
    args: crate::call::FunctionArgumentView<'_>,
) -> crate::BuiltinResult<()> {
    use crate::builtins::common::validation::{
        native_integer_value_is_exact_f64_async, value_has_logical_class,
        value_has_native_integer_class,
    };
    if args.len() < 3 {
        return Ok(());
    }
    let explicit = args.len() >= 5 && !is_option_arg(&args[3])?;
    let (samples, queries) = if explicit {
        (
            args.range(..3).expect("interp2 samples"),
            args.range(3..5).expect("interp2 queries"),
        )
    } else {
        (
            args.range(..1).expect("interp2 samples"),
            args.range(1..3).expect("interp2 queries"),
        )
    };
    for value in samples {
        if value_has_native_integer_class(value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &INTERP2_INTEGER_SAMPLE_EXTENSION,
                NAME,
            )?;
            if !native_integer_value_is_exact_f64_async(value).await? {
                return Err(interp2_invalid_input(
                    "integer samples must be exactly representable as double",
                ));
            }
        }
    }
    for value in queries {
        if value_has_native_integer_class(value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &INTERP2_INTEGER_QUERY_EXTENSION,
                NAME,
            )?;
            if !native_integer_value_is_exact_f64_async(value).await? {
                return Err(interp2_invalid_input(
                    "integer query coordinates must be exactly representable as double",
                ));
            }
        }
    }
    if samples
        .iter()
        .chain(queries.iter())
        .any(|value| value_has_logical_class(value))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &INTERP2_LOGICAL_INPUT_EXTENSION,
            NAME,
        )?;
    }
    let option_start = if explicit { 5 } else { 3 };
    for option in args.iter().skip(option_start) {
        if value_has_native_integer_class(option) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &INTERP2_INTEGER_EXTRAPOLATION_EXTENSION,
                NAME,
            )?;
            if !native_integer_value_is_exact_f64_async(option).await? {
                return Err(interp2_invalid_input(
                    "integer extrapolation value must be exactly representable as double",
                ));
            }
        }
        if value_has_logical_class(option) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &INTERP2_LOGICAL_INPUT_EXTENSION,
                NAME,
            )?;
        }
    }
    Ok(())
}

struct ParsedInterp2 {
    x_axis: Vec<f64>,
    y_axis: Vec<f64>,
    z: GridSamples,
    xq: Vec<f64>,
    yq: Vec<f64>,
    output_shape: Vec<usize>,
    method: InterpMethod,
    extrap: Extrapolation,
}

impl ParsedInterp2 {
    async fn parse(
        mut args: crate::call::BuiltinArguments,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> crate::BuiltinResult<Self> {
        if args.len() < 3 {
            return Err(interp2_invalid_argument(
                "expected Z, Xq, and Yq or X, Y, Z, Xq, and Yq",
            ));
        }

        let mut method = InterpMethod::Linear;
        let mut extrap = Extrapolation::Nan;
        let mut explicit_extrap = false;
        let explicit_axes = args.len() >= 5 && !is_option_arg(&args[3])?;
        let (x_axis, y_axis, z, xq_value, yq_value, options) = if explicit_axes {
            let x = args.pop_front().expect("X");
            let y = args.pop_front().expect("Y");
            let z_value = args.pop_front().expect("Z");
            let z = grid_samples(z_value).await?;
            let (rows, cols) = z.dimensions();
            let (x_axis, y_axis) = axes_from_values(x, y, rows, cols, runtime).await?;
            let xq = args.pop_front().expect("Xq");
            let yq = args.pop_front().expect("Yq");
            (x_axis, y_axis, z, xq, yq, args)
        } else {
            let z_value = args.pop_front().expect("Z");
            let z = grid_samples(z_value).await?;
            let (rows, cols) = z.dimensions();
            let x_axis: Vec<f64> = (1..=cols).map(|v| v as f64).collect();
            let y_axis: Vec<f64> = (1..=rows).map(|v| v as f64).collect();
            let xq = args.pop_front().expect("Xq");
            let yq = args.pop_front().expect("Yq");
            (x_axis, y_axis, z, xq, yq, args)
        };

        validate_axis(&x_axis, "X", runtime).await?;
        validate_axis(&y_axis, "Y", runtime).await?;
        let xq = query_points(xq_value, NAME).await?;
        let yq = query_points(yq_value, NAME).await?;
        let (xq_values, yq_values, output_shape) = align_queries(xq, yq, runtime).await?;

        for option in &options {
            if let Some(parsed) = parse_extrapolation(option, NAME).await? {
                extrap = parsed;
                explicit_extrap = true;
                continue;
            }
            if let Some(parsed) = parse_method(option, NAME)? {
                match parsed {
                    InterpMethod::Linear | InterpMethod::Nearest | InterpMethod::Spline => {
                        method = parsed
                    }
                    _ => {
                        return Err(interp2_invalid_argument(
                            "only linear, nearest, and spline methods are supported",
                        ));
                    }
                }
                continue;
            }
            return Err(interp2_error_with_message(
                "interp2: unsupported interpolation option",
                &INTERP2_ERROR_INVALID_ARGUMENT,
            ));
        }

        if method != InterpMethod::Spline && z.is_complex() {
            return Err(interp2_invalid_input(
                "complex Z requires the spline method in this implementation",
            ));
        }
        if method == InterpMethod::Spline {
            if !matches!(z.dtype(), NumericDType::F32 | NumericDType::F64) {
                return Err(interp2_invalid_input(
                    "spline sample values must be single or double",
                ));
            }
            if !explicit_extrap {
                extrap = Extrapolation::Extrapolate;
            }
        }

        Ok(Self {
            x_axis,
            y_axis,
            z,
            xq: xq_values,
            yq: yq_values,
            output_shape,
            method,
            extrap,
        })
    }
}

fn is_option_arg(value: &Value) -> crate::BuiltinResult<bool> {
    Ok(crate::builtins::common::random_args::keyword_of(value)?.is_some())
}

enum GridSamples {
    Real(Tensor),
    Complex(ComplexTensor),
}

impl GridSamples {
    fn dimensions(&self) -> (usize, usize) {
        let shape = match self {
            Self::Real(t) => t.shape(),
            Self::Complex(t) => t.shape(),
        };
        match shape {
            [] => (0, 0),
            [columns] => (1, *columns),
            [rows, columns, ..] => (*rows, *columns),
        }
    }

    fn is_complex(&self) -> bool {
        matches!(self, Self::Complex(_))
    }

    fn dtype(&self) -> NumericDType {
        match self {
            Self::Real(t) => t.numeric_dtype(),
            Self::Complex(t) => t.numeric_dtype(),
        }
    }

    fn at(&self, index: usize, imaginary: bool) -> f64 {
        match self {
            Self::Real(t) => {
                if imaginary {
                    0.0
                } else {
                    tensor::tensor_value_f64(t, index)
                }
            }
            Self::Complex(t) => {
                if let Some(values) = t.as_f64_slice() {
                    let v = values[index];
                    if imaginary {
                        v.1
                    } else {
                        v.0
                    }
                } else if let Some(values) = t.as_f32_slice() {
                    let v = values[index];
                    if imaginary {
                        f64::from(v.1)
                    } else {
                        f64::from(v.0)
                    }
                } else {
                    unreachable!("floating complex samples admitted")
                }
            }
        }
    }
}

async fn grid_samples(value: Value) -> crate::BuiltinResult<GridSamples> {
    let gathered = dispatcher::gather_if_needed_async(&value).await?;
    let z = match gathered {
        Value::ComplexTensor(t)
            if matches!(t.numeric_dtype(), NumericDType::F32 | NumericDType::F64) =>
        {
            GridSamples::Complex(t)
        }
        other => GridSamples::Real(
            tensor::value_into_tensor_for(NAME, other)
                .map_err(|err| interp2_invalid_input(&err))?,
        ),
    };
    let rank = match &z {
        GridSamples::Real(t) => t.shape().len(),
        GridSamples::Complex(t) => t.shape().len(),
    };
    if rank > 2 {
        return Err(interp2_invalid_argument("Z must be a 2-D matrix"));
    }
    let (rows, cols) = z.dimensions();
    if rows < 2 || cols < 2 {
        return Err(interp2_invalid_argument(
            "Z must have at least two rows and two columns",
        ));
    }
    Ok(z)
}

async fn axes_from_values(
    x: Value,
    y: Value,
    rows: usize,
    cols: usize,
    runtime: Option<&crate::context::RuntimeContext>,
) -> crate::BuiltinResult<(Vec<f64>, Vec<f64>)> {
    let x_axis = axis_from_value(x, rows, cols, true, runtime).await?;
    let y_axis = axis_from_value(y, rows, cols, false, runtime).await?;
    Ok((x_axis, y_axis))
}

async fn axis_from_value(
    value: Value,
    rows: usize,
    cols: usize,
    is_x: bool,
    runtime: Option<&crate::context::RuntimeContext>,
) -> crate::BuiltinResult<Vec<f64>> {
    let gathered = dispatcher::gather_if_needed_async(&value).await?;
    let tensor_value = tensor::value_into_tensor_for(NAME, gathered.clone());
    if let Ok(t) = tensor_value {
        if is_vector_shape(t.shape()) {
            let expected = if is_x { cols } else { rows };
            if t.len() != expected {
                return Err(interp2_invalid_argument(
                    "axis vector length must match Z dimensions",
                ));
            }
            return Ok(tensor::tensor_into_values_f64(t));
        }
        if interp2_matrix_dimensions(&t) == (rows, cols) {
            let axis: Vec<f64> = if is_x {
                (0..cols)
                    .map(|col| tensor::tensor_value_f64(&t, col * rows))
                    .collect()
            } else {
                (0..rows)
                    .map(|row| tensor::tensor_value_f64(&t, row))
                    .collect()
            };
            let mut budget = runtime.map(crate::context::RuntimeContext::cpu_cooperation);
            for col in 0..cols {
                for row in 0..rows {
                    if (row + col * rows).is_multiple_of(4096) {
                        if let Some(budget) = budget.as_mut() {
                            budget
                                .checkpoint(if row == 0 && col == 0 { 0 } else { 4096 })
                                .await
                                .map_err(|source| source.into_runtime_error())?;
                        }
                    }
                    let expected = if is_x { axis[col] } else { axis[row] };
                    if tensor::tensor_value_f64(&t, row + col * rows) != expected {
                        return Err(interp2_invalid_input(
                            "X and Y matrices must be full meshgrid coordinates",
                        ));
                    }
                }
            }
            return Ok(axis);
        }
    }
    let label = if is_x { "X" } else { "Y" };
    vector_from_value(gathered, label, NAME).await
}

async fn validate_axis(
    axis: &[f64],
    label: &str,
    runtime: Option<&crate::context::RuntimeContext>,
) -> crate::BuiltinResult<()> {
    if axis.len() < 2 {
        return Err(interp2_invalid_argument(format!(
            "{label} axis must contain at least two points"
        )));
    }
    let mut budget = runtime.map(crate::context::RuntimeContext::cpu_cooperation);
    for (index, &point) in axis.iter().enumerate() {
        if index.is_multiple_of(4096) {
            if let Some(budget) = budget.as_mut() {
                budget
                    .checkpoint(if index == 0 { 0 } else { 4096 })
                    .await
                    .map_err(|source| source.into_runtime_error())?;
            }
        }
        if !point.is_finite() {
            return Err(interp2_invalid_argument(format!(
                "{label} axis must be finite"
            )));
        }
        if index > 0 && point <= axis[index - 1] {
            return Err(interp2_invalid_argument(format!(
                "{label} axis must be strictly increasing"
            )));
        }
    }
    Ok(())
}

async fn align_queries(
    xq: super::pp::QueryPoints,
    yq: super::pp::QueryPoints,
    runtime: Option<&crate::context::RuntimeContext>,
) -> crate::BuiltinResult<(Vec<f64>, Vec<f64>, Vec<usize>)> {
    match (xq.values.len(), yq.values.len()) {
        (1, 1) => Ok((xq.values, yq.values, vec![1, 1])),
        (1, len) => Ok((vec![xq.values[0]; len], yq.values, yq.shape)),
        (len, 1) => Ok((xq.values, vec![yq.values[0]; len], xq.shape)),
        (columns, rows)
            if xq.shape.as_slice() == [1, columns] && yq.shape.as_slice() == [rows, 1] =>
        {
            let count = rows
                .checked_mul(columns)
                .ok_or_else(|| interp2_invalid_argument("query grid size overflow"))?;
            let mut xs = Vec::new();
            let mut ys = Vec::new();
            xs.try_reserve_exact(count)
                .map_err(|_| interp2_invalid_argument("query grid exceeds memory"))?;
            ys.try_reserve_exact(count)
                .map_err(|_| interp2_invalid_argument("query grid exceeds memory"))?;
            let mut budget = runtime.map(crate::context::RuntimeContext::cpu_cooperation);
            let mut completed = 0u64;
            for &x in &xq.values {
                for &y in &yq.values {
                    xs.push(x);
                    ys.push(y);
                    completed += 1;
                    if completed == 4096 {
                        if let Some(budget) = budget.as_mut() {
                            budget
                                .checkpoint(completed)
                                .await
                                .map_err(|source| source.into_runtime_error())?;
                        }
                        completed = 0;
                    }
                }
            }
            if let Some(budget) = budget.as_mut() {
                budget
                    .checkpoint(completed)
                    .await
                    .map_err(|source| source.into_runtime_error())?;
            }
            Ok((xs, ys, vec![rows, columns]))
        }
        (left, right) if left == right && xq.shape == yq.shape => {
            Ok((xq.values, yq.values, xq.shape))
        }
        _ => Err(interp2_invalid_argument(
            "Xq and Yq must be scalar or matching-size arrays",
        )),
    }
}

fn evaluate_grid(parsed: &ParsedInterp2) -> crate::BuiltinResult<Vec<f64>> {
    let mut out = Vec::with_capacity(parsed.xq.len());
    for (&xq, &yq) in parsed.xq.iter().zip(parsed.yq.iter()) {
        let value = match parsed.method {
            InterpMethod::Linear => eval_bilinear(parsed, xq, yq),
            InterpMethod::Nearest => eval_nearest(parsed, xq, yq),
            _ => unreachable!("interp2 parse rejects cubic methods"),
        };
        out.push(value);
    }
    Ok(out)
}

fn eval_bilinear(parsed: &ParsedInterp2, xq: f64, yq: f64) -> f64 {
    if !xq.is_finite() || !yq.is_finite() {
        return f64::NAN;
    }
    let allow = matches!(parsed.extrap, Extrapolation::Extrapolate);
    let Some(col) = interval_index(&parsed.x_axis, xq, allow) else {
        return out_of_range_value(&parsed.extrap);
    };
    let Some(row) = interval_index(&parsed.y_axis, yq, allow) else {
        return out_of_range_value(&parsed.extrap);
    };
    let x0 = parsed.x_axis[col];
    let x1 = parsed.x_axis[col + 1];
    let y0 = parsed.y_axis[row];
    let y1 = parsed.y_axis[row + 1];
    let tx = (xq - x0) / (x1 - x0);
    let ty = (yq - y0) / (y1 - y0);
    let z00 = z_at(&parsed.z, row, col);
    let z10 = z_at(&parsed.z, row, col + 1);
    let z01 = z_at(&parsed.z, row + 1, col);
    let z11 = z_at(&parsed.z, row + 1, col + 1);
    (1.0 - tx) * (1.0 - ty) * z00 + tx * (1.0 - ty) * z10 + (1.0 - tx) * ty * z01 + tx * ty * z11
}

fn eval_nearest(parsed: &ParsedInterp2, xq: f64, yq: f64) -> f64 {
    if !xq.is_finite() || !yq.is_finite() {
        return f64::NAN;
    }
    let Some(col) = nearest_index(&parsed.x_axis, xq, &parsed.extrap) else {
        return out_of_range_value(&parsed.extrap);
    };
    let Some(row) = nearest_index(&parsed.y_axis, yq, &parsed.extrap) else {
        return out_of_range_value(&parsed.extrap);
    };
    z_at(&parsed.z, row, col)
}

fn z_at(z: &GridSamples, row: usize, col: usize) -> f64 {
    z.at(row + col * z.dimensions().0, false)
}

fn nearest_index(axis: &[f64], q: f64, extrap: &Extrapolation) -> Option<usize> {
    if q < axis[0] {
        return matches!(extrap, Extrapolation::Extrapolate).then_some(0);
    }
    let last = axis.len() - 1;
    if q > axis[last] {
        return matches!(extrap, Extrapolation::Extrapolate).then_some(last);
    }
    match axis.binary_search_by(|probe| probe.partial_cmp(&q).unwrap()) {
        Ok(index) => Some(index),
        Err(index) => {
            let left = index.saturating_sub(1);
            let right = index.min(last);
            if (q - axis[left]).abs() <= (axis[right] - q).abs() {
                Some(left)
            } else {
                Some(right)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::IntegerStorage;
    use std::rc::Rc;

    struct SplineManagedParallel(crate::context::host_effect::ManagedHostEffectLedger);

    impl crate::context::RuntimeParallelService for SplineManagedParallel {
        fn supports(&self, _: crate::context::ParallelCapability) -> bool {
            false
        }
        fn placement_resources(&self) -> crate::context::RuntimeParallelResources {
            let snapshot = self.0.snapshot();
            crate::context::RuntimeParallelResources {
                memory_available_bytes: Some(snapshot.free_bytes()),
                epoch: snapshot.epoch,
                ..crate::context::RuntimeParallelResources::default()
            }
        }
        fn managed_host_effect_ledger(
            &self,
        ) -> Option<crate::context::host_effect::ManagedHostEffectLedger> {
            Some(self.0.clone())
        }
    }

    fn managed_runtime(
        ledger: crate::context::host_effect::ManagedHostEffectLedger,
    ) -> crate::context::RuntimeContext {
        crate::context::RuntimeContext::new(Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(
            crate::context::RuntimeServicePorts::default()
                .with_parallel(Rc::new(SplineManagedParallel(ledger))),
        )
    }

    fn row(values: &[f64]) -> Value {
        Value::Tensor(Tensor::new(values.to_vec(), vec![1, values.len()]).expect("tensor"))
    }

    fn integer_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        let tensor = Tensor::new_integer(storage, shape).expect("integer tensor");
        Value::Tensor(tensor)
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn interp2_sample_grid_preserves_dense_rank_boundaries() {
        for (data, shape, dimensions) in [
            (vec![7.0], Vec::new(), (0, 0)),
            (vec![1.0, 2.0, 3.0, 4.0], vec![4], (1, 4)),
        ] {
            let z = Tensor::new(data, shape).unwrap();
            assert_eq!(interp2_matrix_dimensions(&z), dimensions);
            let error = block_on(interp2_builtin(vec![
                Value::Tensor(z),
                Value::Num(1.5),
                Value::Num(1.5),
            ]))
            .expect_err("Z needs two rows and two columns");
            assert_eq!(
                error.identifier(),
                INTERP2_ERROR_INVALID_ARGUMENT.identifier
            );
            assert!(error
                .message()
                .contains("at least two rows and two columns"));
        }
    }

    #[test]
    fn interp2_implicit_axes_bilinear_scalar() {
        let z = Value::Tensor(Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).expect("tensor"));
        let value =
            block_on(interp2_builtin(vec![z, Value::Num(1.5), Value::Num(1.5)])).expect("interp2");
        let Value::Num(result) = value else {
            panic!("expected scalar");
        };
        assert!((result - 2.5).abs() < 1e-12);
    }

    #[test]
    fn interp2_vector_axes_nearest() {
        let z = Value::Tensor(Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).expect("tensor"));
        let value = block_on(interp2_builtin(vec![
            row(&[10.0, 20.0]),
            row(&[100.0, 200.0]),
            z,
            Value::Num(18.0),
            Value::Num(120.0),
            Value::String("nearest".to_string().into()),
        ]))
        .expect("interp2");
        assert_eq!(value, Value::Num(2.0));
    }

    #[test]
    fn interp2_spline_reproduces_independent_bivariate_quadratic() {
        let x = [0.0, 0.7, 2.0, 4.0];
        let y = [-1.0, 0.0, 1.5, 3.0];
        let polynomial = |x: f64, y: f64| x * x + 3.0 * x * y + 2.0 * y * y;
        let z: Vec<f64> = x
            .iter()
            .flat_map(|&col| y.iter().map(move |&row| polynomial(col, row)))
            .collect();
        let result = block_on(interp2_builtin(vec![
            row(&x),
            row(&y),
            Value::Tensor(Tensor::new(z.clone(), vec![4, 4]).unwrap()),
            Value::Num(1.2),
            Value::Num(0.4),
            Value::String("spline".into()),
        ]))
        .expect("tensor-product spline");
        let Value::Num(actual) = result else {
            panic!("double scalar expected")
        };
        assert!((actual - polynomial(1.2, 0.4)).abs() < 1e-9);
        let single = block_on(interp2_builtin(vec![
            row(&x),
            row(&y),
            Value::Tensor(
                Tensor::from_f32(
                    z.into_iter().map(|value| value as f32).collect(),
                    vec![4, 4],
                )
                .unwrap(),
            ),
            Value::Num(1.2),
            Value::Num(0.4),
            Value::String("spline".into()),
        ]))
        .expect("single spline");
        let Value::Tensor(single) = single else {
            panic!("single scalar must retain its tensor class")
        };
        assert_eq!(single.numeric_dtype(), NumericDType::F32);
        assert_eq!(single.shape(), &[1, 1]);
        assert!((single.materialize_f64()[0] - polynomial(1.2, 0.4)).abs() < 1e-5);
    }

    #[test]
    fn interp2_spline_preserves_complex_single_and_query_meshgrid_shape() {
        let x = [0.0, 1.0, 2.0, 3.0];
        let y = [0.0, 1.0, 2.0, 3.0];
        let z: Vec<(f32, f32)> = x
            .iter()
            .flat_map(|&col| {
                y.iter()
                    .map(move |&row| ((col + row) as f32, (2.0 * col - row) as f32))
            })
            .collect();
        let samples = ComplexTensor::from_f32(z, vec![4, 4]).unwrap();
        let result = block_on(interp2_builtin(vec![
            row(&x),
            row(&y),
            Value::ComplexTensor(samples),
            row(&[0.5, 1.5, 2.5]),
            Value::Tensor(Tensor::new(vec![0.25, 1.25], vec![2, 1]).unwrap()),
            Value::String("spline".into()),
        ]))
        .expect("complex single spline");
        let Value::ComplexTensor(output) = result else {
            panic!("complex output expected")
        };
        assert_eq!(output.shape(), &[2, 3]);
        assert_eq!(output.numeric_dtype(), NumericDType::F32);
        let values = output.materialize_f64();
        assert!((values[0].0 - 0.75).abs() < 1e-5);
        assert!((values[0].1 - 0.75).abs() < 1e-5);
        assert!((values[5].0 - 3.75).abs() < 1e-5);
        assert!((values[5].1 - 3.75).abs() < 1e-5);
    }

    #[test]
    fn interp2_finite_spline_retains_only_the_exact_complex_output_claim() {
        let ledger =
            crate::context::host_effect::ManagedHostEffectLedger::new_with_epoch(1_000_000, 17);
        let runtime = managed_runtime(ledger.clone());
        let samples = ComplexTensor::from_f32(
            (0..16).map(|n| (n as f32, -(n as f32))).collect(),
            vec![4, 4],
        )
        .unwrap();
        let args = || {
            vec![
                Value::ComplexTensor(samples.clone()),
                row(&[1.25, 2.25, 3.25]),
                Value::Tensor(Tensor::new(vec![1.5, 2.5], vec![2, 1]).unwrap()),
                Value::String("spline".into()),
            ]
        };
        let output = block_on(interp2_builtin_registered(
            Some(runtime.clone()),
            crate::builtins::common::test_support::builtin_arguments(args()),
        ))
        .expect("managed complex spline");
        let Value::ComplexTensor(output) = output else {
            panic!("complex output expected")
        };
        assert_eq!(output.shape(), &[2, 3]);
        assert_eq!(output.numeric_dtype(), NumericDType::F32);
        let retained =
            crate::context::host_allocation::HostAllocationPlan::retained_complex_tensor_bytes(
                NumericDType::F32,
                &[2, 3],
            )
            .unwrap();
        assert_eq!(ledger.snapshot().live_reserved_bytes, retained);
        assert_eq!(ledger.snapshot().live_claimed_bytes, retained);
        drop(output);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);

        let narrow = crate::context::host_effect::ManagedHostEffectLedger::new_with_epoch(32, 18);
        let failure = block_on(interp2_builtin_registered(
            Some(managed_runtime(narrow.clone())),
            crate::builtins::common::test_support::builtin_arguments(args()),
        ))
        .expect_err("finite admission must precede spline materialization");
        assert_eq!(
            failure.identifier(),
            Some("RunMat:Interpolation:HostAllocation")
        );
        assert_eq!(narrow.snapshot().live_reserved_bytes, 0);

        let owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
        let resident = owner
            .allocate_zeroed(
                vec![4, 4],
                runmat_accelerate_api::NumericElementType::F32,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap();
        let failure = block_on(interp2_builtin_registered(
            Some(managed_runtime(narrow.clone())),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::GpuTensor(resident.handle),
                Value::Num(1.5),
                Value::Num(1.5),
                Value::String("spline".into()),
            ]),
        ))
        .expect_err("finite capacity must reject before provider download");
        assert_eq!(
            failure.identifier(),
            Some("RunMat:Interpolation:HostAllocation")
        );
        assert_eq!(narrow.snapshot().live_reserved_bytes, 0);
    }

    #[test]
    fn interp2_rejects_nonrectilinear_full_meshgrid() {
        let x = Tensor::new(vec![0.0, 0.0, 1.0, 1.1], vec![2, 2]).unwrap();
        let y = Tensor::new(vec![0.0, 1.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let z = Tensor::new(vec![0.0, 1.0, 1.0, 2.0], vec![2, 2]).unwrap();
        let error = block_on(interp2_builtin(vec![
            Value::Tensor(x),
            Value::Tensor(y),
            Value::Tensor(z),
            Value::Num(0.5),
            Value::Num(0.5),
            Value::String("spline".into()),
        ]))
        .expect_err("malformed meshgrid must not be sampled by first row only");
        assert!(error.message().contains("full meshgrid"));
    }

    #[test]
    fn interp2_vector_axes_read_typed_integer_storage_exactly() {
        let args = || {
            vec![
                integer_tensor(IntegerStorage::I16(vec![10, 20]), vec![1, 2]),
                integer_tensor(IntegerStorage::I16(vec![100, 200]), vec![1, 2]),
                integer_tensor(IntegerStorage::I16(vec![1, 3, 2, 4]), vec![2, 2]),
                Value::Num(18.0),
                Value::Num(120.0),
                Value::String("nearest".to_string().into()),
            ]
        };
        let error =
            block_on(interp2_builtin(args())).expect_err("compatible mode rejects integer grids");
        assert_eq!(
            error.identifier(),
            INTERP2_INTEGER_SAMPLE_EXTENSION.error_identifier
        );
        let _guard = crate::compatibility::push_runmat_extensions_enabled(true);
        let value = block_on(interp2_builtin(args())).expect("RunMat interp2");

        assert_eq!(value, Value::Num(2.0));
    }

    #[test]
    fn interp2_full_grid_axes_read_typed_integer_storage_exactly() {
        let _guard = crate::compatibility::push_runmat_extensions_enabled(true);
        let x_grid = integer_tensor(IntegerStorage::I16(vec![10, 10, 20, 20]), vec![2, 2]);
        let y_grid = integer_tensor(IntegerStorage::I16(vec![100, 200, 100, 200]), vec![2, 2]);
        let value = block_on(interp2_builtin(vec![
            x_grid,
            y_grid,
            integer_tensor(IntegerStorage::I16(vec![1, 3, 2, 4]), vec![2, 2]),
            Value::Num(18.0),
            Value::Num(120.0),
            Value::String("nearest".to_string().into()),
        ]))
        .expect("interp2");

        assert_eq!(value, Value::Num(2.0));
    }

    #[test]
    fn interp2_integer_extrapolation_is_independently_gated() {
        let args = || {
            vec![
                Value::Tensor(Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).expect("tensor")),
                Value::Num(0.0),
                Value::Num(0.0),
                Value::String("linear".to_string().into()),
                Value::Int(runmat_value::IntValue::I16(99)),
            ]
        };
        let error = block_on(interp2_builtin(args()))
            .expect_err("compatible mode rejects integer extrapolation");
        assert_eq!(
            error.identifier(),
            INTERP2_INTEGER_EXTRAPOLATION_EXTENSION.error_identifier
        );
        let _guard = crate::compatibility::push_runmat_extensions_enabled(true);
        assert_eq!(
            block_on(interp2_builtin(args())).expect("RunMat extrapolation"),
            Value::Num(99.0)
        );
    }

    #[test]
    fn interp2_descriptor_signatures_cover_surface() {
        let labels: Vec<&str> = INTERP2_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert!(labels.contains(&"Vq = interp2(Z, Xq, Yq)"));
        assert!(labels.contains(&"Vq = interp2(X, Y, Z, Xq, Yq)"));
        assert!(labels.contains(&"Vq = interp2(X, Y, Z, Xq, Yq, method, extrap)"));
    }

    #[test]
    fn interp2_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = INTERP2_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert!(codes.contains(&"RM.INTERP2.INVALID_ARGUMENT"));
        assert!(codes.contains(&"RM.INTERP2.INVALID_INPUT"));
        assert!(codes.contains(&"RM.INTERP2.INTERNAL"));
    }

    #[test]
    fn interp2_too_few_args_uses_stable_identifier() {
        let err = block_on(interp2_builtin(vec![Value::Num(1.0), Value::Num(2.0)]))
            .expect_err("expected interp2 argument error");
        assert_eq!(err.identifier(), INTERP2_ERROR_INVALID_ARGUMENT.identifier);
    }
}
