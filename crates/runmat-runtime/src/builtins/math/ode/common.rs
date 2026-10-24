use runmat_builtins::BuiltinExtensionDescriptor;
use runmat_value::{Tensor, Value};

use crate::builtins::common::tensor;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[derive(Clone, Copy)]
pub(crate) struct OdeCompatibilityExtensions {
    pub(crate) integer_tspan: &'static BuiltinExtensionDescriptor,
    pub(crate) integer_y0: &'static BuiltinExtensionDescriptor,
    pub(crate) integer_option: &'static BuiltinExtensionDescriptor,
    pub(crate) integer_callback: &'static BuiltinExtensionDescriptor,
    pub(crate) logical_numeric: &'static BuiltinExtensionDescriptor,
    pub(crate) resident_input: &'static BuiltinExtensionDescriptor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OdeMethod {
    Ode45,
    Ode23,
    Ode15s,
}

impl OdeMethod {
    pub(super) fn default_refine(self) -> usize {
        match self {
            Self::Ode45 => 4,
            Self::Ode23 | Self::Ode15s => 1,
        }
    }
    pub(super) fn embedded_error_order(self) -> f64 {
        match self {
            // Embedded RK error estimate uses the lower-order formula.
            Self::Ode45 => 4.0,
            Self::Ode23 => 2.0,
            Self::Ode15s => 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct OdeOptions {
    pub rel_tol: f64,
    pub abs_tol: f64,
    pub initial_step: Option<f64>,
    pub max_step: Option<f64>,
    pub max_steps: usize,
    pub refine: Option<usize>,
}

#[derive(Clone)]
pub(crate) struct OdeInput {
    pub tspan: Vec<f64>,
    pub y0: Vec<f64>,
    pub y_shape: Vec<usize>,
    pub scalar_state: bool,
    pub(super) extensions: OdeCompatibilityExtensions,
    pub(super) runtime: Option<crate::context::RuntimeContext>,
}

pub(crate) struct OdeResult {
    pub t: Vec<f64>,
    pub y_rows: Vec<Vec<f64>>,
}

#[derive(Clone, Copy)]
enum OdeInputRole {
    TimeSpan,
    InitialState,
}

impl OdeInputRole {
    fn extension(
        self,
        extensions: OdeCompatibilityExtensions,
    ) -> &'static BuiltinExtensionDescriptor {
        match self {
            Self::TimeSpan => extensions.integer_tspan,
            Self::InitialState => extensions.integer_y0,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::TimeSpan => "tspan",
            Self::InitialState => "y0",
        }
    }
}

pub(crate) fn ode_error(name: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(name).build()
}

pub(crate) fn check_cancellation(
    _name: &str,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<()> {
    if let Some(runtime) = runtime.filter(|runtime| runtime.is_cancelled()) {
        return Err(crate::execution::ExecutionServiceError::Cancelled
            .into_runtime_error_in_context(Some(runtime)));
    }
    Ok(())
}

pub(crate) async fn parse_ode_input(
    name: &str,
    tspan: Value,
    y0: Value,
    extensions: OdeCompatibilityExtensions,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<OdeInput> {
    check_cancellation(name, runtime)?;
    check_numeric_compatibility(name, OdeInputRole::TimeSpan, &tspan, extensions)?;
    check_host_integer_exactness(name, OdeInputRole::TimeSpan, &tspan)?;
    validate_host_operand(name, OdeInputRole::TimeSpan, &tspan)?;
    check_numeric_compatibility(name, OdeInputRole::InitialState, &y0, extensions)?;
    check_host_integer_exactness(name, OdeInputRole::InitialState, &y0)?;
    validate_host_operand(name, OdeInputRole::InitialState, &y0)?;
    let tspan_value = prepare_ode_numeric_value(name, OdeInputRole::TimeSpan, tspan).await?;
    let tspan = super::values::real_vector(name, tspan_value).await?;
    validate_time_points(name, &tspan)?;

    let y0_value = prepare_ode_numeric_value(name, OdeInputRole::InitialState, y0).await?;
    let (y0, y_shape, scalar_state) = match y0_value {
        Value::Num(n) => (vec![n], vec![1, 1], true),
        Value::Int(i) => (vec![i.to_f64()], vec![1, 1], true),
        Value::Bool(b) => (vec![if b { 1.0 } else { 0.0 }], vec![1, 1], true),
        Value::Tensor(tensor) => {
            let values = tensor::tensor_values_f64(&tensor);
            if values.is_empty() {
                return Err(ode_error(name, format!("{name}: y0 cannot be empty")));
            }
            (values, tensor.shape().to_vec(), false)
        }
        Value::LogicalArray(logical) => {
            if logical.data.is_empty() {
                return Err(ode_error(name, format!("{name}: y0 cannot be empty")));
            }
            (
                logical
                    .data
                    .iter()
                    .map(|v| if *v == 0 { 0.0 } else { 1.0 })
                    .collect(),
                logical.shape.to_vec(),
                false,
            )
        }
        other => {
            return Err(ode_error(
                name,
                format!("{name}: y0 must be real numeric, got {other:?}"),
            ))
        }
    };

    validate_state(name, &y0)?;

    Ok(OdeInput {
        tspan,
        y0,
        y_shape,
        scalar_state,
        extensions,
        runtime: runtime.cloned(),
    })
}

fn validate_host_operand(name: &str, role: OdeInputRole, value: &Value) -> BuiltinResult<()> {
    match value {
        Value::GpuTensor(_) => Ok(()),
        Value::Num(number) => validate_host_series(name, role, 1, |_| *number),
        Value::Int(integer) => validate_host_series(name, role, 1, |_| integer.to_f64()),
        Value::Bool(boolean) => {
            validate_host_series(name, role, 1, |_| if *boolean { 1.0 } else { 0.0 })
        }
        Value::Tensor(tensor) => validate_host_series(name, role, tensor.len(), |index| {
            tensor
                .numeric_value_at(index)
                .expect("index is below authoritative tensor length")
                .materialize_f64()
        }),
        Value::LogicalArray(array) => validate_host_series(name, role, array.data.len(), |index| {
            if array.data[index] == 0 {
                0.0
            } else {
                1.0
            }
        }),
        other => Err(ode_error(
            name,
            format!(
                "{name}: {} must be real numeric, got {other:?}",
                role.label()
            ),
        )),
    }
}

fn validate_host_series(
    name: &str,
    role: OdeInputRole,
    len: usize,
    value_at: impl FnMut(usize) -> f64,
) -> BuiltinResult<()> {
    match role {
        OdeInputRole::TimeSpan => validate_time_series(name, len, value_at),
        OdeInputRole::InitialState => validate_state_series(name, len, value_at),
    }
}

fn validate_time_points(name: &str, values: &[f64]) -> BuiltinResult<()> {
    validate_time_series(name, values.len(), |index| values[index])
}

fn validate_time_series(
    name: &str,
    len: usize,
    mut value_at: impl FnMut(usize) -> f64,
) -> BuiltinResult<()> {
    if len < 2 {
        return Err(ode_error(
            name,
            format!("{name}: tspan must contain at least two time points"),
        ));
    }
    let first = value_at(0);
    let last = value_at(len - 1);
    let forward = last > first;
    let mut previous = first;
    for index in 0..len {
        let current = value_at(index);
        if !current.is_finite() {
            return Err(ode_error(
                name,
                format!("{name}: tspan values must be finite"),
            ));
        }
        if index > 0 && (previous == current || (current > previous) != forward) {
            return Err(ode_error(
                name,
                format!("{name}: tspan values must be strictly monotonic"),
            ));
        }
        previous = current;
    }
    Ok(())
}

fn validate_state(name: &str, values: &[f64]) -> BuiltinResult<()> {
    validate_state_series(name, values.len(), |index| values[index])
}

fn validate_state_series(
    name: &str,
    len: usize,
    mut value_at: impl FnMut(usize) -> f64,
) -> BuiltinResult<()> {
    if len == 0 {
        return Err(ode_error(name, format!("{name}: y0 cannot be empty")));
    }
    for index in 0..len {
        if !value_at(index).is_finite() {
            return Err(ode_error(name, format!("{name}: y0 values must be finite")));
        }
    }
    Ok(())
}

async fn prepare_ode_numeric_value(
    name: &str,
    role: OdeInputRole,
    value: Value,
) -> BuiltinResult<Value> {
    if matches!(value, Value::GpuTensor(_))
        && crate::builtins::common::validation::value_contains_native_integer_class(&value)
        && !crate::builtins::common::validation::native_integer_value_is_exact_f64_async(&value)
            .await?
    {
        return Err(ode_error(
            name,
            format!(
                "{name}: integer {} must be exactly representable as double",
                role.label()
            ),
        ));
    }
    match value {
        Value::GpuTensor(handle) => {
            crate::dispatcher::gather_if_needed_async(&Value::GpuTensor(handle)).await
        }
        value => Ok(value),
    }
}

fn check_host_integer_exactness(
    name: &str,
    role: OdeInputRole,
    value: &Value,
) -> BuiltinResult<()> {
    if !matches!(value, Value::GpuTensor(_))
        && crate::builtins::common::validation::value_contains_native_integer_class(value)
        && !crate::builtins::common::validation::native_integer_value_is_exact_f64(value)
    {
        return Err(ode_error(
            name,
            format!(
                "{name}: integer {} must be exactly representable as double",
                role.label()
            ),
        ));
    }
    Ok(())
}

fn check_numeric_compatibility(
    name: &str,
    role: OdeInputRole,
    value: &Value,
    extensions: OdeCompatibilityExtensions,
) -> BuiltinResult<()> {
    if crate::builtins::common::validation::value_contains_native_integer_class(value) {
        crate::compatibility::ensure_builtin_extension_enabled(role.extension(extensions), name)?;
    }
    if matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
    {
        crate::compatibility::ensure_builtin_extension_enabled(extensions.logical_numeric, name)?;
    }
    if crate::builtins::common::validation::value_contains_explicit_gpu(value) {
        crate::compatibility::ensure_builtin_extension_enabled(extensions.resident_input, name)?;
    }
    Ok(())
}

pub(crate) fn build_ode_output(name: &str, result: OdeResult) -> BuiltinResult<Value> {
    let t = Tensor::new(result.t.clone(), vec![result.t.len(), 1])
        .map(Value::Tensor)
        .map_err(|e| ode_error(name, format!("{name}: {e}")))?;
    let y = rows_to_matrix_value(name, &result.y_rows)?;

    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::output_list(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::output_list(vec![y]));
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![t, y],
        ));
    }
    Ok(y)
}

fn rows_to_matrix_value(name: &str, rows: &[Vec<f64>]) -> BuiltinResult<Value> {
    if rows.is_empty() {
        return Tensor::new(Vec::new(), vec![0, 0])
            .map(Value::Tensor)
            .map_err(|e| ode_error(name, format!("{name}: {e}")));
    }
    let row_count = rows.len();
    let col_count = rows[0].len();
    let mut data = vec![0.0; row_count * col_count];
    for (row_idx, row) in rows.iter().enumerate() {
        if row.len() != col_count {
            return Err(ode_error(
                name,
                format!("{name}: internal solver produced inconsistent row lengths"),
            ));
        }
        for (col_idx, value) in row.iter().enumerate() {
            data[row_idx + col_idx * row_count] = *value;
        }
    }
    Tensor::new(data, vec![row_count, col_count])
        .map(Value::Tensor)
        .map_err(|e| ode_error(name, format!("{name}: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::IntegerStorage;

    #[test]
    fn parse_ode_input_reads_typed_integer_storage_exactly() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let tspan = Tensor::new_integer(IntegerStorage::U16(vec![0, 2]), vec![1, 2]).unwrap();
        let y0 = Tensor::new_integer(IntegerStorage::I16(vec![3, -4]), vec![2, 1]).unwrap();

        let input = block_on(parse_ode_input(
            "ode_test",
            Value::Tensor(tspan),
            Value::Tensor(y0),
            super::super::ode45::ODE_COMPATIBILITY_EXTENSIONS,
            None,
        ))
        .unwrap();

        assert_eq!(input.tspan, vec![0.0, 2.0]);
        assert_eq!(input.y0, vec![3.0, -4.0]);
        assert_eq!(input.y_shape, vec![2, 1]);
        assert!(!input.scalar_state);
    }

    #[test]
    fn ode_integer_metadata_covers_every_shared_role_for_each_solver() {
        let capability_sets = [
            &runmat_builtins::ODE15S_INTEGER_CAPABILITIES,
            &runmat_builtins::ODE23_INTEGER_CAPABILITIES,
            &runmat_builtins::ODE45_INTEGER_CAPABILITIES,
        ];
        for capabilities in capability_sets {
            assert_eq!(capabilities.len(), 4);
            for input in capabilities.iter().flat_map(|capability| capability.inputs) {
                assert_eq!(
                    input.availability,
                    runmat_builtins::BuiltinIntegerInputAvailability::RunMatOnly
                );
            }
            assert!(capabilities.iter().all(|capability| {
                capability.backend == runmat_builtins::BuiltinIntegerBackendRule::GatherFallback
            }));
        }
    }
}
