use super::super::{
    common, invalid, numeric,
    options::{
        self,
        admission::{self, Role},
    },
    ScalarCooperation,
};
use crate::{context::RuntimeContext, BuiltinResult};
use runmat_value::{Tensor, Value};

pub(super) fn begin(
    name: &'static str,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<ScalarCooperation> {
    crate::context::host_allocation::require_unbounded_workspace(runtime,name,
        "finite host-memory execution is unsupported; genetic operator workspaces have no complete finite bound")?;
    Ok(ScalarCooperation::for_builtin(
        runmat_builtins::BuiltinCatalogIdentity { name },
        runtime,
    ))
}
pub(super) fn count_metadata(value: &Value, runtime: Option<&RuntimeContext>) -> BuiltinResult<()> {
    admission::numeric_metadata(options::Owner::Solver, Role::Count, value, runtime)
}
pub(super) async fn count(
    value: &Value,
    label: &str,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<usize> {
    let value =
        crate::dispatcher::gather_if_needed_in_context_async(value, cooperation.runtime()).await?;
    common::option_usize_value("ga", Some(&value), label, 0)
}
pub(super) fn vector_metadata(
    value: &Value,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<()> {
    numeric::preflight(
        value,
        "genetic operator vector",
        common::constrained_numeric::Geometry::Vector,
        runtime,
    )
}
pub(super) async fn vector(
    value: Value,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    let common::constrained_numeric::Numeric::Real { values, .. } =
        common::constrained_numeric::read(
            value,
            "genetic operator vector",
            common::constrained_numeric::Geometry::Vector,
            &numeric::INPUT_POLICY,
            cooperation,
        )
        .await?
    else {
        return Err(invalid("genetic operator requires a real numeric vector"));
    };
    Ok(values)
}
pub(super) fn options(value: &Value) -> BuiltinResult<()> {
    if matches!(value, Value::Struct(_)) || super::super::inputs::raw::unspecified(value) {
        Ok(())
    } else {
        Err(invalid(
            "genetic operator options must be a scalar structure or []",
        ))
    }
}
pub(super) fn finish(
    values: Vec<f64>,
    row: bool,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<Value> {
    if crate::output_count::current_output_count_in(runtime) == Some(0) {
        return Ok(Value::output_list(Vec::new()));
    }
    let n = values.len();
    Tensor::new(values, if row { vec![1, n] } else { vec![n, 1] })
        .map(Value::Tensor)
        .map_err(invalid)
}
