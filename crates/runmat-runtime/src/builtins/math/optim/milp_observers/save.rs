//! Save genuine event points in the selected logical base workspace.
use super::*;
use crate::context::{RuntimeBaseWorkspaceService, RuntimeContext, RuntimeWorkspaceService};
use runmat_value::{StructValue, Tensor};
const KIND: MilpObserverKind = MilpObserverKind::SaveSolutions;

pub(super) async fn update(
    x: &Value,
    values: &StructValue,
    state: OutputState,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    if state == OutputState::Done {
        return Ok(());
    }
    if state == OutputState::Iter && matches!(x,Value::Tensor(t) if t.is_empty()) {
        return Ok(());
    }
    let runtime = cooperation.runtime().cloned().ok_or_else(|| {
        crate::context::RuntimeCapabilityError::new(
            crate::context::RuntimeCapability::BaseWorkspace,
            "savemilpsolutions requires its selected base workspace",
        )
        .into_runtime_error()
    })?;
    let base = runtime
        .service_ports()
        .require_base_workspace("savemilpsolutions base workspace")
        .map_err(crate::context::RuntimeCapabilityError::into_runtime_error)?
        .clone();
    let (points, objectives) = if state == OutputState::Init {
        (
            super::super::common::output_callback::empty(Some(&runtime))?,
            super::super::common::output_callback::empty(Some(&runtime))?,
        )
    } else {
        let fval = values
            .fields
            .get("fval")
            .ok_or_else(|| invalid(KIND, "optimValues.fval is required for a new point"))?;
        // Complete class/intent checks for both needed operands before transfer.
        numeric::preflight(KIND, x)?;
        numeric::preflight(KIND, fval)?;
        let x = numeric::gather(x, cooperation).await?;
        if matches!(&x,Value::Tensor(t) if t.shape().len()!=2 || t.shape()[1]!=1) {
            return Err(invalid(KIND, "new event x must be a decision column"));
        }
        let fval = numeric::gather(fval, cooperation).await?;
        let objective = numeric::scalar(KIND, &fval)?;
        let n = len(&x)?;
        if n == 0 {
            return Ok(());
        }
        for index in 0..n {
            if !at(&x, index)?.is_finite() {
                return Err(invalid(KIND, "event coordinates must be finite"));
            }
            if index % 1024 == 1023 {
                cooperation.completed(1024).await?;
            }
        }
        let old_points = lookup(base.as_ref(), &runtime, "xIntSol")?;
        let old_objectives = lookup(base.as_ref(), &runtime, "fIntSol")?;
        if old_points.is_some() != old_objectives.is_some() {
            return Err(invalid(
                KIND,
                "xIntSol and fIntSol must have matching saved histories",
            ));
        }
        if matches!(old_objectives.as_ref(),Some(Value::Tensor(t)) if !t.is_empty()
            && (t.shape().len()!=2 || (t.shape()[0]!=1 && t.shape()[1]!=1)))
        {
            return Err(invalid(KIND, "fIntSol must remain a vector"));
        }
        let columns = match old_points.as_ref() {
            None => 0,
            Some(value) if len(value)? == 0 => 0,
            Some(Value::Num(_)) if n == 1 => 1,
            Some(Value::Tensor(t))
                if t.shape().len() == 2 && t.shape()[0] == n && t.integer_storage().is_none() =>
            {
                t.shape()[1]
            }
            _ => return Err(invalid(
                KIND,
                "xIntSol row geometry differs from the new event point; call with init to reset",
            )),
        };
        if old_objectives.as_ref().map(len).transpose()?.unwrap_or(0) != columns {
            return Err(invalid(KIND, "fIntSol length differs from xIntSol columns"));
        }
        let next = columns
            .checked_add(1)
            .ok_or_else(|| invalid(KIND, "saved solution count overflows"))?;
        let count = n
            .checked_mul(next)
            .ok_or_else(|| invalid(KIND, "saved solution geometry overflows"))?;
        let mut points = runmat_value::exact_capacity_vec(count).map_err(allocation)?;
        let mut objectives = runmat_value::exact_capacity_vec(next).map_err(allocation)?;
        if let Some(old) = old_points.as_ref() {
            for index in 0..n * columns {
                points.push(at(old, index)?);
                if index % 1024 == 1023 {
                    cooperation.completed(1024).await?;
                }
            }
        }
        for index in 0..n {
            points.push(at(&x, index)?);
            if index % 1024 == 1023 {
                cooperation.completed(1024).await?;
            }
        }
        if let Some(old) = old_objectives.as_ref() {
            for index in 0..columns {
                objectives.push(at(old, index)?);
                if index % 1024 == 1023 {
                    cooperation.completed(1024).await?;
                }
            }
        }
        objectives.push(objective);
        (
            Value::Tensor(
                Tensor::new(points, vec![n, next]).map_err(|error| invalid(KIND, error))?,
            ),
            Value::Tensor(
                Tensor::new(objectives, vec![1, next]).map_err(|error| invalid(KIND, error))?,
            ),
        )
    };
    // Both replacements are prepared before the first write. These are the
    // existing ordered logical assignments (including bound globals), not an
    // invented private workspace or a replayable transaction.
    cooperation.completed(0).await?;
    let variables = runtime.session_variable_authority();
    base.assign_logical_ref_with_context(&variables, &runtime, "xIntSol", &points)?;
    base.assign_logical_ref_with_context(&variables, &runtime, "fIntSol", &objectives)?;
    cooperation.completed(0).await
}
fn lookup(
    base: &dyn RuntimeBaseWorkspaceService,
    runtime: &RuntimeContext,
    name: &str,
) -> BuiltinResult<Option<Value>> {
    let variables = runtime.session_variable_authority();
    if let Some(projection) = base.invocation_projection() {
        if let Some(target) = projection.global_target(name)? {
            return match target {
                crate::workspace::GlobalBindingTarget::Named(name) => {
                    variables.global_value(name.as_str())
                }
                crate::workspace::GlobalBindingTarget::LegacySlot(slot) => {
                    variables.global_slot(slot)
                }
            };
        }
        projection.try_lookup_with_context(runtime, name)
    } else {
        crate::workspace::lookup_logical_base_value(&base.owner(), &variables, name)
    }
}
fn len(value: &Value) -> BuiltinResult<usize> {
    match value {
        Value::Num(_) => Ok(1),
        Value::Tensor(t) if t.integer_storage().is_none() => Ok(t.len()),
        _ => Err(invalid(
            KIND,
            "saved histories and event points must be real floating data",
        )),
    }
}
fn at(value: &Value, index: usize) -> BuiltinResult<f64> {
    match value {
        Value::Num(value) if index == 0 => Ok(*value),
        Value::Tensor(t) if index < t.len() && t.integer_storage().is_none() => {
            Ok(crate::builtins::common::tensor::tensor_value_f64(t, index))
        }
        _ => Err(invalid(KIND, "saved history geometry changed")),
    }
}
fn allocation(source: runmat_value::ExactHostNumericAllocationError) -> RuntimeError {
    crate::build_runtime_error("savemilpsolutions: history allocation failed")
        .with_builtin(KIND.name())
        .with_source(source)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
