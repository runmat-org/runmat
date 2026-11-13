//! Ordinary original-context observer publication; no objective replay.
use super::{
    evaluation::{Counts, Point},
    workspace,
};
use crate::builtins::math::optim::common::{
    self,
    output_callback::{CallbackForm, ObserverGroup, OutputState},
    ScalarCooperation,
};
use crate::BuiltinResult;
use runmat_value::Value;

pub(super) struct Snapshot<'a> {
    pub x: &'a [f64],
    pub point: Option<&'a Point>,
    pub shape: &'a [usize],
    pub scalar: bool,
    pub iterations: usize,
    pub counts: Counts,
    pub violation: Option<f64>,
    pub optimality: Option<f64>,
    pub step: f64,
    pub alpha: Option<f64>,
    pub direction: &'a [f64],
}
pub(super) async fn notify(
    callback: Option<&Value>,
    phase: OutputState,
    snapshot: Snapshot<'_>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    notify_with_info(callback, phase, snapshot, None, cooperation).await
}
pub(super) async fn notify_reflective(
    callback: Option<&Value>,
    phase: OutputState,
    snapshot: Snapshot<'_>,
    info: super::publication::ReflectiveInfo,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    notify_with_info(callback, phase, snapshot, Some(info), cooperation).await
}
async fn notify_with_info(
    callback: Option<&Value>,
    phase: OutputState,
    snapshot: Snapshot<'_>,
    reflective: Option<super::publication::ReflectiveInfo>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    let Some(callback) = callback else {
        return Ok(false);
    };
    common::output_callback::validate(callback)?;
    let x =
        common::output_callback::vector(snapshot.x, snapshot.shape, snapshot.scalar, cooperation)
            .await?;
    let runtime = cooperation.runtime().cloned();
    let gradient = match snapshot
        .point
        .filter(|point| !point.projected_objective)
        .and_then(|value| value.gradient.as_deref())
    {
        Some(gradient) => {
            common::output_callback::vector(gradient, snapshot.shape, snapshot.scalar, cooperation)
                .await?
        }
        None => common::output_callback::empty(runtime.as_ref())?,
    };
    let direction = if snapshot.direction.is_empty() {
        common::output_callback::empty(runtime.as_ref())?
    } else {
        common::output_callback::vector(
            snapshot.direction,
            &[snapshot.direction.len(), 1],
            false,
            cooperation,
        )
        .await?
    };
    let optional = |value: Option<f64>| -> BuiltinResult<Value> {
        value
            .map(Value::Num)
            .map(Ok)
            .unwrap_or_else(|| common::output_callback::empty(runtime.as_ref()))
    };
    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
    fields.insert("iteration", Value::Num(snapshot.iterations as f64));
    fields.insert("funccount", Value::Num(snapshot.counts.points as f64));
    fields.insert("fval", optional(snapshot.point.map(|value| value.f))?);
    fields.insert("gradient", gradient);
    fields.insert("constrviolation", optional(snapshot.violation)?);
    fields.insert("firstorderopt", optional(snapshot.optimality)?);
    fields.insert("stepsize", Value::Num(snapshot.step));
    fields.insert("searchdirection", direction);
    fields.insert("lssteplength", optional(snapshot.alpha)?);
    if let Some(info) = reflective {
        fields.insert("cgiterations", Value::Num(info.cg_iterations as f64));
        fields.insert("trustregionradius", Value::Num(info.radius));
    }
    let metadata = crate::execution::RootedValue::in_optional_context(
        Value::Struct(fields.finish()),
        runtime.as_ref(),
    )?;
    common::output_callback::invoke_groups(
        "fmincon",
        &[ObserverGroup {
            value: Some(callback),
            form: CallbackForm::Handles,
            select: None,
        }],
        x,
        metadata,
        Value::from(phase.name()),
        cooperation,
    )
    .await
}

pub(super) async fn copy(
    values: &[f64],
    owner: runmat_builtins::BuiltinCatalogIdentity,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    let mut output = workspace::capacity(values.len(), owner)?;
    for &value in values {
        output.push(value);
        cooperation.completed(1).await?;
    }
    Ok(output)
}
