//! Original-shaped vector objective observations, through the existing owned
//! callback port. Neither final values nor gradients replay the objective.
use super::super::{
    evaluation::{Attainment, Binding, Evaluator},
    outcome::{Outcome, Parameters},
};
use crate::builtins::math::optim::common::{
    self,
    multiobjective::{inputs::empty, invalid, Kind},
    output_callback::{CallbackForm, ObserverGroup, OutputState},
};
use crate::BuiltinResult;
use runmat_value::Value;

pub(in crate::builtins::math::optim::fmincon) async fn notify(
    kind: Kind,
    outcome: &Outcome,
    evaluator: &mut Evaluator<'_>,
    parameters: &Parameters<'_>,
    phase: OutputState,
    direction: &[f64],
    alpha: Option<f64>,
) -> BuiltinResult<bool> {
    if parameters.display == runmat_builtins::OptimDisplay::Iter
        || (phase == OutputState::Done
            && parameters.display == runmat_builtins::OptimDisplay::Final)
    {
        let _entry = evaluator
            .cooperation
            .runtime()
            .map(crate::context::RuntimeContext::enter);
        crate::console::record_console_line(crate::console::ConsoleStream::Stdout,
            format!("{} goal-attainment-sqp: iteration={} funcCount={} attainment={:?} constraintViolation={:?}{}",
                kind.name(), outcome.iterations, evaluator.budget.counts().points,
                outcome.point.as_ref().map(|point| point.f), outcome.violation,
                if phase == OutputState::Done { outcome.termination.description() } else { "" }))?;
    }
    let Some(callback) = parameters.callback else {
        return Ok(false);
    };
    let variables = evaluator.original_variables(outcome.x.len());
    let x = common::output_callback::vector(
        &outcome.x[..variables],
        &evaluator.shape,
        evaluator.scalar,
        evaluator.cooperation,
    )
    .await?;
    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
    fields.insert("iteration", Value::Num(outcome.iterations as f64));
    fields.insert(
        "funccount",
        Value::Num(evaluator.budget.counts().points as f64),
    );
    let fval = if let Some(payload) = outcome
        .point
        .as_ref()
        .and_then(|point| point.objectives.as_ref())
    {
        common::output_callback::vector(
            &payload.values,
            &payload.shape,
            false,
            evaluator.cooperation,
        )
        .await?
    } else {
        empty()
    };
    fields.insert("fval", fval);
    fields.insert(
        "attainfactor",
        outcome
            .point
            .as_ref()
            .map(|point| Value::Num(point.f))
            .unwrap_or_else(empty),
    );
    for (name, value) in [
        ("constrviolation", outcome.violation),
        ("firstorderopt", outcome.optimality),
        ("lssteplength", alpha),
    ] {
        fields.insert(name, value.map(Value::Num).unwrap_or_else(empty));
    }
    fields.insert("stepsize", Value::Num(outcome.step));
    let direction = if direction.is_empty() {
        empty()
    } else {
        common::output_callback::vector(
            &direction[..variables],
            &[variables, 1],
            false,
            evaluator.cooperation,
        )
        .await?
    };
    fields.insert("searchdirection", direction);
    let gradient = if let (
        Some(point),
        Some(Attainment {
            binding: Binding::Bound(formulation),
            ..
        }),
    ) = (
        outcome
            .point
            .as_ref()
            .filter(|point| !point.projected_constraints),
        &evaluator.attainment,
    ) {
        if let Some(matrix) = point.inequality_jacobian.as_ref() {
            let original = formulation
                .original_jacobian(matrix, evaluator.cooperation)
                .await?;
            Value::SparseTensor(
                runmat_value::SparseTensor::new(
                    original.m,
                    original.n,
                    original.colptr,
                    original.rowval,
                    original.nzval,
                )
                .map_err(|source| invalid(kind, source))?,
            )
        } else {
            empty()
        }
    } else {
        empty()
    };
    fields.insert("gradient", gradient);
    let runtime = evaluator.cooperation.runtime().cloned();
    let metadata = crate::execution::RootedValue::in_optional_context(
        Value::Struct(fields.finish()),
        runtime.as_ref(),
    )?;
    common::output_callback::invoke_groups(
        kind.name(),
        &[ObserverGroup {
            value: Some(callback),
            form: CallbackForm::Handles,
            select: None,
        }],
        x,
        metadata,
        Value::from(phase.name()),
        evaluator.cooperation,
    )
    .await
}
