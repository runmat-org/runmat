//! Original-shape outputs and original constraint multiplier projection. The
//! augmented objective rows participate in KKT but are not user nonlcon rows.
use super::super::{
    model::Geometry,
    outcome::{Outcome, Termination},
    validation,
};
use crate::builtins::math::optim::common::{
    self,
    multiobjective::{inputs::empty, invalid, Kind},
    ScalarCooperation,
};
use crate::BuiltinResult;
use runmat_value::{Tensor, Value};

fn column(values: Vec<f64>, kind: Kind) -> BuiltinResult<Value> {
    let count = values.len();
    Tensor::new(values, vec![count, 1])
        .map(Value::Tensor)
        .map_err(|source| invalid(kind, source))
}
pub(super) fn flag(termination: Termination) -> i32 {
    match termination {
        Termination::Step => 4,
        Termination::Function => 5,
        other => other.flag(),
    }
}
pub(super) async fn finalize(
    kind: Kind,
    mut outcome: Outcome,
    geometry: &Geometry,
    shape: &[usize],
    scalar: bool,
    generated: usize,
    inconsistent_bounds: bool,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<crate::execution::RootedValueSequence> {
    let requested = crate::output_count::current_output_count_in(cooperation.runtime());
    let count = requested.unwrap_or(1);
    let mut values = super::super::workspace::capacity(count, cooperation_owner(kind))?;
    let original_variables = geometry.variables - 1;
    let reported = outcome.point.as_ref().map(|point| {
        if kind == Kind::Minimax {
            point
                .objectives
                .as_ref()
                .expect("vector objective payload")
                .values
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max)
        } else {
            point.f
        }
    });
    if count > 0 {
        outcome.x.truncate(original_variables);
        values.push(common::vector_to_value(
            kind.name(),
            outcome.x,
            shape,
            scalar,
        )?);
    }
    if count >= 2 {
        values.push(
            match outcome
                .point
                .as_mut()
                .and_then(|point| point.objectives.take())
            {
                Some(payload) => Tensor::new(payload.values, payload.shape)
                    .map(Value::Tensor)
                    .map_err(|source| invalid(kind, source))?,
                None => empty(),
            },
        );
    }
    if count >= 3 {
        values.push(reported.map(Value::Num).unwrap_or_else(empty));
    }
    if count >= 4 {
        values.push(Value::Num(flag(outcome.termination) as f64));
    }
    if count >= 5 {
        let mut fields = runmat_value::UnboundedStructValueBuilder::new();
        for (name, value) in [
            ("iterations", outcome.iterations),
            ("funcCount", outcome.counts.points),
            ("objectiveCallCount", outcome.counts.objective_calls),
            ("constraintCallCount", outcome.counts.constraint_calls),
        ] {
            fields.insert(name, Value::Num(value as f64));
        }
        fields.insert("algorithm", Value::from("goal-attainment-sqp"));
        fields.insert("stepsize", Value::Num(outcome.step));
        fields.insert(
            "lssteplength",
            outcome.alpha.map(Value::Num).unwrap_or_else(empty),
        );
        fields.insert(
            "constrviolation",
            outcome.violation.map(Value::Num).unwrap_or_else(empty),
        );
        fields.insert(
            "firstorderopt",
            outcome.optimality.map(Value::Num).unwrap_or_else(empty),
        );
        fields.insert("message", Value::from(if inconsistent_bounds {
            "No finite point satisfies the supplied lower and upper bounds; the objective was not evaluated."
        } else { outcome.termination.description() }));
        values.push(Value::Struct(fields.finish()));
    }
    if count >= 6 {
        if let Some(constraints) = outcome
            .constraints
            .filter(|_| outcome.multipliers_available)
        {
            let point = outcome.point.as_ref().expect("multiplier evaluated point");
            let mut groups = validation::split(
                &constraints,
                &outcome.multipliers,
                geometry.variables,
                geometry.inequality.rhs.len(),
                geometry.equality.rhs.len(),
                point.inequality.len(),
                point.equality.len(),
                cooperation_owner(kind),
                cooperation,
            )
            .await?;
            if groups.ineqnonlin.len() < generated {
                return Err(invalid(
                    kind,
                    "generated objective multiplier projection is inconsistent",
                ));
            }
            drop(groups.ineqnonlin.drain(..generated));
            groups.lower.truncate(original_variables);
            groups.upper.truncate(original_variables);
            let mut fields = runmat_value::UnboundedStructValueBuilder::new();
            for (name, group) in [
                "ineqlin",
                "eqlin",
                "ineqnonlin",
                "eqnonlin",
                "lower",
                "upper",
            ]
            .into_iter()
            .zip([
                groups.ineqlin,
                groups.eqlin,
                groups.ineqnonlin,
                groups.eqnonlin,
                groups.lower,
                groups.upper,
            ]) {
                fields.insert(name, column(group, kind)?);
            }
            values.push(Value::Struct(fields.finish()));
        } else {
            values.push(empty());
        }
    }
    let sequence = if requested.is_none() {
        runmat_value::ValueSequence::single(values.remove(0))
    } else {
        runmat_value::ValueSequence::comma_separated(values)
    }
    .map_err(crate::sequence::sequence_error_to_runtime)?;
    match cooperation.runtime() {
        Some(runtime) => crate::execution::RootedValueSequence::in_context(sequence, runtime),
        None => crate::execution::RootedValueSequence::new(sequence)
            .map_err(|source| source.into_runtime_error()),
    }
}
fn cooperation_owner(kind: Kind) -> runmat_builtins::BuiltinCatalogIdentity {
    runmat_builtins::BuiltinCatalogIdentity { name: kind.name() }
}
