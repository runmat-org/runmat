//! Original accepted x/F, honest calls, and finite constraint multipliers only.
use super::super::{model::Geometry, outcome::Outcome, validation, workspace};
use super::{bernstein::invalid, description, exitflag};
use crate::builtins::math::optim::common::{self, solver_timing::SolverTiming, ScalarCooperation};
use crate::BuiltinResult;
use runmat_value::{Tensor, Value};
const OWNER: runmat_builtins::BuiltinCatalogIdentity =
    runmat_builtins::BuiltinCatalogIdentity { name: "fseminf" };
fn column(v: Vec<f64>) -> BuiltinResult<Value> {
    let n = v.len();
    Tensor::new(v, vec![n, 1])
        .map(Value::Tensor)
        .map_err(invalid)
}
pub(super) async fn finalize(
    out: Outcome,
    geometry: &Geometry,
    shape: &[usize],
    scalar: bool,
    timing: Option<SolverTiming>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<crate::execution::RootedValueSequence> {
    let flag = exitflag(out.termination)?;
    let requested = crate::output_count::current_output_count_in(cooperation.runtime());
    let count = requested.unwrap_or(1);
    let mut values = workspace::capacity(count, OWNER)?;
    if count > 0 {
        values.push(common::vector_to_value("fseminf", out.x, shape, scalar)?);
    }
    if count >= 2 {
        values.push(
            out.point
                .as_ref()
                .map(|p| Value::Num(p.f))
                .unwrap_or_else(super::super::empty),
        );
    }
    if count >= 3 {
        values.push(Value::Num(flag as f64));
    }
    if count >= 4 {
        let mut fields = runmat_value::UnboundedStructValueBuilder::new();
        for (name, v) in [
            ("iterations", out.iterations),
            ("funcCount", out.counts.points),
            ("objectiveCallCount", out.counts.objective_calls),
            ("constraintCallCount", out.counts.constraint_calls),
        ] {
            fields.insert(name, Value::Num(v as f64));
        }
        for (name, v) in [
            ("constrviolation", out.violation),
            ("firstorderopt", out.optimality),
            ("lssteplength", out.alpha),
        ] {
            fields.insert(name, v.map(Value::Num).unwrap_or_else(super::super::empty));
        }
        fields.insert("stepsize", Value::Num(out.step));
        fields.insert(
            "algorithm",
            Value::from("active-set semi-infinite SQP with adaptive interpolated extrema"),
        );
        fields.insert("message", Value::from(description(out.termination)));
        fields.insert(
            "wallclocktime",
            Value::Num(
                timing
                    .ok_or_else(SolverTiming::unavailable)?
                    .elapsed()
                    .as_secs_f64(),
            ),
        );
        values.push(Value::Struct(fields.finish()));
    }
    if count >= 5 {
        if let Some(model) = out.constraints.filter(|_| out.multipliers_available) {
            let point = out
                .point
                .as_ref()
                .ok_or_else(|| invalid("multipliers lack accepted point"))?;
            let finite = point
                .semi_infinite
                .as_ref()
                .ok_or_else(|| invalid("multipliers lack accepted mesh"))?
                .finite;
            let mut groups = validation::split(
                &model,
                &out.multipliers,
                geometry.variables,
                geometry.inequality.rhs.len(),
                geometry.equality.rhs.len(),
                point.inequality.len(),
                point.equality.len(),
                OWNER,
                cooperation,
            )
            .await?;
            // Peak rows are internal discretization constraints, not c(x).
            groups.ineqnonlin.truncate(finite.inequality);
            groups.eqnonlin.truncate(finite.equality);
            let mut fields = runmat_value::UnboundedStructValueBuilder::new();
            for (name, v) in [
                "lower",
                "upper",
                "ineqlin",
                "eqlin",
                "ineqnonlin",
                "eqnonlin",
            ]
            .into_iter()
            .zip([
                groups.lower,
                groups.upper,
                groups.ineqlin,
                groups.eqlin,
                groups.ineqnonlin,
                groups.eqnonlin,
            ]) {
                fields.insert(name, column(v)?);
            }
            values.push(Value::Struct(fields.finish()));
        } else {
            values.push(super::super::empty());
        }
    }
    let seq = if requested.is_none() {
        runmat_value::ValueSequence::single(values.remove(0))
    } else {
        runmat_value::ValueSequence::comma_separated(values)
    }
    .map_err(crate::sequence::sequence_error_to_runtime)?;
    match cooperation.runtime() {
        Some(runtime) => crate::execution::RootedValueSequence::in_context(seq, runtime),
        None => crate::execution::RootedValueSequence::new(seq).map_err(|e| e.into_runtime_error()),
    }
}
