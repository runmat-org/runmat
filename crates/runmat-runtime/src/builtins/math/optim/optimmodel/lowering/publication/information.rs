//! Selected solver metadata; ordinary LP fields are conserved.
use super::*;
use crate::builtins::math::optim::{
    linprog::compiled as lp, lsqlin::compiled as lsq, quadprog::compiled as qp,
};
pub(in crate::builtins::math::optim::optimmodel::lowering) async fn publish(
    outcome: &mut report::Outcome,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    if let report::Outcome::Integer(outcome) = outcome {
        let fields = outcome
            .information
            .take()
            .ok_or_else(|| policy.invalid("integer model information was not requested"))?;
        let solver = Value::from("intlinprog");
        let algorithm =
            Value::from(crate::builtins::math::optim::intlinprog::compiled::algorithm());
        return crate::execution::structure::construct_struct_from_fields_in_optional_context(
            c.runtime(),
            fields
                .fields
                .iter()
                .chain(std::iter::once(("Solver", &solver)))
                .chain(std::iter::once(("algorithm", &algorithm))),
        );
    }
    let mut numeric = runmat_value::exact_capacity_vec(6).map_err(|e| policy.storage(e))?;
    let (algorithm, solver, message) = match outcome {
        report::Outcome::Integer(_) => {
            unreachable!("integer information was moved before continuous fields")
        }
        report::Outcome::Quadratic { outcome, timing } => {
            numeric.extend([("iterations",outcome.iterations as f64),("constrviolation",outcome.violation),("firstorderopt",outcome.optimality),("dualitygap",outcome.gap),("wallclocktime",timing.as_ref().ok_or_else(crate::builtins::math::optim::common::solver_timing::SolverTiming::unavailable)?.elapsed().as_secs_f64())]);
            (
                qp::algorithm(outcome),
                "quadprog",
                outcome.message.to_owned(),
            )
        }
        report::Outcome::Linear(out) => {
            numeric.extend([
                ("iterations", out.iterations as f64),
                ("constrviolation", out.constrviolation),
                ("primalfeasibility", out.primalfeasibility),
                ("dualfeasibility", out.dualfeasibility),
                ("dualitygap", out.dualitygap),
                ("firstorderopt", out.firstorderopt),
            ]);
            (lp::algorithm(), "linprog", std::mem::take(&mut out.message))
        }
        report::Outcome::LeastSquares { outcome, timing } => {
            numeric.extend([("iterations",outcome.iterations as f64),
                ("firstorderopt",outcome.optimality),
                ("wallclocktime",timing.as_ref().ok_or_else(crate::builtins::math::optim::common::solver_timing::SolverTiming::unavailable)?.elapsed().as_secs_f64())]);
            match &outcome.method {
                lsq::Method::InteriorPoint => numeric.extend([
                    ("constrviolation", outcome.violation),
                    ("dualitygap", outcome.gap),
                ]),
                lsq::Method::Reflective { cg_iterations } => {
                    numeric.push(("cgiterations", *cg_iterations as f64))
                }
                lsq::Method::ActiveSet { .. } => {
                    numeric.push(("constrviolation", outcome.violation))
                }
            }
            (
                lsq::algorithm(outcome),
                "lsqlin",
                outcome.message.to_owned(),
            )
        }
    };
    let mut fields = Vec::new();
    for (name, value) in numeric {
        fields.try_reserve(1).map_err(|e| policy.reserve(e))?;
        fields.push((
            name,
            RootedValue::in_optional_context(Value::Num(value), c.runtime())?,
        ));
        c.completed(1).await?;
    }
    for (name, value) in [
        ("algorithm", algorithm.to_owned()),
        ("message", message),
        ("Solver", solver.to_owned()),
    ] {
        fields.try_reserve(1).map_err(|e| policy.reserve(e))?;
        fields.push((
            name,
            RootedValue::in_optional_context(
                Value::String(Utf8Text::new(value).into()),
                c.runtime(),
            )?,
        ));
        c.completed(1).await?;
    }
    if matches!(outcome, report::Outcome::LeastSquares { outcome,.. } if matches!(&outcome.method,lsq::Method::InteriorPoint))
    {
        fields.try_reserve(1).map_err(|e| policy.reserve(e))?;
        fields.push((
            "linearsolver",
            RootedValue::in_optional_context(Value::from("sparse"), c.runtime())?,
        ));
    }
    crate::execution::structure::construct_struct_from_fields_in_optional_context(
        c.runtime(),
        fields.iter().map(|(name, value)| (*name, value.value())),
    )
}
