//! Public positional/problem grammar. Numeric admission, callback dispatch and
//! model construction remain with their existing typed owners.
use super::{invalid, Kind};
use crate::BuiltinResult;
use runmat_value::{Tensor, Value};

pub(crate) enum Goals {
    Minimax,
    Attainment { goal: Value, weight: Value },
}
pub(crate) struct Inputs {
    pub objective: Value,
    pub x0: Value,
    pub goals: Goals,
    pub a: Value,
    pub b: Value,
    pub aeq: Value,
    pub beq: Value,
    pub lower: Value,
    pub upper: Value,
    pub nonlinear: Value,
    pub options: Value,
}
pub(crate) fn empty() -> Value {
    Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("empty shape"))
}

pub(crate) fn parse(kind: Kind, arguments: crate::call::BuiltinArguments) -> BuiltinResult<Inputs> {
    let values = if arguments.len() == 1 {
        let Value::Struct(problem) = arguments.into_iter().next().expect("one argument") else {
            return Err(invalid(
                kind,
                "single-argument form requires a scalar problem structure",
            ));
        };
        let required: &[&str] = match kind {
            Kind::Minimax => &["objective", "x0", "solver", "options"],
            Kind::GoalAttain => &["objective", "x0", "goal", "weight", "solver", "options"],
        };
        for &name in required {
            if !problem.fields.contains_key(name) {
                return Err(invalid(kind, format!("problem requires {name}")));
            }
        }
        let solver = crate::builtins::math::optim::common::field_name(&problem.fields["solver"])?;
        if runmat_builtins::OptimSolver::parse(&solver) != Some(kind.solver()) {
            return Err(invalid(kind, "problem belongs to another solver"));
        }
        let mut values = Vec::new();
        for name in ["objective", "x0"] {
            values.push(problem.fields[name].clone());
        }
        if kind == Kind::GoalAttain {
            values.push(problem.fields["goal"].clone());
            values.push(problem.fields["weight"].clone());
        }
        for name in [
            "Aineq", "bineq", "Aeq", "beq", "lb", "ub", "nonlcon", "options",
        ] {
            values.push(problem.fields.get(name).cloned().unwrap_or_else(empty));
        }
        values
    } else {
        let valid = match kind {
            Kind::Minimax => matches!(arguments.len(), 2 | 4 | 6 | 8 | 9 | 10),
            Kind::GoalAttain => matches!(arguments.len(), 4 | 6 | 8 | 10 | 11 | 12),
        };
        if !valid {
            return Err(invalid(kind, "expected objective/x0, goals and weights when required, complete linear/equality/bound pairs, optional nonlcon and options"));
        }
        let mut values: Vec<_> = arguments.into_iter().collect();
        values.resize_with(if kind == Kind::Minimax { 10 } else { 12 }, empty);
        values
    };
    let mut values = values.into_iter();
    let objective = values.next().expect("objective grammar");
    let x0 = values.next().expect("initial point grammar");
    let goals = match kind {
        Kind::Minimax => Goals::Minimax,
        Kind::GoalAttain => Goals::Attainment {
            goal: values.next().expect("goal grammar"),
            weight: values.next().expect("weight grammar"),
        },
    };
    Ok(Inputs {
        objective,
        x0,
        goals,
        a: values.next().expect("linear inequality grammar"),
        b: values.next().expect("linear rhs grammar"),
        aeq: values.next().expect("equality grammar"),
        beq: values.next().expect("equality rhs grammar"),
        lower: values.next().expect("lower bound grammar"),
        upper: values.next().expect("upper bound grammar"),
        nonlinear: values.next().expect("nonlinear grammar"),
        options: values.next().expect("options grammar"),
    })
}
