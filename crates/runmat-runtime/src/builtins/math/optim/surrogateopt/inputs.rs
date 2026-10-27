//! Documented positional/problem grammar. All selected numerical metadata is
//! admitted before the first transfer; only one original constrained model
//! is assembled from the actual arrays, without a dense sparse-data mirror.
use super::{
    invalid,
    numeric::{self, DoubleArray, Geometry},
};
use crate::builtins::math::optim::common::retained_control::Retained;
use crate::{
    builtins::math::optim::common::ScalarCooperation, call::BuiltinArguments, BuiltinResult,
};
use runmat_value::{StructValue, Value};
mod linear;

pub(super) struct Raw {
    pub objective: Value,
    pub options: Value,
    pub rng: Option<Value>,
    lower: Value,
    upper: Value,
    integers: Value,
    inequality: Value,
    rhs: Value,
    equality: Value,
    equality_rhs: Value,
}
pub(super) enum Form {
    Solve(Retained<Raw>),
    Checkpoint { path: Value, options: Value },
}
fn text(value: &Value) -> Option<String> {
    crate::builtins::common::validation::value_text_scalar_string(value)
}
fn callable(value: &Value) -> bool {
    matches!(
        value,
        Value::ScopedFunctionHandle(_)
            | Value::FunctionHandle(_)
            | Value::BoundFunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::Closure(_)
    ) || text(value).is_some_and(|name| !name.trim().is_empty())
}
fn empty(cooperation: &ScalarCooperation) -> BuiltinResult<Value> {
    crate::builtins::math::optim::common::output_callback::empty(cooperation.runtime())
}
pub(super) fn grammar(
    args: &BuiltinArguments,
    cooperation: &ScalarCooperation,
) -> BuiltinResult<Form> {
    if matches!(args.len(), 1 | 2) && args.first().is_some_and(|value| text(value).is_some()) {
        return Ok(Form::Checkpoint {
            path: args[0].clone(),
            options: match args.get(1) {
                Some(value) => value.clone(),
                None => empty(cooperation)?,
            },
        });
    }
    let raw = if args.len() == 1 {
        let Value::Struct(problem) = &args[0] else {
            return Err(invalid(
                "one argument must be a problem structure or checkpoint path",
            ));
        };
        problem_form(problem, cooperation)?
    } else {
        if !matches!(args.len(), 3 | 4 | 5 | 8 | 9) {
            return Err(invalid("expected objconstr,lb,ub; optional intcon; full intcon,A,b,Aeq,beq; and optional options"));
        }
        let fourth_options = args.len() == 4 && matches!(args[3], Value::Struct(_));
        Raw {
            objective: args[0].clone(),
            lower: args[1].clone(),
            upper: args[2].clone(),
            integers: if args.len() >= 4 && !fourth_options {
                args[3].clone()
            } else {
                empty(cooperation)?
            },
            inequality: if args.len() >= 8 {
                args[4].clone()
            } else {
                empty(cooperation)?
            },
            rhs: if args.len() >= 8 {
                args[5].clone()
            } else {
                empty(cooperation)?
            },
            equality: if args.len() >= 8 {
                args[6].clone()
            } else {
                empty(cooperation)?
            },
            equality_rhs: if args.len() >= 8 {
                args[7].clone()
            } else {
                empty(cooperation)?
            },
            options: if fourth_options {
                args[3].clone()
            } else if matches!(args.len(), 5 | 9) {
                args[args.len() - 1].clone()
            } else {
                empty(cooperation)?
            },
            rng: None,
        }
    };
    if !callable(&raw.objective) {
        return Err(invalid("objective must be a function name or handle"));
    }
    if !matches!(&raw.options, Value::Struct(_))
        && !matches!(&raw.options, Value::Tensor(value) if value.is_empty())
    {
        return Err(invalid("options must be an options structure or []"));
    }
    for (value, role, geometry) in [
        (&raw.lower, "lb", Geometry::Vector),
        (&raw.upper, "ub", Geometry::Vector),
        (&raw.integers, "intcon", Geometry::Vector),
        (&raw.inequality, "A", Geometry::MatrixAny),
        (&raw.rhs, "b", Geometry::Vector),
        (&raw.equality, "Aeq", Geometry::MatrixAny),
        (&raw.equality_rhs, "beq", Geometry::Vector),
    ] {
        numeric::preflight(value, role, geometry, cooperation)?;
    }
    Ok(Form::Solve(Retained::new(
        raw,
        cooperation,
        super::CONTROL_STORAGE,
    )?))
}
fn problem_form(problem: &StructValue, cooperation: &ScalarCooperation) -> BuiltinResult<Raw> {
    const ALLOWED: &[&str] = &[
        "objective",
        "lb",
        "ub",
        "intcon",
        "Aineq",
        "bineq",
        "Aeq",
        "beq",
        "solver",
        "options",
        "rngstate",
    ];
    if problem.fields.keys().any(|name| !ALLOWED.contains(&name)) {
        return Err(invalid("unsupported surrogateopt problem field"));
    }
    let required = |name: &str| {
        problem
            .fields
            .get(name)
            .cloned()
            .ok_or_else(|| invalid(format!("problem requires {name}")))
    };
    if text(&required("solver")?).is_none_or(|name| !name.eq_ignore_ascii_case("surrogateopt")) {
        return Err(invalid("problem.solver must be surrogateopt"));
    }
    let optional = |name: &str| -> BuiltinResult<Value> {
        match problem.fields.get(name) {
            Some(value) => Ok(value.clone()),
            None => empty(cooperation),
        }
    };
    Ok(Raw {
        objective: required("objective")?,
        lower: required("lb")?,
        upper: required("ub")?,
        options: required("options")?,
        integers: optional("intcon")?,
        inequality: optional("Aineq")?,
        rhs: optional("bineq")?,
        equality: optional("Aeq")?,
        equality_rhs: optional("beq")?,
        rng: problem.fields.get("rngstate").cloned(),
    })
}
