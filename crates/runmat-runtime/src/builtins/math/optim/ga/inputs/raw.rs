//! Public grammar only. No provider transfer or callback execution occurs here.
use super::super::{invalid, POLICY};
use crate::BuiltinResult;
use runmat_value::{Tensor, Value};

pub(in super::super) struct Raw {
    pub objective: Value,
    pub variables: Value,
    pub inequalities: Value,
    pub inequality_rhs: Value,
    pub equalities: Value,
    pub equality_rhs: Value,
    pub lower: Value,
    pub upper: Value,
    pub nonlinear: Value,
    pub integers: Value,
    pub options: Value,
    pub rng: Option<Value>,
}

pub(in super::super) fn empty() -> Value {
    Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("empty numeric shape"))
}
pub(in super::super) fn unspecified(value: &Value) -> bool {
    matches!(value, Value::Tensor(value) if value.is_empty())
}
pub(in super::super) fn callable(value: &Value) -> bool {
    matches!(
        value,
        Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle(_)
            | Value::Closure(_)
            | Value::String(_)
            | Value::CharArray(_)
    ) || matches!(value, Value::StringArray(value) if value.data().len()==1)
}

/// The R2026a operator problem is a context argument, not a second positional
/// grammar. Canonical FitnessFcn takes precedence over its legacy alias.
pub(in super::super) fn creation_problem(
    variables: Value,
    problem: &Value,
    options: Value,
) -> BuiltinResult<Raw> {
    let Value::Struct(problem) = problem else {
        return Err(invalid("creation requires its scalar problem context"));
    };
    let field = |names: &[&str]| {
        names
            .iter()
            .find_map(|name| problem.fields.get(name))
            .cloned()
            .unwrap_or_else(empty)
    };
    let objective = field(&["FitnessFcn", "fitnessfcn"]);
    if !callable(&objective) {
        return Err(invalid("creation problem requires FitnessFcn"));
    }
    Ok(Raw {
        objective,
        variables,
        inequalities: field(&["Aineq"]),
        inequality_rhs: field(&["Bineq", "bineq"]),
        equalities: field(&["Aeq"]),
        equality_rhs: field(&["Beq", "beq"]),
        lower: field(&["lb"]),
        upper: field(&["ub"]),
        nonlinear: field(&["nonlcon"]),
        integers: field(&["intcon"]),
        options,
        rng: None,
    })
}

#[derive(Clone, Copy)]
enum Field {
    Objective,
    Variables,
    Inequalities,
    InequalityRhs,
    Equalities,
    EqualityRhs,
    Lower,
    Upper,
    Nonlinear,
    Integers,
    Options,
    Rng,
    Solver,
}
impl Field {
    fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "fitnessfcn" => Self::Objective,
            "nvars" => Self::Variables,
            "Aineq" => Self::Inequalities,
            "Bineq" | "bineq" => Self::InequalityRhs,
            "Aeq" => Self::Equalities,
            "Beq" | "beq" => Self::EqualityRhs,
            "lb" => Self::Lower,
            "ub" => Self::Upper,
            "nonlcon" => Self::Nonlinear,
            "intcon" => Self::Integers,
            "options" => Self::Options,
            "rngstate" => Self::Rng,
            "solver" => Self::Solver,
            _ => return None,
        })
    }
    fn slot(self) -> Option<usize> {
        Some(match self {
            Self::Objective => 0,
            Self::Variables => 1,
            Self::Inequalities => 2,
            Self::InequalityRhs => 3,
            Self::Equalities => 4,
            Self::EqualityRhs => 5,
            Self::Lower => 6,
            Self::Upper => 7,
            Self::Nonlinear => 8,
            Self::Integers => 9,
            Self::Options => 10,
            Self::Rng | Self::Solver => return None,
        })
    }
}

pub(in super::super) fn parse(
    args: crate::call::BuiltinArguments,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<Raw> {
    let mut rng = None;
    let values = if args.len() == 1 {
        let Value::Struct(problem) = args.into_iter().next().expect("one argument") else {
            return Err(invalid(
                "single-argument form requires a scalar problem structure",
            ));
        };
        for field in ["fitnessfcn", "nvars", "options"] {
            if !problem.fields.contains_key(field) {
                return Err(invalid(format!("problem requires {field}")));
            }
        }
        let mut values = POLICY.capacity(11, runtime)?;
        values.resize_with(11, empty);
        let mut supplied = [false; 11];
        for (name, value) in &problem.fields {
            let field = Field::parse(name)
                .ok_or_else(|| invalid(format!("unknown ga problem field {name}")))?;
            if let Some(slot) = field.slot() {
                if std::mem::replace(&mut supplied[slot], true) {
                    return Err(invalid(format!("duplicate ga problem role {name}")));
                }
                values[slot] = value.clone();
            } else {
                match field {
                    Field::Rng => rng = Some(value.clone()),
                    Field::Solver => {
                        let name =
                            crate::builtins::common::validation::value_text_scalar_string(value)
                                .ok_or_else(|| invalid("problem solver must be ga"))?;
                        if !name.eq_ignore_ascii_case("ga") {
                            return Err(invalid("problem solver must be ga"));
                        }
                    }
                    _ => unreachable!("slot-bearing role handled above"),
                }
            }
        }
        values
    } else {
        if !matches!(args.len(), 2 | 4 | 6 | 8 | 9 | 10 | 11) {
            return Err(invalid("expected fun,nvars, complete A/b,Aeq/beq,lb/ub pairs, then optional nonlcon,intcon,options"));
        }
        let count = args.len();
        let mut values = POLICY.capacity(11, runtime)?;
        values.extend(args);
        values.resize_with(11, empty);
        // The tenth positional argument is options in the continuous form,
        // intcon in the integer form. Only an actual struct selects options;
        // numeric [] has the same default meaning in either absent role.
        if count == 10 && matches!(values[9], Value::Struct(_)) {
            values.swap(9, 10);
        }
        values
    };
    let mut values = values.into_iter();
    let raw = Raw {
        objective: values.next().expect("objective slot"),
        variables: values.next().expect("variable slot"),
        inequalities: values.next().expect("inequality slot"),
        inequality_rhs: values.next().expect("inequality rhs slot"),
        equalities: values.next().expect("equality slot"),
        equality_rhs: values.next().expect("equality rhs slot"),
        lower: values.next().expect("lower slot"),
        upper: values.next().expect("upper slot"),
        nonlinear: values.next().expect("nonlinear slot"),
        integers: values.next().expect("integer slot"),
        options: values.next().expect("option slot"),
        rng,
    };
    if !callable(&raw.objective) {
        return Err(invalid(
            "fitnessfcn must be one function handle or function name",
        ));
    }
    if !matches!(&raw.options, Value::Struct(_)) && !unspecified(&raw.options) {
        return Err(invalid("options must be a scalar options structure or []"));
    }
    Ok(raw)
}
