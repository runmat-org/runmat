//! Public argument shape only. No callback, provider transfer or capture walk
//! occurs while selecting the original problem fields.
use super::invalid;
use crate::{call::FunctionArgumentView, BuiltinResult};
use runmat_value::Value;

pub(super) struct Input<'a> {
    pub objective: &'a Value,
    pub initial: &'a Value,
    pub inequality: Option<(&'a Value, &'a Value)>,
    pub equality: Option<(&'a Value, &'a Value)>,
    pub lower: Option<&'a Value>,
    pub upper: Option<&'a Value>,
    pub constraints: Option<&'a Value>,
    pub options: Option<&'a Value>,
    pub rngstate: Option<&'a Value>,
}

impl<'a> Input<'a> {
    pub fn parse(args: FunctionArgumentView<'a>) -> BuiltinResult<Self> {
        let input = if args.len() == 1 {
            Self::problem(args.get(0).expect("one-input problem"))?
        } else {
            if !matches!(args.len(), 2 | 4 | 6 | 8 | 9 | 10) {
                return Err(invalid("expected fun,x0 with complete constraint pairs, optional nonlcon/options, or one problem structure"));
            }
            Self {
                objective: args.get(0).expect("objective"),
                initial: args.get(1).expect("initial point"),
                inequality: args.get(2).zip(args.get(3)),
                equality: args.get(4).zip(args.get(5)),
                lower: args.get(6),
                upper: args.get(7),
                constraints: args.get(8).filter(|value| !empty(value)),
                options: args.get(9).filter(|value| !empty(value)),
                rngstate: None,
            }
        };
        callable(input.objective, "objective")?;
        if let Some(constraints) = input.constraints {
            callable(constraints, "nonlcon")?;
        }
        Ok(input)
    }

    fn problem(value: &'a Value) -> BuiltinResult<Self> {
        let Value::Struct(problem) = value else {
            return Err(invalid(
                "one-input form requires a scalar problem structure",
            ));
        };
        for name in problem.fields.keys() {
            if !matches!(
                name,
                "objective"
                    | "x0"
                    | "Aineq"
                    | "bineq"
                    | "Aeq"
                    | "beq"
                    | "lb"
                    | "ub"
                    | "nonlcon"
                    | "solver"
                    | "options"
                    | "rngstate"
            ) {
                return Err(invalid(format!("unknown problem field {name}")));
            }
        }
        let required = |name| {
            problem
                .fields
                .get(name)
                .ok_or_else(|| invalid(format!("problem.{name} is required")))
        };
        let solver =
            crate::builtins::common::validation::value_text_scalar_string(required("solver")?)
                .ok_or_else(|| invalid("problem.solver must be patternsearch"))?;
        if !solver.eq_ignore_ascii_case("patternsearch") {
            return Err(invalid("problem belongs to another solver"));
        }
        Ok(Self {
            objective: required("objective")?,
            initial: required("x0")?,
            inequality: Some((required("Aineq")?, required("bineq")?)),
            equality: Some((required("Aeq")?, required("beq")?)),
            lower: Some(required("lb")?),
            upper: Some(required("ub")?),
            constraints: Some(required("nonlcon")?).filter(|value| !empty(value)),
            options: Some(required("options")?).filter(|value| !empty(value)),
            rngstate: problem.fields.get("rngstate"),
        })
    }
}

pub(super) fn empty(value: &Value) -> bool {
    matches!(value, Value::Tensor(value) if value.is_empty())
}

fn callable(value: &Value, role: &str) -> BuiltinResult<()> {
    if matches!(
        value,
        Value::ScopedFunctionHandle(_)
            | Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle(_)
            | Value::Closure(_)
    ) || crate::builtins::common::validation::value_text_scalar_string(value)
        .is_some_and(|name| !name.trim().is_empty())
    {
        Ok(())
    } else {
        Err(invalid(format!(
            "{role} must be a function handle or function name"
        )))
    }
}
