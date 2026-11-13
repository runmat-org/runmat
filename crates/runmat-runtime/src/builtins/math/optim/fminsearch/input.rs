use super::*;
pub(super) struct Input<'a> {
    pub function: &'a Value,
    pub x0: &'a Value,
    pub options: Option<&'a Value>,
}
impl<'a> Input<'a> {
    pub fn parse(args: crate::call::FunctionArgumentView<'a>) -> BuiltinResult<Self> {
        match args.len() {
            1 => {
                let Value::Struct(problem) = args.get(0).expect("admitted problem arity") else {
                    return Err(invalid(
                        "one-input form requires a scalar problem structure",
                    ));
                };
                for name in problem.fields.keys() {
                    if !matches!(name, "objective" | "x0" | "solver" | "options") {
                        return Err(invalid(format!("unknown problem field {name}")));
                    }
                }
                let function = problem
                    .fields
                    .get("objective")
                    .ok_or_else(|| invalid("problem.objective is required"))?;
                let x0 = problem
                    .fields
                    .get("x0")
                    .ok_or_else(|| invalid("problem.x0 is required"))?;
                let solver = problem
                    .fields
                    .get("solver")
                    .and_then(crate::builtins::common::validation::value_text_scalar_string)
                    .ok_or_else(|| invalid("problem.solver must be fminsearch"))?;
                if !solver.eq_ignore_ascii_case(NAME) {
                    return Err(invalid("problem belongs to another solver"));
                }
                let options = problem.fields.get("options");
                validate_function(function)?;
                Ok(Self {
                    function,
                    x0,
                    options,
                })
            }
            2 | 3 => {
                let function = args.get(0).expect("admitted function");
                validate_function(function)?;
                Ok(Self {
                    function,
                    x0: args.get(1).expect("admitted initial point"),
                    options: args.get(2),
                })
            }
            _ => Err(invalid(
                "expected fun,x0,optional options, or one problem structure",
            )),
        }
    }
}
fn validate_function(value: &Value) -> BuiltinResult<()> {
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
        Err(invalid(
            "objective must be a function handle or function name",
        ))
    }
}
pub(super) fn admit_numeric(value: &Value) -> BuiltinResult<()> {
    use crate::builtins::common::residency::{host_gather_intent, HostGatherIntent};
    if host_gather_intent(value)? == HostGatherIntent::ExplicitOrUnproven {
        let mut error=invalid("host fallback requires authenticated automatic residency; use gather explicitly for device operands");
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        return Err(error);
    }
    Ok(())
}
pub(super) async fn initial(
    value: &Value,
    runtime: Option<&crate::context::RuntimeContext>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<common::InitialGuess> {
    admit_numeric(value)?;
    let fact = crate::value_fact::value_fact(value);
    if !matches!(
        fact.kind,
        runmat_types::ValueKindFact::Numeric(runmat_types::NumericFact {
            class: runmat_types::NumericClass::Double,
            domain: runmat_types::NumericDomain::Real
        })
    ) || fact.storage == runmat_types::StorageFact::Sparse
        || fact.shape.element_count() == Some(0)
    {
        return Err(invalid("x0 must be a nonempty finite real double array"));
    }
    if let Some(n) = fact.shape.element_count() {
        simplex::geometry(n)?;
    }
    let value = crate::dispatcher::gather_if_needed_in_context_async(value, runtime).await?;
    let scalar = matches!(value, Value::Num(_));
    let tensor =
        crate::builtins::common::tensor::value_into_tensor_for(NAME, value).map_err(invalid)?;
    if tensor.is_empty() {
        return Err(invalid("x0 must be nonempty"));
    }
    simplex::geometry(tensor.len())?;
    let shape = tensor.shape().to_vec();
    let mut values = buffer(tensor.len())?;
    for first in (0..tensor.len()).step_by(1024) {
        let end = first.saturating_add(1024).min(tensor.len());
        for index in first..end {
            let value = crate::builtins::common::tensor::tensor_value_f64(&tensor, index);
            if !value.is_finite() {
                return Err(invalid("x0 must be finite"));
            }
            values.push(value);
        }
        cooperation.completed((end - first) as u64).await?;
    }
    Ok(common::InitialGuess {
        values,
        shape,
        scalar,
    })
}
