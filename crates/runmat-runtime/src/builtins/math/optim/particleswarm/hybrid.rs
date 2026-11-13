//! The actual selected local solver is called only after swarm convergence.
//! Its original source, options, result lease and call errors are not aliased.
use super::{common, inputs, invalid, ScalarCooperation};
use crate::{execution::RootedValue, BuiltinResult};
use runmat_builtins::OptimHybridSolver;
use runmat_value::Value;
pub(super) struct Selected {
    kind: OptimHybridSolver,
    target: RootedValue,
    options: Option<Value>,
}
pub(super) struct Result {
    pub point: RootedValue,
    pub fitness: f64,
    pub flag: f64,
    pub evaluations: usize,
}
pub(super) fn select(
    hybrid: Option<common::swarm_options::Hybrid>,
    lower: &[f64],
    upper: &[f64],
    cooperation: &ScalarCooperation,
) -> BuiltinResult<Option<Selected>> {
    let Some(hybrid) = hybrid else {
        return Ok(None);
    };
    if !hybrid.kind.accepts_bounds()
        && lower
            .iter()
            .zip(upper)
            .any(|(&a, &b)| a.is_finite() || b.is_finite())
    {
        return Err(invalid(
            "bounded problems require fmincon or patternsearch as HybridFcn",
        ));
    }
    let target = {
        let _scope = cooperation
            .runtime()
            .map(crate::context::RuntimeContext::enter);
        crate::canonicalize_callback_handle_for_semantic_resolution(hybrid.target)?
    };
    // A real selected user/external/method target has its own authority. Only
    // an unresolved builtin spelling depends on a compiled runtime binding.
    if !matches!(
        &target,
        Value::BoundFunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
    ) && !crate::builtin::with_runtime_builtin_bindings_by_name(hybrid.kind.name(), |bindings| {
        !bindings.is_empty()
    }) {
        return Err(invalid(format!(
            "the selected {} hybrid has no executable binding",
            hybrid.kind.name()
        )));
    }
    Ok(Some(Selected {
        kind: hybrid.kind,
        target: RootedValue::in_optional_context(target, cooperation.runtime())?,
        options: hybrid.options,
    }))
}
impl Selected {
    pub async fn run(
        &self,
        objective: &Value,
        point: Value,
        lower: &[f64],
        upper: &[f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Result> {
        let mut arguments = vec![objective.clone(), point];
        if self.kind.accepts_bounds() {
            let lower =
                common::output_callback::vector(lower, &[1, lower.len()], false, cooperation)
                    .await?;
            let upper =
                common::output_callback::vector(upper, &[1, upper.len()], false, cooperation)
                    .await?;
            for _ in 0..4 {
                arguments.push(inputs::empty());
            }
            arguments.push(lower);
            arguments.push(upper);
            arguments.push(inputs::empty());
        }
        arguments.push(self.options.clone().unwrap_or_else(inputs::empty));
        let returned = common::call_function_sequence_in_context(
            cooperation.runtime(),
            self.target.value(),
            arguments,
            4,
        )
        .await?;
        if returned.len() != 4 {
            return Err(invalid("hybrid solver must return x,fval,exitflag,output"));
        }
        // Keep the one original result forest alive through every numerical
        // read and transfer the returned x into its own original-context root.
        let mut values = {
            let _scope = cooperation
                .runtime()
                .map(crate::context::RuntimeContext::enter);
            returned.try_into_values()?
        };
        let x = values.next().expect("four outputs");
        inputs::double(&x, "hybrid x")?;
        let x = match x {
            Value::Num(v) if lower.len() == 1 => {
                Value::Tensor(runmat_value::Tensor::new(vec![v], vec![1, 1]).map_err(invalid)?)
            }
            Value::Tensor(v) if v.len() == lower.len() => {
                Value::Tensor(v.reshape(vec![1, lower.len()]).map_err(invalid)?)
            }
            _ => return Err(invalid("hybrid x must contain nvars doubles")),
        };
        let Value::Tensor(tensor) = &x else {
            unreachable!("row x")
        };
        for (index, &value) in tensor.as_f64_slice().expect("double x").iter().enumerate() {
            if !value.is_finite() || value < lower[index] || value > upper[index] {
                return Err(invalid("hybrid returned a nonfinite or out-of-bounds x"));
            }
            cooperation.completed(1).await?;
        }
        let fitness = scalar(&values.next().expect("fval"), "hybrid fval")?;
        let flag = scalar(&values.next().expect("flag"), "hybrid exitflag")?;
        if !fitness.is_finite() || !flag.is_finite() {
            return Err(invalid("hybrid fval and exitflag must be finite"));
        }
        let output = values.next().expect("output");
        let Value::Struct(output) = &output else {
            return Err(invalid("hybrid output must be a scalar structure"));
        };
        let evaluations = output
            .fields
            .get("funcCount")
            .or_else(|| output.fields.get("funccount"))
            .ok_or_else(|| invalid("hybrid output requires its actual function count"))?;
        let evaluations =
            common::option_usize_value("particleswarm", Some(evaluations), "hybrid funcCount", 0)?;
        let point = RootedValue::in_optional_context(x, cooperation.runtime())?;
        Ok(Result {
            point,
            fitness,
            flag,
            evaluations,
        })
    }
}
fn scalar(value: &Value, role: &str) -> BuiltinResult<f64> {
    inputs::double(value, role)?;
    match value {
        Value::Num(v) => Ok(*v),
        Value::Tensor(v) if v.len() == 1 => Ok(v.as_f64_slice().expect("double scalar")[0]),
        _ => Err(invalid(format!("{role} must be a scalar"))),
    }
}
