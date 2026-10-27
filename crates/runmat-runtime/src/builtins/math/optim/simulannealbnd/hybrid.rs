//! The selected local solver executes with actual remaining evaluation budget.
use super::{
    common,
    evaluation::{self, Evaluator},
    inputs::{self, Domain},
    invalid,
    options::Options,
    state::State,
    ScalarCooperation,
};
use crate::BuiltinResult;
use runmat_value::Value;
pub(super) struct Selected {
    pub kind: runmat_builtins::OptimHybridSolver,
    pub target: crate::execution::RootedValue,
    pub options: Option<Value>,
}
impl Selected {
    pub fn new(
        hybrid: common::swarm_options::Hybrid,
        c: &ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let target = {
            let _entry = c.runtime().map(crate::context::RuntimeContext::enter);
            crate::canonicalize_callback_handle_for_semantic_resolution(hybrid.target)?
        };
        if !matches!(
            &target,
            Value::BoundFunctionHandle(_)
                | Value::ExternalFunctionHandle(_)
                | Value::MethodFunctionHandle(_)
        ) && !crate::builtin::with_runtime_builtin_bindings_by_name(hybrid.kind.name(), |v| {
            !v.is_empty()
        }) {
            return Err(invalid(format!(
                "selected {} hybrid has no executable binding",
                hybrid.kind.name()
            )));
        }
        Ok(Self {
            kind: hybrid.kind,
            target: crate::execution::RootedValue::in_optional_context(target, c.runtime())?,
            options: hybrid.options,
        })
    }
}
pub(super) fn admit(
    options: &Options,
    domain: &Domain,
    _c: &ScalarCooperation,
) -> BuiltinResult<()> {
    let Some(selected) = &options.hybrid else {
        return Ok(());
    };
    if domain.bounded() && !selected.kind.accepts_bounds() {
        return Err(invalid(
            "bounded HybridFcn must be fmincon or patternsearch",
        ));
    }
    Ok(())
}
pub(super) async fn run(
    state: &mut State,
    domain: &Domain,
    options: &Options,
    evaluator: &mut Evaluator,
    c: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    let Some(selected) = &options.hybrid else {
        return Ok(());
    };
    admit(options, domain, c)?;
    let remaining = options.evaluations.saturating_sub(evaluator.count);
    if remaining == 0 {
        return Ok(());
    }
    let mut args = vec![
        evaluator.function.value().clone(),
        state.best.value().clone(),
    ];
    if selected.kind.accepts_bounds() {
        let Domain::Double { lower, upper, .. } = domain else {
            return Err(invalid("custom data cannot use HybridFcn"));
        };
        for _ in 0..4 {
            args.push(inputs::empty());
        }
        args.push(common::output_callback::vector(lower, &[lower.len(), 1], false, c).await?);
        args.push(common::output_callback::vector(upper, &[upper.len(), 1], false, c).await?);
        args.push(inputs::empty());
    }
    let mut hybrid_options = runmat_value::UnboundedStructValueBuilder::new();
    if let Some(Value::Struct(fields)) = &selected.options {
        for (name, value) in &fields.fields {
            if runmat_builtins::OptimOptionField::parse(name)
                == Some(runmat_builtins::OptimOptionField::MaxFunEvals)
            {
                continue;
            }
            hybrid_options.insert(name, value.clone());
            c.completed(1).await?;
        }
    }
    let configured = if let Some(Value::Struct(fields)) = &selected.options {
        fields
            .fields
            .iter()
            .find_map(|(name, value)| {
                (runmat_builtins::OptimOptionField::parse(name)
                    == Some(runmat_builtins::OptimOptionField::MaxFunEvals))
                .then_some(value)
            })
            .filter(|v| !inputs::is_empty(v))
            .map(|v| {
                common::option_usize_value(
                    "simulannealbnd",
                    Some(v),
                    "hybrid evaluation limit",
                    remaining,
                )
            })
            .transpose()?
            .unwrap_or(remaining)
    } else {
        remaining
    };
    let field = if selected.kind == runmat_builtins::OptimHybridSolver::Fminsearch {
        "MaxFunEvals"
    } else {
        "MaxFunctionEvaluations"
    };
    let limit = configured.min(remaining);
    let mut public_limit = limit as f64;
    // A generated double count may narrow a maximum, never round it above
    // the original exact structural budget.
    if public_limit as u128 > limit as u128 {
        public_limit = public_limit.next_down();
    }
    hybrid_options.insert(field, Value::Num(public_limit));
    args.push(Value::Struct(hybrid_options.finish()));
    let returned = evaluation::invoke(selected.target.value(), &args, 4, c).await?;
    if returned.len() != 4 {
        return Err(invalid("hybrid must return x,fval,exitflag,output"));
    }
    let mut values = {
        let _entry = c.runtime().map(crate::context::RuntimeContext::enter);
        returned.try_into_values()?
    };
    let point = domain
        .validate_trial(values.next().expect("four"), c)
        .await?;
    let fitness = inputs::scalar(&values.next().expect("four"), "hybrid fval")?;
    let flag = inputs::scalar(&values.next().expect("four"), "hybrid exitflag")?;
    if !fitness.is_finite() || !flag.is_finite() {
        return Err(invalid("hybrid fval and flag must be finite"));
    }
    let Value::Struct(output) = values.next().expect("four") else {
        return Err(invalid("hybrid output must be a structure"));
    };
    let actual = output
        .fields
        .get("funcCount")
        .or_else(|| output.fields.get("funccount"))
        .ok_or_else(|| invalid("hybrid output must provide its actual function count"))?;
    let actual =
        common::option_usize_value("simulannealbnd", Some(actual), "hybrid function count", 0)?;
    evaluator.count = evaluator
        .count
        .checked_add(actual)
        .ok_or_else(|| invalid("function count overflow"))?;
    if actual > configured.min(remaining) {
        return Err(invalid(
            "hybrid exceeded its imposed remaining evaluation budget",
        ));
    }
    if fitness < state.best_f {
        state.best_f = fitness;
        state.best =
            crate::execution::RootedValue::in_optional_context(point.value().clone(), c.runtime())?;
    }
    if fitness < state.current_f {
        state.current_f = fitness;
        state.current = point;
    }
    Ok(())
}
