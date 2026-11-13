//! Canonical QN result publication within the final numerical owner's lease.
use super::*;

pub(in crate::builtins::math::optim::fminunc) async fn finalize(
    outcome: MinimizeOutcome,
    shape: &[usize],
    scalar: bool,
    _options: &FminuncOptions,
    cooperation: &mut crate::builtins::math::optim::common::ScalarCooperation,
) -> BuiltinResult<crate::execution::RootedValueSequence> {
    let runtime = cooperation.runtime();
    let requested = match crate::output_count::current_output_count() {
        None => return publish_result(vector_to_value(NAME, outcome.x, shape, scalar)?, runtime),
        Some(0) => return publish_result(Value::output_list(Vec::new()), runtime),
        Some(count @ 1..=6) => count,
        Some(_) => {
            return Err(fminunc_error_with_detail(
                &FMINUNC_ERROR_INVALID_ARGUMENT,
                "too many output arguments; maximum is 6",
            ))
        }
    };
    // Build only requested small metadata before moving the consumed numerical
    // buffers. The final Hessian's original root remains live through publication.
    let metadata = (requested >= 4).then(|| build_output_struct(&outcome));
    let mut outputs = Vec::with_capacity(requested);
    outputs.push(vector_to_value(NAME, outcome.x, shape, scalar)?);
    if requested >= 2 {
        outputs.push(Value::Num(outcome.fval));
    }
    if requested >= 3 {
        outputs.push(Value::Num(outcome.exitflag as f64));
    }
    if let Some(metadata) = metadata {
        outputs.push(Value::Struct(metadata));
    }
    if requested >= 5 {
        outputs.push(vector_to_value(NAME, outcome.grad, shape, scalar)?);
    }
    if requested >= 6 {
        let hessian = outcome.hessian.ok_or_else(|| {
            quasi_newton::hessian_unavailable("no final Hessian estimate was constructed")
        })?;
        return hessian.with_owned(|value| {
            outputs.push(value);
            publish_result(Value::output_list(outputs), runtime)
        });
    }
    publish_result(
        crate::output_count::output_list_with_padding(requested, outputs),
        runtime,
    )
}

pub(in crate::builtins::math::optim::fminunc) fn build_output_struct(
    outcome: &MinimizeOutcome,
) -> StructValue {
    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
    fields.insert("iterations", Value::Num(outcome.iterations as f64));
    fields.insert("funcCount", Value::Num(outcome.func_count as f64));
    fields.insert("algorithm", Value::from(outcome.algorithm));
    fields.insert("firstorderopt", Value::Num(outcome.firstorderopt));
    fields.insert("stepsize", Value::Num(outcome.stepsize));
    fields.insert("message", Value::from(outcome.message.clone()));
    fields.finish()
}

#[cfg(test)]
mod publication_tests {
    use super::*;
    use futures::executor::block_on;

    #[test]
    fn one_output_does_not_invert_unused_hessian() {
        let _outputs = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
        let outcome = MinimizeOutcome {
            x: vec![2.0],
            fval: 0.0,
            grad: vec![0.0],
            hessian: None,
            algorithm: "quasi-newton bfgs",
            iterations: 0,
            func_count: 1,
            exitflag: 1,
            firstorderopt: 0.0,
            stepsize: 0.0,
            message: String::new(),
            termination: TerminationReason::Optimality,
        };
        let options = FminuncOptions {
            tol_x: 1.0e-6,
            tol_fun: 1.0e-6,
            max_iter: 1,
            max_fun_evals: 1,
            display: DisplayMode::Off,
            specify_objective_gradient: false,
            function_tolerance: DEFAULT_TOL_FUN,
            trust_region: None,
            trust_region_fallback: false,
            update: runmat_builtins::OptimQuasiNewtonUpdate::Bfgs,
            objective_limit: -1e20,
            parallel: crate::execution::CallbackParallelMode::Off,
            scaling: Default::default(),
        };
        let mut cooperation = crate::builtins::math::optim::common::ScalarCooperation::new(
            runmat_builtins::OptimSolver::Fminunc,
            None,
        );
        let value = legacy_test_value(
            block_on(finalize(outcome, &[1, 1], true, &options, &mut cooperation)).unwrap(),
        )
        .unwrap();
        assert!(
            matches!(value, Value::OutputList(values) if matches!(values.as_slice(), [Value::Num(number)] if *number == 2.0))
        );
    }
}
