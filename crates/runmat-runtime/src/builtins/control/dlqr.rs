//! Discrete state-feedback synthesis through the existing LQR DARE owner.
//!
//! The joined LQR request accessor is the single owner of
//! Q/R/N validation, finite solution checks, gain and closed-loop poles.

use runmat_macros::runtime_builtin;
use runmat_value::Value;

use super::lqr::RiccatiApi;
use super::lqr::{LqrDomain, LqrRequest, ManagedRiccatiRequest};
use crate::builtins::control::tf_model::control_error;
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPolicy, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::{BuiltinResult, RuntimeError};

const NAME: &str = "dlqr";

fn invalid(message: impl Into<String>) -> RuntimeError {
    control_error(NAME, "RunMat:dlqr:InvalidArgument", message)
}

fn admission(runtime: Option<&RuntimeContext>, error: HostAllocationError) -> RuntimeError {
    let cancellation = runtime.map(RuntimeContext::cancellation_context);
    crate::builtin::builtin_control_error(cancellation.as_ref(), NAME, error)
}

#[runtime_builtin(
    name = "dlqr",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::control::dlqr"
)]
async fn dlqr_builtin(
    runtime: Option<RuntimeContext>,
    a: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let out_count = crate::output_count::current_output_count();
    let requested = out_count.unwrap_or(1);
    if requested > 3 {
        return Err(invalid("dlqr supports at most three outputs K, S, P"));
    }
    if !(3..=4).contains(&rest.len()) {
        return Err(invalid("dlqr expects (A,B,Q,R[,N])"));
    }
    let policy = HostAllocationReceipt::policy(runtime.as_ref())
        .map_err(|error| admission(runtime.as_ref(), error))?;
    let mut values = rest.into_iter();
    let b = values.next().expect("shape checked B");
    let q = values.next().expect("shape checked Q");
    let r = values.next().expect("shape checked R");
    let n = values.next();
    if policy == HostAllocationPolicy::Managed {
        let runtime = runtime
            .as_ref()
            .ok_or_else(|| admission(None, HostAllocationError::MissingManagedOwner))?;
        return ManagedRiccatiRequest::new(
            a,
            b,
            q,
            Some(r),
            n,
            LqrDomain::Discrete,
            RiccatiApi::Dlqr,
            runtime,
        )?
        .execute(out_count, runtime)
        .await;
    }
    let request = LqrRequest::prepare_standard_matrix(
        a,
        b,
        q,
        Some(r),
        n,
        LqrDomain::Discrete,
        RiccatiApi::Dlqr,
        runtime.as_ref(),
    )
    .await?;
    let solution = request.solve(runtime.as_ref()).await?;
    let outputs = solution.into_outputs(requested)?;
    if let Some(count) = out_count {
        Ok(crate::output_count::output_list_with_padding(
            count, outputs,
        ))
    } else {
        Ok(outputs
            .into_iter()
            .next()
            .expect("one standard DARE output"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn scalar_dare_solution_has_small_residual_and_stable_closed_loop() {
        // A=B=Q=R=1. Positive stabilizing root solves S²-S-1=0;
        // K=S/(1+S), and |A-BK|<1.
        let _outputs = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
        let outputs = block_on(dlqr_builtin(
            None,
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(1.0),
                Value::Num(1.0),
                Value::Num(1.0),
            ]),
        ))
        .expect("standard DARE");
        let Value::OutputList(values) = outputs else {
            panic!("three DARE outputs");
        };
        let Value::Tensor(k) = &values[0] else {
            panic!("gain matrix");
        };
        let gain = k.as_f64_slice().expect("double gain")[0];
        let Value::Tensor(s) = &values[1] else {
            panic!("Riccati matrix");
        };
        let Value::Tensor(poles) = &values[2] else {
            panic!("pole column");
        };
        let solution = s.as_f64_slice().unwrap()[0];
        let expected_s = (1.0 + 5.0_f64.sqrt()) / 2.0;
        assert!((gain - expected_s / (1.0 + expected_s)).abs() < 1e-8);
        assert!((solution * solution - solution - 1.0).abs() < 1e-8);
        assert!((poles.as_f64_slice().unwrap()[0] - (1.0 - gain)).abs() < 1e-8);
        assert!(poles.as_f64_slice().unwrap()[0].abs() < 1.0);
    }
}
