//! Standard continuous algebraic Riccati equation, backed by the shared
//! ordered-QZ LQR substrate.
//!
//! Descriptor pencils, factor outputs and diagnosis reports are not yet
//! exposed by this entry.

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

const NAME: &str = "care";

fn invalid(message: impl Into<String>) -> RuntimeError {
    control_error(NAME, "RunMat:care:InvalidArgument", message)
}

fn unsupported(message: impl Into<String>) -> RuntimeError {
    control_error(NAME, "RunMat:care:UnsupportedForm", message)
}

fn admission(runtime: Option<&RuntimeContext>, error: HostAllocationError) -> RuntimeError {
    let cancellation = runtime.map(RuntimeContext::cancellation_context);
    crate::builtin::builtin_control_error(cancellation.as_ref(), NAME, error)
}

#[runtime_builtin(
    name = "care",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::control::care"
)]
async fn care_builtin(
    runtime: Option<RuntimeContext>,
    a: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let out_count = crate::output_count::current_output_count();
    let requested = out_count.unwrap_or(1);
    if requested > 3 {
        return Err(unsupported(
            "care diagnosis and factor outputs are not yet supported",
        ));
    }
    if !(2..=4).contains(&rest.len()) {
        return Err(if rest.len() > 4 {
            unsupported("care generalized descriptor E and factor forms are not yet supported")
        } else {
            invalid("care expects (A,B,Q[,R[,S]])")
        });
    }
    let policy = HostAllocationReceipt::policy(runtime.as_ref())
        .map_err(|error| admission(runtime.as_ref(), error))?;
    let mut values = rest.into_iter();
    let b = values.next().expect("shape checked B");
    let q = values.next().expect("shape checked Q");
    let r = values.next();
    let s = values.next();
    if policy == HostAllocationPolicy::Managed {
        let runtime = runtime
            .as_ref()
            .ok_or_else(|| admission(None, HostAllocationError::MissingManagedOwner))?;
        return ManagedRiccatiRequest::new(
            a,
            b,
            q,
            r,
            s,
            LqrDomain::Continuous,
            RiccatiApi::Care,
            runtime,
        )?
        .execute(out_count, runtime)
        .await;
    }
    // This is the same validated matrix request and ordered-QZ CARE owner
    // used by lqr. Matrix arguments are moved into that request once.
    let request = LqrRequest::prepare_standard_matrix(
        a,
        b,
        q,
        r,
        s,
        LqrDomain::Continuous,
        RiccatiApi::Care,
        runtime.as_ref(),
    )
    .await?;
    let solution = request.solve(runtime.as_ref()).await?;
    let outputs = solution.into_care_outputs(requested)?;
    if let Some(count) = out_count {
        Ok(crate::output_count::output_list_with_padding(
            count, outputs,
        ))
    } else {
        Ok(outputs
            .into_iter()
            .next()
            .expect("one standard CARE output"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn one_state_standard_solution_satisfies_riccati_residual_and_stability() {
        // A=-1, B=1, Q=1, R=1. The stabilizing CARE root is sqrt(2)-1,
        // gain equals that root, and the closed-loop pole is -sqrt(2).
        let _outputs = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
        let outputs = block_on(care_builtin(
            None,
            Value::Num(-1.0),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(1.0),
                Value::Num(1.0),
                Value::Num(1.0),
            ]),
        ))
        .expect("standard CARE");
        let Value::OutputList(values) = outputs else {
            panic!("three CARE outputs");
        };
        let Value::Tensor(solution) = &values[0] else {
            panic!("matrix output");
        };
        let x = solution.as_f64_slice().expect("double solution")[0];
        assert!((-2.0 * x - x * x + 1.0).abs() < 1e-8);
        let Value::Tensor(poles) = &values[1] else {
            panic!("pole column");
        };
        let Value::Tensor(gain) = &values[2] else {
            panic!("gain matrix");
        };
        assert!((gain.as_f64_slice().unwrap()[0] - x).abs() < 1e-8);
        assert!((poles.as_f64_slice().unwrap()[0] - (-1.0 - x)).abs() < 1e-8);
        assert!(poles.as_f64_slice().unwrap()[0] < 0.0);

        // CARE is an equation solver, not a nonnegative-cost constraint.
        // Negative R and negative Q each have a real stabilizing branch here.
        for (q, r, expected) in [
            (1.0, -1.0, 2.0 - 3.0_f64.sqrt()),
            (-1.0, 1.0, -2.0 + 3.0_f64.sqrt()),
        ] {
            let output = block_on(care_builtin(
                None,
                Value::Num(-2.0),
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::Num(1.0),
                    Value::Num(q),
                    Value::Num(r),
                ]),
            ))
            .expect("standard indefinite CARE has a stabilizing solution");
            let Value::OutputList(values) = output else {
                panic!("three CARE outputs");
            };
            let Value::Tensor(solution) = &values[0] else {
                panic!("matrix solution");
            };
            let x = solution.as_f64_slice().unwrap()[0];
            assert!((x - expected).abs() < 1e-8);
            assert!((-4.0 * x - x * x / r + q).abs() < 1e-8);
            let Value::Tensor(poles) = &values[1] else {
                panic!("pole column");
            };
            let Value::Tensor(gain) = &values[2] else {
                panic!("gain matrix");
            };
            let gain = gain.as_f64_slice().unwrap()[0];
            assert!((gain - x / r).abs() < 1e-8);
            assert!((poles.as_f64_slice().unwrap()[0] - (-2.0 - gain)).abs() < 1e-8);
            assert!(poles.as_f64_slice().unwrap()[0] < 0.0);
        }
    }
}
