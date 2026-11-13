//! One admitted host request shared by the managed lqr/care/dlqr entries.

use runmat_value::Value;

use super::managed_input::ManagedRealMatrix;
use super::managed_output::ManagedRiccatiOutputView;
use super::managed_validation::{
    ManagedValidationPlan, ManagedValidationResult, RiccatiMatrixSlices,
};
use super::managed_weights::ManagedWeightPlan;
use super::{LqrDomain, RiccatiApi, RiccatiIssue};
use crate::builtins::control::tf_model::{state_matrix_dimensions, validate_sample_time, SS_CLASS};
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::BuiltinResult;

pub(in crate::builtins::control) struct ManagedRiccatiRequest {
    a: Value,
    b: Value,
    q: Value,
    r: Option<Value>,
    cross: Option<Value>,
    n: usize,
    m: usize,
    pub(super) domain: LqrDomain,
    pub(super) api: RiccatiApi,
}

impl ManagedRiccatiRequest {
    pub(in crate::builtins::control) async fn execute(
        self,
        requested_outputs: Option<usize>,
        runtime: &RuntimeContext,
    ) -> BuiltinResult<Value> {
        let count = requested_outputs.unwrap_or(1);
        let api = self.api;
        let domain = self.domain;
        let (n, m) = (self.n, self.m);
        let (plan, qz_plan) = self.plan(count, runtime)?;
        let receipt = HostAllocationReceipt::admit(Some(runtime), plan)
            .map_err(|error| api.admission(Some(runtime), error))?
            .ok_or_else(|| {
                api.admission(Some(runtime), HostAllocationError::MissingManagedOwner)
            })?;
        let inputs = self.copy_inputs(&receipt, runtime)?;
        ManagedWeightPlan::new(n, m, api)
            .map_err(|error| api.admission(Some(runtime), error))?
            .validate(
                inputs.q.as_slice(),
                inputs.r.as_slice(),
                inputs.cross.as_slice(),
                &receipt,
                runtime,
            )?;
        let qz_workspace =
            crate::builtins::math::linalg::factor::qz::riccati::ManagedRiccatiWorkspace::prepare(
                &qz_plan, &receipt, runtime,
            )
            .map_err(|error| api.admission(Some(runtime), error))?;
        let qz_domain = match domain {
            LqrDomain::Continuous => {
                crate::builtins::math::linalg::factor::qz::riccati::RiccatiTimeDomain::Continuous
            }
            LqrDomain::Discrete => {
                crate::builtins::math::linalg::factor::qz::riccati::RiccatiTimeDomain::Discrete
            }
        };
        let factor =
            crate::builtins::math::linalg::factor::qz::riccati::stabilizing_subspace_managed(
                inputs,
                qz_plan,
                qz_workspace,
                qz_domain,
                runtime,
                &receipt,
                api.name(),
                |message| api.error(RiccatiIssue::SolverFailed, message),
            )
            .await?;
        let validation_plan = ManagedValidationPlan::new(n, m)
            .map_err(|error| api.admission(Some(runtime), error))?;
        let validation = ManagedValidationResult::prepare(
            &validation_plan,
            RiccatiMatrixSlices {
                a: factor.inputs.a.as_slice(),
                b: factor.inputs.b.as_slice(),
                q: factor.inputs.q.as_slice(),
                r: factor.inputs.r.as_slice(),
                cross: factor.inputs.cross.as_slice(),
                solution: factor.workspace.solution(),
            },
            domain,
            &receipt,
            runtime,
            api,
        )?;
        let (gain, closed_loop, n, m) = validation.into_parts();
        let (factor, closed_loop) =
            crate::builtins::math::linalg::factor::qz::riccati::closed_loop_poles_managed(
                factor,
                closed_loop,
                runtime,
                &receipt,
                api.name(),
                |message| api.error(RiccatiIssue::SolverFailed, message),
            )
            .await?;
        let validation = ManagedValidationResult::from_parts(gain, closed_loop, n, m);
        validation.validate_poles(factor.workspace.poles(), domain, api)?;
        let output = ManagedRiccatiOutputView {
            gain: validation.gain(),
            gain_shape: validation.gain_shape(),
            solution: factor.workspace.solution(),
            solution_shape: [n, n],
            poles: factor.workspace.poles(),
        };
        let outputs = output.publish(api, count, &receipt, runtime)?;
        receipt
            .checkpoint(runtime)
            .map_err(|error| api.admission(Some(runtime), error))?;
        if requested_outputs.is_some() {
            Ok(Value::output_list(outputs))
        } else {
            crate::sequence::into_first_output(outputs)?.ok_or_else(|| {
                api.error(RiccatiIssue::Internal, "Riccati produced no first output")
            })
        }
    }

    pub(super) fn from_lqr(
        first: Value,
        mut rest: crate::call::BuiltinArguments,
        runtime: &RuntimeContext,
    ) -> BuiltinResult<Self> {
        match first {
            Value::Object(object) if object.is_class(SS_CLASS) => {
                let sample_time = super::scalar_property(&object, "Ts")?;
                validate_sample_time(sample_time, "lqr")?;
                let a = object.property("A").cloned().ok_or_else(|| {
                    super::lqr_error(
                        &runmat_builtins::LQR_ERROR_INVALID_MODEL,
                        "ss object missing A property",
                    )
                })?;
                let b = object.property("B").cloned().ok_or_else(|| {
                    super::lqr_error(
                        &runmat_builtins::LQR_ERROR_INVALID_MODEL,
                        "ss object missing B property",
                    )
                })?;
                let q = rest.pop_front().expect("validated Q");
                let r = rest.pop_front().expect("validated R");
                Self::new(
                    a,
                    b,
                    q,
                    Some(r),
                    rest.pop_front(),
                    if sample_time > 0.0 {
                        LqrDomain::Discrete
                    } else {
                        LqrDomain::Continuous
                    },
                    RiccatiApi::Lqr,
                    runtime,
                )
            }
            Value::Object(_) => Err(super::lqr_error(
                &runmat_builtins::LQR_ERROR_UNSUPPORTED_MODEL,
                "lqr(sys,Q,R) requires a state-space ss model",
            )),
            a => {
                let b = rest.pop_front().expect("validated B");
                let q = rest.pop_front().expect("validated Q");
                let r = rest.pop_front().expect("validated R");
                Self::new(
                    a,
                    b,
                    q,
                    Some(r),
                    rest.pop_front(),
                    LqrDomain::Continuous,
                    RiccatiApi::Lqr,
                    runtime,
                )
            }
        }
    }

    pub(in crate::builtins::control) fn new(
        a: Value,
        b: Value,
        q: Value,
        r: Option<Value>,
        cross: Option<Value>,
        domain: LqrDomain,
        api: RiccatiApi,
        runtime: &RuntimeContext,
    ) -> BuiltinResult<Self> {
        for (label, value) in [("A", &a), ("B", &b), ("Q", &q)] {
            host_preflight(label, value, api, runtime)?;
        }
        if let Some(value) = r.as_ref() {
            host_preflight("R", value, api, runtime)?;
        }
        if let Some(value) = cross.as_ref() {
            host_preflight("N", value, api, runtime)?;
        }
        let a_shape = host_shape(&a);
        let b_shape = host_shape(&b);
        let q_shape = host_shape(&q);
        let (n, a_cols) = a_shape;
        let m = b_shape.1;
        if n == 0 || m == 0 || a_cols != n || b_shape.0 != n || q_shape != (n, n) {
            return Err(api.error(
                RiccatiIssue::InvalidDimensions,
                "standard Riccati A, B, Q dimensions do not agree",
            ));
        }
        if r.as_ref().is_some_and(|value| host_shape(value) != (m, m)) {
            return Err(api.error(RiccatiIssue::InvalidDimensions, "R must be m-by-m"));
        }
        if cross
            .as_ref()
            .is_some_and(|value| host_shape(value) != (n, m))
        {
            return Err(api.error(RiccatiIssue::InvalidDimensions, "N must be n-by-m"));
        }
        if !symmetric(&q, n, runtime, api)? {
            return Err(api.error(RiccatiIssue::InvalidDimensions, "Q must be symmetric"));
        }
        if let Some(value) = r.as_ref() {
            if !symmetric(value, m, runtime, api)? {
                return Err(api.error(RiccatiIssue::InvalidDimensions, "R must be symmetric"));
            }
        }
        Ok(Self {
            a,
            b,
            q,
            r,
            cross,
            n,
            m,
            domain,
            api,
        })
    }

    pub(super) fn plan(
        &self,
        count: usize,
        runtime: &RuntimeContext,
    ) -> BuiltinResult<(
        HostAllocationPlan,
        crate::builtins::math::linalg::factor::qz::riccati::RiccatiWorkspacePlan,
    )> {
        let api = self.api;
        let qz = crate::builtins::math::linalg::factor::qz::riccati::RiccatiWorkspacePlan::new(
            self.n, self.m,
        )
        .map_err(|error| api.admission(Some(runtime), error))?;
        let mut input = 0_u64;
        for (rows, cols) in [
            (self.n, self.n),
            (self.n, self.m),
            (self.n, self.n),
            (self.m, self.m),
            (self.n, self.m),
        ] {
            input = input
                .checked_add(
                    ManagedRealMatrix::quoted_bytes(rows, cols)
                        .map_err(|error| api.admission(Some(runtime), error))?,
                )
                .ok_or_else(|| api.admission(Some(runtime), HostAllocationError::Overflow))?;
        }
        let weights = ManagedWeightPlan::new(self.n, self.m, api)
            .map_err(|error| api.admission(Some(runtime), error))?
            .scratch_bytes()
            .map_err(|error| api.admission(Some(runtime), error))?;
        let validation = ManagedValidationPlan::new(self.n, self.m)
            .map_err(|error| api.admission(Some(runtime), error))?
            .scratch_bytes()
            .map_err(|error| api.admission(Some(runtime), error))?;
        let qz_bytes = qz
            .scratch_bytes()
            .map_err(|error| api.admission(Some(runtime), error))?;
        let concurrent = qz_bytes
            .checked_add(validation)
            .ok_or_else(|| api.admission(Some(runtime), HostAllocationError::Overflow))?;
        let scratch = input
            .checked_add(weights.max(concurrent))
            .ok_or_else(|| api.admission(Some(runtime), HostAllocationError::Overflow))?;
        let retained = ManagedRiccatiOutputView::retained_upper_bytes(api, count, self.n, self.m)
            .map_err(|error| api.admission(Some(runtime), error))?;
        let plan = HostAllocationPlan::new(scratch, retained)
            .map_err(|error| api.admission(Some(runtime), error))?;
        Ok((plan, qz))
    }

    pub(super) fn copy_inputs(
        self,
        receipt: &HostAllocationReceipt,
        runtime: &RuntimeContext,
    ) -> BuiltinResult<crate::builtins::math::linalg::factor::qz::riccati::ManagedRiccatiInputs>
    {
        let (n, m, api) = (self.n, self.m, self.api);
        let a = ManagedRealMatrix::from_host_value("A", self.a, n, n, receipt, runtime, api)?;
        let b = ManagedRealMatrix::from_host_value("B", self.b, n, m, receipt, runtime, api)?;
        let q = ManagedRealMatrix::from_host_value("Q", self.q, n, n, receipt, runtime, api)?;
        let r = match self.r {
            Some(value) => {
                ManagedRealMatrix::from_host_value("R", value, m, m, receipt, runtime, api)?
            }
            None => ManagedRealMatrix::implicit(m, m, true, receipt, runtime, api)?,
        };
        let cross = match self.cross {
            Some(value) => {
                ManagedRealMatrix::from_host_value("N", value, n, m, receipt, runtime, api)?
            }
            None => ManagedRealMatrix::implicit(n, m, false, receipt, runtime, api)?,
        };
        Ok(
            crate::builtins::math::linalg::factor::qz::riccati::ManagedRiccatiInputs {
                a: a.into_parts().2,
                b: b.into_parts().2,
                q: q.into_parts().2,
                r: r.into_parts().2,
                cross: cross.into_parts().2,
            },
        )
    }
}

fn host_preflight(
    label: &'static str,
    value: &Value,
    api: RiccatiApi,
    runtime: &RuntimeContext,
) -> BuiltinResult<()> {
    match value {
        Value::Tensor(tensor) => {
            if tensor.shape().len() > 2 {
                return Err(api.error(
                    RiccatiIssue::InvalidDimensions,
                    format!("{label} must be a 2-D matrix"),
                ));
            }
            tensor
                .validate_shape()
                .map_err(|message| api.error(RiccatiIssue::InvalidDimensions, message))?;
            for start in (0..tensor.len()).step_by(4096) {
                preflight_checkpoint(runtime, api)?;
                for index in start..start.saturating_add(4096).min(tensor.len()) {
                    let element = tensor.numeric_value_at(index).ok_or_else(|| {
                        api.error(
                            RiccatiIssue::InvalidDimensions,
                            format!("{label} storage changed"),
                        )
                    })?;
                    if !element.materialize_f64().is_finite() {
                        return Err(api.error(
                            RiccatiIssue::UnsupportedInput,
                            format!("{label} must contain finite real values"),
                        ));
                    }
                }
            }
            Ok(())
        }
        Value::Num(number) if number.is_finite() => Ok(()),
        Value::Int(_) => Ok(()),
        Value::GpuTensor(_) => {
            Err(api.admission(Some(runtime), HostAllocationError::MissingManagedOwner))
        }
        Value::Num(_) => Err(api.error(
            RiccatiIssue::UnsupportedInput,
            format!("{label} must contain finite real values"),
        )),
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(api.error(
            RiccatiIssue::UnsupportedInput,
            format!("{label} must contain real numeric values"),
        )),
        _ => Err(api.error(
            RiccatiIssue::UnsupportedInput,
            format!("{label} must be a real numeric matrix"),
        )),
    }
}

fn preflight_checkpoint(runtime: &RuntimeContext, api: RiccatiApi) -> BuiltinResult<()> {
    if runtime.is_cancelled() {
        return Err(api.admission(Some(runtime), HostAllocationError::Cancelled));
    }
    HostAllocationReceipt::policy(Some(runtime))
        .map(|_| ())
        .map_err(|error| api.admission(Some(runtime), error))
}

fn host_shape(value: &Value) -> (usize, usize) {
    match value {
        Value::Tensor(tensor) => state_matrix_dimensions(tensor),
        Value::Num(_) | Value::Int(_) => (1, 1),
        _ => (0, 0), // host_preflight rejects nonmatrix classes first
    }
}

fn element(value: &Value, index: usize) -> f64 {
    match value {
        Value::Tensor(tensor) => tensor
            .numeric_value_at(index)
            .expect("validated shape")
            .materialize_f64(),
        Value::Num(number) => *number,
        Value::Int(integer) => integer.to_f64(),
        _ => unreachable!("validated host numeric matrix"),
    }
}

fn symmetric(
    value: &Value,
    n: usize,
    runtime: &RuntimeContext,
    api: RiccatiApi,
) -> BuiltinResult<bool> {
    let mut checked = 0_usize;
    for col in 0..n {
        for row in (col + 1)..n {
            checked += 1;
            if checked == 4096 {
                preflight_checkpoint(runtime, api)?;
                checked = 0;
            }
            let a = element(value, col * n + row);
            let b = element(value, row * n + col);
            let scale = a.abs().max(b.abs()).max(1.0);
            if (a - b).abs() > 1.0e-8 * scale {
                return Ok(false);
            }
        }
    }
    preflight_checkpoint(runtime, api)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::host_effect::ManagedHostEffectLedger;
    use crate::context::{
        ParallelCapability, RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
    };
    use futures::executor::block_on;
    use std::rc::Rc;

    struct ManagedParallel(ManagedHostEffectLedger);

    impl RuntimeParallelService for ManagedParallel {
        fn supports(&self, _: ParallelCapability) -> bool {
            false
        }
        fn placement_resources(&self) -> RuntimeParallelResources {
            let snapshot = self.0.snapshot();
            RuntimeParallelResources {
                memory_available_bytes: Some(snapshot.free_bytes()),
                epoch: snapshot.epoch,
                ..RuntimeParallelResources::default()
            }
        }
        fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
            Some(self.0.clone())
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn managed_scalar_care_retains_exact_outputs_and_rejects_underbudget() {
        // A=-1, B=Q=R=1: S=K=sqrt(2)-1 and the pole is -sqrt(2).
        let ledger = ManagedHostEffectLedger::new(1_000_000);
        let context =
            RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
                .with_service_ports(
                    RuntimeServicePorts::default()
                        .with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
                );
        let request = ManagedRiccatiRequest::new(
            Value::Num(-1.0),
            Value::Num(1.0),
            Value::Num(1.0),
            Some(Value::Num(1.0)),
            None,
            LqrDomain::Continuous,
            RiccatiApi::Care,
            &context,
        )
        .unwrap();
        let (plan, _) = request.plan(3, &context).unwrap();
        let result = block_on(request.execute(Some(3), &context)).expect("managed CARE");
        let Value::OutputList(outputs) = result else {
            panic!("three outputs")
        };
        let Value::Tensor(solution) = &outputs[0] else {
            panic!("solution tensor")
        };
        let Value::Tensor(poles) = &outputs[1] else {
            panic!("pole tensor")
        };
        let Value::Tensor(gain) = &outputs[2] else {
            panic!("gain tensor")
        };
        let expected = 2.0_f64.sqrt() - 1.0;
        assert!((solution.as_f64_slice().unwrap()[0] - expected).abs() < 1.0e-8);
        assert!((gain.as_f64_slice().unwrap()[0] - expected).abs() < 1.0e-8);
        assert!((poles.as_f64_slice().unwrap()[0] + 2.0_f64.sqrt()).abs() < 1.0e-8);
        assert!(ledger.snapshot().live_claimed_bytes > 0);
        drop(outputs);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

        let low = ManagedHostEffectLedger::new(
            plan.scratch_peak_bytes() + plan.retained_output_bytes() - 1,
        );
        let low_context =
            RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
                .with_service_ports(
                    RuntimeServicePorts::default()
                        .with_parallel(Rc::new(ManagedParallel(low.clone()))),
                );
        let request = ManagedRiccatiRequest::new(
            Value::Num(-1.0),
            Value::Num(1.0),
            Value::Num(1.0),
            Some(Value::Num(1.0)),
            None,
            LqrDomain::Continuous,
            RiccatiApi::Care,
            &low_context,
        )
        .unwrap();
        assert!(block_on(request.execute(Some(3), &low_context)).is_err());
        assert_eq!(low.snapshot().live_reserved_bytes, 0);
    }
}
