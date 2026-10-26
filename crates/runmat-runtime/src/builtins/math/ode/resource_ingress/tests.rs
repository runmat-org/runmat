use super::*;
use crate::context::host_allocation::HostAllocationError;
use crate::context::host_effect::HostEffectAllocationError;
use crate::context::{
    host_effect::ManagedHostEffectLedger, ParallelCapability, RuntimeParallelResources,
    RuntimeParallelService, RuntimeServicePorts,
};
use runmat_value::{Tensor, Value};
use std::{
    cell::Cell,
    rc::Rc,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

struct Policy {
    finite: Cell<bool>,
    ledger: ManagedHostEffectLedger,
}
impl RuntimeParallelService for Policy {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }
    fn placement_resources(&self) -> RuntimeParallelResources {
        let snapshot = self.ledger.snapshot();
        RuntimeParallelResources {
            memory_available_bytes: self.finite.get().then_some(snapshot.free_bytes()),
            epoch: snapshot.epoch,
            ..RuntimeParallelResources::default()
        }
    }
    fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
        Some(self.ledger.clone())
    }
}

fn assert_unsupported(error: &crate::RuntimeError) {
    assert_eq!(
        error.identifier(),
        Some("RunMat:Placement:InsufficientResourceEvidence")
    );
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert!(matches!(
        std::error::Error::source(error)
            .and_then(|cause| cause.downcast_ref::<HostAllocationError>()),
        Some(HostAllocationError::Resource(
            HostEffectAllocationError::Unavailable
        ))
    ));
}

#[test]
fn finite_ode_frontier_is_unavailable_evidence_not_a_capacity_or_missing_owner_claim() {
    require_unbounded("ode45", None).unwrap();
    for capacity in [0, 8 * 1024 * 1024] {
        let ledger = ManagedHostEffectLedger::new(capacity);
        let policy = Rc::new(Policy {
            finite: Cell::new(false),
            ledger: ledger.clone(),
        });
        let runtime =
            RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
                .with_service_ports(RuntimeServicePorts::default().with_parallel(policy.clone()));
        require_unbounded("ode45", Some(&runtime)).unwrap();
        policy.finite.set(true);
        for name in ["ode45", "ode23", "ode15s"] {
            assert_unsupported(&require_unbounded(name, Some(&runtime)).unwrap_err());
        }
        let snapshot = ledger.snapshot();
        assert_eq!(
            snapshot.peak_reserved_bytes, 0,
            "policy lookup is not an allocation grant"
        );
        assert_eq!(snapshot.peak_claimed_bytes, 0);
    }
}

#[test]
fn all_ode_public_entries_preserve_unbounded_execution_and_refuse_finite_before_rhs() {
    let ledger = ManagedHostEffectLedger::new(8 * 1024 * 1024);
    let policy = Rc::new(Policy {
        finite: Cell::new(false),
        ledger: ledger.clone(),
    });
    let runtime = crate::user_functions::test_support::exact_fixture_runtime()
        .with_service_ports(RuntimeServicePorts::default().with_parallel(policy.clone()));
    let _fixtures = crate::user_functions::test_support::install_exact_named_forwarders(
        &runtime,
        &[("ode_resource_rhs", 952)],
    )
    .unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let callback_calls = calls.clone();
    let entry = runtime.enter();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |function, _, _| {
            assert_eq!(function, 952);
            callback_calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async {
                crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(0.0))
            })
        },
    )))
    .unwrap();
    drop(entry);
    let rhs = Value::FunctionHandle("ode_resource_rhs".into());
    let times = Value::Tensor(Tensor::new(vec![0.0, 0.1], vec![1, 2]).unwrap());
    for (name, args, has_callback) in [
        (
            "ode45",
            vec![rhs.clone(), times.clone(), Value::Num(1.0)],
            true,
        ),
        (
            "ode23",
            vec![rhs.clone(), times.clone(), Value::Num(1.0)],
            true,
        ),
        ("ode15s", vec![rhs.clone(), times, Value::Num(1.0)], true),
        ("fminunc", vec![rhs.clone(), Value::Num(1.0)], true),
        ("fsolve", vec![rhs, Value::Num(1.0)], true),
        (
            "linprog",
            vec![Value::Num(1.0), Value::Num(-1.0), Value::Num(-1.0)],
            false,
        ),
    ] {
        policy.finite.set(false);
        calls.store(0, Ordering::SeqCst);
        let result = futures::executor::block_on(crate::call_builtin_async_in_context(
            &runtime, name, &args,
        ))
        .expect("ordinary public ODE execution remains supported");
        assert_eq!(calls.load(Ordering::SeqCst) > 0, has_callback);
        assert!(match result.value() {
            Value::Num(value) => (*value - 1.0).abs() < 1e-12,
            Value::Tensor(values) => values
                .materialize_f64()
                .iter()
                .all(|value| (*value - 1.0).abs() < 1e-12),
            _ => false,
        });
        drop(result);
        policy.finite.set(true);
        calls.store(0, Ordering::SeqCst);
        let error = futures::executor::block_on(crate::call_builtin_async_in_context(
            &runtime, name, &args,
        ))
        .expect_err("unknown ODE storage cannot enter finite execution");
        assert_unsupported(&error);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "finite refusal precedes RHS effects"
        );
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }
}
