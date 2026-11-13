use super::*;
use crate::context::{
    host_effect::ManagedHostEffectLedger, ParallelCapability, RuntimeParallelResources,
    RuntimeParallelService, RuntimeServicePorts,
};
use std::rc::Rc;

struct Managed(ManagedHostEffectLedger);
impl RuntimeParallelService for Managed {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }
    fn placement_resources(&self) -> RuntimeParallelResources {
        let snapshot = self.0.snapshot();
        RuntimeParallelResources {
            memory_available_bytes: Some(snapshot.free_bytes()),
            epoch: snapshot.epoch,
            ..Default::default()
        }
    }
    fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
        Some(self.0.clone())
    }
}

fn runtime(budget: u64) -> (RuntimeContext, ManagedHostEffectLedger) {
    let ledger = ManagedHostEffectLedger::new_with_epoch(budget, 295);
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default().with_parallel(Rc::new(Managed(ledger.clone()))),
        );
    (runtime, ledger)
}

fn program() -> NumericRegionProgram {
    NumericRegionProgram {
        nodes: vec![
            NumericRegionNode::Input(0),
            NumericRegionNode::Constant(2.0),
            NumericRegionNode::Binary {
                operation: NumericBinaryOperation::Multiply,
                left: 0,
                right: 1,
            },
        ],
        outputs: vec![2],
    }
}

#[test]
fn managed_numeric_outputs_retain_exact_payload_shape_and_collection_claims() {
    runmat_gc::gc_test_context(|| {
        let (runtime, ledger) = runtime(1_000_000);
        let input = Value::Tensor(Tensor::new(vec![1.0, -2.0, 3.0], vec![1, 3]).unwrap());
        let NumericRegionOwnedExecution::Completed(outputs) =
            execute_in_context(&program(), &[&input], &runtime).unwrap()
        else {
            panic!("managed numeric execution must remain supported")
        };
        assert!(ledger.snapshot().live_claimed_bytes > 0);
        let Value::Tensor(output) = &outputs.values()[0] else {
            panic!("dense output")
        };
        assert_eq!(output.as_f64_slice().unwrap(), &[2.0, -4.0, 6.0]);
        assert_eq!(output.shape(), &[1, 3]);
        let retained = output.clone();
        drop(outputs);
        assert!(
            ledger.snapshot().live_claimed_bytes > 0,
            "payload and shape claims outlive temporary output slots"
        );
        drop(retained);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    });
}

#[test]
fn finite_denial_and_pre_cancel_do_not_allocate_or_publish_numeric_outputs() {
    runmat_gc::gc_test_context(|| {
        let input = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2]).unwrap());
        let (limited, ledger) = runtime(1);
        let error = execute_in_context(&program(), &[&input], &limited).unwrap_err();
        assert!(std::error::Error::source(&error).is_some());
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        let (cancelled, ledger) = runtime(1_000_000);
        let cancellation = runmat_execution::cancellation::CancellationSource::new();
        let cancelled = cancelled.with_cancellation_token(cancellation.token());
        assert!(cancellation.cancel(runmat_execution::CancellationReason::User));
        assert!(cancelled.is_cancelled());
        assert!(matches!(
            execute_in_context(&program(), &[&input], &cancelled).unwrap(),
            NumericRegionOwnedExecution::Cancelled
        ));
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    });
}

#[test]
fn existing_program_limits_guard_fixed_stack_storage_before_access() {
    let (runtime, ledger) = runtime(1_000_000);
    for program in [
        NumericRegionProgram {
            nodes: vec![NumericRegionNode::Input(0); MAX_NODES + 1],
            outputs: vec![0],
        },
        NumericRegionProgram {
            nodes: vec![NumericRegionNode::Input(0)],
            outputs: vec![0; MAX_OUTPUTS + 1],
        },
    ] {
        assert!(execute_in_context(&program, &[&Value::Num(1.0)], &runtime).is_err());
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    }
}
