use futures::executor::block_on;
use runmat_value::{StringArray, Value};

#[test]
fn managed_string_partition_moves_unique_text_and_refunds_exact_output() {
    use crate::context::host_effect::ManagedHostEffectLedger;
    use crate::context::{
        ParallelCapability, RuntimeContext, RuntimeParallelResources, RuntimeParallelService,
        RuntimeServicePorts,
    };
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
    let ledger = ManagedHostEffectLedger::new(1 << 20);
    let context = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
        );
    let input = StringArray::new(vec!["alpha".into(), "beta".into()], vec![1, 2]).unwrap();
    let source_ptr = input.data()[0]
        .as_str()
        .expect("present text fixture")
        .as_ptr();
    let Value::Cell(output) = block_on(super::super::num2cell_builtin(
        Some(context.clone()),
        Value::StringArray(input),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap() else {
        panic!("expected cell output")
    };
    let Value::String(first) = &output.data()[0] else {
        panic!("expected text child")
    };
    assert_eq!(first.as_str(), Some("alpha"));
    assert_eq!(
        first.as_str().expect("present text fixture").as_ptr(),
        source_ptr
    );
    assert!(first.has_retained_claim());
    let retained = ledger.snapshot().live_reserved_bytes;
    assert!(retained > 0);
    let alias = output.clone();
    drop(output);
    assert_eq!(ledger.snapshot().live_reserved_bytes, retained);
    drop(alias);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

    let grouped = StringArray::new(
        vec!["a".into(), "b".into(), "c".into(), "d".into()],
        vec![2, 2],
    )
    .unwrap();
    let source_ptr = grouped.data()[0]
        .as_str()
        .expect("present text fixture")
        .as_ptr();
    let Value::Cell(output) = block_on(super::super::num2cell_builtin(
        Some(context),
        Value::StringArray(grouped),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
    ))
    .unwrap() else {
        panic!("expected grouped cell output")
    };
    assert_eq!(output.shape.as_slice(), &[1, 2]);
    let Value::StringArray(first) = &output.data()[0] else {
        panic!("expected grouped text child")
    };
    assert_eq!(first.shape.as_slice(), &[2, 1]);
    assert_eq!(
        first.data()[0]
            .as_str()
            .expect("present text fixture")
            .as_ptr(),
        source_ptr
    );
    assert_eq!(first.data()[0].as_str(), Some("a"));
    assert_eq!(first.data()[1].as_str(), Some("b"));
    drop(output);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}
