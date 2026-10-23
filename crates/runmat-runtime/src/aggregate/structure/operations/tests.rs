use super::*;
use crate::context::{
    host_effect::ManagedHostEffectLedger, ParallelCapability, RuntimeParallelResources,
    RuntimeParallelService, RuntimeServicePorts,
};
use runmat_execution::{cancellation::CancellationSource, CancellationReason};
use std::rc::Rc;

struct Parallel(ManagedHostEffectLedger);
impl RuntimeParallelService for Parallel {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }
    fn placement_resources(&self) -> RuntimeParallelResources {
        let snapshot = self.0.snapshot();
        RuntimeParallelResources {
            cpu_millicores_available: 1000,
            memory_available_bytes: Some(snapshot.free_bytes()),
            epoch: snapshot.epoch,
        }
    }
    fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
        Some(self.0.clone())
    }
}
fn runtime(capacity: u64) -> (RuntimeContext, ManagedHostEffectLedger, CancellationSource) {
    let ledger = ManagedHostEffectLedger::new_with_epoch(capacity, 1);
    let cancellation = CancellationSource::new();
    let runtime = RuntimeContext::with_cancellation(
        Rc::new(crate::execution::RuntimeExecutionService::new()),
        cancellation.token(),
    )
    .with_service_ports(
        RuntimeServicePorts::default().with_parallel(Rc::new(Parallel(ledger.clone()))),
    );
    (runtime, ledger, cancellation)
}
fn array() -> StructArray {
    let values = (1..=4)
        .map(|number| {
            let mut value = runmat_value::UnboundedStructValueBuilder::new();
            value.insert("value", Value::Num(number as f64));
            value.finish()
        })
        .collect();
    StructArray::new(values, vec![2, 2]).unwrap()
}
fn numbers(value: &StructArray) -> Vec<f64> {
    value
        .field_values("value")
        .unwrap()
        .iter()
        .map(|value| {
            if let Value::Num(value) = value {
                *value
            } else {
                panic!("numeric field expected")
            }
        })
        .collect()
}

#[test]
fn managed_transforms_preserve_column_order_shape_and_original_column_claims() {
    let (runtime, ledger, _) = runtime(1 << 20);
    let _entry = runtime.enter();
    let source = array();
    let original = source.column_owner("value").unwrap().as_ptr();
    let reframed = reshape(source.clone(), vec![1, 4]).unwrap();
    assert_eq!(reframed.shape(), &[1, 4]);
    assert_eq!(reframed.column_owner("value").unwrap().as_ptr(), original);
    assert!(ledger.snapshot().live_reserved_bytes > 0);
    drop(reframed);
    assert!(
        ledger.snapshot().live_reserved_bytes > 0,
        "original column alias retains promotion claim"
    );
    let permuted = permute(source.clone(), &[1, 0]).unwrap();
    assert_eq!(numbers(&permuted), vec![1., 3., 2., 4.]);
    let flipped = flip(source.clone(), 0).unwrap();
    assert_eq!(numbers(&flipped), vec![2., 1., 4., 3.]);
    let shifted = circular_shift(source.clone(), &[1, 0]).unwrap();
    assert_eq!(numbers(&shifted), vec![2., 1., 4., 3.]);
    let repeated = tile(&source, &[1, 2]).unwrap();
    assert_eq!(repeated.shape(), &[2, 4]);
    assert_eq!(numbers(&repeated), vec![1., 2., 3., 4., 1., 2., 3., 4.]);
    let scalar = select(&source, &[2], &[1, 1]).unwrap();
    assert!(
        matches!(scalar,Value::Struct(value)if value.fields.get("value")==Some(&Value::Num(3.)))
    );
    let joined = concatenate(
        1,
        &[
            Value::StructArray(source.clone()),
            Value::StructArray(source.clone()),
        ],
    )
    .unwrap();
    let Value::StructArray(joined) = joined else {
        panic!("array expected")
    };
    assert_eq!(joined.shape(), &[2, 4]);
    assert_eq!(numbers(&joined), vec![1., 2., 3., 4., 1., 2., 3., 4.]);
    {
        use crate::object::indexing::ObjectIndexComponent;
        use crate::sequence::{
            AssignmentStepSpec, SequenceDestinationBuilder, SequenceEndpointSpec,
        };
        let baseline = ledger.snapshot();
        let root = Value::StructArray(source.clone());
        let selectors = crate::object::indexing::array_in_context(Some(&runtime), 2, |axis| {
            Ok(ObjectIndexComponent::Value(Value::Num(if axis == 0 {
                3.
            } else {
                2.
            })))
        })
        .unwrap();
        let mut builder = SequenceDestinationBuilder::new_in_context(&runtime, &root)
            .unwrap()
            .with_assignment_creation(
                runmat_types::AssignmentCreationPolicy::CreateStructFieldPath,
            );
        futures::executor::block_on(builder.push_step_in_context(
            &runtime,
            AssignmentStepSpec::Parentheses { selectors },
            None,
        ))
        .unwrap();
        let prepared = futures::executor::block_on(builder.finish_in_context(
            &runtime,
            SequenceEndpointSpec::Fixed {
                step: AssignmentStepSpec::Member("extra".into()),
                mutation: runmat_types::PlaceMutationKind::MemberAssign,
            },
        ))
        .unwrap();
        let updated = futures::executor::block_on(prepared.assign_fixed_in_context(
            &runtime,
            root,
            Value::Num(13.),
            None,
        ))
        .unwrap();
        let Value::StructArray(updated_array) = updated.value() else {
            panic!("grown structure")
        };
        assert_eq!(updated_array.shape(), &[3, 2]);
        let values = updated_array.field_values("value").unwrap();
        assert_eq!(&values[..2], &[Value::Num(1.), Value::Num(2.)]);
        assert_eq!(&values[3..5], &[Value::Num(3.), Value::Num(4.)]);
        assert!(matches!(&values[2],Value::Tensor(value) if value.is_empty()));
        assert!(matches!(&values[5],Value::Tensor(value) if value.is_empty()));
        let extra = updated_array.field_values("extra").unwrap();
        assert_eq!(extra[5], Value::Num(13.));
        assert!(extra[..5]
            .iter()
            .all(|value| matches!(value,Value::Tensor(value) if value.is_empty())));
        assert_eq!(source.shape(), &[2, 2]);
        assert!(source.field_values("extra").is_none());
        drop(updated);
        assert_eq!(
            ledger.snapshot().live_reserved_bytes,
            baseline.live_reserved_bytes
        );
        assert_eq!(
            ledger.snapshot().live_claimed_bytes,
            baseline.live_claimed_bytes
        );
    }
    drop((permuted, flipped, shifted, repeated, joined, source));
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}
#[test]
fn managed_structure_operations_deny_and_cancel_without_leaked_claims() {
    let source = array();
    let (denied, ledger, _) = runtime(1);
    {
        let _entry = denied.enter();
        assert!(permute(source.clone(), &[1, 0]).is_err());
    }
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    let (runtime, ledger, cancellation) = runtime(1 << 20);
    let _entry = runtime.enter();
    cancellation.cancel(CancellationReason::User);
    let error = flip(source, 0).unwrap_err();
    assert!(error
        .identifier()
        .is_some_and(|identifier| identifier.ends_with(":Cancelled")));
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}
