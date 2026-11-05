use super::*;
use crate::context::{
    host_effect::ManagedHostEffectLedger, ParallelCapability, RuntimeParallelResources,
    RuntimeParallelService, RuntimeServicePorts,
};
use std::{cell::Cell, pin::Pin, rc::Rc, task::Poll};

#[test]
fn logical_binary_uses_the_finite_double_output_owner() {
    use crate::context::preparation::{AdmittedBuiltinPreparation, BuiltinPreparationAdmission};
    use runmat_builtins::{BuiltinBindingIdentity, BuiltinPreparationFormId, MINUS_CATALOG_ENTRY};
    use runmat_types::OperatorKind;
    use runmat_value::{IntValue, LogicalArray, Tensor};

    let ledger = ManagedHostEffectLedger::new_with_epoch(1_000_000, 297);
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default().with_parallel(Rc::new(Managed(ledger.clone()))),
        );
    let admission = BuiltinPreparationAdmission {
        binding: BuiltinBindingIdentity {
            builtin: MINUS_CATALOG_ENTRY.identity,
            variant: "default",
        },
        form: BuiltinPreparationFormId(0),
        signature: 0,
        input_count: 2,
        requested_outputs: None,
    };
    for (left, right, expected) in [
        (
            Value::LogicalArray(LogicalArray::new(vec![1, 0, 1, 0], vec![2, 2]).unwrap()),
            Value::Num(5.0),
            vec![-4.0, -5.0, -4.0, -5.0],
        ),
        (
            Value::Num(5.0),
            Value::LogicalArray(LogicalArray::new(vec![1, 0, 1, 0], vec![2, 2]).unwrap()),
            vec![4.0, 5.0, 4.0, 5.0],
        ),
    ] {
        let selected = super::super::selection::prepare(
            Some(&runtime),
            MINUS_CATALOG_ENTRY.identity,
            OperatorKind::Subtract,
            2,
            left.clone(),
            right.clone(),
        )
        .unwrap();
        assert!(matches!(selected.branch, Branch::Admitted(_)));
        let request =
            AdmittedBuiltinPreparation::new(admission, vec![left.clone(), right.clone()]).unwrap();
        let output = request
            .prepare_host_numerical_output(&runtime)
            .unwrap()
            .unwrap();
        let result =
            managed::execute_value(&runtime, output, OperatorKind::Subtract, &left, &right)
                .unwrap();
        let Value::Tensor(result) = result else {
            panic!("finite logical arithmetic changed result carrier");
        };
        assert_eq!(result.as_f64_slice(), Some(expected.as_slice()));
    }
    let logical = Value::LogicalArray(LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap());
    let single = Value::Tensor(Tensor::from_f32(vec![2.0, 3.0], vec![1, 2]).unwrap());
    let selected = super::super::selection::prepare(
        Some(&runtime),
        MINUS_CATALOG_ENTRY.identity,
        OperatorKind::Subtract,
        2,
        logical.clone(),
        single.clone(),
    )
    .unwrap();
    assert!(matches!(selected.branch, Branch::Admitted(_)));
    let output = AdmittedBuiltinPreparation::new(admission, vec![logical.clone(), single.clone()])
        .unwrap()
        .prepare_host_numerical_output(&runtime)
        .unwrap()
        .unwrap();
    let value = managed::execute_value(&runtime, output, OperatorKind::Subtract, &logical, &single)
        .unwrap();
    let Value::Tensor(value) = value else {
        panic!("logical and single retain the inferred single result")
    };
    assert_eq!(value.as_f32_slice(), Some(&[-1.0, -3.0][..]));
    for other in [Value::Int(IntValue::I64(2)), Value::Complex(2.0, 1.0)] {
        let selected = super::super::selection::prepare(
            Some(&runtime),
            MINUS_CATALOG_ENTRY.identity,
            OperatorKind::Subtract,
            2,
            logical.clone(),
            other,
        )
        .unwrap();
        assert!(matches!(selected.branch, Branch::Ordinary));
    }
    assert!(ledger.snapshot().live_claimed_bytes > 0);
    drop(value);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}

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

struct Body {
    ledger: ManagedHostEffectLedger,
    dropped: Rc<Cell<bool>>,
    storage: [u8; 4096],
}
impl Future for Body {
    type Output = BuiltinResult<RootedValue>;
    fn poll(self: Pin<&mut Self>, _: &mut std::task::Context<'_>) -> Poll<Self::Output> {
        assert_eq!(self.storage[0], 17);
        Poll::Pending
    }
}
impl Drop for Body {
    fn drop(&mut self) {
        let snapshot = self.ledger.snapshot();
        assert!(snapshot.live_claimed_bytes + snapshot.live_reserved_bytes > 0);
        self.dropped.set(true);
    }
}

#[test]
fn selected_body_is_small_outside_its_original_funded_owner() {
    for budget in [0, 1_000_000] {
        let ledger = ManagedHostEffectLedger::new_with_epoch(budget, 296);
        let runtime =
            RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
                .with_service_ports(
                    RuntimeServicePorts::default().with_parallel(Rc::new(Managed(ledger.clone()))),
                );
        let constructed = Cell::new(false);
        let dropped = Rc::new(Cell::new(false));
        let inputs =
            crate::execution::RootedValueLease::for_values(std::iter::empty::<&Value>()).unwrap();
        let future = admit_selected(Some(&runtime), inputs, || {
            assert!(!constructed.replace(true), "construct only the chosen body");
            assert!(ledger.snapshot().live_reserved_bytes > 0);
            Body {
                ledger: ledger.clone(),
                dropped: dropped.clone(),
                storage: [17; 4096],
            }
        });
        assert!(std::mem::size_of_val(&future) < std::mem::size_of::<Body>());
        assert_eq!(constructed.get(), budget != 0);
        drop(future);
        assert_eq!(dropped.get(), budget != 0);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }
}
