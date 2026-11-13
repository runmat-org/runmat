use super::*;
use crate::context::host_effect::ManagedHostEffectLedger;
use crate::context::{
    ParallelCapability, RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
};
use runmat_value::{ComplexElement, IntegerComplexStorage, IntegerStorage};
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

fn managed_runtime(ledger: &ManagedHostEffectLedger) -> RuntimeContext {
    RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
        )
}

#[test]
fn exact_numerical_permutations_retain_native_outputs_and_refund() {
    let ledger = ManagedHostEffectLedger::new(1 << 20);
    let runtime = managed_runtime(&ledger);
    let order = [2, 1];

    let integer = Tensor::new_integer(IntegerStorage::U64(vec![1, u64::MAX, 3, 4]), vec![2, 2])
        .expect("integer input");
    let integer = tensor(&runtime, "permute", &integer, &order)
        .expect("managed integer permutation")
        .expect("managed owner");
    assert_eq!(
        integer.integer_storage(),
        Some(&IntegerStorage::U64(vec![1, 3, u64::MAX, 4]))
    );

    let single = ComplexTensor::from_f32(
        vec![(1.0, -1.0), (2.0, -2.0), (3.0, -3.0), (4.0, -4.0)],
        vec![2, 2],
    )
    .expect("complex input");
    let single = complex(&runtime, "permute", &single, &order)
        .expect("managed complex permutation")
        .expect("managed owner");
    assert_eq!(
        single.as_f32_slice(),
        Some(
            [
                ComplexElement(1.0, -1.0),
                ComplexElement(3.0, -3.0),
                ComplexElement(2.0, -2.0),
                ComplexElement(4.0, -4.0),
            ]
            .as_slice()
        )
    );

    let paired = ComplexTensor::new_integer(
        IntegerComplexStorage::new(
            IntegerStorage::U64(vec![1, u64::MAX, 3, 4]),
            IntegerStorage::U64(vec![5, 6, 7, 8]),
        )
        .expect("paired storage"),
        vec![2, 2],
    )
    .expect("paired input");
    let paired = complex(&runtime, "permute", &paired, &order)
        .expect("managed paired permutation")
        .expect("managed owner");
    let paired_storage = paired.integer_storage().expect("paired integer output");
    assert_eq!(
        paired_storage.real(),
        &IntegerStorage::U64(vec![1, 3, u64::MAX, 4])
    );
    assert_eq!(
        paired_storage.imag(),
        &IntegerStorage::U64(vec![5, 7, 6, 8])
    );

    let logical_input = LogicalArray::new(vec![1, 0, 1, 0], vec![2, 2]).expect("logical input");
    let logical = logical(&runtime, "permute", &logical_input, &order)
        .expect("managed logical permutation")
        .expect("managed owner");
    assert_eq!(logical.data.as_slice(), &[1, 1, 0, 0]);

    let empty_input = Tensor::from_f32(Vec::new(), vec![0, 3]).expect("empty input");
    let empty = tensor(&runtime, "permute", &empty_input, &order)
        .expect("managed empty permutation")
        .expect("managed owner");
    assert_eq!(empty.shape(), &[3, 0]);
    assert!(empty.as_f32_slice().is_some_and(<[f32]>::is_empty));

    let retained = ledger.snapshot();
    assert!(retained.live_claimed_bytes > 0);
    assert!(retained.live_reserved_bytes >= retained.live_claimed_bytes);
    drop((integer, single, paired, logical, empty));
    let refunded = ledger.snapshot();
    assert_eq!(refunded.live_claimed_bytes, 0);
    assert_eq!(refunded.live_reserved_bytes, 0);

    let denied = ManagedHostEffectLedger::new(0);
    let denied_runtime = managed_runtime(&denied);
    let input = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("input");
    assert!(tensor(&denied_runtime, "permute", &input, &order).is_err());
    assert_eq!(input.as_f32_slice(), Some([1.0, 2.0, 3.0, 4.0].as_slice()));
    assert_eq!(denied.snapshot().live_claimed_bytes, 0);
    assert_eq!(denied.snapshot().live_reserved_bytes, 0);
}
