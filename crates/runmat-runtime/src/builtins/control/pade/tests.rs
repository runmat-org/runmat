use super::{output, scalar_delay, scalar_order};
use crate::context::host_allocation::HostAllocationPlan;
use crate::context::host_effect::ManagedHostEffectLedger;
use crate::context::{
    ParallelCapability, RuntimeContext, RuntimeParallelResources, RuntimeParallelService,
    RuntimeServicePorts,
};
use runmat_value::{NumericDType, Tensor, Value};
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

fn pair(value: Value) -> (Vec<f64>, Vec<f64>) {
    let Value::OutputList(values) = value else {
        panic!("two outputs");
    };
    let [Value::Tensor(num), Value::Tensor(den)] = values.as_slice() else {
        panic!("two real coefficient tensors");
    };
    (
        num.as_f64_slice().expect("double numerator").to_vec(),
        den.as_f64_slice().expect("double denominator").to_vec(),
    )
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn pure_delay_coefficients_follow_closed_forms_and_zero_delay() {
    let (num, den) = pair(output::coefficients(1.8, 2, NumericDType::F64, None).unwrap());
    for (actual, expected) in num.iter().zip([1.0, -10.0 / 3.0, 100.0 / 27.0]) {
        assert!((actual - expected).abs() < 1e-12);
    }
    for (actual, expected) in den.iter().zip([1.0, 10.0 / 3.0, 100.0 / 27.0]) {
        assert!((actual - expected).abs() < 1e-12);
    }
    let (first_num, first_den) =
        pair(output::coefficients(2.0, 1, NumericDType::F64, None).unwrap());
    assert_eq!(first_num, vec![-1.0, 1.0]);
    assert_eq!(first_den, vec![1.0, 1.0]);
    let (zero_num, zero_den) = pair(output::coefficients(0.0, 3, NumericDType::F64, None).unwrap());
    assert_eq!(zero_num, vec![1.0]);
    assert_eq!(zero_den, vec![1.0]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn delay_and_order_preserve_typed_scalar_boundaries() {
    let single = Value::Tensor(Tensor::from_f32(vec![1.5], vec![1, 1]).unwrap());
    assert_eq!(scalar_delay(&single).unwrap(), (1.5, NumericDType::F32));
    assert!(scalar_delay(&Value::Num(-1.0)).is_err());
    assert_eq!(scalar_order(&Value::Num(2.0)).unwrap(), 2);
    assert_eq!(
        scalar_order(&Value::Int(runmat_value::IntValue::U64(2))).unwrap(),
        2
    );
    assert!(scalar_order(&Value::Num(1.5)).is_err());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn managed_exact_budget_retains_both_tensors_and_list_until_last_owner() {
    let tensor = HostAllocationPlan::retained_tensor_bytes(NumericDType::F64, &[1, 3]).unwrap();
    let list_physical = runmat_value::OutputListStorage::requested_bytes(2).unwrap() as u64;
    let list = HostAllocationPlan::retained_exact_output_bytes(list_physical, 1).unwrap();
    let budget = tensor * 2 + list;
    let ledger = ManagedHostEffectLedger::new(budget);
    let context = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
        );
    let output = output::coefficients(1.8, 2, NumericDType::F64, Some(&context)).unwrap();
    let Value::OutputList(values) = &output else {
        panic!("admitted two-output list")
    };
    assert!(values.has_retained_claim());
    assert_eq!(ledger.snapshot().live_claimed_bytes, budget);
    let alias = output.clone();
    drop(output);
    assert_eq!(ledger.snapshot().live_claimed_bytes, budget);
    drop(alias);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
}
