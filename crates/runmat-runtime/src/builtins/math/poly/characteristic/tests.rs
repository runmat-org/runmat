use super::*;
use crate::context::host_allocation::HostAllocationPlan;
use crate::context::host_effect::ManagedHostEffectLedger;
use crate::context::{
    ParallelCapability, RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
};
use futures::executor::block_on;
use runmat_value::{ComplexElement, ComplexTensor, NumericDType, Tensor};
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

fn managed_context(bytes: u64) -> (RuntimeContext, ManagedHostEffectLedger) {
    let ledger = ManagedHostEffectLedger::new(bytes);
    let context = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
        );
    (context, ledger)
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn roots_and_nondiagonal_characteristic_polynomial() {
    let roots = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
    let Value::Tensor(result) = block_on(poly_builtin(None, roots)).unwrap() else {
        panic!("real coefficient row")
    };
    assert_eq!(result.as_f64_slice().unwrap(), &[1.0, -3.0, 2.0]);

    // Column-major A=[1 2;3 4] has characteristic lambda^2-5lambda-2.
    let matrix = Value::Tensor(Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap());
    let Value::Tensor(result) = block_on(poly_builtin(None, matrix)).unwrap() else {
        panic!("real characteristic row")
    };
    for (actual, expected) in result.as_f64_slice().unwrap().iter().zip([1.0, -5.0, -2.0]) {
        assert!((actual - expected).abs() < 1e-9);
    }

    // A real Schur 2-by-2 conjugate block contributes a real quadratic.
    let rotation = Value::Tensor(Tensor::new(vec![0.0, 1.0, -1.0, 0.0], vec![2, 2]).unwrap());
    let Value::Tensor(result) = block_on(poly_builtin(None, rotation)).unwrap() else {
        panic!("real characteristic row")
    };
    for (actual, expected) in result.as_f64_slice().unwrap().iter().zip([1.0, 0.0, 1.0]) {
        assert!((actual - expected).abs() < 1e-9);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn complex_single_and_empty_keep_domain_shape_and_class() {
    let roots = Value::ComplexTensor(
        ComplexTensor::from_f32(vec![(0.0, 1.0), (0.0, -1.0)], vec![2, 1]).unwrap(),
    );
    let Value::ComplexTensor(result) = block_on(poly_builtin(None, roots)).unwrap() else {
        panic!("complex row")
    };
    assert_eq!(result.numeric_dtype(), NumericDType::F32);
    assert_eq!(result.shape(), &[1, 3]);
    assert_eq!(
        result.as_f32_slice().unwrap(),
        &[
            ComplexElement(1.0, 0.0),
            ComplexElement(0.0, 0.0),
            ComplexElement(1.0, 0.0),
        ]
    );

    let diagonal = Value::ComplexTensor(
        ComplexTensor::from_f32(
            vec![(0.0, 1.0), (0.0, 0.0), (0.0, 0.0), (0.0, -1.0)],
            vec![2, 2],
        )
        .unwrap(),
    );
    let Value::ComplexTensor(result) = block_on(poly_builtin(None, diagonal)).unwrap() else {
        panic!("complex characteristic row")
    };
    assert_eq!(result.numeric_dtype(), NumericDType::F32);
    for (actual, expected) in result.as_f32_slice().unwrap().iter().zip([
        ComplexElement(1.0_f32, 0.0_f32),
        ComplexElement(0.0, 0.0),
        ComplexElement(1.0, 0.0),
    ]) {
        assert!((actual.0 - expected.0).abs() < 1e-5);
        assert!((actual.1 - expected.1).abs() < 1e-5);
    }

    let empty = Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap());
    let Value::Tensor(result) = block_on(poly_builtin(None, empty)).unwrap() else {
        panic!("real constant")
    };
    assert_eq!(result.shape(), &[1, 1]);
    assert_eq!(result.as_f64_slice().unwrap(), &[1.0]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn managed_vector_retains_exact_output_claim_and_matrix_refuses_private_factor() {
    let bytes =
        HostAllocationPlan::retained_complex_tensor_bytes(NumericDType::F32, &[1, 3]).unwrap();
    let (context, ledger) = managed_context(bytes);
    let roots = Value::ComplexTensor(
        ComplexTensor::from_f32(vec![(0.0, 1.0), (0.0, -1.0)], vec![1, 2]).unwrap(),
    );
    let output = block_on(poly_builtin(Some(context.clone()), roots)).unwrap();
    assert_eq!(ledger.snapshot().live_claimed_bytes, bytes);
    drop(output);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    let matrix = Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]).unwrap());
    assert!(block_on(poly_builtin(Some(context), matrix)).is_err());
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
}
