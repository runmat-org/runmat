use super::*;
use crate::context::host_allocation::HostAllocationPlan;
use crate::context::host_effect::ManagedHostEffectLedger;
use crate::context::{
    ParallelCapability, RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
};
use futures::executor::block_on;
use runmat_value::{ComplexStorage, ComplexTensor, NumericDType, NumericStorage, Tensor};
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

fn matrix(
    dtype: NumericDType,
    complex: bool,
    values: &[(f64, f64)],
    rows: usize,
    cols: usize,
) -> Value {
    if complex {
        Value::ComplexTensor(
            ComplexTensor::from_f64_values_with_dtype(values.to_vec(), vec![rows, cols], dtype)
                .unwrap(),
        )
    } else {
        let storage = match dtype {
            NumericDType::F32 => {
                NumericStorage::F32(values.iter().map(|value| value.0 as f32).collect())
            }
            NumericDType::F64 => NumericStorage::F64(values.iter().map(|value| value.0).collect()),
            _ => unreachable!("fixture uses floating classes"),
        };
        Value::Tensor(Tensor::from_numeric_storage(storage, vec![rows, cols]).unwrap())
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn managed_native_outputs_admit_exact_backing_and_release_after_last_alias() {
    for (dtype, complex) in [
        (NumericDType::F32, false),
        (NumericDType::F64, false),
        (NumericDType::F32, true),
        (NumericDType::F64, true),
    ] {
        let coefficient = if complex { (0.0, 1.0) } else { (1.0, 0.0) };
        let a = matrix(
            dtype,
            complex,
            &[(0.0, 0.0), (-2.0, 0.0), coefficient, (-3.0, 0.0)],
            2,
            2,
        );
        let b = matrix(dtype, complex, &[(0.0, 0.0), (1.0, 0.0)], 2, 1);
        let expected = matrix(
            dtype,
            complex,
            &[(0.0, 0.0), (1.0, 0.0), coefficient, (-3.0, 0.0)],
            2,
            2,
        );
        let bytes = if complex {
            HostAllocationPlan::retained_complex_tensor_bytes(dtype, &[2, 2])
        } else {
            HostAllocationPlan::retained_tensor_bytes(dtype, &[2, 2])
        }
        .unwrap();
        let (context, ledger) = managed_context(bytes);
        let mut result = controllability(&a, &b, Some(&context)).unwrap();
        assert_eq!(result, expected);
        assert_eq!(ledger.snapshot().live_claimed_bytes, bytes);
        assert_eq!(ledger.snapshot().live_reserved_bytes, bytes);
        let alias = result.clone();
        // A shared view must refuse mutation, not silently allocate COW
        // storage outside the exact output's retained claim.
        match &mut result {
            Value::Tensor(tensor) => {
                assert!(tensor.try_as_f32_slice_mut_unique().is_none());
                assert!(tensor.try_as_f64_slice_mut_unique().is_none());
            }
            Value::ComplexTensor(tensor) => {
                assert!(tensor.try_as_f32_slice_mut_unique().is_none());
                assert!(tensor.try_as_f64_slice_mut_unique().is_none());
            }
            _ => panic!("controllability output must retain matrix storage"),
        }
        drop(result);
        assert_eq!(ledger.snapshot().live_claimed_bytes, bytes);
        drop(alias);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

        let (context, ledger) = managed_context(bytes - 1);
        assert!(controllability(&a, &b, Some(&context)).is_err());
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }

    let bytes = HostAllocationPlan::retained_tensor_bytes(NumericDType::F64, &[1, 1]).unwrap();
    let (context, ledger) = managed_context(bytes);
    let result = controllability(&Value::Num(2.0), &Value::Num(3.0), Some(&context)).unwrap();
    let Value::Tensor(result) = result else {
        panic!("scalar inputs produce a matrix")
    };
    assert_eq!(result.as_f64_slice(), Some(&[3.0][..]));
    drop(result);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn two_state_real_and_complex_single_products_keep_order_and_class() {
    // Column-major A=[0 1;-2 -3], B=[0;1] => [B,AB]=[0 1;1 -3].
    let a = Value::Tensor(
        Tensor::from_numeric_storage(NumericStorage::F32(vec![0.0, -2.0, 1.0, -3.0]), vec![2, 2])
            .unwrap(),
    );
    let b = Value::Tensor(
        Tensor::from_numeric_storage(NumericStorage::F32(vec![0.0, 1.0]), vec![2, 1]).unwrap(),
    );
    let Value::Tensor(result) = controllability(&a, &b, None).unwrap() else {
        panic!("real single output");
    };
    assert_eq!(result.as_f32_slice(), Some(&[0.0, 1.0, 1.0, -3.0][..]));
    assert_eq!(result.shape(), &[2, 2]);

    // Multiple inputs and a third power block exercise both assembly strides:
    // A=diag(2,3,4), B=[1 4;2 5;3 6].
    let a_multi = Value::Tensor(
        Tensor::new(
            vec![2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0],
            vec![3, 3],
        )
        .unwrap(),
    );
    let b_multi =
        Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap());
    let Value::Tensor(result) = controllability(&a_multi, &b_multi, None).unwrap() else {
        panic!("real double output");
    };
    assert_eq!(result.shape(), &[3, 6]);
    assert_eq!(
        result.as_f64_slice(),
        Some(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 6.0, 12.0, 8.0, 15.0, 24.0, 4.0, 18.0, 48.0,
                16.0, 45.0, 96.0,
            ][..]
        )
    );

    let a = Value::ComplexTensor(
        ComplexTensor::from_complex_storage(
            ComplexStorage::F32(
                vec![
                    ComplexElement(0.0, 0.0),
                    ComplexElement(-2.0, 0.0),
                    ComplexElement(0.0, 1.0),
                    ComplexElement(-3.0, 0.0),
                ]
                .into(),
            ),
            vec![2, 2],
        )
        .unwrap(),
    );
    let b = Value::ComplexTensor(
        ComplexTensor::from_complex_storage(
            ComplexStorage::F32(vec![ComplexElement(0.0, 0.0), ComplexElement(1.0, 0.0)].into()),
            vec![2, 1],
        )
        .unwrap(),
    );
    let Value::ComplexTensor(result) = controllability(&a, &b, None).unwrap() else {
        panic!("complex single output");
    };
    assert_eq!(
        result.as_f32_slice(),
        Some(
            &[
                ComplexElement(0.0, 0.0),
                ComplexElement(1.0, 0.0),
                ComplexElement(0.0, 1.0),
                ComplexElement(-3.0, 0.0),
            ][..]
        )
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn statement_form_still_validates_arguments() {
    let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    let error = block_on(ctrb_builtin(
        None,
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("missing B must fail even when no result is requested");
    assert_eq!(error.identifier(), Some("RunMat:ctrb:InvalidArgument"));
}
