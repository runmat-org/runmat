use runmat_value::{ComplexTensor, NumericDType, SparseTensor, Value};

use super::{eps, flintmax, realmax, realmin};

#[test]
fn eps_finite_host_output_claim_is_admitted_before_sparse_expansion() {
    use crate::context::host_effect::ManagedHostEffectLedger;
    use crate::context::{
        ParallelCapability, RuntimeContext, RuntimeParallelResources, RuntimeParallelService,
        RuntimeServicePorts,
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
                ..RuntimeParallelResources::default()
            }
        }
        fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
            Some(self.0.clone())
        }
    }

    let bytes = crate::context::host_allocation::HostAllocationPlan::retained_tensor_bytes(
        NumericDType::F32,
        &[2, 2],
    )
    .unwrap();
    let sparse = || SparseTensor::new_f32(2, 2, vec![0, 1, 1], vec![1], vec![2.0]).unwrap();
    let limited = ManagedHostEffectLedger::new_with_epoch(bytes - 1, 1);
    let runtime = |ledger: ManagedHostEffectLedger| {
        RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
            .with_service_ports(
                RuntimeServicePorts::default().with_parallel(Rc::new(Managed(ledger))),
            )
    };
    assert!(futures::executor::block_on(super::super::eps::value(
        &Value::SparseTensor(sparse()),
        Some(&runtime(limited.clone())),
    ))
    .is_err());
    assert_eq!(limited.snapshot().live_reserved_bytes, 0);

    let ledger = ManagedHostEffectLedger::new_with_epoch(bytes, 1);
    let context = runtime(ledger.clone());
    let output = futures::executor::block_on(super::super::eps::value(
        &Value::SparseTensor(sparse()),
        Some(&context),
    ))
    .unwrap();
    let Value::Tensor(output) = output else {
        panic!("sparse eps output must be dense");
    };
    assert_eq!(
        output.as_f32_slice().unwrap()[1],
        runmat_value::spacing_f32(2.0)
    );
    assert_eq!(ledger.snapshot().live_claimed_bytes, bytes);
    drop(output);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
}

#[test]
fn eps_value_spacing_retains_class_shape_and_complex_magnitude() {
    assert_eq!(eps(vec![]).unwrap(), Value::Num(f64::EPSILON));
    assert_eq!(
        eps(vec![Value::Num(0.0)]).unwrap(),
        Value::Num(f64::from_bits(1))
    );
    assert_eq!(
        eps(vec![Value::Complex(3.0, 4.0)]).unwrap(),
        Value::Num(runmat_value::spacing_f64(5.0))
    );
    let input = runmat_value::Tensor::from_f32(
        vec![0.0, f32::MIN_POSITIVE, -1.0, f32::INFINITY],
        vec![2, 1, 2],
    )
    .unwrap();
    let Value::Tensor(output) = eps(vec![Value::Tensor(input)]).unwrap() else {
        panic!("single eps must preserve tensor storage")
    };
    assert_eq!(output.shape(), &[2, 1, 2]);
    let values = output.as_f32_slice().unwrap();
    assert_eq!(values[0].to_bits(), 1);
    assert_eq!(values[1].to_bits(), 1);
    assert_eq!(values[2], f32::EPSILON);
    assert!(values[3].is_nan());
    let sparse = SparseTensor::new_f32(2, 2, vec![0, 1, 1], vec![1], vec![2.0]).unwrap();
    let Value::Tensor(sparse_spacing) = eps(vec![Value::SparseTensor(sparse)]).unwrap() else {
        panic!("implicit sparse zeros have nonzero spacing and require dense output")
    };
    assert_eq!(sparse_spacing.shape(), &[2, 2]);
    assert_eq!(
        sparse_spacing.as_f32_slice().unwrap(),
        &[
            f32::from_bits(1),
            runmat_value::spacing_f32(2.0),
            f32::from_bits(1),
            f32::from_bits(1)
        ]
    );
    let mut distant_column_ptrs = vec![0; 8_194];
    *distant_column_ptrs.last_mut().unwrap() = 1;
    let distant = SparseTensor::new_f32(1, 8_193, distant_column_ptrs, vec![0], vec![2.0]).unwrap();
    let Value::Tensor(distant_spacing) = eps(vec![Value::SparseTensor(distant)]).unwrap() else {
        panic!("one distant stored entry still produces a dense spacing output")
    };
    let distant_values = distant_spacing.as_f32_slice().unwrap();
    assert_eq!(distant_values[0].to_bits(), 1);
    assert_eq!(distant_values[8_192], runmat_value::spacing_f32(2.0));
    assert!(eps(vec![Value::Num(1.0), Value::Num(2.0)]).is_err());
}

#[test]
fn supports_single_and_double() {
    assert_eq!(realmax(Vec::new()).unwrap(), Value::Num(f64::MAX));
    for (output, expected) in [
        (
            realmin(vec![Value::from("single")]).unwrap(),
            f32::MIN_POSITIVE,
        ),
        (
            flintmax(vec![Value::from("single")]).unwrap(),
            2f32.powi(24),
        ),
    ] {
        let Value::Tensor(output) = output else {
            panic!("single limit must retain native single storage")
        };
        assert_eq!(output.as_f32_slice(), Some([expected].as_slice()));
        assert_eq!(output.shape(), vec![1, 1]);
    }
}

#[test]
fn like_preserves_complexity_and_sparse_single_storage() {
    let complex = ComplexTensor::from_f32(vec![(1.0, 2.0)], vec![1, 1]).unwrap();
    let output = realmax(vec![Value::from("like"), Value::ComplexTensor(complex)])
        .expect("complex single like form");
    let Value::ComplexTensor(output) = output else {
        panic!("expected complex single scalar")
    };
    assert_eq!(output.numeric_dtype(), NumericDType::F32);
    assert_eq!(output.shape(), vec![1, 1]);

    let sparse = SparseTensor::new_f32(2, 2, vec![0, 1, 1], vec![0], vec![3.0]).unwrap();
    let output = realmin(vec![Value::from("like"), Value::SparseTensor(sparse)])
        .expect("sparse single like form");
    let Value::SparseTensor(output) = output else {
        panic!("expected sparse single scalar")
    };
    assert_eq!(output.numeric_dtype(), Some(NumericDType::F32));
    assert_eq!(output.as_f32_slice(), Some([f32::MIN_POSITIVE].as_slice()));
    assert_eq!((output.rows, output.cols), (1, 1));

    let sparse =
        SparseTensor::new_complex_f32(2, 1, vec![0, 1], vec![1], vec![(3.0, -4.0)]).unwrap();
    let output = flintmax(vec![Value::from("like"), Value::SparseTensor(sparse)])
        .expect("sparse complex single like form");
    let Value::SparseTensor(output) = output else {
        panic!("expected sparse complex single scalar")
    };
    assert_eq!(output.numeric_dtype(), Some(NumericDType::F32));
    assert!(output.is_complex());
    assert_eq!(
        output.as_complex_f32_slice(),
        Some([runmat_value::ComplexElement(2f32.powi(24), 0.0)].as_slice())
    );
}
