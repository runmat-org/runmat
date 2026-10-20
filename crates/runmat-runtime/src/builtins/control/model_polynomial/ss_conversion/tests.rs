use super::*;
use futures::executor::block_on;
use runmat_value::{ComplexTensor, IntegerComplexStorage, IntegerStorage, Tensor};

fn object_with_properties(
    properties: impl IntoIterator<Item = (&'static str, Value)>,
) -> ObjectInstance {
    let mut object = runmat_value::UnboundedObjectBuilder::new("ss");
    for (name, value) in properties {
        object.insert(name.to_owned(), value);
    }
    object.finish()
}

#[test]
fn polynomial_cancellation_keeps_invocation_namespace_and_typed_cause() {
    use crate::context::host_effect::HostEffectAllocationError;
    use crate::execution::{ExecutionServiceError, RuntimeExecutionService};
    use runmat_execution::{cancellation::CancellationSource, CancellationReason};
    let cancellation = CancellationSource::new();
    let runtime = RuntimeContext::new(std::rc::Rc::new(RuntimeExecutionService::new()))
        .with_cancellation_token(cancellation.token());
    runtime.set_error_namespace("ControlSession");
    cancellation.cancel(CancellationReason::User);
    let mut cooperation = PolynomialCooperation::new(Some(&runtime));
    let work_error = block_on(completed_polynomial_work(&mut cooperation, 0, "ss2tf"))
        .expect_err("cancelled invocation cannot enter polynomial work");
    for error in [
        work_error,
        allocation_error("ss2tf", Some(&runtime), HostAllocationError::Cancelled),
        allocation_error(
            "ss2tf",
            Some(&runtime),
            HostAllocationError::Resource(HostEffectAllocationError::Cancelled),
        ),
    ] {
        assert_eq!(error.identifier(), Some("ControlSession:Cancelled"));
        assert!(matches!(
            error
                .source
                .as_deref()
                .and_then(|source| source.downcast_ref::<ExecutionServiceError>()),
            Some(ExecutionServiceError::Cancelled)
        ));
    }
}

#[test]
fn selected_input_uses_one_column_and_all_output_rows() {
    let a = Value::Num(-2.0);
    let b = Value::Tensor(Tensor::new(vec![1.0, 3.0], vec![1, 2]).unwrap());
    let c = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap());
    let d = Value::Tensor(Tensor::new(vec![0.0, 0.0, 1.0, 2.0], vec![2, 2]).unwrap());
    let (num, den) = block_on(selected_input_polynomials(
        [&a, &b, &c, &d],
        1,
        "ss2tf",
        None,
        None,
    ))
    .unwrap();
    assert_eq!(
        den,
        vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)]
    );
    assert_eq!(
        num[0],
        vec![Complex64::new(1.0, 0.0), Complex64::new(5.0, 0.0)]
    );
    assert_eq!(
        num[1],
        vec![Complex64::new(2.0, 0.0), Complex64::new(10.0, 0.0)]
    );
}

#[test]
fn cooperative_polynomial_stepper_matches_synchronous_recurrence() {
    let a = DMatrix::from_column_slice(
        2,
        2,
        &[
            Complex64::new(0.0, 0.0),
            Complex64::new(-3.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(-2.0, 0.0),
        ],
    );
    let b = DMatrix::from_column_slice(2, 1, &[Complex64::new(1.0, 0.0); 2]);
    let c = DMatrix::from_column_slice(1, 2, &[Complex64::new(2.0, 0.0); 2]);
    let denominator = characteristic_polynomial_from_matrix(&a).unwrap();
    let numerator =
        state_space_numerator(&a, &b, &c, Complex64::new(0.5, 0.0), &denominator).unwrap();
    let mut cooperation = PolynomialCooperation::new(None);
    assert_eq!(
        block_on(characteristic_polynomial_cooperative(
            &a,
            &mut cooperation,
            "ss2tf",
            None,
        ))
        .unwrap(),
        denominator
    );
    assert_eq!(
        block_on(state_space_numerator_cooperative(
            &a,
            &b,
            0,
            &c,
            0,
            Complex64::new(0.5, 0.0),
            &denominator,
            &mut cooperation,
            "ss2tf",
            None,
        ))
        .unwrap(),
        numerator
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn state_matrix_property_preserves_rank_one_row_geometry() {
    for value in [
        Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![2]).unwrap()),
        Value::ComplexTensor(ComplexTensor::new(vec![(2.0, 1.0), (3.0, 4.0)], vec![2]).unwrap()),
    ] {
        let object = object_with_properties([("B", value)]);
        let matrix = matrix_property(&object, "B").unwrap();
        assert_eq!(matrix.shape(), (1, 2));
        assert_eq!(matrix[(0, 0)].re, 2.0);
        assert_eq!(matrix[(0, 1)].re, 3.0);
    }
}

#[test]
fn ss_conversion_preflights_then_selectively_downloads_nested_properties() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let state = Tensor::new(vec![-2.0], vec![1, 1]).unwrap();
        let unrelated = Tensor::new(vec![7.0, 8.0], vec![1, 2]).unwrap();
        let resident_state =
            crate::builtins::common::gpu_helpers::upload_tensor(provider, &state).unwrap();
        let resident_unrelated =
            crate::builtins::common::gpu_helpers::upload_tensor(provider, &unrelated).unwrap();

        for builtin in ["pzmap", "rlocus"] {
            let incomplete =
                object_with_properties([("A", Value::GpuTensor(resident_state.clone()))]);
            provider.reset_telemetry();
            let error = block_on(ss_object_to_tf_async(incomplete, builtin))
                .expect_err("missing properties must fail before download");
            assert_eq!(
                error.identifier(),
                Some(if builtin == "pzmap" {
                    "RunMat:pzmap:InvalidModel"
                } else {
                    "RunMat:rlocus:InvalidModel"
                })
            );
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

            let complete = object_with_properties([
                ("A", Value::GpuTensor(resident_state.clone())),
                ("B", Value::Num(1.0)),
                ("C", Value::Num(1.0)),
                ("D", Value::Num(0.0)),
                ("Ts", Value::Num(0.0)),
                ("InputDelay", Value::Num(0.0)),
                ("OutputDelay", Value::Num(0.0)),
                ("Unrelated", Value::GpuTensor(resident_unrelated.clone())),
            ]);
            provider.reset_telemetry();
            let model = block_on(ss_object_to_tf_async(complete, builtin))
                .expect("resident state-space conversion");
            assert_eq!(
                model.denominator,
                vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)]
            );
            assert_eq!(provider.telemetry_snapshot().download_bytes, 8);

            let delayed = object_with_properties([
                ("A", Value::GpuTensor(resident_state.clone())),
                ("B", Value::Num(1.0)),
                ("C", Value::Num(1.0)),
                ("D", Value::Num(0.0)),
                ("Ts", Value::Num(0.0)),
                ("InputDelay", Value::Num(1.0e-14)),
                ("OutputDelay", Value::Num(0.0)),
            ]);
            provider.reset_telemetry();
            let error = block_on(ss_object_to_tf_async(delayed, builtin))
                .expect_err("any nonzero delay is unsupported");
            assert_eq!(
                error.identifier(),
                Some(if builtin == "pzmap" {
                    "RunMat:pzmap:UnsupportedModel"
                } else {
                    "RunMat:rlocus:UnsupportedModel"
                })
            );
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        }
    });
}

#[test]
fn state_space_numerator_retains_small_real_and_complex_gains() {
    let object = object_with_properties([
        ("A", Value::Num(-1.0)),
        ("B", Value::Num(1.0e-7)),
        ("C", Value::Complex(0.0, 1.0e-7)),
        ("D", Value::Num(0.0)),
        ("Ts", Value::Num(0.0)),
        ("InputDelay", Value::Num(0.0)),
        ("OutputDelay", Value::Num(0.0)),
    ]);
    let model = ss_object_to_tf(&object, "pzmap").expect("small nonzero gain");
    assert_eq!(
        model.denominator,
        vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)]
    );
    assert_eq!(model.numerator.len(), 1);
    assert_eq!(model.numerator[0].re, 0.0);
    assert!((model.numerator[0].im - 1.0e-14).abs() < 1.0e-28);
}

#[test]
fn numeric_values_reads_typed_integer_storage_exactly() {
    let tensor = Tensor::new_integer(IntegerStorage::I16(vec![-1, 0, 2]), vec![1, 3])
        .expect("typed integer tensor");

    assert_eq!(
        numeric_values(&Value::Tensor(tensor), "A").expect("numeric values"),
        vec![
            Complex64::new(-1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(2.0, 0.0),
        ]
    );
}

#[test]
fn numeric_values_reads_complex_integer_storage_exactly() {
    let storage = IntegerComplexStorage::new(
        IntegerStorage::I16(vec![1, -2]),
        IntegerStorage::I16(vec![3, -4]),
    )
    .expect("complex integer storage");
    let tensor = ComplexTensor::new_integer(storage, vec![1, 2]).unwrap();

    assert_eq!(
        numeric_values(&Value::ComplexTensor(tensor), "A").expect("numeric values"),
        vec![Complex64::new(1.0, 3.0), Complex64::new(-2.0, -4.0)]
    );
}

#[test]
fn matrix_property_reads_typed_integer_storage_exactly() {
    let matrix = Tensor::new_integer(IntegerStorage::I16(vec![1, 3, 2, 4]), vec![2, 2])
        .expect("typed integer matrix");
    let object = object_with_properties([("A", Value::Tensor(matrix))]);

    let parsed = matrix_property(&object, "A").expect("matrix property");

    assert_eq!(parsed[(0, 0)], Complex64::new(1.0, 0.0));
    assert_eq!(parsed[(1, 0)], Complex64::new(3.0, 0.0));
    assert_eq!(parsed[(0, 1)], Complex64::new(2.0, 0.0));
    assert_eq!(parsed[(1, 1)], Complex64::new(4.0, 0.0));
}

#[test]
fn matrix_property_reads_complex_integer_storage_exactly() {
    let storage = IntegerComplexStorage::new(
        IntegerStorage::I16(vec![1, 3, 2, 4]),
        IntegerStorage::I16(vec![-1, -3, -2, -4]),
    )
    .expect("complex integer storage");
    let matrix = ComplexTensor::new_integer(storage, vec![2, 2]).unwrap();
    let object = object_with_properties([("A", Value::ComplexTensor(matrix))]);

    let parsed = matrix_property(&object, "A").expect("matrix property");

    assert_eq!(parsed[(0, 0)], Complex64::new(1.0, -1.0));
    assert_eq!(parsed[(1, 0)], Complex64::new(3.0, -3.0));
    assert_eq!(parsed[(0, 1)], Complex64::new(2.0, -2.0));
    assert_eq!(parsed[(1, 1)], Complex64::new(4.0, -4.0));
}
