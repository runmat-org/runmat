use super::*;
use runmat_value::{IntegerStorage, Tensor};

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fgets_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = FGETS_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"tline = fgets(fid)"));
    assert!(labels.contains(&"tline = fgets(fid, nchar)"));
    assert!(labels.contains(&"[tline, terminators] = fgets(fid, ...)"));
}

#[test]
fn fgets_integer_capabilities_cover_file_identifier_and_character_limit() {
    assert_eq!(INTEGER_CAPABILITIES.len(), 2);
    for capability in INTEGER_CAPABILITIES {
        assert_eq!(capability.inputs[0].classes.len(), 8);
        assert_eq!(
            capability.inputs[0].availability,
            BuiltinIntegerInputAvailability::RunMatOnly
        );
    }
}

#[test]
fn fgets_typed_single_and_resident_roles_are_independently_gated() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);

    let integer_id = preflight_fid(&Value::Int(IntValue::I32(3))).unwrap_err();
    assert_eq!(
        integer_id.identifier(),
        Some("RunMat:compatibility:FgetsIntegerIdExtension")
    );
    let integer_nchar = preflight_nchar(&Value::Int(IntValue::I32(3))).unwrap_err();
    assert_eq!(
        integer_nchar.identifier(),
        Some("RunMat:compatibility:FgetsIntegerNcharExtension")
    );

    let single_id = Tensor::new_with_dtype(vec![3.0], vec![1, 1], NumericDType::F32)
        .expect("single identifier");
    let single_id = preflight_fid(&Value::Tensor(single_id)).unwrap_err();
    assert_eq!(
        single_id.identifier(),
        Some("RunMat:compatibility:FgetsSingleIdExtension")
    );
    let single_nchar =
        Tensor::new_with_dtype(vec![3.0], vec![1, 1], NumericDType::F32).expect("single nchar");
    let single_nchar = preflight_nchar(&Value::Tensor(single_nchar)).unwrap_err();
    assert_eq!(
        single_nchar.identifier(),
        Some("RunMat:compatibility:FgetsSingleNcharExtension")
    );

    test_support::with_test_provider(|provider| {
        let host = [3.0f64];
        let view = HostTensorView {
            data: &host,
            shape: &[1, 1],
        };
        let resident_id = Value::GpuTensor(provider.upload(&view).expect("upload identifier"));
        let resident_id = preflight_fid(&resident_id).unwrap_err();
        assert_eq!(
            resident_id.identifier(),
            Some("RunMat:compatibility:FgetsResidentIdExtension")
        );

        let resident_nchar = Value::GpuTensor(provider.upload(&view).expect("upload nchar"));
        let resident_nchar = preflight_nchar(&resident_nchar).unwrap_err();
        assert_eq!(
            resident_nchar.identifier(),
            Some("RunMat:compatibility:FgetsResidentNcharExtension")
        );
    });
}

#[test]
fn fgets_nchar_rejects_nonfinite_fractional_negative_and_out_of_range_doubles() {
    for invalid in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN, -1.0, 1.5] {
        assert!(futures::executor::block_on(parse_nchar(
            crate::call::FunctionArgumentView::from_slice(&[Value::Num(invalid)])
        ))
        .is_err());
    }
    let out_of_range = usize::MAX as f64 + 1.0;
    assert!(futures::executor::block_on(parse_nchar(
        crate::call::FunctionArgumentView::from_slice(&[Value::Num(out_of_range)])
    ))
    .is_err());
}

#[test]
fn typed_fid_and_nchar_parsers_preserve_integer_bounds() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    assert_eq!(parse_fid(&Value::Int(IntValue::U16(7))).unwrap(), 7);
    assert!(parse_fid(&Value::Int(IntValue::U64(u64::MAX))).is_err());

    let fid_tensor =
        Tensor::new_integer(IntegerStorage::U16(vec![7]), vec![1, 1]).expect("typed fid tensor");
    assert_eq!(parse_fid(&Value::Tensor(fid_tensor)).unwrap(), 7);
    let fid_too_large = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
        .expect("typed fid tensor");
    assert!(parse_fid(&Value::Tensor(fid_too_large)).is_err());

    let nchar = futures::executor::block_on(parse_nchar(
        crate::call::FunctionArgumentView::from_slice(&[Value::Int(IntValue::U16(64))]),
    ))
    .unwrap();
    assert_eq!(nchar, Some(64));
    let nchar_tensor =
        Tensor::new_integer(IntegerStorage::U16(vec![64]), vec![1, 1]).expect("typed nchar tensor");
    let nchar_from_tensor = futures::executor::block_on(parse_nchar(
        crate::call::FunctionArgumentView::from_slice(&[Value::Tensor(nchar_tensor)]),
    ))
    .unwrap();
    assert_eq!(nchar_from_tensor, Some(64));
    let negative_nchar_tensor =
        Tensor::new_integer(IntegerStorage::I8(vec![-1]), vec![1, 1]).expect("typed nchar tensor");
    assert!(futures::executor::block_on(parse_nchar(
        crate::call::FunctionArgumentView::from_slice(&[Value::Tensor(negative_nchar_tensor)])
    ))
    .is_err());
    assert!(futures::executor::block_on(parse_nchar(
        crate::call::FunctionArgumentView::from_slice(&[Value::Int(IntValue::I8(-1))])
    ))
    .is_err());
}

#[test]
fn fgets_typed_scalar_parameters_ignore_poisoned_f64_mirrors() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let classes = [
        IntegerStorage::I8(vec![7]),
        IntegerStorage::I16(vec![7]),
        IntegerStorage::I32(vec![7]),
        IntegerStorage::I64(vec![7]),
        IntegerStorage::U8(vec![7]),
        IntegerStorage::U16(vec![7]),
        IntegerStorage::U32(vec![7]),
        IntegerStorage::U64(vec![7]),
    ];
    for storage in classes {
        let tensor = Tensor::new_integer(storage, vec![1, 1]).expect("typed scalar");
        let value = Value::Tensor(tensor);
        assert_eq!(parse_fid(&value).unwrap(), 7);
        assert_eq!(
            futures::executor::block_on(parse_nchar(
                crate::call::FunctionArgumentView::from_slice(&[value])
            ))
            .unwrap(),
            Some(7)
        );
    }
}

#[test]
fn invalid_limit_and_resident_geometry_precede_provider_download() {
    let owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
    let scalar = owner
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("resident identifier");
    let vector = owner
        .allocate_zeroed(
            vec![2, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("resident limit");
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    assert!(
        crate::builtins::common::gpu_helpers::exact_provider_for_handle(&scalar.handle).is_none()
    );

    let invalid_host_limit = run_evaluate(
        &Value::GpuTensor(scalar.handle.clone()),
        &[Value::Num(f64::NAN)],
    )
    .expect_err("invalid limit before identifier gather");
    assert_eq!(
        invalid_host_limit.identifier(),
        FGETS_ERROR_INVALID_INPUT.identifier
    );

    let invalid_resident_limit =
        run_evaluate(&Value::Num(3.0), &[Value::GpuTensor(vector.handle.clone())])
            .expect_err("invalid resident geometry before gather");
    assert_eq!(
        invalid_resident_limit.identifier(),
        FGETS_ERROR_INVALID_INPUT.identifier
    );
    assert_eq!(scalar.probe.retirements(), 0);
    assert_eq!(vector.probe.retirements(), 0);
}
