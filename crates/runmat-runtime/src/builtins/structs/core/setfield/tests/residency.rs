use super::*;

fn unregistered_resident_handle() -> runmat_accelerate_api::GpuTensorHandle {
    runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![2, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("authentic unregistered allocation")
        .handle
}

#[test]
fn setfield_direct_resident_replacement_preserves_handle_without_provider_access() {
    let handle = unregistered_resident_handle();
    let root = StructValue::new();
    let updated_owner = run_setfield(
        Value::Struct(root),
        vec![Value::from("values"), Value::GpuTensor(handle.clone())],
    )
    .expect("direct resident assignment");
    let Value::Struct(updated) = updated_owner.value() else {
        panic!("expected structure");
    };
    let weak = handle.downgrade();
    assert_eq!(
        updated.fields.get("values"),
        Some(&Value::GpuTensor(handle))
    );
    assert!(weak.acquire().is_ok(), "replacement retains its allocation");
    drop(updated_owner);
    assert!(weak.acquire().is_err());
}

#[test]
fn setfield_strict_indexed_resident_field_rejects_before_provider_access() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let mut root = runmat_value::UnboundedStructValueBuilder::new();
    root.fields.insert(
        "values".to_string(),
        Value::GpuTensor(unregistered_resident_handle()),
    );
    let selector = CellArray::new_with_shape(vec![Value::Int(IntValue::I32(1))], vec![1, 1])
        .expect("selector");
    let error = run_setfield(
        Value::Struct(root.finish()),
        vec![
            Value::from("values"),
            Value::Cell(selector),
            Value::Num(5.0),
        ],
    )
    .expect_err("strict indexed resident assignment");
    assert_eq!(
        error.identifier(),
        SETFIELD_INDEXED_RESIDENT_EXTENSION.error_identifier
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn setfield_gpu_tensor_indexing_gathers_to_host() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    use runmat_accelerate::backend::wgpu::provider::{register_wgpu_provider, WgpuProviderOptions};
    use runmat_accelerate_api::HostTensorView;

    if runmat_accelerate_api::provider().is_none()
        && register_wgpu_provider(WgpuProviderOptions::default()).is_err()
    {
        runmat_accelerate::simple_provider::register_inprocess_provider();
    }

    let provider = runmat_accelerate_api::provider().expect("accel provider");
    let data = [1.0, 2.0, 3.0, 4.0];
    let shape = [2usize, 2usize];
    let view = HostTensorView {
        data: &data,
        shape: &shape,
    };
    let handle = provider.upload(&view).expect("upload");

    let mut root = runmat_value::UnboundedStructValueBuilder::new();
    root.fields
        .insert("values".to_string(), Value::GpuTensor(handle));

    let index_cell = CellArray::new_with_shape(
        vec![Value::Int(IntValue::I32(2)), Value::Int(IntValue::I32(2))],
        vec![1, 2],
    )
    .unwrap();

    let updated = run_setfield(
        Value::Struct(root.finish()),
        vec![
            Value::from("values"),
            Value::Cell(index_cell),
            Value::Num(99.0),
        ],
    )
    .expect("setfield gpu value");

    match updated.value() {
        Value::Struct(st) => {
            let values = st.fields.get("values").expect("values field");
            match values {
                Value::Tensor(tensor) => {
                    assert_eq!(tensor.shape(), &[2, 2]);
                    assert_eq!(tensor.numeric_value_at(3), Some(NumericScalar::F32(99.0)));
                }
                other => panic!("expected tensor after gather, got {other:?}"),
            }
        }
        other => panic!("expected struct result, got {other:?}"),
    }
}
