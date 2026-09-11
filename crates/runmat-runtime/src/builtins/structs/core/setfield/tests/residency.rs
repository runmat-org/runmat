use super::*;

fn fake_resident_handle() -> runmat_accelerate_api::GpuTensorHandle {
    runmat_accelerate_api::GpuTensorHandle {
        shape: vec![2, 1],
        device_id: u32::MAX - 3,
        buffer_id: u64::MAX - 3,
        descriptor: Default::default(),
    }
}

#[test]
fn setfield_direct_resident_replacement_preserves_handle_without_provider_access() {
    let handle = fake_resident_handle();
    let root = StructValue::new();
    let updated = run_setfield(
        Value::Struct(root),
        vec![Value::from("values"), Value::GpuTensor(handle.clone())],
    )
    .expect("direct resident assignment");
    let Value::Struct(updated) = updated else {
        panic!("expected structure");
    };
    assert_eq!(
        updated.fields.get("values"),
        Some(&Value::GpuTensor(handle))
    );
}

#[test]
fn setfield_strict_indexed_resident_field_rejects_before_provider_access() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let mut root = StructValue::new();
    root.fields.insert(
        "values".to_string(),
        Value::GpuTensor(fake_resident_handle()),
    );
    let selector = CellArray::new_with_shape(vec![Value::Int(IntValue::I32(1))], vec![1, 1])
        .expect("selector");
    let error = run_setfield(
        Value::Struct(root),
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

    let mut root = StructValue::new();
    root.fields
        .insert("values".to_string(), Value::GpuTensor(handle));

    let index_cell = CellArray::new_with_shape(
        vec![Value::Int(IntValue::I32(2)), Value::Int(IntValue::I32(2))],
        vec![1, 2],
    )
    .unwrap();

    let updated = run_setfield(
        Value::Struct(root),
        vec![
            Value::from("values"),
            Value::Cell(index_cell),
            Value::Num(99.0),
        ],
    )
    .expect("setfield gpu value");

    match updated {
        Value::Struct(st) => {
            let values = st.fields.get("values").expect("values field");
            match values {
                Value::Tensor(tensor) => {
                    assert_eq!(tensor.shape, vec![2, 2]);
                    assert_eq!(tensor.numeric_value_at(3), Some(NumericScalar::F32(99.0)));
                }
                other => panic!("expected tensor after gather, got {other:?}"),
            }
        }
        other => panic!("expected struct result, got {other:?}"),
    }
}
