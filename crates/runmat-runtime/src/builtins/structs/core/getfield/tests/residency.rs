use super::*;

#[test]
fn getfield_direct_resident_integer_field_preserves_handle_without_provider_access() {
    let handle = runmat_accelerate_api::GpuTensorHandle {
        shape: vec![2, 1],
        device_id: u32::MAX - 1,
        buffer_id: u64::MAX - 1,
        descriptor: Default::default(),
    }
    .with_numeric_descriptor(
        runmat_accelerate_api::NumericElementType::U64,
        runmat_accelerate_api::GpuTensorStorage::Real,
    );
    let mut st = StructValue::new();
    st.fields
        .insert("values".to_string(), Value::GpuTensor(handle.clone()));

    let result = run_getfield(Value::Struct(st), vec![Value::from("values")])
        .expect("direct resident field");
    assert_eq!(result, Value::GpuTensor(handle.clone()));
}

#[test]
fn getfield_strict_indexed_resident_field_rejects_before_provider_access() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let handle = runmat_accelerate_api::GpuTensorHandle {
        shape: vec![2, 1],
        device_id: u32::MAX - 2,
        buffer_id: u64::MAX - 2,
        descriptor: Default::default(),
    };
    let mut st = StructValue::new();
    st.fields
        .insert("values".to_string(), Value::GpuTensor(handle));
    let selector = CellArray::new_with_shape(vec![Value::Int(IntValue::I32(1))], vec![1, 1])
        .expect("selector");

    let error = run_getfield(
        Value::Struct(st),
        vec![Value::from("values"), Value::Cell(selector)],
    )
    .expect_err("strict indexed resident access");
    assert_eq!(
        error.identifier(),
        GETFIELD_INDEXED_RESIDENT_EXTENSION.error_identifier
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn getfield_gpu_tensor_indexing() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let _ = wgpu_backend::register_wgpu_provider(wgpu_backend::WgpuProviderOptions::default());
    let provider = runmat_accelerate_api::provider().expect("wgpu provider");

    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let view = HostTensorView {
        data: &tensor.materialize_f64(),
        shape: &tensor.shape,
    };
    let handle = provider.upload(&view).expect("upload");

    let mut st = StructValue::new();
    st.fields
        .insert("values".to_string(), Value::GpuTensor(handle.clone()));

    let direct = run_getfield(Value::Struct(st.clone()), vec![Value::from("values")])
        .expect("direct gpu field");
    match direct {
        Value::GpuTensor(out) => assert_eq!(out.buffer_id, handle.buffer_id),
        other => panic!("expected gpu tensor, got {other:?}"),
    }

    let idx_cell = CellArray::new(vec![Value::CharArray(CharArray::new_row("end"))], 1, 1).unwrap();
    let indexed = run_getfield(
        Value::Struct(st),
        vec![Value::from("values"), Value::Cell(idx_cell)],
    )
    .expect("gpu indexed field");
    let Value::Tensor(indexed) = indexed else {
        panic!("expected class-preserving scalar tensor");
    };
    assert_eq!(indexed.shape, vec![1, 1]);
    assert_eq!(indexed.numeric_dtype(), runmat_value::NumericDType::F32);
    assert_eq!(indexed.materialize_f64(), vec![3.0]);
}
