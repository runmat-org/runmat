use super::*;

#[test]
fn getfield_direct_resident_integer_field_preserves_handle_without_provider_access() {
    let handle = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![2, 1],
            runmat_accelerate_api::NumericElementType::U64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("authentic unregistered allocation")
        .handle;
    let mut st = runmat_value::UnboundedStructValueBuilder::new();
    st.fields
        .insert("values".to_string(), Value::GpuTensor(handle.clone()));

    let result = run_getfield(Value::Struct(st.finish()), vec![Value::from("values")])
        .expect("direct resident field");
    assert_eq!(result.value(), &Value::GpuTensor(handle.clone()));
    let weak = handle.downgrade();
    drop(handle);
    assert!(
        weak.acquire().is_ok(),
        "returned field retains its allocation"
    );
    drop(result);
    assert!(weak.acquire().is_err());
}

#[test]
fn getfield_strict_indexed_resident_field_rejects_before_provider_access() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let handle = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![2, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("authentic unregistered allocation")
        .handle;
    let mut st = runmat_value::UnboundedStructValueBuilder::new();
    st.fields
        .insert("values".to_string(), Value::GpuTensor(handle));
    let selector = CellArray::new_with_shape(vec![Value::Int(IntValue::I32(1))], vec![1, 1])
        .expect("selector");

    let error = run_getfield(
        Value::Struct(st.finish()),
        vec![Value::from("values"), Value::Cell(selector)],
    )
    .expect_err("strict indexed resident access");
    assert_eq!(
        error.identifier(),
        GETFIELD_INDEXED_RESIDENT_EXTENSION.error_identifier
    );
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
#[test]
fn getfield_gpu_tensor_indexing() {
    futures::executor::block_on(getfield_gpu_tensor_indexing_case());
}

#[cfg(all(feature = "wgpu", target_arch = "wasm32"))]
#[wasm_bindgen_test::wasm_bindgen_test]
async fn getfield_gpu_tensor_indexing() {
    getfield_gpu_tensor_indexing_case().await;
}

#[cfg(feature = "wgpu")]
async fn getfield_gpu_tensor_indexing_case() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    #[cfg(not(target_arch = "wasm32"))]
    wgpu_backend::register_wgpu_provider(wgpu_backend::WgpuProviderOptions::default())
        .unwrap_or_else(|error| panic!("native WGPU provider initialization failed: {error:#}"));
    #[cfg(target_arch = "wasm32")]
    wgpu_backend::register_wgpu_provider_async(wgpu_backend::WgpuProviderOptions::default())
        .await
        .unwrap_or_else(|error| panic!("browser WGPU provider initialization failed: {error:#}"));
    let provider = runmat_accelerate_api::provider().expect("wgpu provider");

    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let view = HostTensorView {
        data: &tensor.materialize_f64(),
        shape: tensor.shape(),
    };
    let handle = provider.upload(&view).expect("upload");

    let mut st = runmat_value::UnboundedStructValueBuilder::new();
    st.fields
        .insert("values".to_string(), Value::GpuTensor(handle.clone()));
    let st = st.finish();

    let direct = getfield_builtin(
        None,
        Value::Struct(st.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("values")]),
    )
    .await
    .expect("direct gpu field");
    match direct.value() {
        Value::GpuTensor(out) => assert!(out.shares_allocation_with(&handle)),
        other => panic!("expected gpu tensor, got {other:?}"),
    }

    let idx_cell = CellArray::new(vec![Value::CharArray(CharArray::new_row("end"))], 1, 1).unwrap();
    let indexed = getfield_builtin(
        None,
        Value::Struct(st),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("values"),
            Value::Cell(idx_cell),
        ]),
    )
    .await
    .expect("gpu indexed field");
    let Value::Tensor(indexed) = indexed.value() else {
        panic!("expected class-preserving scalar tensor");
    };
    assert_eq!(indexed.shape(), &[1, 1]);
    assert_eq!(indexed.numeric_dtype(), runmat_value::NumericDType::F32);
    assert_eq!(indexed.materialize_f64(), vec![3.0]);
}
