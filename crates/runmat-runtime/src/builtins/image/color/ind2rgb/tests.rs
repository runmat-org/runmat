use super::*;

// Terminal test projection: the color result is numerical or resident GPU,
// with no GC-managed child that could outlive the owning completion.
async fn ind2rgb_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    indexed: Value,
    map: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    super::ind2rgb_builtin(
        runtime,
        indexed,
        map,
        crate::builtins::common::test_support::builtin_arguments(rest),
    )
    .await
    .map(|owned| owned.with_owned(|value| value))
}
use crate::builtins::common::{gpu_helpers, test_support};
use futures::executor::block_on;
use runmat_value::{IntegerStorage, LogicalArray, Tensor};

fn call(indexed: Tensor, map: Tensor) -> BuiltinResult<Tensor> {
    let value = block_on(ind2rgb_builtin(
        None,
        Value::Tensor(indexed),
        Value::Tensor(map),
        Vec::new(),
    ))?;
    let Value::Tensor(out) = value else {
        panic!("expected tensor");
    };
    Ok(out)
}

fn values(tensor: &Tensor) -> Vec<f64> {
    tensor.materialize_f64()
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn indexed_image_requires_matrix_rank_even_for_singleton_payloads() {
    for shape in [Vec::new(), vec![1], vec![2]] {
        let len = shape.iter().product();
        let indexed = Tensor::new(vec![1.0; len], shape).unwrap();
        let map = Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 3]).unwrap();
        let error = call(indexed, map).expect_err("indexed tensor must be a matrix");
        assert_eq!(error.identifier(), IND2RGB_ERROR_INVALID_INPUT.identifier);
        assert_eq!(
            error.message(),
            "ind2rgb: indexed image must be an MxN matrix"
        );
    }
}

#[test]
fn converts_one_based_double_indices() {
    let indexed = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
    let map = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 3]).unwrap();
    let out = call(indexed, map).unwrap();
    assert_eq!(out.shape(), &[1, 2, 3]);
    assert_eq!(out.numeric_dtype(), NumericDType::F64);
    assert_eq!(values(&out), vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
}

#[test]
fn converts_zero_based_uint8_indices() {
    let indexed = Tensor::new_with_dtype(vec![0.0, 1.0], vec![1, 2], NumericDType::U8).unwrap();
    let map = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 3]).unwrap();
    let out = call(indexed, map).unwrap();
    assert_eq!(values(&out), vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
}

#[test]
fn empty_index_image_retains_empty_rgb_geometry() {
    let indexed = Tensor::new_integer(IntegerStorage::U8(vec![]), vec![0, 2]).unwrap();
    let map = Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 3]).unwrap();
    let out = call(indexed, map).unwrap();
    assert_eq!(out.shape(), &[0, 2, 3]);
    assert!(out.is_empty());
}

#[test]
fn accepts_single_indices_and_returns_double() {
    let indexed = Tensor::new_with_dtype(vec![1.0, 2.0], vec![2, 1], NumericDType::F32).unwrap();
    let map = Tensor::new(vec![1.0, 0.0, 0.0, 0.5, 0.0, 1.0], vec![2, 3]).unwrap();
    let out = call(indexed, map).unwrap();
    assert_eq!(out.shape(), &[2, 1, 3]);
    assert_eq!(out.numeric_dtype(), NumericDType::F64);
    assert_eq!(values(&out), vec![1.0, 0.0, 0.0, 0.5, 0.0, 1.0]);
}

#[test]
fn ind2rgb_reads_typed_integer_indices_exactly() {
    let indexed = Tensor::new_integer(IntegerStorage::U16(vec![0, 1]), vec![2, 1]).unwrap();
    let map = Tensor::new(vec![1.0, 0.0, 0.0, 0.5, 0.0, 1.0], vec![2, 3]).unwrap();

    let out = call(indexed, map).unwrap();

    assert_eq!(out.shape(), &[2, 1, 3]);
    assert_eq!(out.numeric_dtype(), NumericDType::F64);
    assert_eq!(values(&out), vec![1.0, 0.0, 0.0, 0.5, 0.0, 1.0]);
}

#[test]
fn clips_indices_to_colormap_range() {
    let indexed = Tensor::new(vec![-10.0, 100.0], vec![1, 2]).unwrap();
    let map = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 3]).unwrap();
    let out = call(indexed, map).unwrap();
    assert_eq!(values(&out), vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
}

#[test]
fn rejects_nonintegral_floating_indices_instead_of_rounding_extension() {
    let indexed = Tensor::new(vec![1.5], vec![1, 1]).unwrap();
    let map = Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 3]).unwrap();
    let err = call(indexed, map).expect_err("fractional floating index");
    assert_eq!(err.identifier(), IND2RGB_ERROR_INVALID_INDEX.identifier);
}

#[test]
fn rejects_nonscalar_index_stack_shape() {
    let indexed = Tensor::new(vec![1.0, 1.0], vec![1, 1, 2]).unwrap();
    let map = Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 3]).unwrap();
    let err = call(indexed, map).expect_err("indexed image must be a matrix");
    assert!(err.message().contains("MxN matrix"));
}

#[test]
fn rejects_unsupported_index_classes() {
    for storage in [
        IntegerStorage::I8(vec![1]),
        IntegerStorage::I16(vec![1]),
        IntegerStorage::I32(vec![1]),
        IntegerStorage::I64(vec![1]),
        IntegerStorage::U32(vec![1]),
        IntegerStorage::U64(vec![1]),
    ] {
        let indexed = Tensor::new_integer(storage, vec![1, 1]).unwrap();
        let map = Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 3]).unwrap();
        let err = call(indexed, map).unwrap_err();
        assert!(err.message().contains("indexed image is not supported"));
    }
}

#[test]
fn rejects_non_double_colormap() {
    let indexed = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
    let map = Tensor::new_with_dtype(vec![1.0, 0.0, 0.0], vec![1, 3], NumericDType::F32).unwrap();
    let err = call(indexed, map).unwrap_err();
    assert!(err.message().contains("colormap is not supported"));
}

#[test]
fn rejects_logical_indexed_image_and_colormap() {
    let logical_indexed = LogicalArray::new(vec![1], vec![1, 1]).unwrap();
    let map = Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 3]).unwrap();
    let err = block_on(ind2rgb_builtin(
        None,
        Value::LogicalArray(logical_indexed),
        Value::Tensor(map),
        Vec::new(),
    ))
    .unwrap_err();
    assert!(err.message().contains("logical indexed image"));

    let indexed = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
    let logical_map = LogicalArray::new(vec![1, 0, 0], vec![1, 3]).unwrap();
    let err = block_on(ind2rgb_builtin(
        None,
        Value::Tensor(indexed),
        Value::LogicalArray(logical_map),
        Vec::new(),
    ))
    .unwrap_err();
    assert!(err.message().contains("logical colormap"));
}

#[test]
fn rejects_non_colormap_map_shape() {
    let indexed = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
    let map = Tensor::new(vec![1.0; 12], vec![2, 2, 3]).unwrap();
    let err = block_on(ind2rgb_builtin(
        None,
        Value::Tensor(indexed),
        Value::Tensor(map),
        Vec::new(),
    ))
    .unwrap_err();
    assert!(err.message().contains("map must be an Nx3 colormap"));
}

#[test]
fn ind2rgb_descriptor_signatures_cover_surface() {
    let labels: Vec<&str> = IND2RGB_DESCRIPTOR
        .signatures
        .iter()
        .map(|signature| signature.label)
        .collect();
    assert_eq!(labels, vec!["RGB = ind2rgb(X, map)"]);
}

#[test]
fn ind2rgb_integer_capabilities_distinguish_supported_indices_and_rejected_maps() {
    assert_eq!(IND2RGB_INTEGER_CAPABILITIES.len(), 3);
    assert_eq!(
        IND2RGB_INTEGER_CAPABILITIES[0].inputs[0].availability,
        BuiltinIntegerInputAvailability::Documented
    );
    assert_eq!(
        IND2RGB_INTEGER_CAPABILITIES[2].inputs[0].availability,
        BuiltinIntegerInputAvailability::Rejected
    );
}

#[test]
fn ind2rgb_gpu_fallback_restores_to_owner_and_preserves_source() {
    test_support::with_test_provider(|provider| {
        let indexed = Tensor::new_integer(IntegerStorage::U8(vec![0, 1]), vec![1, 2]).unwrap();
        let source = gpu_helpers::upload_tensor(provider, &indexed).expect("upload indexed");
        let source = source
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("publish provenance view");
        let map = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 3]).unwrap();

        let result = block_on(ind2rgb_builtin(
            None,
            Value::GpuTensor(source.clone()),
            Value::Tensor(map),
            Vec::new(),
        ))
        .expect("ind2rgb");
        let Value::GpuTensor(output) = result else {
            panic!("expected restored gpu output");
        };
        assert_eq!(output.shape(), &[1, 2, 3]);
        assert!(
            runmat_accelerate_api::provider_for_handle(&output).is_some_and(|owner| {
                runmat_accelerate_api::same_provider_instance(owner, provider)
            })
        );
        assert_eq!(
            test_support::gather(Value::GpuTensor(output))
                .unwrap()
                .materialize_f64(),
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        );
        assert_eq!(
            test_support::gather(Value::GpuTensor(source))
                .unwrap()
                .integer_storage(),
            Some(&IntegerStorage::U8(vec![0, 1]))
        );
    });
}

#[test]
fn ind2rgb_rejects_unsupported_resident_integer_before_download() {
    let handle = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::I16,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("admit owned fixture")
        .handle;
    let map = Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 3]).unwrap();
    let err = block_on(ind2rgb_builtin(
        None,
        Value::GpuTensor(handle.clone()),
        Value::Tensor(map),
        Vec::new(),
    ))
    .expect_err("unsupported resident integer");
    assert_eq!(err.identifier(), IND2RGB_ERROR_INVALID_INPUT.identifier);
}

#[test]
fn ind2rgb_descriptor_errors_have_stable_codes() {
    let codes: Vec<&str> = IND2RGB_DESCRIPTOR
        .errors
        .iter()
        .map(|error| error.code)
        .collect();
    assert!(codes.contains(&"RM.IND2RGB.TOO_MANY_INPUTS"));
    assert!(codes.contains(&"RM.IND2RGB.TOO_MANY_OUTPUTS"));
    assert!(codes.contains(&"RM.IND2RGB.INVALID_INPUT"));
    assert!(codes.contains(&"RM.IND2RGB.INVALID_COLORMAP"));
    assert!(codes.contains(&"RM.IND2RGB.INVALID_INDEX"));
    assert!(codes.contains(&"RM.IND2RGB.INTERNAL"));
}

#[test]
fn ind2rgb_too_many_args_uses_stable_identifier() {
    let err = block_on(ind2rgb_builtin(
        None,
        Value::Num(1.0),
        Value::Num(2.0),
        vec![Value::Num(3.0)],
    ))
    .expect_err("expected argument error");
    assert_eq!(err.identifier(), IND2RGB_ERROR_TOO_MANY_INPUTS.identifier);
}
