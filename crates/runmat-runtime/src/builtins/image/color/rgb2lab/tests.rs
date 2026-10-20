use super::*;

// Terminal test projection: the color result is numerical or resident GPU,
// with no GC-managed child that could outlive the owning completion.
async fn rgb2lab_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    rgb: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    super::rgb2lab_builtin(
        runtime,
        rgb,
        crate::builtins::common::test_support::builtin_arguments(rest),
    )
    .await
    .map(|owned| owned.with_owned(|value| value))
}
use crate::builtins::common::{gpu_helpers, test_support};
use futures::executor::block_on;
use runmat_builtins::BuiltinIntegerOutputClassRule;
use runmat_value::{IntegerStorage, Tensor};

fn call(tensor: Tensor) -> BuiltinResult<Tensor> {
    let Value::Tensor(out) =
        block_on(rgb2lab_builtin(None, Value::Tensor(tensor), Vec::new())).expect("rgb2lab")
    else {
        panic!("expected tensor");
    };
    Ok(out)
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected}, got {actual}"
    );
}

fn values(tensor: &Tensor) -> Vec<f64> {
    tensor.materialize_f64()
}

#[test]
fn converts_white_to_lab_reference() {
    let rgb = Tensor::new(vec![1.0, 1.0, 1.0], vec![1, 1, 3]).unwrap();
    let out = call(rgb).unwrap();
    let values = values(&out);
    assert_close(values[0], 100.0, 1e-4);
    assert_close(values[1], 0.0, 1e-3);
    assert_close(values[2], 0.0, 1e-3);
}

#[test]
fn rgb2lab_preserves_single_output_class() {
    let rgb = Tensor::from_f32(vec![1.0, 1.0, 1.0], vec![1, 1, 3]).unwrap();
    let out = call(rgb).expect("single RGB");
    assert_eq!(out.numeric_dtype(), NumericDType::F32);
    let values = values(&out);
    assert_close(values[0], 100.0, 1e-4);
    assert_close(values[1], 0.0, 1e-3);
    assert_close(values[2], 0.0, 1e-3);
}

#[test]
fn converts_rgb_colormap_to_lab_references() {
    let rgb = Tensor::new(
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        vec![4, 3],
    )
    .unwrap();
    let out = call(rgb).unwrap();
    assert_eq!(out.shape(), &[4, 3]);
    let expected = vec![
        0.0, 53.2408, 87.7347, 32.2970, 0.0, 80.0925, -86.1827, 79.1875, 0.0, 67.2032, 83.1793,
        -107.8602,
    ];
    for (actual, expected) in values(&out).iter().zip(expected) {
        assert_close(*actual, expected, 1e-3);
    }
}

#[test]
fn accepts_documented_single_uint8_and_uint16_colormaps() {
    let single = Tensor::from_f32(vec![1.0, 1.0, 1.0], vec![1, 3]).unwrap();
    assert_eq!(call(single).unwrap().numeric_dtype(), NumericDType::F32);
    for storage in [
        IntegerStorage::U8(vec![255, 255, 255]),
        IntegerStorage::U16(vec![65535, 65535, 65535]),
    ] {
        let out = call(Tensor::new_integer(storage, vec![1, 3]).unwrap()).unwrap();
        assert_eq!(out.numeric_dtype(), NumericDType::F64);
        assert_close(values(&out)[0], 100.0, 1e-4);
    }
}

#[test]
fn supports_documented_color_space_and_white_point_options() {
    let rgb = Tensor::new(vec![0.2, 0.3, 0.4], vec![1, 3]).unwrap();
    let Value::Tensor(adobe) = block_on(rgb2lab_builtin(
        None,
        Value::Tensor(rgb.clone()),
        vec![
            Value::String("ColorSpace".into()),
            Value::String("adobe-rgb-1998".into()),
        ],
    ))
    .expect("Adobe RGB option") else {
        panic!("expected tensor")
    };
    for (actual, expected) in values(&adobe).iter().zip([30.1783, -5.6902, -20.8223]) {
        assert_close(*actual, expected, 2e-2);
    }

    let Value::Tensor(d50) = block_on(rgb2lab_builtin(
        None,
        Value::Tensor(rgb),
        vec![
            Value::String("WhitePoint".into()),
            Value::String("d50".into()),
        ],
    ))
    .expect("D50 option") else {
        panic!("expected tensor")
    };
    for (actual, expected) in values(&d50).iter().zip([31.3294, -4.0732, -18.1750]) {
        assert_close(*actual, expected, 2e-2);
    }
}

#[test]
fn custom_white_point_and_repeated_option_use_final_value() {
    let rgb = Tensor::new(vec![0.2, 0.3, 0.4], vec![1, 3]).unwrap();
    let custom = Tensor::new(vec![0.9642, 1.0, 0.8251], vec![1, 3]).unwrap();
    let Value::Tensor(out) = block_on(rgb2lab_builtin(
        None,
        Value::Tensor(rgb),
        vec![
            Value::from("WhitePoint"),
            Value::from("D65"),
            Value::from("WhitePoint"),
            Value::Tensor(custom),
        ],
    ))
    .expect("last custom white point") else {
        panic!("Lab colormap");
    };
    for (actual, expected) in values(&out).iter().zip([31.3294, -4.0732, -18.1750]) {
        assert_close(*actual, expected, 2e-2);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn white_point_requires_exact_row_geometry_before_numeric_conversion() {
    let rgb = || Tensor::new(vec![1.0, 1.0, 1.0], vec![1, 3]).unwrap();
    let coordinates = [0.9642, 1.0, 0.8251];
    let row = Tensor::new(coordinates.to_vec(), vec![1, 3]).unwrap();
    assert!(block_on(rgb2lab_builtin(
        None,
        Value::Tensor(rgb()),
        vec![Value::from("WhitePoint"), Value::Tensor(row)],
    ))
    .is_ok());
    for shape in [vec![3], vec![3, 1]] {
        let wrong = Tensor::new(coordinates.to_vec(), shape).unwrap();
        let error = block_on(rgb2lab_builtin(
            None,
            Value::Tensor(rgb()),
            vec![Value::from("WhitePoint"), Value::Tensor(wrong)],
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), RGB2LAB_ERROR_INVALID_OPTION.identifier);
    }
}

#[test]
fn converts_m_by_n_by_three_by_p_image_stacks_without_crossing_planes() {
    let rgb = Tensor::new(vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0], vec![1, 1, 3, 2]).unwrap();
    let out = call(rgb).expect("stacked RGB");
    assert_eq!(out.shape(), &[1, 1, 3, 2]);
    let values = values(&out);
    assert_close(values[0], 100.0, 1e-4);
    assert_close(values[1], 0.0, 1e-3);
    assert_close(values[2], 0.0, 1e-3);
    assert_close(values[3], 53.2408, 1e-3);
    assert_close(values[4], 80.0925, 1e-3);
    assert_close(values[5], 67.2032, 1e-3);
}

#[test]
fn scales_uint8_rgb_before_lab_conversion() {
    let rgb =
        Tensor::new_with_dtype(vec![255.0, 0.0, 0.0], vec![1, 1, 3], NumericDType::U8).unwrap();
    let out = call(rgb).unwrap();
    assert_eq!(out.numeric_dtype(), NumericDType::F64);
    let values = values(&out);
    assert_close(values[0], 53.2408, 1e-3);
    assert_close(values[1], 80.0925, 1e-3);
    assert_close(values[2], 67.2032, 1e-3);
}

#[test]
fn rgb2lab_reads_typed_integer_rgb_storage_exactly() {
    let rgb = Tensor::new_integer(IntegerStorage::U8(vec![255, 0, 0]), vec![1, 1, 3]).unwrap();

    let out = call(rgb).unwrap();

    assert_eq!(out.numeric_dtype(), NumericDType::F64);
    let values = values(&out);
    assert_close(values[0], 53.2408, 1e-3);
    assert_close(values[1], 80.0925, 1e-3);
    assert_close(values[2], 67.2032, 1e-3);
}

#[test]
fn rgb2lab_accepts_uint16_and_rejects_other_integer_classes() {
    let white = Tensor::new_integer(IntegerStorage::U16(vec![65535; 3]), vec![1, 1, 3]).unwrap();
    let out = call(white).expect("uint16 RGB");
    assert_eq!(out.numeric_dtype(), NumericDType::F64);
    let values = values(&out);
    assert_close(values[0], 100.0, 1e-4);
    assert_close(values[1], 0.0, 1e-3);
    assert_close(values[2], 0.0, 1e-3);

    for storage in [
        IntegerStorage::I8(vec![0; 3]),
        IntegerStorage::I16(vec![0; 3]),
        IntegerStorage::I32(vec![0; 3]),
        IntegerStorage::I64(vec![0; 3]),
        IntegerStorage::U32(vec![0; 3]),
        IntegerStorage::U64(vec![0; 3]),
    ] {
        let input = Tensor::new_integer(storage, vec![1, 1, 3]).unwrap();
        let err = block_on(rgb2lab_builtin(None, Value::Tensor(input), Vec::new()))
            .expect_err("unsupported integer class");
        assert_eq!(err.identifier(), RGB2LAB_ERROR_INVALID_INPUT.identifier);
    }
}

#[test]
fn rgb2lab_descriptor_signatures_cover_surface() {
    let labels: Vec<&str> = RGB2LAB_DESCRIPTOR
        .signatures
        .iter()
        .map(|signature| signature.label)
        .collect();
    assert_eq!(labels, vec!["LAB = rgb2lab(RGB, Name=Value)"]);
}

#[test]
fn rgb2lab_integer_capabilities_record_supported_and_rejected_classes() {
    assert_eq!(RGB2LAB_INTEGER_CAPABILITIES.len(), 2);
    assert_eq!(
        RGB2LAB_INTEGER_CAPABILITIES[0].inputs[0].availability,
        BuiltinIntegerInputAvailability::Documented
    );
    assert_eq!(
        RGB2LAB_INTEGER_CAPABILITIES[1].inputs[0].availability,
        BuiltinIntegerInputAvailability::Rejected
    );
    assert_eq!(
        RGB2LAB_INTEGER_CAPABILITIES[0].output_class,
        BuiltinIntegerOutputClassRule::Double
    );
}

#[test]
fn rgb2lab_explicit_gpu_form_is_extension_gated_before_download() {
    let handle = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1, 3],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("admit owned fixture")
        .handle;
    let handle = handle
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .expect("publish provenance view");
    let _guard = crate::compatibility::push_runmat_extensions_enabled(false);
    let err = block_on(rgb2lab_builtin(
        None,
        Value::GpuTensor(handle.clone()),
        Vec::new(),
    ))
    .expect_err("explicit GPU form must be gated");
    assert_eq!(err.identifier(), RGB2LAB_EXTENSIONS[0].error_identifier);
}

#[test]
fn rgb2lab_enabled_explicit_gpu_form_restores_output_to_exact_owner() {
    test_support::with_test_provider(|provider| {
        let rgb =
            Tensor::new_integer(IntegerStorage::U8(vec![255, 0, 0]), vec![1, 1, 3]).expect("RGB");
        let source = gpu_helpers::upload_tensor(provider, &rgb).expect("upload RGB");
        let source = source
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("publish provenance view");
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let value = block_on(rgb2lab_builtin(
            None,
            Value::GpuTensor(source.clone()),
            Vec::new(),
        ))
        .expect("explicit resident rgb2lab");
        let Value::GpuTensor(output) = value else {
            panic!("explicit gpuArray output must remain resident");
        };
        assert_eq!(output.device_id(), source.device_id());
        assert_ne!(output.buffer_id(), source.buffer_id());
        assert_eq!(
            runmat_accelerate_api::handle_precision(&output),
            Some(runmat_accelerate_api::ProviderPrecision::F64)
        );
        assert!(runmat_accelerate_api::handle_is_explicit(&output));
        output.release().expect("free output");
        source.release().expect("free source");
    });
}

#[test]
fn rgb2lab_descriptor_errors_have_stable_codes() {
    let codes: Vec<&str> = RGB2LAB_DESCRIPTOR
        .errors
        .iter()
        .map(|error| error.code)
        .collect();
    assert!(codes.contains(&"RM.RGB2LAB.TOO_MANY_INPUTS"));
    assert!(codes.contains(&"RM.RGB2LAB.TOO_MANY_OUTPUTS"));
    assert!(codes.contains(&"RM.RGB2LAB.INVALID_INPUT"));
    assert!(codes.contains(&"RM.RGB2LAB.INTERNAL"));
    assert!(codes.contains(&"RM.RGB2LAB.INVALID_OPTION"));
}

#[test]
fn rgb2lab_malformed_name_value_uses_stable_identifier() {
    let err = block_on(rgb2lab_builtin(
        None,
        Value::Num(1.0),
        vec![Value::Num(2.0)],
    ))
    .expect_err("expected argument error");
    assert_eq!(err.identifier(), RGB2LAB_ERROR_INVALID_OPTION.identifier);
}
