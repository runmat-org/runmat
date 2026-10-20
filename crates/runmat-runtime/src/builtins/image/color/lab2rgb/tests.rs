use super::*;

// Terminal test projection: the color result is numerical or resident GPU,
// with no GC-managed child that could outlive the owning completion.
async fn lab2rgb_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    lab: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    super::lab2rgb_builtin(
        runtime,
        lab,
        crate::builtins::common::test_support::builtin_arguments(rest),
    )
    .await
    .map(|owned| owned.with_owned(|value| value))
}
use crate::builtins::common::{gpu_helpers, test_support};
use futures::executor::block_on;
use runmat_value::{IntegerStorage, LogicalArray, Tensor};

fn call(tensor: Tensor) -> BuiltinResult<Tensor> {
    let value = block_on(lab2rgb_builtin(None, Value::Tensor(tensor), Vec::new()))?;
    let Value::Tensor(out) = value else {
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
fn converts_white_lab_to_rgb() {
    let lab = Tensor::new(vec![100.0, 0.0, 0.0], vec![1, 1, 3]).unwrap();
    let out = call(lab).unwrap();
    let values = values(&out);
    assert_close(values[0], 1.0, 1e-4);
    assert_close(values[1], 1.0, 1e-4);
    assert_close(values[2], 1.0, 1e-4);
}

#[test]
fn converts_lab_colormap_references_to_rgb() {
    let lab = Tensor::new(
        vec![
            0.0, 53.2408, 87.7347, 32.2970, 0.0, 80.0925, -86.1827, 79.1875, 0.0, 67.2032, 83.1793,
            -107.8602,
        ],
        vec![4, 3],
    )
    .unwrap();
    let out = call(lab).unwrap();
    assert_eq!(out.shape(), &[4, 3]);
    let expected = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    for (actual, expected) in values(&out).iter().zip(expected) {
        assert_close(*actual, expected, 1e-3);
    }
}

#[test]
fn preserves_single_precision_metadata() {
    let lab =
        Tensor::new_with_dtype(vec![100.0, 0.0, 0.0], vec![1, 1, 3], NumericDType::F32).unwrap();
    let out = call(lab).unwrap();
    assert_eq!(out.numeric_dtype(), NumericDType::F32);
    let values = values(&out);
    assert_close(values[0], 1.0, 1e-4);
    assert_close(values[1], 1.0, 1e-4);
    assert_close(values[2], 1.0, 1e-4);
}

#[test]
fn output_type_constructs_authoritative_uint8_and_uint16_storage() {
    let lab = Tensor::new(vec![70.0, 5.0, 10.0], vec![1, 1, 3]).unwrap();
    for (name, dtype) in [("uint8", NumericDType::U8), ("uint16", NumericDType::U16)] {
        let value = block_on(lab2rgb_builtin(
            None,
            Value::Tensor(lab.clone()),
            vec![Value::from("OutputType"), Value::from(name)],
        ))
        .expect("integer OutputType");
        let Value::Tensor(output) = value else {
            panic!("expected tensor output");
        };
        assert_eq!(output.numeric_dtype(), dtype);
        assert!(output.integer_storage().is_some());
    }
}

#[test]
fn explicit_resident_input_restores_output_to_exact_owner() {
    test_support::with_test_provider(|provider| {
        let lab = Tensor::new(vec![70.0, 5.0, 10.0], vec![1, 1, 3]).unwrap();
        let source = gpu_helpers::upload_tensor(provider, &lab).expect("upload LAB");
        let source = source
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("publish provenance view");
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let value = block_on(lab2rgb_builtin(
            None,
            Value::GpuTensor(source.clone()),
            vec![Value::from("OutputType"), Value::from("uint8")],
        ))
        .expect("explicit resident lab2rgb");
        let Value::GpuTensor(output) = value else {
            panic!("explicit gpuArray output must remain resident");
        };
        assert_eq!(output.device_id(), source.device_id());
        assert_ne!(output.buffer_id(), source.buffer_id());
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&output),
            Some(runmat_accelerate_api::IntegerElementType::U8)
        );
        assert!(runmat_accelerate_api::handle_is_explicit(&output));
        output.release().expect("free output");
        source.release().expect("free source");
    });
}

#[test]
fn floating_output_preserves_out_of_gamut_components() {
    let lab = Tensor::new(vec![50.0, 200.0, 200.0], vec![1, 1, 3]).unwrap();
    let output = call(lab).unwrap();
    assert!(output
        .materialize_f64()
        .iter()
        .any(|value| *value < 0.0 || *value > 1.0));
}

#[test]
fn lab2rgb_rejects_every_integer_class_and_logical() {
    for storage in [
        IntegerStorage::I8(vec![0; 3]),
        IntegerStorage::I16(vec![0; 3]),
        IntegerStorage::I32(vec![0; 3]),
        IntegerStorage::I64(vec![0; 3]),
        IntegerStorage::U8(vec![0; 3]),
        IntegerStorage::U16(vec![0; 3]),
        IntegerStorage::U32(vec![0; 3]),
        IntegerStorage::U64(vec![0; 3]),
    ] {
        let lab = Tensor::new_integer(storage, vec![1, 1, 3]).unwrap();
        let err = call(lab).unwrap_err();
        assert_eq!(err.identifier(), LAB2RGB_ERROR_INVALID_INPUT.identifier);
    }

    let logical = LogicalArray::new(vec![1, 0, 0], vec![1, 1, 3]).unwrap();
    let err = block_on(lab2rgb_builtin(
        None,
        Value::LogicalArray(logical),
        Vec::new(),
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), LAB2RGB_ERROR_INVALID_INPUT.identifier);
}

#[test]
fn lab2rgb_descriptor_signatures_cover_surface() {
    let labels: Vec<&str> = LAB2RGB_DESCRIPTOR
        .signatures
        .iter()
        .map(|signature| signature.label)
        .collect();
    assert_eq!(labels, vec!["RGB = lab2rgb(LAB, OutputType=type)"]);
}

#[test]
fn lab2rgb_descriptor_errors_have_stable_codes() {
    let codes: Vec<&str> = LAB2RGB_DESCRIPTOR
        .errors
        .iter()
        .map(|error| error.code)
        .collect();
    assert!(codes.contains(&"RM.LAB2RGB.TOO_MANY_INPUTS"));
    assert!(codes.contains(&"RM.LAB2RGB.TOO_MANY_OUTPUTS"));
    assert!(codes.contains(&"RM.LAB2RGB.INVALID_INPUT"));
    assert!(codes.contains(&"RM.LAB2RGB.INTERNAL"));
    assert!(codes.contains(&"RM.LAB2RGB.INVALID_OPTION"));
}

#[test]
fn lab2rgb_too_many_args_uses_stable_identifier() {
    let err = block_on(lab2rgb_builtin(
        None,
        Value::Num(1.0),
        vec![Value::Num(2.0), Value::Num(3.0), Value::Num(4.0)],
    ))
    .expect_err("expected argument error");
    assert_eq!(err.identifier(), LAB2RGB_ERROR_TOO_MANY_INPUTS.identifier);
}

#[test]
fn resident_integer_input_rejects_before_provider_lookup() {
    let handle = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1, 3],
            runmat_accelerate_api::NumericElementType::U8,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("admit owned fixture")
        .handle;
    let error = block_on(lab2rgb_builtin(None, Value::GpuTensor(handle), Vec::new()))
        .expect_err("integer resident LAB must reject");
    assert_eq!(error.identifier(), LAB2RGB_ERROR_INVALID_INPUT.identifier);
    assert!(!error.message().to_ascii_lowercase().contains("provider"));
}

#[test]
fn resident_floating_descriptor_must_match_exact_owner_before_download() {
    test_support::with_test_provider(|provider| {
        let lab = Tensor::new_with_dtype(vec![70.0, 5.0, 10.0], vec![1, 1, 3], NumericDType::F32)
            .expect("LAB");
        let source = gpu_helpers::upload_tensor(provider, &lab).expect("upload LAB");
        let mut wire = source.wire_descriptor();
        wire.physical = runmat_accelerate_api::allocation::GpuPhysicalDescriptor::new(
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
            3,
            24,
            24,
        )
        .unwrap();
        assert!(
            provider.acquire_live(&wire).is_err(),
            "contradictory physical facts cannot mint a live input"
        );
        assert_eq!(
            source.physical().element_type(),
            runmat_accelerate_api::NumericElementType::F32
        );
        block_on(lab2rgb_builtin(None, Value::GpuTensor(source), Vec::new()))
            .expect("authentic input remains usable");
    });
}
