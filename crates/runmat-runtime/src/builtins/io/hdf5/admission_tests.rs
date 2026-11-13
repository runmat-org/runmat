use super::test_arguments::{h5read_builtin, h5write_builtin, hdf5read_builtin, hdf5write_builtin};
use futures::executor::block_on;

use super::*;

fn unregistered_resident() -> runmat_accelerate_api::test_support::TestAllocation {
    runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("test allocation")
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn invalid_text_and_selection_precede_resident_download() {
    let resident = unregistered_resident();
    let original = resident.handle.wire_descriptor();
    let invalid_filename = block_on(h5read_builtin(
        None,
        Value::Bool(true),
        Value::String("/data".into()),
        vec![Value::GpuTensor(resident.handle.clone()), Value::Num(1.0)],
    ))
    .expect_err("invalid filename before any provider access");
    assert_eq!(invalid_filename.identifier(), ERR_ARGUMENT.identifier);

    let invalid_selection = block_on(h5read_builtin(
        None,
        Value::String("never-opened.h5".into()),
        Value::String("/data".into()),
        vec![Value::Bool(true), Value::GpuTensor(resident.handle.clone())],
    ))
    .expect_err("invalid host selection before resident download or file open");
    assert_eq!(invalid_selection.identifier(), ERR_ARGUMENT.identifier);
    assert_eq!(resident.handle.wire_descriptor(), original);

    let paged_filename = Value::CharArray(
        runmat_value::CharArray::from_unicode_scalars_with_shape(vec!['a', 'b'], vec![1, 1, 2])
            .expect("paged character array"),
    );
    let paged = block_on(h5read_builtin(
        None,
        paged_filename,
        Value::String("/data".into()),
        vec![Value::GpuTensor(resident.handle.clone()), Value::Num(1.0)],
    ))
    .expect_err("paged text is rejected before resident selection download");
    assert_eq!(paged.identifier(), ERR_ARGUMENT.identifier);
    assert_eq!(resident.handle.wire_descriptor(), original);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn singleton_trailing_character_pages_remain_scalar_text() {
    let row = Value::CharArray(
        runmat_value::CharArray::from_unicode_scalars_with_shape(vec!['a', 'b'], vec![1, 2, 1])
            .expect("singleton trailing page"),
    );
    assert_eq!(string_scalar(&row, H5READ_NAME, "filename").unwrap(), "ab");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn invalid_write_payload_precedes_resident_selection_and_file_creation() {
    let resident = unregistered_resident();
    let error = block_on(h5write_builtin(
        None,
        Value::String("never-created.h5".into()),
        Value::String("/data".into()),
        Value::Complex(1.0, 2.0),
        vec![Value::GpuTensor(resident.handle.clone()), Value::Num(1.0)],
    ))
    .expect_err("unsupported payload before resident selection download");
    assert_eq!(error.identifier(), ERR_UNSUPPORTED.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn metadata_location_controls_reject_before_provider_download() {
    let resident = unregistered_resident();
    let error = block_on(h5info_builtin(
        Value::String("never-opened.h5".into()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Bool(false),
            Value::GpuTensor(resident.handle.clone()),
        ]),
    ))
    .expect_err("too many location controls before provider access");
    assert_eq!(error.identifier(), ERR_ARGUMENT.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fixed_output_arity_rejects_all_eight_before_file_access() {
    let file = || Value::String("never-created-output-count.h5".into());
    let object = || Value::String("/data".into());
    {
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        for result in [
            block_on(h5read_builtin(None, file(), object(), Vec::new())),
            block_on(hdf5read_builtin(None, file(), object(), Vec::new())),
            block_on(h5info_builtin(
                file(),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            )),
            block_on(hdf5info_builtin(
                file(),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            )),
            block_on(h5disp_builtin(
                file(),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            )),
        ] {
            assert_eq!(
                result.expect_err("too many outputs").identifier(),
                ERR_ARGUMENT.identifier
            );
        }
    }
    {
        let _outputs = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
        for result in [
            block_on(h5write_builtin(
                None,
                file(),
                object(),
                Value::Num(1.0),
                Vec::new(),
            )),
            block_on(hdf5write_builtin(
                None,
                file(),
                object(),
                Value::Num(1.0),
                Vec::new(),
            )),
            block_on(h5writeatt_builtin(
                None,
                file(),
                object(),
                Value::String("units".into()),
                Value::String("m/s".into()),
            )),
        ] {
            assert_eq!(
                result.expect_err("no outputs allowed").identifier(),
                ERR_ARGUMENT.identifier
            );
        }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn nested_implicit_output_does_not_inherit_enclosing_presentation_count() {
    let _presentation = crate::output_context::push_output_count(3).expect("output-count scope");
    let _invocation = crate::output_count::push_output_count(None).expect("output-count scope");
    let error = block_on(h5write_builtin(
        None,
        Value::Bool(true),
        Value::String("/A".into()),
        Value::Num(1.0),
        Vec::new(),
    ))
    .expect_err("invalid filename, not an inherited output-count error");
    assert_eq!(error.identifier(), ERR_ARGUMENT.identifier);
    assert!(error.message().contains("filename"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn captured_h5disp_text_is_gated_before_native_file_access() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let _captured = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
    let denied = block_on(h5disp_builtin(
        Value::String("never-opened.h5".into()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("captured text is a RunMat-only form");
    assert_eq!(
        denied.identifier(),
        H5DISP_CAPTURE_EXTENSION.error_identifier
    );
    drop(_captured);

    let _statement = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    let invalid = block_on(h5disp_builtin(
        Value::Bool(true),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("statement form must continue into ordinary argument validation");
    assert_eq!(invalid.identifier(), ERR_ARGUMENT.identifier);
    drop(_statement);

    let _enabled = crate::compatibility::push_runmat_extensions_enabled(true);
    let _captured = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
    let invalid = block_on(h5disp_builtin(
        Value::Bool(true),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("enabled captured form must continue into argument validation");
    assert_eq!(invalid.identifier(), ERR_ARGUMENT.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn h5disp_invalid_mode_and_encoding_precede_native_path_capture() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let _captured = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
    for rest in [
        vec![Value::String("/data".into()), Value::String("full".into())],
        vec![
            Value::String("/data".into()),
            Value::String("TextEncoding".into()),
            Value::String("Latin-1".into()),
        ],
        vec![
            Value::String("/data".into()),
            Value::String("Other".into()),
            Value::String("UTF-8".into()),
        ],
    ] {
        let error = block_on(h5disp_builtin(
            Value::String("never-opened-display.h5".into()),
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
        .expect_err("invalid controls before native path capture");
        assert_eq!(error.identifier(), ERR_ARGUMENT.identifier);
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test]
fn supported_hdf5_forms_report_native_target_unavailability_in_browser() {
    let read = block_on(h5read_builtin(
        Value::String("fixture.h5".into()),
        Value::String("/A".into()),
        Vec::new(),
    ))
    .expect_err("native HDF5 backend absent on browser");
    assert_eq!(read.identifier(), ERR_WASM.identifier);

    let write = block_on(h5write_builtin(
        Value::String("fixture.h5".into()),
        Value::String("/A".into()),
        Value::Num(1.0),
        Vec::new(),
    ))
    .expect_err("native HDF5 backend absent on browser");
    assert_eq!(write.identifier(), ERR_WASM.identifier);
}
