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
        Value::Bool(true),
        Value::String("/data".into()),
        vec![Value::GpuTensor(resident.handle.clone()), Value::Num(1.0)],
    ))
    .expect_err("invalid filename before any provider access");
    assert_eq!(invalid_filename.identifier(), ERR_ARGUMENT.identifier);

    let invalid_selection = block_on(h5read_builtin(
        Value::String("never-opened.h5".into()),
        Value::String("/data".into()),
        vec![Value::Bool(true), Value::GpuTensor(resident.handle.clone())],
    ))
    .expect_err("invalid host selection before resident download or file open");
    assert_eq!(invalid_selection.identifier(), ERR_ARGUMENT.identifier);
    assert_eq!(resident.handle.wire_descriptor(), original);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn invalid_write_payload_precedes_resident_selection_and_file_creation() {
    let resident = unregistered_resident();
    let error = block_on(h5write_builtin(
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
        vec![
            Value::Bool(false),
            Value::GpuTensor(resident.handle.clone()),
        ],
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
        let _outputs = crate::output_count::push_output_count(Some(2));
        for result in [
            block_on(h5read_builtin(file(), object(), Vec::new())),
            block_on(hdf5read_builtin(file(), object(), Vec::new())),
            block_on(h5info_builtin(file(), Vec::new())),
            block_on(hdf5info_builtin(file(), Vec::new())),
            block_on(h5disp_builtin(file(), Vec::new())),
        ] {
            assert_eq!(
                result.expect_err("too many outputs").identifier(),
                ERR_ARGUMENT.identifier
            );
        }
    }
    {
        let _outputs = crate::output_count::push_output_count(Some(1));
        for result in [
            block_on(h5write_builtin(
                file(),
                object(),
                Value::Num(1.0),
                Vec::new(),
            )),
            block_on(hdf5write_builtin(
                file(),
                object(),
                Value::Num(1.0),
                Vec::new(),
            )),
            block_on(h5writeatt_builtin(
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
