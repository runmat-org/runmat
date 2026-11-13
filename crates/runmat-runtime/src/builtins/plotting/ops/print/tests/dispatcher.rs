use super::*;
#[path = "dispatcher/fixtures.rs"]
mod fixtures;

use fixtures::*;
#[cfg(not(feature = "plot-core"))]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[cfg(not(feature = "plot-core"))]
use runmat_accelerate_api::ThreadProviderGuard;

#[cfg(feature = "plot-core")]
#[test]
fn dispatcher_uses_selected_compatibility_and_requested_output_count() {
    let _plot = setup();
    let _filesystem_lock = runmat_filesystem::provider_override_lock();
    block_on(plot_builtin(vec![
        Value::Tensor(tensor(&[0.0, 1.0, 2.0])),
        Value::Tensor(tensor(&[2.0, 1.0, 3.0])),
    ]))
    .expect("plot fixture");
    let filesystem = Arc::new(CountingFilesystem::new());
    let _filesystem = runmat_filesystem::replace_provider(filesystem.clone());
    let matlab = runtime(crate::context::RuntimeLanguageMode::Matlab);
    let runmat = runtime(crate::context::RuntimeLanguageMode::RunMat);

    let ambient_runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let ambient_output =
        crate::output_count::push_output_count(Some(1)).expect("output-count scope");
    assert_no_outputs(block_on(dispatch(&matlab, "/print/matlab-statement.png", 0)).unwrap());
    drop(ambient_output);
    let ambient_output =
        crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    let error = block_on(dispatch(&matlab, "/print/matlab-status.png", 1)).unwrap_err();
    assert_eq!(
        error.identifier(),
        PRINT_STATUS_OUTPUT_EXTENSION.error_identifier
    );
    drop(ambient_output);
    let ambient_output =
        crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    let error = block_on(dispatch(&matlab, "/print/matlab-two.png", 2)).unwrap_err();
    assert_eq!(error.identifier(), PRINT_ERROR_TOO_MANY_OUTPUTS.identifier);
    drop(ambient_output);
    drop(ambient_runmat);

    let ambient_matlab = crate::compatibility::push_runmat_extensions_enabled(false);
    let ambient_output =
        crate::output_count::push_output_count(Some(1)).expect("output-count scope");
    assert_no_outputs(block_on(dispatch(&runmat, "/print/runmat-statement.png", 0)).unwrap());
    drop(ambient_output);
    let ambient_output =
        crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    assert_eq!(
        block_on(dispatch(&runmat, "/print/runmat-status.png", 1)).unwrap(),
        Value::Bool(true)
    );
    drop(ambient_output);
    let ambient_output =
        crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    let error = block_on(dispatch(&runmat, "/print/runmat-two.png", 2)).unwrap_err();
    assert_eq!(error.identifier(), PRINT_ERROR_TOO_MANY_OUTPUTS.identifier);
    drop(ambient_output);
    drop(ambient_matlab);

    assert!(filesystem
        .inner
        .read_project_path("/print/matlab-statement.png")
        .is_ok());
    assert!(filesystem
        .inner
        .read_project_path("/print/runmat-statement.png")
        .is_ok());
    assert!(filesystem
        .inner
        .read_project_path("/print/runmat-status.png")
        .is_ok());
    assert!(filesystem
        .inner
        .read_project_path("/print/matlab-status.png")
        .is_err());
    assert!(filesystem.capture_count() > 0);
}

#[cfg(not(feature = "plot-core"))]
fn assert_renderer_unavailable(error: &RuntimeError) {
    assert_eq!(error.identifier(), PRINT_ERROR_RENDER.identifier);
    let mut source = std::error::Error::source(error);
    let mut found = false;
    while let Some(error) = source {
        found |= error
            .downcast_ref::<crate::builtins::plotting::engine::PngExportUnavailable>()
            .is_some();
        source = error.source();
    }
    assert!(
        found,
        "typed PNG renderer capability cause must be retained"
    );
}

#[cfg(not(feature = "plot-core"))]
#[test]
fn feature_disabled_dispatch_fails_before_filesystem_capture() {
    let _plot = setup();
    let _filesystem_lock = runmat_filesystem::provider_override_lock();
    let filesystem = Arc::new(CountingFilesystem::new());
    let _filesystem = runmat_filesystem::replace_provider(filesystem.clone());
    let matlab = runtime(crate::context::RuntimeLanguageMode::Matlab);
    let runmat = runtime(crate::context::RuntimeLanguageMode::RunMat);

    let ambient_runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let ambient_output =
        crate::output_count::push_output_count(Some(1)).expect("output-count scope");
    assert_renderer_unavailable(
        &block_on(dispatch(&matlab, "/print/matlab-statement.png", 0)).unwrap_err(),
    );
    drop(ambient_output);
    let ambient_output =
        crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    let error = block_on(dispatch(&matlab, "/print/matlab-status.png", 1)).unwrap_err();
    assert_eq!(
        error.identifier(),
        PRINT_STATUS_OUTPUT_EXTENSION.error_identifier
    );
    drop(ambient_output);
    let ambient_output =
        crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    let error = block_on(dispatch(&matlab, "/print/matlab-two.png", 2)).unwrap_err();
    assert_eq!(error.identifier(), PRINT_ERROR_TOO_MANY_OUTPUTS.identifier);
    drop(ambient_output);
    drop(ambient_runmat);

    let ambient_matlab = crate::compatibility::push_runmat_extensions_enabled(false);
    let ambient_output =
        crate::output_count::push_output_count(Some(1)).expect("output-count scope");
    assert_renderer_unavailable(
        &block_on(dispatch(&runmat, "/print/runmat-statement.png", 0)).unwrap_err(),
    );
    drop(ambient_output);
    let ambient_output =
        crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    assert_renderer_unavailable(
        &block_on(dispatch(&runmat, "/print/runmat-status.png", 1)).unwrap_err(),
    );
    drop(ambient_output);
    let ambient_output =
        crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    let error = block_on(dispatch(&runmat, "/print/runmat-two.png", 2)).unwrap_err();
    assert_eq!(error.identifier(), PRINT_ERROR_TOO_MANY_OUTPUTS.identifier);
    drop(ambient_output);
    drop(ambient_matlab);

    assert_eq!(filesystem.capture_count(), 0);
}

#[cfg(not(feature = "plot-core"))]
#[test]
fn known_invalid_host_argument_precedes_renderer_capability() {
    let _plot = setup();
    let _filesystem_lock = runmat_filesystem::provider_override_lock();
    let filesystem = Arc::new(CountingFilesystem::new());
    let _filesystem = runmat_filesystem::replace_provider(filesystem.clone());
    let runtime = runtime(crate::context::RuntimeLanguageMode::RunMat);
    let nested = runmat_value::CellArray::new(vec![Value::from("plot.png")], 1, 1)
        .expect("one-cell fixture");
    let error = block_on(crate::call_builtin_async_with_outputs_in_context(
        &runtime,
        "print",
        &[Value::Cell(nested)],
        0,
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), PRINT_ERROR_INVALID_INPUT.identifier);
    assert_eq!(filesystem.capture_count(), 0);
}

#[cfg(not(feature = "plot-core"))]
#[test]
fn feature_disabled_dispatch_does_not_download_resident_handle() {
    let _plot = setup();
    let _acceleration = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(CountingDownloadProvider {
        inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
        downloads: AtomicUsize::new(0),
    }));
    let _provider = ThreadProviderGuard::set(Some(provider));
    let handle = current_figure_handle();
    let host = Tensor::new(vec![f64::from(handle.as_u32())], vec![1, 1]).unwrap();
    let resident = crate::builtins::common::gpu_helpers::upload_tensor(provider, &host).unwrap();
    let runtime = runtime(crate::context::RuntimeLanguageMode::RunMat);
    let error = block_on(crate::call_builtin_async_with_outputs_in_context(
        &runtime,
        "print",
        &[
            Value::GpuTensor(resident),
            Value::from("/never-captured.png"),
        ],
        0,
    ))
    .unwrap_err();
    assert_renderer_unavailable(&error);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
}
