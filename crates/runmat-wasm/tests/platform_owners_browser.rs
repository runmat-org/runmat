#![cfg(target_arch = "wasm32")]

use runmat_runtime::builtins::plotting::{
    capture_figure_persistence_snapshot, close_figure, import_figure,
};
use runmat_runtime::execution::value_codec::{decode_inline_value, encode_inline_value};
use runmat_runtime::runtime_error::{error_from_exception, exception_from_error};
use runmat_value::Value;
use std::error::Error;
use wasm_bindgen_test::wasm_bindgen_test;

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn caught_error_codec_and_rethrow_preserve_runtime_metadata() {
    let error = runmat_runtime::build_runtime_error("browser failure")
        .with_identifier("RunMat:Test:BrowserOwner")
        .with_phase("execution")
        .with_gpu_gather_retry(runmat_runtime::GpuGatherRetry::Never)
        .with_call_frames(vec![runmat_runtime::CallFrame {
            function: "origin".into(),
            source_id: None,
            span: Some((2, 5)),
            source_owner: None,
            diagnostic_source: None,
        }])
        .build();
    let value = Value::MException(exception_from_error(error));
    let encoded = encode_inline_value(&value).expect("encode source-less caught error");
    let Value::MException(decoded) =
        decode_inline_value(&encoded).expect("decode browser-local caught error")
    else {
        panic!("expected decoded exception");
    };
    let retained = decoded.clone();
    let original = retained.original().expect("decoded runtime owner");
    let reraised = error_from_exception(decoded);
    let shared = reraised
        .source()
        .unwrap()
        .downcast_ref::<runmat_runtime::runtime_error::SharedRuntimeErrorSource>()
        .expect("rethrow retains the exact caught owner");
    assert!(std::ptr::eq(shared.original(), original.as_ref()));
    assert_eq!(reraised.identifier(), Some("RunMat:Test:BrowserOwner"));
    assert_eq!(reraised.context.phase.as_deref(), Some("execution"));
    assert_eq!(
        reraised.gpu_gather_retry,
        runmat_runtime::GpuGatherRetry::Never
    );
    assert_eq!(reraised.context.call_frames[0].function, "origin");
    assert_eq!(original.context.call_frames[0].span, Some((2, 5)));
    assert_eq!(
        retained
            .stack
            .iter()
            .map(|name| name.as_str())
            .collect::<Vec<_>>(),
        ["origin"]
    );
    let recaught = Value::MException(exception_from_error(reraised));
    let encoded = encode_inline_value(&recaught).expect("encode shared rethrow owner");
    let Value::MException(repeated) =
        decode_inline_value(&encoded).expect("decode shared rethrow owner")
    else {
        panic!("expected repeated exception");
    };
    assert_eq!(repeated.identifier, retained.identifier);
    assert_eq!(repeated.stack, retained.stack);
    assert_eq!(
        repeated.original().unwrap().gpu_gather_retry,
        runmat_runtime::GpuGatherRetry::Never
    );
}

#[wasm_bindgen_test]
async fn figure_snapshots_share_then_detach_on_mutation_and_outlive_registry_entry() {
    let mut figure = runmat_plot::plots::Figure::new();
    figure.set_name("before");
    let handle = import_figure(figure);
    let snapshot = capture_figure_persistence_snapshot(&[handle, handle], 2, usize::MAX)
        .expect("capture shared browser figure");
    assert_eq!(snapshot.entries.len(), 2);
    assert!(std::ptr::eq(
        snapshot.entries[0].figure.as_ref(),
        snapshot.entries[1].figure.as_ref(),
    ));
    assert!(snapshot.entries[0]
        .scene_lease
        .shares_lease(&snapshot.entries[1].scene_lease));
    runmat_runtime::call_builtin_async(
        "set",
        &[
            Value::Num(f64::from(handle.as_u32())),
            Value::String("Name".into()),
            Value::String("after".into()),
        ],
    )
    .await
    .expect("mutate registry figure through the public builtin");
    let successor =
        capture_figure_persistence_snapshot(&[handle], 1, usize::MAX).expect("capture successor");
    assert_eq!(snapshot.entries[0].figure.name.as_deref(), Some("before"));
    assert_eq!(successor.entries[0].figure.name.as_deref(), Some("after"));
    assert!(!std::ptr::eq(
        snapshot.entries[0].figure.as_ref(),
        successor.entries[0].figure.as_ref(),
    ));
    assert!(!snapshot.entries[0]
        .scene_lease
        .shares_lease(&successor.entries[0].scene_lease));
    close_figure(Some(handle)).expect("close registry figure");
    assert_eq!(snapshot.entries[0].figure.name.as_deref(), Some("before"));
    assert_eq!(successor.entries[0].figure.name.as_deref(), Some("after"));
}
