use super::*;
use crate::builtins::common::test_support;
use crate::builtins::io::filetext::{fopen, registry};
#[cfg(not(target_arch = "wasm32"))]
use runmat_time::system_time_now;
use runmat_value::{CellArray, IntValue, IntegerStorage, LogicalArray, StringArray, Tensor};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::time::UNIX_EPOCH;

mod support;
use support::FailingFlushProvider;

fn unwrap_error_message(err: crate::RuntimeError) -> String {
    err.message().to_string()
}

fn run_evaluate(args: &[Value]) -> BuiltinResult<FcloseEval> {
    futures::executor::block_on(evaluate(args))
}

fn run_fopen(args: &[Value]) -> BuiltinResult<fopen::FopenEval> {
    futures::executor::block_on(fopen::evaluate(args))
}

fn registry_guard() -> registry::RegistryTestGuard {
    registry::test_guard()
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fclose_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = FCLOSE_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"status = fclose()"));
    assert!(labels.contains(&"status = fclose(fid)"));
    assert!(labels.contains(&"status = fclose(\"all\")"));
    assert!(labels.contains(&"[status, msg] = fclose(...)"));
}

#[test]
fn typed_file_identifier_parser_rejects_unrepresentable_uint64() {
    assert_eq!(parse_scalar_fid(&Value::Int(IntValue::U16(7))).unwrap(), 7);
    assert!(parse_scalar_fid(&Value::Int(IntValue::U64(u64::MAX))).is_err());
}

#[test]
fn fclose_vector_parser_reads_typed_integer_storage_exactly() {
    let ids = Tensor::new_integer(IntegerStorage::U16(vec![7, 9]), vec![1, 2]).expect("typed fids");

    assert_eq!(collect_file_ids(&Value::Tensor(ids)).unwrap(), vec![7, 9]);

    let too_large =
        Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).expect("typed fid");
    assert!(collect_file_ids(&Value::Tensor(too_large)).is_err());
}

#[test]
fn fclose_typed_identifier_tensors_ignore_poisoned_f64_mirrors() {
    let classes = [
        IntegerStorage::I8(vec![7]),
        IntegerStorage::I16(vec![7]),
        IntegerStorage::I32(vec![7]),
        IntegerStorage::I64(vec![7]),
        IntegerStorage::U8(vec![7]),
        IntegerStorage::U16(vec![7]),
        IntegerStorage::U32(vec![7]),
        IntegerStorage::U64(vec![7]),
    ];

    for storage in classes {
        let tensor = Tensor::new_integer(storage, vec![1, 1]).expect("typed fid");
        assert_eq!(collect_file_ids(&Value::Tensor(tensor)).unwrap(), vec![7]);
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn unique_path(prefix: &str) -> PathBuf {
    let now = system_time_now()
        .duration_since(UNIX_EPOCH)
        .expect("time went backwards");
    let filename = format!("{}_{}_{}.tmp", prefix, now.as_secs(), now.subsec_nanos());
    std::env::temp_dir().join(filename)
}

#[cfg(target_arch = "wasm32")]
fn unique_path(prefix: &str) -> PathBuf {
    test_support::filesystem::unique_path(prefix)
}

fn open_temp_file(prefix: &str) -> (f64, PathBuf) {
    let path = unique_path(prefix);
    {
        let mut file = runmat_filesystem::File::create(&path).unwrap();
        writeln!(&mut file, "data").unwrap();
    }
    let eval = run_fopen(&[Value::from(path.to_string_lossy().to_string())]).expect("fopen");
    let fid = eval.as_open().unwrap().fid;
    assert!(fid >= 3.0, "expected valid file identifier");
    (fid, path)
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fclose_closes_single_file() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let (fid, path) = open_temp_file("fclose_single");
    let eval = run_evaluate(&[Value::Num(fid)]).expect("fclose");
    assert_eq!(eval.status(), 0.0);
    assert!(eval.message().is_empty());
    assert!(registry::info_for(fid as i32).is_none());
    test_support::fs::remove_file(path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fclose_invalid_identifier_returns_error() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let eval = run_evaluate(&[Value::Num(9999.0)]).expect("fclose");
    assert_eq!(eval.status(), -1.0);
    assert_eq!(eval.message(), FCLOSE_ERROR_INVALID_IDENTIFIER.message);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fclose_all_closes_everything() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = registry_guard();
    registry::reset_for_tests();
    let (fid, path) = open_temp_file("fclose_all");
    let eval = run_evaluate(&[Value::from("all")]).expect("fclose all");
    assert_eq!(eval.status(), 0.0);
    assert!(registry::info_for(fid as i32).is_none());
    let infos = registry::list_infos();
    assert!(infos.iter().all(|info| info.id < 3));
    test_support::fs::remove_file(path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fclose_no_args_closes_all() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = registry_guard();
    registry::reset_for_tests();
    let (fid, path) = open_temp_file("fclose_no_args");
    let eval = run_evaluate(&[]).expect("fclose");
    assert_eq!(eval.status(), 0.0);
    assert!(registry::info_for(fid as i32).is_none());
    test_support::fs::remove_file(path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fclose_vector_of_fids_closes_each() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path1 = unique_path("fclose_vec1");
    test_support::fs::write(&path1, "a").unwrap();
    let fid1 = run_fopen(&[Value::from(path1.to_string_lossy().to_string())])
        .expect("open 1")
        .as_open()
        .unwrap()
        .fid;
    let path2 = unique_path("fclose_vec2");
    test_support::fs::write(&path2, "b").unwrap();
    let fid2 = run_fopen(&[Value::from(path2.to_string_lossy().to_string())])
        .expect("open 2")
        .as_open()
        .unwrap()
        .fid;
    let tensor = Tensor::new(vec![fid1, fid2], vec![2, 1]).expect("tensor construction");
    let eval = run_evaluate(&[Value::Tensor(tensor)]).expect("fclose");
    assert_eq!(eval.status(), 0.0);
    assert!(registry::info_for(fid1 as i32).is_none());
    assert!(registry::info_for(fid2 as i32).is_none());
    test_support::fs::remove_file(path1).unwrap();
    test_support::fs::remove_file(path2).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fclose_repeat_returns_error_message() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let (fid, path) = open_temp_file("fclose_repeat");
    let first = run_evaluate(&[Value::Num(fid)]).expect("fclose");
    assert_eq!(first.status(), 0.0);
    let second = run_evaluate(&[Value::Num(fid)]).expect("fclose second");
    assert_eq!(second.status(), -1.0);
    assert_eq!(second.message(), FCLOSE_ERROR_INVALID_IDENTIFIER.message);
    test_support::fs::remove_file(path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fclose_keeps_file_registered_when_flush_fails() {
    let _guard = registry_guard();
    let _provider_lock = runmat_filesystem::provider_override_lock();
    registry::reset_for_tests();
    let fail_flush = Arc::new(AtomicBool::new(true));
    let provider = Arc::new(FailingFlushProvider {
        fail_flush: fail_flush.clone(),
        cancel_on_ready: None,
    });
    let _provider_guard = runmat_filesystem::replace_provider(provider);

    let eval = run_fopen(&[Value::from("flush_fail.txt"), Value::from("w")]).expect("fopen");
    let fid = eval.as_open().expect("open result").fid;
    assert!(fid >= 3.0);

    let first = run_evaluate(&[Value::Num(fid)]).expect("first fclose");
    assert_eq!(first.status(), -1.0);
    assert!(first.message().contains("flush failed"));
    assert!(registry::info_for(fid as i32).is_some());
    let handle = registry::shared_handle(fid as i32).expect("handle remains registered");
    assert!(handle.lock().expect("handle lock").is_some());

    fail_flush.store(false, Ordering::SeqCst);
    let second = run_evaluate(&[Value::Num(fid)]).expect("second fclose");
    assert_eq!(second.status(), 0.0);
    assert!(registry::info_for(fid as i32).is_none());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn concurrent_fclose_only_one_call_closes_file() {
    let _guard = registry_guard();
    let _provider_lock = runmat_filesystem::provider_override_lock();
    registry::reset_for_tests();
    let _provider_guard = runmat_filesystem::replace_provider(Arc::new(FailingFlushProvider {
        fail_flush: Arc::new(AtomicBool::new(false)),
        cancel_on_ready: None,
    }));

    let eval = run_fopen(&[Value::from("yielding_close.txt"), Value::from("w")]).expect("fopen");
    let fid = eval.as_open().expect("open result").fid as i32;
    assert!(fid >= 3);

    let (first, second) = futures::executor::block_on(futures::future::join(
        registry::close_async(fid),
        registry::close_async(fid),
    ));
    let closed_count = [first.expect("first close"), second.expect("second close")]
        .into_iter()
        .filter(Option::is_some)
        .count();

    assert_eq!(closed_count, 1);
    assert!(registry::info_for(fid).is_none());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dropping_pending_close_restores_identifier_and_exact_handle() {
    use std::future::Future;
    use std::task::{Context, Poll};

    let _guard = registry_guard();
    let _provider_lock = runmat_filesystem::provider_override_lock();
    registry::reset_for_tests();
    let _provider_guard = runmat_filesystem::replace_provider(Arc::new(FailingFlushProvider {
        fail_flush: Arc::new(AtomicBool::new(false)),
        cancel_on_ready: None,
    }));

    let eval = run_fopen(&[Value::from("pending_close.txt"), Value::from("w")]).expect("fopen");
    let fid = eval.as_open().expect("open result").fid as i32;
    let original = registry::shared_handle(fid).expect("registered handle");
    let mut future = Box::pin(registry::close_async(fid));
    let mut context = Context::from_waker(futures::task::noop_waker_ref());
    assert!(matches!(
        Future::poll(future.as_mut(), &mut context),
        Poll::Pending
    ));
    assert!(registry::info_for(fid).is_none());
    drop(future);

    assert!(registry::info_for(fid).is_some());
    let restored = registry::shared_handle(fid).expect("restored handle");
    assert!(Arc::ptr_eq(&original, &restored));
    assert!(restored.lock().expect("handle lock").is_some());
    let closed = futures::executor::block_on(registry::close_async(fid)).expect("retry");
    assert!(closed.is_some());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn cancelled_pending_flush_restores_session_identifier_and_exact_handle() {
    use crate::context::RuntimeContext;
    use crate::execution::RuntimeExecutionService;
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use std::future::Future;
    use std::rc::Rc;
    use std::task::{Context, Poll};

    let _guard = registry_guard();
    let _provider_lock = runmat_filesystem::provider_override_lock();
    registry::reset_for_tests();
    let _provider_guard = runmat_filesystem::replace_provider(Arc::new(FailingFlushProvider {
        fail_flush: Arc::new(AtomicBool::new(false)),
        cancel_on_ready: None,
    }));
    let cancellation = CancellationSource::new();
    let runtime = RuntimeContext::with_cancellation(
        Rc::new(RuntimeExecutionService::new()),
        cancellation.token(),
    );
    let opened = futures::executor::block_on(fopen::evaluate_in(
        Some(&runtime),
        &[Value::from("pending_session_close.txt"), Value::from("w")],
    ))
    .expect("session open");
    let fid = opened.as_open().expect("open form").fid as i32;
    let original = registry::shared_handle(fid).expect("registered handle");
    let fid_value = [Value::Num(fid as f64)];
    let mut call = Box::pin(evaluate_in(Some(&runtime), &fid_value));
    let mut context = Context::from_waker(futures::task::noop_waker_ref());
    assert!(matches!(call.as_mut().poll(&mut context), Poll::Pending));
    assert!(registry::info_for_in(fid, Some(&runtime)).is_none());
    cancellation.cancel(CancellationReason::User);
    let error = futures::executor::block_on(call).expect_err("cancelled flush");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    let restored = registry::shared_handle(fid).expect("restored handle");
    assert!(Arc::ptr_eq(&original, &restored));
    assert!(restored.lock().expect("handle lock").is_some());
    futures::executor::block_on(registry::close_async_in(fid, Some(&runtime)))
        .expect("later close");
    registry::reset_for_tests();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn cancellation_during_ready_flush_cannot_commit_identifier_removal() {
    use crate::context::RuntimeContext;
    use crate::execution::RuntimeExecutionService;
    use runmat_execution::cancellation::CancellationSource;
    use std::rc::Rc;

    let _guard = registry_guard();
    let _provider_lock = runmat_filesystem::provider_override_lock();
    registry::reset_for_tests();
    let cancellation = CancellationSource::new();
    let _provider_guard = runmat_filesystem::replace_provider(Arc::new(FailingFlushProvider {
        fail_flush: Arc::new(AtomicBool::new(false)),
        cancel_on_ready: Some(cancellation.clone()),
    }));
    let runtime = RuntimeContext::with_cancellation(
        Rc::new(RuntimeExecutionService::new()),
        cancellation.token(),
    );
    let opened = futures::executor::block_on(fopen::evaluate_in(
        Some(&runtime),
        &[Value::from("ready_flush.txt"), Value::from("w")],
    ))
    .expect("session open");
    let fid = opened.as_open().expect("open form").fid as i32;
    let original = registry::shared_handle(fid).expect("registered handle");
    let error = futures::executor::block_on(evaluate_in(Some(&runtime), &[Value::Num(fid as f64)]))
        .expect_err("inline cancellation before close publication");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert!(registry::info_for_in(fid, Some(&runtime)).is_some());
    let restored = registry::shared_handle(fid).expect("restored handle");
    assert!(Arc::ptr_eq(&original, &restored));
    assert!(restored.lock().expect("handle lock").is_some());
    futures::executor::block_on(registry::close_async_in(fid, Some(&runtime)))
        .expect("later close");
    registry::reset_for_tests();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fclose_standard_stream_bool_argument() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = registry_guard();
    registry::reset_for_tests();
    let eval = run_evaluate(&[Value::Bool(true)]).expect("fclose stdout");
    assert_eq!(eval.status(), 0.0);
    assert!(eval.message().is_empty());
    let outputs = eval.outputs();
    assert_eq!(outputs.len(), 2);
    assert!(matches!(outputs[1], Value::CharArray(ref ca) if ca.rows == 1 && ca.cols == 0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fclose_logical_array_converts_to_numeric_ids() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = registry_guard();
    registry::reset_for_tests();
    let logical = LogicalArray::new(vec![1u8, 0u8, 1u8], vec![3]).expect("logical array");
    let eval = run_evaluate(&[Value::LogicalArray(logical)]).expect("fclose logical");
    assert_eq!(eval.status(), 0.0);
    assert!(eval.message().is_empty());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fclose_cell_array_closes_each_entry() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = registry_guard();
    registry::reset_for_tests();
    let (fid1, path1) = open_temp_file("fclose_cell1");
    let (fid2, path2) = open_temp_file("fclose_cell2");
    let cell = CellArray::new(vec![Value::Num(fid1), Value::Num(fid2)], 1, 2).expect("cell");
    let eval = run_evaluate(&[Value::Cell(cell)]).expect("fclose cell");
    assert_eq!(eval.status(), 0.0);
    assert!(registry::info_for(fid1 as i32).is_none());
    assert!(registry::info_for(fid2 as i32).is_none());
    test_support::fs::remove_file(path1).unwrap();
    test_support::fs::remove_file(path2).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fclose_tensor_with_non_integer_entries_errors() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = registry_guard();
    registry::reset_for_tests();
    let tensor = Tensor::new(vec![1.5], vec![1, 1]).expect("tensor");
    let err = unwrap_error_message(run_evaluate(&[Value::Tensor(tensor)]).unwrap_err());
    assert_eq!(
        err,
        format!(
            "{}: file identifier must be an integer",
            FCLOSE_ERROR_INVALID_INPUT.message
        )
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fclose_string_array_all_equivalent() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = registry_guard();
    registry::reset_for_tests();
    let strings = StringArray::new(vec!["all".to_string()], vec![1]).expect("string array");
    let eval = run_evaluate(&[Value::StringArray(strings)]).expect("fclose all");
    assert_eq!(eval.status(), 0.0);
    assert!(eval.message().is_empty());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fclose_accepts_empty_tensor() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = registry_guard();
    registry::reset_for_tests();
    let tensor = Tensor::new(Vec::new(), vec![0, 0]).expect("tensor");
    let eval = run_evaluate(&[Value::Tensor(tensor)]).expect("fclose");
    assert_eq!(eval.status(), 0.0);
    assert!(eval.message().is_empty());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fclose_errors_on_non_numeric_input() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = registry_guard();
    registry::reset_for_tests();
    let err = unwrap_error_message(run_evaluate(&[Value::from("not-a-fid")]).unwrap_err());
    assert_eq!(
        err,
        format!(
            "{}: file identifier must be numeric or 'all'",
            FCLOSE_ERROR_INVALID_INPUT.message
        )
    );
}

#[test]
fn fclose_strict_extension_rejection_precedes_close_side_effects() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let (fid, path) = open_temp_file("fclose_preflight");
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);

    let integer = run_evaluate(&[Value::Int(IntValue::I64(fid as i64))]).unwrap_err();
    assert_eq!(
        integer.identifier(),
        Some("RunMat:compatibility:FcloseIntegerIdExtension")
    );
    assert!(registry::info_for(fid as i32).is_some());

    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let message = futures::executor::block_on(fclose_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(fid)]),
    ))
    .unwrap_err();
    assert_eq!(
        message.identifier(),
        Some("RunMat:compatibility:FcloseMessageOutputExtension")
    );
    assert!(registry::info_for(fid as i32).is_some());

    drop(_outputs);
    run_evaluate(&[Value::Num(fid)]).expect("close after rejected forms");
    test_support::fs::remove_file(path).unwrap();
}

#[test]
fn fclose_integer_extension_accepts_exact_scalar_and_documented_char_all() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let (fid, path) = open_temp_file("fclose_integer_extension");
    {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let result =
            run_evaluate(&[Value::Int(IntValue::U64(fid as u64))]).expect("integer file id");
        assert_eq!(result.status(), 0.0);
    }
    test_support::fs::remove_file(path).unwrap();

    let all = run_evaluate(&[Value::CharArray(runmat_value::CharArray::new_row("all"))])
        .expect("documented char all");
    assert_eq!(all.status(), 0.0);
}
