use super::*;

#[path = "cooperation/full_call.rs"]
mod full_call;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn abandoned_cooperative_file_read_restores_registered_owner_and_cursor() {
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy, ExecutionServiceError};
    use runmat_execution::cancellation::CancellationSource;
    use std::future::Future;
    use std::num::NonZeroU64;
    use std::task::{Context, Poll};

    let _guard = registry::test_guard();
    registry::reset_for_tests();
    let path = file_fixture("abandoned.txt", b"12\n34\n");
    let mut options = OpenOptions::new();
    options.read(true);
    let mut file = block_on(options.open_async(&path)).expect("open file");
    std::io::Read::read_exact(&mut file, &mut [0; 3]).expect("advance cursor");
    let handle = Arc::new(StdMutex::new(Some(file)));
    let fid = registry::register_file(RegisteredFile {
        path: path.to_path_buf(),
        permission: "r".to_string(),
        access: registry::FileAccess::READ,
        machinefmt: "native".to_string(),
        encoding: "UTF-8".to_string(),
        handle: handle.clone(),
    });
    let source = CancellationSource::new();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).expect("quantum")),
        source.token(),
        || Box::pin(std::future::pending::<Result<(), ExecutionServiceError>>()),
    );
    let mut cooperation = TextscanCooperation::with_budget(budget);
    let format = parse_format_spec("%f").expect("format");
    let scan_options = TextscanOptions::default();
    let mut call = Box::pin(input::parse_registered_file(
        fid,
        &format,
        None,
        &scan_options,
        &mut cooperation,
    ));
    let mut cx = Context::from_waker(futures::task::noop_waker_ref());
    assert!(matches!(
        Future::poll(call.as_mut(), &mut cx),
        Poll::Pending
    ));
    assert!(matches!(
        registry::acquire_file(fid),
        Err(registry::FileOperationError::Busy)
    ));
    assert_eq!(
        block_on(registry::close_async(fid))
            .expect_err("pending reader blocks close")
            .kind(),
        std::io::ErrorKind::WouldBlock
    );
    drop(call);
    let mut resumed = registry::acquire_file(fid).expect("restored operation");
    assert_eq!(
        std::io::Seek::stream_position(resumed.file_mut()).expect("position"),
        3
    );
    let mut tail = String::new();
    std::io::Read::read_to_string(resumed.file_mut(), &mut tail).expect("tail");
    assert_eq!(tail, "34\n");
    drop(resumed);
    block_on(registry::close_async(fid))
        .expect("close")
        .expect("entry");
    registry::reset_for_tests();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn abandoned_output_projection_rolls_back_the_same_registered_cursor() {
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy, ExecutionServiceError};
    use runmat_execution::cancellation::CancellationSource;
    use std::future::Future;
    use std::num::NonZeroU64;
    use std::task::{Context, Poll};

    let _guard = registry::test_guard();
    registry::reset_for_tests();
    let path = file_fixture("output-abandoned.txt", b"10\n20\n");
    let mut open = OpenOptions::new();
    open.read(true);
    let file = block_on(open.open_async(&path)).expect("open file");
    let fid = registry::register_file(RegisteredFile {
        path: path.to_path_buf(),
        permission: "r".to_string(),
        access: registry::FileAccess::READ,
        machinefmt: "native".to_string(),
        encoding: "UTF-8".to_string(),
        handle: Arc::new(StdMutex::new(Some(file))),
    });
    let format = parse_format_spec("%f").expect("format");
    let options = TextscanOptions::default();
    let mut parsing = TextscanCooperation::new(None);
    let mut parsed = block_on(input::parse_registered_file(
        fid,
        &format,
        Some(1),
        &options,
        &mut parsing,
    ))
    .expect("parsed first row, not published");
    assert_eq!(
        block_on(registry::close_async(fid))
            .expect_err("owner still retained")
            .kind(),
        std::io::ErrorKind::WouldBlock
    );
    let source = CancellationSource::new();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).expect("quantum")),
        source.token(),
        || Box::pin(std::future::pending::<Result<(), ExecutionServiceError>>()),
    );
    let mut publication = TextscanCooperation::with_budget(budget);
    let mut output = Box::pin(build_output(
        std::mem::take(&mut parsed.columns),
        &options,
        &mut publication,
    ));
    let mut cx = Context::from_waker(futures::task::noop_waker_ref());
    assert!(matches!(
        Future::poll(output.as_mut(), &mut cx),
        Poll::Pending
    ));
    drop(output);
    drop(parsed);
    let mut operation = registry::acquire_file(fid).expect("owner restored on output drop");
    assert_eq!(
        std::io::Seek::stream_position(operation.file_mut()).expect("cursor"),
        0
    );
    let mut all = String::new();
    std::io::Read::read_to_string(operation.file_mut(), &mut all).expect("original source");
    assert_eq!(all, "10\n20\n");
    drop(operation);
    block_on(registry::close_async(fid))
        .expect("close")
        .expect("entry");
    registry::reset_for_tests();
}
