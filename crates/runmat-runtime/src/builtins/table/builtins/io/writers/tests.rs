use super::*;
use crate::{
    builtins::common::test_support::builtin_arguments, execution::RuntimeExecutionService,
};
use futures::executor::block_on;
use std::{rc::Rc, sync::Arc};
#[test]
fn selected_writer_publication_and_preflight_preserve_exact_identity_and_bytes() {
    let _lock = runmat_filesystem::provider_override_lock();
    let provider = Arc::new(runmat_filesystem::MemoryFsProvider::with_current_dir(
        "/table-write",
    ));
    let _guard = runmat_filesystem::replace_provider(provider.clone());
    runmat_filesystem::set_current_dir("/table-write").unwrap();
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let value = block_on(crate::builtins::table::table_builtin(
        None,
        builtin_arguments(vec![
            Value::Tensor(Tensor::new(vec![3., 4.], vec![2, 1]).unwrap()),
            Value::from("VariableNames"),
            Value::Cell(CellArray::new(vec![Value::from("Count")], 1, 1).unwrap()),
        ]),
    ))
    .unwrap();
    block_on(writetable_builtin(
        Some(runtime.clone()),
        value.clone(),
        builtin_arguments(vec![Value::from("out.csv")]),
    ))
    .unwrap();
    assert_eq!(
        provider.read_project_path("/table-write/out.csv").unwrap(),
        b"Count\n3\n4\n"
    );
    let error = block_on(writetimetable_builtin(
        Some(runtime.clone()),
        value.clone(),
        builtin_arguments(vec![Value::from("out.csv")]),
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:writetimetable:InvalidArgument")
    );
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    let error = block_on(writetable_builtin(
        Some(runtime.clone()),
        value.clone(),
        builtin_arguments(vec![
            Value::from("out.csv"),
            Value::from("Unknown"),
            Value::Bool(true),
        ]),
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:writetable:InvalidArgument")
    );
    assert_eq!(
        provider.read_project_path("/table-write/out.csv").unwrap(),
        b"Count\n3\n4\n"
    );
    let _outputs = crate::output_count::push_output_count(Some(2)).unwrap();
    let error = block_on(writetable_builtin(
        Some(runtime),
        value,
        builtin_arguments(vec![Value::from("out.csv")]),
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:writetable:InvalidArgument")
    );
    assert_eq!(
        provider.read_project_path("/table-write/out.csv").unwrap(),
        b"Count\n3\n4\n"
    );
}
#[test]
fn writer_io_cause_and_cancellation_are_terminal() {
    use runmat_execution::{cancellation::CancellationSource, CancellationReason};
    use std::error::Error;
    let _lock = runmat_filesystem::provider_override_lock();
    let provider = Arc::new(runmat_filesystem::MemoryFsProvider::with_current_dir(
        "/table-write",
    ));
    let _guard = runmat_filesystem::replace_provider(provider.clone());
    runmat_filesystem::set_current_dir("/table-write").unwrap();
    let value = block_on(crate::builtins::table::table_builtin(
        None,
        builtin_arguments(vec![]),
    ))
    .unwrap();
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let error = block_on(writetable_builtin(
        Some(runtime),
        value.clone(),
        builtin_arguments(vec![Value::from("/")]),
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:writetable:IOError"));
    assert!(error.source().is_some());
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    let source = CancellationSource::new();
    let runtime =
        RuntimeContext::with_cancellation(Rc::new(RuntimeExecutionService::new()), source.token());
    source.cancel(CancellationReason::User);
    let error = block_on(writetable_builtin(
        Some(runtime),
        value,
        builtin_arguments(vec![Value::from("cancelled.csv")]),
    ))
    .unwrap_err();
    assert!(error
        .identifier()
        .is_some_and(|id| id.ends_with("Cancelled")));
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert!(provider
        .read_project_path("/table-write/cancelled.csv")
        .is_err());
}
