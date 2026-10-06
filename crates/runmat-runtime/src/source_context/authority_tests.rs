use super::*;
use std::rc::Rc;

fn context() -> crate::context::RuntimeContext {
    crate::context::RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
}

#[test]
fn nested_selected_program_sources_keep_exact_current_file_across_scopes() {
    let runtime = context()
        .with_program_sources([
            ProgramSourceRecord {
                ordinal: runmat_types::ProgramSourceId(0),
                name: Arc::from("primary.m"),
                fullpath_name: None,
                text: Arc::from("primary text"),
            },
            ProgramSourceRecord {
                ordinal: runmat_types::ProgramSourceId(1),
                name: Arc::from("companion.m"),
                fullpath_name: None,
                text: Arc::from("companion text"),
            },
        ])
        .expect("admit source records");
    let primary = runtime
        .clone()
        .with_current_program_source(runmat_types::ProgramSourceId(0))
        .expect("select primary");
    let companion = primary
        .clone()
        .with_current_program_source(runmat_types::ProgramSourceId(1))
        .expect("select companion");
    futures::executor::block_on(primary.clone().scope(async {
        assert_eq!(current_source_info().unwrap().name.as_ref(), "primary.m");
        companion
            .clone()
            .scope(async {
                assert_eq!(current_source_info().unwrap().name.as_ref(), "companion.m");
            })
            .await;
        assert_eq!(current_source_info().unwrap().name.as_ref(), "primary.m");
    }));
    assert!(current_source_info().is_none());
    assert!(matches!(
        runtime.with_current_program_source(runmat_types::ProgramSourceId(2)),
        Err(crate::context::ProgramSourceSelectionError::Missing(
            runmat_types::ProgramSourceId(2)
        ))
    ));
}

#[test]
fn equal_numeric_ids_in_distinct_sessions_never_exchange_source_authority() {
    let first = context();
    let second = context();
    let first_lease = lease_source_context_in_context(&first, "first.m", None, "error('first')")
        .expect("first source");
    let second_lease =
        lease_source_context_in_context(&second, "second.m", None, "error('second')")
            .expect("second source");
    assert_eq!(first_lease.source_id(), second_lease.source_id());
    assert_ne!(first_lease.source_ref(), second_lease.source_ref());
    assert_eq!(
        first_lease.source_ref().resolve_in(&first),
        Some(first_lease.source_id())
    );
    assert_eq!(
        second_lease.source_ref().resolve_in(&second),
        Some(second_lease.source_id())
    );
    assert_eq!(first_lease.source_ref().resolve_in(&second), None);
    assert_eq!(second_lease.source_ref().resolve_in(&first), None);
}

#[test]
fn fork_keeps_issued_origin_and_exact_text_after_parent_lease_retires() {
    let parent = context();
    let lease = lease_source_context_in_context(&parent, "captured.m", None, "captured = 42;")
        .expect("parent source");
    let reference = lease.source_ref();
    let child = parent.fork_parallel_lab(crate::context::RuntimeServicePorts::default());
    assert_eq!(reference.resolve_in(&child), Some(lease.source_id()));
    drop(lease);
    assert_eq!(reference.resolve_in(&parent), None);
    assert_eq!(reference.resolve_in(&child), Some(reference.source_id()));
    assert_eq!(
        source_info_in_context(&child, reference.source_id())
            .unwrap()
            .text
            .as_ref(),
        "captured = 42;"
    );
    let unrelated = context();
    let collision =
        lease_source_context_in_context(&unrelated, "captured.m", None, "captured = 42;")
            .expect("same text, other owner");
    assert_eq!(collision.source_id(), reference.source_id());
    assert_eq!(reference.resolve_in(&unrelated), None);
}
