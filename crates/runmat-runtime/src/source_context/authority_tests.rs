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
fn selected_source_lineage_inherits_nested_overlay_but_not_sibling_frame() {
    let bound = context()
        .with_program_sources([
            ProgramSourceRecord {
                ordinal: runmat_types::ProgramSourceId(0),
                name: Arc::from("primary.m"),
                fullpath_name: None,
                text: Arc::from("primary text"),
            },
            ProgramSourceRecord {
                ordinal: runmat_types::ProgramSourceId(1),
                name: Arc::from("eval.m"),
                fullpath_name: None,
                text: Arc::from("nested eval text"),
            },
        ])
        .expect("admit sources");
    let selected = bound
        .with_current_program_source(runmat_types::ProgramSourceId(0))
        .expect("select primary function");
    let sibling = selected
        .clone()
        .with_current_program_source(runmat_types::ProgramSourceId(0))
        .expect("issue distinct sibling selection");
    let eval = selected
        .program_source_id(runmat_types::ProgramSourceId(1))
        .expect("admitted nested source");
    futures::executor::block_on(selected.clone().scope(async {
        let overlay = replace_current_source_id_in_context(&selected, Some(eval));
        assert_eq!(current_source_info().unwrap().name.as_ref(), "eval.m");
        selected
            .clone()
            .scope(async {
                assert_eq!(current_source_info().unwrap().name.as_ref(), "eval.m");
            })
            .await;
        sibling
            .scope(async {
                assert_eq!(current_source_info().unwrap().name.as_ref(), "primary.m");
            })
            .await;
        assert_eq!(current_source_info().unwrap().name.as_ref(), "eval.m");
        drop(overlay);
        assert_eq!(current_source_info().unwrap().name.as_ref(), "primary.m");
    }));
    assert!(current_source_info().is_none());
}

#[test]
fn retired_admitted_source_fails_before_current_file_changes() {
    let runtime = context()
        .with_program_sources([ProgramSourceRecord {
            ordinal: runmat_types::ProgramSourceId(0),
            name: Arc::from("retired.m"),
            fullpath_name: None,
            text: Arc::from("retired text"),
        }])
        .expect("admit source");
    let id = runtime
        .program_source_id(runmat_types::ProgramSourceId(0))
        .unwrap();
    // A session teardown/revocation removes the issuing authority. Keep the
    // compiled binding in this test so selection must reject the stale ref.
    runtime
        .state()
        .source
        .borrow_mut()
        .issued_authorities
        .remove(&id);
    assert!(matches!(
        runtime.with_current_program_source(runmat_types::ProgramSourceId(0)),
        Err(crate::context::ProgramSourceSelectionError::Retired(
            runmat_types::ProgramSourceId(0)
        ))
    ));
}

#[test]
fn unbound_native_embedding_clears_ambient_file_for_its_scope() {
    let _outer = replace_current_source_context(Some("unrelated.m"), Some("outer text"));
    assert_eq!(current_source_info().unwrap().name.as_ref(), "unrelated.m");
    let unbound = context()
        .with_current_program_source(runmat_types::ProgramSourceId(0))
        .expect("unbound direct embedding is permitted");
    futures::executor::block_on(unbound.scope(async {
        assert!(current_source_info().is_none());
    }));
    assert_eq!(current_source_info().unwrap().name.as_ref(), "unrelated.m");
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
    let child = parent
        .fork_parallel_lab(crate::context::RuntimeServicePorts::default())
        .expect("idle session fork");
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
