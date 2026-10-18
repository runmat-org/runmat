use super::*;
use futures::channel::oneshot;
use futures::future::{select, Either};
use std::rc::Rc;

#[test]
fn debug_frame_snapshot_uses_source_catalog() {
    let source_id = SourceId(7);
    let _catalog = source_context::replace_source_catalog_with_fullpaths(vec![(
        source_id,
        "demo.m".to_string(),
        Some("/tmp/demo.m".to_string()),
        "a = 1;\nb = 2;\n".to_string(),
    )])
    .expect("debug source catalog admission");
    let _guard = push_frame("demo", Some(source_id), Some((8, 13)));
    let frames = current_frames();
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].function, "demo");
    assert_eq!(frames[0].file, "/tmp/demo.m");
    assert_eq!(frames[0].line, 2);
}

#[test]
fn out_of_order_guard_drop_preserves_the_other_selected_frame() {
    let runtime = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let _selected = runtime.enter();
    let outer = push_frame("outer", None, None);
    let inner = push_frame("inner", None, None);
    assert_eq!(current_function_name().as_deref(), Some("inner"));
    drop(outer);
    assert_eq!(current_function_name().as_deref(), Some("inner"));
    assert_eq!(current_frames().len(), 1);
    drop(inner);
    assert!(current_frames().is_empty());
}

#[test]
fn poll_snapshot_retains_active_frame_after_parent_guard_retires() {
    let mut parent = DebugFrameProjection::default();
    let outer = parent.push(DebugFrame {
        function: "outer".into(),
        source_id: None,
        span: None,
    });
    let inner = parent.push(DebugFrame {
        function: "inner".into(),
        source_id: None,
        span: None,
    });
    let snapshot = parent.snapshot();
    parent.remove(outer);
    parent.remove(inner);
    assert!(parent.frames().is_empty());
    assert_eq!(
        snapshot
            .frames()
            .iter()
            .map(|frame| frame.function.as_str())
            .collect::<Vec<_>>(),
        vec!["inner", "outer"]
    );
}

#[test]
fn linked_invocation_frame_does_not_mutate_its_parent() {
    let runtime = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let first = runtime.clone().with_debug_frame(DebugFrame {
        function: "first".into(),
        source_id: None,
        span: None,
    });
    let second = runtime.clone().with_debug_frame(DebugFrame {
        function: "second".into(),
        source_id: None,
        span: None,
    });
    {
        let _selected = first.enter();
        assert_eq!(current_function_name().as_deref(), Some("first"));
    }
    {
        let _selected = second.enter();
        assert_eq!(current_function_name().as_deref(), Some("second"));
    }
    let _selected = runtime.enter();
    assert!(current_frames().is_empty());
}

#[test]
fn two_pending_siblings_keep_distinct_frames_and_drop_exact_owner() {
    let base = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let first = base.clone().with_debug_frame(DebugFrame {
        function: "first".into(),
        source_id: None,
        span: None,
    });
    let second = base.with_debug_frame(DebugFrame {
        function: "second".into(),
        source_id: None,
        span: None,
    });
    let (first_start_tx, first_start_rx) = oneshot::channel();
    let (first_gate_tx, first_gate_rx) = oneshot::channel::<()>();
    let (second_start_tx, second_start_rx) = oneshot::channel();
    let (second_gate_tx, second_gate_rx) = oneshot::channel::<()>();
    let mut first = Box::pin(first.scope(async move {
        assert_eq!(current_function_name().as_deref(), Some("first"));
        first_start_tx.send(()).unwrap();
        let _ = first_gate_rx.await;
        current_function_name()
    }));
    let mut second = Box::pin(second.scope(async move {
        assert_eq!(current_function_name().as_deref(), Some("second"));
        second_start_tx.send(()).unwrap();
        second_gate_rx.await.expect("second gate remains live");
        current_function_name()
    }));
    assert!(matches!(
        futures::executor::block_on(select(first_start_rx, first.as_mut())),
        Either::Left((Ok(()), _))
    ));
    assert!(matches!(
        futures::executor::block_on(select(second_start_rx, second.as_mut())),
        Either::Left((Ok(()), _))
    ));
    drop(first);
    assert!(first_gate_tx.send(()).is_err());
    second_gate_tx.send(()).expect("second future retains gate");
    assert_eq!(
        futures::executor::block_on(second.as_mut()).as_deref(),
        Some("second")
    );
}
