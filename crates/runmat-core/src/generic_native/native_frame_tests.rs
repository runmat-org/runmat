//! Two live native invocations from one admitted source binding own separate
//! pending call futures and frame projections, despite sharing a session.

use crate::{ExecutableSource, RunMatSession};
use futures::channel::oneshot;
use futures::executor::block_on;
use futures::future::{select, Either};
use runmat_native_executor::execute::{NativeInvocation, NativeInvocationStep};
use runmat_runtime::interaction::{replace_async_handler, InteractionResponse};
use runmat_value::Value;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

const SOURCE: &str =
    "function frames = pendingNativeFrame()\ninput('frame gate: ');\nframes = dbstack();\nend\n";

fn suspend(invocation: &mut NativeInvocation) -> (u64, u64) {
    loop {
        match invocation.advance().expect("native step") {
            NativeInvocationStep::Suspended {
                continuation,
                generation,
            } => return (continuation, generation),
            NativeInvocationStep::Deoptimized { target, .. }
                if target == runmat_runtime::native::NativeResumeKind::GENERIC_NATIVE =>
            {
                invocation
                    .resume_deoptimization()
                    .expect("generic native deoptimization");
            }
            other => panic!("native input did not suspend: {other:?}"),
        }
    }
}

#[test]
fn sibling_native_pending_frames_remain_isolated_after_out_of_order_drop() {
    runmat_gc::gc_test_context(|| {
        let mut session = RunMatSession::with_options(false, false).expect("session");
        let unit = block_on(session.compile_executable_unit(
            ExecutableSource::new("native-sibling-frames@1", "pendingNativeFrame.m", SOURCE),
            None,
        ))
        .expect("compile native pending function");
        let compiled = super::compile::compile(&unit, Some("pendingNativeFrame"))
            .expect("compile generic native product");
        let runtime = unit
            .bind_runtime_sources(session.runtime_context().clone())
            .expect("admit source binding");
        let (first_reached_tx, first_reached_rx) = oneshot::channel();
        let (second_reached_tx, second_reached_rx) = oneshot::channel();
        let (first_release_tx, first_release_rx) = oneshot::channel();
        let (second_release_tx, second_release_rx) = oneshot::channel();
        let turns = Arc::new(Mutex::new(VecDeque::from([
            (first_reached_tx, first_release_rx),
            (second_reached_tx, second_release_rx),
        ])));
        let calls = Arc::new(AtomicUsize::new(0));
        let turns_for_handler = Arc::clone(&turns);
        let calls_for_handler = Arc::clone(&calls);
        let scope = runtime.enter();
        let _handler = replace_async_handler(Some(Arc::new(move |_| {
            let (reached, release) = turns_for_handler
                .lock()
                .expect("turn queue")
                .pop_front()
                .expect("native input must not replay");
            calls_for_handler.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move {
                reached.send(()).expect("reached receiver");
                release
                    .await
                    .map_err(|_| "native input abandoned".to_string())?;
                Ok(InteractionResponse::Line("0".into()))
            })
        })));
        drop(scope);

        let mut first = compiled
            .executor
            .begin(compiled.entrypoint, Vec::new(), 1, runtime.clone())
            .expect("first native invocation");
        let mut second = compiled
            .executor
            .begin(compiled.entrypoint, Vec::new(), 1, runtime)
            .expect("second native invocation");
        let (first_continuation, first_generation) = suspend(&mut first);
        let (second_continuation, second_generation) = suspend(&mut second);
        let mut first_pending =
            Box::pin(first.resume_suspension(first_continuation, first_generation));
        let mut second_pending =
            Box::pin(second.resume_suspension(second_continuation, second_generation));
        block_on(async {
            match select(first_pending.as_mut(), first_reached_rx).await {
                Either::Left((outcome, _)) => panic!("first call completed early: {outcome:?}"),
                Either::Right((reached, _)) => reached.expect("first handler reached"),
            }
            match select(second_pending.as_mut(), second_reached_rx).await {
                Either::Left((outcome, _)) => panic!("second call completed early: {outcome:?}"),
                Either::Right((reached, _)) => reached.expect("second handler reached"),
            }
        });
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert!(runmat_runtime::debug_context::current_frames().is_empty());
        drop(first_pending);
        assert!(
            first_release_tx.send(()).is_err(),
            "first pending owner retired"
        );
        second_release_tx.send(()).expect("second remains live");
        block_on(second_pending).expect("second input completion");
        let output = block_on(async {
            loop {
                match second.advance().expect("second native step") {
                    NativeInvocationStep::Completed(output) => break output,
                    NativeInvocationStep::Suspended {
                        continuation,
                        generation,
                    } => second
                        .resume_suspension(continuation, generation)
                        .await
                        .expect("complete remaining native operation"),
                    NativeInvocationStep::Deoptimized { target, .. }
                        if target == runmat_runtime::native::NativeResumeKind::GENERIC_NATIVE =>
                    {
                        second
                            .resume_deoptimization()
                            .expect("resume generic native operation");
                    }
                    other => panic!("unexpected remaining native transition: {other:?}"),
                }
            }
        });
        assert_eq!(calls.load(Ordering::SeqCst), 2, "input must not replay");
        // The canonical structure representation normalizes exactly one frame
        // to a scalar; an abandoned sibling would produce a non-scalar array.
        let Value::Struct(frame) = &output.outputs[0] else {
            panic!("dbstack must contain exactly one live frame");
        };
        assert_eq!(
            frame.fields.get("name"),
            Some(&Value::String("pendingNativeFrame".into()))
        );
        assert!(runmat_runtime::debug_context::current_frames().is_empty());
    });
}
