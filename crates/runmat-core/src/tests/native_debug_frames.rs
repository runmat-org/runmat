//! The generic native and linked AOT lanes share HostState's selected frame.
//! These public Core calls prove native functions do not inherit a script or
//! parent function frame while an async builtin is pending.

use super::*;
use futures::channel::oneshot;
use futures::future::{select, Either};
use runmat_value::Value;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

const SOURCE: &str = r#"
function [name, frames, file] = nativeFrameOuter()
  [name, frames, file] = nativeFrameInner();
end
function [name, frames, file] = nativeFrameInner()
  info = getcallinfo();
  name = info.name;
  frames = dbstack();
  file = mfilename();
end
function frames = nativeFramePending()
  input('frame gate: ');
  frames = dbstack();
end
"#;

fn frame_name(value: &Value, index: usize) -> Option<&Value> {
    let Value::StructArray(frames) = value else {
        return None;
    };
    frames.field_values("name")?.get(index)
}

fn invocation(function: &str, requested_outputs: usize) -> ProcedureInvocation {
    ProcedureInvocation {
        target: ProcedureTarget::Function(function.into()),
        arguments: Vec::new(),
        requested_outputs,
    }
}

#[test]
fn forced_native_nested_functions_project_exact_debug_frames() {
    runmat_gc::gc_test_context(|| {
        let mut session = RunMatSession::with_options(false, false).expect("session");
        let unit = block_on(session.compile_executable_unit(
            ExecutableSource::new("native-frame@1", "nativeFrames.m", SOURCE),
            None,
        ))
        .expect("compile native frame functions");
        let values = block_on(session.invoke_executable(
            &unit,
            invocation("nativeFrameOuter", 3),
            &InvocationControl::default().force_generic_native(),
        ))
        .expect("native nested call");
        let values = values.as_slice();
        assert_eq!(values[0], Value::String("nativeFrameInner".into()));
        assert_eq!(values[2], Value::String("nativeFrames".into()));
        assert_eq!(
            frame_name(&values[1], 0),
            Some(&Value::String("nativeFrameInner".into()))
        );
        assert_eq!(
            frame_name(&values[1], 1),
            Some(&Value::String("nativeFrameOuter".into()))
        );
        assert!(runmat_runtime::debug_context::current_frames().is_empty());
    });
}

#[test]
fn pending_native_frame_retires_on_drop_and_fresh_call_has_one_frame() {
    runmat_gc::gc_test_context(|| {
        let mut session = RunMatSession::with_options(false, false).expect("session");
        let unit = block_on(session.compile_executable_unit(
            ExecutableSource::new("native-frame-pending@1", "nativeFrames.m", SOURCE),
            None,
        ))
        .expect("compile native frame functions");
        let (reached_tx, reached_rx) = oneshot::channel();
        let (release_tx, release_rx) = oneshot::channel::<()>();
        let reached = Arc::new(Mutex::new(Some(reached_tx)));
        let gate = Arc::new(Mutex::new(Some(release_rx)));
        let calls = Arc::new(AtomicUsize::new(0));
        let reached_for_handler = Arc::clone(&reached);
        let gate_for_handler = Arc::clone(&gate);
        let calls_for_handler = Arc::clone(&calls);
        session.install_async_input_handler(move |_| {
            let ordinal = calls_for_handler.fetch_add(1, Ordering::SeqCst);
            let reached = reached_for_handler.lock().expect("reached lock").take();
            let gate = gate_for_handler.lock().expect("gate lock").take();
            async move {
                if ordinal == 0 {
                    reached
                        .expect("first reached sender")
                        .send(())
                        .expect("reached receiver");
                    gate.expect("first input gate")
                        .await
                        .map_err(|_| "native frame gate abandoned".to_string())?;
                }
                Ok(crate::execution::InputResponse::Line("ok".into()))
            }
        });
        let control = InvocationControl::default().force_generic_native();
        let mut pending = Box::pin(session.invoke_executable(
            &unit,
            invocation("nativeFramePending", 1),
            &control,
        ));
        block_on(async {
            match select(pending.as_mut(), reached_rx).await {
                Either::Left((outcome, _)) => {
                    panic!("native call finished before gate: {outcome:?}")
                }
                Either::Right((reached, _)) => reached.expect("handler reached"),
            }
        });
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(runmat_runtime::debug_context::current_frames().is_empty());
        drop(pending);
        assert!(
            release_tx.send(()).is_err(),
            "dropped call retired pending input"
        );
        assert!(runmat_runtime::debug_context::current_frames().is_empty());

        let values = block_on(session.invoke_executable(
            &unit,
            invocation("nativeFramePending", 1),
            &control,
        ))
        .expect("fresh native frame call");
        assert_eq!(calls.load(Ordering::SeqCst), 2, "no input replay");
        let frames = &values.as_slice()[0];
        assert_eq!(
            frame_name(frames, 0),
            Some(&Value::String("nativeFramePending".into()))
        );
        assert_eq!(frame_name(frames, 1), None, "no retired sibling frame");
    });
}
