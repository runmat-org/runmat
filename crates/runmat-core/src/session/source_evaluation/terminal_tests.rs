use super::*;
use runmat_runtime::context::{RuntimeServicePorts, RuntimeWorkspaceService};
use runmat_runtime::interaction::{replace_async_handler, InteractionResponse};
use runmat_runtime::workspace::{BaseWorkspaceOwner, BaseWorkspaceService};
use runmat_value::{CharArray, Tensor, Value};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Mutex,
};
use std::task::Poll;

#[test]
fn captured_source_terminal_modes_preserve_original_owner_and_committed_prefix() {
    runmat_gc::gc_test_context(|| {
        for terminal in ["success", "error", "cancel", "drop"] {
            let cancellation = runmat_execution::cancellation::CancellationSource::new();
            let runtime = RuntimeContext::with_cancellation(
                Rc::new(runmat_runtime::execution::RuntimeExecutionService::new()),
                cancellation.token(),
            );
            let owner = BaseWorkspaceOwner::new();
            let evaluator = SessionSourceEvaluator::new(owner.clone(), runtime.local_identity());
            let _publication = evaluator
                .publish_for_execution(runmat_vm::FunctionRegistry::default(), None)
                .unwrap();
            let bound = evaluator.bind_background_authority().unwrap();
            let base = Rc::new(BaseWorkspaceService::new(owner.clone()));
            let original = Tensor::new(vec![11.0, 12.0], vec![1, 2]).unwrap();
            let pointer = original.as_f64_slice().unwrap().as_ptr();
            RuntimeWorkspaceService::assign_with_context(
                base.as_ref(),
                &runtime,
                "original",
                Value::Tensor(original),
            )
            .unwrap();
            let runtime = runtime.with_service_ports(
                RuntimeServicePorts::default()
                    .with_workspace(base.clone())
                    .with_base_workspace(base)
                    .with_source_evaluation(bound),
            );
            let entered = Arc::new(AtomicBool::new(false));
            let observed = entered.clone();
            let (sender, receiver) = futures::channel::oneshot::channel::<bool>();
            let receiver = Arc::new(Mutex::new(Some(receiver)));
            let _input = {
                let _scope = runtime.enter();
                replace_async_handler(Some(Arc::new(move |_| {
                    let receiver = receiver.lock().unwrap().take().expect("one source input");
                    let observed = observed.clone();
                    Box::pin(async move {
                        observed.store(true, Ordering::Release);
                        if receiver
                            .await
                            .map_err(|_| "abandoned source input".to_owned())?
                        {
                            Ok(InteractionResponse::Line("kept".into()))
                        } else {
                            Err("fixture input failed".into())
                        }
                    })
                })))
            };
            let request = SessionSourceEvaluation::prepare(
                &runtime,
                Arc::from("prefix = 91; x = input('hold', 's'); after_suspend = 99;"),
                Arc::from("<captured-source-terminal>"),
                0,
            )
            .unwrap();
            let mut pending = request.evaluate(runtime.clone());
            futures::executor::block_on(futures::future::poll_fn(|cx| {
                assert!(
                    pending.as_mut().poll(cx).is_pending(),
                    "source must reach real pending input"
                );
                if entered.load(Ordering::Acquire) {
                    Poll::Ready(())
                } else {
                    Poll::Pending
                }
            }));
            match terminal {
                "success" | "error" => {
                    sender.send(terminal == "success").unwrap();
                    let result = futures::executor::block_on(pending.as_mut());
                    if terminal == "success" {
                        result.expect("source completes");
                    } else {
                        assert!(result
                            .unwrap_err()
                            .to_string()
                            .contains("fixture input failed"));
                    }
                }
                "cancel" => {
                    cancellation.cancel(runmat_execution::CancellationReason::User);
                    let error = futures::executor::block_on(pending.as_mut()).unwrap_err();
                    assert!(
                        runmat_runtime::runtime_error::error_has_control_cancellation_origin(
                            &error
                        )
                    );
                }
                "drop" => {}
                _ => unreachable!(),
            }
            drop(pending);
            let snapshot = owner.snapshot().unwrap();
            let Some(Value::Tensor(original)) = snapshot.get("original") else {
                panic!("original owner lost")
            };
            assert_eq!(original.as_f64_slice().unwrap().as_ptr(), pointer);
            assert_eq!(original.as_f64_slice(), Some([11.0, 12.0].as_slice()));
            assert_eq!(
                snapshot.get("prefix"),
                Some(&Value::Num(91.0)),
                "{terminal}"
            );
            if terminal == "success" {
                assert_eq!(
                    snapshot.get("x"),
                    Some(&Value::CharArray(CharArray::new_row("kept")))
                );
                assert_eq!(snapshot.get("after_suspend"), Some(&Value::Num(99.0)));
            } else {
                assert!(snapshot.get("x").is_none());
                assert!(snapshot.get("after_suspend").is_none());
            }
        }
    });
}
