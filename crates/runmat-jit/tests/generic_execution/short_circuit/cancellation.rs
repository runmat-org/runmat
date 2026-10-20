use super::*;
use futures::FutureExt;
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use runmat_native_executor::execute::NativeInvocationStep;

struct DropProbe(Arc<AtomicUsize>);
impl Drop for DropProbe {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn retained_generated_invocation_releases_suspended_work_on_cancel_or_resume_abandonment() {
    for cancel in [false, true] {
        let cancellation = CancellationSource::new();
        let runtime = runtime_context().with_cancellation_token(cancellation.token());
        let drops = Arc::new(AtomicUsize::new(0));
        let observed = drops.clone();
        let calls = Arc::new(AtomicUsize::new(0));
        let observed_calls = calls.clone();
        let activation = runtime.enter();
        let invoker = runmat_runtime::user_functions::install_semantic_function_invoker(Some(
            Arc::new(move |_, _, _| {
                observed_calls.fetch_add(1, Ordering::SeqCst);
                let probe = DropProbe(observed.clone());
                Box::pin(async move {
                    let _probe = probe;
                    futures::future::pending::<()>().await;
                    rooted_sequence(ValueSequence::single(Value::Bool(true)).unwrap())
                        .map_err(|error| *error)
                })
            }),
        ))
        .expect("call-port scope");
        drop(activation);
        let executor = compile_executor(fixtures::caught_capture_fixture()).unwrap();
        let mut invocation = executor.begin(FunctionId(0), vec![], 1, runtime).unwrap();
        let NativeInvocationStep::Suspended {
            mut continuation,
            mut generation,
        } = invocation.advance().unwrap()
        else {
            panic!("generated body must suspend");
        };
        drop(invocation.resume_suspension(continuation, generation));
        assert_eq!(
            drops.load(Ordering::SeqCst),
            0,
            "unpolled resume preserves work"
        );
        let resume = loop {
            let mut resume = Box::pin(invocation.resume_suspension(continuation, generation));
            match resume.as_mut().now_or_never() {
                None => break resume,
                Some(Ok(())) => {
                    drop(resume);
                    let NativeInvocationStep::Suspended {
                        continuation: next_continuation,
                        generation: next_generation,
                    } = invocation.advance().unwrap()
                    else {
                        panic!("generated callback must suspend");
                    };
                    continuation = next_continuation;
                    generation = next_generation;
                }
                Some(Err(error)) => panic!("generated callback resume failed: {error}"),
            }
        };
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        if cancel {
            cancellation.cancel(CancellationReason::User);
            assert!(matches!(
                futures::executor::block_on(resume),
                Err(runmat_native_executor::NativeExecutorError::Cancelled)
            ));
        } else {
            drop(resume);
        }
        assert_eq!(
            drops.load(Ordering::SeqCst),
            1,
            "retained invocation released pending work"
        );
        assert!(invocation.advance().is_err());
        assert!(futures::executor::block_on(
            invocation.resume_suspension(continuation, generation)
        )
        .is_err());
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "no effect replay after terminal outcome"
        );
        drop(invocation);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        drop(invoker);
    }
}
