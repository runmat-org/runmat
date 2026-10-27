use super::*;
#[path = "short_circuit/fixtures.rs"]
mod fixtures;
#[path = "short_circuit/receiver_replay.rs"]
mod receiver_replay;
#[path = "short_circuit/cancellation.rs"]
mod cancellation;

#[test]
fn suspended_descendant_argument_defers_to_exact_child_without_replay() {
    let runtime = runtime_context();
    let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
    let observed = Arc::clone(&calls);
    let activation = runtime.enter();
    let invoker = runmat_runtime::user_functions::install_semantic_function_invoker(Some(
        Arc::new(move |function, arguments, _| {
            observed.lock().unwrap().push(function);
            if function == 11 {
                assert_eq!(arguments, vec![Value::Bool(true)]);
            }
            Box::pin(async move {
                if function == 9 {
                    let mut yielded = false;
                    futures::future::poll_fn(|cx| {
                        if yielded {
                            std::task::Poll::Ready(())
                        } else {
                            yielded = true;
                            cx.waker().wake_by_ref();
                            std::task::Poll::Pending
                        }
                    })
                    .await;
                    rooted_sequence(ValueSequence::single(Value::Num(1.0)).unwrap())
                        .map_err(|error| *error)
                } else {
                    rooted_sequence(ValueSequence::single(Value::Bool(true)).unwrap())
                        .map_err(|error| *error)
                }
            })
        }),
    ))
    .expect("call-port scope");
    drop(activation);
    let executor = compile_executor(fixtures::nested_argument_fixture()).unwrap();
    let result =
        futures::executor::block_on(executor.invoke_async(FunctionId(0), vec![], 1, runtime))
            .unwrap();
    assert_eq!(result.outputs, vec![Value::Bool(true)]);
    assert_eq!(*calls.lock().unwrap(), vec![8, 9, 11, 10]);
    drop(invoker);
}

fn boolean(value: bool) -> MirOperand {
    MirOperand::Constant(MirConstant::Bool(value))
}

fn nested_call(function: usize, local: usize, span: Span) -> MirRvalue {
    MirRvalue::ShortCircuit {
        left: boolean(true),
        op: runmat_mir::MirShortCircuitOp::And,
        right_temps: vec![statement(
            local,
            contextual_call(function, vec![], span),
            span,
        )],
        right: MirOperand::Local(MirLocalId(local)),
    }
}

fn capture_fixture(taken: bool, count: usize) -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 10 };
    let MirRvalue::Call(mut consume) = contextual_call(9, vec![], span) else {
        unreachable!()
    };
    consume.args = vec![MirCallArg::CapturedSequence(
        runmat_mir::MirSequenceLocalId(0),
    )];
    consume.arg_spans = vec![span];
    let steps = vec![
        // Mutating the left local makes rereading it after suspension observably wrong.
        statement(1, MirRvalue::Use(boolean(false)), span),
        statement(2, nested_call(8, 3, span), span),
        MirStmt {
            kind: MirStmtKind::CaptureSequence {
                destination: runmat_mir::MirSequenceLocalId(0),
                source: runmat_mir::MirExpansionSource::CellContents {
                    base: MirOperand::Local(MirLocalId(0)),
                    indexing: MirIndexing {
                        argument_spans: None,
                        kind: runmat_types::IndexKind::Brace,
                        plan: MirIndexPlan::Cell,
                        components: vec![MirIndexComponent::Colon],
                        result_context: runmat_types::IndexResultContext::ReadCommaList,
                        cell_expand_all: true,
                    },
                },
            },
            span,
        },
        statement(4, MirRvalue::Call(consume), span),
        // Must receive a fresh ordinal even when the first nested region was
        // skipped during resume of the enclosing short-circuit region.
        statement(5, nested_call(10, 6, span), span),
    ];
    lower_body(
        "short_capture",
        8,
        vec![
            statement(
                0,
                MirRvalue::Aggregate {
                    kind: MirAggregateKind::Cell,
                    row_lengths: vec![count],
                    elements: (0..count)
                        .map(|index| {
                            MirAggregateElement::Single(MirOperand::Constant(MirConstant::Number(
                                (index + 1).to_string(),
                            )))
                        })
                        .collect(),
                },
                span,
            ),
            statement(1, MirRvalue::Use(boolean(taken)), span),
            statement(
                7,
                MirRvalue::ShortCircuit {
                    left: MirOperand::Local(MirLocalId(1)),
                    op: runmat_mir::MirShortCircuitOp::And,
                    right_temps: steps,
                    right: MirOperand::Local(MirLocalId(5)),
                },
                span,
            ),
        ],
        MirOperand::Local(MirLocalId(7)),
        span,
    )
}

#[test]
fn captured_empty_and_multi_cells_resume_without_effect_or_branch_replay() {
    for count in [0, 3] {
        let runtime = runtime_context();
        let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
        let observed = Arc::clone(&calls);
        let activation = runtime.enter();
        let invoker = runmat_runtime::user_functions::install_semantic_function_invoker(Some(
            Arc::new(move |function, arguments, _| {
                observed.lock().unwrap().push(function);
                if function == 9 {
                    assert_eq!(
                        arguments,
                        (0..count)
                            .map(|index| Value::Num((index + 1) as f64))
                            .collect::<Vec<_>>()
                    );
                }
                Box::pin(async move {
                    if function == 9 {
                        let mut yielded = false;
                        futures::future::poll_fn(|cx| {
                            if yielded {
                                std::task::Poll::Ready(())
                            } else {
                                yielded = true;
                                cx.waker().wake_by_ref();
                                std::task::Poll::Pending
                            }
                        })
                        .await;
                        runmat_gc::gc_collect_major().map_err(|error| {
                            runmat_runtime::RuntimeError::new(error.to_string())
                        })?;
                    }
                    rooted_sequence(ValueSequence::single(Value::Bool(true)).unwrap())
                        .map_err(|error| *error)
                })
            }),
        ))
        .expect("call-port scope");
        drop(activation);
        let executor = compile_executor(capture_fixture(true, count)).unwrap();
        let result =
            futures::executor::block_on(executor.invoke_async(FunctionId(0), vec![], 1, runtime))
                .unwrap();
        assert_eq!(result.outputs, vec![Value::Bool(true)]);
        assert_eq!(*calls.lock().unwrap(), vec![8, 9, 10]);
        drop(invoker);
    }
}

#[test]
fn skipped_rhs_never_captures_or_invokes_calls() {
    let runtime = runtime_context();
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&calls);
    let activation = runtime.enter();
    let invoker = runmat_runtime::user_functions::install_semantic_function_invoker(Some(
        Arc::new(move |_, _, _| {
            observed.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { Err(runmat_runtime::RuntimeError::new("skipped RHS executed")) })
        }),
    ))
    .expect("call-port scope");
    drop(activation);
    let executor = compile_executor(capture_fixture(false, 3)).unwrap();
    assert_eq!(
        futures::executor::block_on(executor.invoke_async(FunctionId(0), vec![], 1, runtime,))
            .unwrap()
            .outputs,
        vec![Value::Bool(false)]
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    drop(invoker);
}

#[test]
fn caught_failure_clears_unconsumed_capture_before_next_short_circuit() {
    let runtime = runtime_context();
    let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
    let observed = Arc::clone(&calls);
    let activation = runtime.enter();
    let invoker = runmat_runtime::user_functions::install_semantic_function_invoker(Some(
        Arc::new(move |function, arguments, _| {
            observed.lock().unwrap().push(function);
            if function == 10 {
                assert_eq!(arguments, vec![Value::Bool(true), Value::Bool(false)]);
            }
            Box::pin(async move {
                if function == 9 {
                    let mut yielded = false;
                    futures::future::poll_fn(|cx| {
                        if yielded {
                            std::task::Poll::Ready(())
                        } else {
                            yielded = true;
                            cx.waker().wake_by_ref();
                            std::task::Poll::Pending
                        }
                    })
                    .await;
                    Err(runmat_runtime::runtime_error::semantic_error(
                        "ExpectedShortCircuitFailure",
                        "fixture failure",
                    ))
                } else {
                    rooted_sequence(ValueSequence::single(Value::Bool(true)).unwrap())
                        .map_err(|error| *error)
                }
            })
        }),
    ))
    .expect("call-port scope");
    drop(activation);
    let executor = compile_executor(fixtures::caught_capture_fixture()).unwrap();
    let result =
        futures::executor::block_on(executor.invoke_async(FunctionId(0), vec![], 1, runtime))
            .unwrap();
    assert_eq!(result.outputs, vec![Value::Bool(true)]);
    assert_eq!(*calls.lock().unwrap(), vec![9, 10]);
    drop(invoker);
}

#[test]
fn abandoning_suspended_rhs_drops_owned_future_without_replaying_effects() {
    struct DropProbe(Arc<AtomicUsize>);
    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let runtime = runtime_context();
    let drops = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&drops);
    let calls = Arc::new(AtomicUsize::new(0));
    let observed_calls = Arc::clone(&calls);
    let activation = runtime.enter();
    let invoker = runmat_runtime::user_functions::install_semantic_function_invoker(Some(
        Arc::new(move |_, _, _| {
            observed_calls.fetch_add(1, Ordering::SeqCst);
            let probe = DropProbe(Arc::clone(&observed));
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
    let mut invocation = executor
        .begin_request(runmat_native_executor::NativeInvocationRequest {
            function: FunctionId(0),
            captures: vec![],
            arguments: vec![],
            requested_outputs: 1,
            runtime,
            deoptimization: Default::default(),
            osr_target: None,
            workspace: None,
        })
        .unwrap();
    // A generated pre-callback stage may complete before the semantic RHS is
    // polled. Abandon the exact pending callback, not an earlier ready stage.
    loop {
        let runmat_native_executor::execute::NativeInvocationStep::Suspended {
            continuation,
            generation,
        } = invocation.advance().unwrap()
        else {
            panic!("generated RHS must suspend");
        };
        let mut resume = Box::pin(invocation.resume_suspension(continuation, generation));
        match futures::executor::block_on(async { futures::poll!(resume.as_mut()) }) {
            std::task::Poll::Ready(Ok(())) => drop(resume),
            std::task::Poll::Ready(Err(error)) => panic!("RHS resume failed: {error}"),
            std::task::Poll::Pending => {
                assert_eq!(calls.load(Ordering::SeqCst), 1);
                drop(resume);
                break;
            }
        }
    }
    drop(invocation);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    drop(invoker);
}
