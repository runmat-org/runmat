use super::*;

fn fixture() -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 9 };
    let index = |local| {
        MirSubscriptStep::Index(MirIndexing {
            argument_spans: None,
            kind: runmat_types::IndexKind::Paren,
            plan: MirIndexPlan::Slice,
            components: vec![MirIndexComponent::ContextualExpr(
                runmat_mir::MirExpressionRegion::from_lowered(
                    vec![statement(local, MirRvalue::End, span)],
                    MirOperand::Local(MirLocalId(local)),
                )
                .unwrap(),
            )],
            result_context: runmat_types::IndexResultContext::ReadSingle,
            cell_expand_all: false,
        })
    };
    let chain = MirRvalue::SubscriptChain(MirSubscriptChain {
        root: MirOperand::Local(MirLocalId(0)),
        steps: vec![
            MirSubscriptStep::Member(runmat_hir::MemberName("child".into())),
            index(1),
            MirSubscriptStep::Member(runmat_hir::MemberName("next".into())),
            index(2),
        ],
        sequence_use: SequenceUse::RequireSingle,
        context: ObjectIndexingContext::Expression,
    });
    lower_body_with_inputs(
        "receiver_replay",
        5,
        vec![statement(
            4,
            MirRvalue::ShortCircuit {
                left: boolean(true),
                op: runmat_mir::MirShortCircuitOp::And,
                right_temps: vec![statement(3, chain, span)],
                right: MirOperand::Local(MirLocalId(3)),
            },
            span,
        )],
        MirOperand::Local(MirLocalId(4)),
        span,
        vec![BindingId(0)],
    )
}

async fn yield_once() {
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
}

#[test]
fn cached_receiver_preserves_ordinal_before_later_and_deeper_suspensions() {
    let runtime = runtime_context();
    let activation = runtime.enter();
    let resolver = runmat_runtime::user_functions::install_semantic_function_resolver(Some(
        Arc::new(|name| match name {
            "receiver_replay_subsref" => Some(13),
            "receiver_replay_end" => Some(14),
            _ => None,
        }),
    ))
    .expect("call-port scope");
    let method = |name: &str, function_name: &str| runmat_runtime::class_registry::RuntimeMethod {
        name: name.into(),
        is_static: false,
        is_abstract: false,
        is_sealed: false,
        access: runmat_types::MemberAccess::Public,
        function_name: function_name.into(),
        implicit_class_argument: None,
        role: runmat_types::ClassMethodRole::Ordinary,
    };
    runmat_runtime::class_registry::register_class(runmat_runtime::class_registry::RuntimeClass {
        events: Default::default(),
        name: "ReceiverReplayObject".into(),
        parent: None,
        properties: Default::default(),
        methods: [
            (
                "subsref".into(),
                method("subsref", "receiver_replay_subsref"),
            ),
            ("end".into(), method("end", "receiver_replay_end")),
        ]
        .into_iter()
        .collect(),
    });
    let subsrefs = Arc::new(AtomicUsize::new(0));
    let ends = Arc::new(AtomicUsize::new(0));
    let calls = Arc::clone(&subsrefs);
    let end_calls = Arc::clone(&ends);
    let invoker = runmat_runtime::user_functions::install_semantic_function_invoker(Some(
        Arc::new(move |function, _, outputs| {
            assert_eq!(outputs, 1);
            let ordinal = match function {
                13 => calls.fetch_add(1, Ordering::SeqCst),
                14 => end_calls.fetch_add(1, Ordering::SeqCst),
                _ => panic!("unexpected function {function}"),
            };
            Box::pin(async move {
                // Suspend at the second preparation, then the second end, then
                // final dispatch: three resumptions at different identity depths.
                if ordinal > 0 {
                    yield_once().await;
                }
                let value = match function {
                    13 if ordinal < 2 => {
                        Value::Object(runmat_value::ObjectInstance::new("ReceiverReplayObject"))
                    }
                    13 => Value::Num(42.0),
                    14 => Value::Num(1.0),
                    _ => unreachable!(),
                };
                rooted_sequence(ValueSequence::single(value).unwrap()).map_err(|error| *error)
            })
        }),
    ))
    .expect("call-port scope");
    drop(activation);
    let executor = compile_executor(fixture()).unwrap();
    let result = futures::executor::block_on(executor.invoke_async(
        FunctionId(0),
        vec![Value::Object(runmat_value::ObjectInstance::new(
            "ReceiverReplayObject",
        ))],
        1,
        runtime,
    ))
    .unwrap();
    assert_eq!(result.outputs, vec![Value::Bool(true)]);
    assert_eq!(subsrefs.load(Ordering::SeqCst), 3);
    assert_eq!(ends.load(Ordering::SeqCst), 2);
    drop(invoker);
    drop(resolver);
}
