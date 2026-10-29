use super::*;

#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn browser_folded_min_learns_across_separate_vm_invocations() {
    // The browser provider is a singleton. Registration must precede the first
    // compile or execution, but no test may assume its pipeline cache starts cold.
    runmat_runtime::builtins::wasm_registry::register_all();
    let provider = register_wgpu_provider_async(WgpuProviderOptions::default())
        .await
        .expect("actual browser WebGPU adapter required; unavailable is failure");
    assert_eq!(provider.precision(), ProviderPrecision::F32);
    let bytecode = test_helpers::compile_source(
        "x = gpuArray(reshape(single(1:16384), 128, 128)); y = min(x + x); z = gather(y);",
    )
    .expect("compile one authentic folded Single minimum program");
    assert_compiled_folded_graph(&bytecode, 1);

    let session = Rc::new(PlacementSession::default());
    let placement = Rc::new(CapturingPlacement {
        session: session.clone(),
        reduction_selection: RefCell::new(Vec::new()),
        delivered: Arc::new(Mutex::new(Vec::new())),
        invalidate_selected_provider: Cell::new(false),
        revoked_before_publication: Cell::new(0),
    });
    let runtime = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(RuntimeServicePorts::default().with_placement(placement.clone()));
    let output_slot = bytecode
        .var_names
        .iter()
        .find_map(|(slot, name)| (name == "z").then_some(*slot))
        .expect("published gathered output");
    let expected = expected_column_min(2.0);

    for call in 0..6 {
        let receipt_start = placement.delivered.lock().unwrap().len();
        let prior_traces: HashSet<_> = runmat_accelerate::placement_report()
            .traces
            .iter()
            .map(|trace| trace.correlation)
            .collect();
        let vars = vec![Value::Num(0.0); bytecode.var_count];
        let runmat_vm::InterpreterOutcome::Completed(completion) = runmat_vm::interpret_workspace(
            &bytecode,
            runmat_vm::InterpreterVars::from_owned(vars),
            None,
            runtime.clone(),
            None,
            None,
        )
        .await
        .outcome
        .expect("independent browser VM installation must execute folded minimum");
        let vars = &completion.values;
        let output_value = vars.get(output_slot).expect("completed output slot");
        let Value::Tensor(output) = &*output_value else {
            panic!(
                "call {call} did not publish gathered tensor: {:?}",
                vars.get(output_slot).as_deref()
            );
        };
        assert_eq!(output.shape(), vec![1, 128], "call {call} output shape");
        assert_eq!(output.numeric_dtype(), NumericDType::F32);
        assert_eq!(
            output.materialize_f64(),
            expected,
            "call {call} column minimum"
        );
        runtime.scope(async {}).await;

        let selections = placement.reduction_selection.borrow();
        assert_eq!(selections.len(), call + 1, "one fused selection per call");
        let selected = &selections[call];
        assert_eq!(selected.signature.operation, "fusion.reduction");
        assert_eq!(
            selected.signature.provider_provenance,
            Some(
                typed_reduction_placement::feedback::selected_provider_capability_digest(
                    selected.device_id
                )
            ),
            "selected browser signature must carry the real WebGPU capability snapshot"
        );
        assert_eq!(selected.signature.region, None);
        assert_eq!(selected.kind, ExecutionCandidateKind::ProviderFusion);
        assert!(matches!(
            selected.signature.revision.authority,
            PlacementExecutionAuthorityRevision::RuntimeBindingSession(_)
        ));
        assert!(matches!(
            selected.preparation,
            CandidatePreparationState::Cold | CandidatePreparationState::Warm
        ));
        let report = runmat_accelerate::placement_report();
        assert!(
            report.traces.iter().any(|trace| {
                !prior_traces.contains(&trace.correlation)
                    && trace.operation == selected.signature.operation
                    && trace.region == selected.signature.region
                    && trace.events.iter().any(|event| {
                        event.kind == PlacementEventKind::Selected
                            && event.variant == Some(PlacementVariant::ProviderFusion)
                    })
                    && trace.events.iter().any(|event| {
                        event.kind == PlacementEventKind::Queue
                            && event.variant == Some(PlacementVariant::ProviderFusion)
                    })
            }),
            "call {call} selected reduction must newly queue on actual WebGPU"
        );

        let receipts = placement.delivered.lock().unwrap();
        let new_receipts = &receipts[receipt_start..];
        // The reduction's fence belongs to this VM exit. The separate
        // `single` conversion can remain pending at the nonblocking drain;
        // its direct binding can also choose a shared-runtime candidate.
        // Capture remains unfiltered, including any late direct receipt.
        let fused = new_receipts
            .iter()
            .filter(|feedback| feedback.kind == ExecutionCandidateKind::ProviderFusion)
            .collect::<Vec<_>>();
        assert_eq!(
            fused.len(),
            1,
            "call {call} must fence exactly one fused receipt; new feedback: {new_receipts:?}"
        );
        for feedback in new_receipts
            .iter()
            .filter(|feedback| feedback.kind != ExecutionCandidateKind::ProviderFusion)
        {
            assert_eq!(feedback.signature.operation, "single");
            assert!(matches!(
                feedback.kind,
                ExecutionCandidateKind::ProviderOperation | ExecutionCandidateKind::SharedRuntime
            ));
            assert!(feedback.succeeded);
            assert!(feedback.total_elapsed_ns > 0);
        }
        assert!(
            receipts
                .iter()
                .filter(|feedback| feedback.kind != ExecutionCandidateKind::ProviderFusion)
                .count()
                <= call + 1,
            "at most one accepted direct observation per executed conversion: {receipts:?}"
        );
        let receipt = fused[0];
        assert_eq!(receipt.signature, selected.signature);
        assert_eq!(receipt.candidate, selected.candidate);
        assert_eq!(receipt.kind, selected.kind);
        assert!(receipt.succeeded, "call {call} must be accepted");
        assert!(receipt.total_elapsed_ns > 0, "call {call} must be timed");
    }

    let selections = placement.reduction_selection.borrow();
    let first_revision = &selections[0].signature.revision;
    assert!(
        selections
            .iter()
            .all(|selected| &selected.signature.revision == first_revision),
        "same bytecode, runtime, and placement session must retain one semantic revision across installations"
    );
    let learned = selections
        .iter()
        .find(|candidate| {
            candidate.preparation == CandidatePreparationState::Warm
                && selections
                    .iter()
                    .filter(|selected| {
                        selected.preparation == CandidatePreparationState::Warm
                            && selected.signature == candidate.signature
                            && selected.candidate == candidate.candidate
                            && selected.kind == candidate.kind
                    })
                    .count()
                    >= 2
        })
        .expect("two genuine Warm selections must share one exact physical feedback series");
    assert!(matches!(
        session.observed_candidate_duration(
            &learned.signature,
            &learned.candidate,
            learned.kind,
        ),
        CandidateObservationEvidence::Observed(observed) if observed.elapsed_ns > 0
    ));
    assert!(session.profile_snapshot().feedback.iter().all(|series| {
        selections
            .iter()
            .all(|selected| series.signature != selected.signature)
    }));

    let request = learned.request.clone();
    let _ = session.plan(request.clone()).unwrap();
    assert!(session.plan(request.clone()).unwrap().from_cache);
    let provider_digest =
        typed_reduction_placement::feedback::selected_provider_capability_digest(learned.device_id);
    session.invalidate(runmat_execution::PlacementInvalidation::Provider {
        digest: provider_digest,
    });
    assert!(matches!(
        session.observed_candidate_duration(&learned.signature, &learned.candidate, learned.kind),
        CandidateObservationEvidence::Unavailable(CandidateCostUnavailable::NoObservations)
    ));
    assert!(
        !session.plan(request).unwrap().from_cache,
        "matching provider invalidation retires the exact browser VM decision"
    );
}

#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn browser_selected_provider_invalidation_before_publication_suppresses_receipt() {
    runmat_runtime::builtins::wasm_registry::register_all();
    let provider = register_wgpu_provider_async(WgpuProviderOptions::default())
        .await
        .expect("actual browser WebGPU adapter required; unavailable is failure");
    assert_eq!(provider.precision(), ProviderPrecision::F32);
    let bytecode = test_helpers::compile_source(
        "x = gpuArray(reshape(single(1:16384), 128, 128)); y = min(x + x); z = gather(y);",
    )
    .expect("compile authentic browser folded minimum");
    assert_compiled_folded_graph(&bytecode, 1);
    let session = Rc::new(PlacementSession::default());
    let placement = Rc::new(CapturingPlacement {
        session: session.clone(),
        reduction_selection: RefCell::new(Vec::new()),
        delivered: Arc::new(Mutex::new(Vec::new())),
        invalidate_selected_provider: Cell::new(true),
        revoked_before_publication: Cell::new(0),
    });
    let runtime = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(RuntimeServicePorts::default().with_placement(placement.clone()));
    let vars = vec![Value::Num(0.0); bytecode.var_count];
    let runmat_vm::InterpreterOutcome::Completed(completion) = runmat_vm::interpret_workspace(
        &bytecode,
        runmat_vm::InterpreterVars::from_owned(vars),
        None,
        runtime.clone(),
        None,
        None,
    )
    .await
    .outcome
    .expect("provider invalidation revokes learning, not the valid browser execution");
    let vars = &completion.values;
    let slot = bytecode
        .var_names
        .iter()
        .find_map(|(slot, name)| (name == "z").then_some(*slot))
        .expect("published gathered output");
    let output_value = vars.get(slot).expect("completed output slot");
    let Value::Tensor(output) = &*output_value else {
        panic!("browser VM did not publish a numeric tensor");
    };
    assert_eq!(output.shape(), vec![1, 128]);
    assert_eq!(output.numeric_dtype(), NumericDType::F32);
    assert_eq!(output.materialize_f64(), expected_column_min(2.0));
    runtime.scope(async {}).await;
    assert_eq!(placement.revoked_before_publication.get(), 1);
    let selections = placement.reduction_selection.borrow();
    assert_eq!(selections.len(), 1);
    let selected = &selections[0];
    assert!(placement.delivered.lock().unwrap().iter().all(|feedback| {
        feedback.signature != selected.signature
            || feedback.candidate != selected.candidate
            || feedback.kind != selected.kind
    }));
    assert!(matches!(
        session.observed_candidate_duration(
            &selected.signature,
            &selected.candidate,
            selected.kind
        ),
        CandidateObservationEvidence::Unavailable(CandidateCostUnavailable::NoObservations)
    ));
}
