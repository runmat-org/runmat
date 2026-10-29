use super::*;

/// One interpreter installation executes the *same* compiled reduction five
/// times. Cross-invocation learning is qualified separately; this case proves
/// repeat observations within the same installed region.
#[test]
#[ignore = "requires an actual WGPU adapter; unavailable hardware fails when explicitly run"]
fn actual_vm_folded_min_cold_then_repeated_warm_full_call_receipts() {
    runmat_gc::gc_test_context(|| {
        register_wgpu_provider(WgpuProviderOptions::default())
            .expect("actual WGPU adapter required for repeated installed reduction");
        let bytecode = test_helpers::compile_source(
            "x = gpuArray(reshape(single(1:16384), 128, 128)); for k = 1:5; y = min(x + x); end; z = gather(y);",
        )
        .expect("compile authentic loop-owned folded minimum");
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
        let vars = vec![Value::Num(0.0); bytecode.var_count];
        let runmat_vm::InterpreterOutcome::Completed(completion) =
            block_on(runmat_vm::interpret_workspace(
                &bytecode,
                runmat_vm::InterpreterVars::from_owned(vars),
                None,
                runtime.clone(),
                None,
                None,
            ))
            .outcome
            .expect("five actual VM folded reduction executions");
        let vars = &completion.values;
        let slot = bytecode
            .var_names
            .iter()
            .find_map(|(slot, name)| (name == "z").then_some(*slot))
            .expect("published gathered output");
        let output_value = vars.get(slot).expect("completed output slot");
        let Value::Tensor(output) = &*output_value else {
            panic!(
                "VM did not publish gathered numeric output: {:?}",
                vars.get(slot).as_deref()
            );
        };
        assert_eq!(output.shape(), vec![1, 128]);
        assert_eq!(output.numeric_dtype(), NumericDType::F32);
        assert_eq!(output.materialize_f64(), expected_column_min(2.0));
        block_on(runtime.scope(async {}));

        let selections = placement.reduction_selection.borrow();
        assert_eq!(
            selections.len(),
            5,
            "same installed region must plan five times"
        );
        let cold = &selections[0];
        assert_eq!(cold.preparation, CandidatePreparationState::Cold);
        assert!(matches!(
            cold.signature.revision.authority,
            PlacementExecutionAuthorityRevision::RuntimeBindingSession(_)
        ));
        for (index, selected) in selections.iter().enumerate() {
            let repeated_physical = selections[..index].iter().any(|earlier| {
                earlier.candidate == selected.candidate && earlier.kind == selected.kind
            });
            let expected = if repeated_physical {
                CandidatePreparationState::Warm
            } else {
                CandidatePreparationState::Cold
            };
            assert_eq!(
                selected.preparation, expected,
                "each newly selected physical recipe is cold, then its repeats are warm"
            );
            assert_eq!(selected.signature.revision, cold.signature.revision);
        }
        // The canonical planner may switch physical selections after feedback. Resource
        // epoch may also change while old y is retained. Find an observed
        // repeated Warm series using every exact signature/candidate fact.
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
            .expect(
                "five calls must yield two repeated Warm receipts for one exact physical candidate",
            );
        assert_ne!(cold.signature, learned.signature);

        let receipts = placement.delivered.lock().unwrap();
        let fusion_receipts = receipts
            .iter()
            .filter(|feedback| {
                feedback.kind == runmat_execution::ExecutionCandidateKind::ProviderFusion
            })
            .collect::<Vec<_>>();
        let unrelated_receipts = receipts
            .iter()
            .filter(|feedback| {
                feedback.kind != runmat_execution::ExecutionCandidateKind::ProviderFusion
            })
            .collect::<Vec<_>>();
        assert_eq!(
            fusion_receipts.len(),
            5,
            "each real VM exit must deposit one fused receipt; all feedback: {receipts:?}"
        );
        assert_eq!(unrelated_receipts.len(), 1, "one direct operation receipt");
        assert_eq!(
            unrelated_receipts[0].kind,
            runmat_execution::ExecutionCandidateKind::ProviderOperation
        );
        assert_eq!(unrelated_receipts[0].signature.operation, "single");
        assert!(unrelated_receipts[0].succeeded);
        assert!(unrelated_receipts[0].total_elapsed_ns > 0);
        for selected in selections.iter() {
            let planned = selections
                .iter()
                .filter(|item| {
                    item.signature == selected.signature
                        && item.candidate == selected.candidate
                        && item.kind == selected.kind
                })
                .count();
            let delivered = fusion_receipts
                .iter()
                .filter(|item| {
                    item.signature == selected.signature
                        && item.candidate == selected.candidate
                        && item.kind == selected.kind
                        && item.succeeded
                        && item.total_elapsed_ns > 0
                })
                .count();
            assert_eq!(
                delivered, planned,
                "exactly one valid receipt per selected call"
            );
        }
        assert!(matches!(
            session.observed_candidate_duration(&cold.signature, &cold.candidate, cold.kind),
            CandidateObservationEvidence::Unavailable(
                CandidateCostUnavailable::InsufficientObservations {
                    successful: 1,
                    failed: 0
                }
            )
        ));
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
    });
}
