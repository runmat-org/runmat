use super::*;
use runmat_accelerate::placement::{ExplorationMode, PlacementSessionConfig};
use runmat_accelerate_api::{AccelProvider, ProviderPrecision, ReductionTwoPassMode};
use runmat_execution::{
    CandidateExecutionLocation, ExecutionCandidateKind, PlacementFeedback, PlacementSignature,
};
use std::collections::HashSet;

#[path = "auto_two_pass/placement.rs"]
mod placement;
use placement::{AutoPlacement, AutoSelectedRoute};

const LARGE_MIN_SOURCE: &str =
    "x = gpuArray(reshape(single(1:1048580), 262145, 4)); y = min(x + x); z = gather(y);";
const LARGE_HOST_MIN_SOURCE: &str =
    "x = reshape(single(1:1048580), 262145, 4); y = min(x + x); z = gather(y);";
const LARGE_MIN_EXPECTED: [f64; 4] = [2.0, 524292.0, 1048582.0, 1572872.0];

struct ActualAutoRun {
    signature: PlacementSignature,
    candidate: String,
    kind: ExecutionCandidateKind,
    preparation: CandidatePreparationState,
    feedback: PlacementFeedback,
    route: ActualAutoPhysical,
    offered_provider_candidates: usize,
}

#[derive(Clone, Copy)]
enum ActualAutoPhysical {
    Host,
    Provider { pass_count: u64, partial_bytes: u64 },
}

fn actual_runtime(session: Rc<PlacementSession>) -> (Rc<AutoPlacement>, RuntimeContext) {
    let placement = Rc::new(AutoPlacement::new(session));
    let runtime = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(RuntimeServicePorts::default().with_placement(placement.clone()));
    (placement, runtime)
}

fn execute_actual_auto(
    bytecode: &runmat_vm::Bytecode,
    runtime: &RuntimeContext,
    placement: &AutoPlacement,
) -> ActualAutoRun {
    let selection_start = placement.selections.borrow().len();
    let receipt_start = placement.base.delivered.lock().unwrap().len();
    let prior: HashSet<_> = runmat_accelerate::placement_report()
        .traces
        .iter()
        .map(|trace| trace.correlation)
        .collect();
    let mut vars = vec![Value::Num(0.0); bytecode.var_count];
    block_on(runmat_vm::interpret_with_vars_in_context(
        bytecode,
        &mut vars,
        None,
        runtime.clone(),
    ))
    .expect("actual Auto selected reduction must execute");
    let slot = bytecode
        .var_names
        .iter()
        .find_map(|(slot, name)| (name == "z").then_some(*slot))
        .expect("published gathered output");
    let Value::Tensor(output) = &vars[slot] else {
        panic!("Auto VM did not publish gathered numeric output");
    };
    assert_eq!(output.shape(), [1, 4]);
    assert_eq!(output.numeric_dtype(), NumericDType::F32);
    assert_eq!(output.materialize_f64(), LARGE_MIN_EXPECTED);
    block_on(runtime.scope(async {}));

    let selections = placement.selections.borrow();
    assert_eq!(
        selections.len(),
        selection_start + 1,
        "one new canonical Auto reduction selection"
    );
    let selected = &selections[selection_start];
    assert_eq!(selected.signature.operation, "fusion.reduction");
    assert!(matches!(
        selected.signature.revision.authority,
        PlacementExecutionAuthorityRevision::RuntimeBindingSession(_)
    ));
    let offered_devices = selected
        .request
        .graph
        .nodes
        .iter()
        .flat_map(|node| &node.candidates)
        .filter_map(|candidate| {
            (candidate.descriptor.kind == ExecutionCandidateKind::ProviderFusion)
                .then_some(candidate.descriptor.execution_location)
        })
        .map(|location| {
            let CandidateExecutionLocation::Provider { device_id } = location else {
                panic!("offered provider route has actual provider location");
            };
            device_id
        })
        .collect::<Vec<_>>();
    assert_eq!(
        offered_devices.len(),
        2,
        "both physical offers remain admitted"
    );
    let offered_device = offered_devices[0];
    assert!(offered_devices
        .iter()
        .all(|device| *device == offered_device));
    assert_eq!(
        selected.signature.provider_provenance,
        Some(typed_reduction_placement::selected_provider_capability_digest(offered_device)),
        "host and provider choices must retain the actual offered provider's provenance"
    );
    let offered_provider_candidates = selected
        .request
        .graph
        .nodes
        .iter()
        .flat_map(|node| &node.candidates)
        .filter(|candidate| candidate.descriptor.kind == ExecutionCandidateKind::ProviderFusion)
        .count();
    assert_eq!(
        offered_provider_candidates, 2,
        "Auto must offer both legal physical strategies"
    );

    let report = runmat_accelerate::placement_report();
    let traces = report
        .traces
        .iter()
        .filter(|trace| {
            !prior.contains(&trace.correlation) && trace.operation == selected.signature.operation
        })
        .collect::<Vec<_>>();
    let selected_provider_traces = traces
        .iter()
        .filter(|trace| {
            trace.events.iter().any(|event| {
                event.kind == PlacementEventKind::Selected
                    && event.variant == Some(PlacementVariant::ProviderFusion)
            })
        })
        .collect::<Vec<_>>();
    let typed_queue = traces
        .iter()
        .flat_map(|trace| &trace.events)
        .filter(|event| {
            event.kind == PlacementEventKind::Queue
                && event.variant == Some(PlacementVariant::ProviderFusion)
                && event.reason.as_deref() == Some("typed_reduction_submitted")
        })
        .collect::<Vec<_>>();
    let route = match selected.route {
        AutoSelectedRoute::Provider { device_id } => {
            assert_eq!(device_id, offered_device);
            assert_eq!(selected.kind, ExecutionCandidateKind::ProviderFusion);
            assert_eq!(
                selected_provider_traces.len(),
                1,
                "one selected fused trace"
            );
            assert_eq!(typed_queue.len(), 1, "one typed submitted-route trace");
            let queue = typed_queue[0];
            assert_eq!(queue.duration_ns, None, "Queue is not timing");
            let [attribute] = queue.attributes.as_slice() else {
                panic!("typed Queue must expose exactly one pass-count attribute");
            };
            assert_eq!(attribute.key, "pass_count");
            ActualAutoPhysical::Provider {
                pass_count: attribute.value,
                partial_bytes: queue.bytes.expect("selected partial-byte projection"),
            }
        }
        AutoSelectedRoute::Host => {
            assert_eq!(selected.kind, ExecutionCandidateKind::SharedRuntime);
            assert_eq!(selected.preparation, CandidatePreparationState::Ready);
            assert!(
                selected_provider_traces.is_empty(),
                "host route did not select provider fusion"
            );
            assert!(
                typed_queue.is_empty(),
                "host route did not submit a typed provider kernel"
            );
            ActualAutoPhysical::Host
        }
    };

    let receipts = placement.base.delivered.lock().unwrap();
    let reduction_receipts = receipts[receipt_start..]
        .iter()
        .filter(|feedback| feedback.signature.operation == selected.signature.operation)
        .collect::<Vec<_>>();
    let [receipt] = reduction_receipts.as_slice() else {
        panic!("one exact fenced FullCall reduction receipt per invocation");
    };
    assert_eq!(receipt.signature, selected.signature);
    assert_eq!(receipt.candidate, selected.candidate);
    assert_eq!(receipt.kind, selected.kind);
    assert!(receipt.succeeded);
    assert!(receipt.total_elapsed_ns > 0);
    let result = ActualAutoRun {
        signature: selected.signature.clone(),
        candidate: selected.candidate.clone(),
        kind: selected.kind,
        preparation: selected.preparation,
        feedback: (**receipt).clone(),
        route,
        offered_provider_candidates,
    };
    drop(receipts);
    drop(selections);
    result
}

fn assert_session_only_feedback(session: &PlacementSession, run: &ActualAutoRun) {
    assert!(session.profile_snapshot().feedback.iter().all(|series| {
        series.signature != run.signature
            || series.candidate != run.candidate
            || series.kind != run.kind
    }));
}

/// Default Auto is a profitability policy, not a request to force the
/// provider's heuristic preference. Both physical strategies are offered, and
/// the canonical cold cost currently selects the cheaper SinglePass route.
#[test]
#[ignore = "requires an actual WGPU adapter; unavailable hardware fails when explicitly run"]
fn actual_default_auto_min_selects_single_with_typed_queue_and_fenced_full_call() {
    runmat_gc::gc_test_context(|| {
        let provider = register_wgpu_provider(WgpuProviderOptions::default())
            .expect("actual WGPU adapter required for default Auto qualification");
        assert_eq!(
            provider.reduction_two_pass_mode(),
            ReductionTwoPassMode::Auto
        );
        assert_eq!(provider.precision(), ProviderPrecision::F32);
        let bytecode = test_helpers::compile_source(LARGE_MIN_SOURCE)
            .expect("compile authentic large folded Single minimum");
        assert_compiled_folded_graph(&bytecode, 1);
        let session = Rc::new(PlacementSession::default());
        let (placement, runtime) = actual_runtime(session.clone());

        let selected = execute_actual_auto(&bytecode, &runtime, placement.as_ref());
        let ActualAutoPhysical::Provider {
            pass_count,
            partial_bytes,
        } = selected.route
        else {
            panic!("default cold Auto must select a provider route");
        };
        assert_eq!(pass_count, 1, "default cold Auto selects SinglePass");
        assert_eq!(partial_bytes, 0);
        assert_eq!(selected.offered_provider_candidates, 2);
        assert!(matches!(
            session.observed_candidate_duration(
                &selected.signature,
                &selected.candidate,
                selected.kind,
            ),
            CandidateObservationEvidence::Unavailable(
                CandidateCostUnavailable::InsufficientObservations {
                    successful: 1,
                    failed: 0
                }
            )
        ));
        assert_session_only_feedback(session.as_ref(), &selected);
        assert_eq!(placement.base.revoked_before_publication.get(), 0);
    });
}

/// Canonical transactional exploration can execute Auto's legal alternate
/// without changing provider mode or production costs. The first cold TwoPass
/// is a distinct cost-query bucket; only two later receipts in the same exact
/// warm signature/candidate/kind can reach the evidence threshold.
#[test]
#[ignore = "requires an actual WGPU adapter; unavailable hardware fails when explicitly run"]
fn actual_auto_exploration_executes_two_pass_twice_in_one_live_session() {
    runmat_gc::gc_test_context(|| {
        let provider = register_wgpu_provider(WgpuProviderOptions::default())
            .expect("actual WGPU adapter required for Auto exploration qualification");
        assert_eq!(
            provider.reduction_two_pass_mode(),
            ReductionTwoPassMode::Auto
        );
        assert_eq!(provider.precision(), ProviderPrecision::F32);
        // A selected SharedRuntime alternative requires genuine host F32
        // Add operands; a resident gpuArray source cannot certify that route.
        // The separate default-Auto test retains the resident-input case.
        let bytecode = test_helpers::compile_source(LARGE_HOST_MIN_SOURCE)
            .expect("compile authentic large folded Single minimum");
        assert_compiled_folded_graph(&bytecode, 1);
        let session = Rc::new(PlacementSession::new(PlacementSessionConfig {
            leave_absolute_margin_ns: 0,
            leave_relative_margin_basis_points: 0,
            exploration: ExplorationMode::TransactionalEvery { interval: 1 },
            exploration_max_overhead_basis_points: u32::MAX,
            ..PlacementSessionConfig::default()
        }));
        let (placement, runtime) = actual_runtime(session.clone());

        let first = execute_actual_auto(&bytecode, &runtime, placement.as_ref());
        let ActualAutoPhysical::Provider {
            pass_count,
            partial_bytes,
        } = first.route
        else {
            panic!("first explored Auto route must be provider TwoPass");
        };
        assert_eq!(pass_count, 2, "exploration selects actual TwoPass");
        assert_eq!(first.preparation, CandidatePreparationState::Cold);
        assert!(partial_bytes > 0);
        assert_eq!(first.offered_provider_candidates, 2);
        assert!(matches!(
            session.observed_candidate_duration(&first.signature, &first.candidate, first.kind),
            CandidateObservationEvidence::Unavailable(
                CandidateCostUnavailable::InsufficientObservations {
                    successful: 1,
                    failed: 0
                }
            )
        ));

        // Cold and warm cost queries intentionally produce different runtime
        // facts. Keep invoking the same real source/session until two actual
        // warm TwoPass selections have the identical full placement key.
        // Every intervening host route, SinglePass, or differently keyed
        // provider route passes its own numeric, admission, trace, and exact
        // accepted FullCall receipt checks.
        let mut invocation_count = 1;
        let mut warm_two = Vec::<ActualAutoRun>::new();
        for _ in 0..12 {
            let selected = execute_actual_auto(&bytecode, &runtime, placement.as_ref());
            invocation_count += 1;
            assert_eq!(selected.offered_provider_candidates, 2);
            match selected.route {
                ActualAutoPhysical::Host => {
                    assert_eq!(selected.kind, ExecutionCandidateKind::SharedRuntime);
                }
                ActualAutoPhysical::Provider {
                    pass_count: 1,
                    partial_bytes,
                } => {
                    assert_eq!(selected.kind, ExecutionCandidateKind::ProviderFusion);
                    assert_eq!(partial_bytes, 0);
                }
                ActualAutoPhysical::Provider {
                    pass_count: 2,
                    partial_bytes,
                } => {
                    assert_eq!(selected.kind, ExecutionCandidateKind::ProviderFusion);
                    assert!(partial_bytes > 0);
                    assert_eq!(selected.preparation, CandidatePreparationState::Warm);
                    assert_ne!(selected.signature, first.signature);
                    let prior_exact = warm_two.iter().any(|prior| {
                        prior.signature == selected.signature
                            && prior.candidate == selected.candidate
                            && prior.kind == selected.kind
                    });
                    warm_two.push(selected);
                    if prior_exact {
                        break;
                    }
                }
                ActualAutoPhysical::Provider { pass_count, .. } => {
                    panic!("unexpected physical reduction pass count {pass_count}")
                }
            }
        }
        let warm_second = warm_two
            .last()
            .expect("bounded exploration must execute a warm TwoPass route");
        let warm_first = warm_two[..warm_two.len() - 1]
            .iter()
            .find(|prior| {
                prior.signature == warm_second.signature
                    && prior.candidate == warm_second.candidate
                    && prior.kind == warm_second.kind
            })
            .expect(
                "bounded authentic exploration did not produce two exact warm TwoPass receipts",
            );
        assert_eq!(warm_first.signature, warm_second.signature);
        assert_eq!(warm_first.candidate, warm_second.candidate);
        assert_eq!(warm_first.kind, warm_second.kind);
        assert!(warm_first.feedback.succeeded && warm_second.feedback.succeeded);
        assert!(warm_first.feedback.total_elapsed_ns > 0);
        assert!(warm_second.feedback.total_elapsed_ns > 0);
        let exact_two_receipts = placement
            .base
            .delivered
            .lock()
            .unwrap()
            .iter()
            .filter(|feedback| {
                feedback.signature == warm_first.signature
                    && feedback.candidate == warm_first.candidate
                    && feedback.kind == warm_first.kind
            })
            .count();
        assert_eq!(
            exact_two_receipts, 2,
            "two exact warm TwoPass receipts only"
        );
        let exact_cold_receipts = placement
            .base
            .delivered
            .lock()
            .unwrap()
            .iter()
            .filter(|feedback| {
                feedback.signature == first.signature
                    && feedback.candidate == first.candidate
                    && feedback.kind == first.kind
            })
            .count();
        assert_eq!(
            exact_cold_receipts, 1,
            "cold TwoPass stays in its own bucket"
        );
        assert!(matches!(
            session.observed_candidate_duration(&first.signature, &first.candidate, first.kind),
            CandidateObservationEvidence::Unavailable(
                CandidateCostUnavailable::InsufficientObservations {
                    successful: 1,
                    failed: 0
                }
            )
        ));
        assert!(matches!(
            session.observed_candidate_duration(
                &warm_second.signature,
                &warm_second.candidate,
                warm_second.kind,
            ),
            CandidateObservationEvidence::Observed(duration) if duration.elapsed_ns > 0
        ));
        assert_session_only_feedback(session.as_ref(), &first);
        assert_session_only_feedback(session.as_ref(), warm_second);
        assert_eq!(placement.selections.borrow().len(), invocation_count);
        assert_eq!(placement.base.revoked_before_publication.get(), 0);
    });
}
