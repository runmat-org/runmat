use super::*;
use runmat_accelerate::placement::{ExplorationMode, PlacementSessionConfig};
use runmat_accelerate_api::{
    allocation::AllocationReleaseOutcome, AccelProvider, GpuHandleProvenance, GpuTensorHandle,
    GpuTensorStorage, HostNumericDataOwned, HostNumericDataView, HostNumericTensorView,
    ProviderPrecision, ReductionTwoPassMode,
};
use runmat_execution::{
    CandidateExecutionLocation, CandidateOutputPresentation, CandidateOutputResidency,
    CandidateResourceDemand, ExecutionCandidateDescriptor, ExecutionCandidateKind,
    PlacementFeedback, PlacementSignature, ResourceByteBound,
};
use std::collections::{HashMap, HashSet};

#[path = "auto_two_pass/placement.rs"]
mod placement;
use placement::{AutoPlacement, AutoSelectedRoute};

const LARGE_MIN_SOURCE: &str =
    "x = gpuArray(reshape(single(1:1048580), 262145, 4)); y = min(x + x); z = gather(y);";
const LARGE_EXTERNAL_MIN_SOURCE: &str = "y = min(x + x); z = gather(y);";
const LARGE_MIN_EXPECTED: [f64; 4] = [2.0, 524292.0, 1048582.0, 1572872.0];

struct ActualAutoRun {
    signature: PlacementSignature,
    candidate: String,
    kind: ExecutionCandidateKind,
    preparation: CandidatePreparationState,
    feedback: PlacementFeedback,
    route: ActualAutoPhysical,
    offered_provider_candidates: usize,
    offered_host_candidates: usize,
    selected_descriptor: ExecutionCandidateDescriptor,
    selected_resources: CandidateResourceDemand,
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

fn compile_external_x() -> runmat_vm::Bytecode {
    let variables = HashMap::from([("x".to_string(), 0)]);
    let ast = runmat_parser::parse(LARGE_EXTERNAL_MIN_SOURCE)
        .expect("parse external Automatic input source");
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::new(&variables))
        .expect("lower external Automatic input source");
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly)
        .expect("lower external Automatic input MIR");
    runmat_vm::compile(&hir.assembly, &mir, hir.assembly.entrypoints[0].id)
        .expect("compile external Automatic input bytecode")
}

#[test]
fn external_automatic_input_uses_reconciled_compiler_stack_certificate() {
    let bytecode = compile_external_x();
    let (graph, groups) = bytecode
        .reconciled_runtime_fusion_graph()
        .expect("external Automatic source must reconcile");
    let reduction_groups = groups
        .iter()
        .filter(|group| group.kind == runmat_accelerate::FusionKind::Reduction)
        .collect::<Vec<_>>();
    let [group] = reduction_groups.as_slice() else {
        panic!("expected one reduction group, got {reduction_groups:?}");
    };
    assert_eq!(
        group.span,
        runmat_accelerate::InstrSpan { start: 2, end: 5 }
    );
    let prefix = bytecode
        .instructions
        .get(..group.span.start)
        .expect("instructions before the reduction span");
    assert_eq!(prefix.len(), 2);
    assert!(prefix
        .iter()
        .all(|instruction| matches!(instruction, runmat_vm::Instr::LoadVar(0))));
    let layout = group
        .stack_layout
        .as_ref()
        .expect("compiler-owned reduction stack layout");
    assert_eq!(layout.required_stack_operands, 2);
    assert_eq!(layout.bindings, layout.compiler_variable_bindings);
    assert_eq!(layout.compiler_variable_bindings.len(), 2);
    assert_eq!(layout.compiler_variable_bindings[0].stack_offset, 0);
    assert_eq!(layout.compiler_variable_bindings[1].stack_offset, 1);
    let value_id = layout.compiler_variable_bindings[0].value_id;
    assert_eq!(layout.compiler_variable_bindings[1].value_id, value_id);
    assert!(matches!(
        graph.value(value_id).map(|value| &value.origin),
        Some(runmat_accelerate::ValueOrigin::Variable {
            kind: runmat_accelerate::VarKind::Global,
            index: 0
        })
    ));
    assert!(group.nodes.iter().all(|node| graph.node(*node).is_some()));
    assert_eq!(group.nodes.len(), 2, "one Add and one minimum");
}

fn named_slot(bytecode: &runmat_vm::Bytecode, name: &str) -> usize {
    bytecode
        .var_names
        .iter()
        .find_map(|(slot, candidate)| (candidate == name).then_some(*slot))
        .expect("named external input or output")
}

fn execute_actual_auto(
    bytecode: &runmat_vm::Bytecode,
    runtime: &RuntimeContext,
    placement: &AutoPlacement,
    external_x: Option<&GpuTensorHandle>,
) -> ActualAutoRun {
    let selection_start = placement.selections.borrow().len();
    let receipt_start = placement.base.delivered.lock().unwrap().len();
    let prior: HashSet<_> = runmat_accelerate::placement_report()
        .traces
        .iter()
        .map(|trace| trace.correlation)
        .collect();
    let mut vars = vec![Value::Num(0.0); bytecode.var_count];
    if let Some(external_x) = external_x {
        vars[named_slot(bytecode, "x")] = Value::GpuTensor(external_x.clone());
    }
    let runmat_vm::InterpreterOutcome::Completed(completion) =
        block_on(runmat_vm::interpret_workspace(
            bytecode,
            runmat_vm::InterpreterVars::from_owned(vars),
            None,
            runtime.clone(),
            None,
            None,
        ))
        .outcome
        .expect("actual Auto selected reduction must execute");
    let vars = &completion.values;
    let slot = bytecode
        .var_names
        .iter()
        .find_map(|(slot, name)| (name == "z").then_some(*slot))
        .expect("published gathered output");
    let output_value = vars.get(slot).expect("completed output slot");
    let Value::Tensor(output) = &*output_value else {
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
        Some(
            typed_reduction_placement::feedback::selected_provider_capability_digest(
                offered_device
            )
        ),
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
    let offered_host_candidates = selected
        .request
        .graph
        .nodes
        .iter()
        .flat_map(|node| &node.candidates)
        .filter(|candidate| candidate.descriptor.kind == ExecutionCandidateKind::SharedRuntime)
        .count();
    assert!(
        offered_host_candidates <= 1,
        "one bounded whole-work host alternative"
    );
    let selected_offer = selected
        .request
        .graph
        .nodes
        .iter()
        .flat_map(|node| &node.candidates)
        .find(|candidate| {
            candidate.descriptor.identity == selected.candidate
                && candidate.descriptor.kind == selected.kind
        })
        .expect("selected Auto route was offered by the exact request");

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
        offered_host_candidates,
        selected_descriptor: selected_offer.descriptor.clone(),
        selected_resources: selected_offer.resources,
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

        let selected = execute_actual_auto(&bytecode, &runtime, placement.as_ref(), None);
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
        assert_eq!(
            selected.offered_host_candidates, 0,
            "explicit gpuArray intent is never offered implicit host readback"
        );
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
        // The external value is an ordinary compiler entry-stack binding.
        // Automatic residency is supplied by the selected provider owner, so
        // host readback must use the compiler certificate rather than infer an
        // in-span LoadVar or override explicit gpuArray intent.
        let bytecode = compile_external_x();
        assert_compiled_folded_graph(&bytecode, 1);
        let shape = [262_145, 4];
        let input = (1..=1_048_580)
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        let resident = provider
            .upload_numeric_raw(&HostNumericTensorView {
                data: HostNumericDataView::F32(&input),
                shape: &shape,
                storage: GpuTensorStorage::Real,
            })
            .expect("upload exact Single source")
            .with_provenance(GpuHandleProvenance::Automatic)
            .expect("Automatic provider provenance");
        let session = Rc::new(PlacementSession::new(PlacementSessionConfig {
            leave_absolute_margin_ns: 0,
            leave_relative_margin_basis_points: 0,
            exploration: ExplorationMode::TransactionalEvery { interval: 1 },
            exploration_max_overhead_basis_points: u32::MAX,
            ..PlacementSessionConfig::default()
        }));
        let (placement, runtime) = actual_runtime(session.clone());

        let first = execute_actual_auto(&bytecode, &runtime, placement.as_ref(), Some(&resident));
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
        assert_eq!(
            first.offered_host_candidates, 1,
            "external Automatic source admits one checked readback route"
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

        // Cold and warm cost queries intentionally produce different runtime
        // facts. Keep invoking the same real source/session until two actual
        // warm TwoPass selections have the identical full placement key.
        // Every intervening host route, SinglePass, or differently keyed
        // provider route passes its own numeric, admission, trace, and exact
        // accepted FullCall receipt checks.
        let mut invocation_count = 1;
        let mut warm_two = Vec::<ActualAutoRun>::new();
        let mut selected_host = None;
        for _ in 0..12 {
            let selected =
                execute_actual_auto(&bytecode, &runtime, placement.as_ref(), Some(&resident));
            invocation_count += 1;
            assert_eq!(selected.offered_provider_candidates, 2);
            assert_eq!(
                selected.offered_host_candidates, 1,
                "all real trials retain the executable host alternative"
            );
            match selected.route {
                ActualAutoPhysical::Host => {
                    assert_eq!(selected.kind, ExecutionCandidateKind::SharedRuntime);
                    selected_host.get_or_insert(selected);
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
        let host = selected_host.expect("canonical exploration must select the resident CPU route");
        assert_eq!(
            host.selected_descriptor.execution_location,
            CandidateExecutionLocation::Host
        );
        assert_eq!(
            host.selected_descriptor.output_residency,
            CandidateOutputResidency::Host
        );
        let CandidateOutputPresentation::ProviderReadbacksHost { readbacks } =
            &host.selected_descriptor.presentation
        else {
            panic!("resident CPU route needs an authenticated provider readback");
        };
        let readbacks = readbacks.iter().collect::<Vec<_>>();
        assert_eq!(readbacks.len(), 1, "one exact resident source owner");
        assert_eq!(readbacks[0].device_id, provider.device_id());
        assert_eq!(
            readbacks[0].queue_slots,
            provider.download_numeric_queue_slots(&resident)
        );
        assert_eq!(
            readbacks[0].provider_scratch_bytes,
            provider.download_numeric_provider_scratch_bound(&resident)
        );
        assert_eq!(host.selected_resources.cpu_millicores, 1_000);
        assert_eq!(host.selected_resources.queue_slots, 0);
        assert_eq!(
            host.selected_resources.scratch_bytes,
            ResourceByteBound::Known(
                host.selected_descriptor
                    .cost
                    .scratch_bytes
                    .expect("typed host scratch bound")
            )
        );
        assert!(matches!(
            host.selected_resources.retained_bytes,
            ResourceByteBound::Known(bytes) if bytes > 0
        ));
        assert_eq!(host.feedback.kind, ExecutionCandidateKind::SharedRuntime);
        assert!(host.feedback.succeeded && host.feedback.total_elapsed_ns > 0);
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
        let downloaded = block_on(provider.download_numeric(&resident))
            .expect("original Automatic source remains readable");
        assert_eq!(downloaded.shape, shape);
        let HostNumericDataOwned::F32(actual) = downloaded.data else {
            panic!("original Automatic source lost Single storage");
        };
        assert_eq!(actual, input);
        assert_eq!(
            resident.release().expect("retire original source owner"),
            AllocationReleaseOutcome::Retired
        );
        assert_eq!(placement.selections.borrow().len(), invocation_count);
        assert_eq!(placement.base.revoked_before_publication.get(), 0);
    });
}
