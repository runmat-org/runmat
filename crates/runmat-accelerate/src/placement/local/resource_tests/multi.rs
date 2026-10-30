//! Physical candidate-set fixtures, separate from legacy one-record resource tests.
use super::super::multi::{
    plan_local_multi_with_provider, LocalMultiPlacementRequest, LocalPhysicalCandidate,
    LocalProviderAdmission,
};
use super::*;
use crate::placement::tensor_representation;
use runmat_accelerate_api::test_support::TestAllocationOwner;
use runmat_accelerate_api::ProviderFeasibility;
use std::sync::atomic::{AtomicUsize, Ordering};
mod checked_recipe;

#[test]
fn quote_availability_does_not_bypass_physical_admission_or_erase_receipts() {
    let provider = MultiProvider {
        device: 7,
        owner: None,
        reject: None,
        queries: AtomicUsize::new(0),
    };
    for quoted in [false, true] {
        for host_available in [false, true] {
            for cpu_price in [Some(100), None] {
                let capture = Rc::new(Capture::default());
                let mut route = physical(19, 1, CandidatePreparationState::Cold);
                route.fallback = if quoted {
                    // Exact fixture quote, but its scratch plus retained output
                    // exceeds this provider's actual finite capacity.
                    runmat_execution::CandidateCostEvidence::Estimated(cost(1, 1 << 20))
                } else {
                    runmat_execution::CandidateCostEvidence::Unavailable(
                        runmat_execution::CandidateCostUnavailable::NoEstimate,
                    )
                };
                let outcome = try_plan_physical_set_with_admission(
                    &provider,
                    &capture,
                    cpu_price,
                    vec![route],
                    if host_available {
                        LocalCpuAdmission::Legacy
                    } else {
                        LocalCpuAdmission::Unavailable
                    },
                    LocalProviderAdmission::Available,
                );
                if !host_available {
                    let error =
                        outcome.expect_err("finite provider budget requires a real scratch bound");
                    assert_eq!(
                        error.identifier(),
                        Some(if quoted {
                            "RunMat:Placement:NoLegalCandidate"
                        } else {
                            "RunMat:Placement:InsufficientResourceEvidence"
                        })
                    );
                    assert!(capture.requests.borrow().is_empty());
                    continue;
                }
                let outcome = outcome.expect("the independently admitted host remains legal");
                let LocalPlacementOutcome::Selected {
                    selected,
                    cpu_cost,
                    provider_cost,
                    predicted_total_ns,
                    provider_query,
                    provider_index,
                    feedback,
                    decision_kind,
                    ..
                } = outcome
                else {
                    panic!("legal host/provider-only execution")
                };
                assert!(!selected.kind.is_provider());
                assert_eq!(provider_index, None);
                assert_eq!(provider_query.workload.elements, Some(19));
                assert_eq!(provider_cost.is_some(), quoted);
                assert_eq!(predicted_total_ns, cpu_price);
                assert_eq!(cpu_cost.estimate().is_some(), cpu_price.is_some());
                if !quoted || cpu_price.is_none() {
                    assert!(
                        feedback.is_none(),
                        "unpriced partial calls cannot mint complete-route learning"
                    );
                }
                assert_eq!(
                    decision_kind,
                    if cpu_price.is_some() {
                        runmat_execution::WholeWorkDecisionKind::Uncompared
                    } else {
                        runmat_execution::WholeWorkDecisionKind::Unmeasured
                    }
                );
                assert_eq!(
                    capture.requests.borrow().len(),
                    usize::from(quoted && cpu_price.is_some())
                );
            }
        }
    }
}

struct MultiProvider {
    device: u32,
    owner: Option<TestAllocationOwner>,
    reject: Option<(u64, ProviderRejectionCode)>,
    queries: AtomicUsize,
}

fn registered_fixture() -> &'static MultiProvider {
    let owner = TestAllocationOwner::new();
    let device = owner.device_id();
    let provider: &'static MultiProvider = Box::leak(Box::new(MultiProvider {
        device,
        owner: Some(owner),
        reject: None,
        queries: AtomicUsize::new(0),
    }));
    // Retained for the process/worker lifetime under its unique owner device.
    unsafe { runmat_accelerate_api::register_device_provider(provider) };
    provider
}

impl AccelProvider for MultiProvider {
    fn device_id(&self) -> u32 {
        self.device
    }
    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        panic!("multi-candidate planning must not upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        panic!("multi-candidate planning must not download")
    }
    fn device_info(&self) -> String {
        "multi-physical-fixture".into()
    }
    fn query_feasibility(&self, query: &ProviderFeasibilityQuery) -> ProviderFeasibility {
        self.queries.fetch_add(1, Ordering::SeqCst);
        if let Some((marker, code)) = &self.reject {
            if query.workload.elements == Some(*marker) {
                return ProviderFeasibility::rejected(*code, "fixture strategy unavailable");
            }
        }
        let output_bytes = query.outputs.iter().try_fold(0_u64, |total, output| {
            total.checked_add(output.checked_byte_len()?)
        });
        let transient_bytes = output_bytes.and_then(|output| {
            output.checked_add(
                query
                    .workload
                    .typed_reduction
                    .as_ref()
                    .map_or(0, |typed| typed.work.partial_requested_bytes),
            )
        });
        ProviderFeasibility::supported(ProviderResourceEstimate {
            output_bytes,
            transient_bytes,
            dispatches: Some(query.workload.elements.unwrap_or(1) as u32),
        })
    }
    fn capability_snapshot(&self) -> ProviderCapabilitySnapshot {
        let mut snapshot = ProviderCapabilitySnapshot::conservative(self);
        snapshot.concurrency.transactional_results = true;
        snapshot
    }
    fn placement_resources(&self) -> ProviderResourceSnapshot {
        ProviderResourceSnapshot {
            device_id: self.device_id(),
            capacity_bytes: Some(1 << 20),
            ..Default::default()
        }
    }
}

fn physical(
    marker: u64,
    ns: u64,
    preparation: CandidatePreparationState,
) -> LocalPhysicalCandidate<'static> {
    LocalPhysicalCandidate {
        authority: None,
        physical_identity: Some(Digest::sha256([marker as u8])),
        provider_kind: ExecutionCandidateKind::ProviderFusion,
        data_inputs: &[],
        source_inputs: Some(vec![]),
        prepared_inputs: Some(vec![]),
        input_adaptation: None,
        outputs: vec![ProviderRepresentation {
            element_type: runmat_accelerate_api::ProviderElementType::F64,
            storage: runmat_accelerate_api::ProviderStorage::DenseReal,
            layout: runmat_accelerate_api::ProviderLayout::ColumnMajorContiguous,
            shape: vec![1, 1],
            residency: runmat_accelerate_api::ProviderResidency::Device,
        }],
        workload: ProviderWorkload {
            elements: Some(marker),
            ..Default::default()
        },
        preparation,
        preparation_program: None,
        shader_requirements: None,
        fallback: runmat_execution::CandidateCostEvidence::Estimated(cost(ns, 0)),
        required_upload_bytes: Some(0),
        required_download_bytes: 0,
        downstream_materialization: false,
    }
}

fn plan_physical_set<'a>(
    provider: &MultiProvider,
    capture: &Rc<Capture>,
    cpu_ns: u64,
    physical: Vec<LocalPhysicalCandidate<'a>>,
) -> LocalPlacementOutcome {
    try_plan_physical_set(provider, capture, cpu_ns, physical).unwrap()
}

fn try_plan_physical_set<'a>(
    provider: &MultiProvider,
    capture: &Rc<Capture>,
    cpu_ns: u64,
    physical: Vec<LocalPhysicalCandidate<'a>>,
) -> Result<LocalPlacementOutcome, Box<runmat_runtime::RuntimeError>> {
    try_plan_physical_set_with_admission(
        provider,
        capture,
        Some(cpu_ns),
        physical,
        LocalCpuAdmission::Legacy,
        LocalProviderAdmission::Available,
    )
}

fn try_plan_physical_set_with_admission<'a>(
    provider: &MultiProvider,
    capture: &Rc<Capture>,
    cpu_ns: Option<u64>,
    physical: Vec<LocalPhysicalCandidate<'a>>,
    cpu_admission: LocalCpuAdmission,
    implementation: LocalProviderAdmission,
) -> Result<LocalPlacementOutcome, Box<runmat_runtime::RuntimeError>> {
    try_plan_physical_set_with_context(
        provider,
        capture,
        cpu_ns,
        physical,
        cpu_admission,
        implementation,
        true,
    )
}

fn try_plan_physical_set_with_context<'a>(
    provider: &MultiProvider,
    capture: &Rc<Capture>,
    cpu_ns: Option<u64>,
    physical: Vec<LocalPhysicalCandidate<'a>>,
    cpu_admission: LocalCpuAdmission,
    implementation: LocalProviderAdmission,
    with_context: bool,
) -> Result<LocalPlacementOutcome, Box<runmat_runtime::RuntimeError>> {
    let runtime = with_context.then(|| {
        let runtime = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(RuntimeServicePorts::default().with_placement(capture.clone()));
        if let Some(policy) = runmat_runtime::context::current_acceleration_policy() {
            runtime.with_acceleration_policy(policy)
        } else {
            runtime
        }
    });
    plan_local_multi_with_provider(
        provider,
        LocalMultiPlacementRequest {
            operation: "fusion.reduction",
            semantic_identity: Some(Digest::sha256(b"checked scalar and controls")),
            observation_contract: TotalCandidateObservationContract::PreparationBodyCleanup,
            prepared_authority: None,
            vm_right: None,
            vm_decline: None,
            source_decline: LocalSourceDecline::None,
            family: ProviderOperationFamily::Fusion,
            cpu: LocalCpuCandidate::host(
                "cpu.shared.fusion.reduction".into(),
                cpu_ns.map_or(
                    runmat_execution::CandidateCostEvidence::Unavailable(
                        runmat_execution::CandidateCostUnavailable::NoEstimate,
                    ),
                    |ns| runmat_execution::CandidateCostEvidence::Estimated(cost(ns, 0)),
                ),
            ),
            cpu_admission,
            publication: LocalPublicationSemantics::Atomic,
            physical,
            compare_profitability: true,
            runtime: runtime.as_ref(),
        },
        PlacementPolicy::default(),
        implementation,
    )
}

#[test]
fn scoped_no_fallback_keeps_resident_work_on_provider_but_preserves_host_only_work() {
    use runmat_runtime::context::{RuntimeAccelerationPolicy, RuntimeProviderPreference};
    let provider = registered_fixture();
    let allocation = provider
        .owner
        .as_ref()
        .unwrap()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .unwrap();
    let resident = Value::GpuTensor(
        allocation
            .handle
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
            .unwrap(),
    );
    let leaves = [&resident];
    for fallback in [false, true] {
        let runtime = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
        .with_acceleration_policy(RuntimeAccelerationPolicy {
            enabled: true,
            provider: RuntimeProviderPreference::Auto,
            automatic_offload: true,
            allow_host_fallback: fallback,
            calibrate: false,
        });
        let _scope = runtime.enter();
        let representation = tensor_representation(&resident, provider.precision()).unwrap();
        // Host cost includes the real 5us synchronization plus 20us+byte
        // transfer prior, not just its 1ns body. Make the provider slower than
        // that entire legal alternative; no-fallback must still retain it.
        let mut route = physical(19, 1_000_000, CandidatePreparationState::Ready);
        route.data_inputs = &leaves;
        route.source_inputs = Some(vec![representation.clone()]);
        route.prepared_inputs = Some(vec![representation]);
        let outcome = plan_physical_set(provider, &Rc::new(Capture::default()), 1, vec![route]);
        let LocalPlacementOutcome::Selected { selected, .. } = outcome else {
            panic!("selected route")
        };
        assert_eq!(selected.kind.is_provider(), !fallback);
        let host_only = plan_physical_set(
            provider,
            &Rc::new(Capture::default()),
            1,
            vec![physical(19, 10_000, CandidatePreparationState::Cold)],
        );
        let LocalPlacementOutcome::Selected { selected, .. } = host_only else {
            panic!("host route")
        };
        assert!(
            !selected.kind.is_provider(),
            "ordinary CPU work is legal under no-fallback"
        );
    }
}

#[test]
fn plain_cast_source_decline_requires_typed_provider_feasibility_rejection() {
    for rejected_code in [
        None,
        Some(ProviderRejectionCode::UnsupportedOperation),
        Some(ProviderRejectionCode::UnsupportedElementType),
        Some(ProviderRejectionCode::UnsupportedStorage),
        Some(ProviderRejectionCode::UnsupportedLayout),
        Some(ProviderRejectionCode::UnsupportedRank),
        Some(ProviderRejectionCode::InvalidShape),
        Some(ProviderRejectionCode::ResourceLimit),
        Some(ProviderRejectionCode::ProviderUnavailable),
        Some(ProviderRejectionCode::UnknownResourceEstimate),
    ] {
        let provider = MultiProvider {
            device: 7,
            owner: None,
            reject: rejected_code.map(|code| (19, code)),
            queries: AtomicUsize::new(0),
        };
        for source_decline in [
            LocalSourceDecline::None,
            LocalSourceDecline::PlainFloatingCast,
        ] {
            let outcome = plan_local_multi_with_provider(
                &provider,
                LocalMultiPlacementRequest {
                    operation: "single",
                    semantic_identity: None,
                    observation_contract: TotalCandidateObservationContract::PreparationBodyCleanup,
                    prepared_authority: None,
                    vm_right: None,
                    vm_decline: None,
                    source_decline,
                    family: ProviderOperationFamily::Elementwise,
                    cpu: LocalCpuCandidate::host(
                        "cpu.unavailable.cast".into(),
                        runmat_execution::CandidateCostEvidence::Estimated(cost(1000, 0)),
                    ),
                    cpu_admission: LocalCpuAdmission::Unavailable,
                    publication: LocalPublicationSemantics::Atomic,
                    physical: vec![physical(19, 10, CandidatePreparationState::Ready)],
                    compare_profitability: true,
                    runtime: None,
                },
                PlacementPolicy::default(),
                LocalProviderAdmission::Available,
            );
            if rejected_code.is_none() {
                assert!(matches!(
                    outcome.unwrap(),
                    LocalPlacementOutcome::Selected {
                        provider_index: Some(0),
                        ..
                    }
                ));
            } else if source_decline == LocalSourceDecline::PlainFloatingCast
                && matches!(
                    &rejected_code,
                    Some(
                        ProviderRejectionCode::UnsupportedOperation
                            | ProviderRejectionCode::UnsupportedElementType
                            | ProviderRejectionCode::UnsupportedStorage
                            | ProviderRejectionCode::UnsupportedLayout
                            | ProviderRejectionCode::UnsupportedRank
                    )
                )
            {
                match outcome.unwrap() {
                    LocalPlacementOutcome::ProviderRejected {
                        code,
                        feedback: None,
                        ..
                    } => assert_eq!(Some(code), rejected_code),
                    other => panic!("expected exact provider rejection, got {other:?}"),
                }
            } else {
                assert_eq!(
                    outcome.unwrap_err().identifier(),
                    Some("RunMat:Placement:NoLegalCandidate")
                );
            }
        }
    }
}

#[test]
fn context_free_overflowing_price_refuses_without_panicking_and_keeps_legal_alternative() {
    let provider = MultiProvider {
        device: 7,
        owner: None,
        reject: None,
        queries: AtomicUsize::new(0),
    };
    for legal_alternative in [false, true] {
        let capture = Rc::new(Capture::default());
        let mut routes = vec![physical(19, u64::MAX, CandidatePreparationState::Ready)];
        if legal_alternative {
            routes.push(physical(20, 100, CandidatePreparationState::Ready));
        }
        let result = try_plan_physical_set_with_context(
            &provider,
            &capture,
            Some(1000),
            routes,
            LocalCpuAdmission::Unavailable,
            LocalProviderAdmission::Available,
            false,
        );
        if legal_alternative {
            assert!(matches!(
                result.unwrap(),
                LocalPlacementOutcome::Selected {
                    provider_index: Some(1),
                    ..
                }
            ));
        } else {
            assert_eq!(
                result
                    .expect_err("unpriceable numeric candidates fail closed")
                    .identifier(),
                Some("RunMat:Placement:NoLegalCandidate")
            );
        }
        assert!(capture.requests.borrow().is_empty());
    }
    // An invalid provider price cannot create a comparison with a valid host
    // or reach a service graph, regardless of session availability.
    for with_context in [false, true] {
        let capture = Rc::new(Capture::default());
        let outcome = try_plan_physical_set_with_context(
            &provider,
            &capture,
            Some(1000),
            vec![physical(19, u64::MAX, CandidatePreparationState::Ready)],
            LocalCpuAdmission::Legacy,
            LocalProviderAdmission::Available,
            with_context,
        )
        .unwrap();
        assert!(matches!(
            outcome,
            LocalPlacementOutcome::Selected {
                provider_index: None,
                decision_kind: runmat_execution::WholeWorkDecisionKind::Uncompared,
                predicted_total_ns: Some(1000),
                ..
            }
        ));
        let requests = capture.requests.borrow();
        assert_eq!(requests.len(), usize::from(with_context));
        if with_context {
            let candidates = &requests[0].graph.nodes[0].candidates;
            assert_eq!(candidates.len(), 1);
            assert!(!candidates[0].descriptor.kind.is_provider());
        }
    }
    // Missing CPU prices use whole-work selection. An invalid provider quote
    // cannot poison that set, even when no observation service/right exists.
    for with_context in [false, true] {
        for host_available in [false, true] {
            let capture = Rc::new(Capture::default());
            let outcome = try_plan_physical_set_with_context(
                &provider,
                &capture,
                None,
                vec![physical(19, u64::MAX, CandidatePreparationState::Ready)],
                if host_available {
                    LocalCpuAdmission::Legacy
                } else {
                    LocalCpuAdmission::Unavailable
                },
                LocalProviderAdmission::Available,
                with_context,
            );
            if host_available {
                assert!(matches!(
                    outcome.unwrap(),
                    LocalPlacementOutcome::Selected {
                        provider_index: None,
                        decision_kind: runmat_execution::WholeWorkDecisionKind::Unmeasured,
                        predicted_total_ns: None,
                        feedback: None,
                        ..
                    }
                ));
            } else {
                assert_eq!(
                    outcome.unwrap_err().identifier(),
                    Some("RunMat:Placement:NoLegalCandidate")
                );
            }
            assert!(capture.requests.borrow().is_empty());
        }
        // A genuinely unavailable quote still enters physical resource
        // admission. It cannot hide the later valid route's original ordinal.
        for cpu_price in [None, Some(1000)] {
            let capture = Rc::new(Capture::default());
            let mut unquoted = physical(20, 1, CandidatePreparationState::Cold);
            unquoted.fallback = runmat_execution::CandidateCostEvidence::Unavailable(
                runmat_execution::CandidateCostUnavailable::NoEstimate,
            );
            let before = provider.queries.load(Ordering::SeqCst);
            let outcome = try_plan_physical_set_with_context(
                &provider,
                &capture,
                cpu_price,
                vec![
                    physical(19, u64::MAX, CandidatePreparationState::Ready),
                    unquoted,
                    physical(21, 100, CandidatePreparationState::Ready),
                ],
                LocalCpuAdmission::Unavailable,
                LocalProviderAdmission::Available,
                with_context,
            )
            .unwrap();
            let LocalPlacementOutcome::Selected {
                provider_index,
                provider_query,
                provider_cost,
                predicted_total_ns,
                decision_kind,
                feedback,
                ..
            } = outcome
            else {
                panic!("valid physical route remains executable");
            };
            assert_eq!(provider_index, Some(2));
            assert_eq!(provider_query.workload.elements, Some(21));
            assert_eq!(
                provider_cost.and_then(|cost| cost.checked_total_ns()),
                Some(100)
            );
            assert_eq!(predicted_total_ns, Some(100));
            assert_eq!(
                decision_kind,
                runmat_execution::WholeWorkDecisionKind::Uncompared
            );
            assert!(feedback.is_none());
            assert_eq!(provider.queries.load(Ordering::SeqCst) - before, 3);
            assert!(capture.requests.borrow().is_empty());
        }
    }
}

#[test]
fn source_unavailability_and_provider_rejection_preserve_exact_host_demand_and_identity() {
    use runmat_execution::{CandidateResourceDemand, ResourceByteBound};
    let mut identities = Vec::new();
    for implementation in [
        LocalProviderAdmission::Available,
        LocalProviderAdmission::ImplementationUnavailable,
    ] {
        for retained in [12, 16] {
            let provider = MultiProvider {
                device: 0,
                owner: None,
                reject: Some((7, ProviderRejectionCode::UnsupportedOperation)),
                queries: AtomicUsize::new(0),
            };
            let capture = Rc::new(Capture {
                graph_observations: true,
                ..Default::default()
            });
            let demand = CandidateResourceDemand {
                cpu_millicores: 1_000,
                retained_bytes: ResourceByteBound::Known(retained),
                scratch_bytes: ResourceByteBound::Known(30),
                queue_slots: 0,
                provider_host_workspace: None,
                provider_input_stages: None,
                peak_live: None,
            };
            let outcome = try_plan_physical_set_with_admission(
                &provider,
                &capture,
                Some(1_000),
                vec![physical(7, 1, CandidatePreparationState::Unknown)],
                LocalCpuAdmission::ExactReadbacks {
                    resources: Box::new(demand),
                    download_bytes: 12,
                    readback_count: 1,
                },
                implementation,
            )
            .unwrap();
            let LocalPlacementOutcome::ProviderRejected {
                feedback: Some(feedback),
                ..
            } = outcome
            else {
                panic!("actual session must return its host rejection observation right");
            };
            assert!(feedback.is_current());
            assert_eq!(
                provider.queries.load(Ordering::SeqCst),
                usize::from(matches!(implementation, LocalProviderAdmission::Available))
            );
            let requests = capture.requests.borrow();
            assert_eq!(requests.len(), 1, "one canonical host-only selection");
            let candidates = &requests[0].graph.nodes[0].candidates;
            assert_eq!(candidates.len(), 1);
            assert_eq!(candidates[0].resources, demand);
            assert_eq!(
                candidates[0].descriptor.kind,
                ExecutionCandidateKind::SharedRuntime
            );
            assert_eq!(
                candidates[0].descriptor.cost.components.download_ns,
                super::super::transfer_prior_ns(12)
            );
            assert_eq!(
                candidates[0].descriptor.cost.components.synchronization_ns,
                5_000
            );
            assert!(!identities.contains(&requests[0].signature.runtime_facts));
            identities.push(requests[0].signature.runtime_facts);
            let selected_signature = requests[0].signature.clone();
            let selected_candidate = candidates[0].descriptor.identity.clone();
            drop(requests);
            feedback.observe(13, true);
            let profile = capture.session.profile_snapshot();
            assert_eq!(profile.feedback.len(), 1);
            assert_eq!(profile.feedback[0].samples, 1);
            assert_eq!(profile.feedback[0].failures, 0);
            assert_eq!(profile.feedback[0].total_elapsed_ns, 13);
            assert_eq!(profile.feedback[0].signature, selected_signature);
            assert_eq!(profile.feedback[0].candidate, selected_candidate);
            assert_eq!(
                profile.feedback[0].kind,
                ExecutionCandidateKind::SharedRuntime
            );
        }
    }
}

#[test]
fn one_service_graph_compares_cpu_and_both_physical_strategies() {
    let provider = MultiProvider {
        reject: None,
        device: 0,
        owner: None,
        queries: AtomicUsize::new(0),
    };
    let capture = Rc::new(Capture::default());
    let selected = plan_physical_set(
        &provider,
        &capture,
        1_000_000,
        vec![
            physical(1, 10_000, CandidatePreparationState::Ready),
            physical(2, 20_000, CandidatePreparationState::Ready),
        ],
    );
    let LocalPlacementOutcome::Selected {
        provider_index,
        provider_query,
        ..
    } = selected
    else {
        panic!("one graph must select a candidate");
    };
    assert_eq!(provider_index, Some(0));
    assert_eq!(provider_query.workload.elements, Some(1));
    assert_eq!(
        capture.requests.borrow()[0].graph.nodes[0].candidates.len(),
        3
    );

    let cpu = plan_physical_set(
        &provider,
        // A separate cost-prior scenario starts a fresh session. The earlier
        // identical signature deliberately retains its cached decision.
        &Rc::new(Capture::default()),
        1_000,
        vec![
            physical(1, 10_000, CandidatePreparationState::Ready),
            physical(2, 20_000, CandidatePreparationState::Ready),
        ],
    );
    assert!(matches!(
        cpu,
        LocalPlacementOutcome::Selected {
            provider_index: None,
            ..
        }
    ));

    let selective = MultiProvider {
        reject: Some((1, ProviderRejectionCode::UnsupportedOperation)),
        device: 0,
        owner: None,
        queries: AtomicUsize::new(0),
    };
    let remaining = plan_physical_set(
        &selective,
        &Rc::new(Capture::default()),
        1_000_000,
        vec![
            physical(1, 10_000, CandidatePreparationState::Ready),
            physical(2, 20_000, CandidatePreparationState::Ready),
        ],
    );
    assert!(matches!(
        remaining,
        LocalPlacementOutcome::Selected {
            provider_index: Some(1),
            ..
        }
    ));
}

#[test]
fn physical_candidate_reorder_keeps_series_but_cold_state_separates_it() {
    let provider = MultiProvider {
        reject: None,
        device: 0,
        owner: None,
        queries: AtomicUsize::new(0),
    };
    let capture = Rc::new(Capture::default());
    let first = plan_physical_set(
        &provider,
        &capture,
        1_000_000,
        vec![
            physical(1, 10_000, CandidatePreparationState::Ready),
            physical(2, 20_000, CandidatePreparationState::Ready),
        ],
    );
    let second = plan_physical_set(
        &provider,
        &capture,
        1_000_000,
        vec![
            physical(2, 20_000, CandidatePreparationState::Ready),
            physical(1, 10_000, CandidatePreparationState::Ready),
        ],
    );
    assert!(matches!(
        first,
        LocalPlacementOutcome::Selected {
            provider_index: Some(0),
            ..
        }
    ));
    assert!(matches!(
        second,
        LocalPlacementOutcome::Selected {
            provider_index: Some(1),
            ..
        }
    ));
    assert_eq!(
        capture.requests.borrow()[0].signature,
        capture.requests.borrow()[1].signature,
    );
    let _ = plan_physical_set(
        &provider,
        &capture,
        1_000_000,
        vec![
            physical(1, 10_000, CandidatePreparationState::Ready),
            physical(2, 20_000, CandidatePreparationState::Cold),
        ],
    );
    assert_ne!(
        capture.requests.borrow()[1].signature,
        capture.requests.borrow()[2].signature,
    );
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn typed_reduction_refuses_floating_cast_as_input_adapter_before_query() {
    use runmat_accelerate_api::{
        reduction::{ReductionStrategy, SelectedReductionWork},
        ProviderPreparationRoute, ProviderResidency, ProviderTypedReductionWorkload,
    };

    let provider = MultiProvider {
        reject: None,
        device: 0,
        owner: None,
        queries: AtomicUsize::new(0),
    };
    let source_value = Value::Num(3.0);
    let leaves = [&source_value];
    let source = tensor_representation(&source_value, provider.precision()).unwrap();
    for (target, bytes) in [(ProviderPrecision::F64, 0), (ProviderPrecision::F32, 8)] {
        let base = physical(1, 10_000, CandidatePreparationState::Ready);
        let mut prepared = source.clone();
        prepared.residency = ProviderResidency::Device;
        prepared.element_type = target.into();
        let selection_identity = base.physical_identity.unwrap();
        let output = base.outputs[0].clone();
        let candidate = LocalPhysicalCandidate {
            authority: None,
            data_inputs: &leaves,
            source_inputs: Some(vec![source.clone()]),
            prepared_inputs: Some(vec![prepared.clone()]),
            input_adaptation: Some(ProviderPreparationRoute::FloatingCast { target }),
            workload: ProviderWorkload {
                typed_reduction: Some(ProviderTypedReductionWorkload {
                    recipe_identity: Digest::sha256(b"checked-recipe-fixture"),
                    selection_identity,
                    strategy: ReductionStrategy::SinglePass,
                    work: SelectedReductionWork {
                        output_requested_bytes: 8,
                        partial_requested_bytes: 0,
                        dispatches: 1,
                        workgroup_bytes: 64,
                        uniform_bytes: 0,
                        producer_scalar_work_units: 1,
                        terminal_scalar_work_units: 0,
                    },
                    inputs: vec![prepared],
                    output,
                    input_preparation: None,
                }),
                ..Default::default()
            },
            required_upload_bytes: Some(bytes),
            ..base
        };
        let valid_first = LocalPhysicalCandidate {
            authority: None,
            data_inputs: &leaves,
            source_inputs: Some(vec![source.clone()]),
            prepared_inputs: Some(vec![source.clone()]),
            required_upload_bytes: Some(8),
            ..physical(2, 20_000, CandidatePreparationState::Ready)
        };
        assert!(try_plan_physical_set(
            &provider,
            &Rc::new(Capture::default()),
            1_000_000,
            vec![valid_first, candidate],
        )
        .is_err());
        assert_eq!(provider.queries.load(Ordering::SeqCst), 0);
    }
}
