use super::*;
mod multi;
use runmat_accelerate_api::{
    AccelDownloadFuture, GpuTensorHandle, HostTensorView, ProviderCapabilitySnapshot,
    ProviderCostQuery, ProviderFeasibility, ProviderPrecision, ProviderResourceEstimate,
};
use runmat_execution::{
    CandidateExecutionLocation, CandidateOutputResidency, EstimateConfidence, EstimateSource,
    ExecutionCandidateDescriptor, ExecutionCostComponents, PlacementDecision, PlacementFeedback,
    PlacementInvalidation, PlacementPlanRequest, ProviderResourceSnapshot,
};
use runmat_runtime::context::{RuntimeContext, RuntimePlacementService, RuntimeServicePorts};
use std::{cell::RefCell, rc::Rc};

#[test]
fn host_readback_phase_peak_and_retained_output_are_charged_once_at_exact_quota() {
    use crate::placement::partition::{
        single_candidate_resources_admitted, CandidateResourceFacts,
    };
    use runmat_execution::{
        CandidateOutputPresentation, CandidateResourceDemand, PlacementResourceSnapshot,
        ResourceByteBound,
    };
    for (preparation_peak, body_extra, total) in [(40, 20, 40), (10, 30, 42)] {
        let scratch =
            crate::fusion_exec::checked_extra_host_scratch(12, preparation_peak, body_extra)
                .unwrap();
        assert_eq!(12 + scratch, total);
        let demand = CandidateResourceDemand {
            cpu_millicores: 1_000,
            retained_bytes: ResourceByteBound::Known(12),
            scratch_bytes: ResourceByteBound::Known(scratch),
            queue_slots: 0,
            provider_host_workspace: None,
            provider_input_stages: None,
            peak_live: None,
        };
        for (available, expected) in [(total - 1, false), (total, true)] {
            let snapshot = PlacementResourceSnapshot {
                cpu_millicores_available: 1_000,
                memory_available_bytes: Some(available),
                ..Default::default()
            };
            assert_eq!(
                single_candidate_resources_admitted(
                    CandidateResourceFacts {
                        location: CandidateExecutionLocation::Host,
                        output_residency: CandidateOutputResidency::Host,
                        presentation: &CandidateOutputPresentation::Direct,
                        kind: runmat_execution::ExecutionCandidateKind::SharedRuntime,
                        demand,
                        transactional_results: false,
                    },
                    &snapshot,
                    false
                )
                .is_ok(),
                expected
            );
        }
    }
    assert!(crate::fusion_exec::checked_extra_host_scratch(u64::MAX, 0, 1).is_none());
}

struct Provider {
    output: Option<u64>,
    capacity: u64,
}

impl AccelProvider for Provider {
    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        panic!("planning must not upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        panic!("planning must not download")
    }
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }
    fn device_info(&self) -> String {
        "physical-resource-fixture".into()
    }
    fn query_feasibility(&self, _: &ProviderFeasibilityQuery) -> ProviderFeasibility {
        ProviderFeasibility::supported(ProviderResourceEstimate {
            output_bytes: self.output,
            // Deliberately unspecified: this packet does not reinterpret the
            // old total-working-set transient estimate as extra allocation.
            transient_bytes: None,
            dispatches: Some(1),
        })
    }
    fn capability_snapshot(&self) -> ProviderCapabilitySnapshot {
        let mut snapshot = ProviderCapabilitySnapshot::conservative(self);
        snapshot.concurrency.transactional_results = true;
        snapshot
    }
    fn operation_observation_capability(
        &self,
        _: &ProviderFeasibilityQuery,
    ) -> runmat_accelerate_api::observation::ProviderObservationCapability {
        use runmat_accelerate_api::observation::*;
        ProviderObservationCapability::Supported(ProviderObservationOverhead {
            submissions: 2,
            requested_peak_buffer_bytes: 16,
            clear_commands: 2,
            map_callbacks: 2,
            wgpu_compute_query: None,
        })
    }
    fn estimate_cost(
        &self,
        query: &ProviderCostQuery,
    ) -> Option<runmat_accelerate_api::ProviderCostEstimate> {
        query
            .context
            .observation
            .map(|overhead| runmat_accelerate_api::ProviderCostEstimate {
                cost: cost(1, overhead.requested_peak_buffer_bytes),
            })
    }
    fn placement_resources(&self) -> ProviderResourceSnapshot {
        ProviderResourceSnapshot {
            device_id: self.device_id(),
            capacity_bytes: Some(self.capacity),
            ..Default::default()
        }
    }
}

#[derive(Default)]
struct Capture {
    session: crate::placement::PlacementSession,
    requests: RefCell<Vec<PlacementPlanRequest>>,
    /// Opt into forwarding the actual session's graph observation authority.
    /// Default captures retain the legacy plan-only, non-learning contract.
    graph_observations: bool,
}
impl RuntimePlacementService for Capture {
    fn plan_with_observation_permit(
        &self,
        request: PlacementPlanRequest,
    ) -> Result<runmat_runtime::context::RuntimeGraphPlacementSelection, runmat_runtime::RuntimeError>
    {
        if self.graph_observations {
            self.requests.borrow_mut().push(request.clone());
            RuntimePlacementService::plan_with_observation_permit(&self.session, request)
        } else {
            self.plan(request).map(|decision| {
                runmat_runtime::context::RuntimeGraphPlacementSelection {
                    decision,
                    observation_permit: None,
                }
            })
        }
    }
    fn plan(
        &self,
        request: PlacementPlanRequest,
    ) -> Result<PlacementDecision, runmat_runtime::RuntimeError> {
        self.requests.borrow_mut().push(request.clone());
        RuntimePlacementService::plan(&self.session, request)
    }
    fn observe(&self, feedback: PlacementFeedback) -> Result<(), runmat_runtime::RuntimeError> {
        RuntimePlacementService::observe(&self.session, feedback)
    }
    fn invalidate(&self, invalidation: PlacementInvalidation) {
        RuntimePlacementService::invalidate(&self.session, invalidation);
    }
}

fn cost(ns: u64, scratch_bytes: u64) -> ExecutionCostEstimate {
    ExecutionCostEstimate {
        components: ExecutionCostComponents {
            execution_ns: ns,
            ..Default::default()
        },
        scratch_bytes: Some(scratch_bytes),
        confidence: EstimateConfidence::High,
        source: EstimateSource::StaticPrior,
    }
}

fn plan(
    provider: &Provider,
    capture: &Rc<Capture>,
    automatic: bool,
    scratch: u64,
) -> LocalPlacementOutcome {
    plan_with_program(provider, capture, automatic, scratch, None)
}

fn plan_with_program(
    provider: &Provider,
    capture: &Rc<Capture>,
    automatic: bool,
    scratch: u64,
    program: Option<runmat_accelerate_api::ProviderPreparationProgram<'_>>,
) -> LocalPlacementOutcome {
    plan_with_outputs(provider, capture, automatic, scratch, program, vec![])
}

fn plan_with_outputs(
    provider: &Provider,
    capture: &Rc<Capture>,
    automatic: bool,
    scratch: u64,
    program: Option<runmat_accelerate_api::ProviderPreparationProgram<'_>>,
    outputs: Vec<ProviderRepresentation>,
) -> LocalPlacementOutcome {
    try_plan_with_outputs(provider, capture, automatic, scratch, program, outputs).unwrap()
}

fn try_plan_with_outputs(
    provider: &Provider,
    capture: &Rc<Capture>,
    automatic: bool,
    scratch: u64,
    program: Option<runmat_accelerate_api::ProviderPreparationProgram<'_>>,
    outputs: Vec<ProviderRepresentation>,
) -> Result<LocalPlacementOutcome, Box<runmat_runtime::RuntimeError>> {
    let runtime = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(RuntimeServicePorts::default().with_placement(capture.clone()));
    plan_local(
        provider,
        LocalPlacementRequest {
            compound: None,
            operation: "resource.fixture",
            semantic_identity: None,
            observation_contract: TotalCandidateObservationContract::PreparationBodyCleanup,
            prepared_authority: None,
            source_decline: LocalSourceDecline::None,
            family: ProviderOperationFamily::Elementwise,
            provider_kind: ExecutionCandidateKind::ProviderOperation,
            inputs: &[],
            outputs,
            workload: ProviderWorkload::default(),
            preparation: CandidatePreparationState::Ready,
            preparation_program: program,
            shader_requirements: None,
            cpu: ExecutionCandidateDescriptor {
                identity: "cpu.fixture".into(),
                region: None,
                kind: ExecutionCandidateKind::SharedRuntime,
                execution_location: CandidateExecutionLocation::Host,
                preparation: CandidatePreparationState::Ready,
                cost: cost(1_000_000, 0),
                output_residency: CandidateOutputResidency::Host,
                presentation: runmat_execution::CandidateOutputPresentation::Direct,
                guards: vec![],
            }
            .into(),
            provider_fallback: runmat_execution::CandidateCostEvidence::Estimated(cost(1, scratch)),
            required_upload_bytes: Some(0),
            required_download_bytes: 0,
            downstream_materialization: false,
            compare_profitability: automatic,
            runtime: Some(&runtime),
        },
        PlacementPolicy::default(),
    )
}

fn is_provider(outcome: LocalPlacementOutcome) -> bool {
    let LocalPlacementOutcome::Selected { selected, .. } = outcome else {
        panic!("expected selected candidate")
    };
    selected.kind.is_provider()
}

#[test]
fn full_call_authority_rejects_non_session_revision_variants() {
    let provider = crate::placement::revision::local(
        None,
        Digest::sha256(b"provider-revision"),
        PlacementPolicy::default(),
    );
    assert!(session_revision_only(provider.clone()).is_none());
    let mut session = provider;
    session.authority =
        runmat_execution::PlacementExecutionAuthorityRevision::RuntimeBindingSession(
            Digest::sha256(b"selected-binding-session"),
        );
    assert_eq!(session_revision_only(session.clone()), Some(session));
}

#[test]
fn observation_scratch_participates_in_admission_before_any_execution() {
    use runmat_accelerate_api::ProviderPreparationProgram;
    for capacity in [23, 24] {
        let provider = Provider {
            output: Some(8),
            capacity,
        };
        // Uninstrumented numerical feasibility is independent of optional
        // completion tracking; its 8-byte output fits either resource limit.
        let capture = Rc::new(Capture::default());
        assert!(is_provider(plan(&provider, &capture, true, 0)));
        let observed = Rc::new(Capture::default());
        let outcome = plan_with_program(
            &provider,
            &observed,
            true,
            0,
            Some(ProviderPreparationProgram::Elementwise {
                shader: "fixture, never compiled",
            }),
        );
        assert_eq!(is_provider(outcome), capacity == 24);
        let requests = observed.requests.borrow();
        let candidates = &requests[0].graph.nodes[0].candidates;
        assert_eq!(candidates.len(), if capacity == 24 { 2 } else { 1 });
        if capacity == 24 {
            let demand = candidates[1].resources;
            assert_eq!(demand.retained_bytes, 8.into());
            assert_eq!(demand.scratch_bytes, 16.into());
        }
        assert_eq!(
            requests[0].graph.nodes[0].candidates[0]
                .resources
                .retained_bytes,
            runmat_execution::ResourceByteBound::Unavailable(
                runmat_execution::ResourceBoundUnavailable::NoRetainedValueBound
            )
        );
    }
}

#[test]
fn admitted_sum_does_not_turn_unknown_peak_bytes_into_a_sixteen_byte_bound() {
    use runmat_accelerate_api::ProviderSumWorkload;
    use runmat_types::{NumericClass, NumericDomain, ReductionNaN, SumOutputClass};

    let provider = Provider {
        output: Some(8),
        capacity: 24,
    };
    let capture = Rc::new(Capture::default());
    let runtime = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(RuntimeServicePorts::default().with_placement(capture.clone()));
    let outcome = plan_local(
        &provider,
        LocalPlacementRequest {
            compound: None,
            operation: "sum",
            semantic_identity: None,
            observation_contract: TotalCandidateObservationContract::PreparationBodyCleanup,
            prepared_authority: None,
            source_decline: LocalSourceDecline::None,
            family: ProviderOperationFamily::Reduction,
            provider_kind: ExecutionCandidateKind::ProviderOperation,
            inputs: &[],
            outputs: vec![],
            workload: ProviderWorkload {
                sum: Some(ProviderSumWorkload {
                    invocation: runmat_types::SumInvocation::Sum,
                    axes: vec![0],
                    missing: ReductionNaN::Include,
                    output: SumOutputClass::Double,
                    input_class: NumericClass::Double,
                    input_domain: NumericDomain::Real,
                    accumulation_class: NumericClass::Double,
                    publication_class: NumericClass::Double,
                }),
                ..Default::default()
            },
            preparation: CandidatePreparationState::Cold,
            preparation_program: None,
            shader_requirements: None,
            cpu: ExecutionCandidateDescriptor {
                identity: "cpu.sum".into(),
                region: None,
                kind: ExecutionCandidateKind::SharedRuntime,
                execution_location: CandidateExecutionLocation::Host,
                preparation: CandidatePreparationState::Ready,
                cost: cost(1_000_000, 0),
                output_residency: CandidateOutputResidency::Host,
                presentation: runmat_execution::CandidateOutputPresentation::Direct,
                guards: vec![],
            }
            .into(),
            provider_fallback: runmat_execution::CandidateCostEvidence::Estimated(cost(1, 16)),
            required_upload_bytes: Some(0),
            required_download_bytes: 0,
            downstream_materialization: false,
            compare_profitability: true,
            runtime: Some(&runtime),
        },
        PlacementPolicy::default(),
    )
    .unwrap();
    assert!(
        !is_provider(outcome),
        "unknown peak cannot satisfy finite quota"
    );
    assert_eq!(
        capture.requests.borrow()[0].graph.nodes[0].candidates.len(),
        1
    );
}

#[test]
fn explicit_unobserved_selection_does_not_price_or_start_fences() {
    let provider = Provider {
        output: Some(8),
        capacity: 24,
    };
    let capture = Rc::new(Capture::default());
    let outcome = plan_with_program(
        &provider,
        &capture,
        false,
        0,
        Some(
            runmat_accelerate_api::ProviderPreparationProgram::Elementwise {
                shader: "never compiled",
            },
        ),
    );
    let LocalPlacementOutcome::Selected {
        provider_observation,
        provider_cost,
        ..
    } = outcome
    else {
        panic!("explicit provider selection");
    };
    assert!(!provider_observation);
    assert_eq!(
        provider_cost.expect("fixture physical prior").scratch_bytes,
        Some(0)
    );
    assert!(capture.requests.borrow().is_empty());
}

#[test]
fn physical_output_bytes_reach_budget_even_without_query_output_facts() {
    // A packed or complex physical result is provider-owned. Do not substitute
    // semantic element width or this fixture's F64 default for its actual bytes.
    for bytes in [0, 12, 48, 128] {
        let capture = Rc::new(Capture::default());
        let provider = Provider {
            output: Some(bytes),
            capacity: 64,
        };
        assert_eq!(is_provider(plan(&provider, &capture, true, 0)), bytes <= 64);
        let requests = capture.requests.borrow();
        let candidates = &requests[0].graph.nodes[0].candidates;
        assert_eq!(candidates.len(), if bytes <= 64 { 2 } else { 1 });
        if bytes <= 64 {
            let demand = candidates[1].resources;
            assert_eq!(demand.retained_bytes, bytes.into());
            assert_eq!(demand.scratch_bytes, 0.into());
        }
    }
}

#[test]
fn cpu_output_templates_do_not_prove_backing_allocation_bytes() {
    use runmat_accelerate_api::{
        ProviderElementType, ProviderLayout, ProviderResidency, ProviderStorage,
    };
    let output = ProviderRepresentation {
        element_type: ProviderElementType::F64,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape: vec![2, 3],
        residency: ProviderResidency::Host,
    };
    let provider = Provider {
        output: Some(48),
        capacity: 128,
    };
    for template in [
        output.clone(),
        ProviderRepresentation {
            layout: ProviderLayout::Strided,
            ..output.clone()
        },
        ProviderRepresentation {
            layout: ProviderLayout::Opaque,
            ..output.clone()
        },
        ProviderRepresentation {
            storage: ProviderStorage::Sparse,
            ..output
        },
    ] {
        let capture = Rc::new(Capture::default());
        let _ = plan_with_outputs(&provider, &capture, true, 0, None, vec![template]);
        assert_eq!(
            capture.requests.borrow()[0].graph.nodes[0].candidates[0]
                .resources
                .retained_bytes,
            runmat_execution::ResourceByteBound::Unavailable(
                runmat_execution::ResourceBoundUnavailable::NoRetainedValueBound
            )
        );
    }
}

#[test]
fn overflowing_output_template_is_rejected_before_provider_planning() {
    use runmat_accelerate_api::{
        ProviderElementType, ProviderLayout, ProviderResidency, ProviderStorage,
    };
    let capture = Rc::new(Capture::default());
    let error = match try_plan_with_outputs(
        &Provider {
            output: Some(1),
            capacity: 128,
        },
        &capture,
        true,
        0,
        None,
        vec![ProviderRepresentation {
            element_type: ProviderElementType::F64,
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape: vec![u64::MAX, 2],
            residency: ProviderResidency::Host,
        }],
    ) {
        Err(error) => error,
        Ok(_) => panic!("unrepresentable output cannot enter the graph"),
    };
    assert_eq!(
        error.identifier(),
        Some("RunMat:Placement:ResourceOverflow")
    );
    assert!(capture.requests.borrow().is_empty());
}

#[test]
fn known_output_and_scratch_overflow_fails_closed_without_saturation() {
    let capture = Rc::new(Capture::default());
    let provider = Provider {
        output: Some(u64::MAX),
        capacity: u64::MAX,
    };
    assert!(!is_provider(plan(&provider, &capture, true, 1)));
    assert_eq!(
        capture.requests.borrow()[0].graph.nodes[0].candidates.len(),
        1,
        "overflowing provider demand is filtered before service selection"
    );
    let error = try_plan_with_outputs(&provider, &capture, false, 1, None, vec![])
        .expect_err("a mandatory overflowing route cannot use the host alternative");
    assert_eq!(
        error.identifier(),
        Some("RunMat:Placement:NoLegalCandidate")
    );
}

#[test]
fn unknown_is_not_zero_or_exceeded_capacity_and_explicit_execution_stays_supported() {
    for graph_observations in [false, true] {
        let capture = Rc::new(Capture {
            graph_observations,
            ..Default::default()
        });
        let provider = Provider {
            output: None,
            capacity: u64::MAX,
        };
        let LocalPlacementOutcome::ProviderRejected {
            code: ProviderRejectionCode::UnknownResourceEstimate,
            feedback,
            ..
        } = plan(&provider, &capture, true, 0)
        else {
            panic!("unknown provider output must retain the legal host rejection route")
        };
        assert_eq!(feedback.is_some(), graph_observations);
        if let Some(feedback) = feedback.as_ref() {
            assert_eq!(feedback.candidate, "cpu.fixture");
            assert!(feedback.is_current());
        }
        let requests = capture.requests.borrow();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].graph.nodes[0].candidates.len(), 1);
        assert_eq!(
            requests[0].graph.nodes[0].candidates[0].descriptor.kind,
            ExecutionCandidateKind::SharedRuntime
        );
        drop(requests);
        if let Some(feedback) = feedback {
            feedback.observe(13, true);
        }
        let profile = capture.session.profile_snapshot();
        assert_eq!(profile.feedback.len(), usize::from(graph_observations));
        if graph_observations {
            assert_eq!(profile.feedback[0].samples, 1);
            assert_eq!(profile.feedback[0].failures, 0);
            assert_eq!(profile.feedback[0].total_elapsed_ns, 13);
            assert_eq!(profile.feedback[0].candidate, "cpu.fixture");
            assert_eq!(
                profile.feedback[0].kind,
                ExecutionCandidateKind::SharedRuntime
            );
        }
        let error = try_plan_with_outputs(&provider, &capture, false, 0, None, vec![])
            .expect_err("explicit intent cannot prove unavailable retained bytes");
        assert_eq!(
            error.identifier(),
            Some("RunMat:Placement:InsufficientResourceEvidence")
        );
        assert_eq!(capture.requests.borrow().len(), 1);
        let encoded =
            serde_json::to_string(&ProviderRejectionCode::UnknownResourceEstimate).unwrap();
        assert_eq!(encoded, "\"unknown_resource_estimate\"");
        assert_eq!(
            serde_json::from_str::<ProviderRejectionCode>(&encoded).unwrap(),
            ProviderRejectionCode::UnknownResourceEstimate
        );
    }
}

#[test]
fn changed_physical_output_estimates_do_not_reuse_feedback_identity() {
    let capture = Rc::new(Capture::default());
    for bytes in [12, 16] {
        assert!(is_provider(plan(
            &Provider {
                output: Some(bytes),
                capacity: 64
            },
            &capture,
            true,
            0
        )));
    }
    let requests = capture.requests.borrow();
    assert_ne!(
        requests[0].signature.runtime_facts,
        requests[1].signature.runtime_facts
    );
}
