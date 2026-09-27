use super::*;
use runmat_accelerate_api::{
    AccelDownloadFuture, GpuTensorHandle, HostTensorView, ProviderCapabilitySnapshot,
    ProviderPrecision, ProviderResourceEstimate,
};
use runmat_execution::{
    EstimateConfidence, EstimateSource, ExecutionCostComponents, PlacementDecision,
    PlacementFeedback, PlacementInvalidation, ProviderResourceSnapshot,
};
use runmat_runtime::context::{RuntimeContext, RuntimePlacementService, RuntimeServicePorts};
use std::{cell::RefCell, rc::Rc};

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
    fn query_feasibility(&self, query: &ProviderFeasibilityQuery) -> ProviderFeasibility {
        assert!(
            query.outputs.is_empty(),
            "direct calls infer outputs at the provider"
        );
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
}
impl RuntimePlacementService for Capture {
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
        scratch_bytes,
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
    let runtime = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(RuntimeServicePorts::default().with_placement(capture.clone()));
    plan_local(
        provider,
        LocalPlacementRequest {
            operation: "resource.fixture",
            semantic_identity: None,
            family: ProviderOperationFamily::Elementwise,
            provider_kind: ExecutionCandidateKind::ProviderOperation,
            inputs: &[],
            outputs: vec![],
            workload: ProviderWorkload::default(),
            preparation: CandidatePreparationState::Ready,
            preparation_program: program,
            cpu: ExecutionCandidateDescriptor {
                identity: "cpu.fixture".into(),
                region: None,
                kind: ExecutionCandidateKind::SharedRuntime,
                execution_location: CandidateExecutionLocation::Host,
                preparation: CandidatePreparationState::Ready,
                cost: cost(1_000_000, 0),
                output_residency: CandidateOutputResidency::Host,
                guards: vec![],
            },
            provider_fallback: cost(1, scratch),
            required_upload_bytes: Some(0),
            required_download_bytes: 0,
            downstream_materialization: false,
            compare_profitability: automatic,
            runtime: Some(&runtime),
        },
        PlacementPolicy::default(),
    )
    .unwrap()
}

fn is_provider(outcome: LocalPlacementOutcome) -> bool {
    let LocalPlacementOutcome::Selected { selected, .. } = outcome else {
        panic!("expected selected candidate")
    };
    selected.kind.is_provider()
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
        let demand = requests[0].graph.nodes[0].candidates[1].resources;
        assert_eq!(demand.retained_bytes, 8);
        assert_eq!(demand.scratch_bytes, 16);
    }
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
    assert_eq!(provider_cost.scratch_bytes, 0);
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
        let demand = requests[0].graph.nodes[0].candidates[1].resources;
        assert_eq!(demand.retained_bytes, bytes);
        assert_eq!(demand.scratch_bytes, 0);
    }
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
        capture.requests.borrow()[0].graph.nodes[0].candidates[1]
            .resources
            .retained_bytes,
        u64::MAX
    );
}

#[test]
fn unknown_is_not_zero_or_exceeded_capacity_and_explicit_execution_stays_supported() {
    let capture = Rc::new(Capture::default());
    let provider = Provider {
        output: None,
        capacity: u64::MAX,
    };
    assert!(matches!(
        plan(&provider, &capture, true, 0),
        LocalPlacementOutcome::ProviderRejected {
            code: ProviderRejectionCode::UnknownResourceEstimate,
            ..
        }
    ));
    assert!(capture.requests.borrow().is_empty());
    assert!(is_provider(plan(&provider, &capture, false, 0)));
    assert!(capture.requests.borrow().is_empty());
    let encoded = serde_json::to_string(&ProviderRejectionCode::UnknownResourceEstimate).unwrap();
    assert_eq!(encoded, "\"unknown_resource_estimate\"");
    assert_eq!(
        serde_json::from_str::<ProviderRejectionCode>(&encoded).unwrap(),
        ProviderRejectionCode::UnknownResourceEstimate
    );
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
