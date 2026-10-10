//! Physical candidate-set fixtures, separate from legacy one-record resource tests.
use super::super::multi::{plan_local_multi, LocalMultiPlacementRequest, LocalPhysicalCandidate};
use super::*;
use crate::placement::tensor_representation;
use runmat_accelerate_api::test_support::TestAllocationOwner;
use runmat_accelerate_api::ProviderFeasibility;
use runmat_execution::{CandidateExecutionLocation, CandidateOutputResidency};
use std::sync::atomic::{AtomicUsize, Ordering};
mod checked_recipe;

struct MultiProvider {
    device: u32,
    owner: Option<TestAllocationOwner>,
    reject: Option<u64>,
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
        if self
            .reject
            .is_some_and(|marker| query.workload.elements == Some(marker))
        {
            return ProviderFeasibility::rejected(
                ProviderRejectionCode::UnsupportedOperation,
                "fixture strategy unavailable",
            );
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
        fallback: cost(ns, 0),
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
    let runtime = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(RuntimeServicePorts::default().with_placement(capture.clone()));
    plan_local_multi(
        provider,
        LocalMultiPlacementRequest {
            operation: "fusion.reduction",
            semantic_identity: Some(Digest::sha256(b"checked scalar and controls")),
            observation_contract: TotalCandidateObservationContract::PreparationBodyCleanup,
            prepared_authority: None,
            vm_right: None,
            family: ProviderOperationFamily::Fusion,
            cpu: ExecutionCandidateDescriptor {
                identity: "cpu.shared.fusion.reduction".into(),
                region: None,
                kind: ExecutionCandidateKind::SharedRuntime,
                execution_location: CandidateExecutionLocation::Host,
                preparation: CandidatePreparationState::Ready,
                cost: cost(cpu_ns, 0),
                output_residency: CandidateOutputResidency::Host,
                presentation: runmat_execution::CandidateOutputPresentation::Direct,
                guards: vec![],
            },
            physical,
            compare_profitability: true,
            runtime: Some(&runtime),
        },
        PlacementPolicy::default(),
    )
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
        reject: Some(1),
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
