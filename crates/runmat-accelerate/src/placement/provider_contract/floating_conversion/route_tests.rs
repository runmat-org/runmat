use super::*;
use crate::placement::local::{plan_local, LocalPlacementOutcome, LocalPlacementRequest};
use runmat_accelerate_api::{ProviderElementwiseGeometry, ProviderPreparationProgram};
use runmat_execution::{
    CandidateExecutionLocation, CandidateOutputResidency, CandidatePreparationState,
    EstimateConfidence, EstimateSource, ExecutionCandidateDescriptor, ExecutionCandidateKind,
    ExecutionCostComponents, ExecutionCostEstimate,
};
use runmat_value::{Tensor, Value};

/// Real numerical/resource authority with no executable hooks: any speculative
/// transfer during planning is a test failure, not a simulated GPU execution.
struct PlanningProvider;
impl AccelProvider for PlanningProvider {
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }
    fn elementwise_geometry(&self) -> Option<ProviderElementwiseGeometry> {
        Some(ProviderElementwiseGeometry {
            workgroup_size: 256,
        })
    }
    fn device_info(&self) -> String {
        "route-planning".into()
    }
    fn upload(
        &self,
        _: &runmat_accelerate_api::HostTensorView<'_>,
    ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
        panic!("planning cannot upload")
    }
    fn download<'a>(
        &'a self,
        _: &'a runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        panic!("planning cannot download")
    }
    fn query_feasibility(&self, query: &ProviderFeasibilityQuery) -> ProviderFeasibility {
        validate(self, true, query).unwrap()
    }
}

#[test]
fn unresolved_template_preserves_hook_support_without_alias_resource_or_receipt_claims() {
    use runmat_accelerate_api::observation::ProviderObservationCapability;
    let provider = PlanningProvider;
    let mut request = query(
        SINGLE_CATALOG_ENTRY.identity.name,
        ProviderElementType::F32,
        &[1, 8],
    );
    for residency in [ProviderResidency::Host, ProviderResidency::Device] {
        request.inputs[0].residency = residency;
        request.workload.preparation_route = None;
        assert_eq!(
            validate(&provider, true, &request),
            Some(ProviderFeasibility::supported(
                ProviderResourceEstimate::default()
            ))
        );
        assert!(unresolved_adaptation(&request));
        assert!(work::physical_work(&provider, &request).is_none());
        assert!(matches!(
            work::observation_capability(&provider, &request),
            ProviderObservationCapability::Unsupported
        ));

        request.workload.preparation_route = ProviderPreparationProgram::FloatingCast {
            target: ProviderPrecision::F32,
        }
        .route();
        assert!(!unresolved_adaptation(&request));
        let physical = work::physical_work(&provider, &request).unwrap();
        assert_eq!(physical.estimate.output_bytes, Some(32));
        assert_eq!(physical.estimate.dispatches, Some(0));

        request.workload.preparation_route = ProviderPreparationProgram::FloatingCast {
            target: ProviderPrecision::F64,
        }
        .route();
        assert!(unresolved_adaptation(&request));
        assert!(work::physical_work(&provider, &request).is_none());
    }
}

#[test]
fn exact_plain_double_has_resource_work_without_enabling_completion_observation() {
    use runmat_accelerate_api::observation::ProviderObservationCapability;
    let provider = PlanningProvider;
    let mut request = query(
        DOUBLE_CATALOG_ENTRY.identity.name,
        ProviderElementType::F64,
        &[1, 8],
    );
    assert!(work::physical_work(&provider, &request).is_none());
    request.workload.preparation_route = ProviderPreparationProgram::FloatingCast {
        target: ProviderPrecision::F64,
    }
    .route();
    for residency in [ProviderResidency::Host, ProviderResidency::Device] {
        request.inputs[0].residency = residency;
        let physical = work::physical_work(&provider, &request).unwrap();
        assert_eq!(physical.estimate.output_bytes, Some(64));
        assert_eq!(physical.estimate.dispatches, Some(0));
        assert_eq!(
            physical.allocations,
            u32::from(residency == ProviderResidency::Host)
        );
        assert!(matches!(
            work::observation_capability(&provider, &request),
            ProviderObservationCapability::Unsupported
        ));
    }
}

#[test]
fn placement_derives_route_only_from_admitted_program_not_legacy_workload() {
    let provider = PlanningProvider;
    let input = Value::Tensor(Tensor::from_f32(vec![1.25; 8], vec![1, 8]).unwrap());
    let plain = ProviderPreparationProgram::FloatingCast {
        target: ProviderPrecision::F32,
    };
    let cpu_cost = ExecutionCostEstimate {
        components: ExecutionCostComponents {
            execution_ns: 1_000_000,
            ..Default::default()
        },
        scratch_bytes: Some(0),
        confidence: EstimateConfidence::Prior,
        source: EstimateSource::StaticPrior,
    };
    for program in [None, Some(plain)] {
        let result = plan_local(
            &provider,
            LocalPlacementRequest {
                compound: None,
                operation: SINGLE_CATALOG_ENTRY.identity.name,
                semantic_identity: None,
                observation_contract:
                    crate::placement::TotalCandidateObservationContract::PreparationBodyCleanup,
                prepared_authority: None,
                source_decline: crate::placement::LocalSourceDecline::None,
                family: ProviderOperationFamily::Elementwise,
                provider_kind: ExecutionCandidateKind::ProviderOperation,
                inputs: &[&input],
                outputs: vec![],
                // A legacy request cannot smuggle a plain route through workload.
                workload: ProviderWorkload {
                    preparation_route: plain.route(),
                    ..Default::default()
                },
                preparation: CandidatePreparationState::Ready,
                preparation_program: program,
                shader_requirements: None,
                cpu: ExecutionCandidateDescriptor {
                    identity: "cpu".into(),
                    region: None,
                    kind: ExecutionCandidateKind::SharedRuntime,
                    execution_location: CandidateExecutionLocation::Host,
                    preparation: CandidatePreparationState::Ready,
                    cost: cpu_cost,
                    output_residency: CandidateOutputResidency::Host,
                    presentation: runmat_execution::CandidateOutputPresentation::Direct,
                    guards: vec![],
                }
                .into(),
                provider_fallback: runmat_execution::CandidateCostEvidence::Estimated(cpu_cost),
                required_upload_bytes: None,
                required_download_bytes: 0,
                downstream_materialization: false,
                compare_profitability: true,
                runtime: None,
            },
            crate::placement::PlacementPolicy::default(),
        )
        .unwrap();
        if program.is_none() {
            assert!(matches!(
                result,
                LocalPlacementOutcome::ProviderRejected {
                    code: ProviderRejectionCode::UnknownResourceEstimate,
                    ..
                }
            ));
        } else {
            let LocalPlacementOutcome::Selected { provider_query, .. } = result else {
                panic!("plain cast candidate must remain available")
            };
            assert_eq!(provider_query.workload.preparation_route, plain.route());
        }
    }
}
