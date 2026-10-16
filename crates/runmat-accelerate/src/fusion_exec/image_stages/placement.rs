//! One physical candidate for the exact move-owned staged session. A missing
//! whole-program quote remains unpriced; this owner never sums per-hook costs.

use super::*;
use crate::placement::{
    plan_local_multi, tensor_representation, LocalCpuAdmission, LocalCpuCandidate,
    LocalMultiPlacementRequest, LocalPhysicalCandidate, LocalPlacementOutcome, PlacementPolicy,
    PreparedPhysicalAuthority, TotalCandidateObservationContract,
};
use runmat_accelerate_api::{
    ProviderElementType, ProviderLayout, ProviderOperationFamily, ProviderRepresentation,
    ProviderResidency, ProviderStorage, ProviderWorkload,
};
use runmat_execution::{
    CandidateCostEvidence, CandidateCostUnavailable, CandidatePreparationState,
    ExecutionCandidateKind,
};

fn output_representation(precision: ProviderPrecision, shape: [u64; 3]) -> ProviderRepresentation {
    ProviderRepresentation {
        element_type: ProviderElementType::from(precision),
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape: shape.into(),
        residency: ProviderResidency::Device,
    }
}

impl PreparedImageStages {
    /// Placement consumes this prepared session's physical bounds and exact
    /// provider snapshot. A host candidate is deliberately absent: rejecting
    /// an optimization merely resumes the original selected VM statements.
    pub(crate) fn select(
        &self,
        input: &Value,
        runtime: &runmat_runtime::context::RuntimeContext,
        decline: Option<&runmat_runtime::context::VmFusionDeclineRight>,
    ) -> ProviderExecutionResult<LocalPlacementOutcome> {
        let Value::GpuTensor(handle) = input else {
            return Err(ProviderExecutionError::from(anyhow!(
                "selected image stages require their original resident input"
            )));
        };
        if handle.view_identity() != self.input_view_identity()
            || !self.matches_pre_effect_snapshot(self.session.provider)?
        {
            return Err(ProviderExecutionError::from(anyhow!(
                "selected image stage source or provider snapshot changed"
            )));
        }
        let selected = self.session.selected_mean_plans.as_ref().ok_or_else(|| {
            ProviderExecutionError::from(anyhow!("selected image moments have no physical plan"))
        })?;
        let (mean, _) = selected.public_mean_carrier().ok_or_else(|| {
            ProviderExecutionError::from(anyhow!("selected image mean has no public carrier"))
        })?;
        let (sigma, _) = selected.public_sigma_carrier().ok_or_else(|| {
            ProviderExecutionError::from(anyhow!("selected image sigma has no public carrier"))
        })?;
        let output_program = self
            .session
            .program
            .output_program
            .as_ref()
            .ok_or_else(|| {
                ProviderExecutionError::from(anyhow!("selected image output program is absent"))
            })?;
        let mut outputs = vec![mean, sigma];
        outputs.extend(
            output_program
                .calls
                .iter()
                .filter(|call| call.checkpoint)
                .map(|call| output_representation(call.output_precision, call.output_shape)),
        );
        let source =
            tensor_representation(input, self.session.provider.precision()).ok_or_else(|| {
                ProviderExecutionError::from(anyhow!(
                    "selected image input has no physical representation"
                ))
            })?;
        let data_inputs = [input];
        let no_quote = CandidateCostEvidence::Unavailable(CandidateCostUnavailable::NoEstimate);
        plan_local_multi(
            self.session.provider,
            LocalMultiPlacementRequest {
                operation: "fusion.image_normalize",
                semantic_identity: Some(self.semantic_identity()),
                observation_contract: TotalCandidateObservationContract::PreparationBodyCleanup,
                prepared_authority: None,
                vm_right: None,
                vm_decline: decline,
                family: ProviderOperationFamily::Fusion,
                cpu: LocalCpuCandidate::host("cpu.shared.fusion.image_normalize".into(), no_quote),
                cpu_admission: LocalCpuAdmission::Unavailable,
                physical: vec![LocalPhysicalCandidate {
                    authority: Some(PreparedPhysicalAuthority::ImageStages(self)),
                    physical_identity: Some(self.semantic_identity()),
                    provider_kind: ExecutionCandidateKind::ProviderFusion,
                    data_inputs: &data_inputs,
                    source_inputs: Some(vec![source.clone()]),
                    prepared_inputs: Some(vec![source]),
                    input_adaptation: None,
                    outputs,
                    workload: ProviderWorkload {
                        image_stages: Some(self.session.program.clone()),
                        ..Default::default()
                    },
                    preparation: CandidatePreparationState::Cold,
                    preparation_program: None,
                    shader_requirements: None,
                    fallback: no_quote,
                    required_upload_bytes: Some(0),
                    required_download_bytes: 0,
                    downstream_materialization: false,
                }],
                compare_profitability: true,
                runtime: Some(runtime),
            },
            PlacementPolicy::default(),
        )
        .map_err(|error| ProviderExecutionError::from_source(*error))
    }
}
