//! Resource shape of admitted physical candidates.
use super::{
    AccelProvider, CandidateResourceDemand, Digest, ExecutionCandidateDescriptor,
    ProviderCostQuery, ResourceBoundUnavailable, ResourceByteBound,
};
use runmat_execution::CandidatePeakLiveDemand;

pub(super) struct Admitted {
    pub(super) exact_resources: Option<CandidateResourceDemand>,
    pub(super) exact_residency: Option<runmat_execution::CandidateOutputResidency>,
    pub(super) original_index: usize,
    pub(super) descriptor: Option<ExecutionCandidateDescriptor>,
    pub(super) identity: String,
    pub(super) kind: runmat_execution::ExecutionCandidateKind,
    pub(super) cost: runmat_execution::CandidateCostEvidence,
    pub(super) cost_query: ProviderCostQuery,
    pub(super) output_bytes: ResourceByteBound,
    pub(super) transient_bytes: Option<u64>,
    pub(super) transactional: bool,
    pub(super) input_resources: Option<crate::placement::TypedReductionInputResources>,
}

impl Admitted {
    pub(super) fn output_size(&self) -> Option<u64> {
        match self.output_bytes {
            ResourceByteBound::Known(bytes) => Some(bytes),
            ResourceByteBound::Unavailable(_) => None,
        }
    }
    pub(super) fn priced(&self) -> &ExecutionCandidateDescriptor {
        self.descriptor
            .as_ref()
            .expect("numeric branch admits only estimated costs")
    }
}

pub(super) fn candidate_name(
    provider: &dyn AccelProvider,
    operation: &str,
    physical_identity: Option<Digest>,
) -> String {
    match physical_identity {
        Some(identity) => format!(
            "provider.{}.{}.{}",
            provider.device_id(),
            operation,
            identity
        ),
        None => format!("provider.{}.{}", provider.device_id(), operation),
    }
}

pub(super) fn candidate_resource(admitted: &Admitted) -> CandidateResourceDemand {
    if let Some(resources) = admitted.exact_resources {
        return resources;
    }
    let final_created = admitted
        .input_resources
        .map_or(ResourceByteBound::Known(0), |resources| {
            resources.final_created_live_bytes
        });
    let scratch = if admitted.cost.estimate().is_none() {
        // The quote was also the legacy carrier for additional scratch
        // (including observation buffers). Absence cannot become zero here;
        // exact body/input-stage facts remain retained independently below.
        ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoScratchBound)
    } else if let Some(typed) = admitted
        .cost_query
        .operation
        .workload
        .typed_reduction
        .as_ref()
    {
        // Transient bytes is total peak (output plus partials); the graph
        // separately charges retained output, so scratch is its extra part.
        match (admitted.transient_bytes, final_created) {
            (Some(transient), ResourceByteBound::Known(created)) => transient
                .checked_add(created)
                .and_then(|peak| peak.checked_sub(admitted.output_size()?))
                .and_then(|extra| {
                    admitted
                        .cost
                        .estimate()
                        .and_then(|cost| cost.scratch_bytes)
                        .unwrap_or(0)
                        .max(typed.work.partial_requested_bytes)
                        .checked_add(created)
                        .map(|cost_extra| ResourceByteBound::Known(cost_extra.max(extra)))
                })
                .unwrap_or(ResourceByteBound::Unavailable(
                    ResourceBoundUnavailable::NoPeakMemoryBound,
                )),
            (None, _) => {
                ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoPeakMemoryBound)
            }
            (_, ResourceByteBound::Unavailable(reason)) => ResourceByteBound::Unavailable(reason),
        }
    } else if admitted.cost_query.operation.workload.sum.is_some()
        && admitted.transient_bytes.is_none()
    {
        ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoPeakMemoryBound)
    } else {
        admitted
            .cost
            .estimate()
            .and_then(|cost| cost.scratch_bytes)
            .map_or(
                ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoScratchBound),
                Into::into,
            )
    };
    // The prepared handles remain owned until the final dispatch completes.
    // With an exact typed peak, publish the complete incremental live set so
    // resource admission cannot merely peak preparation against a kernel that
    // omits its still-live prepared inputs.
    let peak_live = if admitted.input_resources.is_some()
        && admitted
            .cost_query
            .operation
            .workload
            .typed_reduction
            .is_some()
    {
        match scratch {
            ResourceByteBound::Known(extra) => admitted
                .output_size()
                .and_then(|bytes| bytes.checked_add(extra))
                .map(|peak_bytes| CandidatePeakLiveDemand {
                    peak_bytes,
                    already_live_bytes: 0,
                }),
            ResourceByteBound::Unavailable(_) => None,
        }
    } else {
        None
    };
    CandidateResourceDemand {
        cpu_millicores: 0,
        retained_bytes: admitted.output_bytes,
        scratch_bytes: scratch,
        queue_slots: 1,
        provider_host_workspace: admitted
            .input_resources
            .map(|resources| resources.host_workspace),
        provider_input_stages: admitted.input_resources.map(|resources| resources.stages),
        peak_live,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_accelerate_api::{
        reduction::{ReductionStrategy, SelectedReductionWork},
        ProviderCostContext, ProviderElementType, ProviderFeasibilityQuery, ProviderLayout,
        ProviderOperationFamily, ProviderOperationIdentity, ProviderRepresentation,
        ProviderResidency, ProviderStorage, ProviderTypedReductionWorkload, ProviderWorkload,
    };
    use runmat_execution::{
        CandidateCostEvidence, CandidateExecutionLocation, CandidateOutputPresentation,
        CandidateOutputResidency, CandidatePreparationState, EstimateConfidence, EstimateSource,
        ExecutionCandidateKind, ExecutionCostComponents, ExecutionCostEstimate,
        ProviderInputStageDemands,
    };

    fn admitted(transient_bytes: Option<u64>) -> Admitted {
        let output = ProviderRepresentation {
            element_type: ProviderElementType::F64,
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape: vec![2],
            residency: ProviderResidency::Device,
        };
        let cost = ExecutionCostEstimate {
            components: ExecutionCostComponents::default(),
            scratch_bytes: Some(0),
            confidence: EstimateConfidence::High,
            source: EstimateSource::StaticPrior,
        };
        Admitted {
            exact_resources: None,
            exact_residency: None,
            original_index: 0,
            identity: "provider.1.fusion.reduction.fixture".into(),
            kind: ExecutionCandidateKind::ProviderFusion,
            cost: CandidateCostEvidence::Estimated(cost),
            descriptor: Some(ExecutionCandidateDescriptor {
                identity: "provider.1.fusion.reduction.fixture".into(),
                region: None,
                kind: ExecutionCandidateKind::ProviderFusion,
                execution_location: CandidateExecutionLocation::Provider { device_id: 1 },
                preparation: CandidatePreparationState::Ready,
                cost,
                output_residency: CandidateOutputResidency::Provider { device_id: 1 },
                presentation: CandidateOutputPresentation::Direct,
                guards: vec![],
            }),
            cost_query: ProviderCostQuery {
                context: ProviderCostContext::default(),
                operation: ProviderFeasibilityQuery {
                    operation: ProviderOperationIdentity::new("fusion.reduction"),
                    family: ProviderOperationFamily::Fusion,
                    inputs: vec![],
                    outputs: vec![output.clone()],
                    workload: ProviderWorkload {
                        typed_reduction: Some(ProviderTypedReductionWorkload {
                            recipe_identity: Digest::sha256(b"recipe"),
                            selection_identity: Digest::sha256(b"two-pass"),
                            strategy: ReductionStrategy::TwoPass,
                            work: SelectedReductionWork {
                                output_requested_bytes: 16,
                                partial_requested_bytes: 64,
                                dispatches: 2,
                                workgroup_bytes: 128,
                                uniform_bytes: 0,
                                producer_scalar_work_units: 8,
                                terminal_scalar_work_units: 0,
                            },
                            inputs: vec![],
                            output,
                            input_preparation: None,
                        }),
                        ..Default::default()
                    },
                },
                preparation: CandidatePreparationState::Ready,
                required_upload_bytes: 0,
                required_download_bytes: 0,
                downstream_materialization: false,
            },
            output_bytes: ResourceByteBound::Known(16),
            transient_bytes,
            transactional: true,
            input_resources: None,
        }
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn typed_graph_uses_checked_partial_peak_not_zero_fallback() {
        let known = candidate_resource(&admitted(Some(80)));
        assert_eq!(known.retained_bytes, ResourceByteBound::Known(16));
        assert_eq!(known.scratch_bytes, ResourceByteBound::Known(64));
        let unknown = candidate_resource(&admitted(None));
        assert_eq!(
            unknown.scratch_bytes,
            ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoPeakMemoryBound)
        );
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn typed_graph_includes_prepared_inputs_in_final_kernel_peak() {
        let mut selected = admitted(Some(80));
        selected.input_resources = Some(crate::placement::TypedReductionInputResources {
            route: runmat_accelerate_api::ProviderPreparationRoute::TypedReductionInputs {
                identity: Digest::sha256(b"inputs"),
            },
            required_upload_bytes: 32,
            host_workspace: ResourceByteBound::Known(0),
            stages: ProviderInputStageDemands::sequential([]).unwrap(),
            final_created_live_bytes: ResourceByteBound::Known(32),
        });
        let demand = candidate_resource(&selected);
        assert_eq!(demand.retained_bytes, ResourceByteBound::Known(16));
        assert_eq!(demand.scratch_bytes, ResourceByteBound::Known(96));
        assert_eq!(
            demand.peak_live,
            Some(CandidatePeakLiveDemand {
                peak_bytes: 112,
                already_live_bytes: 0,
            })
        );
        selected.transient_bytes = None;
        let unknown = candidate_resource(&selected);
        assert_eq!(
            unknown.scratch_bytes,
            ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoPeakMemoryBound)
        );
        assert_eq!(unknown.peak_live, None);
        selected.transient_bytes = Some(80);
        selected
            .input_resources
            .as_mut()
            .unwrap()
            .final_created_live_bytes = ResourceByteBound::Known(u64::MAX);
        let overflow = candidate_resource(&selected);
        assert_eq!(
            overflow.scratch_bytes,
            ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoPeakMemoryBound)
        );
        assert_eq!(overflow.peak_live, None);
        selected
            .input_resources
            .as_mut()
            .unwrap()
            .final_created_live_bytes =
            ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoPeakMemoryBound);
        let unbounded = candidate_resource(&selected);
        assert_eq!(
            unbounded.scratch_bytes,
            ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoPeakMemoryBound)
        );
        assert_eq!(unbounded.peak_live, None);
    }
}
