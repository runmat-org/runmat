//! Resident, exact-class physical projection of one selected checked recipe.
//! Missing preparation/adaptation facts are not silently priced as zero work.
use super::*;
#[cfg(feature = "wgpu")]
use runmat_accelerate_api::observation::{
    ProviderObservationCapability, ProviderObservationOverhead,
};
use runmat_accelerate_api::reduction::ReductionStrategy;
use runmat_accelerate_api::ProviderPreparationRoute;

pub(crate) fn validate(
    wgpu: bool,
    query: &ProviderFeasibilityQuery,
) -> Option<ProviderFeasibility> {
    analyze(wgpu, query).map(|result| match result {
        Some(work) => ProviderFeasibility::supported(work),
        None => ProviderFeasibility::rejected(
            ProviderRejectionCode::UnsupportedOperation,
            "fusion.reduction.typed_physical_route_unavailable",
        ),
    })
}

#[cfg(feature = "wgpu")]
pub(crate) fn physical_work(query: &ProviderFeasibilityQuery) -> Option<ProviderResourceEstimate> {
    analyze(true, query)?
}

#[cfg(feature = "wgpu")]
pub(crate) fn observation_capability(
    query: &ProviderFeasibilityQuery,
) -> ProviderObservationCapability {
    let Some(work) = physical_work(query) else {
        return ProviderObservationCapability::Unsupported;
    };
    let Some(submissions) = work.dispatches else {
        return ProviderObservationCapability::Unsupported;
    };
    ProviderObservationCapability::Supported(ProviderObservationOverhead {
        submissions,
        requested_peak_buffer_bytes: u64::from(submissions)
            * crate::backend::wgpu::dispatch::observation::FENCE_REQUESTED_BYTES,
        clear_commands: submissions,
        map_callbacks: submissions,
        wgpu_compute_query: crate::backend::wgpu::dispatch::observation::prospective_encoder_query(
            submissions,
            submissions,
        ),
    })
}

fn analyze(
    wgpu: bool,
    query: &ProviderFeasibilityQuery,
) -> Option<Option<ProviderResourceEstimate>> {
    if query.operation.0 != fusion_operation_token(&FusionKind::Reduction)
        || query.workload.typed_reduction.is_none()
    {
        return None;
    }
    let selected = query.workload.typed_reduction.as_ref()?;
    let input_preparation = match (
        selected.input_preparation.as_ref(),
        query.workload.preparation_route,
    ) {
        (None, None) => true,
        (Some(inventory), Some(ProviderPreparationRoute::TypedReductionInputs { identity })) => {
            identity == inventory.identity()
                && inventory.recipe_identity() == selected.recipe_identity
                && inventory
                    .entries()
                    .iter()
                    .map(|entry| &entry.prepared)
                    .eq(selected.inputs.iter())
        }
        _ => false,
    };
    if !wgpu
        || query.family != ProviderOperationFamily::Fusion
        // Only the checked ordered typed inventory can describe adaptation.
        // A generic floating-cast label cannot account for this multi-input route.
        || !input_preparation
        || query.outputs.len() != 1
        || query.inputs != selected.inputs
        || selected.work.checked_total_scalar_work_units().is_none()
        || query.workload.scalar_work_units != selected.work.checked_total_scalar_work_units()
        || query.outputs[0] != selected.output
        || selected.output.residency != ProviderResidency::Device
        || selected.output.storage != ProviderStorage::DenseReal
        || selected.output.layout != ProviderLayout::ColumnMajorContiguous
        || query.outputs[0].checked_byte_len() != Some(selected.work.output_requested_bytes)
        || selected.inputs.iter().any(|input| {
            input.residency != ProviderResidency::Device
                || input.storage != ProviderStorage::DenseReal
                || !match input.layout {
                    ProviderLayout::ColumnMajorContiguous => true,
                    // The checked recipe can borrow only the API's rank-two
                    // transpose view. Do not turn the serialized Strided tag
                    // into admission for arbitrary-rank strided storage.
                    ProviderLayout::Strided => input.shape.len() == 2,
                    ProviderLayout::Opaque => false,
                }
                || !matches!(
                    input.element_type,
                    ProviderElementType::F32 | ProviderElementType::F64
                )
        })
        || selected.work.dispatches
            != match selected.strategy {
                ReductionStrategy::SinglePass => 1,
                ReductionStrategy::TwoPass => 2,
            }
        || selected.work.output_requested_bytes == 0
        || (selected.strategy == ReductionStrategy::SinglePass
            && selected.work.partial_requested_bytes != 0)
        || (selected.strategy == ReductionStrategy::TwoPass
            && selected.work.partial_requested_bytes == 0)
    {
        return Some(None);
    }
    let Some(transient_bytes) = selected
        .work
        .output_requested_bytes
        .checked_add(selected.work.partial_requested_bytes)
    else {
        return Some(None);
    };
    Some(Some(ProviderResourceEstimate {
        transient_bytes: Some(transient_bytes),
        output_bytes: Some(selected.work.output_requested_bytes),
        dispatches: Some(selected.work.dispatches),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_accelerate_api::{
        reduction::{ReductionStrategy, SelectedReductionWork},
        ProviderTypedReductionWorkload, ProviderWorkload,
    };
    use runmat_execution::Digest;

    #[test]
    fn selected_two_pass_resource_projection_is_exact_and_rejects_wrong_class() {
        let input = |element_type| ProviderRepresentation {
            element_type,
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape: vec![4, 2],
            residency: ProviderResidency::Device,
        };
        let mut query = ProviderFeasibilityQuery {
            operation: ProviderOperationIdentity::new(fusion_operation_token(
                &FusionKind::Reduction,
            )),
            family: ProviderOperationFamily::Fusion,
            inputs: vec![
                input(ProviderElementType::F32),
                input(ProviderElementType::F64),
            ],
            outputs: vec![ProviderRepresentation {
                shape: vec![1, 2],
                ..input(ProviderElementType::F64)
            }],
            workload: ProviderWorkload {
                scalar_work_units: Some(14),
                typed_reduction: Some(ProviderTypedReductionWorkload {
                    recipe_identity: Digest::sha256(b"recipe"),
                    selection_identity: Digest::sha256(b"selected"),
                    strategy: ReductionStrategy::TwoPass,
                    work: SelectedReductionWork {
                        output_requested_bytes: 16,
                        partial_requested_bytes: 64,
                        dispatches: 2,
                        workgroup_bytes: 128,
                        uniform_bytes: 0,
                        producer_scalar_work_units: 8,
                        terminal_scalar_work_units: 6,
                    },
                    inputs: vec![
                        input(ProviderElementType::F32),
                        input(ProviderElementType::F64),
                    ],
                    output: ProviderRepresentation {
                        shape: vec![1, 2],
                        ..input(ProviderElementType::F64)
                    },
                    input_preparation: None,
                }),
                ..Default::default()
            },
        };
        assert!(matches!(
            validate(true, &query),
            Some(ProviderFeasibility::Supported { estimate })
                if estimate.output_bytes == Some(16)
                    && estimate.transient_bytes == Some(80)
                    && estimate.dispatches == Some(2)
        ));
        query.workload.scalar_work_units = Some(8);
        assert!(matches!(
            validate(true, &query),
            Some(ProviderFeasibility::Rejected { .. })
        ));
        query.workload.scalar_work_units = Some(14);
        #[cfg(feature = "wgpu")]
        {
            let ProviderObservationCapability::Supported(overhead) = observation_capability(&query)
            else {
                panic!("audited two-pass reduction");
            };
            assert_eq!(overhead.wgpu_compute_query.unwrap().max_query_sets, 2);
        }
        query.workload.preparation_route = Some(
            runmat_accelerate_api::ProviderPreparationRoute::FloatingCast {
                target: runmat_accelerate_api::ProviderPrecision::F64,
            },
        );
        assert!(matches!(
            validate(true, &query),
            Some(ProviderFeasibility::Rejected { .. })
        ));
        #[cfg(feature = "wgpu")]
        {
            assert!(physical_work(&query).is_none());
            assert!(matches!(
                observation_capability(&query),
                ProviderObservationCapability::Unsupported
            ));
        }
        query.workload.preparation_route = None;
        assert!(matches!(
            validate(true, &query),
            Some(ProviderFeasibility::Supported { .. })
        ));
        query.outputs[0].element_type = ProviderElementType::F32;
        assert!(matches!(
            validate(true, &query),
            Some(ProviderFeasibility::Rejected { .. })
        ));
        query.outputs[0].element_type = ProviderElementType::F64;
        query.outputs[0].shape = vec![2, 1];
        assert!(matches!(
            validate(true, &query),
            Some(ProviderFeasibility::Rejected { .. })
        ));
        query.outputs[0].shape = vec![1, 2];
        query.inputs[0].shape = vec![2, 4];
        assert!(matches!(
            validate(true, &query),
            Some(ProviderFeasibility::Rejected { .. })
        ));
        query.inputs[0].shape = vec![4, 2];
        // A strided source cannot claim a contiguous prepared binding. This is
        // also the class-changing transpose case until a priced preparation
        // route supplies a real contiguous destination.
        query.inputs[0].layout = ProviderLayout::Strided;
        assert!(matches!(
            validate(true, &query),
            Some(ProviderFeasibility::Rejected { .. })
        ));
        query.workload.typed_reduction.as_mut().unwrap().inputs[0].layout = ProviderLayout::Strided;
        // Exact agreement with the checked selected projection admits the
        // provider's same-class borrowed transpose route without transfer.
        assert!(matches!(
            validate(true, &query),
            Some(ProviderFeasibility::Supported { .. })
        ));
        query.inputs[0].shape.push(1);
        query.workload.typed_reduction.as_mut().unwrap().inputs[0]
            .shape
            .push(1);
        assert!(matches!(
            validate(true, &query),
            Some(ProviderFeasibility::Rejected { .. })
        ));
        query.inputs[0].shape.pop();
        query.workload.typed_reduction.as_mut().unwrap().inputs[0]
            .shape
            .pop();
        query.inputs[0].layout = ProviderLayout::Opaque;
        query.workload.typed_reduction.as_mut().unwrap().inputs[0].layout = ProviderLayout::Opaque;
        assert!(matches!(
            validate(true, &query),
            Some(ProviderFeasibility::Rejected { .. })
        ));
        query.inputs[0].layout = ProviderLayout::ColumnMajorContiguous;
        query.workload.typed_reduction.as_mut().unwrap().inputs[0].layout =
            ProviderLayout::ColumnMajorContiguous;
        assert!(matches!(
            validate(false, &query),
            Some(ProviderFeasibility::Rejected { .. })
        ));
    }
}
