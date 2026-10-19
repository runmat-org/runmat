//! Canonical floating-conversion physical proposals between ordered mean calls.
//! The selected source binding is separate from the provider's route query.

use super::*;
use crate::fusion::image_reduction::ImageSelectedMomentConversion;
use runmat_accelerate_api::{
    ProviderFeasibility, ProviderFeasibilityQuery, ProviderFloatingCastPhysicalEvidence,
    ProviderImageMomentConversion, ProviderOperationFamily, ProviderOperationIdentity,
    ProviderPreparationRoute, ProviderRealElementwiseOperation, ProviderRealElementwiseRecipe,
    ProviderRealElementwiseRequest, ProviderResourceEstimate, ProviderWorkload,
};
use runmat_builtins::{DOUBLE_CATALOG_ENTRY, SINGLE_CATALOG_ENTRY};
use runmat_execution::Digest;

#[derive(Clone)]
pub(crate) struct SelectedMomentCastPlan {
    pub(crate) source_input: u64,
    pub(crate) source_output: u64,
    pub(crate) binding: ImageSelectedMomentConversion,
    pub(crate) semantic_identity: Digest,
    pub(crate) source: ProviderRepresentation,
    pub(crate) source_bytes: u64,
    pub(crate) output: ProviderRepresentation,
    pub(crate) output_bytes: u64,
    pub(crate) resources: ProviderResourceEstimate,
    /// Owner-issued physical work for a genuine class-changing cast. A
    /// same-class alias has no submission or fresh allocation. Generic
    /// feasibility alone is not evidence for staged observation or costing.
    pub(crate) physical: Option<ProviderFloatingCastPhysicalEvidence>,
    /// The same native cast, sealed as an executable recipe for finite
    /// request-local host metadata admission. Ordinary unbounded providers
    /// may continue to use their established unary conversion hook.
    pub(crate) real_recipe: Option<ProviderRealElementwiseRecipe>,
}

fn query(
    provider: &'static dyn AccelProvider,
    source: &ProviderRepresentation,
    target: ProviderPrecision,
    binding: &ImageSelectedMomentConversion,
) -> Option<(ProviderRepresentation, ProviderResourceEstimate)> {
    let entry = match target {
        ProviderPrecision::F32 => &SINGLE_CATALOG_ENTRY,
        ProviderPrecision::F64 => &DOUBLE_CATALOG_ENTRY,
    };
    if binding.target() != target
        || !entry
            .bindings
            .iter()
            .any(|variant| entry.binding_identity(variant) == binding.binding())
    {
        return None;
    }
    let output = ProviderRepresentation {
        element_type: match target {
            ProviderPrecision::F32 => ProviderElementType::F32,
            ProviderPrecision::F64 => ProviderElementType::F64,
        },
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape: source.shape.clone(),
        residency: ProviderResidency::Device,
    };
    let request = ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new(binding.binding().builtin.name),
        family: ProviderOperationFamily::Elementwise,
        inputs: vec![source.clone()],
        outputs: vec![output.clone()],
        workload: ProviderWorkload {
            preparation_route: Some(ProviderPreparationRoute::FloatingCast { target }),
            ..ProviderWorkload::default()
        },
    };
    let ProviderFeasibility::Supported { estimate } = provider.query_feasibility(&request) else {
        return None;
    };
    Some((output, estimate))
}

pub(super) fn plan(
    provider: &'static dyn AccelProvider,
    source: &ProviderRepresentation,
    source_bytes: u64,
    call: &ProviderImageMomentConversion,
    binding: &ImageSelectedMomentConversion,
    semantic_identity: Digest,
) -> Option<SelectedMomentCastPlan> {
    let target = moment_precision(&call.output_fact)?;
    if binding.source_input() != call.source_input
        || binding.source_output() != call.source_output
        || binding.target() != target
        || source.storage != ProviderStorage::DenseReal
        || source.layout != ProviderLayout::ColumnMajorContiguous
        || source.residency != ProviderResidency::Device
        || source.checked_byte_len()? != source_bytes
        || call
            .output_fact
            .shape
            .known_dims()?
            .into_iter()
            .collect::<Option<Vec<_>>>()?
            .into_iter()
            .map(|extent| u64::try_from(extent).ok())
            .collect::<Option<Vec<_>>>()?
            != source.shape
    {
        return None;
    }
    let (output, resources) = query(provider, source, target, binding)?;
    let owner_output_bytes = resources.output_bytes?;
    let output_bytes = if output.element_type == source.element_type {
        // WGPU same-class casts register an alias; inherit the authenticated
        // incoming physical extent rather than minting a new allocation.
        if resources.dispatches != Some(0) || owner_output_bytes != source_bytes {
            return None;
        }
        source_bytes
    } else {
        if resources.dispatches? == 0 {
            return None;
        }
        owner_output_bytes
    };
    resources.transient_bytes?;
    let physical = if output.element_type == source.element_type {
        None
    } else {
        provider.floating_cast_physical_evidence(source, target)
    };
    if physical.is_some_and(|stage| {
        let Some(physical_peak) = stage
            .output_requested_bytes
            .checked_add(stage.uniform_buffer_bytes)
        else {
            return true;
        };
        stage.output_requested_bytes != output_bytes
            || stage.dispatches != resources.dispatches.unwrap_or(0)
            || stage.queue_slots == 0
            || resources
                .transient_bytes
                .is_none_or(|bytes| bytes < physical_peak)
    }) {
        return None;
    }
    let real_recipe = ProviderRealElementwiseRecipe::plan(
        provider,
        ProviderRealElementwiseRequest {
            operation: ProviderRealElementwiseOperation::FloatingCast { target },
            source: source.clone(),
            source_buffer_bytes: source_bytes,
        },
    )
    .ok()
    .flatten()
    .filter(|recipe| {
        recipe.output() == &output
            && recipe.physical().output_buffer_bytes == output_bytes
            && recipe.physical().dispatches == resources.dispatches.unwrap_or(0)
    });
    Some(SelectedMomentCastPlan {
        source_input: call.source_input,
        source_output: call.source_output,
        binding: binding.clone(),
        semantic_identity,
        source: source.clone(),
        source_bytes,
        output,
        output_bytes,
        resources,
        physical,
        real_recipe,
    })
}

pub(crate) fn revalidate(
    provider: &'static dyn AccelProvider,
    live: &GpuTensorHandle,
    call: &ProviderImageMomentConversion,
    selected: &SelectedMomentCastPlan,
) -> bool {
    if live.allocation_lease().validate_live().is_err()
        || runmat_accelerate_api::handle_is_logical(live)
        || runmat_accelerate_api::handle_is_transposed(live)
        || runmat_runtime::builtins::common::gpu_helpers::exact_provider_for_handle(live)
            .is_none_or(|owner| !runmat_accelerate_api::same_provider_instance(owner, provider))
    {
        return false;
    }
    let Some(precision) = runmat_accelerate_api::handle_precision(live) else {
        return false;
    };
    let Some(source) = representation(precision, live.shape()) else {
        return false;
    };
    let Some(replanned) = plan(
        provider,
        &source,
        live.physical().requested_bytes(),
        call,
        &selected.binding,
        selected.semantic_identity,
    ) else {
        return false;
    };
    replanned.source_input == selected.source_input
        && replanned.source_output == selected.source_output
        && replanned.source == selected.source
        && replanned.source_bytes == selected.source_bytes
        && replanned.output == selected.output
        && replanned.output_bytes == selected.output_bytes
        && replanned.resources == selected.resources
        && replanned.physical == selected.physical
}
