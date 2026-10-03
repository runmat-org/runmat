//! The exact two-handle provider hook query shared by runtime admission and
//! provider-side physical preflight. Public filter2/imfilter identity and
//! preparation routes remain with their respective domain owners.

use super::filter_output_shape;
use crate::context::provider_operation::{
    select_exact_provider_operation, ExactProviderOperation, ExactProviderReadback,
    SelectedProviderOperation,
};
use crate::context::RuntimeContext;
use crate::BuiltinResult;
use runmat_accelerate_api::{
    GpuTensorHandle, ImfilterOptions, ProviderFeasibilityQuery, ProviderImfilterInputBackings,
    ProviderImfilterWorkload, ProviderLayout, ProviderOperationFamily, ProviderOperationIdentity,
    ProviderRepresentation, ProviderResidency, ProviderStorage, ProviderWorkload,
};
use runmat_execution::{Digest, ResourceByteBound};

/// The narrow no-preparation route: both original operands already belong to
/// the output owner. Other public forms require a separate prepared-stage
/// admission and must not borrow this physical estimate.
pub(super) fn select_direct(
    runtime: Option<&RuntimeContext>,
    semantic_operation: &'static str,
    image: &GpuTensorHandle,
    kernel: &GpuTensorHandle,
    options: &ImfilterOptions,
    original_kernel_readback: bool,
) -> BuiltinResult<Option<SelectedProviderOperation>> {
    let Some(provider) = runmat_accelerate_api::provider_for_handle(image) else {
        return Ok(None);
    };
    let Some(kernel_owner) = runmat_accelerate_api::provider_for_handle(kernel) else {
        return Ok(None);
    };
    if !runmat_accelerate_api::same_provider_instance(provider, kernel_owner) {
        return Ok(None);
    }
    let Some(query) = exact_hook_query(image, kernel, options) else {
        return Ok(None);
    };
    let Some(backings) = query
        .workload
        .imfilter
        .as_ref()
        .and_then(|facts| facts.input_backings.as_ref())
    else {
        return Ok(None);
    };
    let Some(already_live_bytes) = backings.live_unique_requested_bytes() else {
        return Ok(None);
    };
    let Some(physical) = provider.imfilter_physical_evidence(&query) else {
        return Ok(None);
    };
    let domain_facts = Digest::sha256(
        serde_json::to_vec(&(
            semantic_operation,
            options,
            (
                image.allocation_identity(),
                image.view_identity().view(),
                runmat_accelerate_api::handle_provenance(image),
            ),
            (
                kernel.allocation_identity(),
                kernel.view_identity().view(),
                runmat_accelerate_api::handle_provenance(kernel),
            ),
        ))
        .expect("typed imfilter domain facts serialize"),
    );
    let readback = ExactProviderReadback {
        source: kernel,
        prior_created_bytes: ResourceByteBound::Known(0),
    };
    let readbacks = if original_kernel_readback {
        std::slice::from_ref(&readback)
    } else {
        &[]
    };
    select_exact_provider_operation(
        runtime,
        ExactProviderOperation {
            provider,
            query: &query,
            semantic_operation,
            variant: "imfilter.same_owner_direct.v1",
            domain_facts,
            host_scratch: ResourceByteBound::Known(0),
            provider_host_scratch: physical.host_workspace,
            host_cpu_millicores: 1_000,
            already_live_bytes,
            readbacks,
            internal_input_stages: Some(physical.input_stages),
            output_queue_slots: Some(physical.output_queue_slots),
            conditional_host_fallback: None,
        },
    )
}

pub fn exact_hook_query(
    image: &GpuTensorHandle,
    kernel: &GpuTensorHandle,
    options: &ImfilterOptions,
) -> Option<ProviderFeasibilityQuery> {
    let backings = ProviderImfilterInputBackings::from_live_handles([image, kernel]).ok()??;
    let input = backings.live_representations()?.to_vec();
    let output_shape =
        filter_output_shape(image.shape(), kernel.shape(), options.shape, "imfilter")
            .ok()?
            .into_iter()
            .map(|extent| u64::try_from(extent).ok())
            .collect::<Option<Vec<_>>>()?;
    let output = ProviderRepresentation {
        element_type: input[0].element_type,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape: output_shape,
        residency: ProviderResidency::Device,
    };
    Some(ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new("imfilter"),
        family: ProviderOperationFamily::Library,
        inputs: input,
        outputs: vec![output],
        workload: ProviderWorkload {
            imfilter: Some(ProviderImfilterWorkload::from_options(
                options,
                runmat_accelerate_api::handle_is_explicit(image),
                Some(backings),
            )),
            requested_outputs: Some(1),
            ..ProviderWorkload::default()
        },
    })
}
