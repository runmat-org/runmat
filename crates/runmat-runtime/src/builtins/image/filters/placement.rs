//! The exact two-handle provider hook query shared by runtime admission and
//! provider-side physical preflight. Public filter2/imfilter identity and
//! preparation routes remain with their respective domain owners.

use crate::context::provider_operation::{
    select_exact_provider_operation, ExactConditionalProviderUpload, ExactProviderOperation,
    ExactProviderReadback, ExactProviderReadbackPhase, SelectedProviderOperation,
};
use crate::context::RuntimeContext;
use crate::BuiltinResult;
use runmat_accelerate_api::{
    GpuTensorHandle, ImfilterOptions, ProviderElementType, ProviderFeasibilityQuery,
    ProviderLayout, ProviderOperationFamily, ProviderOperationIdentity, ProviderRepresentation,
    ProviderResidency, ProviderStorage, ProviderWorkload,
};
use runmat_execution::{Digest, ResourceByteBound};
mod host_only;
pub(in crate::builtins::image::filters) use host_only::{
    select_host_only_sink, select_host_only_upload,
};
mod resource;
pub(in crate::builtins::image::filters) use resource::{
    conditional_host_bounds, host_scratch_excluding_result,
};
mod query;
pub(in crate::builtins::image::filters) use query::prepared_hook_matches_realized;
pub use query::{exact_hook_query, prepared_hook_query};

/// A genuine Unsupported hook may publish a host value only when the caller
/// has admitted this complete conditional branch before any preparation.
/// Readbacks are original live handles; no owned host payload is serialized.
pub(in crate::builtins::image::filters) struct ConditionalHostRoute<'a> {
    pub(in crate::builtins::image::filters) result_bytes: ResourceByteBound,
    pub(in crate::builtins::image::filters) operation_scratch: ResourceByteBound,
    pub(in crate::builtins::image::filters) readbacks: &'a [&'a GpuTensorHandle],
}

pub(in crate::builtins::image::filters) struct ConditionalProviderRoute<'a> {
    pub(in crate::builtins::image::filters) operation_scratch: ResourceByteBound,
    pub(in crate::builtins::image::filters) readbacks: &'a [&'a GpuTensorHandle],
}

pub(in crate::builtins::image::filters) enum ConditionalFilterRoute<'a> {
    Host(ConditionalHostRoute<'a>),
    Provider(ConditionalProviderRoute<'a>),
}

pub(in crate::builtins::image::filters) struct FilterRouteRequest<'a> {
    pub(in crate::builtins::image::filters) runtime: Option<&'a RuntimeContext>,
    pub(in crate::builtins::image::filters) semantic_operation: &'static str,
    pub(in crate::builtins::image::filters) image: &'a GpuTensorHandle,
    pub(in crate::builtins::image::filters) options: &'a ImfilterOptions,
    pub(in crate::builtins::image::filters) conditional: Option<ConditionalFilterRoute<'a>>,
}

fn exact_conditional_upload(
    runtime: Option<&RuntimeContext>,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    hook_query: &ProviderFeasibilityQuery,
    host_scratch_bytes: ResourceByteBound,
) -> BuiltinResult<Option<ExactConditionalProviderUpload>> {
    let [output] = hook_query.outputs.as_slice() else {
        return Ok(None);
    };
    if !matches!(
        output.element_type,
        ProviderElementType::F32 | ProviderElementType::F64
    ) || output.storage != ProviderStorage::DenseReal
        || output.layout != ProviderLayout::ColumnMajorContiguous
        || output.residency != ProviderResidency::Device
    {
        return Ok(None);
    }
    let Some(transfer_bytes) = output.checked_byte_len() else {
        return Ok(None);
    };
    let mut input = output.clone();
    input.residency = ProviderResidency::Host;
    let upload = ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new("transfer.upload"),
        family: ProviderOperationFamily::Upload,
        inputs: vec![input],
        outputs: vec![output.clone()],
        workload: ProviderWorkload {
            requested_outputs: Some(1),
            ..ProviderWorkload::default()
        },
    };
    let Some(estimate) =
        crate::context::provider_operation::provider_resource_estimate(runtime, provider, &upload)?
    else {
        return Ok(None);
    };
    let Some(physical) = provider.numeric_upload_physical_evidence(&upload) else {
        return Ok(None);
    };
    if estimate.output_bytes != Some(physical.output_requested_bytes)
        || physical.completion_queue_slots == 0
    {
        return Ok(None);
    }
    Ok(Some(ExactConditionalProviderUpload {
        provider_output_requested_bytes: physical.output_requested_bytes,
        transfer_bytes: Some(transfer_bytes),
        host_result_bytes: ResourceByteBound::Known(transfer_bytes),
        host_scratch_bytes,
        provider_scratch_bytes: physical.provider_scratch_bytes,
        queue_slots: physical.completion_queue_slots,
    }))
}

fn add_provider_bytes(bound: ResourceByteBound, retained: u64) -> BuiltinResult<ResourceByteBound> {
    Ok(match bound {
        ResourceByteBound::Known(bytes) => {
            ResourceByteBound::Known(bytes.checked_add(retained).ok_or_else(|| {
                crate::build_runtime_error("prepared kernel and fallback provider bytes overflow")
                    .with_identifier("RunMat:Placement:ResourceOverflow")
                    .build()
            })?)
        }
        ResourceByteBound::Unavailable(reason) => ResourceByteBound::Unavailable(reason),
    })
}

fn missing_physical_route<T>(
    runtime: Option<&RuntimeContext>,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
) -> BuiltinResult<Option<T>> {
    crate::context::provider_operation::unavailable_provider_route(runtime, provider)
}

/// The narrow no-preparation route: both original operands already belong to
/// the output owner. Other public forms require a separate prepared-stage
/// admission and must not borrow this physical estimate.
pub(in crate::builtins::image::filters) fn select_direct(
    request: FilterRouteRequest<'_>,
    kernel: &GpuTensorHandle,
    original_kernel_readback: bool,
) -> BuiltinResult<Option<SelectedProviderOperation>> {
    let FilterRouteRequest {
        runtime,
        semantic_operation,
        image,
        options,
        conditional,
    } = request;
    let (conditional_host, conditional_provider) = match conditional {
        Some(ConditionalFilterRoute::Host(route)) => (Some(route), None),
        Some(ConditionalFilterRoute::Provider(route)) => (None, Some(route)),
        None => (None, None),
    };
    let Some(provider) = runmat_accelerate_api::provider_for_handle(image) else {
        return Ok(None);
    };
    let Some(kernel_owner) = runmat_accelerate_api::provider_for_handle(kernel) else {
        return missing_physical_route(runtime, provider);
    };
    if !runmat_accelerate_api::same_provider_instance(provider, kernel_owner) {
        // The caller's hook still belongs to the authenticated image owner;
        // a foreign kernel does not authorize that owner's physical demand.
        return missing_physical_route(runtime, provider);
    }
    let Some(query) = exact_hook_query(image, kernel, options) else {
        return missing_physical_route(runtime, provider);
    };
    if crate::context::provider_operation::provider_resource_estimate(runtime, provider, &query)?
        .is_none()
    {
        return Ok(None);
    }
    let Some(backings) = query
        .workload
        .imfilter
        .as_ref()
        .and_then(|facts| facts.input_backings.as_ref())
    else {
        return missing_physical_route(runtime, provider);
    };
    let Some(already_live_bytes) = backings.live_unique_requested_bytes() else {
        return missing_physical_route(runtime, provider);
    };
    let Some(physical) = provider.imfilter_physical_evidence(&query) else {
        return missing_physical_route(runtime, provider);
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
        phase: ExactProviderReadbackPhase::Preparation,
    };
    let mut readbacks = Vec::new();
    if original_kernel_readback {
        readbacks.push(readback);
    }
    for source in conditional_host
        .as_ref()
        .into_iter()
        .flat_map(|route| route.readbacks.iter())
        .chain(
            conditional_provider
                .as_ref()
                .into_iter()
                .flat_map(|route| route.readbacks.iter()),
        )
    {
        let source = *source;
        if !readbacks.iter().any(|stage: &ExactProviderReadback<'_>| {
            stage.phase == ExactProviderReadbackPhase::Conditional
                && stage.source.view_identity() == source.view_identity()
        }) {
            readbacks.push(ExactProviderReadback {
                source,
                prior_created_bytes: ResourceByteBound::Known(0),
                phase: ExactProviderReadbackPhase::Conditional,
            });
        }
    }
    let host_scratch = ResourceByteBound::Known(0);
    let conditional_host_scratch = conditional_host
        .as_ref()
        .map_or(ResourceByteBound::Known(0), |route| route.operation_scratch);
    let conditional_upload = conditional_provider
        .as_ref()
        .map(|route| exact_conditional_upload(runtime, provider, &query, route.operation_scratch))
        .transpose()?
        .flatten();
    if conditional_provider.is_some() && conditional_upload.is_none() {
        return missing_physical_route(runtime, provider);
    }
    select_exact_provider_operation(
        runtime,
        ExactProviderOperation {
            provider,
            query: &query,
            semantic_operation,
            variant: "imfilter.same_owner_direct.v1",
            domain_facts,
            calibration_facts: None,
            host_scratch,
            provider_host_scratch: physical.host_workspace,
            host_cpu_millicores: 1_000,
            already_live_bytes,
            readbacks: &readbacks,
            produced_readback: None,
            internal_input_stages: Some(physical.input_stages),
            output_queue_slots: Some(physical.output_queue_slots),
            conditional_host_fallback: conditional_host.map(|route| route.result_bytes),
            conditional_host_scratch,
            conditional_provider_upload: conditional_upload,
            provider_host_mediated: physical.route
                != runmat_accelerate_api::ProviderImfilterPhysicalRoute::Direct,
        },
    )
}

pub(in crate::builtins::image::filters) struct SelectedPreparedFilter {
    pub(in crate::builtins::image::filters) query: ProviderFeasibilityQuery,
    pub(in crate::builtins::image::filters) selection: SelectedProviderOperation,
    pub(in crate::builtins::image::filters) expected_kernel_requested_bytes: u64,
}

pub(in crate::builtins::image::filters) fn select_prepared(
    request: FilterRouteRequest<'_>,
    original_kernel: Option<&GpuTensorHandle>,
    projected_kernel: ProviderRepresentation,
    operation_host_scratch: ResourceByteBound,
) -> BuiltinResult<Option<SelectedPreparedFilter>> {
    let FilterRouteRequest {
        runtime,
        semantic_operation,
        image,
        options,
        conditional,
    } = request;
    let (conditional_host, conditional_provider) = match conditional {
        Some(ConditionalFilterRoute::Host(route)) => (Some(route), None),
        Some(ConditionalFilterRoute::Provider(route)) => (None, Some(route)),
        None => (None, None),
    };
    let Some(provider) = runmat_accelerate_api::provider_for_handle(image) else {
        return Ok(None);
    };
    let Some(query) = prepared_hook_query(image, projected_kernel, options) else {
        return missing_physical_route(runtime, provider);
    };
    if crate::context::provider_operation::provider_resource_estimate(runtime, provider, &query)?
        .is_none()
    {
        return Ok(None);
    }
    let Some(prepared) = query
        .workload
        .imfilter
        .as_ref()
        .and_then(|workload| workload.prepared_kernel.as_ref())
    else {
        return missing_physical_route(runtime, provider);
    };
    let Some(already_live_bytes) = prepared.live_image_requested_bytes() else {
        return missing_physical_route(runtime, provider);
    };
    let Some(physical) = provider.imfilter_physical_evidence(&query) else {
        return missing_physical_route(runtime, provider);
    };
    let Some(expected_kernel_requested_bytes) = physical.prepared_kernel_requested_bytes else {
        return missing_physical_route(runtime, provider);
    };
    let original_facts = original_kernel.map(|source| {
        (
            source.allocation_identity(),
            source.view_identity().view(),
            runmat_accelerate_api::handle_provenance(source),
        )
    });
    let domain_facts = Digest::sha256(
        serde_json::to_vec(&(
            semantic_operation,
            options,
            image.allocation_identity(),
            image.view_identity().view(),
            original_facts,
            &query,
        ))
        .expect("typed prepared filter facts serialize"),
    );
    let readback = original_kernel.map(|source| ExactProviderReadback {
        source,
        prior_created_bytes: ResourceByteBound::Known(0),
        phase: ExactProviderReadbackPhase::Preparation,
    });
    let mut readbacks = readback.into_iter().collect::<Vec<_>>();
    for source in conditional_host
        .as_ref()
        .into_iter()
        .flat_map(|route| route.readbacks.iter())
        .chain(
            conditional_provider
                .as_ref()
                .into_iter()
                .flat_map(|route| route.readbacks.iter()),
        )
    {
        let source = *source;
        if !readbacks.iter().any(|stage| {
            stage.phase == ExactProviderReadbackPhase::Conditional
                && stage.source.view_identity() == source.view_identity()
        }) {
            readbacks.push(ExactProviderReadback {
                source,
                prior_created_bytes: ResourceByteBound::Known(
                    if source.view_identity() == image.view_identity() {
                        expected_kernel_requested_bytes
                    } else {
                        0
                    },
                ),
                phase: ExactProviderReadbackPhase::Conditional,
            });
        }
    }
    let host_scratch = operation_host_scratch;
    let conditional_host_scratch = conditional_host
        .as_ref()
        .map_or(ResourceByteBound::Known(0), |route| route.operation_scratch);
    let mut conditional_upload = conditional_provider
        .as_ref()
        .map(|route| exact_conditional_upload(runtime, provider, &query, route.operation_scratch))
        .transpose()?
        .flatten();
    if conditional_provider.is_some() && conditional_upload.is_none() {
        return missing_physical_route(runtime, provider);
    }
    if let Some(upload) = conditional_upload.as_mut() {
        upload.provider_scratch_bytes = add_provider_bytes(
            upload.provider_scratch_bytes,
            expected_kernel_requested_bytes,
        )?;
    }
    let selected = select_exact_provider_operation(
        runtime,
        ExactProviderOperation {
            provider,
            query: &query,
            semantic_operation,
            variant: "imfilter.prepared_kernel_direct.v1",
            domain_facts,
            calibration_facts: None,
            host_scratch,
            provider_host_scratch: physical.host_workspace,
            host_cpu_millicores: 1_000,
            already_live_bytes,
            readbacks: &readbacks,
            produced_readback: None,
            internal_input_stages: Some(physical.input_stages),
            output_queue_slots: Some(physical.output_queue_slots),
            conditional_host_fallback: conditional_host.map(|route| route.result_bytes),
            conditional_host_scratch,
            conditional_provider_upload: conditional_upload,
            provider_host_mediated: physical.route
                != runmat_accelerate_api::ProviderImfilterPhysicalRoute::Direct,
        },
    )?;
    Ok(selected.map(|selection| SelectedPreparedFilter {
        query,
        selection,
        expected_kernel_requested_bytes,
    }))
}
