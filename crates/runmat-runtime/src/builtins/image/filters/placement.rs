//! The exact two-handle provider hook query shared by runtime admission and
//! provider-side physical preflight. Public filter2/imfilter identity and
//! preparation routes remain with their respective domain owners.

use super::imfilter::filter_output_shape;
use crate::context::host_presentation::combine_host_scratch;
use crate::context::provider_operation::{
    select_exact_provider_operation, ExactConditionalProviderUpload, ExactProviderOperation,
    ExactProviderReadback, SelectedProviderOperation,
};
use crate::context::RuntimeContext;
use crate::BuiltinResult;
use runmat_accelerate_api::{
    GpuTensorHandle, ImfilterOptions, ProviderElementType, ProviderFeasibility,
    ProviderFeasibilityQuery, ProviderImfilterInputBackings, ProviderImfilterPreparedKernel,
    ProviderImfilterWorkload, ProviderLayout, ProviderOperationFamily, ProviderOperationIdentity,
    ProviderRepresentation, ProviderResidency, ProviderStorage, ProviderWorkload,
};
use runmat_execution::{Digest, ResourceBoundUnavailable, ResourceByteBound};

/// Checked operation-owned plan/evaluation/conversion workspace shared by
/// filter2 and imfilter's equivalent host continuation. Provider readback
/// payload and staging are accounted separately by the source-owner hooks.
pub(in crate::builtins::image::filters) fn conditional_host_bounds(
    builtin: &str,
    image_shape: &[usize],
    kernel_shape: &[usize],
    options: &ImfilterOptions,
    image_single: bool,
    kernel_single: bool,
    prepared_kernel: bool,
) -> (ResourceByteBound, ResourceByteBound) {
    let counts = (|| {
        let output_shape =
            filter_output_shape(image_shape, kernel_shape, options.shape, builtin).ok()?;
        let count = |shape: &[usize]| {
            shape.iter().try_fold(1_u64, |total, &extent| {
                total.checked_mul(u64::try_from(extent).ok()?)
            })
        };
        Some((
            count(image_shape)?,
            count(kernel_shape)?,
            count(&output_shape)?,
            u64::try_from(image_shape.len().max(kernel_shape.len()).max(2)).ok()?,
        ))
    })();
    let Some((image_len, kernel_len, output_len, rank)) = counts else {
        return (
            ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoScratchBound),
            ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoRetainedValueBound),
        );
    };
    let output_single = image_single || kernel_single;
    let result = (|| {
        let point_width =
            u64::try_from(std::mem::size_of::<super::imfilter::ImfilterKernelPoint>())
                .ok()?
                .checked_add(rank.checked_mul(8)?)?;
        let rank_vectors = rank.checked_mul(16)?.checked_mul(8)?;
        let bytes = [
            kernel_len.checked_mul(point_width)?,
            rank_vectors,
            if image_single {
                image_len.checked_mul(8)?
            } else {
                0
            },
            if kernel_single {
                kernel_len.checked_mul(8)?
            } else {
                0
            },
            output_len.checked_mul(8)?,
            if output_single {
                output_len.checked_mul(12)?
            } else {
                0
            },
            if prepared_kernel {
                kernel_len
                    .checked_mul(if kernel_single { 4 } else { 8 })?
                    .checked_mul(2)?
            } else {
                0
            },
        ]
        .into_iter()
        .try_fold(0_u64, |total, bytes| total.checked_add(bytes))?;
        let output = output_len.checked_mul(if output_single { 4 } else { 8 })?;
        Some((bytes, output))
    })();
    match result {
        Some((scratch, output)) => (
            ResourceByteBound::Known(scratch),
            ResourceByteBound::Known(output),
        ),
        None => (
            ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoScratchBound),
            ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoRetainedValueBound),
        ),
    }
}

/// The bound above includes the final host payload. A conditional provider
/// upload accounts that payload as a separately retained result, so remove
/// exactly its bytes before combining the remaining operation workspace.
pub(in crate::builtins::image::filters) fn host_scratch_excluding_result(
    scratch: ResourceByteBound,
    result: ResourceByteBound,
) -> BuiltinResult<ResourceByteBound> {
    match (scratch, result) {
        (ResourceByteBound::Known(scratch), ResourceByteBound::Known(result)) => scratch
            .checked_sub(result)
            .map(ResourceByteBound::Known)
            .ok_or_else(|| {
                crate::build_runtime_error("conditional host result exceeds its workspace bound")
                    .with_identifier("RunMat:Placement:InvalidConditionalPresentation")
                    .build()
            }),
        (ResourceByteBound::Unavailable(reason), _)
        | (_, ResourceByteBound::Unavailable(reason)) => Ok(ResourceByteBound::Unavailable(reason)),
    }
}

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
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    hook_query: &ProviderFeasibilityQuery,
    host_scratch_bytes: ResourceByteBound,
) -> Option<ExactConditionalProviderUpload> {
    let [output] = hook_query.outputs.as_slice() else {
        return None;
    };
    if !matches!(
        output.element_type,
        ProviderElementType::F32 | ProviderElementType::F64
    ) || output.storage != ProviderStorage::DenseReal
        || output.layout != ProviderLayout::ColumnMajorContiguous
        || output.residency != ProviderResidency::Device
    {
        return None;
    }
    let transfer_bytes = output.checked_byte_len()?;
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
    let ProviderFeasibility::Supported { estimate } = provider.query_feasibility(&upload) else {
        return None;
    };
    let physical = provider.numeric_upload_physical_evidence(&upload)?;
    if estimate.output_bytes != Some(physical.output_requested_bytes)
        || physical.completion_queue_slots == 0
    {
        return None;
    }
    Some(ExactConditionalProviderUpload {
        provider_output_requested_bytes: physical.output_requested_bytes,
        transfer_bytes: Some(transfer_bytes),
        host_result_bytes: ResourceByteBound::Known(transfer_bytes),
        host_scratch_bytes,
        provider_scratch_bytes: physical.provider_scratch_bytes,
        queue_slots: physical.completion_queue_slots,
    })
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
    let Some(runtime) = runtime.filter(|runtime| runtime.service_ports().placement().is_some())
    else {
        return Ok(None);
    };
    let provider_resources = provider.placement_resources();
    let host_limited = runtime
        .service_ports()
        .parallel()
        .is_some_and(|port| port.placement_resources().memory_available_bytes.is_some());
    if host_limited
        || provider_resources.capacity_bytes.is_some()
        || provider_resources.scratch_available_bytes.is_some()
        || provider_resources.queue_limit.is_some()
    {
        return Err(crate::build_runtime_error(
            "imfilter provider route has no proven physical continuation",
        )
        .with_identifier("RunMat:Placement:InsufficientResourceEvidence")
        .build());
    }
    Ok(None)
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
            stage.source.view_identity() == source.view_identity()
        }) {
            readbacks.push(ExactProviderReadback {
                source,
                prior_created_bytes: ResourceByteBound::Known(0),
            });
        }
    }
    let host_scratch = if let Some(route) = conditional_host.as_ref() {
        combine_host_scratch(ResourceByteBound::Known(0), route.operation_scratch)?
    } else {
        ResourceByteBound::Known(0)
    };
    let conditional_upload = conditional_provider
        .as_ref()
        .and_then(|route| exact_conditional_upload(provider, &query, route.operation_scratch));
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
            host_scratch,
            provider_host_scratch: physical.host_workspace,
            host_cpu_millicores: 1_000,
            already_live_bytes,
            readbacks: &readbacks,
            internal_input_stages: Some(physical.input_stages),
            output_queue_slots: Some(physical.output_queue_slots),
            conditional_host_fallback: conditional_host.map(|route| route.result_bytes),
            conditional_provider_upload: conditional_upload,
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

/// A projected H is not yet a GPU handle. This query only proves the exact
/// physical route and may execute only after the realized upload matches it.
pub fn prepared_hook_query(
    image: &GpuTensorHandle,
    projected_kernel: ProviderRepresentation,
    options: &ImfilterOptions,
) -> Option<ProviderFeasibilityQuery> {
    let prepared =
        ProviderImfilterPreparedKernel::from_live_image(image, projected_kernel).ok()??;
    let input = prepared.live_representations()?.to_vec();
    let image_shape = image.shape();
    let kernel_shape = input[1]
        .shape
        .iter()
        .map(|&extent| usize::try_from(extent).ok())
        .collect::<Option<Vec<_>>>()?;
    let output_shape = filter_output_shape(image_shape, &kernel_shape, options.shape, "imfilter")
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
    let mut workload = ProviderImfilterWorkload::from_options(
        options,
        runmat_accelerate_api::handle_is_explicit(image),
        None,
    );
    workload.prepared_kernel = Some(prepared);
    Some(ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new("imfilter"),
        family: ProviderOperationFamily::Library,
        inputs: input,
        outputs: vec![output],
        workload: ProviderWorkload {
            imfilter: Some(workload),
            requested_outputs: Some(1),
            ..ProviderWorkload::default()
        },
    })
}

pub(in crate::builtins::image::filters) fn prepared_hook_matches_realized(
    query: &ProviderFeasibilityQuery,
    image: &GpuTensorHandle,
    kernel: &GpuTensorHandle,
    options: &ImfilterOptions,
    expected_kernel_requested_bytes: u64,
) -> bool {
    query
        .workload
        .imfilter
        .as_ref()
        .filter(|workload| {
            workload.matches_options(options, runmat_accelerate_api::handle_is_explicit(image))
        })
        .and_then(|workload| workload.prepared_kernel.as_ref())
        .is_some_and(|prepared| {
            prepared
                .admits_realized(image, kernel, expected_kernel_requested_bytes)
                .ok()
                == Some(true)
        })
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
        return Ok(None);
    };
    let Some(prepared) = query
        .workload
        .imfilter
        .as_ref()
        .and_then(|workload| workload.prepared_kernel.as_ref())
    else {
        return Ok(None);
    };
    let Some(already_live_bytes) = prepared.live_image_requested_bytes() else {
        return Ok(None);
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
        if !readbacks
            .iter()
            .any(|stage| stage.source.view_identity() == source.view_identity())
        {
            readbacks.push(ExactProviderReadback {
                source,
                prior_created_bytes: ResourceByteBound::Known(
                    if source.view_identity() == image.view_identity() {
                        expected_kernel_requested_bytes
                    } else {
                        0
                    },
                ),
            });
        }
    }
    let host_scratch = if let Some(route) = conditional_host.as_ref() {
        combine_host_scratch(operation_host_scratch, route.operation_scratch)?
    } else {
        operation_host_scratch
    };
    let mut conditional_upload = conditional_provider
        .as_ref()
        .and_then(|route| exact_conditional_upload(provider, &query, route.operation_scratch));
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
            host_scratch,
            provider_host_scratch: physical.host_workspace,
            host_cpu_millicores: 1_000,
            already_live_bytes,
            readbacks: &readbacks,
            internal_input_stages: Some(physical.input_stages),
            output_queue_slots: Some(physical.output_queue_slots),
            conditional_host_fallback: conditional_host.map(|route| route.result_bytes),
            conditional_provider_upload: conditional_upload,
        },
    )?;
    Ok(selected.map(|selection| SelectedPreparedFilter {
        query,
        selection,
        expected_kernel_requested_bytes,
    }))
}
