//! The exact two-handle provider hook query shared by runtime admission and
//! provider-side physical preflight. Public filter2/imfilter identity and
//! preparation routes remain with their respective domain owners.

use super::imfilter::filter_output_shape;
use crate::context::provider_operation::{
    select_exact_provider_operation, ExactProviderOperation, ExactProviderReadback,
    SelectedProviderOperation,
};
use crate::context::RuntimeContext;
use crate::BuiltinResult;
use runmat_accelerate_api::{
    GpuTensorHandle, ImfilterOptions, ProviderFeasibilityQuery, ProviderImfilterInputBackings,
    ProviderImfilterPreparedKernel, ProviderImfilterWorkload, ProviderLayout,
    ProviderOperationFamily, ProviderOperationIdentity, ProviderRepresentation, ProviderResidency,
    ProviderStorage, ProviderWorkload,
};
use runmat_execution::{Digest, ResourceByteBound};

/// The narrow no-preparation route: both original operands already belong to
/// the output owner. Other public forms require a separate prepared-stage
/// admission and must not borrow this physical estimate.
pub(in crate::builtins::image::filters) fn select_direct(
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
    runtime: Option<&RuntimeContext>,
    semantic_operation: &'static str,
    image: &GpuTensorHandle,
    original_kernel: Option<&GpuTensorHandle>,
    projected_kernel: ProviderRepresentation,
    options: &ImfilterOptions,
    operation_host_scratch: ResourceByteBound,
) -> BuiltinResult<Option<SelectedPreparedFilter>> {
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
        return Ok(None);
    };
    let Some(expected_kernel_requested_bytes) = physical.prepared_kernel_requested_bytes else {
        return Ok(None);
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
    let readbacks = readback.as_ref().map(std::slice::from_ref).unwrap_or(&[]);
    let selected = select_exact_provider_operation(
        runtime,
        ExactProviderOperation {
            provider,
            query: &query,
            semantic_operation,
            variant: "imfilter.prepared_kernel_direct.v1",
            domain_facts,
            host_scratch: operation_host_scratch,
            provider_host_scratch: physical.host_workspace,
            host_cpu_millicores: 1_000,
            already_live_bytes,
            readbacks,
            internal_input_stages: Some(physical.input_stages),
            output_queue_slots: Some(physical.output_queue_slots),
            conditional_host_fallback: None,
        },
    )?;
    Ok(selected.map(|selection| SelectedPreparedFilter {
        query,
        selection,
        expected_kernel_requested_bytes,
    }))
}
