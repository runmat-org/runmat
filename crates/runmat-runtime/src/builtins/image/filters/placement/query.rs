//! Exact two-handle and projected-H imfilter query representations shared by
//! filter2 and imfilter, with realized-upload equality verification.

use super::super::geometry::filter_output_shape;
use runmat_accelerate_api::{
    GpuTensorHandle, ImfilterOptions, ProviderFeasibilityQuery, ProviderImfilterInputBackings,
    ProviderImfilterPreparedKernel, ProviderImfilterWorkload, ProviderLayout,
    ProviderOperationFamily, ProviderOperationIdentity, ProviderRepresentation, ProviderResidency,
    ProviderStorage, ProviderWorkload,
};

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

/// A projected H is not yet a GPU handle. This query only proves its exact
/// physical route and must be checked against the realized upload.
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
