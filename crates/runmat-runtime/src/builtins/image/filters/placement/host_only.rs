//! Class-preserving host filter route facts shared by filter identities.
//! The caller owns the public class decision; generic context admission owns
//! resources, selection, whole-work timing and feedback.

use super::super::geometry::filter_output_shape;
use super::conditional_host_bounds;
use crate::context::host_presentation::{
    combine_host_scratch, select_exact_host_sink_with_readbacks,
    select_exact_host_upload_with_readbacks, ExactHostSink, ExactHostUpload,
    HostPresentationReceipt, HostSinkReceipt,
};
use crate::context::RuntimeContext;
use crate::BuiltinResult;
use runmat_accelerate_api::{
    GpuTensorHandle, ImfilterOptions, ProviderElementType, ProviderLayout, ProviderRepresentation,
    ProviderResidency, ProviderStorage,
};
use runmat_execution::{Digest, ResourceBoundUnavailable, ResourceByteBound};

struct HostOnlyFacts {
    output_shape: Vec<usize>,
    output_bytes: ResourceByteBound,
    host_compute_scratch: ResourceByteBound,
    domain_facts: Digest,
}

fn facts(
    builtin: &'static str,
    image: &GpuTensorHandle,
    kernel_shape: &[usize],
    options: &ImfilterOptions,
    host_element_width: u64,
) -> BuiltinResult<HostOnlyFacts> {
    let output_shape = filter_output_shape(image.shape(), kernel_shape, options.shape, builtin)
        .map_err(|message| {
            crate::build_runtime_error(message)
                .with_builtin(builtin)
                .build()
        })?;
    let count = |shape: &[usize]| {
        shape.iter().try_fold(1_u64, |total, &extent| {
            total.checked_mul(u64::try_from(extent).ok()?)
        })
    };
    let output_bytes = count(&output_shape).and_then(|len| len.checked_mul(host_element_width));
    let (operation_scratch, _) = conditional_host_bounds(
        builtin,
        image.shape(),
        kernel_shape,
        options,
        false,
        false,
        false,
    );
    // The typed integer/logical evaluator converts A into f64 arithmetic
    // storage while the downloaded native payload remains live.
    let converted_image = count(image.shape())
        .and_then(|len| len.checked_mul(8))
        .map(ResourceByteBound::Known)
        .unwrap_or(ResourceByteBound::Unavailable(
            ResourceBoundUnavailable::NoScratchBound,
        ));
    let host_compute_scratch = combine_host_scratch(operation_scratch, converted_image)?;
    Ok(HostOnlyFacts {
        output_shape,
        output_bytes: output_bytes.map(ResourceByteBound::Known).unwrap_or(
            ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoRetainedValueBound),
        ),
        host_compute_scratch,
        domain_facts: Digest::sha256(
            serde_json::to_vec(&(
                image.shape(),
                kernel_shape,
                options.mode,
                options.shape,
                options.padding,
                options.constant_value.to_bits(),
                image.physical().element_type(),
                runmat_accelerate_api::handle_is_logical(image),
                host_element_width,
            ))
            .expect("typed host filter facts serialize"),
        ),
    })
}

pub(in crate::builtins::image::filters) fn select_host_only_sink(
    runtime: Option<&RuntimeContext>,
    builtin: &'static str,
    image: &GpuTensorHandle,
    kernel_shape: &[usize],
    options: &ImfilterOptions,
    readbacks: &[GpuTensorHandle],
    host_element_width: u64,
) -> BuiltinResult<Option<HostSinkReceipt>> {
    let facts = facts(builtin, image, kernel_shape, options, host_element_width)?;
    select_exact_host_sink_with_readbacks(
        runtime,
        ExactHostSink {
            operation: builtin,
            variant: "filters.integer_logical.host_native.v1",
            source: image,
            domain_facts: facts.domain_facts,
            output_bytes: facts.output_bytes,
            host_compute_scratch: facts.host_compute_scratch,
        },
        readbacks,
    )
}

pub(in crate::builtins::image::filters) fn select_host_only_upload(
    runtime: Option<&RuntimeContext>,
    builtin: &'static str,
    image: &GpuTensorHandle,
    kernel_shape: &[usize],
    options: &ImfilterOptions,
    readbacks: &[GpuTensorHandle],
    element_type: ProviderElementType,
) -> BuiltinResult<Option<HostPresentationReceipt>> {
    let facts = facts(
        builtin,
        image,
        kernel_shape,
        options,
        element_type.byte_width(),
    )?;
    let shape = facts
        .output_shape
        .into_iter()
        .map(|extent| u64::try_from(extent).ok())
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            crate::build_runtime_error("filter output shape exceeds provider representation")
                .with_identifier("RunMat:Placement:ResourceOverflow")
                .build()
        })?;
    let output = ProviderRepresentation {
        element_type,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape,
        residency: ProviderResidency::Device,
    };
    select_exact_host_upload_with_readbacks(
        runtime,
        ExactHostUpload {
            operation: builtin,
            variant: "filters.integer_logical.host_exact_owner_upload.v1",
            owner: image,
            domain_facts: facts.domain_facts,
            outputs: vec![output],
            host_compute_scratch: facts.host_compute_scratch,
        },
        readbacks,
    )
}
