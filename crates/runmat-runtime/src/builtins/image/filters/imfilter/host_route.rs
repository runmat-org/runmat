//! Host A with a resident double H: one admitted native readback, host
//! evaluation, and class-preserving host publication.

use super::*;
use crate::builtins::image::filters::placement;
use crate::context::host_presentation::{
    combine_host_scratch, select_exact_host_sink_with_readbacks, ExactHostSink,
};
use runmat_execution::{Digest, ResourceBoundUnavailable, ResourceByteBound};

fn count(shape: &[usize]) -> Option<u64> {
    shape.iter().try_fold(1_u64, |total, &extent| {
        total.checked_mul(u64::try_from(extent).ok()?)
    })
}

fn class_code_and_width(class: ImageOutputClass) -> (u8, u64) {
    match class {
        ImageOutputClass::Double => (0, 8),
        ImageOutputClass::Single => (1, 4),
        ImageOutputClass::Logical => (2, 1),
        ImageOutputClass::Integer(dtype) => {
            let code = match dtype {
                NumericDType::I8 => 3,
                NumericDType::I16 => 4,
                NumericDType::I32 => 5,
                NumericDType::U8 => 6,
                NumericDType::U16 => 7,
                NumericDType::U32 => 8,
                _ => unreachable!("unsupported host A class rejected before route admission"),
            };
            (
                code,
                u64::try_from(dtype.byte_size()).expect("numeric element width fits u64"),
            )
        }
    }
}

pub(super) async fn run(
    runtime: Option<&crate::context::RuntimeContext>,
    image: Value,
    kernel: GpuTensorHandle,
    options: ImfilterOptions,
) -> BuiltinResult<Value> {
    let class = ImageOutputClass::from_host_value(&image)?;
    let image_shape = imfilter_value_shape(&image).expect("admitted numeric host A has a shape");
    validate_public_kernel_shape(kernel.shape())?;
    if runmat_accelerate_api::handle_storage(&kernel) != GpuTensorStorage::Real
        || runmat_accelerate_api::handle_is_logical(&kernel)
        || runmat_accelerate_api::handle_integer_type(&kernel).is_some()
        || runmat_accelerate_api::handle_precision(&kernel) != Some(ProviderPrecision::F64)
    {
        return Err(imfilter_error_with_detail(
            &IMFILTER_ERROR_INVALID_INPUT,
            "H must be a real double array",
        ));
    }
    if !gpu_helpers::gpu_class_metadata_matches(&kernel, Some(ProviderPrecision::F64), None, false)
    {
        return Err(imfilter_error_with_detail(
            &IMFILTER_ERROR_INVALID_INPUT,
            "H GPU class metadata contradicts double storage",
        ));
    }
    let output_shape = filter_output_shape(
        &image_shape,
        kernel.shape(),
        options.shape,
        IMFILTER_BUILTIN,
    )
    .map_err(|error| imfilter_map_error(error.into(), &IMFILTER_ERROR_INVALID_INPUT))?;
    let (class_code, width) = class_code_and_width(class);
    let output_bytes = count(&output_shape).and_then(|len| len.checked_mul(width));
    let (scratch, _) = placement::conditional_host_bounds(
        IMFILTER_BUILTIN,
        &image_shape,
        kernel.shape(),
        &options,
        false,
        false,
        false,
    );
    let image_materialization = count(&image_shape)
        .and_then(|len| len.checked_mul(8))
        .map(ResourceByteBound::Known)
        .unwrap_or(ResourceByteBound::Unavailable(
            ResourceBoundUnavailable::NoScratchBound,
        ));
    let host_compute_scratch = combine_host_scratch(scratch, image_materialization)?;
    let domain_facts = Digest::sha256(
        serde_json::to_vec(&(
            &image_shape,
            kernel.shape(),
            options.mode,
            options.shape,
            options.padding,
            options.constant_value.to_bits(),
            class_code,
        ))
        .expect("typed imfilter host route facts serialize"),
    );
    let sources = [kernel.clone()];
    let receipt = select_exact_host_sink_with_readbacks(
        runtime,
        ExactHostSink {
            operation: IMFILTER_BUILTIN,
            variant: "imfilter.host_image_resident_double_kernel.v1",
            source: &kernel,
            domain_facts,
            output_bytes: output_bytes.map(ResourceByteBound::Known).unwrap_or(
                ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoRetainedValueBound),
            ),
            host_compute_scratch,
        },
        &sources,
    )?;
    let result = match download_kernel_tensor(&kernel).await {
        Ok(kernel_tensor) => imfilter_host_value(image, kernel_tensor, options),
        Err(error) => Err(error),
    };
    if let Some(receipt) = receipt {
        receipt.finish(result.is_ok());
    }
    result
}
