//! Typed owner-upload admission shared by the two resident RGB conversions.
//! The receipt begins before Sink(Gather) preparation and finishes only after
//! exact-owner publication and its provider completion fence.

use super::{common::ColorLayout, rgb_input::ResidentRgbInput};
use crate::context::host_presentation::{
    select_exact_host_upload, ExactHostUpload, HostPresentationReceipt,
};
use crate::context::RuntimeContext;
use crate::BuiltinResult;
use runmat_accelerate_api::observation::ProviderObservationFailure;
use runmat_accelerate_api::{
    handle_integer_type, handle_is_logical, handle_precision, handle_storage, provider_for_handle,
    same_provider_instance, GpuTensorHandle, GpuTensorStorage, IntegerElementType,
    ProviderElementType, ProviderLayout, ProviderPrecision, ProviderRepresentation,
    ProviderResidency, ProviderStorage,
};
use runmat_builtins::RgbConversion;
use runmat_execution::{Digest, ResourceBoundUnavailable, ResourceByteBound};
use runmat_value::{NumericDType, Value};

fn variant(operation: RgbConversion) -> &'static str {
    match operation {
        RgbConversion::Gray => "rgb2gray.host_gather_convert_restore.v1",
        RgbConversion::Hsv => "rgb2hsv.host_gather_convert_restore.v1",
    }
}

pub(crate) struct SelectedRgbPresentation {
    owner: GpuTensorHandle,
    shape: Vec<usize>,
    element_type: ProviderElementType,
    receipt: HostPresentationReceipt,
}

pub(crate) fn select(
    runtime: Option<&RuntimeContext>,
    name: &'static str,
    owner: &GpuTensorHandle,
    input: ResidentRgbInput,
    operation: RgbConversion,
) -> BuiltinResult<Option<SelectedRgbPresentation>> {
    let (shape, pixels) = match input.layout {
        ColorLayout::Truecolor { rows, cols } => {
            let pixels = u64::try_from(rows)
                .ok()
                .and_then(|rows| rows.checked_mul(u64::try_from(cols).ok()?));
            let shape = match operation {
                RgbConversion::Gray => vec![rows, cols],
                RgbConversion::Hsv => vec![rows, cols, 3],
            };
            (shape, pixels)
        }
        ColorLayout::Colormap { rows } => (vec![rows, 3], u64::try_from(rows).ok()),
    };
    let output_dtype = output_dtype(operation, input.layout, pixels, input.dtype);
    let element_type = match output_dtype {
        NumericDType::F32 => ProviderElementType::F32,
        NumericDType::F64 => ProviderElementType::F64,
        NumericDType::U8 => ProviderElementType::U8,
        NumericDType::U16 => ProviderElementType::U16,
        _ => return Ok(None),
    };
    let host_scratch = host_conversion_scratch(operation, input.dtype, pixels);
    let Some(provider_shape) = shape
        .iter()
        .map(|&extent| u64::try_from(extent).ok())
        .collect::<Option<Vec<_>>>()
    else {
        return Ok(None);
    };
    let output = ProviderRepresentation {
        element_type,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape: provider_shape,
        residency: ProviderResidency::Device,
    };
    let facts = Digest::sha256(
        serde_json::to_vec(&(
            owner.shape(),
            owner.physical().element_type(),
            owner.view().layout(),
            &output,
        ))
        .expect("typed RGB presentation facts serialize"),
    );
    let receipt = select_exact_host_upload(
        runtime,
        ExactHostUpload {
            operation: name,
            variant: variant(operation),
            owner,
            domain_facts: facts,
            outputs: vec![output],
            host_compute_scratch: host_scratch,
        },
    )?;
    Ok(receipt.map(|receipt| SelectedRgbPresentation {
        owner: owner.clone(),
        shape,
        element_type,
        receipt,
    }))
}

fn output_dtype(
    operation: RgbConversion,
    layout: ColorLayout,
    pixels: Option<u64>,
    input_dtype: NumericDType,
) -> NumericDType {
    match operation {
        RgbConversion::Gray => match layout {
            ColorLayout::Colormap { .. } => NumericDType::F64,
            ColorLayout::Truecolor { .. }
                if pixels == Some(1) && input_dtype == NumericDType::F32 =>
            {
                NumericDType::F64
            }
            ColorLayout::Truecolor { .. } => input_dtype,
        },
        RgbConversion::Hsv => super::common::image_output_dtype(input_dtype),
    }
}

/// Excludes the provider's returned native readback vector and the final host
/// result; shared admission accounts for those. HSV constructs an f64 work
/// vector, and non-double input also materializes an f64 value vector.
fn host_conversion_scratch(
    operation: RgbConversion,
    input_dtype: NumericDType,
    pixels: Option<u64>,
) -> ResourceByteBound {
    let bytes = match operation {
        RgbConversion::Gray => pixels.and_then(|pixels| pixels.checked_mul(8)),
        RgbConversion::Hsv => {
            pixels
                .and_then(|pixels| pixels.checked_mul(3))
                .and_then(|elements| {
                    elements.checked_mul(if input_dtype == NumericDType::F64 {
                        8
                    } else {
                        16
                    })
                })
        }
    };
    bytes
        .map(ResourceByteBound::Known)
        .unwrap_or(ResourceByteBound::Unavailable(
            ResourceBoundUnavailable::NoScratchBound,
        ))
}

pub(crate) fn publish(
    selected: &mut Option<SelectedRgbPresentation>,
    restore: impl FnOnce() -> BuiltinResult<Value>,
) -> BuiltinResult<Value> {
    let Some(mut selected) = selected.take() else {
        return restore();
    };
    let recorder = selected.receipt.recorder();
    let mut result =
        runmat_accelerate_api::observation::with_provider_observation(Some(recorder), restore);
    if let Ok(Value::GpuTensor(handle)) = &result {
        if let Err(error) = selected.receipt.validate_published_outputs(&[handle]) {
            result = Err(error);
        }
    }
    let published = match &result {
        Ok(value) => Ok(selected.matches(value)),
        Err(error) if error.identifier() == Some("RunMat:Cancelled") => {
            Err(ProviderObservationFailure::Cancelled)
        }
        Err(_) => Err(ProviderObservationFailure::ProviderFailed),
    };
    selected.receipt.finish(published);
    result
}

impl SelectedRgbPresentation {
    fn matches(&self, output: &Value) -> bool {
        let Value::GpuTensor(handle) = output else {
            return false;
        };
        let Some(owner_provider) = provider_for_handle(&self.owner) else {
            return false;
        };
        handle.shape() == self.shape.as_slice()
            && handle.device_id() == self.owner.device_id()
            && handle.allocation_lease().validate_live().is_ok()
            && handle_storage(handle) == GpuTensorStorage::Real
            && !handle_is_logical(handle)
            && !runmat_accelerate_api::handle_is_transposed(handle)
            && !handle.shares_allocation_with(&self.owner)
            && match self.element_type {
                ProviderElementType::F32 => {
                    handle_precision(handle) == Some(ProviderPrecision::F32)
                }
                ProviderElementType::F64 => {
                    handle_precision(handle) == Some(ProviderPrecision::F64)
                }
                ProviderElementType::U8 => {
                    handle_integer_type(handle) == Some(IntegerElementType::U8)
                }
                ProviderElementType::U16 => {
                    handle_integer_type(handle) == Some(IntegerElementType::U16)
                }
                _ => false,
            }
            && provider_for_handle(handle)
                .is_some_and(|provider| same_provider_instance(provider, owner_provider))
    }
}

#[cfg(test)]
#[path = "rgb_presentation/tests.rs"]
mod tests;
