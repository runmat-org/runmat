//! Typed host histogram plus exact owner-upload route. No histogram provider
//! hook exists; this route measures the real gather, binning, and publication.

use super::*;
use crate::context::host_presentation::{
    select_exact_host_upload, ExactHostUpload, HostPresentationReceipt,
};
use runmat_accelerate_api::observation::ProviderObservationFailure;
use runmat_accelerate_api::{
    handle_integer_type, handle_is_logical, handle_precision, handle_storage, provider_for_handle,
    same_provider_instance, GpuTensorHandle, GpuTensorStorage, ProviderElementType, ProviderLayout,
    ProviderPrecision, ProviderRepresentation, ProviderResidency, ProviderStorage,
};
use runmat_execution::{Digest, ResourceBoundUnavailable, ResourceByteBound};

pub(super) struct SelectedImhist {
    owner: GpuTensorHandle,
    bins: usize,
    expected_outputs: usize,
    receipt: HostPresentationReceipt,
}

pub(super) fn select(
    runtime: Option<&crate::context::RuntimeContext>,
    image: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
    outputs: usize,
) -> BuiltinResult<Option<SelectedImhist>> {
    let Value::GpuTensor(owner) = image else {
        return Ok(None);
    };
    if outputs == 0
        || outputs > 2
        || rest.len() > 1
        || !matches!(handle_storage(owner), GpuTensorStorage::Real)
    {
        return Ok(None);
    }
    if matches!(
        handle_integer_type(owner),
        Some(
            runmat_accelerate_api::IntegerElementType::I64
                | runmat_accelerate_api::IntegerElementType::U64
        )
    ) {
        return Ok(None);
    }
    // Resident indexed calls retain their established unsupported diagnostic;
    // neither their colormap nor a resident control is an admitted upload route.
    let bins = match (rest.len(), rest.first()) {
        (0, _) => {
            if handle_is_logical(owner) {
                LOGICAL_BINS
            } else {
                DEFAULT_GRAYSCALE_BINS
            }
        }
        (1, Some(control))
            if !matches!(control, Value::GpuTensor(_)) && !is_colormap_argument(control) =>
        {
            let Some(bins) = call::parse_optional_bin_count(control)? else {
                return Ok(None);
            };
            bins
        }
        _ => return Ok(None),
    };
    if handle_precision(owner).is_none()
        && handle_integer_type(owner).is_none()
        && !handle_is_logical(owner)
    {
        return Ok(None);
    }
    let shape = vec![
        u64::try_from(bins).map_err(|_| internal("bin count exceeds provider shape range"))?,
        1,
    ];
    let output = ProviderRepresentation {
        element_type: ProviderElementType::F64,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape,
        residency: ProviderResidency::Device,
    };
    let facts = Digest::sha256(
        serde_json::to_vec(&(
            owner.shape(),
            handle_precision(owner),
            handle_integer_type(owner),
            handle_is_logical(owner),
            owner.view().layout(),
            bins,
            outputs,
        ))
        .expect("typed imhist facts serialize"),
    );
    // Unrepresentable arithmetic is unknown resource evidence, not an
    // unselected route that could execute without a finite-quota check.
    let host_compute_scratch = host_scratch_bound(owner, bins, outputs)
        .map(ResourceByteBound::Known)
        .unwrap_or(ResourceByteBound::Unavailable(
            ResourceBoundUnavailable::NoScratchBound,
        ));
    let receipt = select_exact_host_upload(
        runtime,
        ExactHostUpload {
            operation: NAME,
            variant: "grayscale.host_gather_bin_restore.v1",
            owner,
            domain_facts: facts,
            outputs: vec![output; outputs],
            host_compute_scratch,
        },
    )?;
    Ok(receipt.map(|receipt| SelectedImhist {
        owner: owner.clone(),
        bins,
        expected_outputs: outputs,
        receipt,
    }))
}

/// Bound only imhist-owned host temporaries outside the published result.
/// Shared admission adds the provider's separately bounded native download
/// payload and readback temporaries. Logical decoding may form a bit vector
/// and then an f64 vector; f32 binning forms an f64 vector. `linspace` always
/// constructs locations, even for a one-output call. Both published tensors
/// move their vectors and clones are COW handles.
fn host_scratch_bound(owner: &GpuTensorHandle, bins: usize, outputs: usize) -> Option<u64> {
    let elements = owner.shape().iter().try_fold(1_u64, |total, &extent| {
        total.checked_mul(u64::try_from(extent).ok()?)
    })?;
    let logical_bits = if handle_is_logical(owner) {
        elements
    } else {
        0
    };
    let f64_conversion =
        if handle_is_logical(owner) || handle_precision(owner) == Some(ProviderPrecision::F32) {
            elements.checked_mul(8)?
        } else {
            0
        };
    let unused_locations = if outputs == 1 {
        u64::try_from(bins).ok()?.checked_mul(8)?
    } else {
        0
    };
    logical_bits
        .checked_add(f64_conversion)?
        .checked_add(unused_locations)
}

#[cfg(test)]
#[path = "placement/tests.rs"]
mod resource_tests;

pub(super) fn publish(
    selected: &mut Option<SelectedImhist>,
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
        Ok(output) => Ok(selected.matches(std::slice::from_ref(output))),
        Err(error) if error.identifier() == Some("RunMat:Cancelled") => {
            Err(ProviderObservationFailure::Cancelled)
        }
        Err(_) => Err(ProviderObservationFailure::ProviderFailed),
    };
    selected.receipt.finish(published);
    result
}

pub(super) fn publish_many(
    selected: &mut Option<SelectedImhist>,
    restore: impl FnOnce() -> BuiltinResult<Vec<Value>>,
) -> BuiltinResult<Vec<Value>> {
    let Some(mut selected) = selected.take() else {
        return restore();
    };
    let recorder = selected.receipt.recorder();
    let mut result =
        runmat_accelerate_api::observation::with_provider_observation(Some(recorder), restore);
    if let Ok(outputs) = &result {
        if let Some(handles) = outputs
            .iter()
            .map(|output| match output {
                Value::GpuTensor(handle) => Some(handle),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
        {
            if let Err(error) = selected.receipt.validate_published_outputs(&handles) {
                result = Err(error);
            }
        }
    }
    let published = match &result {
        Ok(outputs) => Ok(selected.matches(outputs)),
        Err(error) if error.identifier() == Some("RunMat:Cancelled") => {
            Err(ProviderObservationFailure::Cancelled)
        }
        Err(_) => Err(ProviderObservationFailure::ProviderFailed),
    };
    selected.receipt.finish(published);
    result
}

impl SelectedImhist {
    fn matches(&self, outputs: &[Value]) -> bool {
        let Some(owner_provider) = provider_for_handle(&self.owner) else {
            return false;
        };
        outputs.len() == self.expected_outputs
            && outputs.iter().all(|output| {
                let Value::GpuTensor(handle) = output else {
                    return false;
                };
                handle.shape() == [self.bins, 1]
                    && handle.device_id() == self.owner.device_id()
                    && handle_precision(handle) == Some(ProviderPrecision::F64)
                    && handle_storage(handle) == GpuTensorStorage::Real
                    && !handle_is_logical(handle)
                    && !runmat_accelerate_api::handle_is_transposed(handle)
                    && provider_for_handle(handle)
                        .is_some_and(|provider| same_provider_instance(provider, owner_provider))
                    && !handle.shares_allocation_with(&self.owner)
            })
            && (outputs.len() < 2
                || match (&outputs[0], &outputs[1]) {
                    (Value::GpuTensor(left), Value::GpuTensor(right)) => {
                        !left.shares_allocation_with(right)
                    }
                    _ => false,
                })
    }
}
