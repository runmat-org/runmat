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
    rest: &[Value],
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
    let bins = match rest {
        [] => {
            if handle_is_logical(owner) {
                LOGICAL_BINS
            } else {
                DEFAULT_GRAYSCALE_BINS
            }
        }
        [control] if !matches!(control, Value::GpuTensor(_)) && !is_colormap_argument(control) => {
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
            bins,
            outputs,
        ))
        .expect("typed imhist facts serialize"),
    );
    // The image's physical gather staging varies by provider. Do not invent a
    // zero or conflate the output payload with the host algorithm's scratch.
    let receipt = select_exact_host_upload(
        runtime,
        ExactHostUpload {
            operation: NAME,
            variant: "grayscale.host_gather_bin_restore.v1",
            owner,
            domain_facts: facts,
            outputs: vec![output; outputs],
            host_compute_scratch: ResourceByteBound::Unavailable(
                ResourceBoundUnavailable::NoScratchBound,
            ),
        },
    )?;
    Ok(receipt.map(|receipt| SelectedImhist {
        owner: owner.clone(),
        bins,
        expected_outputs: outputs,
        receipt,
    }))
}

pub(super) fn publish(
    selected: &mut Option<SelectedImhist>,
    restore: impl FnOnce() -> BuiltinResult<Value>,
) -> BuiltinResult<Value> {
    let Some(selected) = selected.take() else {
        return restore();
    };
    let recorder = selected.receipt.recorder();
    let result =
        runmat_accelerate_api::observation::with_provider_observation(Some(recorder), restore);
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
    let Some(selected) = selected.take() else {
        return restore();
    };
    let recorder = selected.receipt.recorder();
    let result =
        runmat_accelerate_api::observation::with_provider_observation(Some(recorder), restore);
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
