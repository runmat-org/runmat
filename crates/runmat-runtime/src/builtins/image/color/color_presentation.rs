//! Whole-call host conversion with exact resident-owner publication. Geometry
//! and controls are supplied by each admitted color identity; placement and
//! completion evidence belong to the shared host-presentation service.

use crate::context::host_presentation::{
    select_exact_host_upload_with_readbacks, ExactHostUpload, HostPresentationReceipt,
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
use runmat_execution::{Digest, ResourceBoundUnavailable, ResourceByteBound};
use runmat_value::{NumericDType, Value};

#[derive(Clone, Copy)]
pub(crate) enum ColorHostRoute {
    HsvToRgb,
    LabToRgb,
    RgbToLab,
    IndexedToRgb,
}

impl ColorHostRoute {
    fn identity(self) -> (&'static str, &'static str, u8) {
        match self {
            Self::HsvToRgb => ("hsv2rgb", "hsv.host_gather_convert_restore.v1", 1),
            Self::LabToRgb => ("lab2rgb", "lab.host_gather_convert_restore.v1", 2),
            Self::RgbToLab => ("rgb2lab", "rgb.host_gather_convert_restore.v1", 3),
            Self::IndexedToRgb => ("ind2rgb", "indexed.host_gather_convert_restore.v1", 4),
        }
    }
}

pub(crate) struct ColorHostUpload<'a> {
    pub(crate) route: ColorHostRoute,
    pub(crate) sources: &'a [GpuTensorHandle],
    pub(crate) output_shape: Vec<usize>,
    pub(crate) output_dtype: NumericDType,
    pub(crate) host_compute_scratch: ResourceByteBound,
    pub(crate) control_facts: Digest,
}

pub(crate) struct SelectedColorPresentation {
    sources: Vec<GpuTensorHandle>,
    owner: GpuTensorHandle,
    shape: Vec<usize>,
    element_type: ProviderElementType,
    receipt: HostPresentationReceipt,
}

pub(crate) fn checked_scratch(
    elements: Option<usize>,
    bytes_per_element: u64,
) -> ResourceByteBound {
    elements
        .and_then(|elements| u64::try_from(elements).ok())
        .and_then(|elements| elements.checked_mul(bytes_per_element))
        .map(ResourceByteBound::Known)
        .unwrap_or(ResourceByteBound::Unavailable(
            ResourceBoundUnavailable::NoScratchBound,
        ))
}

pub(crate) fn checked_elements(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1usize, |total, &extent| total.checked_mul(extent))
}

pub(crate) fn select(
    runtime: Option<&RuntimeContext>,
    spec: ColorHostUpload<'_>,
) -> BuiltinResult<Option<SelectedColorPresentation>> {
    let Some(first) = spec.sources.first() else {
        return Ok(None);
    };
    let owner = spec
        .sources
        .iter()
        .find(|source| runmat_accelerate_api::handle_is_explicit(source))
        .unwrap_or(first);
    let element_type = match spec.output_dtype {
        NumericDType::F32 => ProviderElementType::F32,
        NumericDType::F64 => ProviderElementType::F64,
        NumericDType::U8 => ProviderElementType::U8,
        NumericDType::U16 => ProviderElementType::U16,
        _ => return Ok(None),
    };
    let Some(shape) = spec
        .output_shape
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
        shape,
        residency: ProviderResidency::Device,
    };
    let (operation, variant, route_tag) = spec.route.identity();
    let facts = Digest::sha256(
        serde_json::to_vec(&(
            route_tag,
            spec.control_facts,
            &output,
            spec.sources
                .iter()
                .map(|source| {
                    (
                        source.shape(),
                        source.physical().element_type(),
                        source.view().layout(),
                        runmat_accelerate_api::handle_provenance(source),
                    )
                })
                .collect::<Vec<_>>(),
        ))
        .expect("typed color host route serializes"),
    );
    let receipt = select_exact_host_upload_with_readbacks(
        runtime,
        ExactHostUpload {
            operation,
            variant,
            owner,
            domain_facts: facts,
            outputs: vec![output],
            host_compute_scratch: spec.host_compute_scratch,
        },
        spec.sources,
    )?;
    Ok(receipt.map(|receipt| SelectedColorPresentation {
        sources: spec.sources.to_vec(),
        owner: owner.clone(),
        shape: spec.output_shape,
        element_type,
        receipt,
    }))
}

pub(crate) fn publish(
    selected: &mut Option<SelectedColorPresentation>,
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

impl SelectedColorPresentation {
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
            && self
                .sources
                .iter()
                .all(|source| !handle.shares_allocation_with(source))
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
