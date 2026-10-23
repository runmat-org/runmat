//! Pure candidate quote for a selected host-compute/native-upload presentation.

use super::*;

impl<'a> GpuHostUpload<'a> {
    /// Pure pre-effect quote for a whole-work host generation/upload option.
    /// The shared upload candidate builder owns presentation/resource math;
    /// `prepare` later revalidates the same physical owner before any draw.
    pub(crate) fn quote_candidate(
        label: &'static str,
        source: &GpuTensorHandle,
        shape: &[usize],
        class: NumericElementType,
        compute_scratch_bytes: u64,
    ) -> BuiltinResult<Option<GpuHostCandidate>> {
        Self::quote_candidate_with_storage(
            label,
            source,
            shape,
            class,
            GpuTensorStorage::Real,
            compute_scratch_bytes,
        )
    }

    pub(crate) fn quote_candidate_with_storage(
        label: &'static str,
        source: &GpuTensorHandle,
        shape: &[usize],
        class: NumericElementType,
        storage: GpuTensorStorage,
        compute_scratch_bytes: u64,
    ) -> BuiltinResult<Option<GpuHostCandidate>> {
        Self::quote_candidate_for(
            label,
            source,
            shape,
            HostUploadClass::Numeric {
                element: class,
                storage,
            },
            compute_scratch_bytes,
        )
    }

    pub(crate) fn quote_logical_candidate(
        label: &'static str,
        source: &GpuTensorHandle,
        shape: &[usize],
        compute_scratch_bytes: u64,
    ) -> BuiltinResult<Option<GpuHostCandidate>> {
        Self::quote_candidate_for(
            label,
            source,
            shape,
            HostUploadClass::Logical,
            compute_scratch_bytes,
        )
    }

    fn quote_candidate_for(
        label: &'static str,
        source: &GpuTensorHandle,
        shape: &[usize],
        class: HostUploadClass,
        compute_scratch_bytes: u64,
    ) -> BuiltinResult<Option<GpuHostCandidate>> {
        let Some((element_type, storage_class)) = class.provider_representation() else {
            return Ok(None);
        };
        let Some(provider) = gpu_helpers::exact_provider_for_handle(source) else {
            return Ok(None);
        };
        source.allocation_lease().validate_live().map_err(|cause| {
            crate::runtime_error::with_provider_execution_source(
                error(label, format!("{label}: GPU prototype lease: {cause}")),
                runmat_accelerate_api::ProviderExecutionError::from_source(cause)
                    .with_unproven_effects(),
            )
        })?;
        let runtime = crate::context::legacy::active();
        let footprint = ProviderQueryMetadataFootprint {
            operation: "transfer.upload",
            input_ranks: &[shape.len()],
            output_ranks: &[shape.len()],
            preparation_shape_ranks: &[shape.len()],
            preparation_representation_slots: 0,
        };
        let query_bytes = query_metadata::requested_bytes(label, &footprint)?;
        let query_receipt = match query_metadata::admit(runtime.as_ref(), query_bytes) {
            Ok(receipt) => receipt,
            Err(HostAllocationError::Resource(HostEffectAllocationError::ExceedsCapacity)) => {
                return Ok(None);
            }
            Err(source) => {
                return Err(query_metadata::admission_error(
                    label,
                    runtime.as_ref(),
                    source,
                ));
            }
        };
        let host_shape = shape;
        let shape = shape
            .iter()
            .map(|&extent| u64::try_from(extent))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| error(label, format!("{label}: output dimension overflow")))?;
        let output = ProviderRepresentation {
            element_type,
            storage: storage_class,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape,
            residency: ProviderResidency::Device,
        };
        let Some(transfer_bytes) = output.checked_byte_len() else {
            return Ok(None);
        };
        let Some(query) = class_preserving_upload_query(vec![output.clone()]) else {
            return Ok(None);
        };
        let publication_bytes =
            selected_upload_metadata_bytes(std::slice::from_ref(&output), &query.operation, label)?;
        if !footprint.matches(&query) {
            return Err(error(
                label,
                format!("{label}: upload query metadata plan changed"),
            ));
        }
        let ProviderFeasibility::Supported { estimate } = provider.query_feasibility(&query) else {
            return Ok(None);
        };
        let Some(physical) = provider.upload_physical_evidence(&query) else {
            return Ok(None);
        };
        if estimate.output_bytes != Some(physical.output_requested_bytes) {
            return Err(error(
                label,
                format!("{label}: upload estimate contradicts physical evidence"),
            ));
        }
        let registration = provider_registration_host_quote(provider, output.shape.len(), 1, 1)
            .map_err(|cause| error(label, format!("{label}: metadata quote: {cause}")))?;
        let logical_workspace_extra = if class == HostUploadClass::Logical {
            let exact = logical_workspace_requested_bytes(label, provider, host_shape)?;
            let physical = match physical.host_workspace_bytes {
                ResourceByteBound::Known(bytes) => bytes,
                ResourceByteBound::Unavailable(_) => return Ok(None),
            };
            exact.checked_sub(physical).ok_or_else(|| {
                error(
                    label,
                    "logical upload physical workspace exceeds exact stage quote",
                )
            })?
        } else {
            0
        };
        let scratch = match registration {
            Some(quote) => compute_scratch_bytes
                .checked_add(logical_workspace_extra)
                .and_then(|bytes| bytes.checked_add(quote.preparation_bytes))
                .and_then(|bytes| bytes.checked_add(quote.retained_bytes))
                .map(|bytes| bytes.max(query_bytes))
                .and_then(|bytes| bytes.checked_add(publication_bytes))
                .map(ResourceByteBound::Known)
                .ok_or_else(|| error(label, format!("{label}: host quote overflow")))?,
            None => ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoScratchBound),
        };
        let candidate = upload_candidate(
            provider.device_id(),
            transfer_bytes,
            UploadStageResources {
                physical_retained_bytes: Some(physical.output_requested_bytes),
                host_workspace: physical.host_workspace_bytes,
                provider_scratch: physical.provider_scratch_bytes,
                queue_slots: physical.completion_queue_slots,
            },
            scratch,
            Digest::sha256_serialized(&query).expect("typed upload serializes"),
            None,
        );
        Ok(Some(GpuHostCandidate {
            candidate,
            query,
            output,
            requested_bytes: physical.output_requested_bytes,
            query_metadata_bytes: query_bytes,
            publication_metadata_bytes: publication_bytes,
            _query_receipt: query_receipt,
        }))
    }
}
