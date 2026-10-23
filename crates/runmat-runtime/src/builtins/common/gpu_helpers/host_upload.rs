//! One admitted native host-upload owner shared by resident result domains.
//! The result class can differ from a Like prototype (randperm is Double),
//! but the provider and output intent can never silently change.

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::gpu_helpers::query_metadata;
use crate::context::host_allocation::{
    provider_registration_host_quote, HostAllocationError, HostAllocationPlan,
    HostAllocationReceipt, HostNumericWorkspace,
};
use crate::context::host_effect::HostEffectAllocationError;
use crate::context::host_presentation::{
    selected_upload_metadata_bytes, upload_candidate, HostPresentationReceipt, UploadStageResources,
};
use crate::{build_runtime_error, BuiltinResult};
use runmat_accelerate_api::{
    class_preserving_upload_query, upload_logical_protecting_admitted,
    upload_numeric_protecting_admitted, AccelProvider, GpuTensorHandle, GpuTensorStorage,
    HostLogicalTensorView, HostLogicalUploadScratch, HostNumericDataView, HostNumericTensorView,
    NumericElementType, ProviderElementType, ProviderFeasibility, ProviderFeasibilityQuery,
    ProviderLayout, ProviderQueryMetadataFootprint, ProviderRepresentation, ProviderResidency,
    ProviderStorage,
};
use runmat_execution::resource::ProviderHostMetadataAdmission;
use runmat_execution::{
    single_candidate_resources_admitted, CandidateExecutionLocation, CandidateOutputPresentation,
    CandidateOutputResidency, CandidatePreparationState, CandidateResourceDemand,
    CandidateResourceFacts, Digest, ExecutionCandidateKind, PlacementResourceSnapshot,
    ResourceBoundUnavailable, ResourceByteBound, WholeWorkCandidate,
};
use runmat_value::{ComplexTensor, NumericDType};

mod publication;
mod quote;

fn error(label: &'static str, message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message).with_builtin(label).build()
}

fn provider_element(class: NumericElementType) -> ProviderElementType {
    match class {
        NumericElementType::F64 => ProviderElementType::F64,
        NumericElementType::F32 => ProviderElementType::F32,
        NumericElementType::I8 => ProviderElementType::I8,
        NumericElementType::I16 => ProviderElementType::I16,
        NumericElementType::I32 => ProviderElementType::I32,
        NumericElementType::I64 => ProviderElementType::I64,
        NumericElementType::U8 => ProviderElementType::U8,
        NumericElementType::U16 => ProviderElementType::U16,
        NumericElementType::U32 => ProviderElementType::U32,
        NumericElementType::U64 => ProviderElementType::U64,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum HostUploadClass {
    Numeric {
        element: NumericElementType,
        storage: GpuTensorStorage,
    },
    Logical,
}

impl HostUploadClass {
    fn provider_representation(self) -> Option<(ProviderElementType, ProviderStorage)> {
        match self {
            Self::Logical => Some((ProviderElementType::Logical, ProviderStorage::DenseReal)),
            Self::Numeric {
                element,
                storage: GpuTensorStorage::Real,
            } => Some((provider_element(element), ProviderStorage::DenseReal)),
            Self::Numeric {
                element,
                storage: GpuTensorStorage::ComplexInterleaved,
            } => match element {
                NumericElementType::F32 => Some((
                    ProviderElementType::ComplexF32,
                    ProviderStorage::DenseComplexInterleaved,
                )),
                NumericElementType::F64 => Some((
                    ProviderElementType::ComplexF64,
                    ProviderStorage::DenseComplexInterleaved,
                )),
                _ => None,
            },
        }
    }

    fn host_element_bytes(self) -> u64 {
        match self {
            Self::Logical => 1,
            Self::Numeric { element, storage } => {
                (element.element_size() as u64)
                    * if storage == GpuTensorStorage::ComplexInterleaved {
                        2
                    } else {
                        1
                    }
            }
        }
    }
}

fn logical_workspace_requested_bytes(
    label: &'static str,
    provider: &dyn AccelProvider,
    shape: &[usize],
) -> BuiltinResult<u64> {
    let len = if shape.contains(&0) {
        0
    } else {
        shape
            .iter()
            .try_fold(1_usize, |count, &dim| count.checked_mul(dim))
            .ok_or_else(|| error(label, "logical upload workspace extent overflow"))?
    };
    let dtype = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F32 => NumericDType::F32,
        runmat_accelerate_api::ProviderPrecision::F64 => NumericDType::F64,
    };
    HostAllocationPlan::numeric_workspace_bytes(dtype, len)
        .map_err(|source| error(label, format!("logical upload workspace: {source}")))
}

pub(crate) struct GpuHostUpload<'a> {
    label: &'static str,
    provider: &'static dyn AccelProvider,
    source: &'a GpuTensorHandle,
    shape: &'a [usize],
    class: HostUploadClass,
    physical_bytes: u64,
    runtime: Option<crate::context::RuntimeContext>,
    // Rust drops fields in declaration order. The permit releases before the
    // managed headroom owner, including on every early return.
    permit: Option<Box<dyn ProviderHostMetadataAdmission>>,
    receipt: Option<HostAllocationReceipt>,
    selection: Option<HostPresentationReceipt>,
}

pub(crate) struct GpuHostCandidate {
    pub(crate) candidate: WholeWorkCandidate,
    pub(crate) query: ProviderFeasibilityQuery,
    pub(crate) output: ProviderRepresentation,
    pub(crate) requested_bytes: u64,
    pub(crate) query_metadata_bytes: u64,
    pub(crate) publication_metadata_bytes: u64,
    // Query/output shape records remain owned by this candidate. Drop them
    // before releasing the preparation claim, including the unchosen route.
    _query_receipt: Option<HostAllocationReceipt>,
}

impl<'a> GpuHostUpload<'a> {
    /// An independently claimed host result may reuse this exact upload owner
    /// without a child generator or an ambient execution-context lookup.
    pub(crate) fn prepare_borrowed_native_output_in_context(
        runtime: &crate::context::RuntimeContext,
        label: &'static str,
        source: &'a GpuTensorHandle,
        shape: &'a [usize],
        element: NumericElementType,
        storage: GpuTensorStorage,
        compute_scratch_bytes: u64,
    ) -> BuiltinResult<Self> {
        Self::prepare_borrowed_native_output_in_optional_context(
            Some(runtime),
            label,
            source,
            shape,
            element,
            storage,
            compute_scratch_bytes,
        )
    }

    pub(crate) fn prepare_borrowed_native_output_in_optional_context(
        runtime: Option<&crate::context::RuntimeContext>,
        label: &'static str,
        source: &'a GpuTensorHandle,
        shape: &'a [usize],
        element: NumericElementType,
        storage: GpuTensorStorage,
        compute_scratch_bytes: u64,
    ) -> BuiltinResult<Self> {
        Self::prepare_with_context(
            runtime.cloned(),
            label,
            source,
            shape,
            HostUploadClass::Numeric { element, storage },
            compute_scratch_bytes,
            false,
        )
    }

    pub(crate) fn prepare_borrowed_logical_output_in_optional_context(
        runtime: Option<&crate::context::RuntimeContext>,
        label: &'static str,
        source: &'a GpuTensorHandle,
        shape: &'a [usize],
    ) -> BuiltinResult<Self> {
        Self::prepare_with_context(
            runtime.cloned(),
            label,
            source,
            shape,
            HostUploadClass::Logical,
            0,
            false,
        )
    }
    fn reject_selected_publication(&mut self) {
        if let Some(selected) = self.selection.take() {
            selected.finish(Err(
                runmat_accelerate_api::observation::ProviderObservationFailure::PublicationFailed,
            ));
        }
    }

    /// `compute_scratch_bytes` excludes the single native result payload.
    /// All resource-denying checks occur before a stream range is reserved.
    pub(crate) fn prepare(
        label: &'static str,
        source: &'a GpuTensorHandle,
        shape: &'a [usize],
        class: NumericElementType,
        compute_scratch_bytes: u64,
    ) -> BuiltinResult<Self> {
        Self::prepare_with_output_workspace(
            label,
            source,
            shape,
            HostUploadClass::Numeric {
                element: class,
                storage: GpuTensorStorage::Real,
            },
            compute_scratch_bytes,
            true,
        )
    }

    /// The caller owns an independently claimed, borrowable host result.
    /// Upload preparation therefore charges provider metadata and work but
    /// does not reserve a second copy of the same native result payload.
    pub(crate) fn prepare_borrowed_host_output(
        label: &'static str,
        source: &'a GpuTensorHandle,
        shape: &'a [usize],
        class: NumericElementType,
        compute_scratch_bytes: u64,
    ) -> BuiltinResult<Self> {
        Self::prepare_with_output_workspace(
            label,
            source,
            shape,
            HostUploadClass::Numeric {
                element: class,
                storage: GpuTensorStorage::Real,
            },
            compute_scratch_bytes,
            false,
        )
    }

    pub(crate) fn prepare_borrowed_complex_host_output(
        label: &'static str,
        source: &'a GpuTensorHandle,
        shape: &'a [usize],
        class: NumericElementType,
        compute_scratch_bytes: u64,
    ) -> BuiltinResult<Self> {
        Self::prepare_with_output_workspace(
            label,
            source,
            shape,
            HostUploadClass::Numeric {
                element: class,
                storage: GpuTensorStorage::ComplexInterleaved,
            },
            compute_scratch_bytes,
            false,
        )
    }

    pub(crate) fn prepare_borrowed_logical_host_output(
        label: &'static str,
        source: &'a GpuTensorHandle,
        shape: &'a [usize],
    ) -> BuiltinResult<Self> {
        Self::prepare_with_output_workspace(
            label,
            source,
            shape,
            HostUploadClass::Logical,
            0,
            false,
        )
    }

    fn prepare_with_output_workspace(
        label: &'static str,
        source: &'a GpuTensorHandle,
        shape: &'a [usize],
        class: HostUploadClass,
        compute_scratch_bytes: u64,
        owns_native_output_workspace: bool,
    ) -> BuiltinResult<Self> {
        Self::prepare_with_context(
            crate::context::legacy::active(),
            label,
            source,
            shape,
            class,
            compute_scratch_bytes,
            owns_native_output_workspace,
        )
    }

    fn prepare_with_context(
        runtime: Option<crate::context::RuntimeContext>,
        label: &'static str,
        source: &'a GpuTensorHandle,
        shape: &'a [usize],
        class: HostUploadClass,
        compute_scratch_bytes: u64,
        owns_native_output_workspace: bool,
    ) -> BuiltinResult<Self> {
        let (element_type, storage_class) = class
            .provider_representation()
            .ok_or_else(|| error(label, format!("{label}: unsupported upload storage")))?;
        let provider = gpu_helpers::exact_provider_for_handle(source).ok_or_else(|| {
            error(
                label,
                format!("{label}: no provider owns the GPU prototype"),
            )
        })?;
        source.allocation_lease().validate_live().map_err(|cause| {
            crate::runtime_error::with_provider_execution_source(
                error(
                    label,
                    format!("{label}: GPU prototype lease is not live: {cause}"),
                ),
                runmat_accelerate_api::ProviderExecutionError::from_source(cause)
                    .with_unproven_effects(),
            )
        })?;
        let len = if shape.contains(&0) {
            0
        } else {
            shape
                .iter()
                .try_fold(1_usize, |n, &d| n.checked_mul(d))
                .ok_or_else(|| error(label, format!("{label}: output extent overflow")))?
        };
        let native_bytes = u64::try_from(len)
            .ok()
            .and_then(|len| len.checked_mul(class.host_element_bytes()))
            .ok_or_else(|| error(label, format!("{label}: native output extent overflow")))?;
        let footprint = ProviderQueryMetadataFootprint {
            operation: "transfer.upload",
            input_ranks: &[shape.len()],
            output_ranks: &[shape.len()],
            preparation_shape_ranks: &[],
            preparation_representation_slots: 0,
        };
        let query_bytes = query_metadata::requested_bytes(label, &footprint)?;
        let _query_receipt = query_metadata::admit(runtime.as_ref(), query_bytes)
            .map_err(|source| query_metadata::admission_error(label, runtime.as_ref(), source))?;
        let output = ProviderRepresentation {
            element_type,
            storage: storage_class,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape: shape
                .iter()
                .map(|&n| u64::try_from(n))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| error(label, format!("{label}: output dimension overflow")))?,
            residency: ProviderResidency::Device,
        };
        let query = class_preserving_upload_query(vec![output])
            .ok_or_else(|| error(label, format!("{label}: no native upload representation")))?;
        if !footprint.matches(&query) {
            return Err(error(
                label,
                format!("{label}: upload query metadata plan changed"),
            ));
        }
        let ProviderFeasibility::Supported { estimate } = provider.query_feasibility(&query) else {
            return Err(crate::runtime_error::with_provider_execution_source(
                error(
                    label,
                    format!("{label}: provider cannot upload this native class"),
                ),
                runmat_accelerate_api::unsupported_provider_operation(
                    "provider cannot upload the requested native class",
                )
                .into(),
            ));
        };
        let physical = provider.upload_physical_evidence(&query).ok_or_else(|| {
            error(
                label,
                format!("{label}: provider has no physical upload evidence"),
            )
        })?;
        if estimate.output_bytes != Some(physical.output_requested_bytes) {
            return Err(error(
                label,
                format!("{label}: physical output estimate disagrees"),
            ));
        }
        let registration = provider_registration_host_quote(provider, shape.len(), 1, 1)
            .map_err(|source| error(label, format!("{label}: metadata quote: {source}")))?;
        let parallel = runtime
            .as_ref()
            .and_then(|runtime| runtime.service_ports().parallel())
            .map(|service| service.placement_resources())
            .unwrap_or_default();
        if parallel.memory_available_bytes.is_some() && registration.is_none() {
            return Err(error(
                label,
                format!("{label}: provider lacks finite host metadata evidence"),
            ));
        }
        let metadata_bytes = match registration {
            Some(quote) => quote
                .preparation_bytes
                .checked_add(quote.retained_bytes)
                .ok_or_else(|| error(label, format!("{label}: metadata extent overflow")))?,
            None => 0,
        };
        let provider_workspace = match physical.host_workspace_bytes {
            ResourceByteBound::Known(bytes) => bytes,
            ResourceByteBound::Unavailable(_) if parallel.memory_available_bytes.is_some() => {
                return Err(error(
                    label,
                    format!("{label}: provider host workspace is unbounded"),
                ));
            }
            ResourceByteBound::Unavailable(_) => 0,
        };
        let provider_workspace = if class == HostUploadClass::Logical {
            let exact = logical_workspace_requested_bytes(label, provider, shape)?;
            if exact < provider_workspace {
                return Err(error(
                    label,
                    "logical upload workspace quote disagrees with physical owner",
                ));
            }
            exact
        } else {
            provider_workspace
        };
        let quoted_host_peak = compute_scratch_bytes
            .checked_add(metadata_bytes)
            .map(|bytes| bytes.max(query_bytes))
            .ok_or_else(|| error(label, format!("{label}: metadata peak overflow")))?;
        let presentation = CandidateOutputPresentation::ProviderUpload {
            device_id: provider.device_id(),
            transfer_bytes: Some(physical.output_requested_bytes),
            host_result_bytes: Some(ResourceByteBound::Known(native_bytes)),
            host_scratch_bytes: physical.host_workspace_bytes,
            provider_scratch_bytes: physical.provider_scratch_bytes,
            queue_slots: physical.completion_queue_slots,
            preparation: CandidatePreparationState::Ready,
        };
        single_candidate_resources_admitted(
            CandidateResourceFacts {
                location: CandidateExecutionLocation::Host,
                output_residency: CandidateOutputResidency::Provider {
                    device_id: provider.device_id(),
                },
                presentation: &presentation,
                kind: ExecutionCandidateKind::SharedRuntime,
                demand: CandidateResourceDemand {
                    cpu_millicores: 1_000,
                    retained_bytes: ResourceByteBound::Known(physical.output_requested_bytes),
                    scratch_bytes: ResourceByteBound::Known(quoted_host_peak),
                    provider_host_workspace: None,
                    provider_input_stages: None,
                    queue_slots: 0,
                    peak_live: None,
                },
                transactional_results: true,
            },
            &PlacementResourceSnapshot {
                cpu_millicores_available: parallel.cpu_millicores_available,
                memory_available_bytes: parallel.memory_available_bytes,
                cancellation_requested: runtime
                    .as_ref()
                    .is_some_and(|runtime| runtime.is_cancelled()),
                providers: vec![provider.placement_resources()],
                epoch: parallel.epoch,
            },
            false,
        )
        .map_err(|source| {
            error(
                label,
                format!("{label}: host-mediated route not admitted: {source}"),
            )
        })?;
        drop(query);
        drop(_query_receipt);
        let owned_result_workspace = if owns_native_output_workspace {
            native_bytes
        } else {
            0
        };
        let (scratch, retained) = match registration {
            Some(quote) => (
                compute_scratch_bytes
                    .checked_add(owned_result_workspace)
                    .and_then(|bytes| bytes.checked_add(provider_workspace))
                    .and_then(|bytes| bytes.checked_add(quote.preparation_bytes))
                    .ok_or_else(|| error(label, format!("{label}: host extent overflow")))?,
                quote.retained_bytes,
            ),
            None => (
                compute_scratch_bytes
                    .checked_add(owned_result_workspace)
                    .and_then(|bytes| bytes.checked_add(provider_workspace))
                    .ok_or_else(|| error(label, format!("{label}: host extent overflow")))?,
                0,
            ),
        };
        let receipt = HostAllocationPlan::new(scratch, retained)
            .and_then(|plan| HostAllocationReceipt::admit(runtime.as_ref(), plan))
            .map_err(|source| {
                crate::context::host_allocation::host_allocation_runtime_error(
                    runtime.as_ref(),
                    &format!("RunMat:{label}:Allocation"),
                    "native host upload allocation",
                    source,
                )
            })?;
        let permit = if let (Some(runtime), Some(receipt)) = (runtime.as_ref(), receipt.as_ref()) {
            let retained = receipt.retained_view_permit(runtime).map_err(|source| {
                crate::context::host_allocation::host_allocation_runtime_error(
                    Some(runtime),
                    &format!("RunMat:{label}:Allocation"),
                    "native host upload metadata permit",
                    source,
                )
            })?;
            Some(
                provider
                    .prepare_output_host_metadata(Box::new(retained), 1)
                    .map_err(|source| {
                        error(label, format!("{label}: provider metadata: {source}"))
                    })?,
            )
        } else {
            None
        };
        Ok(Self {
            label,
            provider,
            source,
            shape,
            class,
            physical_bytes: physical.output_requested_bytes,
            runtime,
            permit,
            receipt,
            selection: None,
        })
    }

    pub(crate) fn with_selected_host_route(
        mut self,
        selection: HostPresentationReceipt,
        quoted_requested_bytes: u64,
    ) -> BuiltinResult<Self> {
        if self.physical_bytes != quoted_requested_bytes {
            selection.finish(Err(
                runmat_accelerate_api::observation::ProviderObservationFailure::ResourceLimit,
            ));
            return Err(error(
                self.label,
                format!("{}: selected upload physical evidence changed", self.label),
            ));
        }
        self.selection = Some(selection);
        Ok(self)
    }

    pub(crate) fn runtime(&self) -> Option<&crate::context::RuntimeContext> {
        self.runtime.as_ref()
    }

    pub(crate) fn sample_workspace(
        &self,
        dtype: NumericDType,
        len: usize,
    ) -> BuiltinResult<Option<HostNumericWorkspace>> {
        let (Some(runtime), Some(receipt)) = (self.runtime.as_ref(), self.receipt.as_ref()) else {
            return Ok(None);
        };
        receipt
            .temporary_numeric(runtime, dtype, len)
            .map(Some)
            .map_err(|source| {
                crate::context::host_allocation::host_allocation_runtime_error(
                    Some(runtime),
                    &format!("RunMat:{}:Allocation", self.label),
                    "native host upload workspace",
                    source,
                )
            })
    }

    pub(crate) fn checkpoint(&self) -> BuiltinResult<()> {
        if let (Some(runtime), Some(receipt)) = (self.runtime.as_ref(), self.receipt.as_ref()) {
            receipt.checkpoint(runtime).map_err(|source| {
                crate::context::host_allocation::host_allocation_runtime_error(
                    Some(runtime),
                    &format!("RunMat:{}:Allocation", self.label),
                    "native host upload checkpoint",
                    source,
                )
            })?;
        }
        Ok(())
    }
}
