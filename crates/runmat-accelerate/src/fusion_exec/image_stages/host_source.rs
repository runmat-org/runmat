//! Host Single image source: pure exact upload evidence and the same selected
//! recipe body used for resident inputs. No upload is performed in `plan`.

use super::*;
use crate::fusion_exec::outputs::FusionOutputAdmission;
use runmat_accelerate_api::{
    class_preserving_upload_query, HostNumericDataView, HostNumericTensorView,
    ProviderClassPreservingUpload, ProviderFeasibilityQuery, ProviderRepresentation,
    ProviderUploadPhysicalEvidence,
};
use runmat_execution::{
    CandidatePeakLiveDemand, CandidateResourceDemand, Digest, ProviderInputStageDemand,
    ProviderInputStageDemands, ProviderResourceSnapshot, ResourceByteBound,
};
use runmat_value::{NumericDType, Tensor};

pub(crate) struct HostSingleImageSource {
    source: Tensor,
    pub(super) plan: selected_plan::ImageSelectedStagePlan,
    query: ProviderFeasibilityQuery,
    pub(super) upload: ProviderUploadPhysicalEvidence,
    pub(super) route: ProviderClassPreservingUpload,
    semantic_identity: Digest,
    provider_snapshot: ProviderResourceSnapshot,
    provider_metadata_bytes: Option<u64>,
    provider_power_workspace_bytes: Option<u64>,
    _placement_host_preparation:
        Option<runmat_runtime::context::host_allocation::HostAllocationReceipt>,
}

impl HostSingleImageSource {
    pub(crate) fn semantic_identity(&self) -> Digest {
        self.semantic_identity
    }

    pub(crate) fn preparation_route(&self) -> runmat_accelerate_api::ProviderPreparationRoute {
        self.route.route()
    }

    pub(crate) fn prepared_representation(&self) -> &ProviderRepresentation {
        self.route.prepared()
    }

    pub(crate) fn upload_bytes(&self) -> u64 {
        self.upload.output_requested_bytes
    }

    pub(super) fn plan(
        provider: &'static dyn AccelProvider,
        tensor: &Tensor,
        invocation: &ImageSourceInvocation,
    ) -> ProviderExecutionResult<Option<Self>> {
        let Some(words) = tensor.as_f32_slice() else {
            return Ok(None);
        };
        let Ok([batch, height, width]) = <[usize; 3]>::try_from(tensor.shape()) else {
            return Ok(None);
        };
        if tensor.numeric_dtype() != NumericDType::F32
            || [batch, height, width].contains(&0)
            || batch
                .checked_mul(height)
                .and_then(|len| len.checked_mul(width))
                != Some(words.len())
            || invocation.selected_staged_mean_power_host(tensor).is_none()
        {
            return Ok(None);
        }
        let Some(output) = ordered_mean::representation(ProviderPrecision::F32, tensor.shape())
        else {
            return Ok(None);
        };
        let Some(query) = class_preserving_upload_query(vec![output.clone()]) else {
            return Ok(None);
        };
        let Some(upload) = provider.upload_physical_evidence(&query) else {
            return Ok(None);
        };
        if output
            .checked_byte_len()
            .is_none_or(|logical| logical > upload.output_requested_bytes)
        {
            return Ok(None);
        }
        let route = ProviderClassPreservingUpload::new_checked(
            provider.device_id(),
            query.inputs[0].clone(),
            output.clone(),
            upload,
        );
        let Some(route) = route else {
            return Ok(None);
        };
        let Some(plan) = selected_plan::ImageSelectedStagePlan::new(
            provider,
            &Value::Tensor(tensor.clone()),
            output,
            upload.output_requested_bytes,
            invocation,
        )?
        else {
            return Ok(None);
        };
        let provider_snapshot = provider.placement_resources();
        provider_snapshot
            .validate_typed()
            .map_err(ProviderExecutionError::from_source)?;
        if provider_snapshot.device_id != provider.device_id() || provider_snapshot.lost {
            return Ok(None);
        }
        Ok(Some(Self {
            plan,
            source: tensor.clone(),
            query,
            upload,
            route,
            semantic_identity: invocation.semantic_identity(),
            provider_snapshot,
            provider_metadata_bytes: None,
            provider_power_workspace_bytes: None,
            _placement_host_preparation: None,
        }))
    }

    pub(super) fn set_provider_metadata_bytes(&mut self, bytes: u64) {
        self.provider_metadata_bytes = Some(bytes);
    }

    pub(super) fn set_provider_power_workspace_bytes(&mut self, bytes: u64) {
        self.provider_power_workspace_bytes = Some(bytes);
    }

    pub(super) fn finite_power_workspace_bytes(&self) -> Option<u64> {
        self.plan.powers.provider_host_workspace_bytes()
    }

    pub(super) fn finite_metadata_registration_count(&self) -> Option<usize> {
        if !self.plan.means.admitted_casts_supported() {
            return None;
        }
        self.plan.registered_output_count()?.checked_add(1)
    }

    pub(super) fn finite_metadata_derived_view_count(&self) -> Option<usize> {
        3usize.checked_add(self.plan.alias_view_count()?)
    }

    pub(super) fn retain_placement_preparation(
        &mut self,
        receipt: Option<runmat_runtime::context::host_allocation::HostAllocationReceipt>,
    ) {
        self._placement_host_preparation = receipt;
    }

    pub(super) fn take_placement_preparation(
        &mut self,
    ) -> Option<runmat_runtime::context::host_allocation::HostAllocationReceipt> {
        self._placement_host_preparation.take()
    }

    pub(crate) fn source_representation(&self) -> ProviderRepresentation {
        self.route.source().clone()
    }

    pub(crate) fn matches_source_and_provider(
        &self,
        provider: &dyn AccelProvider,
        source: &Value,
    ) -> ProviderExecutionResult<bool> {
        let Value::Tensor(tensor) = source else {
            return Ok(false);
        };
        let Some(bound) = self.source.as_f32_slice() else {
            return Ok(false);
        };
        let Some(actual) = tensor.as_f32_slice() else {
            return Ok(false);
        };
        let current = provider.placement_resources();
        current
            .validate_typed()
            .map_err(ProviderExecutionError::from_source)?;
        Ok(
            runmat_accelerate_api::same_provider_instance(self.plan.provider, provider)
                && current == self.provider_snapshot
                && self.plan.input_rep == *self.route.prepared()
                && tensor.shape() == self.source.shape()
                && actual.len() == bound.len()
                && std::ptr::eq(actual.as_ptr(), bound.as_ptr())
                && provider.upload_physical_evidence(&self.query) == Some(self.upload),
        )
    }

    pub(crate) fn physical_resources(&self) -> Option<CandidateResourceDemand> {
        let stages = self.plan.physical_upper_bound()?;
        let retained = stages
            .requested_bytes
            .checked_add(self.upload.output_requested_bytes)?;
        let upload_scratch = self.upload.provider_scratch_bytes;
        let peak = match upload_scratch {
            ResourceByteBound::Known(bytes) => Some(CandidatePeakLiveDemand {
                peak_bytes: retained.checked_add(bytes)?,
                already_live_bytes: 0,
            }),
            ResourceByteBound::Unavailable(_) => None,
        };
        let upload_stage_peak = match upload_scratch {
            ResourceByteBound::Known(bytes) => {
                ResourceByteBound::Known(self.upload.output_requested_bytes.checked_add(bytes)?)
            }
            ResourceByteBound::Unavailable(reason) => ResourceByteBound::Unavailable(reason),
        };
        let mut demand = CandidateResourceDemand {
            cpu_millicores: 0,
            retained_bytes: ResourceByteBound::Known(retained),
            scratch_bytes: upload_scratch,
            queue_slots: self
                .upload
                .completion_queue_slots
                .max(u32::from(stages.dispatches != 0)),
            provider_host_workspace: match (
                self.provider_metadata_bytes,
                self.provider_power_workspace_bytes,
            ) {
                (Some(metadata), Some(power)) => match self.upload.host_workspace_bytes {
                    ResourceByteBound::Known(bytes) => bytes
                        .checked_add(metadata)
                        .and_then(|bytes| bytes.checked_add(power))
                        .map(ResourceByteBound::Known),
                    unavailable @ ResourceByteBound::Unavailable(_) => Some(unavailable),
                },
                (None, None) => Some(self.upload.host_workspace_bytes),
                _ => None,
            },
            provider_input_stages: Some(ProviderInputStageDemands::sequential([
                ProviderInputStageDemand {
                    device_id: self.plan.provider.device_id(),
                    incremental_peak_bytes: upload_stage_peak,
                    queue_slots: self.upload.completion_queue_slots,
                },
            ])?),
            peak_live: peak,
        };
        if let Some(tail) = self.plan.provider.region_queue_tail_resources() {
            demand.queue_slots = demand.queue_slots.max(tail.queue_slots);
            demand = demand.with_provider_observation_peak(tail.provider_scratch_bytes)?;
            super::prepared::charge_region_tail_host_demand(&mut demand, tail)?;
        }
        Some(demand)
    }

    /// Stage work is issued by the same exact recipes for host and resident
    /// inputs. The upload fence is charged and counted separately.
    pub(crate) fn stage_physical_work(
        &self,
    ) -> Option<runmat_accelerate_api::ProviderImageStagePhysicalWork> {
        let stages = self.plan.physical_upper_bound()?;
        let work = runmat_accelerate_api::ProviderImageStagePhysicalWork {
            requested_bytes: stages.requested_bytes,
            dispatches: stages.dispatches,
            queue_submissions: stages.queue_submissions?,
        };
        work.validate().then_some(work)
    }

    /// Called only after selection, with the same source slice kept live by
    /// the VM request. The selected upload evidence is rechecked before its
    /// first effect; the resulting handle is validated by the common owner.
    pub(super) fn upload_selected(
        self,
        tensor: &Tensor,
        metadata_admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> ProviderExecutionResult<Option<ImageNormalizeStageSession>> {
        let Some(bound_words) = self.source.as_f32_slice() else {
            return Ok(None);
        };
        let Some(actual_words) = tensor.as_f32_slice() else {
            return Ok(None);
        };
        if tensor.numeric_dtype() != NumericDType::F32
            || tensor.shape()
                != [
                    self.plan.desc.batch,
                    self.plan.desc.height,
                    self.plan.desc.width,
                ]
            || bound_words.len() != actual_words.len()
            || !std::ptr::eq(bound_words.as_ptr(), actual_words.as_ptr())
        {
            return Ok(None);
        }
        let current = self.plan.provider.placement_resources();
        current
            .validate_typed()
            .map_err(ProviderExecutionError::from_source)?;
        if current != self.provider_snapshot
            || self.plan.provider.upload_physical_evidence(&self.query) != Some(self.upload)
        {
            return Ok(None);
        }
        let mut output_admission = FusionOutputAdmission::new(&[]);
        let handle = output_admission
            .upload_typed_admitted(
                self.plan.provider,
                &HostNumericTensorView {
                    data: HostNumericDataView::F32(actual_words),
                    shape: tensor.shape(),
                    storage: GpuTensorStorage::Real,
                },
                metadata_admission,
            )?
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
            .map_err(|error| ProviderExecutionError::from_source(error).after_effect())?;
        self.plan
            .bind(&handle)
            .map_err(ProviderExecutionError::after_effect)?
            .ok_or_else(|| {
                ProviderExecutionError::from(anyhow!(
                    "selected host image upload no longer matches its physical recipe"
                ))
                .after_effect()
            })
            .map(Some)
    }
}
