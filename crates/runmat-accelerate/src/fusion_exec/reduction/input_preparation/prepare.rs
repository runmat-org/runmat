//! Consume a selected immutable preparation plan with whole-call effect phase.
use super::{host, invalid, InputAction, TypedInputPlan};
use crate::fusion_exec::outputs::{reject_outputs, FusionOutputAdmission};
use runmat_accelerate_api::reduction::ProviderReductionInputStage;
use runmat_accelerate_api::{
    class_preserving_upload_query, same_provider_instance, GpuTensorHandle, ProviderElementType,
    ProviderExecutionResult,
};
use runmat_runtime::builtins::common::gpu_helpers::exact_provider_for_handle;

impl TypedInputPlan<'_> {
    /// Execute only the already selected ordered stages. Returned handles own
    /// every prepared input until ReductionSubmission has completed.
    pub(in crate::fusion_exec::reduction) async fn prepare(
        self,
    ) -> ProviderExecutionResult<Vec<GpuTensorHandle>> {
        // The plan captured its provider when all sources and proposals were
        // checked. An all-host plan has no resident handle to re-establish
        // this authority, so execution must never accept a replacement facade.
        let provider = self.owner;
        let originals: Vec<&GpuTensorHandle> = self
            .actions
            .iter()
            .filter_map(|action| match action {
                InputAction::Borrow(handle) | InputAction::Cast { source: handle, .. } => {
                    Some(*handle)
                }
                InputAction::Upload { .. } => None,
            })
            .collect();
        // Recheck all resident sources before the first effect, not lazily
        // after an earlier upload has succeeded.
        for handle in &originals {
            if handle.device_id() != provider.device_id()
                || handle.physical().requested_bytes() > self.binding_cap
                || !exact_provider_for_handle(handle)
                    .is_some_and(|owner| same_provider_instance(owner, provider))
                || handle.view().validate(handle.physical()).is_err()
                || handle.allocation_lease().validate_live().is_err()
            {
                return Err(invalid(
                    "typed resident authority changed before preparation",
                ));
            }
        }
        // A selected proposal is immutable, but the provider may change its
        // advertised physical route between planning and execution. Recheck
        // the entire set before any upload/cast starts; later backend checks
        // still own actual allocation authentication.
        for entry in self.inventory.entries() {
            match &entry.stage {
                ProviderReductionInputStage::UploadExact { upload }
                | ProviderReductionInputStage::HostCastThenUpload { upload, .. } => {
                    if upload.output_requested_bytes > self.binding_cap {
                        return Err(invalid("typed selected upload exceeds binding limit"));
                    }
                    let query = class_preserving_upload_query(vec![entry.prepared.clone()])
                        .ok_or_else(|| invalid("typed upload query changed"))?;
                    if provider.upload_physical_evidence(&query).as_ref() != Some(upload) {
                        return Err(invalid("typed upload physical evidence changed"));
                    }
                }
                ProviderReductionInputStage::ResidentCast { physical, .. } => {
                    if physical.output_buffer_bytes > self.binding_cap {
                        return Err(invalid("typed selected cast exceeds binding limit"));
                    }
                }
                ProviderReductionInputStage::BorrowResident { .. } => {}
            }
        }
        for cast in &self.casts {
            if !cast.is_owned_by(provider)
                || provider.propose_real_elementwise(cast.request())?.as_ref()
                    != Some(cast.physical())
            {
                return Err(invalid("typed resident cast proposal changed"));
            }
        }
        // Validate every selected target shape before any provider effect. The
        // executor never truncates a u64 extent into the host's usize domain.
        let prepared_shapes: Vec<Vec<usize>> = self
            .inventory
            .entries()
            .iter()
            .map(|entry| {
                entry
                    .prepared
                    .shape
                    .iter()
                    .map(|&extent| {
                        usize::try_from(extent)
                            .map_err(|_| invalid("typed prepared extent exceeds host usize"))
                    })
                    .collect::<ProviderExecutionResult<Vec<_>>>()
            })
            .collect::<ProviderExecutionResult<Vec<_>>>()?;
        let mut admission = FusionOutputAdmission::with_borrowed_handles(originals.iter().copied());
        let mut prepared = Vec::with_capacity(self.actions.len());
        let mut effected = false;
        for (index, (action, entry)) in self
            .actions
            .into_iter()
            .zip(self.inventory.entries())
            .enumerate()
        {
            let result = match action {
                InputAction::Borrow(handle) => Ok(handle.clone()),
                InputAction::Upload { source, target } => {
                    host::upload(&mut admission, provider, source, target)
                }
                InputAction::Cast {
                    source,
                    recipe_index,
                } => {
                    let protected: Vec<_> =
                        originals.iter().copied().chain(prepared.iter()).collect();
                    match self.casts[recipe_index]
                        .execute_protected(source, &protected)
                        .await
                    {
                        Ok(executed) => admission.admit(
                            executed.output,
                            provider,
                            &prepared_shapes[index],
                            match entry.prepared.element_type {
                                ProviderElementType::F32 => {
                                    runmat_accelerate_api::ProviderPrecision::F32
                                }
                                ProviderElementType::F64 => {
                                    runmat_accelerate_api::ProviderPrecision::F64
                                }
                                _ => unreachable!("checked typed input class"),
                            },
                        ),
                        Err(error) => Err(error),
                    }
                }
            };
            let handle = result.map_err(|error| {
                if effected {
                    error.after_effect()
                } else {
                    error
                }
            })?;
            if !matches!(
                &entry.stage,
                ProviderReductionInputStage::BorrowResident { .. }
            ) {
                effected = true;
            }
            let selected_requested = match &entry.stage {
                ProviderReductionInputStage::UploadExact { upload }
                | ProviderReductionInputStage::HostCastThenUpload { upload, .. } => {
                    Some(upload.output_requested_bytes)
                }
                ProviderReductionInputStage::ResidentCast { physical, .. } => {
                    Some(physical.output_buffer_bytes)
                }
                ProviderReductionInputStage::BorrowResident { .. } => None,
            };
            if selected_requested.is_some_and(|bytes| handle.physical().requested_bytes() != bytes)
            {
                return Err(reject_outputs(
                    vec![handle],
                    "typed prepared allocation differs from selected physical proposal",
                ));
            }
            let expected_payload = entry
                .prepared
                .checked_byte_len()
                .expect("checked inventory");
            if handle.physical().payload_bytes() < expected_payload
                || handle.shape() != prepared_shapes[index]
            {
                return Err(
                    if matches!(
                        &entry.stage,
                        ProviderReductionInputStage::BorrowResident { .. }
                    ) {
                        let error = invalid("typed borrowed input differs from selected binding");
                        if effected {
                            error.after_effect()
                        } else {
                            error
                        }
                    } else {
                        reject_outputs(
                            vec![handle],
                            "typed prepared input differs from selected binding",
                        )
                    },
                );
            }
            prepared.push(handle);
        }
        Ok(prepared)
    }
}
