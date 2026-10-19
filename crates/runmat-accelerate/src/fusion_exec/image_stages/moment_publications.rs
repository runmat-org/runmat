//! Result-only publication. Once a raw result exists every failure is terminal;
//! neither Unsupported nor a cache hit can replay the image or erase a cast.
use super::*;
use runmat_accelerate_api::{moment_precision, ProviderImageMomentPublication};
use runmat_runtime::builtins::common::gpu_helpers::{
    unary_gpu_output_matches_shape, GpuOutputAliasPolicy, UnaryGpuOutputContract,
};

impl ImageNormalizeStageSession {
    pub(super) fn admit_reduction(
        &self,
        output: GpuTensorHandle,
        input: &GpuTensorHandle,
        shape: &[usize],
        precision: Option<ProviderPrecision>,
        scratch: &[GpuTensorHandle],
        published: &[&GpuTensorHandle],
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        let valid = precision.is_some()
            && unary_gpu_output_matches_shape(
                &output,
                input,
                self.provider,
                shape,
                UnaryGpuOutputContract {
                    storage: GpuTensorStorage::Real,
                    precision,
                    integer: None,
                    logical: false,
                    alias: GpuOutputAliasPolicy::RequireDistinctAllocation,
                },
            )
            && !runmat_accelerate_api::handle_is_transposed(&output)
            && scratch
                .iter()
                .chain(published.iter().copied())
                .chain([&self.input])
                .all(|handle| !output.shares_allocation_with(handle));
        if valid {
            Ok(output)
        } else {
            Err(super::super::image_output::reject_untrusted_output(
                output,
                std::iter::once(input)
                    .chain(scratch.iter())
                    .chain(published.iter().copied())
                    .chain([&self.input]),
                "selected image reduction has invalid class, shape or owner",
            )
            .after_effect())
        }
    }

    pub(super) async fn publish_moment(
        &self,
        raw: GpuTensorHandle,
        publication: &ProviderImageMomentPublication,
        scratch: &mut Vec<GpuTensorHandle>,
        published: &[&GpuTensorHandle],
        selected_casts: Option<&[ordered_mean::SelectedMomentCastPlan]>,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        let shape = publication
            .raw_fact
            .shape
            .known_dims()
            .and_then(|dims| dims.into_iter().collect::<Option<Vec<_>>>())
            .ok_or_else(|| {
                ProviderExecutionError::from(anyhow!("selected moment geometry is unproved"))
                    .after_effect()
            })?;
        if raw.shape() != shape
            || runmat_accelerate_api::handle_precision(&raw)
                != moment_precision(&publication.raw_fact)
        {
            return Err(super::super::image_output::reject_untrusted_output(
                raw,
                scratch
                    .iter()
                    .chain(published.iter().copied())
                    .chain([&self.input]),
                "selected raw moment differs from its publication",
            )
            .after_effect());
        }
        if selected_casts.is_some_and(|selected| selected.len() != publication.conversions.len()) {
            return Err(ProviderExecutionError::from(anyhow!(
                "selected moment cast plan count changed"
            ))
            .after_effect());
        }
        if publication.conversions.is_empty() {
            return Ok(raw);
        }
        let mut current = raw.clone();
        scratch.push(raw);
        for (index, call) in publication.conversions.iter().enumerate() {
            let selected = selected_casts.and_then(|casts| casts.get(index));
            if let Some(selected) = selected {
                current
                    .allocation_lease()
                    .validate_live()
                    .map_err(|source| ProviderExecutionError::from(source).after_effect())?;
                if !ordered_mean::casts::revalidate(self.provider, &current, call, selected) {
                    return Err(ProviderExecutionError::from(anyhow!(
                        "selected cast source or physical route changed"
                    ))
                    .after_effect());
                }
            }
            let precision = moment_precision(&call.output_fact).ok_or_else(|| {
                ProviderExecutionError::from(anyhow!("selected conversion class is unproved"))
                    .after_effect()
            })?;
            let intent = runmat_accelerate_api::handle_provenance(&current)
                .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic);
            let metadata_guard = runmat_accelerate_api::ResidencyRestoreGuard::new(&current);
            let direct = if let Some(permit) = self.provider_metadata_permit.as_ref() {
                let recipe = selected
                    .and_then(|selected| selected.real_recipe.as_ref())
                    .ok_or_else(|| {
                        ProviderExecutionError::from(anyhow!(
                            "finite selected moment cast has no native recipe"
                        ))
                        .after_effect()
                    })?;
                let protected = std::iter::once(&self.input)
                    .chain(std::iter::once(&current))
                    .chain(scratch.iter())
                    .chain(published.iter().copied())
                    .collect::<Vec<_>>();
                recipe
                    .execute_protected_admitted(&current, &protected, Some(&**permit))
                    .await
                    .map(|executed| executed.output)
            } else {
                match precision {
                    ProviderPrecision::F32 => self.provider.unary_single(&current).await,
                    ProviderPrecision::F64 => self.provider.unary_double(&current).await,
                }
            };
            // Restore before class/owner validation or propagating a provider
            // failure. The guard also restores on cancellation while awaiting
            // either hook, exactly as ordinary floating conversion does.
            metadata_guard.restore();
            drop(metadata_guard);
            let output = direct.map_err(ProviderExecutionError::after_effect)?;
            if let Some(selected) = selected {
                current
                    .allocation_lease()
                    .validate_live()
                    .map_err(|source| ProviderExecutionError::from(source).after_effect())?;
                if !ordered_mean::casts::revalidate(self.provider, &current, call, selected) {
                    return Err(super::super::image_output::reject_untrusted_output(
                        output,
                        std::iter::once(&current)
                            .chain(scratch.iter())
                            .chain(published.iter().copied())
                            .chain([&self.input]),
                        "selected cast physical route changed during execution",
                    ));
                }
                let same_class_alias = selected.source.element_type == selected.output.element_type;
                if output.physical().requested_bytes() != selected.output_bytes
                    || (same_class_alias && !output.shares_allocation_with(&current))
                {
                    return Err(super::super::image_output::reject_untrusted_output(
                        output,
                        std::iter::once(&current)
                            .chain(scratch.iter())
                            .chain(published.iter().copied())
                            .chain([&self.input]),
                        "selected cast physical output extent changed",
                    ));
                }
            }
            // This is the ordinary floating-conversion allocation contract,
            // including its validated immutable input-alias permission. Equal
            // class is never permission to skip the provider call.
            let valid = unary_gpu_output_matches_shape(
                &output,
                &current,
                self.provider,
                &shape,
                UnaryGpuOutputContract {
                    storage: GpuTensorStorage::Real,
                    precision: Some(precision),
                    integer: None,
                    logical: false,
                    alias: GpuOutputAliasPolicy::AllowInput,
                },
            ) && runmat_accelerate_api::handle_is_transposed(&output)
                == runmat_accelerate_api::handle_is_transposed(&current)
                && scratch
                    .iter()
                    .chain(published.iter().copied())
                    .chain([&self.input])
                    .all(|handle| {
                        handle.shares_allocation_with(&current)
                            || !output.shares_allocation_with(handle)
                    });
            if !valid {
                return Err(super::super::image_output::reject_untrusted_output(
                    output,
                    std::iter::once(&current)
                        .chain(scratch.iter())
                        .chain(published.iter().copied())
                        .chain([&self.input]),
                    "selected moment conversion returned malformed output",
                ));
            }
            let output = output
                .with_provenance(intent)
                .map_err(|cause| ProviderExecutionError::from_source(cause).after_effect())?;
            current = output.clone();
            // Retained references, including certified aliases, are released
            // only by the stage transaction. The final/public reference keeps
            // the common physical lease live through acknowledgement.
            scratch.push(output);
        }
        Ok(current)
    }
}
