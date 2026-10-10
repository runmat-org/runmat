//! Selected Sigma's direct SQRT output and public cast are separate proposals.
//! This joins the sealed source SQRT node to the ordinary output allocator's
//! extent and the existing canonical floating-conversion physical query.

use super::*;
use crate::fusion::image_reduction::{ImageSelectedMomentBindings, ImageSelectedSigmaRaw};
use runmat_accelerate_api::ProviderImageMomentPublication;
use runmat_execution::Digest;

pub(crate) struct SelectedSigmaPublicationPlan {
    raw_source: ImageSelectedSigmaRaw,
    source: ProviderRepresentation,
    source_bytes: u64,
    raw_output_bytes: u64,
    pub(crate) casts: Vec<SelectedMomentCastPlan>,
    pub(crate) operations: Option<super::super::sigma_operations::SelectedSigmaOperationsPlan>,
}

pub(super) fn final_representation(
    call: &OrderedMeanCallPlan,
) -> Option<(ProviderRepresentation, u64)> {
    if let Some(cast) = call.casts.last() {
        return Some((cast.output.clone(), cast.output_bytes));
    }
    let step = call.recipe.steps().last()?;
    Some((
        representation(step.output_precision(), step.output_shape())?,
        step.output_bytes(),
    ))
}

pub(super) fn plan(
    provider: &'static dyn AccelProvider,
    variance: &[OrderedMeanCallPlan],
    publication: &ProviderImageMomentPublication,
    bindings: &ImageSelectedMomentBindings,
) -> ProviderExecutionResult<Option<SelectedSigmaPublicationPlan>> {
    let raw_source = bindings.sigma_raw();
    if raw_source.semantic_identity() != bindings.semantic_identity()
        || !raw_source.matches_publication(publication)
        || publication.conversions.len() != bindings.sigma().len()
    {
        return Ok(None);
    }
    let Some((source, source_bytes)) = variance.last().and_then(final_representation) else {
        return Ok(None);
    };
    let Some(raw_precision) = moment_precision(&publication.raw_fact) else {
        return Ok(None);
    };
    let Some(raw_representation) = representation(
        raw_precision,
        &publication
            .raw_fact
            .shape
            .known_dims()
            .and_then(|dims| dims.into_iter().collect::<Option<Vec<_>>>())
            .ok_or_else(|| ProviderExecutionError::from(anyhow!("Sigma shape is unproved")))?,
    ) else {
        return Ok(None);
    };
    if source != raw_representation {
        return Ok(None);
    }
    let Some(raw_output_bytes) = provider.plan_dense_real_output_buffer_bytes(&source)? else {
        return Ok(None);
    };
    // Scalar Add and unary SQRT retain the dense class/shape, so their actual
    // output allocations must use this exact ordinary numeric allocator key.
    // Their kernel/submission resources are NOT certified by this proposal.
    if raw_output_bytes != source_bytes {
        return Ok(None);
    }
    let mut cast_source = raw_representation;
    let mut cast_source_bytes = raw_output_bytes;
    let mut casts = Vec::with_capacity(publication.conversions.len());
    for (call, binding) in publication.conversions.iter().zip(bindings.sigma()) {
        let Some(selected) = casts::plan(
            provider,
            &cast_source,
            cast_source_bytes,
            call,
            binding,
            bindings.semantic_identity(),
        ) else {
            return Ok(None);
        };
        cast_source = selected.output.clone();
        cast_source_bytes = selected.output_bytes;
        casts.push(selected);
    }
    Ok(Some(SelectedSigmaPublicationPlan {
        raw_source,
        source,
        source_bytes,
        raw_output_bytes,
        casts,
        operations: None,
    }))
}

impl SelectedSigmaPublicationPlan {
    pub(super) fn public_carrier(&self) -> Option<(ProviderRepresentation, u64)> {
        if let Some(cast) = self.casts.last() {
            Some((cast.output.clone(), cast.output_bytes))
        } else {
            Some((self.source.clone(), self.raw_output_bytes))
        }
    }
    pub(crate) fn variance_carrier(&self) -> (&ProviderRepresentation, u64) {
        (&self.source, self.source_bytes)
    }
    pub(crate) fn raw_output_bytes(&self) -> u64 {
        self.raw_output_bytes
    }
    pub(crate) fn revalidate_source(
        &self,
        provider: &'static dyn AccelProvider,
        source: &GpuTensorHandle,
    ) -> ProviderExecutionResult<bool> {
        source
            .allocation_lease()
            .validate_live()
            .map_err(ProviderExecutionError::from_source)?;
        let Some(precision) = runmat_accelerate_api::handle_precision(source) else {
            return Ok(false);
        };
        let owner =
            runmat_runtime::builtins::common::gpu_helpers::exact_provider_for_handle(source);
        if owner.is_none_or(|owner| !runmat_accelerate_api::same_provider_instance(owner, provider))
            || runmat_accelerate_api::handle_is_logical(source)
            || runmat_accelerate_api::handle_is_transposed(source)
            || representation(precision, source.shape()).as_ref() != Some(&self.source)
            || source.physical().requested_bytes() != self.source_bytes
        {
            return Ok(false);
        }
        Ok(provider.plan_dense_real_output_buffer_bytes(&self.source)?
            == Some(self.raw_output_bytes))
    }

    pub(crate) fn revalidate_raw(
        &self,
        provider: &'static dyn AccelProvider,
        raw: &GpuTensorHandle,
    ) -> ProviderExecutionResult<bool> {
        raw.allocation_lease()
            .validate_live()
            .map_err(ProviderExecutionError::from_source)?;
        let Some(precision) = runmat_accelerate_api::handle_precision(raw) else {
            return Ok(false);
        };
        let owner = runmat_runtime::builtins::common::gpu_helpers::exact_provider_for_handle(raw);
        Ok(owner
            .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
            && !runmat_accelerate_api::handle_is_logical(raw)
            && !runmat_accelerate_api::handle_is_transposed(raw)
            && representation(precision, raw.shape()).as_ref() == Some(&self.source)
            && raw.physical().requested_bytes() == self.raw_output_bytes
            && provider.plan_dense_real_output_buffer_bytes(&self.source)?
                == Some(self.raw_output_bytes))
    }

    pub(crate) fn raw_source_identity(&self) -> (u64, u64, Digest) {
        (
            self.raw_source.source_input(),
            self.raw_source.raw_output(),
            self.raw_source.semantic_identity(),
        )
    }
}
