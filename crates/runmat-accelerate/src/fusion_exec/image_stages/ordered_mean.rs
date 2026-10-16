//! Selected one-call mean recipes. The provider owns Double accumulation and
//! physical derivation; this owner binds them to exact image source calls.

use super::*;
use crate::fusion::image_reduction::{ImageSelectedMomentBindings, ImageSelectedMomentConversion};
use runmat_accelerate_api::{
    moment_precision, PowerResultDomain, ProviderElementType, ProviderImageMomentPublication,
    ProviderLayout, ProviderOrderedRealMeanRecipe, ProviderRealMeanNaNMode, ProviderRepresentation,
    ProviderResidency, ProviderStorage,
};
use runmat_runtime::builtins::common::gpu_helpers::{
    unary_gpu_output_matches_shape, GpuOutputAliasPolicy, UnaryGpuOutputContract,
};

pub(super) mod casts;
pub(super) use casts::SelectedMomentCastPlan;
mod sigma;
pub(super) use sigma::SelectedSigmaPublicationPlan;

pub(super) struct OrderedMeanCallPlan {
    pub source_input: u64,
    pub raw_output: u64,
    pub recipe: ProviderOrderedRealMeanRecipe,
    pub casts: Vec<casts::SelectedMomentCastPlan>,
}

pub(super) struct ImageOrderedMeanPlans {
    pub mu: Vec<OrderedMeanCallPlan>,
    pub variance: Vec<OrderedMeanCallPlan>,
    pub sigma: Option<SelectedSigmaPublicationPlan>,
}

impl ImageOrderedMeanPlans {
    pub(super) fn physical_upper_bound(
        &self,
    ) -> Option<super::resources::ImageStagePhysicalUpperBound> {
        let mut bound = super::resources::ImageStagePhysicalUpperBound::default();
        for call in self.mu.iter().chain(&self.variance) {
            bound.add(
                call.recipe.requested_incremental_peak_bytes(),
                u32::try_from(call.recipe.steps().len()).ok()?,
            )?;
            for cast in &call.casts {
                // The cast feasibility peak includes its already-live input.
                // This aggregate charges only the additional allocation and
                // uniforms; the selected source remains live throughout.
                bound.add(
                    cast.resources
                        .transient_bytes?
                        .checked_sub(cast.source_bytes)?,
                    cast.resources.dispatches?,
                )?;
            }
        }
        let sigma = self.sigma.as_ref()?;
        let (add, sqrt) = sigma.operations.as_ref()?.proposals();
        for proposal in [add, sqrt] {
            bound.add(
                proposal.requested_incremental_peak_bytes()?,
                proposal.dispatches,
            )?;
        }
        for cast in &sigma.casts {
            bound.add(
                cast.resources
                    .transient_bytes?
                    .checked_sub(cast.source_bytes)?,
                cast.resources.dispatches?,
            )?;
        }
        Some(bound)
    }

    pub(super) fn public_mean_carrier(&self) -> Option<(ProviderRepresentation, u64)> {
        self.mu.last().and_then(sigma::final_representation)
    }

    pub(super) fn public_sigma_carrier(&self) -> Option<(ProviderRepresentation, u64)> {
        self.sigma
            .as_ref()
            .and_then(SelectedSigmaPublicationPlan::public_carrier)
    }
}

pub(super) fn representation(
    precision: ProviderPrecision,
    shape: &[usize],
) -> Option<ProviderRepresentation> {
    Some(ProviderRepresentation {
        element_type: match precision {
            ProviderPrecision::F32 => ProviderElementType::F32,
            ProviderPrecision::F64 => ProviderElementType::F64,
        },
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape: shape
            .iter()
            .map(|&extent| u64::try_from(extent).ok())
            .collect::<Option<_>>()?,
        residency: ProviderResidency::Device,
    })
}

#[cfg(test)]
pub(super) fn representation_for_test(
    source: &GpuTensorHandle,
) -> anyhow::Result<ProviderRepresentation> {
    let precision = runmat_accelerate_api::handle_precision(source)
        .ok_or_else(|| anyhow!("test ordered mean source has no real precision"))?;
    representation(precision, source.shape())
        .ok_or_else(|| anyhow!("test ordered mean source shape overflows"))
}

fn one_call(
    provider: &'static dyn AccelProvider,
    source: &ProviderRepresentation,
    source_bytes: u64,
    call: &ProviderImageMeanCall,
    publication: &ProviderImageMomentPublication,
    bindings: Option<&[ImageSelectedMomentConversion]>,
    semantic_identity: Option<runmat_execution::Digest>,
) -> ProviderExecutionResult<Option<OrderedMeanCallPlan>> {
    let axes = match call.hook {
        ProviderImageMeanHook::ReduceNd(axes) => axes.map(usize::from).to_vec(),
        ProviderImageMeanHook::ReduceDim(axis) => vec![usize::from(axis)],
    };
    let Some(output_precision) = moment_precision(&publication.raw_fact) else {
        return Ok(None);
    };
    let input_precision = match source.element_type {
        ProviderElementType::F32 => ProviderPrecision::F32,
        ProviderElementType::F64 => ProviderPrecision::F64,
        _ => return Ok(None),
    };
    // Current Default and Native mean forms both publish the source class;
    // provider preference never changes the language-level output class.
    if output_precision != input_precision {
        return Ok(None);
    }
    let Some(recipe) = provider
        .plan_ordered_real_mean(
            source,
            source_bytes,
            &axes,
            output_precision,
            ProviderRealMeanNaNMode::Include,
        )
        .map_err(ProviderExecutionError::from)?
    else {
        return Ok(None);
    };
    let expected = call
        .shape
        .iter()
        .map(|&extent| usize::try_from(extent).ok())
        .collect::<Option<Vec<_>>>();
    if recipe.source() != source
        || recipe.source_buffer_bytes() != source_bytes
        || recipe.axes() != axes
        || recipe.nan_mode() != ProviderRealMeanNaNMode::Include
        || recipe.output_precision() != output_precision
        || recipe.steps().last().map(|step| step.output_shape()) != expected.as_deref()
        || publication
            .raw_fact
            .shape
            .known_dims()
            .and_then(|dims| dims.into_iter().collect::<Option<Vec<_>>>())
            != expected
    {
        return Ok(None);
    }
    if publication.conversions.len() != bindings.map_or(0, |bindings| bindings.len()) {
        return Ok(None);
    }
    let Some(final_step) = recipe.steps().last() else {
        return Ok(None);
    };
    let mut casts = Vec::new();
    let Some(mut cast_source) = representation(output_precision, final_step.output_shape()) else {
        return Ok(None);
    };
    let mut cast_source_bytes = final_step.output_bytes();
    for (cast, binding) in publication.conversions.iter().zip(bindings.unwrap_or(&[])) {
        let Some(identity) = semantic_identity else {
            return Ok(None);
        };
        let Some(selected) = casts::plan(
            provider,
            &cast_source,
            cast_source_bytes,
            cast,
            binding,
            identity,
        ) else {
            return Ok(None);
        };
        cast_source = selected.output.clone();
        cast_source_bytes = selected.output_bytes;
        casts.push(selected);
    }
    Ok(Some(OrderedMeanCallPlan {
        source_input: publication.source_input,
        raw_output: publication.raw_output,
        recipe,
        casts,
    }))
}

fn route(
    provider: &'static dyn AccelProvider,
    initial: ProviderRepresentation,
    initial_bytes: u64,
    calls: &[ProviderImageMeanCall],
    publications: &[ProviderImageMomentPublication],
    bindings: Option<&[Vec<ImageSelectedMomentConversion>]>,
    semantic_identity: Option<runmat_execution::Digest>,
) -> ProviderExecutionResult<Option<Vec<OrderedMeanCallPlan>>> {
    if calls.is_empty()
        || calls.len() != publications.len()
        || bindings.is_some_and(|bindings| bindings.len() != calls.len())
    {
        return Ok(None);
    }
    let mut source = initial;
    let mut source_bytes = initial_bytes;
    let mut planned = Vec::with_capacity(calls.len());
    for (index, (call, publication)) in calls.iter().zip(publications).enumerate() {
        if index > 0 && publication.source_input != publications[index - 1].public_output {
            return Ok(None);
        }
        let Some(next) = one_call(
            provider,
            &source,
            source_bytes,
            call,
            publication,
            bindings.map(|bindings| bindings[index].as_slice()),
            semantic_identity,
        )?
        else {
            return Ok(None);
        };
        let Some(final_step) = next.recipe.steps().last() else {
            return Ok(None);
        };
        source = representation(final_step.output_precision(), final_step.output_shape())
            .ok_or_else(|| ProviderExecutionError::from(anyhow!("ordered mean shape overflows")))?;
        source_bytes = final_step.output_bytes();
        if let Some(cast) = next.casts.last() {
            source = cast.output.clone();
            source_bytes = cast.output_bytes;
        }
        planned.push(next);
    }
    Ok(Some(planned))
}

pub(super) fn plan_selected(
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
    program: &ProviderImageStageWorkload,
    powers: &power::ImagePowerPlans,
    bindings: Option<&ImageSelectedMomentBindings>,
) -> ProviderExecutionResult<Option<ImageOrderedMeanPlans>> {
    let Some(moments) = program.moment_program.as_ref() else {
        return Ok(None);
    };
    if !moments.validate() {
        return Ok(None);
    }
    let Some(selected_powers) = program.selected_powers.as_ref() else {
        return Ok(None);
    };
    let Some(input_precision) = runmat_accelerate_api::handle_precision(input) else {
        return Ok(None);
    };
    let Some(input_rep) = representation(input_precision, input.shape()) else {
        return Ok(None);
    };
    let Some(mu) = route(
        provider,
        input_rep,
        input.physical().requested_bytes(),
        &program.mu,
        &moments.mu,
        bindings.map(ImageSelectedMomentBindings::mu),
        bindings.map(ImageSelectedMomentBindings::semantic_identity),
    )?
    else {
        return Ok(None);
    };
    let Some(prepared_mu) = provider
        .prepare_ordered_real_mean_against_recipe(input, &mu[0].recipe)
        .map_err(ProviderExecutionError::from)?
    else {
        return Ok(None);
    };
    if !prepared_mu
        .validate_live_source(input)
        .map_err(ProviderExecutionError::from)?
        || prepared_mu.recipe() != &mu[0].recipe
    {
        return Ok(None);
    }
    let Some(square_work) = powers.square.physical_work(PowerResultDomain::Real) else {
        return Ok(None);
    };
    let Some(square_bytes) =
        provider.plan_power_output_buffer_bytes(&powers.square, PowerResultDomain::Real)?
    else {
        return Ok(None);
    };
    // The variance source is the future POWER allocation, not just its
    // logical shape. Preserve the exact proposal's allocator-rounded extent.
    if square_work
        .output
        .checked_byte_len()
        .is_none_or(|logical| logical > square_bytes)
        || powers
            .square
            .exact_physical_proposal()
            .is_some_and(|proposal| proposal.output_buffer_bytes != square_bytes)
    {
        return Ok(None);
    }
    let Some(variance) = route(
        provider,
        square_work.output.clone(),
        square_bytes,
        &program.variance,
        &moments.variance,
        bindings.map(ImageSelectedMomentBindings::variance),
        bindings.map(ImageSelectedMomentBindings::semantic_identity),
    )?
    else {
        return Ok(None);
    };
    if variance[0].source_input != u64::from(selected_powers.square.output) {
        return Ok(None);
    }
    let sigma = if let Some(bindings) = bindings {
        let Some(sigma) = sigma::plan(provider, &variance, &moments.sigma, bindings)? else {
            return Ok(None);
        };
        Some(sigma)
    } else {
        None
    };
    Ok(Some(ImageOrderedMeanPlans {
        mu,
        variance,
        sigma,
    }))
}

pub(super) struct OrderedMeanExecutionContext<'a> {
    pub current: &'a GpuTensorHandle,
    pub image_input: &'a GpuTensorHandle,
    pub scratch: &'a [GpuTensorHandle],
    pub published: &'a [&'a GpuTensorHandle],
    pub prior_effect: bool,
}

pub(super) fn execute_call(
    provider: &'static dyn AccelProvider,
    context: OrderedMeanExecutionContext<'_>,
    call: &ProviderImageMeanCall,
    publication: &ProviderImageMomentPublication,
    planned: &OrderedMeanCallPlan,
) -> ProviderExecutionResult<GpuTensorHandle> {
    let OrderedMeanExecutionContext {
        current,
        image_input,
        scratch,
        published,
        prior_effect,
    } = context;
    let axes = match call.hook {
        ProviderImageMeanHook::ReduceNd(axes) => axes.map(usize::from).to_vec(),
        ProviderImageMeanHook::ReduceDim(axis) => vec![usize::from(axis)],
    };
    if planned.source_input != publication.source_input
        || planned.raw_output != publication.raw_output
        || planned.recipe.axes() != axes
        || planned.recipe.nan_mode() != ProviderRealMeanNaNMode::Include
        || moment_precision(&publication.raw_fact) != Some(planned.recipe.output_precision())
    {
        return Err(ProviderExecutionError::from(anyhow!(
            "selected mean source call changed after admission"
        ))
        .after_effect());
    }
    let selected = provider
        .prepare_ordered_real_mean_against_recipe(current, &planned.recipe)
        .map_err(|source| {
            let error = ProviderExecutionError::from(source);
            if prior_effect {
                error.after_effect()
            } else {
                error
            }
        })?
        .ok_or_else(|| {
            ProviderExecutionError::from(anyhow!("ordered mean live view differs from recipe"))
                .after_effect()
        })?;
    if !selected.validate_live_source(current).map_err(|source| {
        let error = ProviderExecutionError::from(source);
        if prior_effect {
            error.after_effect()
        } else {
            error
        }
    })? || selected.recipe() != &planned.recipe
    {
        return Err(ProviderExecutionError::from(anyhow!(
            "ordered mean source lease or recipe changed"
        ))
        .after_effect());
    }
    for borrowed in std::iter::once(current)
        .chain(std::iter::once(image_input))
        .chain(scratch.iter())
        .chain(published.iter().copied())
    {
        borrowed
            .allocation_lease()
            .validate_live()
            .map_err(|source| {
                let error = ProviderExecutionError::from(source);
                if prior_effect {
                    error.after_effect()
                } else {
                    error
                }
            })?;
    }
    let executed = provider
        .execute_ordered_real_mean(selected)
        .map_err(ProviderExecutionError::after_effect)?;
    for borrowed in std::iter::once(current)
        .chain(std::iter::once(image_input))
        .chain(scratch.iter())
        .chain(published.iter().copied())
    {
        borrowed
            .allocation_lease()
            .validate_live()
            .map_err(|source| ProviderExecutionError::from(source).after_effect())?;
    }
    let Some(final_step) = planned.recipe.steps().last() else {
        return Err(super::super::image_output::reject_untrusted_output(
            executed.output,
            std::iter::once(current)
                .chain([image_input])
                .chain(scratch.iter())
                .chain(published.iter().copied()),
            "ordered mean selected recipe has no final step",
        ));
    };
    let expected_submissions = u32::try_from(planned.recipe.steps().len()).ok();
    let borrowed_alias = executed.output.shares_allocation_with(current)
        || executed.output.shares_allocation_with(image_input)
        || scratch
            .iter()
            .chain(published.iter().copied())
            .any(|borrowed| executed.output.shares_allocation_with(borrowed));
    let valid = executed.requested_incremental_peak_bytes
        == planned.recipe.requested_incremental_peak_bytes()
        && executed.observed_incremental_peak_bytes
            == planned.recipe.requested_incremental_peak_bytes()
        && Some(executed.selected_submissions) == expected_submissions
        && executed.observed_output_bytes == final_step.output_bytes()
        && executed.observed_output_bytes == executed.output.physical().requested_bytes()
        && !borrowed_alias
        && unary_gpu_output_matches_shape(
            &executed.output,
            current,
            provider,
            final_step.output_shape(),
            UnaryGpuOutputContract {
                storage: GpuTensorStorage::Real,
                precision: Some(final_step.output_precision()),
                integer: None,
                logical: false,
                alias: GpuOutputAliasPolicy::RequireDistinctAllocation,
            },
        )
        && !runmat_accelerate_api::handle_is_transposed(&executed.output);
    if !valid {
        if borrowed_alias {
            // A forged borrowed alias must not be retired by result cleanup.
            return Err(ProviderExecutionError::from(anyhow!(
                "ordered mean returned a protected allocation"
            ))
            .after_effect());
        }
        return Err(super::super::outputs::reject_outputs(
            vec![executed.output],
            "ordered mean physical completion changed",
        ));
    }
    Ok(executed.output)
}
