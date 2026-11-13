//! Exact image recipes before a source is physically bound. Host inputs may
//! be considered by placement without manufacturing a future GPU handle.

use super::*;
use runmat_accelerate_api::{ProviderElementType, ProviderRepresentation};

pub(super) struct ImageSelectedStagePlan {
    pub provider: &'static dyn AccelProvider,
    pub desc: ImageNormalizeDescriptor,
    pub program: ProviderImageStageWorkload,
    pub input_rep: ProviderRepresentation,
    pub input_bytes: u64,
    pub powers: power::ImagePowerPlans,
    pub means: ordered_mean::ImageOrderedMeanPlans,
    pub binary: selected_binary::SelectedBinaryPlans,
    pub real: selected_real_output::SelectedRealPlans,
}

impl ImageSelectedStagePlan {
    pub(super) fn new(
        provider: &'static dyn AccelProvider,
        input: &Value,
        input_rep: ProviderRepresentation,
        input_bytes: u64,
        invocation: &ImageSourceInvocation,
    ) -> ProviderExecutionResult<Option<Self>> {
        if !invocation.selected_stage_covers_source() {
            return Ok(None);
        }
        let Some((desc, source)) = invocation.selected_staged_mean_power_for_value(input) else {
            return Ok(None);
        };
        let Some(frozen) = invocation.requested_staged_workload() else {
            return Ok(None);
        };
        let Some(selected_powers) = frozen.selected_powers.clone() else {
            return Ok(None);
        };
        let Some(sigma) = invocation.selected_sigma_operations() else {
            return Ok(None);
        };
        let Some(output_program) = invocation.requested_output_program() else {
            return Ok(None);
        };
        let Some(mut expected) =
            source.staged_workload_selected(desc, selected_powers, sigma.control())
        else {
            return Ok(None);
        };
        expected.moment_program = frozen.moment_program.clone();
        expected.output_program = Some(output_program.clone());
        if &expected != frozen
            || !frozen.validate()
            || frozen.moment_program.is_none()
            || input_rep.element_type != ProviderElementType::F32
            || input_rep.storage != runmat_accelerate_api::ProviderStorage::DenseReal
            || input_rep.layout != runmat_accelerate_api::ProviderLayout::ColumnMajorContiguous
            || input_rep.residency != runmat_accelerate_api::ProviderResidency::Device
            || input_rep.shape != frozen.input_shape
            || input_rep
                .checked_byte_len()
                .is_none_or(|bytes| bytes > input_bytes)
            || desc.epsilon <= 0.0
            || !desc.epsilon.is_finite()
        {
            return Ok(None);
        }
        let program = frozen.clone();
        let Some(powers) = power::plan_selected(provider, &program)? else {
            return Ok(None);
        };
        let Some(bindings) = invocation.selected_moment_bindings() else {
            return Ok(None);
        };
        let Some(mut means) = ordered_mean::plan_selected_source(
            provider,
            input_rep.clone(),
            input_bytes,
            &program,
            &powers,
            Some(&bindings),
        )?
        else {
            return Ok(None);
        };
        let Some((publication, moments)) =
            means.sigma.as_mut().zip(program.moment_program.as_ref())
        else {
            return Ok(None);
        };
        let Some(sigma_plan) =
            sigma_operations::plan(provider, &sigma, publication, moments, &bindings)?
        else {
            return Ok(None);
        };
        if !sigma_plan.matches_program(&program) {
            return Ok(None);
        }
        publication.operations = Some(sigma_plan);
        let Some(binary_calls) = invocation.selected_binary_calls() else {
            return Ok(None);
        };
        if binary_calls.semantic_identity() != invocation.semantic_identity() {
            return Ok(None);
        }
        let Some(binary) = selected_binary::plan_source(
            provider,
            input_rep.clone(),
            input_bytes,
            &binary_calls,
            &powers,
            &means,
            &program,
        )?
        else {
            return Ok(None);
        };
        let Some(real_calls) = invocation.selected_real_output_calls() else {
            return Ok(None);
        };
        if real_calls.semantic_identity() != invocation.semantic_identity() {
            return Ok(None);
        }
        let Some(real) =
            selected_real_output::plan(provider, &real_calls, &binary, &powers, &program)?
        else {
            return Ok(None);
        };
        Ok(Some(Self {
            provider,
            desc: desc.clone(),
            program,
            input_rep,
            input_bytes,
            powers,
            means,
            binary,
            real,
        }))
    }

    pub(super) fn physical_upper_bound(&self) -> Option<resources::ImageStagePhysicalUpperBound> {
        let mut bound = resources::ImageStagePhysicalUpperBound::default();
        bound.combine(self.means.physical_upper_bound()?)?;
        bound.combine(self.binary.physical_upper_bound()?)?;
        bound.combine(self.real.physical_upper_bound()?)?;
        bound.combine(self.powers.physical_upper_bound()?)?;
        Some(bound)
    }

    pub(super) fn registered_output_count(&self) -> Option<usize> {
        self.means
            .registered_output_count()?
            .checked_add(self.binary.registered_output_count())?
            .checked_add(self.real.registered_output_count())?
            .checked_add(self.powers.registered_output_count()?)
    }

    pub(super) fn alias_view_count(&self) -> Option<usize> {
        self.means
            .alias_view_count()
            .checked_add(self.real.alias_view_count())
    }

    pub(super) fn bind(
        self,
        input: &GpuTensorHandle,
    ) -> ProviderExecutionResult<Option<ImageNormalizeStageSession>> {
        input
            .allocation_lease()
            .validate_live()
            .map_err(ProviderExecutionError::from_source)?;
        if input.device_id() != self.provider.device_id()
            || !exact_provider_for_handle(input).is_some_and(|owner| {
                runmat_accelerate_api::same_provider_instance(owner, self.provider)
            })
            || input.physical().element_type() != NumericElementType::F32
            || input.physical().storage() != GpuTensorStorage::Real
            || runmat_accelerate_api::handle_is_logical(input)
            || runmat_accelerate_api::handle_is_transposed(input)
            || !gpu_class_metadata_matches(input, Some(ProviderPrecision::F32), None, false)
            || input.physical().requested_bytes() != self.input_bytes
            || ordered_mean::representation(ProviderPrecision::F32, input.shape())
                != Some(self.input_rep.clone())
            || !ordered_mean::validate_first_live_source(self.provider, input, &self.means)?
        {
            return Ok(None);
        }
        Ok(Some(ImageNormalizeStageSession {
            provider: self.provider,
            input: input.clone(),
            desc: self.desc,
            program: self.program,
            selected_power_plans: Some(self.powers),
            selected_mean_plans: Some(self.means),
            selected_binary_plans: Some(self.binary),
            selected_real_plans: Some(self.real),
            reserved_mu: None,
            cold_mu: None,
            retained_view_permit: None,
            provider_metadata_permit: None,
            state: State::BeforeMean,
        }))
    }
}
