//! Test provider's selected paired POWER geometry and completion. The test
//! buffers contain zero; a positive scalar exponent therefore has a real
//! result. These receipts exercise binding and lifetime, not WGPU timing.

use super::*;
use runmat_accelerate_api::{
    power_service::{
        PowerEvaluationPhysicalProposal, PowerEvaluationRecipe, PowerPairPhysicalProjection,
        PowerProbeObserved, PowerProbePhysicalProposal, PowerProbeRecipe,
    },
    PowerInputDescription, ProviderElementType, ProviderExecutedRealElementwise,
    ProviderExecutionError, ProviderLayout, ProviderRealElementwiseObserved,
    ProviderRealElementwiseOutputRoute, ProviderRealElementwisePhysicalProposal, ProviderResidency,
    ProviderStorage,
};

fn representation(handle: &GpuTensorHandle) -> Option<ProviderRepresentation> {
    ProviderFeasibilityQuery::exact_numeric_download(handle)?
        .inputs
        .into_iter()
        .next()
}

fn physical_pair(
    inputs: &[ProviderRepresentation; 2],
    descriptions: &[PowerInputDescription; 2],
) -> Option<PowerPairPhysicalProjection> {
    for (input, description) in inputs.iter().zip(descriptions) {
        if input.storage != ProviderStorage::DenseReal
            || input.layout != ProviderLayout::ColumnMajorContiguous
            || input.residency != ProviderResidency::Device
            || !matches!(
                input.element_type,
                ProviderElementType::F32 | ProviderElementType::F64
            )
            || input.checked_byte_len()? != description.requested_bytes()
        {
            return None;
        }
    }
    if inputs[0].element_type != inputs[1].element_type {
        return None;
    }
    let input_shapes = inputs.each_ref().map(|input| {
        input
            .shape
            .iter()
            .copied()
            .map(usize::try_from)
            .collect::<Result<Vec<_>, _>>()
            .ok()
    });
    let output_shape = runmat_types::geometry::broadcast_concrete_shape(
        input_shapes[0].as_ref()?,
        input_shapes[1].as_ref()?,
    )
    .ok()?;
    if output_shape.contains(&0) {
        return None;
    }
    let shape = output_shape
        .iter()
        .copied()
        .map(u64::try_from)
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    let output = |domain| {
        let (element_type, storage) = match (inputs[0].element_type, domain) {
            (ProviderElementType::F32, PowerResultDomain::Real) => {
                (ProviderElementType::F32, ProviderStorage::DenseReal)
            }
            (ProviderElementType::F64, PowerResultDomain::Real) => {
                (ProviderElementType::F64, ProviderStorage::DenseReal)
            }
            (ProviderElementType::F32, PowerResultDomain::Complex) => (
                ProviderElementType::ComplexF32,
                ProviderStorage::DenseComplexInterleaved,
            ),
            (ProviderElementType::F64, PowerResultDomain::Complex) => (
                ProviderElementType::ComplexF64,
                ProviderStorage::DenseComplexInterleaved,
            ),
            _ => return None,
        };
        let representation = ProviderRepresentation {
            element_type,
            storage,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape: shape.clone(),
            residency: ProviderResidency::Device,
        };
        Some(PowerEvaluationPhysicalProposal {
            work: ProviderRealElementwisePhysicalProposal {
                output_route: ProviderRealElementwiseOutputRoute::FreshAllocation,
                output_buffer_bytes: representation.checked_byte_len()?,
                workgroup_size: 64,
                chunk_capacity: 64,
                uniform_allocations: 0,
                uniform_buffer_bytes: 0,
                dispatches: 1,
                queue_submissions: 1,
                pipeline_key: Some(72),
            },
            output: representation,
            shader_identity: runmat_execution::Digest::sha256(b"staged-test-power-evaluation-v1"),
        })
    };
    let input_bytes = inputs[0]
        .checked_byte_len()?
        .checked_add(inputs[1].checked_byte_len()?)?;
    Some(PowerPairPhysicalProjection {
        probe: PowerProbePhysicalProposal {
            output_shape,
            snapshot_copy: false,
            shader_identity: runmat_execution::Digest::sha256(b"staged-test-power-probe-v1"),
            flag_pipeline_key: 72,
            phase_data_limits: [input_bytes; 4],
            dispatches: 1,
            queue_submissions: 1,
            host_workspace_bytes: 0,
        },
        real: output(PowerResultDomain::Real)?,
        complex: output(PowerResultDomain::Complex)?,
    })
}

impl StagedProvider {
    pub(super) fn staged_numeric_upload_evidence(
        &self,
        query: &ProviderFeasibilityQuery,
    ) -> Option<runmat_accelerate_api::ProviderNumericUploadPhysicalEvidence> {
        if query.operation.0 != "transfer.upload"
            || query.family != ProviderOperationFamily::Upload
            || query.inputs.len() != 1
            || query.outputs.len() != 1
            || query.inputs[0].residency != ProviderResidency::Host
            || query.outputs[0].residency != ProviderResidency::Device
            || query.inputs[0].element_type != query.outputs[0].element_type
            || query.inputs[0].shape != query.outputs[0].shape
        {
            return None;
        }
        Some(
            runmat_accelerate_api::ProviderNumericUploadPhysicalEvidence {
                output_requested_bytes: query.outputs[0].checked_byte_len()?,
                provider_scratch_bytes: runmat_execution::ResourceByteBound::Known(0),
                completion_queue_slots: 1,
            },
        )
    }

    pub(super) fn staged_floating_cast_evidence(
        &self,
        input: &ProviderRepresentation,
        target: ProviderPrecision,
    ) -> Option<runmat_accelerate_api::ProviderFloatingCastPhysicalEvidence> {
        let (source, lanes) = match (input.element_type, input.storage) {
            (ProviderElementType::F32, ProviderStorage::DenseReal) => (ProviderPrecision::F32, 1),
            (ProviderElementType::F64, ProviderStorage::DenseReal) => (ProviderPrecision::F64, 1),
            (ProviderElementType::ComplexF32, ProviderStorage::DenseComplexInterleaved) => {
                (ProviderPrecision::F32, 2)
            }
            (ProviderElementType::ComplexF64, ProviderStorage::DenseComplexInterleaved) => {
                (ProviderPrecision::F64, 2)
            }
            _ => return None,
        };
        if source == target
            || input.layout != ProviderLayout::ColumnMajorContiguous
            || input.residency != ProviderResidency::Device
        {
            return None;
        }
        Some(
            runmat_accelerate_api::ProviderFloatingCastPhysicalEvidence {
                output_requested_bytes: input
                    .checked_element_count()?
                    .checked_mul(if target == ProviderPrecision::F32 {
                        4
                    } else {
                        8
                    })?
                    .checked_mul(lanes)?,
                uniform_buffer_bytes: 0,
                dispatches: 1,
                queue_slots: 1,
            },
        )
    }

    pub(super) fn staged_power_pair_geometry(
        &self,
        inputs: &[ProviderRepresentation; 2],
        descriptions: &[PowerInputDescription; 2],
        frozen: Option<&PowerProbePhysicalProposal>,
    ) -> ProviderExecutionResult<Option<PowerPairPhysicalProjection>> {
        Ok(physical_pair(inputs, descriptions)
            .filter(|physical| frozen.is_none_or(|expected| expected == &physical.probe)))
    }

    pub(super) fn staged_power_probe(
        &self,
        base: &GpuTensorHandle,
        exponent: &GpuTensorHandle,
        frozen: Option<&PowerProbePhysicalProposal>,
    ) -> ProviderExecutionResult<Option<PowerProbePhysicalProposal>> {
        let (Some(base_rep), Some(exponent_rep)) = (representation(base), representation(exponent))
        else {
            return Ok(None);
        };
        let descriptions = [base, exponent].map(|handle| PowerInputDescription::Live {
            physical: *handle.physical(),
            view: handle.view().clone(),
        });
        Ok(self
            .staged_power_pair_geometry(&[base_rep, exponent_rep], &descriptions, frozen)?
            .map(|physical| physical.probe))
    }

    pub(super) fn staged_power_probe_execute(
        &self,
        recipe: &PowerProbeRecipe,
    ) -> ProviderExecutionResult<PowerProbeObserved> {
        let [base, exponent] = recipe.sources();
        if !recipe.is_owned_by(self)
            || self
                .staged_power_probe(base, exponent, Some(recipe.physical()))?
                .as_ref()
                != Some(recipe.physical())
        {
            return Err(ProviderExecutionError::from(anyhow!("POWER probe changed")));
        }
        Ok(PowerProbeObserved {
            domain: PowerResultDomain::Real,
            dispatches: recipe.physical().dispatches,
            queue_submissions: recipe.physical().queue_submissions,
        })
    }

    pub(super) fn staged_power_evaluation(
        &self,
        base: &GpuTensorHandle,
        exponent: &GpuTensorHandle,
        domain: PowerResultDomain,
    ) -> ProviderExecutionResult<Option<PowerEvaluationPhysicalProposal>> {
        let (Some(base_rep), Some(exponent_rep)) = (representation(base), representation(exponent))
        else {
            return Ok(None);
        };
        let descriptions = [base, exponent].map(|handle| PowerInputDescription::Live {
            physical: *handle.physical(),
            view: handle.view().clone(),
        });
        Ok(self
            .staged_power_pair_geometry(&[base_rep, exponent_rep], &descriptions, None)?
            .map(|physical| match domain {
                PowerResultDomain::Real => physical.real,
                PowerResultDomain::Complex => physical.complex,
            }))
    }

    pub(super) fn staged_power_evaluation_execute(
        &self,
        recipe: &PowerEvaluationRecipe,
    ) -> ProviderExecutionResult<ProviderExecutedRealElementwise> {
        let [base, exponent] = recipe.probe().sources();
        if !recipe.probe().is_owned_by(self)
            || self
                .staged_power_evaluation(base, exponent, recipe.domain())?
                .as_ref()
                != Some(recipe.physical())
        {
            return Err(ProviderExecutionError::from(anyhow!(
                "POWER evaluation changed"
            )));
        }
        if recipe.domain() != PowerResultDomain::Real {
            return Err(ProviderExecutionError::from(anyhow!(
                "zero-backed staged POWER fixture has no complex execution"
            )));
        }
        let output = self
            .produce_class(
                recipe
                    .physical()
                    .output
                    .shape
                    .iter()
                    .copied()
                    .map(usize::try_from)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(ProviderExecutionError::from_source)?,
                StageHook::Pow,
                base.physical().element_type(),
            )
            .map_err(ProviderExecutionError::from)?;
        let work = &recipe.physical().work;
        Ok(ProviderExecutedRealElementwise {
            output,
            observed: ProviderRealElementwiseObserved {
                output_buffer_bytes: Some(work.output_buffer_bytes),
                uniform_allocations: work.uniform_allocations,
                uniform_buffer_bytes: work.uniform_buffer_bytes,
                dispatches: work.dispatches,
                queue_submissions: work.queue_submissions,
            },
        })
    }
}
