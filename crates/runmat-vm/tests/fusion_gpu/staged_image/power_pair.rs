use super::*;
use runmat_accelerate_api::{
    power_service::{
        PowerEvaluationPhysicalProposal, PowerEvaluationRecipe, PowerPairPhysicalProjection,
        PowerProbeObserved, PowerProbePhysicalProposal, PowerProbeRecipe,
    },
    NumericElementType, PowerInputDescription, PowerResultDomain, ProviderElementType,
    ProviderExecutedRealElementwise, ProviderExecutionError, ProviderExecutionResult,
    ProviderLayout, ProviderRealElementwiseObserved, ProviderRealElementwiseOutputRoute,
    ProviderRealElementwisePhysicalProposal, ProviderRepresentation, ProviderResidency,
    ProviderStorage,
};
use runmat_runtime::builtins::common::broadcast::{broadcast_index, compute_strides};

fn shape(representation: &ProviderRepresentation) -> Option<Vec<usize>> {
    representation
        .shape
        .iter()
        .copied()
        .map(usize::try_from)
        .collect::<Result<_, _>>()
        .ok()
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
    let output_shape =
        runmat_types::geometry::broadcast_concrete_shape(&shape(&inputs[0])?, &shape(&inputs[1])?)
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
        let element_type = match (inputs[0].element_type, domain) {
            (ProviderElementType::F32, PowerResultDomain::Real) => ProviderElementType::F32,
            (ProviderElementType::F64, PowerResultDomain::Real) => ProviderElementType::F64,
            (ProviderElementType::F32, PowerResultDomain::Complex) => {
                ProviderElementType::ComplexF32
            }
            (ProviderElementType::F64, PowerResultDomain::Complex) => {
                ProviderElementType::ComplexF64
            }
            _ => return None,
        };
        let representation = ProviderRepresentation {
            element_type,
            storage: if domain == PowerResultDomain::Real {
                ProviderStorage::DenseReal
            } else {
                ProviderStorage::DenseComplexInterleaved
            },
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
                pipeline_key: Some(1),
            },
            output: representation,
            shader_identity: runmat_execution::Digest::sha256(b"vm-software-power-evaluation-v1"),
        })
    };
    let input_bytes = inputs[0]
        .checked_byte_len()?
        .checked_add(inputs[1].checked_byte_len()?)?;
    // The software probe reads each real input into an f64 snapshot and
    // retains both shape and stride vectors while deciding the domain.
    let probe_host_workspace_bytes = inputs.iter().try_fold(0_u64, |bytes, input| {
        let values = input.checked_element_count()?.checked_mul(8)?;
        let axes = u64::try_from(input.shape.len())
            .ok()?
            .checked_mul(u64::try_from(std::mem::size_of::<usize>()).ok()?)?
            .checked_mul(2)?;
        bytes.checked_add(values)?.checked_add(axes)
    })?;
    Some(PowerPairPhysicalProjection {
        probe: PowerProbePhysicalProposal {
            output_shape,
            snapshot_copy: false,
            shader_identity: runmat_execution::Digest::sha256(b"vm-software-power-probe-v1"),
            flag_pipeline_key: 1,
            phase_data_limits: [input_bytes; 4],
            dispatches: 1,
            queue_submissions: 1,
            host_workspace_bytes: probe_host_workspace_bytes,
        },
        real: output(PowerResultDomain::Real)?,
        complex: output(PowerResultDomain::Complex)?,
    })
}

impl TestProvider {
    pub(in super::super) fn staged_power_pair_geometry(
        &self,
        inputs: &[ProviderRepresentation; 2],
        descriptions: &[PowerInputDescription; 2],
        frozen: Option<&PowerProbePhysicalProposal>,
    ) -> ProviderExecutionResult<Option<PowerPairPhysicalProjection>> {
        let physical = physical_pair(inputs, descriptions);
        Ok(physical.filter(|physical| frozen.is_none_or(|frozen| frozen == &physical.probe)))
    }

    pub(in super::super) fn staged_power_probe(
        &self,
        base: &GpuTensorHandle,
        exponent: &GpuTensorHandle,
        frozen: Option<&PowerProbePhysicalProposal>,
    ) -> ProviderExecutionResult<Option<PowerProbePhysicalProposal>> {
        let [Some(base_rep), Some(exponent_rep)] = [base, exponent].map(|handle| {
            ProviderFeasibilityQuery::exact_numeric_download(handle)
                .and_then(|query| query.inputs.into_iter().next())
        }) else {
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

    pub(in super::super) fn staged_power_probe_execute(
        &self,
        recipe: &PowerProbeRecipe,
    ) -> ProviderExecutionResult<PowerProbeObserved> {
        if !recipe.is_owned_by(self) {
            return Err(ProviderExecutionError::from(anyhow!("foreign POWER probe")));
        }
        let [base, exponent] = recipe.sources();
        if self
            .staged_power_probe(base, exponent, Some(recipe.physical()))?
            .as_ref()
            != Some(recipe.physical())
        {
            return Err(ProviderExecutionError::from(anyhow!("POWER probe changed")));
        }
        let (bases, base_shape) = self.pull(base).map_err(ProviderExecutionError::from)?;
        let (exponents, exponent_shape) =
            self.pull(exponent).map_err(ProviderExecutionError::from)?;
        let output_shape = &recipe.physical().output_shape;
        let base_strides = compute_strides(&base_shape);
        let exponent_strides = compute_strides(&exponent_shape);
        let count = output_shape
            .iter()
            .try_fold(1usize, |count, &extent| count.checked_mul(extent))
            .ok_or_else(|| ProviderExecutionError::from(anyhow!("POWER extent overflow")))?;
        let mut domain = PowerResultDomain::Real;
        for index in 0..count {
            let lhs = bases[broadcast_index(index, output_shape, &base_shape, &base_strides)];
            let rhs =
                exponents[broadcast_index(index, output_shape, &exponent_shape, &exponent_strides)];
            domain = domain.join(PowerResultDomain::for_real_pair(lhs, rhs));
            if domain == PowerResultDomain::Complex {
                break;
            }
        }
        Ok(PowerProbeObserved {
            domain,
            dispatches: recipe.physical().dispatches,
            queue_submissions: recipe.physical().queue_submissions,
        })
    }

    pub(in super::super) fn staged_power_evaluation(
        &self,
        base: &GpuTensorHandle,
        exponent: &GpuTensorHandle,
        domain: PowerResultDomain,
    ) -> ProviderExecutionResult<Option<PowerEvaluationPhysicalProposal>> {
        let [Some(base_rep), Some(exponent_rep)] = [base, exponent].map(|handle| {
            ProviderFeasibilityQuery::exact_numeric_download(handle)
                .and_then(|query| query.inputs.into_iter().next())
        }) else {
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

    pub(in super::super) fn staged_power_evaluation_execute(
        &self,
        recipe: &PowerEvaluationRecipe,
    ) -> ProviderExecutionResult<ProviderExecutedRealElementwise> {
        if !recipe.probe().is_owned_by(self)
            || self
                .staged_power_evaluation(
                    &recipe.probe().sources()[0],
                    &recipe.probe().sources()[1],
                    recipe.domain(),
                )?
                .as_ref()
                != Some(recipe.physical())
        {
            return Err(ProviderExecutionError::from(anyhow!(
                "POWER evaluation changed"
            )));
        }
        let [base, exponent] = recipe.probe().sources();
        let (bases, base_shape) = self.pull(base).map_err(ProviderExecutionError::from)?;
        let (exponents, exponent_shape) =
            self.pull(exponent).map_err(ProviderExecutionError::from)?;
        let output_shape = shape(&recipe.physical().output)
            .ok_or_else(|| ProviderExecutionError::from(anyhow!("POWER output shape overflow")))?;
        let count = output_shape
            .iter()
            .try_fold(1usize, |n, &extent| n.checked_mul(extent))
            .ok_or_else(|| ProviderExecutionError::from(anyhow!("POWER output extent overflow")))?;
        let base_strides = compute_strides(&base_shape);
        let exponent_strides = compute_strides(&exponent_shape);
        let output = if recipe.domain() == PowerResultDomain::Complex {
            let mut base_values = Vec::new();
            let mut exponent_values = Vec::new();
            base_values
                .try_reserve_exact(count)
                .map_err(ProviderExecutionError::from_source)?;
            exponent_values
                .try_reserve_exact(count)
                .map_err(ProviderExecutionError::from_source)?;
            for index in 0..count {
                base_values
                    .push(bases[broadcast_index(index, &output_shape, &base_shape, &base_strides)]);
                exponent_values.push(
                    exponents
                        [broadcast_index(index, &output_shape, &exponent_shape, &exponent_strides)],
                );
            }
            let (base, exponent) = if base.physical().element_type() == NumericElementType::F32 {
                let base = runmat_value::ComplexTensor::from_f32(
                    base_values
                        .into_iter()
                        .map(|value| (value as f32, 0.0))
                        .collect(),
                    output_shape.clone(),
                )
                .map_err(|error| ProviderExecutionError::from(anyhow!(error)))?;
                let exponent = runmat_value::ComplexTensor::from_f32(
                    exponent_values
                        .into_iter()
                        .map(|value| (value as f32, 0.0))
                        .collect(),
                    output_shape.clone(),
                )
                .map_err(|error| ProviderExecutionError::from(anyhow!(error)))?;
                (base, exponent)
            } else {
                let base = runmat_value::ComplexTensor::new(
                    base_values.into_iter().map(|value| (value, 0.0)).collect(),
                    output_shape.clone(),
                )
                .map_err(|error| ProviderExecutionError::from(anyhow!(error)))?;
                let exponent = runmat_value::ComplexTensor::new(
                    exponent_values
                        .into_iter()
                        .map(|value| (value, 0.0))
                        .collect(),
                    output_shape.clone(),
                )
                .map_err(|error| ProviderExecutionError::from(anyhow!(error)))?;
                (base, exponent)
            };
            let computed = runmat_runtime::builtins::common::elementwise::elementwise_pow(
                &Value::ComplexTensor(base),
                &Value::ComplexTensor(exponent),
            )
            .map_err(|error| ProviderExecutionError::from(anyhow!(error)))?;
            let Value::ComplexTensor(computed) = computed else {
                return Err(ProviderExecutionError::from(anyhow!(
                    "complex POWER oracle returned non-complex storage"
                )));
            };
            let class = if recipe.physical().output.element_type == ProviderElementType::ComplexF32
            {
                NumericElementType::F32
            } else {
                NumericElementType::F64
            };
            let data = if class == NumericElementType::F32 {
                BufferPayload::ComplexF32(
                    computed
                        .materialize_f64()
                        .into_iter()
                        .map(|(real, imag)| (real as f32, imag as f32))
                        .collect(),
                )
            } else {
                BufferPayload::ComplexF64(computed.materialize_f64())
            };
            self.push_complex(data, output_shape, class)
        } else {
            let class = if recipe.physical().output.element_type == ProviderElementType::F32 {
                NumericElementType::F32
            } else {
                NumericElementType::F64
            };
            let values = (0..count)
                .map(|index| {
                    let base =
                        bases[broadcast_index(index, &output_shape, &base_shape, &base_strides)];
                    let exponent = exponents
                        [broadcast_index(index, &output_shape, &exponent_shape, &exponent_strides)];
                    if class == NumericElementType::F32 {
                        f64::from((base as f32).powf(exponent as f32))
                    } else {
                        base.powf(exponent)
                    }
                })
                .collect();
            self.push_typed(values, output_shape, class)
        };
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
