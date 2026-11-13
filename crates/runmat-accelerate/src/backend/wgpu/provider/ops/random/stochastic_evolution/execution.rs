use super::super::super::{backend_types::NumericBufferRegistration, WgpuProvider};
use super::feasibility::physical_width;
use super::submission::StochasticTileSubmittedAllocations;
use crate::backend::wgpu::dispatch::observation::PreparedSubmittedAllocationRetention;
use crate::backend::wgpu::residency::BufferUsageClass;
use anyhow::{anyhow, ensure, Result};
use runmat_accelerate_api::scalar_program::ScalarLiteral;
use runmat_accelerate_api::stochastic_evolution::{
    ProviderStochasticEvolutionProgress, ProviderStochasticEvolutionTileRequest,
    PROVIDER_STOCHASTIC_EVOLUTION_STEP_OPERATION,
};
use runmat_accelerate_api::{
    GpuTensorStorage, NumericElementType, ProviderExecutionError, ProviderExecutionResult,
    ProviderOperationIdentity,
};
use runmat_execution::resource::{ProviderHostMetadataAdmission, RetainedHostAllocationClaim};
use std::sync::Arc;

impl WgpuProvider {
    pub(crate) async fn stochastic_evolution_tile_exec(
        &self,
        request: ProviderStochasticEvolutionTileRequest<'_>,
        metadata: Option<&dyn ProviderHostMetadataAdmission>,
        completion_control: Option<Arc<dyn RetainedHostAllocationClaim>>,
    ) -> ProviderExecutionResult<ProviderStochasticEvolutionProgress> {
        let (count, total) = request.validate_for(self)?;
        let selected = self
            .propose_stochastic_evolution_step_exec(request.plan)
            .filter(|proposal| proposal == request.proposal)
            .ok_or_else(|| {
                ProviderExecutionError::from(anyhow!(
                    "stochastic evolution tile has no selected physical proposal"
                ))
            })?;
        if metadata.is_some() && completion_control.is_none() {
            return Err(ProviderExecutionError::from(anyhow!(
                "finite stochastic evolution tile lacks its completion-control claim"
            ))
            .with_unproven_effects());
        }
        let source = self
            .get_entry_raw(request.source)
            .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
        let noise = self
            .get_entry_raw(request.noise)
            .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
        let arithmetic = request
            .plan
            .arithmetic_for_step(request.step)
            .ok_or_else(|| {
                ProviderExecutionError::from(anyhow!(
                    "stochastic evolution step is outside its plan"
                ))
            })?;
        let expected_source = match arithmetic.state_class() {
            runmat_types::NumericClass::Single => NumericElementType::F32,
            runmat_types::NumericClass::Double => NumericElementType::F64,
            _ => unreachable!("validated plan has real floating state"),
        };
        let expected_noise = match arithmetic.noise_class() {
            runmat_types::NumericClass::Single => NumericElementType::F32,
            runmat_types::NumericClass::Double => NumericElementType::F64,
            _ => unreachable!("validated plan has real floating noise"),
        };
        if source.element_type != expected_source
            || noise.element_type != expected_noise
            || source.storage != GpuTensorStorage::Real
            || noise.storage != GpuTensorStorage::Real
            || source.len != usize::try_from(total).unwrap_or(usize::MAX)
            || noise.len != usize::try_from(count).unwrap_or(usize::MAX)
        {
            return Err(ProviderExecutionError::from(anyhow!(
                "stochastic evolution live buffers disagree with the selected plan"
            ))
            .with_unproven_effects());
        }
        let pipeline = self
            .pipelines
            .stochastic_evolution
            .get(arithmetic.class_identity())
            .ok_or_else(|| {
                ProviderExecutionError::from(anyhow!("stochastic evolution pipeline is absent"))
            })?;
        let result_width = physical_width(arithmetic.result_class())
            .expect("validated real-floating result class");
        let resource_scope = runmat_accelerate_api::observation::current_provider_resource_scope()
            .ok_or_else(|| {
                ProviderExecutionError::from(anyhow!(
                    "stochastic evolution submission lacks its resource owner"
                ))
                .with_unproven_effects()
            })?;
        // Reserve registers this operation's exact submission token. Capture
        // its prior boundary first; reusing that prepared token at submit does
        // not create a second registration. Completion is still certified only
        // after the actual kernel's fence and physical retirement below.
        let operation =
            ProviderOperationIdentity::new(PROVIDER_STOCHASTIC_EVOLUTION_STEP_OPERATION);
        let observed = runmat_accelerate_api::observation::current_provider_observation()
            .filter(|recorder| recorder.expects_operation(&operation));
        let checkpoint = observed.as_ref().map(
            runmat_accelerate_api::observation::ProviderOperationRecorder::submission_checkpoint,
        );
        let (submission_completion, submission_retirement) =
            PreparedSubmittedAllocationRetention::reserve()
                .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
        let output = if let Some(pending) = request.pending {
            let output = self
                .get_entry_raw(pending.output())
                .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
            let expected_output = match arithmetic.result_class() {
                runmat_types::NumericClass::Single => NumericElementType::F32,
                runmat_types::NumericClass::Double => NumericElementType::F64,
                _ => unreachable!(),
            };
            if output.shape.as_slice() != request.source.shape()
                || output.len != usize::try_from(total).unwrap_or(usize::MAX)
                || output.element_type != expected_output
                || output.storage != GpuTensorStorage::Real
                || output.allocated_bytes
                    != selected
                        .output_buffer_bytes_for_step(request.step)
                        .unwrap_or(0)
            {
                return Err(ProviderExecutionError::from(anyhow!(
                    "stochastic evolution pending output disagrees with its plan"
                ))
                .with_unproven_effects());
            }
            pending.output().clone()
        } else {
            let buffer = self
                .with_emission(|emission, _| {
                    self.create_storage_buffer_checked_with_element_size(
                        emission,
                        usize::try_from(total)?,
                        "runmat-stochastic-evolution-step-out",
                        BufferUsageClass::Generic,
                        result_width,
                    )
                })
                .map_err(|error| ProviderExecutionError::from(error).after_effect())?
                .map_err(|error| ProviderExecutionError::from(error).after_effect())?;
            self.register_numeric_buffer_produced_admitted(
                buffer,
                NumericBufferRegistration {
                    shape: request.source.shape().to_vec(),
                    len: usize::try_from(total).map_err(|error| {
                        ProviderExecutionError::from(anyhow!(error)).after_effect()
                    })?,
                    physical_element_type: match arithmetic.result_class() {
                        runmat_types::NumericClass::Single => NumericElementType::F32,
                        runmat_types::NumericClass::Double => NumericElementType::F64,
                        _ => unreachable!(),
                    },
                    storage: GpuTensorStorage::Real,
                    allocated_bytes: selected
                        .output_buffer_bytes_for_step(request.step)
                        .expect("validated step has an output bound"),
                    usage: BufferUsageClass::Generic,
                },
                false,
                metadata,
            )
            .map_err(|error| ProviderExecutionError::from(error).after_effect())?
        };
        // Registration establishes the private physical lease and resource
        // owner before submit.  Returning this handle remains conditional on
        // the semantic fence and completed-prefix validation below.
        let output_entry = self
            .get_entry_raw(&output)
            .map_err(|error| ProviderExecutionError::from(error).after_effect())?;
        let params = crate::backend::wgpu::params::StochasticEvolutionStepParams {
            offset: u32::try_from(request.offset)
                .map_err(|error| ProviderExecutionError::from(anyhow!(error)).after_effect())?,
            count: u32::try_from(count)
                .map_err(|error| ProviderExecutionError::from(anyhow!(error)).after_effect())?,
            total: u32::try_from(total)
                .map_err(|error| ProviderExecutionError::from(anyhow!(error)).after_effect())?,
            _pad: 0,
        };
        let submission_claim = completion_control.clone();
        let fence = self
            .with_emission(
                |emission, pending| -> Result<runmat_accelerate_api::WgpuBufferOwner> {
                    use crate::backend::wgpu::params::*;
                    let compiled = pipeline.get(emission)?;
                    let uniform = runmat_accelerate_api::WgpuBufferOwner::new(
                        match (arithmetic.drift(), arithmetic.scale()) {
                            (ScalarLiteral::F32Bits(drift), ScalarLiteral::F32Bits(scale)) => self
                                .uniform_buffer(
                                    emission,
                                    &StochasticEvolutionStepParamsF32F32 {
                                        base: params,
                                        drift: f32::from_bits(drift),
                                        scale: f32::from_bits(scale),
                                        _pad: [0; 2],
                                    },
                                    "runmat-stochastic-evolution-step-f32-f32-params",
                                )?,
                            (ScalarLiteral::F32Bits(drift), ScalarLiteral::F64Bits(scale)) => self
                                .uniform_buffer(
                                    emission,
                                    &StochasticEvolutionStepParamsF32F64 {
                                        base: params,
                                        drift: f32::from_bits(drift),
                                        _pad: 0,
                                        scale: f64::from_bits(scale),
                                    },
                                    "runmat-stochastic-evolution-step-f32-f64-params",
                                )?,
                            (ScalarLiteral::F64Bits(drift), ScalarLiteral::F32Bits(scale)) => self
                                .uniform_buffer(
                                    emission,
                                    &StochasticEvolutionStepParamsF64F32 {
                                        base: params,
                                        drift: f64::from_bits(drift),
                                        scale: f32::from_bits(scale),
                                        _pad: 0,
                                    },
                                    "runmat-stochastic-evolution-step-f64-f32-params",
                                )?,
                            (ScalarLiteral::F64Bits(drift), ScalarLiteral::F64Bits(scale)) => self
                                .uniform_buffer(
                                    emission,
                                    &StochasticEvolutionStepParamsF64F64 {
                                        base: params,
                                        drift: f64::from_bits(drift),
                                        scale: f64::from_bits(scale),
                                    },
                                    "runmat-stochastic-evolution-step-f64-f64-params",
                                )?,
                        },
                    );
                    let fence = runmat_accelerate_api::WgpuBufferOwner::new(
                        emission.device().create_buffer(&wgpu::BufferDescriptor {
                            label: Some("runmat-stochastic-evolution-step-fence"),
                            size: selected.completion_fence_bytes_per_tile,
                            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        }),
                    );
                    let bind = emission
                        .device()
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("runmat-stochastic-evolution-step-bind"),
                            layout: &compiled.layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: source.buffer.as_ref().as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: noise.buffer.as_ref().as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: output_entry.buffer.as_ref().as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 3,
                                    resource: uniform.as_ref().as_entire_binding(),
                                },
                            ],
                        });
                    for bytes in [uniform.size(), fence.size()] {
                        pending.retain_provider_resource_until_next_submit(
                            resource_scope.issue_resource_ticket(bytes)?,
                        );
                    }
                    pending.retain_allocations_until_next_submit(
                        submission_completion,
                        StochasticTileSubmittedAllocations::new(
                            request.source.clone(),
                            request.noise.clone(),
                            output.clone(),
                            uniform,
                            fence.clone(),
                            submission_claim,
                        )
                        .into_retention(),
                    )?;
                    let mut encoder = pending.profiled_encoder(
                        emission,
                        &wgpu::CommandEncoderDescriptor {
                            label: Some("runmat-stochastic-evolution-step-encoder"),
                        },
                    );
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("runmat-stochastic-evolution-step-pass"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&compiled.pipeline);
                        pass.set_bind_group(0, &bind, &[]);
                        pass.dispatch_workgroups(
                            pipeline.linear_dispatch_groups(usize::try_from(count)?)?,
                            1,
                            1,
                        );
                    }
                    encoder.clear_buffer(fence.as_ref(), 0, None);
                    self.submit(emission, pending, encoder)?;
                    Ok(fence)
                },
            )
            .map_err(|error| ProviderExecutionError::from(error).after_effect())?
            .map_err(|error| ProviderExecutionError::from(error).after_effect())?;
        let decode = |bytes: &[u8]| -> Result<()> {
            ensure!(
                bytes == [0; 8],
                "stochastic evolution completion fence is malformed"
            );
            Ok(())
        };
        match completion_control {
            Some(claim) => {
                self.map_readback_with_admitted(
                    fence,
                    "stochastic evolution completion",
                    decode,
                    claim,
                )
                .await
            }
            None => {
                self.map_readback_with(fence, "stochastic evolution completion", decode)
                    .await
            }
        }
        .map_err(|error| ProviderExecutionError::from(error).after_effect())?;
        submission_retirement
            .await
            .map_err(|error| ProviderExecutionError::from(error).after_effect())?;
        if let (Some(recorder), Some(checkpoint)) = (observed, checkpoint) {
            recorder.confirm_submitted_provider_execution_after(
                self.device_id(),
                &operation,
                checkpoint,
            );
        }
        ProviderStochasticEvolutionProgress::from_completed_tile(self, request, output)
            .map_err(|error| error.after_effect())
    }
}
