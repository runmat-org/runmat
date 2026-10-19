//! Resident floating casts use target-typed storage, not provider-native unary allocation.
use super::*;
use runmat_accelerate_api::{NumericElementType, WgpuBufferOwner};

#[path = "floating_cast/frame.rs"]
mod frame;

#[cfg(test)]
#[path = "floating_cast/tests.rs"]
mod tests;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct Params {
    len: u32,
    offset: u32,
    padding: [u32; 2],
}

/// Recorded at the actual allocation/submission edges, never reconstructed
/// from a declaration after execution. Alias publication is not an effect.
pub(super) enum CastResourceEvent {
    Output(u64),
    Alias(u64),
    Uniform(u64),
    Submitted { dispatch: bool },
}

impl WgpuProvider {
    pub(crate) async fn floating_cast_exec(
        &self,
        input: &GpuTensorHandle,
        target: NumericPrecision,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.floating_cast_exec_observed(input, target, |_| Ok(()))
            .await
    }

    pub(crate) async fn floating_cast_exec_typed(
        &self,
        input: &GpuTensorHandle,
        target: NumericPrecision,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.floating_cast_exec_typed_after_allocation(input, target, |_| Ok(()))
            .await
    }

    async fn floating_cast_exec_typed_after_allocation(
        &self,
        input: &GpuTensorHandle,
        target: NumericPrecision,
        checkpoint: impl FnMut(&WgpuBufferOwner) -> Result<()>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.floating_cast_exec_observed(input, target, checkpoint)
            .await
    }

    async fn floating_cast_exec_observed(
        &self,
        input: &GpuTensorHandle,
        target: NumericPrecision,
        mut after_allocation: impl FnMut(&WgpuBufferOwner) -> Result<()>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.floating_cast_exec_recorded_with_checkpoint(
            input,
            target,
            None,
            &mut after_allocation,
            |_| Ok(()),
        )
        .await
    }

    pub(super) async fn floating_cast_exec_recorded(
        &self,
        input: &GpuTensorHandle,
        target: NumericPrecision,
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
        record: impl FnMut(CastResourceEvent) -> Result<()>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.floating_cast_exec_recorded_with_checkpoint(
            input,
            target,
            admission,
            |_| Ok(()),
            record,
        )
        .await
    }

    async fn floating_cast_exec_recorded_with_checkpoint(
        &self,
        input: &GpuTensorHandle,
        target: NumericPrecision,
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
        mut after_allocation: impl FnMut(&WgpuBufferOwner) -> Result<()>,
        mut record: impl FnMut(CastResourceEvent) -> Result<()>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        use runmat_accelerate_api::ProviderExecutionError;
        let frozen = frame::FloatingCastFrame::plan(self, input, target)
            .map_err(ProviderExecutionError::from)?;
        let target_type = match frozen.target {
            NumericPrecision::F32 => NumericElementType::F32,
            NumericPrecision::F64 => NumericElementType::F64,
        };
        if frozen.entry.precision == target && !frozen.logical_producer {
            let output = if let Some(admission) = admission {
                let bytes = GpuTensorHandle::view_host_upper_bytes(input.view().shape_capacity())
                    .ok_or_else(|| {
                    ProviderExecutionError::from(anyhow!("floating cast alias view byte overflow"))
                })?;
                let grant = admission
                    .claim_view(bytes)
                    .map_err(ProviderExecutionError::from_source)?;
                input.with_view_retained_grant(input.view().clone().with_logical(false), grant)
            } else {
                input.with_view(input.view().clone().with_logical(false))
            }
            .map_err(ProviderExecutionError::from)?;
            record(CastResourceEvent::Alias(
                output.physical().requested_bytes(),
            ))
            .map_err(ProviderExecutionError::from)?;
            return Ok(output);
        }
        let validated = if let Some(recipe) = &frozen.recipe {
            let pipeline = self
                .prepare_validated_floating_cast_pipeline(
                    recipe,
                    frozen.entry.precision,
                    frozen.target,
                )
                .await?;
            frozen
                .require_live(self, input)
                .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
            pipeline
                .require_floating_cast(recipe, frozen.entry.precision, frozen.target)
                .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
            Some(pipeline)
        } else {
            None
        };
        let dispatch_geometry = if frozen.recipe.is_some() {
            Some(frozen.launch.zip(frozen.chunk_capacity).ok_or_else(|| {
                ProviderExecutionError::from(anyhow!("floating cast dispatch geometry missing"))
            })?)
        } else {
            None
        };
        let produced = std::cell::Cell::new(false);
        // The existing checked raw-word allocator is independent of provider
        // float precision; typed registration below owns the interpretation.
        let output = self
            .with_emission(
                |emission, pending| -> Result<runmat_accelerate_api::WgpuBufferOwner> {
                    let output = self.create_integer_word_buffer(
                        emission,
                        frozen.words,
                        "runmat-floating-cast",
                    )?;
                    produced.set(true);
                    record(CastResourceEvent::Output(output.size()))?;
                    after_allocation(&output)?;
                    if frozen.entry.precision == target {
                        if frozen.scalar_len != 0 {
                            let mut encoder = pending.profiled_encoder(
                                emission,
                                &wgpu::CommandEncoderDescriptor {
                                    label: Some("runmat-logical-to-numeric-copy"),
                                },
                            );
                            encoder.copy_buffer_to_buffer(
                                &frozen.entry.buffer,
                                0,
                                &output,
                                0,
                                frozen.allocated_bytes,
                            );
                            self.submit(emission, pending, encoder)?;
                            record(CastResourceEvent::Submitted { dispatch: false })?;
                        }
                        return Ok(output);
                    }
                    if frozen.scalar_len != 0 {
                        let validated = validated
                            .as_ref()
                            .ok_or_else(|| anyhow!("floating cast pipeline missing"))?;
                        let (launch, capacity) = dispatch_geometry
                            .ok_or_else(|| anyhow!("floating cast dispatch geometry missing"))?;
                        let mut offset = 0;
                        while offset < frozen.scalar_len {
                            let count = (frozen.scalar_len - offset).min(capacity);
                            let params = self.uniform_buffer(
                                emission,
                                &Params {
                                    len: count as u32,
                                    offset: offset as u32,
                                    padding: [0; 2],
                                },
                                "runmat-floating-cast-params",
                            )?;
                            record(CastResourceEvent::Uniform(params.size()))?;
                            let bind_group =
                                emission
                                    .device()
                                    .create_bind_group(&wgpu::BindGroupDescriptor {
                                        label: Some("runmat-floating-cast-bg"),
                                        layout: &validated.layout,
                                        entries: &[
                                            wgpu::BindGroupEntry {
                                                binding: 0,
                                                resource: frozen
                                                    .entry
                                                    .buffer
                                                    .as_ref()
                                                    .as_entire_binding(),
                                            },
                                            wgpu::BindGroupEntry {
                                                binding: 1,
                                                resource: output.as_ref().as_entire_binding(),
                                            },
                                            wgpu::BindGroupEntry {
                                                binding: 2,
                                                resource: params.as_entire_binding(),
                                            },
                                        ],
                                    });
                            let groups = launch.dispatch_groups(count).ok_or_else(|| {
                                anyhow!("floating cast dispatch changed after preparation")
                            })?;
                            crate::backend::wgpu::dispatch::elementwise::run(
                                emission,
                                pending,
                                &validated.pipeline,
                                &bind_group,
                                groups,
                            )?;
                            record(CastResourceEvent::Submitted { dispatch: true })?;
                            offset += count;
                        }
                    }
                    Ok(output)
                },
            )
            .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?
            .map_err(|error| {
                let error = ProviderExecutionError::from(error);
                if produced.get() {
                    error.after_effect()
                } else {
                    error
                }
            })?;
        // Publication is outside device emission: registration/view failure
        // cleanup may invoke residency owners and must not reenter the gate.
        let output = self
            .register_numeric_buffer_produced_admitted(
                output,
                NumericBufferRegistration {
                    shape: frozen.entry.shape,
                    len: frozen.entry.len,
                    physical_element_type: target_type,
                    storage: frozen.entry.storage,
                    allocated_bytes: frozen.allocated_bytes,
                    usage: BufferUsageClass::Generic,
                },
                false,
                admission,
            )
            .map_err(|error| ProviderExecutionError::from(error).after_effect())?;
        self.finish_floating_cast_view(output, frozen.transpose)
            .map_err(|error| ProviderExecutionError::from(error).after_effect())
    }

    fn finish_floating_cast_view(
        &self,
        output: GpuTensorHandle,
        transpose: Option<runmat_accelerate_api::TransposeInfo>,
    ) -> Result<GpuTensorHandle> {
        if let Some(info) = transpose {
            return Ok(output.with_view(output.view().clone().with_layout(
                runmat_accelerate_api::allocation::GpuTensorLayout::Transpose(info),
            ))?);
        }
        Ok(output)
    }
}
