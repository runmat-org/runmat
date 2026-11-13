//! Fixed-work power-of-two integer child route.

use super::*;

impl WgpuProvider {
    pub(crate) fn propose_integer_child_exec(
        &self,
        request: &GpuIntegerRequest<'_>,
    ) -> Option<GpuRandomPhysicalProposal> {
        let count = request.element_count()?;
        request.fixed_work_32()?;
        if count > u32::MAX as usize
            || (request.class == NumericElementType::F64 && self.precision != NumericPrecision::F64)
        {
            return None;
        }
        let width = if request.class == NumericElementType::F64 {
            8
        } else {
            4
        };
        let output_buffer_bytes =
            crate::backend::wgpu::residency::BufferResidency::requested_bytes(count, width)?;
        if output_buffer_bytes > self.adapter_limits.max_buffer_size
            || output_buffer_bytes > u64::from(self.adapter_limits.max_storage_buffer_binding_size)
        {
            return None;
        }
        let chunk_capacity = self.linear_launch.chunk_capacity()?;
        let dispatches = u32::try_from(count.div_ceil(chunk_capacity)).ok()?;
        let uniform_buffer_bytes = u64::from(dispatches).checked_mul(std::mem::size_of::<
            crate::backend::wgpu::params::RandomChildParams,
        >() as u64)?;
        Some(GpuRandomPhysicalProposal {
            output_buffer_bytes,
            uniform_buffer_bytes,
            uniform_allocations: dispatches,
            dispatches,
            queue_submissions: dispatches,
        })
    }

    pub(crate) fn integer_child_exec(
        &self,
        request: &GpuIntegerRequest<'_>,
        metadata: Option<&dyn ProviderHostMetadataAdmission>,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        let Some(physical) = self.propose_integer_child_exec(request) else {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "GPU integer request has no fixed-work physical proposal",
            )
            .into());
        };
        let Some((lower, mask)) = request.fixed_work_32() else {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "GPU integer request is outside fixed-work domain",
            )
            .into());
        };
        let count = request
            .element_count()
            .expect("physical proposal validated count");
        let width = if request.class == NumericElementType::F64 {
            8
        } else {
            4
        };
        let buffer = self
            .with_emission(|emission, _| {
                self.create_storage_buffer_checked_with_element_size(
                    emission,
                    count,
                    "runmat-random-child-integer-out",
                    BufferUsageClass::Generic,
                    width,
                )
            })
            .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?
            .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
        if buffer.size() > physical.output_buffer_bytes {
            return Err(ProviderExecutionError::from(anyhow!(
                "GPU integer output exceeded physical proposal"
            ))
            .with_unproven_effects());
        }
        let operation = ProviderOperationIdentity::new(PROVIDER_GPU_INTEGER_OPERATION);
        let observed = runmat_accelerate_api::observation::current_provider_observation()
            .filter(|recorder| recorder.expects_operation(&operation));
        let before_dispatch = observed.as_ref().map(
            runmat_accelerate_api::observation::ProviderOperationRecorder::submission_checkpoint,
        );
        if count != 0 {
            let pipeline = match request.class {
                NumericElementType::F32 => &self.pipelines.random_child_integer_f32,
                NumericElementType::F64 => &self.pipelines.random_child_integer_f64,
                NumericElementType::I32 | NumericElementType::U32 => {
                    &self.pipelines.random_child_integer_u32
                }
                _ => unreachable!("fixed-work class gate is exhaustive"),
            };
            let chunk_capacity = self.linear_launch.chunk_capacity().ok_or_else(|| {
                ProviderExecutionError::from(anyhow!(
                    "GPU integer request has no frozen launch geometry"
                ))
                .with_unproven_effects()
            })?;
            self.with_emission(|emission, pending| -> Result<()> {
                let mut offset = 0usize;
                while offset < count {
                    let chunk = (count - offset).min(chunk_capacity);
                    let first = request.reservation.first;
                    let seed = request.reservation.seed;
                    let params = crate::backend::wgpu::params::RandomChildParams {
                        offset: u32::try_from(offset)?,
                        chunk: u32::try_from(chunk)?,
                        first0: first as u32,
                        first1: (first >> 32) as u32,
                        first2: (first >> 64) as u32,
                        first3: (first >> 96) as u32,
                        seed0: seed as u32,
                        seed1: (seed >> 32) as u32,
                        integer_lower: lower,
                        integer_mask: mask,
                        _pad0: 0,
                        _pad1: 0,
                    };
                    let params_buffer = self.uniform_buffer(
                        emission,
                        &params,
                        "runmat-random-child-integer-params",
                    )?;
                    let compiled = pipeline.get(emission)?;
                    let bind_group =
                        emission
                            .device()
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("runmat-random-child-integer-bind"),
                                layout: &compiled.layout,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: buffer.as_ref().as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: params_buffer.as_entire_binding(),
                                    },
                                ],
                            });
                    let workgroups = pipeline.linear_dispatch_groups(chunk)?;
                    crate::backend::wgpu::dispatch::creation::run(
                        emission,
                        pending,
                        &compiled.pipeline,
                        &bind_group,
                        workgroups,
                        "runmat-random-child-integer-encoder",
                        "runmat-random-child-integer-pass",
                    )?;
                    offset += chunk;
                }
                Ok(())
            })
            .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?
            .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
        }
        let bytes = buffer.size();
        let output = self
            .register_numeric_buffer_produced_admitted(
                buffer,
                NumericBufferRegistration {
                    shape: request.shape.to_vec(),
                    len: count,
                    physical_element_type: request.class,
                    storage: GpuTensorStorage::Real,
                    allocated_bytes: bytes,
                    usage: BufferUsageClass::Generic,
                },
                false,
                metadata,
            )
            .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
        if let Some(recorder) = observed {
            if count == 0 {
                recorder.confirm_provider_execution(self.device_id(), &operation);
            } else if let Some(checkpoint) = before_dispatch {
                recorder.confirm_submitted_provider_execution_after(
                    self.device_id(),
                    &operation,
                    checkpoint,
                );
            }
        }
        Ok(output)
    }
}
