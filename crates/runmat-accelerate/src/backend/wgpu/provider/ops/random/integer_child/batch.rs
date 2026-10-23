//! Bounded rejection batch and retained pending progress.

use super::*;
use runmat_accelerate_api::WgpuBufferOwner;

impl WgpuProvider {
    pub(crate) fn propose_integer_child_batch_exec(
        &self,
        request: GpuIntegerBatchRequest<'_>,
    ) -> Option<GpuIntegerBatchPhysicalProposal> {
        let count = request.validate()?;
        request.integer.rejection_domain_32()?;
        if count == 0
            || count > u32::MAX as usize
            || (request.integer.class == NumericElementType::F64
                && self.precision != NumericPrecision::F64)
        {
            return None;
        }
        let width = if request.integer.class == NumericElementType::F64 {
            8
        } else {
            4
        };
        let output_buffer_bytes =
            crate::backend::wgpu::residency::BufferResidency::requested_bytes(count, width)?;
        let accepted_buffer_bytes =
            crate::backend::wgpu::residency::BufferResidency::requested_bytes(count, 4)?;
        if [output_buffer_bytes, accepted_buffer_bytes]
            .into_iter()
            .any(|bytes| {
                bytes > self.adapter_limits.max_buffer_size
                    || bytes > u64::from(self.adapter_limits.max_storage_buffer_binding_size)
            })
        {
            return None;
        }
        let chunk_capacity = self.linear_launch.chunk_capacity()?;
        let chunks = u32::try_from(count.div_ceil(chunk_capacity)).ok()?;
        let uniform_buffer_bytes = u64::from(chunks).checked_mul(std::mem::size_of::<
            crate::backend::wgpu::params::RandomIntegerBatchParams,
        >() as u64)?;
        // `entry_for_handle` clones the record shape, then replaces it with a
        // cloned logical view. While authenticating the second pending input,
        // the first result shape and both second-input shapes coexist.
        let pending_view_workspace_bytes = if request.pending.is_some() {
            u64::try_from(request.integer.shape.len())
                .ok()?
                .checked_mul(std::mem::size_of::<usize>() as u64)?
                .checked_mul(3)?
        } else {
            0
        };
        Some(GpuIntegerBatchPhysicalProposal {
            output_buffer_bytes,
            accepted_buffer_bytes,
            count_buffer_bytes: 4,
            uniform_buffer_bytes,
            staging_buffer_bytes: 4,
            readback_control_host_bytes: map_completion_host_upper_bytes()?,
            pending_view_workspace_bytes,
            uniform_allocations: chunks,
            queue_submissions: chunks.checked_add(1)?,
        })
    }

    pub(crate) async fn integer_child_batch_exec(
        &self,
        request: GpuIntegerBatchRequest<'_>,
        metadata: Option<&dyn ProviderHostMetadataAdmission>,
        readback_control: Option<
            std::sync::Arc<dyn runmat_execution::resource::RetainedHostAllocationClaim>,
        >,
    ) -> ProviderExecutionResult<GpuIntegerBatchProgress> {
        let Some(physical) = self.propose_integer_child_batch_exec(request) else {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "GPU integer batch has no physical proposal",
            )
            .into());
        };
        if metadata.is_some() && readback_control.is_none() {
            return Err(ProviderExecutionError::from(anyhow!(
                "finite GPU integer batch lacks its readback-control claim"
            ))
            .with_unproven_effects());
        }
        let (lower, span, accept_limit) = request
            .integer
            .rejection_domain_32()
            .expect("physical proposal validated domain");
        let count = request
            .validate()
            .expect("physical proposal validated count");
        let width = if request.integer.class == NumericElementType::F64 {
            8
        } else {
            4
        };
        let (output_buffer, accepted_buffer) = if let Some(pending) = request.pending {
            let output = self
                .get_entry_raw(&pending.output)
                .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
            let accepted = self
                .get_entry_raw(&pending.accepted)
                .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
            if output.len != count
                || output.shape != request.integer.shape
                || output.element_type != request.integer.class
                || output.storage != GpuTensorStorage::Real
                || output.allocated_bytes > physical.output_buffer_bytes
                || accepted.len != count
                || accepted.shape != request.integer.shape
                || accepted.element_type != NumericElementType::U32
                || accepted.storage != GpuTensorStorage::Real
                || accepted.allocated_bytes > physical.accepted_buffer_bytes
            {
                return Err(ProviderExecutionError::from(anyhow!(
                    "GPU integer pending buffers do not match the selected batch"
                ))
                .with_unproven_effects());
            }
            (output.buffer, accepted.buffer)
        } else {
            let output = self
                .with_emission(|emission, _| {
                    self.create_storage_buffer_checked_with_element_size(
                        emission,
                        count,
                        "runmat-random-child-batch-out",
                        BufferUsageClass::Generic,
                        width,
                    )
                })
                .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?
                .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
            let accepted = self
                .with_emission(|emission, _| {
                    self.create_storage_buffer_checked_with_element_size(
                        emission,
                        count,
                        "runmat-random-child-batch-accepted",
                        BufferUsageClass::Generic,
                        4,
                    )
                })
                .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?
                .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
            (output, accepted)
        };
        let chunk_capacity = self.linear_launch.chunk_capacity().ok_or_else(|| {
            ProviderExecutionError::from(anyhow!("GPU integer batch has no frozen launch geometry"))
                .with_unproven_effects()
        })?;
        let pipeline = match request.integer.class {
            NumericElementType::F32 => &self.pipelines.random_child_batch_f32,
            NumericElementType::F64 => &self.pipelines.random_child_batch_f64,
            NumericElementType::I32 => &self.pipelines.random_child_batch_i32,
            NumericElementType::U32 => &self.pipelines.random_child_batch_u32,
            _ => unreachable!("physical proposal validates native integer class"),
        };
        let operation = ProviderOperationIdentity::new(PROVIDER_GPU_INTEGER_BATCH_OPERATION);
        let observed = runmat_accelerate_api::observation::current_provider_observation()
            .filter(|recorder| recorder.expects_operation(&operation));
        let before_dispatch = observed.as_ref().map(
            runmat_accelerate_api::observation::ProviderOperationRecorder::submission_checkpoint,
        );
        let staging = self
            .with_emission(|emission, pending| -> Result<WgpuBufferOwner> {
                let count_storage = emission.device().create_buffer(&wgpu::BufferDescriptor {
                    label: Some("runmat-random-child-batch-count"),
                    size: 4,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let staging = runmat_accelerate_api::WgpuBufferOwner::new(
                    emission.device().create_buffer(&wgpu::BufferDescriptor {
                        label: Some("runmat-random-child-batch-readback"),
                        size: 4,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
                );
                let mut offset = 0usize;
                while offset < count {
                    let chunk = (count - offset).min(chunk_capacity);
                    let reservation = request.integer.reservation;
                    let params = crate::backend::wgpu::params::RandomIntegerBatchParams {
                        base: crate::backend::wgpu::params::RandomChildParams {
                            offset: u32::try_from(offset)?,
                            chunk: u32::try_from(chunk)?,
                            first0: reservation.first as u32,
                            first1: (reservation.first >> 32) as u32,
                            first2: (reservation.first >> 64) as u32,
                            first3: (reservation.first >> 96) as u32,
                            seed0: reservation.seed as u32,
                            seed1: (reservation.seed >> 32) as u32,
                            integer_lower: 0,
                            integer_mask: 0,
                            _pad0: 0,
                            _pad1: 0,
                        },
                        span0: span as u32,
                        span1: (span >> 32) as u32,
                        accept_limit0: accept_limit as u32,
                        accept_limit1: (accept_limit >> 32) as u32,
                        lower0: lower,
                        lower1: if lower & 0x8000_0000 != 0 {
                            u32::MAX
                        } else {
                            0
                        },
                        attempt0: request.first_attempt as u32,
                        attempt1: (request.first_attempt >> 32) as u32,
                        attempts: request.attempts_per_item,
                        initial: u32::from(request.pending.is_none()),
                        _pad: [0; 2],
                    };
                    let uniform =
                        self.uniform_buffer(emission, &params, "runmat-random-child-batch-params")?;
                    let compiled = pipeline.get(emission)?;
                    let bind_group =
                        emission
                            .device()
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("runmat-random-child-batch-bind"),
                                layout: &compiled.layout,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: output_buffer.as_ref().as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: uniform.as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 2,
                                        resource: accepted_buffer.as_ref().as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 3,
                                        resource: count_storage.as_entire_binding(),
                                    },
                                ],
                            });
                    let mut encoder = pending.profiled_encoder(
                        emission,
                        &wgpu::CommandEncoderDescriptor {
                            label: Some("runmat-random-child-batch-encoder"),
                        },
                    );
                    if offset == 0 {
                        encoder.clear_buffer(&count_storage, 0, None);
                    }
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("runmat-random-child-batch-pass"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&compiled.pipeline);
                        pass.set_bind_group(0, &bind_group, &[]);
                        pass.dispatch_workgroups(pipeline.linear_dispatch_groups(chunk)?, 1, 1);
                    }
                    self.submit(emission, pending, encoder)?;
                    offset += chunk;
                }
                let mut encoder = pending.profiled_encoder(
                    emission,
                    &wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-random-child-batch-count-encoder"),
                    },
                );
                encoder.copy_buffer_to_buffer(&count_storage, 0, &staging, 0, 4);
                self.submit(emission, pending, encoder)?;
                Ok(staging)
            })
            .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?
            .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
        let decode_count = |bytes: &[u8]| {
            ensure!(
                bytes.len() == 4,
                "GPU integer count readback has wrong size"
            );
            Ok(u32::from_le_bytes(bytes.try_into()?))
        };
        let remaining = match readback_control {
            Some(claim) => {
                self.map_readback_with_admitted(
                    staging,
                    "GPU integer batch count",
                    decode_count,
                    claim,
                )
                .await
            }
            None => {
                self.map_readback_with(staging, "GPU integer batch count", decode_count)
                    .await
            }
        }
        .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
        if let (Some(recorder), Some(checkpoint)) = (observed, before_dispatch) {
            recorder.confirm_submitted_provider_execution_after(
                self.device_id(),
                &operation,
                checkpoint,
            );
        }
        let (output, accepted) = if let Some(pending) = request.pending {
            (pending.output.clone(), pending.accepted.clone())
        } else {
            let output = self
                .register_numeric_buffer_produced_admitted(
                    output_buffer,
                    NumericBufferRegistration {
                        shape: request.integer.shape.to_vec(),
                        len: count,
                        physical_element_type: request.integer.class,
                        storage: GpuTensorStorage::Real,
                        allocated_bytes: physical.output_buffer_bytes,
                        usage: BufferUsageClass::Generic,
                    },
                    false,
                    metadata,
                )
                .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
            let accepted = self
                .register_numeric_buffer_produced_admitted(
                    accepted_buffer,
                    NumericBufferRegistration {
                        shape: request.integer.shape.to_vec(),
                        len: count,
                        physical_element_type: NumericElementType::U32,
                        storage: GpuTensorStorage::Real,
                        allocated_bytes: physical.accepted_buffer_bytes,
                        usage: BufferUsageClass::Generic,
                    },
                    false,
                    metadata,
                )
                .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
            (output, accepted)
        };
        if remaining == 0 {
            Ok(GpuIntegerBatchProgress::Complete(GpuIntegerPending {
                output,
                accepted,
                remaining: 0,
            }))
        } else if u64::from(remaining)
            <= request
                .pending
                .map_or(count as u64, |pending| pending.remaining)
        {
            Ok(GpuIntegerBatchProgress::Pending(GpuIntegerPending {
                output,
                accepted,
                remaining: u64::from(remaining),
            }))
        } else {
            Err(ProviderExecutionError::from(anyhow!(
                "GPU integer batch remaining count increased"
            ))
            .with_unproven_effects())
        }
    }
}
