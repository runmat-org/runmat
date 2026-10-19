use crate::backend::wgpu::residency::BufferUsageClass;
use anyhow::{anyhow, Result};
use runmat_accelerate_api::random::{
    GpuNormalTransform, GpuRandomAlgorithm, GpuRandomDistribution, GpuRandomPhysicalProposal,
    GpuRandomRequest,
};
use runmat_accelerate_api::{
    GpuTensorHandle, GpuTensorStorage, NumericElementType, ProviderElementType,
    ProviderExecutionError, ProviderExecutionResult, ProviderFeasibility, ProviderFeasibilityQuery,
    ProviderLayout, ProviderOperationFamily, ProviderPrecision, ProviderRejectionCode,
    ProviderResidency, ProviderResourceEstimate, ProviderStorage,
};
use runmat_execution::resource::ProviderHostMetadataAdmission;

use super::super::{NumericPrecision, WgpuProvider};

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "child/tests.rs"]
mod tests;

impl WgpuProvider {
    pub(crate) fn random_child_feasibility(
        &self,
        query: &ProviderFeasibilityQuery,
    ) -> Option<ProviderFeasibility> {
        if query.workload.gpu_integer.is_some()
            || query.operation.0 == runmat_accelerate_api::random::PROVIDER_GPU_INTEGER_OPERATION
        {
            return self.integer_child_feasibility(query);
        }
        let random_identity =
            query.operation.0 == runmat_accelerate_api::random::PROVIDER_GPU_RANDOM_OPERATION;
        let workload = match query.workload.gpu_random {
            Some(workload) => workload,
            None if random_identity => {
                return Some(ProviderFeasibility::rejected(
                    ProviderRejectionCode::UnknownResourceEstimate,
                    "random.gpu_child.missing_workload",
                ));
            }
            None => return None,
        };
        use ProviderRejectionCode as Code;
        let reject = |code, detail| ProviderFeasibility::rejected(code, detail);
        if !random_identity
            || query.family != ProviderOperationFamily::Library
            || !query.inputs.is_empty()
            || query.outputs.len() != 1
        {
            return Some(reject(Code::InvalidShape, "random.gpu_child.invalid_query"));
        }
        let output = &query.outputs[0];
        let storage = match output.storage {
            ProviderStorage::DenseReal => GpuTensorStorage::Real,
            ProviderStorage::DenseComplexInterleaved => GpuTensorStorage::ComplexInterleaved,
            ProviderStorage::Sparse => {
                return Some(reject(Code::UnsupportedStorage, "random.gpu_child.output"));
            }
        };
        if output.layout != ProviderLayout::ColumnMajorContiguous
            || output.residency != ProviderResidency::Device
        {
            return Some(reject(Code::UnsupportedStorage, "random.gpu_child.output"));
        }
        let precision = match output.element_type {
            ProviderElementType::F32 => ProviderPrecision::F32,
            ProviderElementType::F64 => ProviderPrecision::F64,
            _ => {
                return Some(reject(
                    Code::UnsupportedElementType,
                    "random.gpu_child.class",
                ))
            }
        };
        // The representation already owns the dimensions. A feasibility
        // probe must not allocate a second rank-sized vector before the
        // request's finite host preparation is admitted.
        let Some(count) = (if output.shape.contains(&0) {
            Some(0)
        } else {
            output.shape.iter().try_fold(1usize, |count, extent| {
                count.checked_mul(usize::try_from(*extent).ok()?)
            })
        }) else {
            return Some(reject(Code::InvalidShape, "random.gpu_child.shape"));
        };
        let count = count.checked_mul(match storage {
            GpuTensorStorage::Real => 1,
            GpuTensorStorage::ComplexInterleaved => 2,
        });
        let Some(count) = count else {
            return Some(reject(Code::InvalidShape, "random.gpu_child.components"));
        };
        if u64::try_from(count).ok() != Some(workload.reservation.count)
            || workload
                .reservation
                .first
                .checked_add(u128::from(workload.reservation.count))
                .is_none()
        {
            return Some(reject(Code::InvalidShape, "random.gpu_child.range"));
        }
        let Some(physical) = self.propose_random_child_count(
            count,
            precision,
            workload.reservation.algorithm,
            workload.reservation.normal_transform,
        ) else {
            return Some(reject(
                Code::UnknownResourceEstimate,
                "random.gpu_child.physical_route",
            ));
        };
        Some(ProviderFeasibility::supported(ProviderResourceEstimate {
            transient_bytes: physical.requested_incremental_peak_bytes(),
            output_bytes: Some(physical.output_buffer_bytes),
            dispatches: Some(physical.queue_submissions),
        }))
    }

    pub(crate) fn propose_random_child_exec(
        &self,
        request: &GpuRandomRequest<'_>,
    ) -> Option<GpuRandomPhysicalProposal> {
        self.propose_random_child_count(
            request.element_count()?,
            request.precision,
            request.reservation.algorithm,
            request.reservation.normal_transform,
        )
    }

    fn propose_random_child_count(
        &self,
        count: usize,
        precision: ProviderPrecision,
        algorithm: GpuRandomAlgorithm,
        transform: GpuNormalTransform,
    ) -> Option<GpuRandomPhysicalProposal> {
        if (precision == ProviderPrecision::F64 && self.precision != NumericPrecision::F64)
            || algorithm != GpuRandomAlgorithm::Threefry4x64_20
            || transform != GpuNormalTransform::BoxMuller
            || count > u32::MAX as usize
        {
            return None;
        }
        let element_size = match precision {
            ProviderPrecision::F32 => std::mem::size_of::<f32>(),
            ProviderPrecision::F64 => std::mem::size_of::<f64>(),
        };
        let output_buffer_bytes =
            crate::backend::wgpu::residency::BufferResidency::requested_bytes(count, element_size)?;
        if output_buffer_bytes > self.adapter_limits.max_buffer_size
            || output_buffer_bytes > u64::from(self.adapter_limits.max_storage_buffer_binding_size)
        {
            return None;
        }
        let chunk_capacity = self.linear_launch.chunk_capacity()?;
        let dispatches = count.div_ceil(chunk_capacity);
        let Ok(dispatches) = u32::try_from(dispatches) else {
            return None;
        };
        let uniform_buffer_bytes = u64::from(dispatches).checked_mul(std::mem::size_of::<
            crate::backend::wgpu::params::RandomChildParams,
        >() as u64)?;
        Some(GpuRandomPhysicalProposal {
            output_buffer_bytes,
            uniform_buffer_bytes,
            uniform_allocations: dispatches,
            dispatches,
            // dispatch::creation::run submits each encoded chunk separately.
            queue_submissions: dispatches,
        })
    }

    pub(crate) fn random_child_exec(
        &self,
        request: &GpuRandomRequest<'_>,
        metadata: Option<&dyn ProviderHostMetadataAdmission>,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        let Some(physical) = self.propose_random_child_exec(request) else {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "GPU random request has no physical proposal",
            )
            .into());
        };
        let count = request.element_count().ok_or_else(|| {
            ProviderExecutionError::from(runmat_accelerate_api::unsupported_provider_operation(
                "GPU random shape differs from reserved child-stream range",
            ))
        })?;
        let chunk_capacity = self.linear_launch.chunk_capacity().ok_or_else(|| {
            ProviderExecutionError::from(anyhow!("random child launch extent overflow"))
        })?;
        if (request.precision == ProviderPrecision::F64 && self.precision != NumericPrecision::F64)
            || request.reservation.algorithm != GpuRandomAlgorithm::Threefry4x64_20
            || request.reservation.normal_transform != GpuNormalTransform::BoxMuller
            || count > u32::MAX as usize
        {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "GPU random request is not supported by this physical provider",
            )
            .into());
        }
        let (element_type, element_size) = match request.precision {
            ProviderPrecision::F32 => (NumericElementType::F32, std::mem::size_of::<f32>()),
            ProviderPrecision::F64 => (NumericElementType::F64, std::mem::size_of::<f64>()),
        };
        let buffer = self
            .with_emission(|emission, _| {
                self.create_storage_buffer_checked_with_element_size(
                    emission,
                    count,
                    "runmat-random-child-out",
                    BufferUsageClass::Generic,
                    element_size,
                )
            })
            .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?
            .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
        if buffer.size() > physical.output_buffer_bytes {
            return Err(ProviderExecutionError::from(anyhow!(
                "GPU random output exceeded its physical proposal"
            ))
            .with_unproven_effects());
        }
        let operation = runmat_accelerate_api::ProviderOperationIdentity::new(
            runmat_accelerate_api::random::PROVIDER_GPU_RANDOM_OPERATION,
        );
        let observed = runmat_accelerate_api::observation::current_provider_observation()
            .filter(|recorder| recorder.expects_operation(&operation));
        let before_dispatch = observed.as_ref().map(
            runmat_accelerate_api::observation::ProviderOperationRecorder::submission_checkpoint,
        );
        if count != 0 {
            let pipeline = match (request.precision, request.distribution, request.storage) {
                (ProviderPrecision::F32, GpuRandomDistribution::Uniform, _) => {
                    &self.pipelines.random_child_uniform_f32
                }
                (
                    ProviderPrecision::F32,
                    GpuRandomDistribution::StandardNormal,
                    GpuTensorStorage::Real,
                ) => &self.pipelines.random_child_normal_f32,
                (
                    ProviderPrecision::F32,
                    GpuRandomDistribution::StandardNormal,
                    GpuTensorStorage::ComplexInterleaved,
                ) => &self.pipelines.random_child_complex_normal_f32,
                (ProviderPrecision::F64, GpuRandomDistribution::Uniform, _) => {
                    &self.pipelines.random_child_uniform_f64
                }
                (
                    ProviderPrecision::F64,
                    GpuRandomDistribution::StandardNormal,
                    GpuTensorStorage::Real,
                ) => &self.pipelines.random_child_normal_f64,
                (
                    ProviderPrecision::F64,
                    GpuRandomDistribution::StandardNormal,
                    GpuTensorStorage::ComplexInterleaved,
                ) => &self.pipelines.random_child_complex_normal_f64,
            };
            self.with_emission(|emission, pending| -> Result<()> {
                let mut offset = 0usize;
                while offset < count {
                    let chunk = (count - offset).min(chunk_capacity);
                    let offset_u32 = u32::try_from(offset)?;
                    let chunk_u32 = u32::try_from(chunk)?;
                    let first = request.reservation.first;
                    let seed = request.reservation.seed;
                    let params = crate::backend::wgpu::params::RandomChildParams {
                        offset: offset_u32,
                        chunk: chunk_u32,
                        first0: first as u32,
                        first1: (first >> 32) as u32,
                        first2: (first >> 64) as u32,
                        first3: (first >> 96) as u32,
                        seed0: seed as u32,
                        seed1: (seed >> 32) as u32,
                        integer_lower: 0,
                        integer_mask: 0,
                        _pad0: 0,
                        _pad1: 0,
                    };
                    let params_buffer =
                        self.uniform_buffer(emission, &params, "runmat-random-child-params")?;
                    let compiled = pipeline.get(emission)?;
                    let bind_group =
                        emission
                            .device()
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("runmat-random-child-bind"),
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
                        "runmat-random-child-encoder",
                        "runmat-random-child-pass",
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
                super::super::backend_types::NumericBufferRegistration {
                    shape: request.shape.to_vec(),
                    len: count,
                    physical_element_type: element_type,
                    storage: request.storage,
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
