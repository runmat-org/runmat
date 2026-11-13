//! Deterministic in-process implementation of the same request-bound GPU
//! child contract. This provider has no finite host-metadata receipt, so the
//! Runtime owner must admit it only when its host policy is unbounded.

use super::{InProcessNumericData, InProcessProvider};
use runmat_accelerate_api::random::{
    GpuIntegerRequest, GpuRandomAlgorithm, GpuRandomDistribution, GpuRandomPhysicalProposal,
    GpuRandomRequest, PROVIDER_GPU_RANDOM_OPERATION,
};
use runmat_accelerate_api::{
    GpuTensorHandle, NumericElementType, ProviderExecutionError, ProviderExecutionResult,
    ProviderPrecision,
};
use runmat_execution::resource::ProviderHostMetadataAdmission;

impl InProcessProvider {
    pub(super) fn propose_child_integer(
        &self,
        request: &GpuIntegerRequest<'_>,
    ) -> Option<GpuRandomPhysicalProposal> {
        let count = request.element_count()?;
        request.fixed_work_32()?;
        let width = match request.class {
            NumericElementType::F64 => 8,
            NumericElementType::F32 | NumericElementType::I32 | NumericElementType::U32 => 4,
            _ => return None,
        };
        Some(GpuRandomPhysicalProposal {
            output_buffer_bytes: u64::try_from(count).ok()?.checked_mul(width)?,
            uniform_buffer_bytes: 0,
            uniform_allocations: 0,
            dispatches: u32::from(count != 0),
            queue_submissions: 0,
        })
    }

    pub(super) fn execute_child_integer(
        &self,
        request: &GpuIntegerRequest<'_>,
        metadata: Option<&dyn ProviderHostMetadataAdmission>,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        if metadata.is_some() {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "in-process integer random has no finite metadata registration owner",
            )
            .into());
        }
        let count = request.element_count().ok_or_else(|| {
            ProviderExecutionError::from(runmat_accelerate_api::unsupported_provider_operation(
                "invalid child integer shape",
            ))
        })?;
        let (lower, mask) = request.fixed_work_32().ok_or_else(|| {
            ProviderExecutionError::from(runmat_accelerate_api::unsupported_provider_operation(
                "no fixed-work child integer route",
            ))
        })?;
        self.propose_child_integer(request).ok_or_else(|| {
            ProviderExecutionError::from(runmat_accelerate_api::unsupported_provider_operation(
                "no child integer physical quote",
            ))
        })?;
        let mut words = Vec::new();
        words
            .try_reserve_exact(count)
            .map_err(ProviderExecutionError::from_source)?;
        for item in 0..count {
            let word = request.reservation.word(item as u64, 0).ok_or_else(|| {
                ProviderExecutionError::from(anyhow::anyhow!("child integer reservation exhausted"))
            })?;
            words.push(lower.wrapping_add((word as u32) & mask));
        }
        let data = match request.class {
            NumericElementType::F32 => InProcessNumericData::F32(
                words.into_iter().map(|word| (word as i32) as f32).collect(),
            ),
            NumericElementType::F64 => InProcessNumericData::F64(
                words.into_iter().map(|word| (word as i32) as f64).collect(),
            ),
            NumericElementType::I32 => {
                InProcessNumericData::Integer(super::HostIntegerDataOwned::I32(
                    words.into_iter().map(|word| word as i32).collect(),
                ))
            }
            NumericElementType::U32 => {
                InProcessNumericData::Integer(super::HostIntegerDataOwned::U32(words))
            }
            _ => unreachable!("fixed-work class gate is exhaustive"),
        };
        let output = self
            .allocate_numeric_data(
                data,
                request.shape.to_vec(),
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .map_err(|source| ProviderExecutionError::from(source).with_unproven_effects())?;
        let operation = runmat_accelerate_api::ProviderOperationIdentity::new(
            runmat_accelerate_api::random::PROVIDER_GPU_INTEGER_OPERATION,
        );
        if let Some(recorder) = runmat_accelerate_api::observation::current_provider_observation()
            .filter(|recorder| recorder.expects_operation(&operation))
        {
            recorder.confirm_provider_execution(self.device_id, &operation);
        }
        Ok(output)
    }
    pub(super) fn propose_child_random(
        &self,
        request: &GpuRandomRequest<'_>,
    ) -> Option<GpuRandomPhysicalProposal> {
        let count = request.element_count()?;
        if request.reservation.algorithm != GpuRandomAlgorithm::Threefry4x64_20 {
            return None;
        }
        let bytes_per_component = match request.precision {
            ProviderPrecision::F32 => 4_u64,
            ProviderPrecision::F64 => 8_u64,
        };
        let output_buffer_bytes = u64::try_from(count)
            .ok()?
            .checked_mul(bytes_per_component)?;
        Some(GpuRandomPhysicalProposal {
            output_buffer_bytes,
            uniform_buffer_bytes: 0,
            uniform_allocations: 0,
            dispatches: u32::from(count != 0),
            queue_submissions: 0,
        })
    }

    pub(super) fn execute_child_random(
        &self,
        request: &GpuRandomRequest<'_>,
        metadata: Option<&dyn ProviderHostMetadataAdmission>,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        if metadata.is_some() {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "in-process random has no finite metadata registration owner",
            )
            .into());
        }
        let count = request.element_count().ok_or_else(|| {
            ProviderExecutionError::from(runmat_accelerate_api::unsupported_provider_operation(
                "in-process random geometry differs from reservation",
            ))
        })?;
        self.propose_child_random(request).ok_or_else(|| {
            ProviderExecutionError::from(runmat_accelerate_api::unsupported_provider_operation(
                "in-process random has no exact physical proposal",
            ))
        })?;
        let data = match request.precision {
            ProviderPrecision::F32 => {
                let mut values = Vec::new();
                values
                    .try_reserve_exact(count)
                    .map_err(ProviderExecutionError::from_source)?;
                for item in 0..count {
                    let item = u64::try_from(item).map_err(ProviderExecutionError::from_source)?;
                    let value = match request.distribution {
                        GpuRandomDistribution::Uniform => request.reservation.uniform_f32(item),
                        GpuRandomDistribution::StandardNormal => request.normal_component_f32(item),
                    }
                    .ok_or_else(|| {
                        ProviderExecutionError::from(anyhow::anyhow!(
                            "in-process random reservation exhausted during generation"
                        ))
                    })?;
                    values.push(value);
                }
                InProcessNumericData::F32(values)
            }
            ProviderPrecision::F64 => {
                let mut values = Vec::new();
                values
                    .try_reserve_exact(count)
                    .map_err(ProviderExecutionError::from_source)?;
                for item in 0..count {
                    let item = u64::try_from(item).map_err(ProviderExecutionError::from_source)?;
                    let value = match request.distribution {
                        GpuRandomDistribution::Uniform => request.reservation.uniform_f64(item),
                        GpuRandomDistribution::StandardNormal => request.normal_component_f64(item),
                    }
                    .ok_or_else(|| {
                        ProviderExecutionError::from(anyhow::anyhow!(
                            "in-process random reservation exhausted during generation"
                        ))
                    })?;
                    values.push(value);
                }
                InProcessNumericData::F64(values)
            }
        };
        let output = self
            .allocate_numeric_data(data, request.shape.to_vec(), request.storage)
            .map_err(|source| ProviderExecutionError::from(source).with_unproven_effects())?;
        let operation =
            runmat_accelerate_api::ProviderOperationIdentity::new(PROVIDER_GPU_RANDOM_OPERATION);
        if let Some(recorder) = runmat_accelerate_api::observation::current_provider_observation()
            .filter(|recorder| recorder.expects_operation(&operation))
        {
            recorder.confirm_provider_execution(self.device_id, &operation);
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_accelerate_api::random::{GpuNormalTransform, GpuRandomReservation};
    use runmat_accelerate_api::{AccelProvider, GpuTensorStorage, HostNumericDataOwned};

    #[test]
    fn complex_normal_scales_the_same_reserved_components_in_both_precisions() {
        let provider = InProcessProvider::new();
        let reservation = GpuRandomReservation {
            algorithm: GpuRandomAlgorithm::Threefry4x64_20,
            seed: 17,
            first: 9,
            count: 8,
            normal_transform: GpuNormalTransform::BoxMuller,
        };
        for precision in [ProviderPrecision::F32, ProviderPrecision::F64] {
            let real = GpuRandomRequest {
                shape: &[1, 8],
                precision,
                storage: GpuTensorStorage::Real,
                distribution: GpuRandomDistribution::StandardNormal,
                reservation,
            };
            let complex = GpuRandomRequest {
                shape: &[1, 4],
                storage: GpuTensorStorage::ComplexInterleaved,
                ..real
            };
            let real_output = provider
                .execute_child_random(&real, None)
                .expect("real normal");
            let complex_output = provider
                .execute_child_random(&complex, None)
                .expect("complex normal");
            let real_data = block_on(provider.download_numeric(&real_output))
                .expect("real download")
                .data;
            let complex_data = block_on(provider.download_numeric(&complex_output))
                .expect("complex download")
                .data;
            match (real_data, complex_data) {
                (HostNumericDataOwned::F32(real), HostNumericDataOwned::F32(complex)) => {
                    assert_eq!(real.len(), 8);
                    for (item, (&ordinary, &component)) in real.iter().zip(&complex).enumerate() {
                        assert_eq!(ordinary, reservation.normal_f32(item as u64).unwrap());
                        assert_eq!(component, ordinary * std::f32::consts::FRAC_1_SQRT_2);
                    }
                }
                (HostNumericDataOwned::F64(real), HostNumericDataOwned::F64(complex)) => {
                    assert_eq!(real.len(), 8);
                    for (item, (&ordinary, &component)) in real.iter().zip(&complex).enumerate() {
                        assert_eq!(ordinary, reservation.normal_f64(item as u64).unwrap());
                        assert_eq!(component, ordinary * std::f64::consts::FRAC_1_SQRT_2);
                    }
                }
                classes => panic!("normal output class mismatch: {classes:?}"),
            }
        }
    }
}
