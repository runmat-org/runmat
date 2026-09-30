use super::*;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct NormParams {
    shape: [u32; 4],
    rank: u32,
    reduce_mask: u32,
    groups: u32,
    group_size: u32,
    channel_axis: u32,
    total: u32,
    chunk_offset: u32,
    chunk_count: u32,
    epsilon_low: u32,
    epsilon_high: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Chunk {
    offset: u32,
    count: u32,
    workgroups: u32,
}

fn next_chunk(total: u32, offset: u32) -> Option<Chunk> {
    let lanes =
        crate::backend::wgpu::shaders::deep_learning_normalization::NORMALIZATION_WORKGROUP_SIZE;
    let capacity = crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS.saturating_mul(lanes);
    if offset == total {
        return None;
    }
    let count = (total - offset).min(capacity);
    Some(Chunk {
        offset,
        count,
        workgroups: count.div_ceil(lanes),
    })
}

fn shape_count(shape: &[usize]) -> Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |count, &extent| count.checked_mul(extent))
        .ok_or_else(|| anyhow!("normalization shape product overflows"))
}

fn geometry(request: &DeepLearningLayerNormRequest) -> Result<(usize, usize)> {
    ensure!(
        (1..=4).contains(&request.shape.len()),
        "normalization rank must be one through four"
    );
    ensure!(
        request.channel_axis < request.shape.len(),
        "normalization channel axis exceeds rank"
    );
    ensure!(
        request.reduce_mask != 0 && (request.reduce_mask as usize >> request.shape.len()) == 0,
        "normalization reduction mask is invalid"
    );
    ensure!(
        request.reduce_mask & (1 << request.channel_axis) != 0,
        "normalization must reduce its channel axis"
    );
    ensure!(
        request.epsilon.is_finite() && request.epsilon > 0.0,
        "normalization epsilon must be positive and finite"
    );
    let channels = request.shape[request.channel_axis];
    let mut groups = 1usize;
    let mut group_size = 1usize;
    for (axis, &extent) in request.shape.iter().enumerate() {
        let target = if request.reduce_mask & (1 << axis) != 0 {
            &mut group_size
        } else {
            &mut groups
        };
        *target = target
            .checked_mul(extent)
            .ok_or_else(|| anyhow!("normalization geometry overflows"))?;
    }
    ensure!(
        channels != 0 && group_size != 0,
        "normalization reduction is empty"
    );
    let row = [1, channels];
    let column = [channels, 1];
    ensure!(
        request.scale_shape == request.offset_shape
            && (request.scale_shape.as_slice() == row.as_slice()
                || request.scale_shape.as_slice() == column.as_slice()),
        "normalization Scale/Offset shape mismatch"
    );
    Ok((groups, group_size))
}

impl WgpuProvider {
    pub(crate) fn deep_learning_layer_norm_capability_exec(
        &self,
        input: &GpuTensorHandle,
        scale: Option<&GpuTensorHandle>,
        offset: Option<&GpuTensorHandle>,
        request: &DeepLearningLayerNormRequest,
    ) -> Result<DeepLearningLayerNormCapability> {
        let (groups, group_size) = geometry(request)?;
        ensure!(
            input.shape() == request.shape,
            "normalization input shape differs from admitted geometry"
        );
        if let Some(scale) = scale {
            ensure!(
                scale.shape() == request.scale_shape,
                "normalization Scale descriptor shape mismatch"
            );
        }
        if let Some(offset) = offset {
            ensure!(
                offset.shape() == request.offset_shape,
                "normalization Offset descriptor shape mismatch"
            );
        }
        let total = shape_count(&request.shape)?;
        for handle in std::iter::once(input).chain(scale).chain(offset) {
            let entry = self.get_entry_raw(handle)?;
            ensure!(
                entry.element_type == handle.physical().element_type()
                    && entry.storage == handle.physical().storage()
                    && entry.len == shape_count(handle.shape())?
                    && entry.logical == runmat_accelerate_api::handle_is_logical(handle)
                    && entry.integer_type() == runmat_accelerate_api::handle_integer_type(handle),
                "normalization handle differs from provider-owned descriptor"
            );
            if entry.storage != GpuTensorStorage::Real
                || entry.logical
                || entry.integer_type().is_some()
                || entry.precision != self.precision
            {
                return Ok(DeepLearningLayerNormCapability::UnsupportedOperation);
            }
        }
        if self.precision == crate::backend::wgpu::types::NumericPrecision::F32
            && (!(request.epsilon as f32).is_finite() || request.epsilon as f32 == 0.0)
        {
            return Ok(DeepLearningLayerNormCapability::UnsupportedOperation);
        }
        if total > u32::MAX as usize
            || groups > u32::MAX as usize
            || group_size > u32::MAX as usize
            || groups.checked_mul(3).is_none_or(|n| n > u32::MAX as usize)
            || request
                .shape
                .iter()
                .any(|&extent| extent > u32::MAX as usize)
        {
            return Ok(DeepLearningLayerNormCapability::ResourceLimit);
        }
        let binding_limit = u64::from(self.adapter_limits.max_storage_buffer_binding_size)
            .min(self.adapter_limits.max_buffer_size);
        let element_size = u64::try_from(self.element_size)?;
        if [total, groups * 3, request.shape[request.channel_axis]]
            .iter()
            .any(|&count| {
                u64::try_from(count)
                    .ok()
                    .and_then(|n| n.checked_mul(element_size))
                    .is_none_or(|bytes| bytes > binding_limit)
            })
        {
            return Ok(DeepLearningLayerNormCapability::ResourceLimit);
        }
        Ok(DeepLearningLayerNormCapability::Ready)
    }

    pub(crate) fn deep_learning_layer_norm_exec(
        &self,
        input: &GpuTensorHandle,
        scale: &GpuTensorHandle,
        offset: &GpuTensorHandle,
        request: &DeepLearningLayerNormRequest,
    ) -> Result<GpuTensorHandle> {
        ensure!(
            self.deep_learning_layer_norm_capability_exec(
                input,
                Some(scale),
                Some(offset),
                request
            )? == DeepLearningLayerNormCapability::Ready,
            "normalization route was not admitted by this provider"
        );
        let (groups, group_size) = geometry(request)?;
        let total = shape_count(&request.shape)?;
        let mut shape = [1u32; 4];
        for (target, &extent) in shape.iter_mut().zip(&request.shape) {
            *target = extent as u32;
        }
        let epsilon_bits = match self.precision {
            crate::backend::wgpu::types::NumericPrecision::F32 => {
                (request.epsilon as f32).to_bits() as u64
            }
            crate::backend::wgpu::types::NumericPrecision::F64 => request.epsilon.to_bits(),
        };
        let base = NormParams {
            shape,
            rank: request.shape.len() as u32,
            reduce_mask: request.reduce_mask as u32,
            groups: groups as u32,
            group_size: group_size as u32,
            channel_axis: request.channel_axis as u32,
            total: total as u32,
            chunk_offset: 0,
            chunk_count: 0,
            epsilon_low: epsilon_bits as u32,
            epsilon_high: (epsilon_bits >> 32) as u32,
        };
        if total == 0 {
            let output = self.with_emission(|emission, _| {
                self.create_storage_buffer_checked(emission, 0, "runmat-layer-norm-empty")
            })??;
            return self.register_existing_buffer_with_storage(
                output,
                request.shape.clone(),
                0,
                GpuTensorStorage::Real,
            );
        }
        let input_materialized = self.materialize_nd_reduction_input(input)?;
        let scale_materialized = self.materialize_nd_reduction_input(scale)?;
        let offset_materialized = self.materialize_nd_reduction_input(offset)?;
        let source = self.get_entry(input_materialized.as_ref().unwrap_or(input))?;
        let gamma = self.get_entry(scale_materialized.as_ref().unwrap_or(scale))?;
        let beta = self.get_entry(offset_materialized.as_ref().unwrap_or(offset))?;
        let output = self.with_emission(|emission, pending| -> Result<_> {
            let statistics = self.create_storage_buffer_checked(
                emission,
                groups * 3,
                "runmat-layer-norm-statistics",
            )?;
            let output =
                self.create_storage_buffer_checked(emission, total, "runmat-layer-norm-output")?;
            let stats_pipeline = self
                .pipelines
                .deep_learning_layer_norm_statistics
                .get(emission)?;
            let affine_pipeline = self
                .pipelines
                .deep_learning_layer_norm_affine
                .get(emission)?;
            let mut offset = 0u32;
            while let Some(chunk) = next_chunk(groups as u32, offset) {
                let params = self.uniform_buffer(
                    emission,
                    &NormParams {
                        chunk_offset: chunk.offset,
                        chunk_count: chunk.count,
                        ..base
                    },
                    "runmat-layer-norm-stat-params",
                )?;
                let bind = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-layer-norm-stat-bind"),
                        layout: &stats_pipeline.layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: source.buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: statistics.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: params.as_entire_binding(),
                            },
                        ],
                    });
                crate::backend::wgpu::dispatch::elementwise::run(
                    emission,
                    pending,
                    &stats_pipeline.pipeline,
                    &bind,
                    chunk.workgroups,
                )?;
                offset += chunk.count;
            }
            offset = 0;
            while let Some(chunk) = next_chunk(total as u32, offset) {
                let params = self.uniform_buffer(
                    emission,
                    &NormParams {
                        chunk_offset: chunk.offset,
                        chunk_count: chunk.count,
                        ..base
                    },
                    "runmat-layer-norm-affine-params",
                )?;
                let bind = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-layer-norm-affine-bind"),
                        layout: &affine_pipeline.layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: source.buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: gamma.buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: beta.buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: statistics.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: output.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: params.as_entire_binding(),
                            },
                        ],
                    });
                crate::backend::wgpu::dispatch::elementwise::run(
                    emission,
                    pending,
                    &affine_pipeline.pipeline,
                    &bind,
                    chunk.workgroups,
                )?;
                offset += chunk.count;
            }
            Ok(output)
        })??;
        self.register_existing_buffer_with_storage(
            output,
            request.shape.clone(),
            total,
            GpuTensorStorage::Real,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_cover_each_output_once() {
        for total in [0, 1, 63, 64, 65, 4097] {
            let mut seen = vec![0; total];
            let mut offset = 0;
            while let Some(chunk) = next_chunk(total as u32, offset) {
                for local in 0..chunk.workgroups * 64 {
                    if local < chunk.count {
                        seen[(chunk.offset + local) as usize] += 1;
                    }
                }
                offset += chunk.count;
            }
            assert!(seen.iter().all(|&visits| visits == 1));
        }
    }

    #[test]
    fn typed_geometry_rejects_malformed_reduction_and_parameter_shapes() {
        let mut request = DeepLearningLayerNormRequest {
            shape: vec![2, 3, 2],
            reduce_mask: 0b010,
            channel_axis: 1,
            scale_shape: vec![3, 1],
            offset_shape: vec![3, 1],
            epsilon: 1e-5,
        };
        assert_eq!(geometry(&request).unwrap(), (4, 3));
        request.reduce_mask = 0b001;
        assert!(geometry(&request).is_err());
        request.reduce_mask = 0b010;
        request.offset_shape = vec![1, 2];
        assert!(geometry(&request).is_err());
        request.offset_shape = vec![3, 1];
        request.epsilon = f64::NAN;
        assert!(geometry(&request).is_err());
    }

    #[test]
    #[ignore = "requires actual WGPU device; run explicitly for native precision qualification"]
    fn resident_channel_normalization_matches_independent_cb_oracle() {
        use runmat_accelerate_api::{AccelProvider, HostTensorView};

        let provider = WgpuProvider::new(Default::default()).expect("actual WGPU device required");
        let input = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                shape: &[3, 2],
            })
            .unwrap();
        let scale = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0, 3.0],
                shape: &[3, 1],
            })
            .unwrap();
        let offset = provider
            .upload(&HostTensorView {
                data: &[0.5, -0.5, 1.0],
                shape: &[3, 1],
            })
            .unwrap();
        let request = DeepLearningLayerNormRequest {
            shape: vec![3, 2],
            reduce_mask: 0b01,
            channel_axis: 0,
            scale_shape: vec![3, 1],
            offset_shape: vec![3, 1],
            epsilon: 1e-5,
        };
        assert_eq!(
            provider
                .deep_learning_layer_norm_capability_exec(
                    &input,
                    Some(&scale),
                    Some(&offset),
                    &request
                )
                .unwrap(),
            DeepLearningLayerNormCapability::Ready
        );
        provider.reset_telemetry();
        let output = provider
            .deep_learning_layer_norm_exec(&input, &scale, &offset, &request)
            .unwrap();
        assert_eq!(output.shape(), &[3, 2]);
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        let values = pollster::block_on(provider.download(&output)).unwrap().data;
        let std = (2.0_f64 / 3.0 + 1e-5).sqrt();
        let expected = [
            -1.0 / std + 0.5,
            -0.5,
            3.0 / std + 1.0,
            -1.0 / std + 0.5,
            -0.5,
            3.0 / std + 1.0,
        ];
        let tolerance = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F32 => 2e-5,
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-11,
        };
        for (actual, expected) in values.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < tolerance,
                "actual {actual}, expected {expected}"
            );
        }
        for handle in [input, scale, offset, output] {
            handle.release().unwrap();
        }
    }
}
