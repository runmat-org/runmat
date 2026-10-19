use anyhow::{anyhow, ensure, Result};
use runmat_accelerate_api::GpuTensorHandle;
use std::sync::Arc;

use super::WgpuProvider;
use crate::backend::wgpu::params::{
    ImageNormalizeUniforms, IMAGE_NORMALIZE_FLAG_BIAS, IMAGE_NORMALIZE_FLAG_CLAMP_ZERO,
    IMAGE_NORMALIZE_FLAG_GAIN, IMAGE_NORMALIZE_FLAG_GAMMA,
};
use crate::backend::wgpu::residency::BufferUsageClass;
use crate::backend::wgpu::resources::UniformBufferKey;
use crate::backend::wgpu::types::NumericPrecision;
#[path = "image/imfilter.rs"]
mod imfilter;

impl WgpuProvider {
    pub(crate) async fn image_normalize_exec(
        &self,
        input: &GpuTensorHandle,
        desc: &runmat_accelerate_api::ImageNormalizeDescriptor,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(input)?;
        ensure!(
            entry.shape.len() == 3,
            "image_normalize: expected 3-D tensor, got {:?}",
            entry.shape
        );
        ensure!(
            entry.shape[0] == desc.batch
                && entry.shape[1] == desc.height
                && entry.shape[2] == desc.width,
            "image_normalize: descriptor dims {:?} do not match tensor shape {:?}",
            (desc.batch, desc.height, desc.width),
            entry.shape
        );

        if entry.len == 0 {
            return self.image_normalize_cpu_fallback(input, desc).await;
        }

        match self.precision {
            NumericPrecision::F64 => self.image_normalize_cpu_fallback(input, desc).await,
            NumericPrecision::F32 => {
                ensure!(
                    desc.epsilon.is_finite(),
                    "image_normalize: epsilon must be finite"
                );
                ensure!(
                    desc.epsilon >= 0.0,
                    "image_normalize: epsilon must be non-negative"
                );

                let batches = entry.shape[0];
                let height = entry.shape[1];
                let width = entry.shape[2];
                let plane = height
                    .checked_mul(width)
                    .ok_or_else(|| anyhow!("image_normalize: height*width overflow"))?;
                ensure!(
                    entry.len == plane * batches,
                    "image_normalize: inconsistent tensor length {} vs dims {:?}",
                    entry.len,
                    entry.shape
                );

                let stride_h = batches;
                let stride_w = batches
                    .checked_mul(height)
                    .ok_or_else(|| anyhow!("image_normalize: stride overflow"))?;

                let batches_u32 = u32::try_from(batches)
                    .map_err(|_| anyhow!("image_normalize: batch size too large"))?;
                let height_u32 = u32::try_from(height)
                    .map_err(|_| anyhow!("image_normalize: height too large"))?;
                let width_u32 = u32::try_from(width)
                    .map_err(|_| anyhow!("image_normalize: width too large"))?;
                let plane_u32 = u32::try_from(plane)
                    .map_err(|_| anyhow!("image_normalize: plane size too large"))?;
                let stride_h_u32 = u32::try_from(stride_h)
                    .map_err(|_| anyhow!("image_normalize: stride_h too large"))?;
                let stride_w_u32 = u32::try_from(stride_w)
                    .map_err(|_| anyhow!("image_normalize: stride_w too large"))?;
                let (tuning, cache_hit) =
                    self.resolve_image_normalize_tuning(batches_u32, plane_u32);
                log::debug!(
                    "image_normalize tuning batches={} plane={} lane={} spatial={} values/thread={} cache_hit={}",
                    batches_u32,
                    plane_u32,
                    tuning.lane_count,
                    tuning.spatial_tile,
                    tuning.values_per_thread,
                    cache_hit
                );
                let out_buffer = self.with_emission(|emission, pending| -> Result<_> {
                    let pipeline = self.image_normalize_pipeline(emission, &tuning)?;

                    let mut flags = 0u32;
                    if desc.gain.is_some() {
                        flags |= IMAGE_NORMALIZE_FLAG_GAIN;
                    }
                    if desc.bias.is_some() {
                        flags |= IMAGE_NORMALIZE_FLAG_BIAS;
                    }
                    if desc.gamma.is_some() {
                        flags |= IMAGE_NORMALIZE_FLAG_GAMMA;
                    }
                    if desc.clamp_zero {
                        flags |= IMAGE_NORMALIZE_FLAG_CLAMP_ZERO;
                    }

                    let mut uniforms = ImageNormalizeUniforms {
                        batch_count: 0,
                        height: height_u32,
                        width: width_u32,
                        plane: plane_u32,
                        stride_h: stride_h_u32,
                        stride_w: stride_w_u32,
                        flags,
                        batch_stride: batches_u32,
                        batch_offset: 0,
                        _pad0: 0,
                        epsilon: desc.epsilon as f32,
                        gain: desc.gain.unwrap_or(1.0) as f32,
                        bias: desc.bias.unwrap_or(0.0) as f32,
                        gamma: desc.gamma.unwrap_or(1.0) as f32,
                        _pad1: 0,
                    };

                    let out_buffer = self.create_storage_buffer_checked_with_usage(
                        emission,
                        entry.len,
                        "runmat-image-normalize-out",
                        BufferUsageClass::FusionOut,
                    )?;
                    let uniform_buf = self.kernel_resources.uniform_buffer(
                        emission,
                        UniformBufferKey::ImageNormalizeUniforms,
                        std::mem::size_of::<ImageNormalizeUniforms>() as u64,
                        "runmat-image-normalize-uniform",
                    )?;
                    let stream_hot_cap = self
                        .image_normalize_hot_stream_cap(plane_u32, batches_u32)
                        .max(1);
                    let cold_cap =
                        stream_hot_cap.min((Self::IMAGE_NORMALIZE_STREAM_COLD_CAP).max(1));
                    let chunk_limit = if cache_hit {
                        stream_hot_cap
                    } else {
                        cold_cap.max(1)
                    };

                    let bind_entries = [
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: out_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: uniform_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: out_buffer.as_ref().as_entire_binding(),
                        },
                    ];
                    let layout = &self.pipelines.image_normalize.get(emission)?.layout;
                    let bind_group = self.bind_group_cache.get_or_create(
                        emission,
                        layout,
                        &bind_entries,
                        |emission| {
                            Arc::new(emission.device().create_bind_group(
                                &wgpu::BindGroupDescriptor {
                                    label: Some("runmat-image-normalize-bind"),
                                    layout,
                                    entries: &bind_entries,
                                },
                            ))
                        },
                    )?;

                    let mut offset = 0u32;
                    while offset < batches_u32 {
                        let remaining = batches_u32 - offset;
                        let chunk = remaining.min(chunk_limit).max(1);
                        uniforms.batch_count = chunk;
                        uniforms.batch_offset = offset;
                        emission.write_buffer(
                            uniform_buf.as_ref(),
                            0,
                            bytemuck::bytes_of(&uniforms),
                        );
                        crate::backend::wgpu::dispatch::image_normalize::run(
                            emission,
                            pending,
                            pipeline.as_ref(),
                            bind_group.as_ref(),
                            chunk,
                            tuning.batch_tile,
                        )?;
                        offset += chunk;
                    }

                    Ok(out_buffer)
                })??;
                self.register_existing_buffer_with_usage(
                    out_buffer,
                    entry.shape.clone(),
                    entry.len,
                    BufferUsageClass::FusionOut,
                )
            }
        }
    }
}
