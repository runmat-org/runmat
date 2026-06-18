use anyhow::{anyhow, ensure, Result};
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView, ImfilterOptions, ImfilterPadding};
use runmat_builtins::Tensor;
use std::sync::Arc;
use wgpu::util::DeviceExt;

use super::WgpuProvider;
use crate::backend::wgpu::params::{
    ImageNormalizeUniforms, IMAGE_NORMALIZE_FLAG_BIAS, IMAGE_NORMALIZE_FLAG_CLAMP_ZERO,
    IMAGE_NORMALIZE_FLAG_GAIN, IMAGE_NORMALIZE_FLAG_GAMMA,
};
use crate::backend::wgpu::residency::BufferUsageClass;
use crate::backend::wgpu::resources::UniformBufferKey;
use crate::backend::wgpu::types::NumericPrecision;
use runmat_runtime::builtins::image::filters::imfilter::{
    apply_imfilter_tensor as runtime_apply_imfilter_tensor, build_imfilter_plan,
};

impl WgpuProvider {
    pub(crate) async fn imfilter_exec(
        &self,
        image: &GpuTensorHandle,
        kernel: &GpuTensorHandle,
        options: &ImfilterOptions,
    ) -> Result<GpuTensorHandle> {
        if std::env::var("RUNMAT_WGPU_DISABLE_IMFILTER")
            .ok()
            .and_then(|v| match v.trim().to_ascii_lowercase().as_str() {
                "1" | "true" | "yes" => Some(true),
                "0" | "false" | "no" => Some(false),
                _ => None,
            })
            .unwrap_or(false)
        {
            return self.imfilter_exec_fallback(image, kernel, options).await;
        }
        let image_entry = self.get_entry(image)?;
        let kernel_host = self.download_exec(kernel).await?;
        let kernel_tensor = Tensor::new(kernel_host.data.clone(), kernel_host.shape.clone())
            .map_err(|e| anyhow!("imfilter: {e}"))?;

        let image_shape = if image_entry.shape.is_empty() {
            vec![1usize]
        } else {
            image_entry.shape.clone()
        };

        let plan = match build_imfilter_plan(&image_shape, &kernel_tensor, options, "imfilter") {
            Ok(plan) => plan,
            Err(err) => return Err(anyhow!("{err}")),
        };

        if plan.rank > crate::backend::wgpu::params::IMFILTER_MAX_RANK {
            return self.imfilter_exec_fallback(image, kernel, options).await;
        }

        let image_ext_product = plan
            .image_shape_ext
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| anyhow!("imfilter: image dimensions exceed GPU limits"))?;
        if image_ext_product != image_entry.len {
            return self.imfilter_exec_fallback(image, kernel, options).await;
        }

        let output_len = plan
            .output_shape_ext
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| anyhow!("imfilter: output dimensions exceed GPU limits"))?;
        if output_len > u32::MAX as usize || image_entry.len > u32::MAX as usize {
            return self.imfilter_exec_fallback(image, kernel, options).await;
        }

        let kernel_points_len = plan.kernel_points.len();
        if kernel_points_len > u32::MAX as usize {
            return self.imfilter_exec_fallback(image, kernel, options).await;
        }

        let mut kernel_offsets = Vec::with_capacity(kernel_points_len * plan.rank);
        let mut kernel_values_f64 = Vec::with_capacity(kernel_points_len);
        for point in &plan.kernel_points {
            if point.offsets.len() != plan.rank {
                return self.imfilter_exec_fallback(image, kernel, options).await;
            }
            for &offset in &point.offsets {
                if offset < i32::MIN as isize || offset > i32::MAX as isize {
                    return self.imfilter_exec_fallback(image, kernel, options).await;
                }
                kernel_offsets.push(offset as i32);
            }
            kernel_values_f64.push(point.value);
        }

        if kernel_offsets.len() > u32::MAX as usize {
            return self.imfilter_exec_fallback(image, kernel, options).await;
        }

        let kernel_offsets_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("runmat-imfilter-kernel-offsets"),
                    contents: bytemuck::cast_slice(&kernel_offsets),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let kernel_values_buffer = match self.precision {
            NumericPrecision::F64 => {
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("runmat-imfilter-kernel-values-f64"),
                        contents: bytemuck::cast_slice(&kernel_values_f64),
                        usage: wgpu::BufferUsages::STORAGE,
                    })
            }
            NumericPrecision::F32 => {
                let kernel_values_f32: Vec<f32> =
                    kernel_values_f64.iter().map(|&v| v as f32).collect();
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("runmat-imfilter-kernel-values-f32"),
                        contents: bytemuck::cast_slice(&kernel_values_f32),
                        usage: wgpu::BufferUsages::STORAGE,
                    })
            }
        };

        let out_buffer = self.create_storage_buffer_checked(output_len, "runmat-imfilter-out")?;
        if output_len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, plan.final_shape.clone(), 0));
        }

        let mut image_shape_arr = [crate::backend::wgpu::params::AlignedU32::default();
            crate::backend::wgpu::params::IMFILTER_MAX_RANK];
        let mut image_strides_arr = [crate::backend::wgpu::params::AlignedU32::default();
            crate::backend::wgpu::params::IMFILTER_MAX_RANK];
        let mut output_shape_arr = [crate::backend::wgpu::params::AlignedU32::default();
            crate::backend::wgpu::params::IMFILTER_MAX_RANK];
        let mut base_offset_arr = [crate::backend::wgpu::params::PackedI32::default();
            crate::backend::wgpu::params::IMFILTER_MAX_RANK];

        for i in 0..plan.rank {
            let dim = plan.image_shape_ext[i];
            ensure!(
                dim <= u32::MAX as usize,
                "imfilter: image dimension exceeds GPU limits"
            );
            image_shape_arr[i] = crate::backend::wgpu::params::AlignedU32::new(dim as u32);

            let stride = plan.image_strides[i];
            ensure!(
                stride <= u32::MAX as usize,
                "imfilter: image stride exceeds GPU limits"
            );
            image_strides_arr[i] = crate::backend::wgpu::params::AlignedU32::new(stride as u32);

            let out_dim = plan.output_shape_ext[i];
            ensure!(
                out_dim <= u32::MAX as usize,
                "imfilter: output dimension exceeds GPU limits"
            );
            output_shape_arr[i] = crate::backend::wgpu::params::AlignedU32::new(out_dim as u32);

            let offset = plan.base_offset[i];
            ensure!(
                offset >= i32::MIN as isize && offset <= i32::MAX as isize,
                "imfilter: base offset exceeds GPU limits"
            );
            base_offset_arr[i] =
                crate::backend::wgpu::params::PackedI32::from_scalar(offset as i32);
        }

        let padding_mode = match options.padding {
            ImfilterPadding::Constant => 0u32,
            ImfilterPadding::Replicate => 1u32,
            ImfilterPadding::Symmetric => 2u32,
            ImfilterPadding::Circular => 3u32,
        };

        let kernel_points_u32 = kernel_points_len as u32;
        let image_len_u32 = image_entry.len as u32;

        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;

        {
            let mut enc =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-imfilter-noop"),
                    });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-imfilter-noop-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.imfilter.pipeline);
            drop(pass);
            self.submit(enc);
        }

        self.device_ref().poll(wgpu::Maintain::Poll);
        {
            let enc = self
                .device_ref()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-imfilter-flush-gap"),
                });
            self.submit(enc);
        }

        let mut offset = 0usize;
        while offset < output_len {
            let remaining = output_len - offset;
            let chunk_len = remaining.min(chunk_capacity);
            let offset_u32 = offset as u32;
            let chunk_u32 = chunk_len as u32;

            let params_buffer = match self.precision {
                NumericPrecision::F64 => {
                    let params = crate::backend::wgpu::params::ImfilterParamsF64 {
                        len: chunk_u32,
                        offset: offset_u32,
                        rank: plan.rank as u32,
                        padding: padding_mode,
                        kernel_points: kernel_points_u32,
                        image_len: image_len_u32,
                        _pad0: 0,
                        _pad1: 0,
                        constant_value: options.constant_value,
                        _pad_const: 0.0,
                        image_shape: image_shape_arr,
                        image_strides: image_strides_arr,
                        output_shape: output_shape_arr,
                        base_offset: base_offset_arr,
                        _pad_tail: crate::backend::wgpu::params::AlignedU32::default(),
                    };

                    self.uniform_buffer(&params, "runmat-imfilter-params-f64")
                }
                NumericPrecision::F32 => {
                    let params = crate::backend::wgpu::params::ImfilterParamsF32 {
                        len: chunk_u32,
                        offset: offset_u32,
                        rank: plan.rank as u32,
                        padding: padding_mode,
                        kernel_points: kernel_points_u32,
                        image_len: image_len_u32,
                        _pad0: 0,
                        _pad1: 0,
                        constant_value: options.constant_value as f32,
                        _pad_const: [0.0; 3],
                        image_shape: image_shape_arr,
                        image_strides: image_strides_arr,
                        output_shape: output_shape_arr,
                        base_offset: base_offset_arr,
                        _pad_tail: crate::backend::wgpu::params::AlignedU32::default(),
                    };

                    self.uniform_buffer(&params, "runmat-imfilter-params-f32")
                }
            };

            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-imfilter-bind"),
                    layout: &self.pipelines.imfilter.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: image_entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: kernel_offsets_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: kernel_values_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: out_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });

            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );

            crate::backend::wgpu::dispatch::imfilter::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.imfilter.pipeline,
                &bind_group,
                workgroups,
            );

            offset += chunk_len;
        }

        Ok(self.register_existing_buffer(out_buffer, plan.final_shape.clone(), output_len))
    }

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
                let pipeline = self.image_normalize_pipeline(&tuning)?;

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
                    entry.len,
                    "runmat-image-normalize-out",
                    BufferUsageClass::FusionOut,
                )?;
                let uniform_buf = self.kernel_resources.uniform_buffer(
                    self.device_ref(),
                    UniformBufferKey::ImageNormalizeUniforms,
                    std::mem::size_of::<ImageNormalizeUniforms>() as u64,
                    "runmat-image-normalize-uniform",
                );
                let stream_hot_cap = self
                    .image_normalize_hot_stream_cap(plane_u32, batches_u32)
                    .max(1);
                let cold_cap = stream_hot_cap.min((Self::IMAGE_NORMALIZE_STREAM_COLD_CAP).max(1));
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
                let layout = &self.pipelines.image_normalize.layout;
                let bind_group =
                    self.bind_group_cache
                        .get_or_create(layout, &bind_entries, || {
                            Arc::new(self.device_ref().create_bind_group(
                                &wgpu::BindGroupDescriptor {
                                    label: Some("runmat-image-normalize-bind"),
                                    layout,
                                    entries: &bind_entries,
                                },
                            ))
                        });

                let mut offset = 0u32;
                while offset < batches_u32 {
                    let remaining = batches_u32 - offset;
                    let chunk = remaining.min(chunk_limit).max(1);
                    uniforms.batch_count = chunk;
                    uniforms.batch_offset = offset;
                    self.queue
                        .write_buffer(uniform_buf.as_ref(), 0, bytemuck::bytes_of(&uniforms));
                    crate::backend::wgpu::dispatch::image_normalize::run(
                        self.device_ref(),
                        self.queue_ref(),
                        pipeline.as_ref(),
                        bind_group.as_ref(),
                        chunk,
                        tuning.batch_tile,
                    );
                    offset += chunk;
                }

                Ok(self.register_existing_buffer_with_usage(
                    out_buffer,
                    entry.shape.clone(),
                    entry.len,
                    BufferUsageClass::FusionOut,
                ))
            }
        }
    }

    async fn imfilter_exec_fallback(
        &self,
        image: &GpuTensorHandle,
        kernel: &GpuTensorHandle,
        options: &ImfilterOptions,
    ) -> Result<GpuTensorHandle> {
        let image_host = self.download_exec(image).await?;
        let kernel_host = self.download_exec(kernel).await?;

        let image_tensor = Tensor::new(image_host.data.clone(), image_host.shape.clone())
            .map_err(|e| anyhow!("imfilter: {e}"))?;
        let kernel_tensor = Tensor::new(kernel_host.data.clone(), kernel_host.shape.clone())
            .map_err(|e| anyhow!("imfilter: {e}"))?;

        let result =
            runtime_apply_imfilter_tensor(&image_tensor, &kernel_tensor, options, "imfilter")
                .map_err(|err| anyhow!("{err}"))?;
        let data_owned = result.data;
        let shape_owned = result.shape;
        let view = HostTensorView {
            data: &data_owned,
            shape: &shape_owned,
        };
        let handle = self.upload_exec(&view)?;
        Ok(handle)
    }
}
