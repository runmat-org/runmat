use anyhow::{anyhow, ensure, Result};
use bytemuck::{bytes_of, cast_slice, Pod};
use num_complex::Complex;
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, HostTensorOwned, HostTensorView};
use runmat_runtime::builtins::common::shape::normalize_scalar_shape;
use rustfft::FftPlanner;
use std::sync::Arc;

use crate::backend::wgpu::resources::UniformBufferKey;
use crate::backend::wgpu::types::NumericPrecision;

use super::WgpuProvider;

fn fft_trim_trailing_ones(shape: &mut Vec<usize>, minimum_rank: usize) {
    while shape.len() > minimum_rank && shape.last() == Some(&1) {
        shape.pop();
    }
    *shape = normalize_scalar_shape(shape);
}

fn fft_is_power_of_two(len: usize) -> bool {
    len != 0 && (len & (len - 1)) == 0
}

fn fft_log2_pow2(len: usize) -> Option<u32> {
    if !fft_is_power_of_two(len) {
        return None;
    }
    Some(len.trailing_zeros())
}

fn fft_log3_pow3(mut len: usize) -> Option<u32> {
    if len == 0 {
        return None;
    }
    let mut d = 0u32;
    while len > 1 {
        if !len.is_multiple_of(3) {
            return None;
        }
        len /= 3;
        d += 1;
    }
    Some(d)
}

fn fft_log5_pow5(mut len: usize) -> Option<u32> {
    if len == 0 {
        return None;
    }
    let mut d = 0u32;
    while len > 1 {
        if !len.is_multiple_of(5) {
            return None;
        }
        len /= 5;
        d += 1;
    }
    Some(d)
}

fn fft_factor_smooth_235(mut len: usize) -> Option<Vec<u32>> {
    if len <= 1 {
        return None;
    }
    let mut factors = Vec::new();
    while len.is_multiple_of(5) {
        factors.push(5);
        len /= 5;
    }
    while len.is_multiple_of(3) {
        factors.push(3);
        len /= 3;
    }
    while len.is_multiple_of(2) {
        factors.push(2);
        len /= 2;
    }
    if len == 1 && !factors.is_empty() {
        Some(factors)
    } else {
        None
    }
}

impl WgpuProvider {
    fn fft_uniform_buffer<T: Pod>(&self, data: &T, label: &'static str) -> Arc<wgpu::Buffer> {
        let size = std::mem::size_of::<T>() as u64;
        let buffer = self.kernel_resources.uniform_buffer(
            self.device_ref(),
            UniformBufferKey::LenOpParams,
            size,
            label,
        );
        self.queue_ref()
            .write_buffer(buffer.as_ref(), 0, bytes_of(data));
        buffer
    }

    fn fft_storage_param_buffer<T: Pod>(&self, data: &T, label: &str) -> Arc<wgpu::Buffer> {
        let size = std::mem::size_of::<T>() as u64;
        let buffer = Arc::new(self.device_ref().create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(1),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.queue_ref()
            .write_buffer(buffer.as_ref(), 0, bytes_of(data));
        buffer
    }

    fn fft_debug_enabled() -> bool {
        std::env::var_os("RUNMAT_FFT_DEBUG").is_some()
    }

    fn fft_debug_dump_scalar_buffer(
        &self,
        label: &str,
        buffer: &Arc<wgpu::Buffer>,
        scalar_len: usize,
    ) {
        if !Self::fft_debug_enabled() || scalar_len == 0 {
            return;
        }
        let sample_len = scalar_len.min(32);
        let size_bytes = (sample_len as u64).saturating_mul(self.element_size as u64);
        let staging = self.device_ref().create_buffer(&wgpu::BufferDescriptor {
            label: Some("runmat-fft-debug-staging"),
            size: size_bytes.max(1),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder =
            self.device_ref()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-fft-debug-copy"),
                });
        encoder.copy_buffer_to_buffer(buffer.as_ref(), 0, &staging, 0, size_bytes);
        self.submit(encoder);
        let Ok(bytes) = self.map_readback_bytes_sync(staging, size_bytes, "fft-debug") else {
            eprintln!("[fft-debug] {label}: readback failed");
            return;
        };
        match self.precision {
            NumericPrecision::F64 => {
                let vals = bytemuck::cast_slice::<u8, f64>(&bytes);
                eprintln!("[fft-debug] {label} f64 {:?}", vals);
            }
            NumericPrecision::F32 => {
                let vals = bytemuck::cast_slice::<u8, f32>(&bytes);
                eprintln!("[fft-debug] {label} f32 {:?}", vals);
            }
        }
    }

    fn fft_stage_buffer_pair(
        &self,
        len: usize,
        label_a: &str,
        label_b: &str,
    ) -> (Arc<wgpu::Buffer>, Arc<wgpu::Buffer>) {
        let a = self.create_storage_buffer(len, label_a);
        let mut b = self.create_storage_buffer(len, label_b);
        if Arc::ptr_eq(&a, &b) {
            let size_bytes = (len.max(1) as u64).saturating_mul(self.element_size as u64);
            b = Arc::new(self.device_ref().create_buffer(&wgpu::BufferDescriptor {
                label: Some(label_b),
                size: size_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        if Self::fft_debug_enabled() {
            eprintln!(
                "[fft-debug] stage buffers a={:p} b={:p}",
                Arc::as_ptr(&a),
                Arc::as_ptr(&b)
            );
        }
        (a, b)
    }

    fn fft_twiddle_buffer(
        &self,
        len: usize,
        half_only: bool,
        label: &str,
    ) -> Result<Arc<wgpu::Buffer>> {
        let mode = if half_only { 1u8 } else { 0u8 };
        if let Ok(cache) = self.fft_twiddle_cache.lock() {
            if let Some(existing) = cache.get(&(len, mode)) {
                return Ok(existing.clone());
            }
        }

        let count = if half_only { len / 2 } else { len };
        let twiddle_scalar_len = count
            .checked_mul(2)
            .ok_or_else(|| anyhow!("fft_dim: twiddle buffer length overflow"))?;
        let size_bytes =
            (twiddle_scalar_len.max(1) as u64).saturating_mul(self.element_size as u64);
        let twiddle = Arc::new(self.device_ref().create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        let tau = std::f64::consts::TAU;
        match self.precision {
            NumericPrecision::F64 => {
                let mut tw = Vec::with_capacity(twiddle_scalar_len);
                for k in 0..count {
                    let angle = -tau * (k as f64) / (len as f64);
                    tw.push(angle.cos());
                    tw.push(angle.sin());
                }
                self.queue_ref()
                    .write_buffer(twiddle.as_ref(), 0, cast_slice(&tw));
            }
            NumericPrecision::F32 => {
                let mut tw = Vec::with_capacity(twiddle_scalar_len);
                for k in 0..count {
                    let angle = -tau * (k as f64) / (len as f64);
                    tw.push(angle.cos() as f32);
                    tw.push(angle.sin() as f32);
                }
                self.queue_ref()
                    .write_buffer(twiddle.as_ref(), 0, cast_slice(&tw));
            }
        }

        if let Ok(mut cache) = self.fft_twiddle_cache.lock() {
            cache.insert((len, mode), twiddle.clone());
        }
        Ok(twiddle)
    }

    pub(crate) async fn fft_dim_exec(
        &self,
        handle: &GpuTensorHandle,
        len: Option<usize>,
        dim: usize,
    ) -> Result<GpuTensorHandle> {
        if let Some(native) = self.try_fft_dim_exec_native(handle, len, dim, false)? {
            return Ok(native);
        }
        self.fft_dim_exec_host_fallback(handle, len, dim, false)
            .await
    }

    pub(crate) async fn ifft_dim_exec(
        &self,
        handle: &GpuTensorHandle,
        len: Option<usize>,
        dim: usize,
    ) -> Result<GpuTensorHandle> {
        if let Some(native) = self.try_fft_dim_exec_native(handle, len, dim, true)? {
            return Ok(native);
        }
        self.fft_dim_exec_host_fallback(handle, len, dim, true)
            .await
    }

    pub(crate) fn fft_extract_real_exec(
        &self,
        handle: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(handle)?;
        let mut out_shape = handle.shape.clone();
        if out_shape.last() != Some(&2) {
            return Ok(handle.clone());
        }
        out_shape.pop();
        out_shape = normalize_scalar_shape(&out_shape);

        let complex_len = entry.len / 2;
        ensure!(
            complex_len <= u32::MAX as usize,
            "fft_extract_real: tensor too large for GPU kernel"
        );

        let out_buffer = self.create_storage_buffer(complex_len, "runmat-fft-extract-real-out");
        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;

        let mut offset = 0usize;
        while offset < complex_len {
            let chunk_len = (complex_len - offset).min(chunk_capacity);
            let params = crate::backend::wgpu::params::FftExtractRealParams {
                len: chunk_len as u32,
                offset: offset as u32,
                total: complex_len as u32,
                _pad0: 0,
            };
            let params_buffer = self.fft_uniform_buffer(&params, "runmat-fft-extract-real-params");
            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-fft-extract-real-bind"),
                    layout: &self.pipelines.fft_extract_real.layout,
                    entries: &[
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
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });
            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_len as u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::fft::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.fft_extract_real.pipeline,
                &bind_group,
                workgroups,
                "runmat-fft-extract-real-encoder",
            );
            offset += chunk_len;
        }

        Ok(self.register_existing_buffer(out_buffer, out_shape, complex_len))
    }

    fn try_fft_dim_exec_native(
        &self,
        handle: &GpuTensorHandle,
        len: Option<usize>,
        dim: usize,
        inverse: bool,
    ) -> Result<Option<GpuTensorHandle>> {
        if Self::fft_debug_enabled() {
            eprintln!(
                "[fft-debug] native enter len={len:?} dim={dim} inverse={inverse} shape={:?}",
                handle.shape
            );
        }
        let entry = self.get_entry(handle)?;
        let mut shape = handle.shape.clone();
        let complex_axis = shape.last() == Some(&2);
        if complex_axis {
            shape.pop();
        }
        if shape.is_empty() {
            let scalar_len = if complex_axis {
                entry.len / 2
            } else {
                entry.len
            };
            shape = vec![scalar_len];
        }

        let origin_rank = shape.len();
        while shape.len() <= dim {
            shape.push(1);
        }

        let current_len = shape.get(dim).copied().unwrap_or(0);
        let target_len = len.unwrap_or(current_len);
        let inner_stride = shape[..dim]
            .iter()
            .copied()
            .fold(1usize, |acc, v| acc.saturating_mul(v));
        let outer_stride = shape[dim + 1..]
            .iter()
            .copied()
            .fold(1usize, |acc, v| acc.saturating_mul(v));
        let num_slices = inner_stride.saturating_mul(outer_stride);

        let mut out_shape = shape.clone();
        if dim < out_shape.len() {
            out_shape[dim] = target_len;
        }
        if target_len == 0 || num_slices == 0 {
            fft_trim_trailing_ones(&mut out_shape, origin_rank.max(dim + 1));
            let mut packed_shape = out_shape;
            packed_shape.push(2);
            let buffer = self.create_storage_buffer(0, "runmat-fft-empty-native");
            return Ok(Some(self.register_existing_buffer(buffer, packed_shape, 0)));
        }

        let total_out = target_len.saturating_mul(num_slices);
        let copy_len = current_len.min(target_len);

        ensure!(
            target_len <= u32::MAX as usize
                && inner_stride <= u32::MAX as usize
                && current_len <= u32::MAX as usize
                && copy_len <= u32::MAX as usize
                && total_out <= u32::MAX as usize,
            "fft_dim: dimensions exceed native GPU FFT kernel limits"
        );

        let out_scalar_len = total_out
            .checked_mul(2)
            .ok_or_else(|| anyhow!("fft_dim: output buffer length overflow"))?;

        if !fft_is_power_of_two(target_len) {
            if let Some(factors) = fft_factor_smooth_235(target_len) {
                let has_3 = factors.contains(&3);
                let has_5 = factors.contains(&5);
                if has_3 && !has_5 {
                    if let Some(digits) = fft_log3_pow3(target_len) {
                        return self.try_fft_dim_exec_native_radix3(
                            entry.buffer,
                            shape,
                            dim,
                            origin_rank,
                            current_len,
                            copy_len,
                            target_len,
                            inner_stride,
                            total_out,
                            out_scalar_len,
                            complex_axis,
                            inverse,
                            digits,
                        );
                    }
                } else if has_5 && !has_3 {
                    if let Some(digits) = fft_log5_pow5(target_len) {
                        return self.try_fft_dim_exec_native_radix5(
                            entry.buffer,
                            shape,
                            dim,
                            origin_rank,
                            current_len,
                            copy_len,
                            target_len,
                            inner_stride,
                            total_out,
                            out_scalar_len,
                            complex_axis,
                            inverse,
                            digits,
                        );
                    }
                } else {
                    return self.try_fft_dim_exec_native_mixed(
                        entry.buffer,
                        shape,
                        dim,
                        origin_rank,
                        current_len,
                        copy_len,
                        target_len,
                        inner_stride,
                        total_out,
                        out_scalar_len,
                        complex_axis,
                        inverse,
                        &factors,
                    );
                }
            }
            if let Some(bluestein) = self.try_fft_dim_exec_native_bluestein(
                entry.buffer.clone(),
                shape.clone(),
                dim,
                origin_rank,
                current_len,
                copy_len,
                target_len,
                inner_stride,
                num_slices,
                total_out,
                out_scalar_len,
                complex_axis,
                inverse,
            )? {
                return Ok(Some(bluestein));
            }
            if let Some(digits) = fft_log3_pow3(target_len) {
                return self.try_fft_dim_exec_native_radix3(
                    entry.buffer,
                    shape,
                    dim,
                    origin_rank,
                    current_len,
                    copy_len,
                    target_len,
                    inner_stride,
                    total_out,
                    out_scalar_len,
                    complex_axis,
                    inverse,
                    digits,
                );
            }
            if let Some(digits) = fft_log5_pow5(target_len) {
                return self.try_fft_dim_exec_native_radix5(
                    entry.buffer,
                    shape,
                    dim,
                    origin_rank,
                    current_len,
                    copy_len,
                    target_len,
                    inner_stride,
                    total_out,
                    out_scalar_len,
                    complex_axis,
                    inverse,
                    digits,
                );
            }
            return Ok(None);
        }
        let Some(log2_len) = fft_log2_pow2(target_len) else {
            return Ok(None);
        };
        let (mut stage_a, mut stage_b) =
            self.fft_stage_buffer_pair(out_scalar_len, "runmat-fft-stage-a", "runmat-fft-stage-b");

        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;

        let mut offset = 0usize;
        while offset < total_out {
            let chunk_len = (total_out - offset).min(chunk_capacity);
            let params = crate::backend::wgpu::params::FftInitParams {
                len: chunk_len as u32,
                offset: offset as u32,
                total: total_out as u32,
                target_len: target_len as u32,
                inner_stride: inner_stride as u32,
                current_len: current_len as u32,
                copy_len: copy_len as u32,
                input_complex: if complex_axis { 1 } else { 0 },
            };
            let params_buffer = self.fft_uniform_buffer(&params, "runmat-fft-init-params");
            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-fft-init-bind"),
                    layout: &self.pipelines.fft_init.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: stage_a.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });
            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_len as u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::fft::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.fft_init.pipeline,
                &bind_group,
                workgroups,
                "runmat-fft-init-encoder",
            );
            offset += chunk_len;
        }

        self.fft_debug_dump_scalar_buffer("stage2:init", &stage_a, out_scalar_len);

        let mut stage_span = target_len;
        while stage_span >= 2 {
            let stage_half = stage_span / 2;
            let twiddle_step = target_len / stage_span;
            let mut stage_offset = 0usize;
            while stage_offset < total_out {
                let chunk_len = (total_out - stage_offset).min(chunk_capacity);
                let params = crate::backend::wgpu::params::FftStageParams {
                    len: chunk_len as u32,
                    offset: stage_offset as u32,
                    total: total_out as u32,
                    target_len: target_len as u32,
                    inner_stride: inner_stride as u32,
                    stage_span: stage_span as u32,
                    stage_half: stage_half as u32,
                    twiddle_step: twiddle_step as u32,
                    inverse: if inverse { 1 } else { 0 },
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                };
                if Self::fft_debug_enabled() {
                    eprintln!(
                        "[fft-debug] stage2 params span={} half={} step={} chunk={} off={}",
                        params.stage_span,
                        params.stage_half,
                        params.twiddle_step,
                        params.len,
                        params.offset
                    );
                }
                let params_buffer = self.fft_uniform_buffer(&params, "runmat-fft-stage-params");
                let bind_group = self
                    .device_ref()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-fft-stage-bind"),
                        layout: &self.pipelines.fft_stage.layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: stage_a.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: stage_b.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: params_buffer.as_entire_binding(),
                            },
                        ],
                    });
                let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                    chunk_len as u32,
                    crate::backend::wgpu::config::WORKGROUP_SIZE,
                );
                if Self::fft_debug_enabled() {
                    eprintln!(
                        "[fft-debug] stage2 dispatch workgroups={workgroups} chunk={chunk_len}"
                    );
                }
                crate::backend::wgpu::dispatch::fft::run(
                    self.device_ref(),
                    self.queue_ref(),
                    &self.pipelines.fft_stage.pipeline,
                    &bind_group,
                    workgroups,
                    "runmat-fft-stage-encoder",
                );
                stage_offset += chunk_len;
            }
            std::mem::swap(&mut stage_a, &mut stage_b);
            self.fft_debug_dump_scalar_buffer("stage2:after_stage", &stage_a, out_scalar_len);
            stage_span /= 2;
        }

        let mut reorder_offset = 0usize;
        while reorder_offset < total_out {
            let chunk_len = (total_out - reorder_offset).min(chunk_capacity);
            let params = crate::backend::wgpu::params::FftReorderParams {
                len: chunk_len as u32,
                offset: reorder_offset as u32,
                total: total_out as u32,
                target_len: target_len as u32,
                inner_stride: inner_stride as u32,
                log2_len,
                inverse: if inverse { 1 } else { 0 },
                _pad0: 0,
            };
            let params_buffer = self.fft_uniform_buffer(&params, "runmat-fft-reorder-params");
            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-fft-reorder-bind"),
                    layout: &self.pipelines.fft_reorder.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: stage_a.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: stage_b.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });
            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_len as u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::fft::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.fft_reorder.pipeline,
                &bind_group,
                workgroups,
                "runmat-fft-reorder-encoder",
            );
            reorder_offset += chunk_len;
        }

        self.fft_debug_dump_scalar_buffer("stage2:after_reorder", &stage_b, out_scalar_len);

        fft_trim_trailing_ones(&mut out_shape, origin_rank.max(dim + 1));
        let mut packed_shape = out_shape;
        packed_shape.push(2);
        Ok(Some(self.register_existing_buffer(
            stage_b,
            packed_shape,
            out_scalar_len,
        )))
    }

    #[allow(clippy::too_many_arguments)]
    fn try_fft_dim_exec_native_bluestein(
        &self,
        in_buffer: Arc<wgpu::Buffer>,
        shape: Vec<usize>,
        dim: usize,
        origin_rank: usize,
        current_len: usize,
        copy_len: usize,
        target_len: usize,
        inner_stride: usize,
        num_slices: usize,
        total_out: usize,
        out_scalar_len: usize,
        complex_axis: bool,
        inverse: bool,
    ) -> Result<Option<GpuTensorHandle>> {
        let two_n_minus_1 = target_len
            .checked_mul(2)
            .and_then(|v| v.checked_sub(1))
            .ok_or_else(|| anyhow!("fft_dim: bluestein length overflow"))?;
        let m_len = two_n_minus_1.next_power_of_two();
        if m_len <= target_len {
            return Ok(None);
        }
        if m_len > u32::MAX as usize || target_len > u32::MAX as usize {
            return Ok(None);
        }

        let mut ext_shape = shape.clone();
        ext_shape[dim] = m_len;
        let mut ext_shape_packed = ext_shape.clone();
        ext_shape_packed.push(2);

        let m_total = m_len
            .checked_mul(num_slices)
            .ok_or_else(|| anyhow!("fft_dim: bluestein total overflow"))?;
        let m_out_scalar_len = m_total
            .checked_mul(2)
            .ok_or_else(|| anyhow!("fft_dim: bluestein buffer overflow"))?;

        let chirp_scalar_len = target_len
            .checked_mul(2)
            .ok_or_else(|| anyhow!("fft_dim: chirp buffer overflow"))?;
        let chirp_buffer =
            self.create_storage_buffer(chirp_scalar_len, "runmat-fft-bluestein-chirp");
        let pi = std::f64::consts::PI;
        let sign = if inverse { 1.0 } else { -1.0 };
        match self.precision {
            NumericPrecision::F64 => {
                let mut chirp = Vec::with_capacity(chirp_scalar_len);
                for n in 0..target_len {
                    let angle = sign * pi * (n as f64) * (n as f64) / (target_len as f64);
                    chirp.push(angle.cos());
                    chirp.push(angle.sin());
                }
                self.queue_ref()
                    .write_buffer(chirp_buffer.as_ref(), 0, cast_slice(&chirp));
            }
            NumericPrecision::F32 => {
                let mut chirp = Vec::with_capacity(chirp_scalar_len);
                for n in 0..target_len {
                    let angle = sign * pi * (n as f64) * (n as f64) / (target_len as f64);
                    chirp.push(angle.cos() as f32);
                    chirp.push(angle.sin() as f32);
                }
                self.queue_ref()
                    .write_buffer(chirp_buffer.as_ref(), 0, cast_slice(&chirp));
            }
        }

        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;

        let a_buffer = self.create_storage_buffer(m_out_scalar_len, "runmat-fft-bluestein-a");
        let mut offset = 0usize;
        while offset < m_total {
            let chunk_len = (m_total - offset).min(chunk_capacity);
            let params = crate::backend::wgpu::params::FftBluesteinPrepParams {
                len: chunk_len as u32,
                offset: offset as u32,
                total: m_total as u32,
                target_len: m_len as u32,
                inner_stride: inner_stride as u32,
                current_len: current_len as u32,
                copy_len: copy_len as u32,
                input_complex: if complex_axis { 1 } else { 0 },
            };
            let params_buffer = self.fft_uniform_buffer(&params, "runmat-fft-blue-prep-params");
            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-fft-blue-prep-bind"),
                    layout: &self.pipelines.fft_bluestein_prep.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: in_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: a_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: chirp_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });
            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_len as u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::fft::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.fft_bluestein_prep.pipeline,
                &bind_group,
                workgroups,
                "runmat-fft-blue-prep-encoder",
            );
            offset += chunk_len;
        }

        let b_scalar_len = m_len
            .checked_mul(2)
            .ok_or_else(|| anyhow!("fft_dim: bluestein kernel buffer overflow"))?;
        let b_buffer = self.create_storage_buffer(b_scalar_len, "runmat-fft-bluestein-b");
        let mut boff = 0usize;
        while boff < m_len {
            let chunk_len = (m_len - boff).min(chunk_capacity);
            let params = crate::backend::wgpu::params::FftBluesteinKernelParams {
                len: chunk_len as u32,
                offset: boff as u32,
                total: m_len as u32,
                target_len: target_len as u32,
                m_len: m_len as u32,
                inverse: if inverse { 1 } else { 0 },
                _pad0: 0,
                _pad1: 0,
            };
            let params_buffer = self.fft_uniform_buffer(&params, "runmat-fft-blue-kernel-params");
            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-fft-blue-kernel-bind"),
                    layout: &self.pipelines.fft_bluestein_kernel.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: b_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: chirp_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });
            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_len as u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::fft::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.fft_bluestein_kernel.pipeline,
                &bind_group,
                workgroups,
                "runmat-fft-blue-kernel-encoder",
            );
            boff += chunk_len;
        }

        let a_handle =
            self.register_existing_buffer(a_buffer, ext_shape_packed.clone(), m_out_scalar_len);
        let b_handle = self.register_existing_buffer(b_buffer, vec![m_len, 2], b_scalar_len);
        let Some(a_fft) = self.try_fft_dim_exec_native(&a_handle, Some(m_len), dim, false)? else {
            return Ok(None);
        };
        let Some(b_fft) = self.try_fft_dim_exec_native(&b_handle, Some(m_len), 0, false)? else {
            return Ok(None);
        };

        let c_fft_buf = self.create_storage_buffer(m_out_scalar_len, "runmat-fft-bluestein-cfft");
        let mut poff = 0usize;
        while poff < m_total {
            let chunk_len = (m_total - poff).min(chunk_capacity);
            let params = crate::backend::wgpu::params::FftPointwiseBroadcastParams {
                len: chunk_len as u32,
                offset: poff as u32,
                total: m_total as u32,
                m_len: m_len as u32,
                inner_stride: inner_stride as u32,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };
            let params_buffer =
                self.fft_uniform_buffer(&params, "runmat-fft-blue-pointwise-params");
            let a_entry = self.get_entry(&a_fft)?;
            let b_entry = self.get_entry(&b_fft)?;
            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-fft-blue-pointwise-bind"),
                    layout: &self.pipelines.fft_pointwise_broadcast.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: a_entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: b_entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: c_fft_buf.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });
            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_len as u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::fft::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.fft_pointwise_broadcast.pipeline,
                &bind_group,
                workgroups,
                "runmat-fft-blue-pointwise-encoder",
            );
            poff += chunk_len;
        }

        let c_fft_handle =
            self.register_existing_buffer(c_fft_buf, ext_shape_packed.clone(), m_out_scalar_len);
        let Some(c_time) = self.try_fft_dim_exec_native(&c_fft_handle, Some(m_len), dim, true)?
        else {
            return Ok(None);
        };

        let out_buffer = self.create_storage_buffer(out_scalar_len, "runmat-fft-bluestein-out");
        let mut foff = 0usize;
        while foff < total_out {
            let chunk_len = (total_out - foff).min(chunk_capacity);
            let params = crate::backend::wgpu::params::FftBluesteinFinalizeParams {
                len: chunk_len as u32,
                offset: foff as u32,
                total: total_out as u32,
                target_len: target_len as u32,
                m_len: m_len as u32,
                inner_stride: inner_stride as u32,
                inverse: if inverse { 1 } else { 0 },
                _pad0: 0,
            };
            let params_buffer = self.fft_uniform_buffer(&params, "runmat-fft-blue-finalize-params");
            let c_entry = self.get_entry(&c_time)?;
            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-fft-blue-finalize-bind"),
                    layout: &self.pipelines.fft_bluestein_finalize.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: c_entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: out_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: chirp_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });
            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_len as u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::fft::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.fft_bluestein_finalize.pipeline,
                &bind_group,
                workgroups,
                "runmat-fft-blue-finalize-encoder",
            );
            foff += chunk_len;
        }

        self.free(&a_handle).ok();
        self.free(&b_handle).ok();
        self.free(&a_fft).ok();
        self.free(&b_fft).ok();
        self.free(&c_fft_handle).ok();
        self.free(&c_time).ok();

        let mut out_shape = shape;
        out_shape[dim] = target_len;
        fft_trim_trailing_ones(&mut out_shape, origin_rank.max(dim + 1));
        out_shape.push(2);
        Ok(Some(self.register_existing_buffer(
            out_buffer,
            out_shape,
            out_scalar_len,
        )))
    }

    #[allow(clippy::too_many_arguments)]
    fn try_fft_dim_exec_native_mixed(
        &self,
        in_buffer: Arc<wgpu::Buffer>,
        mut shape: Vec<usize>,
        dim: usize,
        origin_rank: usize,
        current_len: usize,
        copy_len: usize,
        target_len: usize,
        inner_stride: usize,
        total_out: usize,
        out_scalar_len: usize,
        complex_axis: bool,
        inverse: bool,
        factors: &[u32],
    ) -> Result<Option<GpuTensorHandle>> {
        if factors.len() > 16 {
            return Ok(None);
        }

        let (mut stage_a, mut stage_b) = self.fft_stage_buffer_pair(
            out_scalar_len,
            "runmat-fftm-stage-a",
            "runmat-fftm-stage-b",
        );
        let twiddle = self.fft_twiddle_buffer(target_len, false, "runmat-fftm-twiddles")?;

        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;

        let mut offset = 0usize;
        while offset < total_out {
            let chunk_len = (total_out - offset).min(chunk_capacity);
            let params = crate::backend::wgpu::params::FftInitParams {
                len: chunk_len as u32,
                offset: offset as u32,
                total: total_out as u32,
                target_len: target_len as u32,
                inner_stride: inner_stride as u32,
                current_len: current_len as u32,
                copy_len: copy_len as u32,
                input_complex: if complex_axis { 1 } else { 0 },
            };
            let params_buffer = self.fft_uniform_buffer(&params, "runmat-fftm-init-params");
            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-fftm-init-bind"),
                    layout: &self.pipelines.fft_init.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: in_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: stage_a.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });
            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_len as u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::fft::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.fft_init.pipeline,
                &bind_group,
                workgroups,
                "runmat-fftm-init-encoder",
            );
            offset += chunk_len;
        }

        self.fft_debug_dump_scalar_buffer("mixed:init", &stage_a, out_scalar_len);

        let mut stage_span = target_len;
        for &radix in factors {
            if radix == 2 {
                let stage_half = stage_span / 2;
                let twiddle_step = target_len / stage_span;
                let mut stage_offset = 0usize;
                while stage_offset < total_out {
                    let chunk_len = (total_out - stage_offset).min(chunk_capacity);
                    let params = crate::backend::wgpu::params::FftStageParams {
                        len: chunk_len as u32,
                        offset: stage_offset as u32,
                        total: total_out as u32,
                        target_len: target_len as u32,
                        inner_stride: inner_stride as u32,
                        stage_span: stage_span as u32,
                        stage_half: stage_half as u32,
                        twiddle_step: twiddle_step as u32,
                        inverse: if inverse { 1 } else { 0 },
                        _pad0: 0,
                        _pad1: 0,
                        _pad2: 0,
                    };
                    let params_buffer =
                        self.fft_uniform_buffer(&params, "runmat-fftm-stage2-params");
                    let bind_group =
                        self.device_ref()
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("runmat-fftm-stage2-bind"),
                                layout: &self.pipelines.fft_stage.layout,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: stage_a.as_ref().as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: stage_b.as_ref().as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 2,
                                        resource: params_buffer.as_entire_binding(),
                                    },
                                ],
                            });
                    let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                        chunk_len as u32,
                        crate::backend::wgpu::config::WORKGROUP_SIZE,
                    );
                    crate::backend::wgpu::dispatch::fft::run(
                        self.device_ref(),
                        self.queue_ref(),
                        &self.pipelines.fft_stage.pipeline,
                        &bind_group,
                        workgroups,
                        "runmat-fftm-stage2-encoder",
                    );
                    stage_offset += chunk_len;
                }
                self.fft_debug_dump_scalar_buffer("mixed:after_stage2", &stage_b, out_scalar_len);
            } else if radix == 3 {
                let stage_third = stage_span / 3;
                let twiddle_step = target_len / stage_span;
                let mut stage_offset = 0usize;
                while stage_offset < total_out {
                    let chunk_len = (total_out - stage_offset).min(chunk_capacity);
                    let params = crate::backend::wgpu::params::FftStage3Params {
                        len: chunk_len as u32,
                        offset: stage_offset as u32,
                        total: total_out as u32,
                        target_len: target_len as u32,
                        inner_stride: inner_stride as u32,
                        stage_span: stage_span as u32,
                        stage_third: stage_third as u32,
                        twiddle_step: twiddle_step as u32,
                        inverse: if inverse { 1 } else { 0 },
                        _pad0: 0,
                        _pad1: 0,
                        _pad2: 0,
                    };
                    let params_buffer =
                        self.fft_uniform_buffer(&params, "runmat-fftm-stage3-params");
                    let bind_group =
                        self.device_ref()
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("runmat-fftm-stage3-bind"),
                                layout: &self.pipelines.fft_stage3.layout,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: stage_a.as_ref().as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: stage_b.as_ref().as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 2,
                                        resource: twiddle.as_ref().as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 3,
                                        resource: params_buffer.as_entire_binding(),
                                    },
                                ],
                            });
                    let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                        chunk_len as u32,
                        crate::backend::wgpu::config::WORKGROUP_SIZE,
                    );
                    crate::backend::wgpu::dispatch::fft::run(
                        self.device_ref(),
                        self.queue_ref(),
                        &self.pipelines.fft_stage3.pipeline,
                        &bind_group,
                        workgroups,
                        "runmat-fftm-stage3-encoder",
                    );
                    stage_offset += chunk_len;
                }
                self.fft_debug_dump_scalar_buffer("mixed:after_stage3", &stage_b, out_scalar_len);
            } else if radix == 5 {
                let stage_fifth = stage_span / 5;
                let twiddle_step = target_len / stage_span;
                let mut stage_offset = 0usize;
                while stage_offset < total_out {
                    let chunk_len = (total_out - stage_offset).min(chunk_capacity);
                    let params = crate::backend::wgpu::params::FftStage5Params {
                        len: chunk_len as u32,
                        offset: stage_offset as u32,
                        total: total_out as u32,
                        target_len: target_len as u32,
                        inner_stride: inner_stride as u32,
                        stage_span: stage_span as u32,
                        stage_fifth: stage_fifth as u32,
                        twiddle_step: twiddle_step as u32,
                        inverse: if inverse { 1 } else { 0 },
                        _pad0: 0,
                        _pad1: 0,
                        _pad2: 0,
                    };
                    let params_buffer =
                        self.fft_uniform_buffer(&params, "runmat-fftm-stage5-params");
                    let bind_group =
                        self.device_ref()
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("runmat-fftm-stage5-bind"),
                                layout: &self.pipelines.fft_stage5.layout,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: stage_a.as_ref().as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: stage_b.as_ref().as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 2,
                                        resource: twiddle.as_ref().as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 3,
                                        resource: params_buffer.as_entire_binding(),
                                    },
                                ],
                            });
                    let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                        chunk_len as u32,
                        crate::backend::wgpu::config::WORKGROUP_SIZE,
                    );
                    crate::backend::wgpu::dispatch::fft::run(
                        self.device_ref(),
                        self.queue_ref(),
                        &self.pipelines.fft_stage5.pipeline,
                        &bind_group,
                        workgroups,
                        "runmat-fftm-stage5-encoder",
                    );
                    stage_offset += chunk_len;
                }
                self.fft_debug_dump_scalar_buffer("mixed:after_stage5", &stage_b, out_scalar_len);
            } else {
                return Ok(None);
            }

            std::mem::swap(&mut stage_a, &mut stage_b);
            stage_span /= radix as usize;
        }

        let mut radices = [0u32; 16];
        for (i, &r) in factors.iter().enumerate() {
            radices[i] = r;
        }

        let mut reorder_offset = 0usize;
        while reorder_offset < total_out {
            let chunk_len = (total_out - reorder_offset).min(chunk_capacity);
            let params = crate::backend::wgpu::params::FftReorderMixedParams {
                len: chunk_len as u32,
                offset: reorder_offset as u32,
                total: total_out as u32,
                target_len: target_len as u32,
                inner_stride: inner_stride as u32,
                stage_count: factors.len() as u32,
                inverse: if inverse { 1 } else { 0 },
                _pad0: 0,
                radices,
            };
            let params_buffer =
                self.fft_storage_param_buffer(&params, "runmat-fftm-reorder-params");
            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-fftm-reorder-bind"),
                    layout: &self.pipelines.fft_reorder_mixed.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: stage_a.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: stage_b.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });
            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_len as u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::fft::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.fft_reorder_mixed.pipeline,
                &bind_group,
                workgroups,
                "runmat-fftm-reorder-encoder",
            );
            reorder_offset += chunk_len;
        }

        self.fft_debug_dump_scalar_buffer("mixed:after_reorder", &stage_b, out_scalar_len);

        shape[dim] = target_len;
        fft_trim_trailing_ones(&mut shape, origin_rank.max(dim + 1));
        shape.push(2);
        Ok(Some(self.register_existing_buffer(
            stage_b,
            shape,
            out_scalar_len,
        )))
    }

    #[allow(clippy::too_many_arguments)]
    fn try_fft_dim_exec_native_radix3(
        &self,
        in_buffer: Arc<wgpu::Buffer>,
        mut shape: Vec<usize>,
        dim: usize,
        origin_rank: usize,
        current_len: usize,
        copy_len: usize,
        target_len: usize,
        inner_stride: usize,
        total_out: usize,
        out_scalar_len: usize,
        complex_axis: bool,
        inverse: bool,
        digits: u32,
    ) -> Result<Option<GpuTensorHandle>> {
        let (mut stage_a, mut stage_b) = self.fft_stage_buffer_pair(
            out_scalar_len,
            "runmat-fft3-stage-a",
            "runmat-fft3-stage-b",
        );
        let twiddle = self.fft_twiddle_buffer(target_len, false, "runmat-fft3-twiddles")?;

        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;

        let mut offset = 0usize;
        while offset < total_out {
            let chunk_len = (total_out - offset).min(chunk_capacity);
            let params = crate::backend::wgpu::params::FftInitParams {
                len: chunk_len as u32,
                offset: offset as u32,
                total: total_out as u32,
                target_len: target_len as u32,
                inner_stride: inner_stride as u32,
                current_len: current_len as u32,
                copy_len: copy_len as u32,
                input_complex: if complex_axis { 1 } else { 0 },
            };
            let params_buffer = self.fft_uniform_buffer(&params, "runmat-fft3-init-params");
            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-fft3-init-bind"),
                    layout: &self.pipelines.fft_init.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: in_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: stage_a.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });
            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_len as u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::fft::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.fft_init.pipeline,
                &bind_group,
                workgroups,
                "runmat-fft3-init-encoder",
            );
            offset += chunk_len;
        }

        self.fft_debug_dump_scalar_buffer("radix3:init", &stage_a, out_scalar_len);

        let mut stage_span = target_len;
        while stage_span >= 3 {
            let stage_third = stage_span / 3;
            let twiddle_step = target_len / stage_span;
            let mut stage_offset = 0usize;
            while stage_offset < total_out {
                let chunk_len = (total_out - stage_offset).min(chunk_capacity);
                let params = crate::backend::wgpu::params::FftStage3Params {
                    len: chunk_len as u32,
                    offset: stage_offset as u32,
                    total: total_out as u32,
                    target_len: target_len as u32,
                    inner_stride: inner_stride as u32,
                    stage_span: stage_span as u32,
                    stage_third: stage_third as u32,
                    twiddle_step: twiddle_step as u32,
                    inverse: if inverse { 1 } else { 0 },
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                };
                if Self::fft_debug_enabled() {
                    eprintln!(
                        "[fft-debug] stage3 params span={} third={} step={} chunk={} off={}",
                        params.stage_span,
                        params.stage_third,
                        params.twiddle_step,
                        params.len,
                        params.offset
                    );
                }
                let params_buffer = self.fft_uniform_buffer(&params, "runmat-fft3-stage-params");
                let bind_group = self
                    .device_ref()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-fft3-stage-bind"),
                        layout: &self.pipelines.fft_stage3.layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: stage_a.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: stage_b.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: twiddle.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: params_buffer.as_entire_binding(),
                            },
                        ],
                    });
                let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                    chunk_len as u32,
                    crate::backend::wgpu::config::WORKGROUP_SIZE,
                );
                if Self::fft_debug_enabled() {
                    eprintln!(
                        "[fft-debug] stage3 dispatch workgroups={workgroups} chunk={chunk_len}"
                    );
                }
                crate::backend::wgpu::dispatch::fft::run(
                    self.device_ref(),
                    self.queue_ref(),
                    &self.pipelines.fft_stage3.pipeline,
                    &bind_group,
                    workgroups,
                    "runmat-fft3-stage-encoder",
                );
                stage_offset += chunk_len;
            }
            std::mem::swap(&mut stage_a, &mut stage_b);
            self.fft_debug_dump_scalar_buffer("stage3:after_stage", &stage_a, out_scalar_len);
            stage_span /= 3;
        }

        let mut reorder_offset = 0usize;
        while reorder_offset < total_out {
            let chunk_len = (total_out - reorder_offset).min(chunk_capacity);
            let params = crate::backend::wgpu::params::FftReorder3Params {
                len: chunk_len as u32,
                offset: reorder_offset as u32,
                total: total_out as u32,
                target_len: target_len as u32,
                inner_stride: inner_stride as u32,
                digits,
                inverse: if inverse { 1 } else { 0 },
                _pad0: 0,
            };
            let params_buffer = self.fft_uniform_buffer(&params, "runmat-fft3-reorder-params");
            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-fft3-reorder-bind"),
                    layout: &self.pipelines.fft_reorder3.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: stage_a.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: stage_b.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });
            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_len as u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::fft::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.fft_reorder3.pipeline,
                &bind_group,
                workgroups,
                "runmat-fft3-reorder-encoder",
            );
            reorder_offset += chunk_len;
        }

        self.fft_debug_dump_scalar_buffer("stage3:after_reorder", &stage_b, out_scalar_len);

        shape[dim] = target_len;
        fft_trim_trailing_ones(&mut shape, origin_rank.max(dim + 1));
        shape.push(2);
        Ok(Some(self.register_existing_buffer(
            stage_b,
            shape,
            out_scalar_len,
        )))
    }

    #[allow(clippy::too_many_arguments)]
    fn try_fft_dim_exec_native_radix5(
        &self,
        in_buffer: Arc<wgpu::Buffer>,
        mut shape: Vec<usize>,
        dim: usize,
        origin_rank: usize,
        current_len: usize,
        copy_len: usize,
        target_len: usize,
        inner_stride: usize,
        total_out: usize,
        out_scalar_len: usize,
        complex_axis: bool,
        inverse: bool,
        digits: u32,
    ) -> Result<Option<GpuTensorHandle>> {
        let (mut stage_a, mut stage_b) = self.fft_stage_buffer_pair(
            out_scalar_len,
            "runmat-fft5-stage-a",
            "runmat-fft5-stage-b",
        );
        let twiddle = self.fft_twiddle_buffer(target_len, false, "runmat-fft5-twiddles")?;

        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;

        let mut offset = 0usize;
        while offset < total_out {
            let chunk_len = (total_out - offset).min(chunk_capacity);
            let params = crate::backend::wgpu::params::FftInitParams {
                len: chunk_len as u32,
                offset: offset as u32,
                total: total_out as u32,
                target_len: target_len as u32,
                inner_stride: inner_stride as u32,
                current_len: current_len as u32,
                copy_len: copy_len as u32,
                input_complex: if complex_axis { 1 } else { 0 },
            };
            let params_buffer = self.fft_uniform_buffer(&params, "runmat-fft5-init-params");
            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-fft5-init-bind"),
                    layout: &self.pipelines.fft_init.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: in_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: stage_a.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });
            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_len as u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::fft::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.fft_init.pipeline,
                &bind_group,
                workgroups,
                "runmat-fft5-init-encoder",
            );
            offset += chunk_len;
        }

        let mut stage_span = target_len;
        while stage_span >= 5 {
            let stage_fifth = stage_span / 5;
            let twiddle_step = target_len / stage_span;
            let mut stage_offset = 0usize;
            while stage_offset < total_out {
                let chunk_len = (total_out - stage_offset).min(chunk_capacity);
                let params = crate::backend::wgpu::params::FftStage5Params {
                    len: chunk_len as u32,
                    offset: stage_offset as u32,
                    total: total_out as u32,
                    target_len: target_len as u32,
                    inner_stride: inner_stride as u32,
                    stage_span: stage_span as u32,
                    stage_fifth: stage_fifth as u32,
                    twiddle_step: twiddle_step as u32,
                    inverse: if inverse { 1 } else { 0 },
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                };
                let params_buffer = self.fft_uniform_buffer(&params, "runmat-fft5-stage-params");
                let bind_group = self
                    .device_ref()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-fft5-stage-bind"),
                        layout: &self.pipelines.fft_stage5.layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: stage_a.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: stage_b.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: twiddle.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: params_buffer.as_entire_binding(),
                            },
                        ],
                    });
                let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                    chunk_len as u32,
                    crate::backend::wgpu::config::WORKGROUP_SIZE,
                );
                crate::backend::wgpu::dispatch::fft::run(
                    self.device_ref(),
                    self.queue_ref(),
                    &self.pipelines.fft_stage5.pipeline,
                    &bind_group,
                    workgroups,
                    "runmat-fft5-stage-encoder",
                );
                stage_offset += chunk_len;
            }
            std::mem::swap(&mut stage_a, &mut stage_b);
            stage_span /= 5;
        }

        let mut reorder_offset = 0usize;
        while reorder_offset < total_out {
            let chunk_len = (total_out - reorder_offset).min(chunk_capacity);
            let params = crate::backend::wgpu::params::FftReorder5Params {
                len: chunk_len as u32,
                offset: reorder_offset as u32,
                total: total_out as u32,
                target_len: target_len as u32,
                inner_stride: inner_stride as u32,
                digits,
                inverse: if inverse { 1 } else { 0 },
                _pad0: 0,
            };
            let params_buffer = self.fft_uniform_buffer(&params, "runmat-fft5-reorder-params");
            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-fft5-reorder-bind"),
                    layout: &self.pipelines.fft_reorder5.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: stage_a.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: stage_b.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });
            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_len as u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::fft::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.fft_reorder5.pipeline,
                &bind_group,
                workgroups,
                "runmat-fft5-reorder-encoder",
            );
            reorder_offset += chunk_len;
        }

        shape[dim] = target_len;
        fft_trim_trailing_ones(&mut shape, origin_rank.max(dim + 1));
        shape.push(2);
        Ok(Some(self.register_existing_buffer(
            stage_b,
            shape,
            out_scalar_len,
        )))
    }

    #[allow(clippy::too_many_arguments)]
    async fn fft_dim_exec_host_fallback(
        &self,
        handle: &GpuTensorHandle,
        len: Option<usize>,
        dim: usize,
        inverse: bool,
    ) -> Result<GpuTensorHandle> {
        let HostTensorOwned { data, mut shape } =
            <Self as AccelProvider>::download(self, handle).await?;
        let mut complex_axis = false;
        if shape.last() == Some(&2) {
            complex_axis = true;
            shape.pop();
        }
        if shape.is_empty() {
            if complex_axis {
                let inferred = data.len() / 2;
                shape = vec![inferred];
            } else if data.is_empty() {
                shape = vec![0];
            } else {
                shape = vec![data.len()];
            }
        }
        let origin_rank = shape.len();
        while shape.len() <= dim {
            shape.push(1);
        }
        let current_len = shape.get(dim).copied().unwrap_or(0);
        let target_len = len.unwrap_or(current_len);

        let inner_stride = shape[..dim]
            .iter()
            .copied()
            .fold(1usize, |acc, v| acc.saturating_mul(v));
        let outer_stride = shape[dim + 1..]
            .iter()
            .copied()
            .fold(1usize, |acc, v| acc.saturating_mul(v));
        let num_slices = inner_stride.saturating_mul(outer_stride);

        let copy_len = current_len.min(target_len);

        let mut out_shape = shape.clone();
        if dim < out_shape.len() {
            out_shape[dim] = target_len;
        }

        if target_len == 0 || num_slices == 0 {
            fft_trim_trailing_ones(&mut out_shape, origin_rank.max(dim + 1));
            let mut packed_shape = out_shape.clone();
            packed_shape.push(2);
            let buffer = self.create_storage_buffer(0, "runmat-fft-empty");
            return Ok(self.register_existing_buffer(buffer, packed_shape, 0));
        }

        let total_elems = shape
            .iter()
            .copied()
            .fold(1usize, |acc, v| acc.saturating_mul(v));
        let mut input = Vec::with_capacity(total_elems);
        if complex_axis {
            for idx in 0..total_elems {
                let base = idx * 2;
                let re = data.get(base).copied().unwrap_or(0.0);
                let im = data.get(base + 1).copied().unwrap_or(0.0);
                input.push(Complex::new(re, im));
            }
        } else {
            for idx in 0..total_elems {
                let re = data.get(idx).copied().unwrap_or(0.0);
                input.push(Complex::new(re, 0.0));
            }
        }

        let mut planner = FftPlanner::<f64>::new();
        let fft_plan = if target_len > 1 {
            Some(if inverse {
                planner.plan_fft_inverse(target_len)
            } else {
                planner.plan_fft_forward(target_len)
            })
        } else {
            None
        };

        let mut buffer_line = vec![Complex::new(0.0, 0.0); target_len];
        let mut output = vec![Complex::new(0.0, 0.0); target_len.saturating_mul(num_slices)];

        for outer in 0..outer_stride {
            let base_in = outer.saturating_mul(current_len.saturating_mul(inner_stride));
            let base_out = outer.saturating_mul(target_len.saturating_mul(inner_stride));
            for inner in 0..inner_stride {
                buffer_line.fill(Complex::new(0.0, 0.0));
                for (k, slot) in buffer_line.iter_mut().enumerate().take(copy_len) {
                    let src_idx = base_in + inner + k * inner_stride;
                    if src_idx < input.len() {
                        *slot = input[src_idx];
                    }
                }
                if let Some(plan) = &fft_plan {
                    plan.process(&mut buffer_line);
                }
                for (k, value) in buffer_line.iter().enumerate().take(target_len) {
                    let dst_idx = base_out + inner + k * inner_stride;
                    if dst_idx < output.len() {
                        output[dst_idx] = if inverse {
                            *value * (1.0 / (target_len as f64))
                        } else {
                            *value
                        };
                    }
                }
            }
        }

        fft_trim_trailing_ones(&mut out_shape, origin_rank.max(dim + 1));
        let mut packed_shape = out_shape.clone();
        packed_shape.push(2);

        let mut packed = Vec::with_capacity(output.len() * 2);
        for complex in output {
            packed.push(complex.re);
            packed.push(complex.im);
        }

        let view = HostTensorView {
            data: &packed,
            shape: &packed_shape,
        };
        let result = self.upload(&view)?;
        Ok(result)
    }
}
