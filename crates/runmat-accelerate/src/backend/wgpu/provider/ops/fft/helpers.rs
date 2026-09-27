use super::*;

impl WgpuProvider {
    pub(super) fn fft_uniform_buffer<T: Pod>(
        &self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        data: &T,
        label: &'static str,
    ) -> Result<Arc<wgpu::Buffer>> {
        let size = std::mem::size_of::<T>() as u64;
        let buffer = self.kernel_resources.uniform_buffer(
            emission,
            UniformBufferKey::LenOpParams,
            size,
            label,
        )?;
        emission
            .queue()
            .write_buffer(buffer.as_ref(), 0, bytes_of(data));
        Ok(buffer)
    }

    pub(super) fn fft_storage_param_buffer<T: Pod>(
        &self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        data: &T,
        label: &str,
    ) -> Result<Arc<wgpu::Buffer>> {
        let size = std::mem::size_of::<T>() as u64;
        let buffer = Arc::new(emission.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(1),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        emission
            .queue()
            .write_buffer(buffer.as_ref(), 0, bytes_of(data));
        Ok(buffer)
    }

    pub(super) fn fft_debug_enabled() -> bool {
        std::env::var_os("RUNMAT_FFT_DEBUG").is_some()
    }

    pub(super) fn fft_debug_dump_scalar_buffer(
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
        let staging = self
            .with_emission(|emission, pending| -> Result<_> {
                let staging = emission.device().create_buffer(&wgpu::BufferDescriptor {
                    label: Some("runmat-fft-debug-staging"),
                    size: size_bytes.max(1),
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let mut encoder =
                    emission
                        .device()
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("runmat-fft-debug-copy"),
                        });
                encoder.copy_buffer_to_buffer(buffer.as_ref(), 0, &staging, 0, size_bytes);
                self.submit(emission, pending, encoder)?;
                Ok(staging)
            })
            .and_then(|result| result);
        let Ok(staging) = staging else {
            eprintln!("[fft-debug] {label}: staging emission failed");
            return;
        };
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

    pub(super) fn fft_stage_buffer_pair(
        &self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        len: usize,
        label_a: &str,
        label_b: &str,
    ) -> Result<(Arc<wgpu::Buffer>, Arc<wgpu::Buffer>)> {
        let a = self.create_storage_buffer(emission, len, label_a)?;
        let mut b = self.create_storage_buffer(emission, len, label_b)?;
        if Arc::ptr_eq(&a, &b) {
            let size_bytes = (len.max(1) as u64).saturating_mul(self.element_size as u64);
            b = Arc::new(emission.device().create_buffer(&wgpu::BufferDescriptor {
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
        Ok((a, b))
    }

    pub(super) fn fft_twiddle_buffer(
        &self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
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

        crate::backend::wgpu::cache::scope::allow_creation(emission.mode())?;
        let count = if half_only { len / 2 } else { len };
        let twiddle_scalar_len = count
            .checked_mul(2)
            .ok_or_else(|| anyhow!("fft_dim: twiddle buffer length overflow"))?;
        let size_bytes =
            (twiddle_scalar_len.max(1) as u64).saturating_mul(self.element_size as u64);
        let twiddle = Arc::new(emission.device().create_buffer(&wgpu::BufferDescriptor {
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
                emission
                    .queue()
                    .write_buffer(twiddle.as_ref(), 0, cast_slice(&tw));
            }
            NumericPrecision::F32 => {
                let mut tw = Vec::with_capacity(twiddle_scalar_len);
                for k in 0..count {
                    let angle = -tau * (k as f64) / (len as f64);
                    tw.push(angle.cos() as f32);
                    tw.push(angle.sin() as f32);
                }
                emission
                    .queue()
                    .write_buffer(twiddle.as_ref(), 0, cast_slice(&tw));
            }
        }

        if let Ok(mut cache) = self.fft_twiddle_cache.lock() {
            cache.insert((len, mode), twiddle.clone());
        }
        Ok(twiddle)
    }
}
