use super::*;

impl WgpuProvider {
    pub(crate) fn read_scalar_exec(
        &self,
        handle: &GpuTensorHandle,
        linear_index: usize,
    ) -> Result<f64> {
        let entry = self.get_entry(handle)?;
        let elem_size = match entry.precision {
            NumericPrecision::F64 => std::mem::size_of::<f64>() as u64,
            NumericPrecision::F32 => std::mem::size_of::<f32>() as u64,
        };
        let total_bytes = (linear_index as u64)
            .checked_mul(elem_size)
            .ok_or_else(|| anyhow!("read_scalar: index overflow"))?;
        if (linear_index + 1) > entry.len {
            return Err(anyhow!(
                "read_scalar: index {} out of bounds (len {})",
                linear_index + 1,
                entry.len
            ));
        }
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("runmat-read-scalar-staging"),
            size: elem_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runmat-read-scalar-enc"),
            });
        encoder.copy_buffer_to_buffer(entry.buffer.as_ref(), total_bytes, &staging, 0, elem_size);
        self.submit(encoder);
        let bytes = self.map_readback_bytes_sync(staging, elem_size, "read_scalar")?;
        let value = match entry.precision {
            NumericPrecision::F64 => {
                let words: &[f64] = cast_slice(&bytes);
                words.first().copied().unwrap_or(0.0)
            }
            NumericPrecision::F32 => {
                let words: &[f32] = cast_slice(&bytes);
                words.first().copied().unwrap_or(0.0) as f64
            }
        };
        Ok(value)
    }

    pub(crate) fn upload_exec(&self, host: &HostTensorView) -> Result<GpuTensorHandle> {
        let _span = info_span!(
            "gpu.transfer.upload",
            shape = ?host.shape,
            len = host.data.len()
        )
        .entered();
        let len = host.data.len();
        let shape = host.shape.to_vec();
        let bytes = (len as u64).saturating_mul(self.element_size as u64);
        if bytes > self.adapter_limits.max_buffer_size {
            return Err(gpu_per_buffer_limit_error(
                "upload",
                bytes,
                self.adapter_limits.max_buffer_size,
            ));
        }
        let buffer =
            if len == 0 {
                self.create_storage_buffer(0, "runmat-upload-empty")
            } else {
                match self.precision {
                    NumericPrecision::F64 => {
                        let contents = cast_slice(host.data);
                        Arc::new(self.device.create_buffer_init(
                            &wgpu::util::BufferInitDescriptor {
                                label: Some("runmat-upload-buffer"),
                                contents,
                                usage: wgpu::BufferUsages::STORAGE
                                    | wgpu::BufferUsages::COPY_DST
                                    | wgpu::BufferUsages::COPY_SRC,
                            },
                        ))
                    }
                    NumericPrecision::F32 => {
                        let data_f32: Vec<f32> = host.data.iter().map(|v| *v as f32).collect();
                        let contents = cast_slice(&data_f32);
                        Arc::new(self.device.create_buffer_init(
                            &wgpu::util::BufferInitDescriptor {
                                label: Some("runmat-upload-buffer"),
                                contents,
                                usage: wgpu::BufferUsages::STORAGE
                                    | wgpu::BufferUsages::COPY_DST
                                    | wgpu::BufferUsages::COPY_SRC,
                            },
                        ))
                    }
                }
            };
        self.telemetry.record_upload_bytes(bytes);
        Ok(self.register_existing_buffer(buffer, shape, len))
    }

    pub(crate) async fn download_exec(&self, handle: &GpuTensorHandle) -> Result<HostTensorOwned> {
        let span = info_span!(
            "gpu.transfer.download",
            shape = ?handle.shape,
            buffer_id = handle.buffer_id
        );
        let entry = {
            let _guard = span.enter();
            log::trace!(
                "wgpu download id={} shape={:?}",
                handle.buffer_id,
                &handle.shape
            );
            self.get_entry(handle)?
        };
        if let Some(last) = entry.last_submission_id {
            log::trace!(
                "wgpu download id={} last_submission_id={}",
                handle.buffer_id,
                last
            );
        } else {
            log::trace!(
                "wgpu download id={} last_submission_id=<none>",
                handle.buffer_id
            );
        }
        let storage = if runmat_accelerate_api::handle_storage(handle)
            == GpuTensorStorage::ComplexInterleaved
        {
            GpuTensorStorage::ComplexInterleaved
        } else {
            entry.storage.clone()
        };
        if entry.len == 0 {
            return Ok(HostTensorOwned {
                data: Vec::new(),
                shape: handle.shape.clone(),
                storage,
            });
        }

        let size_bytes = (entry.len * self.element_size) as u64;
        let finish_readback = |staging: wgpu::Buffer, size_bytes: u64| -> Result<HostTensorOwned> {
            let slice = staging.slice(..);
            let data = slice.get_mapped_range();
            log::trace!(
                "wgpu download copying data id={} len={} bytes={}",
                handle.buffer_id,
                entry.len,
                size_bytes
            );

            let mut out = vec![0.0f64; entry.len];
            match entry.precision {
                NumericPrecision::F64 => out.copy_from_slice(cast_slice(&data)),
                NumericPrecision::F32 => {
                    let f32_slice: &[f32] = cast_slice(&data);
                    for (dst, src) in out.iter_mut().zip(f32_slice.iter()) {
                        *dst = *src as f64;
                    }
                }
            }
            drop(data);
            staging.unmap();
            log::trace!("wgpu download finished copy id={}", handle.buffer_id);
            self.telemetry.record_download_bytes(size_bytes);

            let lane_factor = match storage {
                GpuTensorStorage::Real => 1usize,
                GpuTensorStorage::ComplexInterleaved => 2usize,
            };
            let mut shape = handle.shape.clone();
            if let Some(info) = runmat_accelerate_api::handle_transpose_info(handle) {
                let base_rows = info.base_rows;
                let base_cols = info.base_cols;
                let logical_len = out.len() / lane_factor;
                if out.len() % lane_factor != 0
                    || base_rows.checked_mul(base_cols) != Some(logical_len)
                {
                    return Err(anyhow!(
                        "download: transpose metadata mismatch for buffer {}",
                        handle.buffer_id
                    ));
                }
                if shape.len() == 2 {
                    let rows_t = base_cols;
                    let cols_t = base_rows;
                    let mut transposed = vec![0.0f64; out.len()];
                    for col in 0..base_cols {
                        for row in 0..base_rows {
                            let src_idx = (row + col * base_rows) * lane_factor;
                            let dst_idx = (col + row * base_cols) * lane_factor;
                            transposed[dst_idx..dst_idx + lane_factor]
                                .copy_from_slice(&out[src_idx..src_idx + lane_factor]);
                        }
                    }
                    out = transposed;
                    shape[0] = rows_t;
                    shape[1] = cols_t;
                }
            }

            log::trace!(
                "wgpu download complete id={} final_shape={:?}",
                handle.buffer_id,
                shape
            );

            Ok(HostTensorOwned {
                data: out,
                shape,
                storage,
            })
        };

        log::trace!(
            "wgpu download creating staging buffer id={} bytes={}",
            handle.buffer_id,
            size_bytes
        );
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("runmat-download-staging"),
            size: size_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runmat-download-encoder"),
            });
        encoder.copy_buffer_to_buffer(entry.buffer.as_ref(), 0, &staging, 0, size_bytes);
        self.submit(encoder);
        let slice = staging.slice(..);
        let (tx, rx) = oneshot::channel();

        let map_buffer_id = handle.buffer_id;
        slice.map_async(wgpu::MapMode::Read, move |res| {
            log::trace!(
                "wgpu download map_async callback id={} status={:?}",
                map_buffer_id,
                res
            );
            let _ = tx.send(res);
        });
        log::trace!(
            "wgpu download awaiting map_async completion id={} bytes={}",
            handle.buffer_id,
            size_bytes
        );
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.device.poll(wgpu::Maintain::Wait);
        }
        let map_result = rx
            .await
            .map_err(|_| anyhow!("map_async callback dropped for buffer {}", handle.buffer_id))?;

        log::trace!("wgpu download map_async success id={}", handle.buffer_id);
        map_result.map_err(|e: wgpu::BufferAsyncError| anyhow!(e))?;
        finish_readback(staging, size_bytes)
    }

    pub(crate) fn free_exec(&self, handle: &GpuTensorHandle) -> Result<()> {
        log::trace!("wgpu free id={}", handle.buffer_id);
        let entry = self
            .buffers
            .lock()
            .expect("buffer mutex poisoned")
            .remove(&handle.buffer_id);
        if let Some(entry) = entry {
            if entry.len > 0 {
                let size_bytes = (entry.len as u64).saturating_mul(self.element_size as u64);
                let poolable_by_size = self.buffer_residency_max_poolable_bytes > 0
                    && size_bytes <= self.buffer_residency_max_poolable_bytes;
                let buffer_ptr = entry.buffer.as_ref() as *const wgpu::Buffer as usize;
                self.bind_group_cache.invalidate_buffer(buffer_ptr);
                let strong_count = Arc::strong_count(&entry.buffer);
                if poolable_by_size && strong_count == 1 {
                    self.buffer_residency
                        .release(entry.usage, entry.len, entry.buffer.clone());
                } else {
                    log::trace!(
                        "buffer_residency: not pooling buffer id={} len={} bytes={} strong_count={} poolable_by_size={}",
                        handle.buffer_id,
                        entry.len,
                        size_bytes,
                        strong_count,
                        poolable_by_size
                    );
                }
            }
        }
        self.kernel_resources.clear_matmul_source(handle.buffer_id);
        runmat_accelerate_api::clear_handle_logical(handle);
        runmat_accelerate_api::clear_handle_storage(handle);
        runmat_accelerate_api::clear_handle_transpose(handle);
        Ok(())
    }

    pub(crate) fn device_info_exec(&self) -> String {
        format!(
            "{} ({:?})",
            self.adapter_info.name, self.adapter_info.backend
        )
    }

    pub(crate) fn device_info_struct_exec(&self) -> ApiDeviceInfo {
        let backend = format!("{:?}", self.adapter_info.backend).to_ascii_lowercase();
        let memory_bytes = if self.adapter_limits.max_buffer_size > 0 {
            Some(self.adapter_limits.max_buffer_size)
        } else {
            None
        };
        ApiDeviceInfo {
            device_id: self.runtime_device_id,
            name: self.adapter_info.name.clone(),
            vendor: canonical_vendor_name(&self.adapter_info),
            memory_bytes,
            backend: Some(backend),
        }
    }
}
