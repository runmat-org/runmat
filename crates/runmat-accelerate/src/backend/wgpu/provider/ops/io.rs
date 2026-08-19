use super::*;
use runmat_accelerate_api::{
    HostIntegerDataOwned, HostIntegerDataView, HostIntegerTensorOwned, HostIntegerTensorView,
    HostNumericDataOwned, HostNumericDataView, HostNumericTensorOwned, HostNumericTensorView,
    IntegerElementType, NumericElementType,
};

fn integer_words(data: HostIntegerDataView<'_>) -> Vec<u32> {
    match data {
        HostIntegerDataView::I8(values) => {
            values.iter().map(|&value| value as i32 as u32).collect()
        }
        HostIntegerDataView::I16(values) => {
            values.iter().map(|&value| value as i32 as u32).collect()
        }
        HostIntegerDataView::I32(values) => values.iter().map(|&value| value as u32).collect(),
        HostIntegerDataView::U8(values) => values.iter().map(|&value| value as u32).collect(),
        HostIntegerDataView::U16(values) => values.iter().map(|&value| value as u32).collect(),
        HostIntegerDataView::U32(values) => values.to_vec(),
        HostIntegerDataView::I64(values) => values
            .iter()
            .flat_map(|&value| {
                let bits = value as u64;
                [bits as u32, (bits >> 32) as u32]
            })
            .collect(),
        HostIntegerDataView::U64(values) => values
            .iter()
            .flat_map(|&value| [value as u32, (value >> 32) as u32])
            .collect(),
    }
}

fn integer_view_from_numeric(data: HostNumericDataView<'_>) -> Result<HostIntegerDataView<'_>> {
    Ok(match data {
        HostNumericDataView::I8(values) => HostIntegerDataView::I8(values),
        HostNumericDataView::I16(values) => HostIntegerDataView::I16(values),
        HostNumericDataView::I32(values) => HostIntegerDataView::I32(values),
        HostNumericDataView::I64(values) => HostIntegerDataView::I64(values),
        HostNumericDataView::U8(values) => HostIntegerDataView::U8(values),
        HostNumericDataView::U16(values) => HostIntegerDataView::U16(values),
        HostNumericDataView::U32(values) => HostIntegerDataView::U32(values),
        HostNumericDataView::U64(values) => HostIntegerDataView::U64(values),
        HostNumericDataView::F64(_) | HostNumericDataView::F32(_) => {
            return Err(anyhow!(
                "integer word packing requested for floating storage"
            ));
        }
    })
}

fn numeric_data_from_bytes(
    element_type: NumericElementType,
    bytes: &[u8],
    len: usize,
) -> Result<HostNumericDataOwned> {
    let expected_bytes = match element_type.integer_type() {
        Some(IntegerElementType::I64 | IntegerElementType::U64) => len
            .checked_mul(2)
            .and_then(|words| words.checked_mul(std::mem::size_of::<u32>())),
        Some(_) => len.checked_mul(std::mem::size_of::<u32>()),
        None => len.checked_mul(element_type.element_size()),
    }
    .ok_or_else(|| anyhow!("numeric download byte length overflow"))?;
    ensure!(
        bytes.len() == expected_bytes,
        "numeric download byte count mismatch: got {}, expected {} for {:?}",
        bytes.len(),
        expected_bytes,
        element_type
    );
    if len == 0 {
        return Ok(match element_type {
            NumericElementType::F64 => HostNumericDataOwned::F64(Vec::new()),
            NumericElementType::F32 => HostNumericDataOwned::F32(Vec::new()),
            NumericElementType::I8 => HostNumericDataOwned::I8(Vec::new()),
            NumericElementType::I16 => HostNumericDataOwned::I16(Vec::new()),
            NumericElementType::I32 => HostNumericDataOwned::I32(Vec::new()),
            NumericElementType::I64 => HostNumericDataOwned::I64(Vec::new()),
            NumericElementType::U8 => HostNumericDataOwned::U8(Vec::new()),
            NumericElementType::U16 => HostNumericDataOwned::U16(Vec::new()),
            NumericElementType::U32 => HostNumericDataOwned::U32(Vec::new()),
            NumericElementType::U64 => HostNumericDataOwned::U64(Vec::new()),
        });
    }
    Ok(match element_type {
        NumericElementType::F64 => HostNumericDataOwned::F64(cast_slice(bytes).to_vec()),
        NumericElementType::F32 => HostNumericDataOwned::F32(cast_slice(bytes).to_vec()),
        NumericElementType::I8
        | NumericElementType::I16
        | NumericElementType::I32
        | NumericElementType::I64
        | NumericElementType::U8
        | NumericElementType::U16
        | NumericElementType::U32
        | NumericElementType::U64 => {
            let integer_type = element_type
                .integer_type()
                .expect("integer numeric element type");
            match integer_data_from_words(integer_type, cast_slice(bytes), len)? {
                HostIntegerDataOwned::I8(values) => HostNumericDataOwned::I8(values),
                HostIntegerDataOwned::I16(values) => HostNumericDataOwned::I16(values),
                HostIntegerDataOwned::I32(values) => HostNumericDataOwned::I32(values),
                HostIntegerDataOwned::I64(values) => HostNumericDataOwned::I64(values),
                HostIntegerDataOwned::U8(values) => HostNumericDataOwned::U8(values),
                HostIntegerDataOwned::U16(values) => HostNumericDataOwned::U16(values),
                HostIntegerDataOwned::U32(values) => HostNumericDataOwned::U32(values),
                HostIntegerDataOwned::U64(values) => HostNumericDataOwned::U64(values),
            }
        }
    })
}

fn transpose_numeric_values<T: Copy>(
    values: &mut Vec<T>,
    lane_factor: usize,
    base_rows: usize,
    base_cols: usize,
) {
    let mut transposed = values.clone();
    for col in 0..base_cols {
        for row in 0..base_rows {
            let src_idx = (row + col * base_rows) * lane_factor;
            let dst_idx = (col + row * base_cols) * lane_factor;
            transposed[dst_idx..dst_idx + lane_factor]
                .copy_from_slice(&values[src_idx..src_idx + lane_factor]);
        }
    }
    *values = transposed;
}

fn transpose_numeric_data(
    data: &mut HostNumericDataOwned,
    lane_factor: usize,
    base_rows: usize,
    base_cols: usize,
) {
    match data {
        HostNumericDataOwned::F64(values) => {
            transpose_numeric_values(values, lane_factor, base_rows, base_cols)
        }
        HostNumericDataOwned::F32(values) => {
            transpose_numeric_values(values, lane_factor, base_rows, base_cols)
        }
        HostNumericDataOwned::I8(values) => {
            transpose_numeric_values(values, lane_factor, base_rows, base_cols)
        }
        HostNumericDataOwned::I16(values) => {
            transpose_numeric_values(values, lane_factor, base_rows, base_cols)
        }
        HostNumericDataOwned::I32(values) => {
            transpose_numeric_values(values, lane_factor, base_rows, base_cols)
        }
        HostNumericDataOwned::I64(values) => {
            transpose_numeric_values(values, lane_factor, base_rows, base_cols)
        }
        HostNumericDataOwned::U8(values) => {
            transpose_numeric_values(values, lane_factor, base_rows, base_cols)
        }
        HostNumericDataOwned::U16(values) => {
            transpose_numeric_values(values, lane_factor, base_rows, base_cols)
        }
        HostNumericDataOwned::U32(values) => {
            transpose_numeric_values(values, lane_factor, base_rows, base_cols)
        }
        HostNumericDataOwned::U64(values) => {
            transpose_numeric_values(values, lane_factor, base_rows, base_cols)
        }
    }
}

fn integer_data_from_words(
    element_type: IntegerElementType,
    words: &[u32],
    len: usize,
) -> Result<HostIntegerDataOwned> {
    if matches!(
        element_type,
        IntegerElementType::I64 | IntegerElementType::U64
    ) && words.len() != len.saturating_mul(2)
    {
        return Err(anyhow!("integer download: 64-bit word count mismatch"));
    }
    if !matches!(
        element_type,
        IntegerElementType::I64 | IntegerElementType::U64
    ) && words.len() != len
    {
        return Err(anyhow!("integer download: word count mismatch"));
    }
    Ok(match element_type {
        IntegerElementType::I8 => {
            HostIntegerDataOwned::I8(words.iter().map(|&v| v as i8).collect())
        }
        IntegerElementType::I16 => {
            HostIntegerDataOwned::I16(words.iter().map(|&v| v as i16).collect())
        }
        IntegerElementType::I32 => {
            HostIntegerDataOwned::I32(words.iter().map(|&v| v as i32).collect())
        }
        IntegerElementType::U8 => {
            HostIntegerDataOwned::U8(words.iter().map(|&v| v as u8).collect())
        }
        IntegerElementType::U16 => {
            HostIntegerDataOwned::U16(words.iter().map(|&v| v as u16).collect())
        }
        IntegerElementType::U32 => HostIntegerDataOwned::U32(words.to_vec()),
        IntegerElementType::I64 => HostIntegerDataOwned::I64(
            words
                .chunks_exact(2)
                .map(|pair| ((pair[0] as u64) | ((pair[1] as u64) << 32)) as i64)
                .collect(),
        ),
        IntegerElementType::U64 => HostIntegerDataOwned::U64(
            words
                .chunks_exact(2)
                .map(|pair| (pair[0] as u64) | ((pair[1] as u64) << 32))
                .collect(),
        ),
    })
}

impl WgpuProvider {
    pub(crate) fn read_scalar_exec(
        &self,
        handle: &GpuTensorHandle,
        linear_index: usize,
    ) -> Result<f64> {
        let entry = self.get_entry(handle)?;
        let elem_size = entry.element_type.element_size() as u64;
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

    pub(crate) fn upload_integer_exec(
        &self,
        host: &HostIntegerTensorView,
    ) -> Result<GpuTensorHandle> {
        let data = match host.data {
            HostIntegerDataView::I8(values) => HostNumericDataView::I8(values),
            HostIntegerDataView::I16(values) => HostNumericDataView::I16(values),
            HostIntegerDataView::I32(values) => HostNumericDataView::I32(values),
            HostIntegerDataView::I64(values) => HostNumericDataView::I64(values),
            HostIntegerDataView::U8(values) => HostNumericDataView::U8(values),
            HostIntegerDataView::U16(values) => HostNumericDataView::U16(values),
            HostIntegerDataView::U32(values) => HostNumericDataView::U32(values),
            HostIntegerDataView::U64(values) => HostNumericDataView::U64(values),
        };
        self.upload_numeric_exec(&HostNumericTensorView {
            data,
            shape: host.shape,
            storage: GpuTensorStorage::Real,
        })
    }

    pub(crate) fn upload_numeric_exec(
        &self,
        host: &HostNumericTensorView,
    ) -> Result<GpuTensorHandle> {
        host.validate()?;
        let element_type = host.element_type();
        let len = host.data.len();
        let integer_words = if element_type.integer_type().is_some() {
            Some(integer_words(integer_view_from_numeric(host.data)?))
        } else {
            None
        };
        let bytes = integer_words.as_ref().map_or_else(
            || (len as u64).saturating_mul(element_type.element_size() as u64),
            |words| (words.len() as u64).saturating_mul(std::mem::size_of::<u32>() as u64),
        );
        if bytes > self.adapter_limits.max_buffer_size {
            return Err(gpu_per_buffer_limit_error(
                "numeric upload",
                bytes,
                self.adapter_limits.max_buffer_size,
            ));
        }
        let buffer = if len == 0 {
            self.create_storage_buffer(0, "runmat-numeric-upload-empty")
        } else {
            let contents = match host.data {
                HostNumericDataView::F64(values) => cast_slice(values),
                HostNumericDataView::F32(values) => cast_slice(values),
                HostNumericDataView::I8(_)
                | HostNumericDataView::I16(_)
                | HostNumericDataView::I32(_)
                | HostNumericDataView::I64(_)
                | HostNumericDataView::U8(_)
                | HostNumericDataView::U16(_)
                | HostNumericDataView::U32(_)
                | HostNumericDataView::U64(_) => cast_slice(
                    integer_words
                        .as_ref()
                        .expect("integer numeric upload words"),
                ),
            };
            Arc::new(
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("runmat-numeric-upload-buffer"),
                        contents,
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::COPY_SRC,
                    }),
            )
        };
        self.telemetry.record_upload_bytes(bytes);
        Ok(self.register_numeric_buffer(
            buffer,
            NumericBufferRegistration {
                shape: host.shape.to_vec(),
                len,
                physical_element_type: element_type,
                storage: host.storage,
                allocated_bytes: bytes,
                usage: crate::backend::wgpu::residency::BufferUsageClass::Generic,
            },
        ))
    }

    pub(crate) async fn download_integer_exec(
        &self,
        handle: &GpuTensorHandle,
    ) -> Result<HostIntegerTensorOwned> {
        let numeric = self.download_numeric_exec(handle).await?;
        ensure!(
            numeric.storage == GpuTensorStorage::Real,
            "legacy integer download cannot represent complex integer storage"
        );
        let data = match numeric.data {
            HostNumericDataOwned::I8(values) => HostIntegerDataOwned::I8(values),
            HostNumericDataOwned::I16(values) => HostIntegerDataOwned::I16(values),
            HostNumericDataOwned::I32(values) => HostIntegerDataOwned::I32(values),
            HostNumericDataOwned::I64(values) => HostIntegerDataOwned::I64(values),
            HostNumericDataOwned::U8(values) => HostIntegerDataOwned::U8(values),
            HostNumericDataOwned::U16(values) => HostIntegerDataOwned::U16(values),
            HostNumericDataOwned::U32(values) => HostIntegerDataOwned::U32(values),
            HostNumericDataOwned::U64(values) => HostIntegerDataOwned::U64(values),
            HostNumericDataOwned::F64(_) | HostNumericDataOwned::F32(_) => {
                return Err(anyhow!(
                    "integer download requested for non-integer gpuArray buffer"
                ));
            }
        };
        Ok(HostIntegerTensorOwned {
            data,
            shape: numeric.shape,
        })
    }

    pub(crate) async fn download_numeric_exec(
        &self,
        handle: &GpuTensorHandle,
    ) -> Result<HostNumericTensorOwned> {
        let entry = self.get_entry_raw(handle)?;
        let bytes = if entry.allocated_bytes == 0 {
            Vec::new()
        } else {
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("runmat-numeric-download-staging"),
                size: entry.allocated_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-numeric-download-encoder"),
                });
            encoder.copy_buffer_to_buffer(
                entry.buffer.as_ref(),
                0,
                &staging,
                0,
                entry.allocated_bytes,
            );
            self.submit(encoder);
            self.map_readback_bytes(staging, entry.allocated_bytes, "numeric download")
                .await?
        };
        let mut data = numeric_data_from_bytes(entry.element_type, &bytes, entry.len)?;
        let lane_factor = match entry.storage {
            GpuTensorStorage::Real => 1usize,
            GpuTensorStorage::ComplexInterleaved => 2usize,
        };
        let mut shape = handle.shape.clone();
        if let Some(info) = runmat_accelerate_api::handle_transpose_info(handle) {
            let logical_len = data.len() / lane_factor;
            ensure!(
                data.len() % lane_factor == 0
                    && info.base_rows.checked_mul(info.base_cols) == Some(logical_len),
                "numeric download: transpose metadata mismatch for buffer {}",
                handle.buffer_id
            );
            if shape.len() == 2 {
                transpose_numeric_data(&mut data, lane_factor, info.base_rows, info.base_cols);
                shape[0] = info.base_cols;
                shape[1] = info.base_rows;
            }
        }
        self.telemetry.record_download_bytes(entry.allocated_bytes);
        let owned = HostNumericTensorOwned {
            data,
            shape,
            storage: entry.storage,
        };
        owned.validate()?;
        Ok(owned)
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
            let entry = self.get_entry_raw(handle)?;
            ensure!(
                entry.integer_type().is_none(),
                "legacy floating download cannot represent native integer storage"
            );
            entry
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
            entry.storage
        };
        if entry.len == 0 {
            return Ok(HostTensorOwned {
                data: Vec::new(),
                shape: handle.shape.clone(),
                storage,
            });
        }

        let size_bytes = entry.allocated_bytes;
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
        if handle.device_id != self.runtime_device_id {
            return Err(anyhow!(
                "free: handle belongs to device {}, but this provider owns device {}",
                handle.device_id,
                self.runtime_device_id
            ));
        }
        log::trace!("wgpu free id={}", handle.buffer_id);
        let entry = self
            .buffers
            .lock()
            .expect("buffer mutex poisoned")
            .remove(&handle.buffer_id);
        if let Some(entry) = entry {
            if entry.len > 0 {
                let size_bytes = entry.allocated_bytes;
                let poolable_by_size = self.buffer_residency_max_poolable_bytes > 0
                    && size_bytes <= self.buffer_residency_max_poolable_bytes;
                let buffer_ptr = entry.buffer.as_ref() as *const wgpu::Buffer as usize;
                self.bind_group_cache.invalidate_buffer(buffer_ptr);
                let strong_count = Arc::strong_count(&entry.buffer);
                if poolable_by_size && strong_count == 1 {
                    self.buffer_residency.release(
                        entry.usage,
                        entry.allocated_bytes,
                        entry.buffer.clone(),
                    );
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
        runmat_accelerate_api::clear_handle_metadata(handle);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_rejects_handles_owned_by_another_device() {
        let Ok(provider) = WgpuProvider::new(WgpuProviderOptions::default()) else {
            return;
        };
        let handle = provider
            .upload_exec(&HostTensorView {
                data: &[1.0, 2.0],
                shape: &[2, 1],
            })
            .expect("upload");
        let foreign = GpuTensorHandle {
            device_id: handle.device_id.wrapping_add(1),
            ..handle.clone()
        };

        let error = provider
            .free_exec(&foreign)
            .expect_err("foreign handle must not free a local buffer");
        assert!(error.to_string().contains("belongs to device"));
        futures::executor::block_on(provider.download_exec(&handle))
            .expect("the locally owned buffer must remain live");
        provider.free_exec(&handle).expect("free local handle");
    }
}
