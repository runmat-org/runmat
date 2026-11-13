use super::allocation_state::AllocationState;
use super::*;

enum NumericReadbackStaging {
    Ordinary(super::allocation_state::NumericDownloadStageHandle),
    Selected(runmat_accelerate_api::PreparedNumericDownloadStage),
}

impl std::borrow::Borrow<wgpu::Buffer> for NumericReadbackStaging {
    fn borrow(&self) -> &wgpu::Buffer {
        match self {
            Self::Ordinary(owner) => owner
                .staging_buffer()
                .expect("ordinary numeric staging owns its buffer"),
            Self::Selected(stage) => stage
                .lease_as_any()
                .downcast_ref::<super::allocation_state::NumericDownloadStageHandle>()
                .and_then(|lease| lease.staging_buffer())
                .expect("selected numeric staging owns the authenticated buffer"),
        }
    }
}

impl NumericReadbackStaging {
    fn owner(&self) -> &super::allocation_state::NumericDownloadStageHandle {
        match self {
            Self::Ordinary(owner) => owner,
            Self::Selected(stage) => stage
                .lease_as_any()
                .downcast_ref()
                .expect("selected numeric staging owns its authenticated owner"),
        }
    }
}
impl super::core::ReadbackResourceOwner for NumericReadbackStaging {
    fn readback_buffer(&self) -> Option<&runmat_accelerate_api::WgpuNumericReadbackBuffer> {
        self.owner().readback_buffer()
    }
    fn validate_scope(&self) -> Result<()> {
        self.owner().validate_scope()
    }
    fn numeric_callback_owner(
        &self,
    ) -> Option<super::allocation_state::NumericDownloadStageHandle> {
        Some(self.owner().callback_owner())
    }
}
#[path = "io/power_probe.rs"]
mod power_probe;
use power_probe::ProbeStaging;
pub(crate) use power_probe::{power_probe_host_workspace_bytes, PowerProbeReadbackObserved};
use runmat_accelerate_api::{
    HostIntegerDataOwned, HostIntegerDataView, HostIntegerTensorOwned, HostIntegerTensorView,
    HostNumericDataOwned, HostNumericDataView, HostNumericTensorOwned, HostNumericTensorView,
    IntegerElementType, NumericElementType,
};

impl WgpuProvider {
    /// Side-effect-free exact admission for this owner’s native readback,
    /// including the logical transpose performed after mapped staging.
    pub(crate) fn numeric_download_feasibility_exec(
        &self,
        source: &GpuTensorHandle,
        query: &runmat_accelerate_api::ProviderFeasibilityQuery,
    ) -> runmat_accelerate_api::ProviderFeasibility {
        use runmat_accelerate_api::{
            ProviderFeasibility, ProviderRejectionCode, ProviderResourceEstimate,
        };
        if !query.matches_numeric_download(source) || source.device_id() != self.device_id() {
            return ProviderFeasibility::rejected(
                ProviderRejectionCode::InvalidShape,
                "provider.download.exact_contract",
            );
        }
        if source.allocation_lease().validate_live().is_err() {
            return ProviderFeasibility::rejected(
                ProviderRejectionCode::ProviderUnavailable,
                "provider.download.source_unavailable",
            );
        }
        let Ok(entry) = self.get_entry_raw(source) else {
            return ProviderFeasibility::rejected(
                ProviderRejectionCode::ProviderUnavailable,
                "provider.download.source_owner",
            );
        };
        if !numeric_readback_record_matches(&entry, source) {
            return ProviderFeasibility::rejected(
                ProviderRejectionCode::UnknownResourceEstimate,
                "provider.download.physical_mismatch",
            );
        }
        let staging = if entry.len == 0 {
            0
        } else {
            entry.allocated_bytes
        };
        if staging > self.adapter_limits.max_buffer_size {
            return ProviderFeasibility::rejected(
                ProviderRejectionCode::ResourceLimit,
                "provider.download.staging_limit",
            );
        }
        ProviderFeasibility::supported(ProviderResourceEstimate {
            // The shared resource hook proves staging, but this estimate does
            // not claim a whole-device peak across other borrowed allocations.
            transient_bytes: None,
            output_bytes: query.outputs[0].checked_byte_len(),
            dispatches: Some(u32::from(staging > 0)),
        })
    }
}

/// Older generic registrations record logical payload bytes in `allocated_bytes`;
/// produced uploads record buffer extent there. The authenticated physical
/// descriptor and actual buffer must agree on extent in either case. In
/// particular an empty payload may own one physical element while requiring
/// no readback staging or transfer.
fn numeric_readback_record_matches(entry: &BufferEntry, source: &GpuTensorHandle) -> bool {
    // BufferEntry counts component lanes, whereas descriptor/view capacity
    // counts logical elements. Keep the exact checked correspondence.
    let lanes = match entry.storage {
        GpuTensorStorage::Real => 1,
        GpuTensorStorage::ComplexInterleaved => 2,
    };
    let Some(component_capacity) = source.physical().element_capacity().checked_mul(lanes) else {
        return false;
    };
    entry.element_type == source.physical().element_type()
        && entry.storage == source.physical().storage()
        && u64::try_from(entry.len).ok() == Some(component_capacity)
        && source.view().checked_element_count().ok() == Some(source.physical().element_capacity())
        && numeric_download_extents_match(
            entry.allocated_bytes,
            entry.buffer.size(),
            source.physical().payload_bytes(),
            source.physical().requested_bytes(),
        )
}

fn numeric_download_extents_match(
    recorded_bytes: u64,
    buffer_bytes: u64,
    payload_bytes: u64,
    requested_bytes: u64,
) -> bool {
    payload_bytes <= requested_bytes
        && buffer_bytes == requested_bytes
        && (recorded_bytes == payload_bytes || recorded_bytes == requested_bytes)
}

fn record_provider_copy(reason: runmat_value::HostCopyReason, bytes: u64) {
    runmat_value::record_host_copy(reason, usize::try_from(bytes).unwrap_or(usize::MAX));
}

fn integer_words(data: HostIntegerDataView<'_>) -> Vec<u32> {
    let count = integer_encoding_words(data.element_type(), data.len())
        .expect("validated integer upload extent");
    let mut output = vec![0; count];
    fill_integer_words(data, &mut output).expect("exact ordinary encoding extent");
    output
}

pub(super) fn integer_encoding_words(element: IntegerElementType, lanes: usize) -> Option<usize> {
    lanes.checked_mul(
        if matches!(element, IntegerElementType::I64 | IntegerElementType::U64) {
            2
        } else {
            1
        },
    )
}

fn fill_integer_words(data: HostIntegerDataView<'_>, output: &mut [u32]) -> Result<()> {
    ensure!(
        Some(output.len()) == integer_encoding_words(data.element_type(), data.len()),
        "native upload encoding workspace extent changed"
    );
    macro_rules! fill {
        ($values:expr, $convert:expr) => {
            for (slot, &value) in output.iter_mut().zip($values) {
                *slot = $convert(value);
            }
        };
    }
    match data {
        HostIntegerDataView::I8(values) => fill!(values, |v: i8| v as i32 as u32),
        HostIntegerDataView::I16(values) => fill!(values, |v: i16| v as i32 as u32),
        HostIntegerDataView::I32(values) => fill!(values, |v: i32| v as u32),
        HostIntegerDataView::U8(values) => fill!(values, |v: u8| u32::from(v)),
        HostIntegerDataView::U16(values) => fill!(values, |v: u16| u32::from(v)),
        HostIntegerDataView::U32(values) => output.copy_from_slice(values),
        HostIntegerDataView::I64(values) => {
            for (words, &value) in output.chunks_exact_mut(2).zip(values) {
                let bits = value as u64;
                words[0] = bits as u32;
                words[1] = (bits >> 32) as u32;
            }
        }
        HostIntegerDataView::U64(values) => {
            for (words, &value) in output.chunks_exact_mut(2).zip(values) {
                words[0] = value as u32;
                words[1] = (value >> 32) as u32;
            }
        }
    }
    Ok(())
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
        NumericElementType::F64 => HostNumericDataOwned::F64(
            runmat_value::exact_capacity_vec_from_slice(cast_slice(bytes))?,
        ),
        NumericElementType::F32 => HostNumericDataOwned::F32(
            runmat_value::exact_capacity_vec_from_slice(cast_slice(bytes))?,
        ),
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
) -> Result<()> {
    let mut transposed = runmat_value::exact_capacity_vec_from_slice(values)?;
    for col in 0..base_cols {
        for row in 0..base_rows {
            let src_idx = (row + col * base_rows) * lane_factor;
            let dst_idx = (col + row * base_cols) * lane_factor;
            transposed[dst_idx..dst_idx + lane_factor]
                .copy_from_slice(&values[src_idx..src_idx + lane_factor]);
        }
    }
    *values = transposed;
    Ok(())
}

fn transpose_numeric_data(
    data: &mut HostNumericDataOwned,
    lane_factor: usize,
    base_rows: usize,
    base_cols: usize,
) -> Result<()> {
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
    }?;
    Ok(())
}

fn integer_data_from_words(
    element_type: IntegerElementType,
    words: &[u32],
    len: usize,
) -> Result<HostIntegerDataOwned> {
    if matches!(
        element_type,
        IntegerElementType::I64 | IntegerElementType::U64
    ) && len.checked_mul(2) != Some(words.len())
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
            HostIntegerDataOwned::I8(runmat_value::exact_capacity_vec_with(len, |i| {
                words[i] as i8
            })?)
        }
        IntegerElementType::I16 => {
            HostIntegerDataOwned::I16(runmat_value::exact_capacity_vec_with(len, |i| {
                words[i] as i16
            })?)
        }
        IntegerElementType::I32 => {
            HostIntegerDataOwned::I32(runmat_value::exact_capacity_vec_with(len, |i| {
                words[i] as i32
            })?)
        }
        IntegerElementType::U8 => {
            HostIntegerDataOwned::U8(runmat_value::exact_capacity_vec_with(len, |i| {
                words[i] as u8
            })?)
        }
        IntegerElementType::U16 => {
            HostIntegerDataOwned::U16(runmat_value::exact_capacity_vec_with(len, |i| {
                words[i] as u16
            })?)
        }
        IntegerElementType::U32 => {
            HostIntegerDataOwned::U32(runmat_value::exact_capacity_vec_from_slice(words)?)
        }
        IntegerElementType::I64 => {
            HostIntegerDataOwned::I64(runmat_value::exact_capacity_vec_with(len, |i| {
                ((words[2 * i] as u64) | ((words[2 * i + 1] as u64) << 32)) as i64
            })?)
        }
        IntegerElementType::U64 => {
            HostIntegerDataOwned::U64(runmat_value::exact_capacity_vec_with(len, |i| {
                (words[2 * i] as u64) | ((words[2 * i + 1] as u64) << 32)
            })?)
        }
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
        let staging = self.with_emission(|emission, pending| -> Result<_> {
            let staging = runmat_accelerate_api::WgpuBufferOwner::new(
                emission.device().create_buffer(&wgpu::BufferDescriptor {
                    label: Some("runmat-read-scalar-staging"),
                    size: elem_size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
            );
            let mut encoder = pending.profiled_encoder(
                emission,
                &wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-read-scalar-enc"),
                },
            );
            encoder.copy_buffer_to_buffer(
                entry.buffer.as_ref(),
                total_bytes,
                &staging,
                0,
                elem_size,
            );
            self.submit(emission, pending, encoder)?;
            Ok(staging)
        })??;
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
        let buffer = self.with_emission(|emission, _| -> Result<_> {
            let buffer = if len == 0 {
                self.create_storage_buffer(emission, 0, "runmat-upload-empty")?
            } else {
                match self.precision {
                    NumericPrecision::F64 => {
                        let contents = cast_slice(host.data);
                        runmat_accelerate_api::WgpuBufferOwner::new(
                            emission.create_initialized_buffer(&wgpu::util::BufferInitDescriptor {
                                label: Some("runmat-upload-buffer"),
                                contents,
                                usage: wgpu::BufferUsages::STORAGE
                                    | wgpu::BufferUsages::COPY_DST
                                    | wgpu::BufferUsages::COPY_SRC,
                            }),
                        )
                    }
                    NumericPrecision::F32 => {
                        let data_f32: Vec<f32> = host.data.iter().map(|v| *v as f32).collect();
                        let contents = cast_slice(&data_f32);
                        runmat_accelerate_api::WgpuBufferOwner::new(
                            emission.create_initialized_buffer(&wgpu::util::BufferInitDescriptor {
                                label: Some("runmat-upload-buffer"),
                                contents,
                                usage: wgpu::BufferUsages::STORAGE
                                    | wgpu::BufferUsages::COPY_DST
                                    | wgpu::BufferUsages::COPY_SRC,
                            }),
                        )
                    }
                }
            };
            Ok(buffer)
        })??;
        self.telemetry.record_upload_bytes(bytes);
        record_provider_copy(runmat_value::HostCopyReason::ProviderUpload, bytes);
        self.register_existing_buffer(buffer, shape, len)
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
        self.upload_numeric_produced(host, false)
    }

    /// Report the first buffer-creation effect to the typed raw hook while
    /// keeping optional request-local metadata admission at registration.
    pub(crate) fn upload_numeric_exec_observed_admitted_with_scratch(
        &self,
        host: &HostNumericTensorView,
        on_buffer_created: impl FnOnce() -> Result<()>,
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
        scratch: Option<runmat_accelerate_api::HostNumericUploadScratch<'_>>,
    ) -> Result<GpuTensorHandle> {
        self.upload_numeric_produced_observed_with_scratch(
            host,
            false,
            on_buffer_created,
            admission,
            scratch,
        )
    }

    pub(crate) fn upload_logical_exec_observed_admitted(
        &self,
        host: &runmat_accelerate_api::HostLogicalTensorView<'_>,
        on_buffer_created: impl FnOnce() -> Result<()>,
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
        scratch: Option<runmat_accelerate_api::HostLogicalUploadScratch<'_>>,
    ) -> Result<GpuTensorHandle> {
        host.validate()?;
        match (self.precision, scratch) {
            (
                NumericPrecision::F32,
                Some(runmat_accelerate_api::HostLogicalUploadScratch::F32(data)),
            ) => {
                anyhow::ensure!(
                    data.len() == host.data.len(),
                    "logical F32 upload workspace extent changed"
                );
                for (&source, destination) in host.data.iter().zip(data.iter_mut()) {
                    *destination = f32::from(source);
                }
                self.upload_numeric_produced_observed(
                    &HostNumericTensorView {
                        data: HostNumericDataView::F32(data),
                        shape: host.shape,
                        storage: GpuTensorStorage::Real,
                    },
                    true,
                    on_buffer_created,
                    admission,
                )
            }
            (
                NumericPrecision::F64,
                Some(runmat_accelerate_api::HostLogicalUploadScratch::F64(data)),
            ) => {
                anyhow::ensure!(
                    data.len() == host.data.len(),
                    "logical F64 upload workspace extent changed"
                );
                for (&source, destination) in host.data.iter().zip(data.iter_mut()) {
                    *destination = f64::from(source);
                }
                self.upload_numeric_produced_observed(
                    &HostNumericTensorView {
                        data: HostNumericDataView::F64(data),
                        shape: host.shape,
                        storage: GpuTensorStorage::Real,
                    },
                    true,
                    on_buffer_created,
                    admission,
                )
            }
            (NumericPrecision::F32, None) if admission.is_none() => {
                let mut data = Vec::new();
                data.try_reserve_exact(host.data.len())
                    .map_err(|source| anyhow::anyhow!("logical F32 upload workspace: {source}"))?;
                data.extend(host.data.iter().map(|&value| f32::from(value)));
                self.upload_numeric_produced_observed(
                    &HostNumericTensorView {
                        data: HostNumericDataView::F32(&data),
                        shape: host.shape,
                        storage: GpuTensorStorage::Real,
                    },
                    true,
                    on_buffer_created,
                    admission,
                )
            }
            (NumericPrecision::F64, None) if admission.is_none() => {
                let mut data = Vec::new();
                data.try_reserve_exact(host.data.len())
                    .map_err(|source| anyhow::anyhow!("logical F64 upload workspace: {source}"))?;
                data.extend(host.data.iter().map(|&value| f64::from(value)));
                self.upload_numeric_produced_observed(
                    &HostNumericTensorView {
                        data: HostNumericDataView::F64(&data),
                        shape: host.shape,
                        storage: GpuTensorStorage::Real,
                    },
                    true,
                    on_buffer_created,
                    admission,
                )
            }
            _ => anyhow::bail!("admitted logical upload lacks matching native workspace"),
        }
    }

    fn upload_numeric_produced(
        &self,
        host: &HostNumericTensorView<'_>,
        logical: bool,
    ) -> Result<GpuTensorHandle> {
        self.upload_numeric_produced_observed(host, logical, || Ok(()), None)
    }

    fn upload_numeric_produced_observed(
        &self,
        host: &HostNumericTensorView<'_>,
        logical: bool,
        on_buffer_created: impl FnOnce() -> Result<()>,
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> Result<GpuTensorHandle> {
        self.upload_numeric_produced_observed_with_scratch(
            host,
            logical,
            on_buffer_created,
            admission,
            None,
        )
    }

    fn upload_numeric_produced_observed_with_scratch(
        &self,
        host: &HostNumericTensorView,
        logical: bool,
        on_buffer_created: impl FnOnce() -> Result<()>,
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
        scratch: Option<runmat_accelerate_api::HostNumericUploadScratch<'_>>,
    ) -> Result<GpuTensorHandle> {
        host.validate()?;
        let element_type = host.element_type();
        let len = host.data.len();
        let mut ordinary_words = None;
        let integer_words = if element_type.integer_type().is_some() {
            let data = integer_view_from_numeric(host.data)?;
            Some(if let Some(scratch) = scratch {
                fill_integer_words(data, scratch.words)?;
                &*scratch.words
            } else {
                ordinary_words.insert(integer_words(data)).as_slice()
            })
        } else {
            ensure!(
                scratch.is_none(),
                "floating upload received integer encoding scratch"
            );
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
        let buffer = self.with_emission(|emission, _| -> Result<_> {
            let buffer = if len == 0 {
                self.create_storage_buffer(emission, 0, "runmat-numeric-upload-empty")?
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
                runmat_accelerate_api::WgpuBufferOwner::new(emission.create_initialized_buffer(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("runmat-numeric-upload-buffer"),
                        contents,
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::COPY_SRC,
                    },
                ))
            };
            on_buffer_created()?;
            Ok(buffer)
        })??;
        self.telemetry.record_upload_bytes(bytes);
        record_provider_copy(runmat_value::HostCopyReason::ProviderUpload, bytes);
        let allocated_bytes = buffer.size();
        self.register_numeric_buffer_produced_admitted(
            buffer,
            NumericBufferRegistration {
                shape: host.shape.to_vec(),
                len,
                physical_element_type: element_type,
                storage: host.storage,
                allocated_bytes,
                usage: crate::backend::wgpu::residency::BufferUsageClass::Generic,
            },
            logical,
            admission,
        )
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
        self.download_numeric_exec_with_claim(handle, None).await
    }

    /// This is the exact concrete future boxed by the admitted provider hook.
    /// Constructing it for a size quote does not poll or allocate it.
    pub(crate) async fn download_numeric_exec_with_claim(
        &self,
        handle: &GpuTensorHandle,
        claim: Option<std::sync::Arc<dyn runmat_execution::resource::RetainedHostAllocationClaim>>,
    ) -> Result<HostNumericTensorOwned> {
        self.download_numeric_exec_with_claim_and_stage(handle, claim, None, None)
            .await
    }

    pub(crate) async fn download_numeric_exec_with_claim_and_stage(
        &self,
        handle: &GpuTensorHandle,
        claim: Option<std::sync::Arc<dyn runmat_execution::resource::RetainedHostAllocationClaim>>,
        mut stage: Option<runmat_accelerate_api::PreparedNumericDownloadStage>,
        readback: Option<&runmat_accelerate_api::NumericReadback<'_>>,
    ) -> Result<HostNumericTensorOwned> {
        if let Some(readback) = readback {
            readback.authorize_provider(self)?;
            ensure!(
                readback.source().view_identity() == handle.view_identity(),
                "numeric readback source changed"
            );
        }
        if let Some(stage) = &stage {
            if !stage.matches_source(handle)
                || !stage
                    .lease_as_any()
                    .downcast_ref::<super::allocation_state::NumericDownloadStageHandle>()
                    .is_some_and(|lease| lease.matches_allocation_owner(&self.allocations))
                || stage.provider_scratch_bytes()
                    != if handle.physical().element_capacity() == 0 {
                        0
                    } else {
                        handle.physical().requested_bytes()
                    }
                || stage.queue_slots() != u32::from(handle.physical().payload_bytes() != 0)
            {
                return Err(anyhow!("numeric download staging authority changed"));
            }
        }
        let entry = self.get_entry_raw(handle)?;
        if readback.is_some() {
            ensure!(
                numeric_readback_record_matches(&entry, handle),
                "numeric readback physical source record changed"
            );
        }
        if stage.is_some() && claim.is_none() {
            return Err(anyhow!("numeric staging lacks admitted completion control"));
        }
        let mut data = if entry.len == 0 {
            numeric_data_from_bytes(entry.element_type, &[], entry.len)?
        } else {
            let callback_stage = stage
                .as_mut()
                .map(|s| {
                    s.take_callback()
                        .ok_or_else(|| anyhow!("numeric staging callback already issued"))
                })
                .transpose()?;
            let staging = if readback.is_some() {
                // Match the allocation record before minting fixed physical
                // authority. No device/encoder callback is admitted here.
                let source = runmat_accelerate_api::WgpuNumericReadbackSource::new(
                    &self.emission_authority,
                    self,
                    handle,
                    &entry.buffer,
                )?;
                let staging = match stage.take() {
                    Some(stage) => NumericReadbackStaging::Selected(stage),
                    None => NumericReadbackStaging::Ordinary(
                        AllocationState::own_numeric_download_stage(
                            &self.allocations,
                            Some(super::allocation_state::NumericStageBuffer::Readback(
                                source.create_stage(entry.allocated_bytes)?,
                            )),
                            entry.allocated_bytes,
                            1,
                            claim.clone(),
                            None,
                            None,
                        )?,
                    ),
                };
                let buffer = staging.owner().readback_buffer().ok_or_else(|| {
                    anyhow!("numeric readback stage lacks exact source authority")
                })?;
                self.submit_numeric_readback(&source, buffer)?;
                staging
            } else {
                self.with_emission(|emission, pending| -> Result<_> {
                    let staging = match stage.take() {
                        Some(stage) => NumericReadbackStaging::Selected(stage),
                        None => NumericReadbackStaging::Ordinary(
                            AllocationState::own_numeric_download_stage(
                                &self.allocations,
                                Some(
                                    runmat_accelerate_api::WgpuBufferOwner::new(
                                        emission.device().create_buffer(&wgpu::BufferDescriptor {
                                            label: Some("runmat-numeric-download-staging"),
                                            size: entry.allocated_bytes,
                                            usage: wgpu::BufferUsages::MAP_READ
                                                | wgpu::BufferUsages::COPY_DST,
                                            mapped_at_creation: false,
                                        }),
                                    )
                                    .into(),
                                ),
                                entry.allocated_bytes,
                                1,
                                claim.clone(),
                                None,
                                None,
                            )?,
                        ),
                    };
                    let mut encoder = pending.profiled_encoder(
                        emission,
                        &wgpu::CommandEncoderDescriptor {
                            label: Some("runmat-numeric-download-encoder"),
                        },
                    );
                    encoder.copy_buffer_to_buffer(
                        entry.buffer.as_ref(),
                        0,
                        std::borrow::Borrow::<wgpu::Buffer>::borrow(&staging),
                        0,
                        entry.allocated_bytes,
                    );
                    self.submit(emission, pending, encoder)?;
                    Ok(staging)
                })??
            };
            match claim {
                Some(claim) => {
                    if let Some(callback_stage) = callback_stage {
                        self.map_readback_with_admitted_stage(
                            staging,
                            "numeric download",
                            |bytes| numeric_data_from_bytes(entry.element_type, bytes, entry.len),
                            claim,
                            callback_stage,
                        )
                        .await?
                    } else {
                        self.map_readback_with_admitted(
                            staging,
                            "numeric download",
                            |bytes| numeric_data_from_bytes(entry.element_type, bytes, entry.len),
                            claim,
                        )
                        .await?
                    }
                }
                None => {
                    self.map_readback_with(staging, "numeric download", |bytes| {
                        numeric_data_from_bytes(entry.element_type, bytes, entry.len)
                    })
                    .await?
                }
            }
        };
        let lane_factor = match entry.storage {
            GpuTensorStorage::Real => 1usize,
            GpuTensorStorage::ComplexInterleaved => 2usize,
        };
        let mut shape = runmat_value::exact_capacity_vec_from_slice(handle.shape())?;
        if let Some(info) = runmat_accelerate_api::handle_transpose_info(handle) {
            let logical_len = data.len() / lane_factor;
            ensure!(
                data.len() % lane_factor == 0
                    && info.base_rows.checked_mul(info.base_cols) == Some(logical_len),
                "numeric download: transpose metadata mismatch for buffer {}",
                handle.buffer_id()
            );
            if shape.len() == 2 {
                transpose_numeric_data(&mut data, lane_factor, info.base_rows, info.base_cols)?;
                shape[0] = info.base_cols;
                shape[1] = info.base_rows;
            }
        }
        let transferred_bytes = if entry.len == 0 {
            0
        } else {
            entry.allocated_bytes
        };
        self.telemetry.record_download_bytes(transferred_bytes);
        record_provider_copy(
            runmat_value::HostCopyReason::ProviderReadback,
            transferred_bytes,
        );
        let owned = HostNumericTensorOwned {
            data,
            shape,
            storage: entry.storage,
        };
        owned.validate()?;
        Ok(owned)
    }

    pub(crate) async fn download_exec(&self, handle: &GpuTensorHandle) -> Result<HostTensorOwned> {
        self.download_exec_observing(handle, None).await
    }

    pub(crate) async fn download_power_probe_observed_exec(
        &self,
        handle: &GpuTensorHandle,
    ) -> Result<(HostTensorOwned, power_probe::PowerProbeReadbackObserved)> {
        let mut observed = power_probe::PowerProbeReadbackObserved::default();
        let host = self
            .download_exec_observing(handle, Some(&mut observed))
            .await?;
        observed.require_scalar(handle.view_identity())?;
        Ok((host, observed))
    }

    async fn download_exec_observing(
        &self,
        handle: &GpuTensorHandle,
        mut probe_observed: Option<&mut power_probe::PowerProbeReadbackObserved>,
    ) -> Result<HostTensorOwned> {
        let span = info_span!(
            "gpu.transfer.download",
            shape = ?handle.shape(),
            buffer_id = handle.buffer_id()
        );
        let entry = {
            let _guard = span.enter();
            log::trace!(
                "wgpu download id={} shape={:?}",
                handle.buffer_id(),
                handle.shape()
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
                handle.buffer_id(),
                last
            );
        } else {
            log::trace!(
                "wgpu download id={} last_submission_id=<none>",
                handle.buffer_id()
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
            ensure!(
                probe_observed.is_none(),
                "POWER scalar readback cannot be empty"
            );
            return Ok(HostTensorOwned {
                data: Vec::new(),
                shape: handle.shape().to_vec(),
                storage,
            });
        }

        let size_bytes = entry.allocated_bytes;
        let finish_readback = |mut out: Vec<f64>| -> Result<HostTensorOwned> {
            log::trace!(
                "wgpu download copying data id={} len={} bytes={}",
                handle.buffer_id(),
                entry.len,
                size_bytes
            );

            log::trace!("wgpu download finished copy id={}", handle.buffer_id());
            self.telemetry.record_download_bytes(size_bytes);
            record_provider_copy(runmat_value::HostCopyReason::ProviderReadback, size_bytes);

            let lane_factor = match storage {
                GpuTensorStorage::Real => 1usize,
                GpuTensorStorage::ComplexInterleaved => 2usize,
            };
            let mut shape = handle.shape().to_vec();
            if let Some(info) = runmat_accelerate_api::handle_transpose_info(handle) {
                let base_rows = info.base_rows;
                let base_cols = info.base_cols;
                let logical_len = out.len() / lane_factor;
                if !out.len().is_multiple_of(lane_factor)
                    || base_rows.checked_mul(base_cols) != Some(logical_len)
                {
                    return Err(anyhow!(
                        "download: transpose metadata mismatch for buffer {}",
                        handle.buffer_id()
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
                handle.buffer_id(),
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
            handle.buffer_id(),
            size_bytes
        );
        let (staging, copy_submission_id, copy_fence_requested_bytes, copy_ticket) = self
            .with_emission(|emission, pending| -> Result<_> {
                let buffer = runmat_accelerate_api::WgpuBufferOwner::new(
                    emission.device().create_buffer(&wgpu::BufferDescriptor {
                        label: Some("runmat-download-staging"),
                        size: size_bytes,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
                );
                let resource_ticket = if probe_observed.is_some() {
                    runmat_accelerate_api::observation::current_provider_resource_scope()
                        .map(|scope| scope.issue_resource_ticket(buffer.size()))
                        .transpose()?
                } else {
                    None
                };
                let staging = ProbeStaging::new(buffer, resource_ticket);
                let mut encoder = pending.profiled_encoder(
                    emission,
                    &wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-download-encoder"),
                    },
                );
                encoder.copy_buffer_to_buffer(
                    entry.buffer.as_ref(),
                    0,
                    &staging.buffer,
                    0,
                    size_bytes,
                );
                let ticket = probe_observed
                    .is_some()
                    .then(|| pending.arm_probe_completion());
                let submission_id = self.submit(emission, pending, encoder)?;
                Ok((
                    staging,
                    submission_id,
                    pending.last_fence_requested_bytes(),
                    ticket,
                ))
            })??;
        if let Some(observed) = probe_observed.as_deref_mut() {
            observed.input_view = Some(handle.view_identity());
            observed.copied_bytes = Some(size_bytes);
            observed.staging_buffer_bytes = Some(staging.requested_bytes());
            observed.copy_submission_id = Some(copy_submission_id);
            observed.copy_fence_requested_bytes = copy_fence_requested_bytes;
            observed.copy_ticket = copy_ticket;
        }
        let mapped_bytes = std::sync::atomic::AtomicU64::new(0);
        let out = self
            .map_readback_with(staging, "download", |data| {
                mapped_bytes.store(data.len() as u64, std::sync::atomic::Ordering::Relaxed);
                let mut out = vec![0.0f64; entry.len];
                match entry.precision {
                    NumericPrecision::F64 => out.copy_from_slice(cast_slice(data)),
                    NumericPrecision::F32 => {
                        let f32_slice: &[f32] = cast_slice(data);
                        for (dst, src) in out.iter_mut().zip(f32_slice.iter()) {
                            *dst = *src as f64;
                        }
                    }
                }
                Ok(out)
            })
            .await?;
        if let Some(observed) = probe_observed {
            observed.mapped_bytes = Some(mapped_bytes.load(std::sync::atomic::Ordering::Relaxed));
        }
        finish_readback(out)
    }

    pub(crate) fn device_info_exec(&self) -> String {
        format!(
            "{} ({:?})",
            self.adapter_info.name, self.adapter_info.backend
        )
    }

    pub(crate) fn device_info_struct_exec(&self) -> ApiDeviceInfo {
        let backend = format!("{:?}", self.adapter_info.backend).to_ascii_lowercase();
        ApiDeviceInfo {
            device_id: self.runtime_device_id,
            name: self.adapter_info.name.clone(),
            vendor: canonical_vendor_name(&self.adapter_info),
            // This is a per-buffer allocation limit, not adapter total/free
            // memory. WGPU exposes no authoritative VRAM byte count here.
            memory_bytes: None,
            backend: Some(backend),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn exact_readback_survives_compute_denial_without_granting_emission() {
        use runmat_runtime::context::{
            RuntimeAccelerationPolicy, RuntimeContext, RuntimeProviderPreference,
        };
        // Install the real Runtime authorizer, never a test-private callback
        // into the process-wide first-install-wins authority slot.
        let runtime = RuntimeContext::new(std::rc::Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
        .with_acceleration_policy(RuntimeAccelerationPolicy {
            enabled: true,
            provider: RuntimeProviderPreference::Auto,
            automatic_offload: false,
            allow_host_fallback: false,
            calibrate: false,
        });
        // A policy transition is not a new session or device epoch. Retain
        // the exact context state that owns the source allocation.
        let readback_runtime =
            runtime
                .clone()
                .with_acceleration_policy(RuntimeAccelerationPolicy {
                    enabled: false,
                    ..runtime.acceleration_policy().unwrap()
                });
        assert_eq!(runtime.local_identity(), readback_runtime.local_identity());
        let provider = crate::backend::wgpu::provider::register_wgpu_provider(
            crate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .expect("actual WGPU adapter required for exact readback gate test");
        // The singleton is static and uniquely owns its registered device ID.
        unsafe { runmat_accelerate_api::register_device_provider(provider) };
        let input = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F32 => {
                HostNumericDataOwned::F32(vec![1.0, -2.0])
            }
            runmat_accelerate_api::ProviderPrecision::F64 => {
                HostNumericDataOwned::F64(vec![1.0, -2.0])
            }
        };
        let creation_scope = runtime.enter();
        let source = provider
            .upload_numeric(&HostNumericTensorView {
                data: input.as_view(),
                // Two component lanes are ONE logical complex element.
                // Exercise the same source-record proof as all complex paths.
                shape: &[1],
                storage: GpuTensorStorage::ComplexInterleaved,
            })
            .unwrap();
        drop(creation_scope);
        let scope = readback_runtime.enter();
        assert!(provider.access.with_shared(|_| ()).is_err());
        let readback = runmat_accelerate_api::numeric_readback_for_handle(&source).unwrap();
        let host = futures::executor::block_on(readback.download_numeric()).unwrap();
        assert_eq!(host.shape, [1]);
        assert_eq!(host.storage, GpuTensorStorage::ComplexInterleaved);
        match host.data {
            HostNumericDataOwned::F32(values) => assert_eq!(values, [1.0, -2.0]),
            HostNumericDataOwned::F64(values) => assert_eq!(values, [1.0, -2.0]),
            other => panic!("unexpected exact readback class {other:?}"),
        }
        assert!(provider.access.with_shared(|_| ()).is_err());
        use runmat_execution::identity::{AttemptId, ExecutionScopeId, PoolId, TaskId, WorkerId};
        let assignment = readback_runtime.enter_execution_assignment(Some(
            runmat_execution::ProgramExecutionAssignment {
                scope_id: ExecutionScopeId::derive(&[b"readback-denied-scope"]),
                pool_id: PoolId::derive(&[b"readback-denied-pool"]),
                task_id: TaskId::derive(&[b"readback-denied-task"]),
                attempt_id: AttemptId::derive(&[b"readback-denied-attempt"]),
                worker_id: WorkerId::derive(&[b"readback-denied-worker"]),
                backend: runmat_execution::PoolBackend::LocalProcesses,
                resources: runmat_execution::resource::ResourceAssignment {
                    accelerator_leases: vec![],
                },
            },
        ));
        assert!(futures::executor::block_on(readback.download_numeric()).is_err());
        drop(assignment);
        drop(scope);
        let _retirement_scope = runtime.enter();
        source.release().unwrap();
    }

    #[test]
    fn numeric_upload_encoding_uses_exact_borrowed_words() {
        let lanes = [u64::MAX, 9_007_199_254_740_993, 0, 1];
        let mut words = [0; 8];
        fill_integer_words(HostIntegerDataView::U64(&lanes), &mut words).unwrap();
        assert_eq!(words, [u32::MAX, u32::MAX, 1, 2_097_152, 0, 0, 1, 0]);
        assert_eq!(integer_encoding_words(IntegerElementType::U64, 4), Some(8));
        assert_eq!(
            integer_encoding_words(IntegerElementType::I64, usize::MAX),
            None
        );
        let mut invalid = [123; 7];
        assert!(fill_integer_words(HostIntegerDataView::U64(&lanes), &mut invalid).is_err());
        assert_eq!(invalid, [123; 7]);
        let mut narrow = [0; 2];
        fill_integer_words(HostIntegerDataView::I8(&[-1, i8::MIN]), &mut narrow).unwrap();
        assert_eq!(narrow, [u32::MAX, (-128i32) as u32]);
    }

    #[test]
    fn exact_download_extent_accepts_payload_or_backing_record_but_not_mismatch() {
        assert!(numeric_download_extents_match(0, 8, 0, 8));
        assert!(numeric_download_extents_match(8, 8, 0, 8));
        assert!(numeric_download_extents_match(24, 24, 24, 24));
        assert!(!numeric_download_extents_match(4, 8, 0, 8));
        assert!(!numeric_download_extents_match(0, 4, 0, 8));
        assert!(!numeric_download_extents_match(8, 8, 16, 8));
    }

    fn exact_empty_download_query(
        handle: &GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderFeasibilityQuery {
        use runmat_accelerate_api::{
            ProviderElementType, ProviderFeasibilityQuery, ProviderLayout, ProviderOperationFamily,
            ProviderOperationIdentity, ProviderRepresentation, ProviderResidency, ProviderStorage,
            ProviderWorkload,
        };
        let element_type = match handle.physical().element_type() {
            NumericElementType::F32 => ProviderElementType::F32,
            NumericElementType::F64 => ProviderElementType::F64,
            _ => panic!("floating empty fixture"),
        };
        let input = ProviderRepresentation {
            element_type,
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape: vec![1, 0],
            residency: ProviderResidency::Device,
        };
        let output = ProviderRepresentation {
            residency: ProviderResidency::Host,
            ..input.clone()
        };
        ProviderFeasibilityQuery {
            operation: ProviderOperationIdentity::new("transfer.download"),
            family: ProviderOperationFamily::Download,
            inputs: vec![input],
            outputs: vec![output],
            workload: ProviderWorkload {
                requested_outputs: Some(1),
                ..ProviderWorkload::default()
            },
        }
    }

    fn admit_empty_download_under_zero_staging_and_queue(
        provider: &WgpuProvider,
        handle: &GpuTensorHandle,
    ) {
        use runmat_execution::{
            CandidateCostEvidence, CandidateCostUnavailable, CandidateExecutionLocation,
            CandidateOutputPresentation, CandidateOutputResidency, CandidatePreparationState,
            CandidateResourceDemand, Digest, ExecutionCandidateKind, PlacementResourceSnapshot,
            PlacementRevision, PlacementSignature, ResourceByteBound, WholeWorkCandidate,
            WholeWorkConstraint, WholeWorkPlacementRequest,
        };
        let digest = Digest::sha256(b"actual-empty-wgpu-readback");
        // Zero payload staging/queue demand still owns the returned shape
        // metadata. Fund its exact canonical quote rather than zero host RAM.
        let host_workspace = provider.download_numeric_host_workspace_bound(handle);
        let ResourceByteBound::Known(host_bytes) = host_workspace else {
            panic!("empty download must have a known host workspace bound");
        };
        let candidate = WholeWorkCandidate {
            kind: ExecutionCandidateKind::SharedRuntime,
            execution_location: CandidateExecutionLocation::Host,
            preparation: CandidatePreparationState::Ready,
            output_residency: CandidateOutputResidency::Host,
            presentation: CandidateOutputPresentation::ProviderDownload {
                device_id: provider.device_id(),
                provider_scratch_bytes: provider.download_numeric_provider_scratch_bound(handle),
                queue_slots: provider.download_numeric_queue_slots(handle),
            },
            route_facts: digest,
            cost: CandidateCostEvidence::Unavailable(CandidateCostUnavailable::NoEstimate),
            resources: CandidateResourceDemand {
                cpu_millicores: 1_000,
                retained_bytes: ResourceByteBound::Known(0),
                scratch_bytes: host_workspace,
                provider_host_workspace: None,
                provider_input_stages: None,
                queue_slots: 0,
                peak_live: None,
            },
            transactional_results: true,
        };
        let mut owner = provider.placement_resources();
        owner.scratch_available_bytes = Some(0);
        owner.queue_depth = Some(0);
        owner.queue_limit = Some(0);
        assert_eq!(provider.download_numeric_queue_slots(handle), 0);
        let request = WholeWorkPlacementRequest {
            signature: PlacementSignature {
                region: None,
                operation: "empty_download".into(),
                runtime_facts: digest,
                provider_provenance: None,
                graph_output_authority: None,
                revision: PlacementRevision {
                    program: None,
                    catalog: digest,
                    compiler: digest,
                    authority: runmat_execution::PlacementExecutionAuthorityRevision::Provider(
                        digest,
                    ),
                    policy: digest,
                },
            },
            constraint: WholeWorkConstraint::AnyLegal,
            candidates: vec![candidate],
            resources: PlacementResourceSnapshot {
                cpu_millicores_available: 1_000,
                memory_available_bytes: Some(host_bytes),
                cancellation_requested: false,
                providers: vec![owner],
                epoch: 0,
            },
            deterministic: true,
            require_transactional_results: true,
        };
        crate::placement::PlacementSession::default()
            .select_whole_work(request)
            .unwrap_or_else(|error| panic!("empty download admission failed: {error:?}"));
    }

    #[test]
    #[ignore = "requires a live WGPU adapter; run explicitly for physical empty-upload qualification"]
    fn actual_empty_numeric_upload_retains_physical_buffer_but_transfers_no_payload() {
        let provider =
            WgpuProvider::new(WgpuProviderOptions::default()).expect("live WGPU adapter required");
        let before = provider.placement_resources().live_bytes;
        let f32_values: [f32; 0] = [];
        let f64_values: [f64; 0] = [];
        let data = match provider.precision() {
            ProviderPrecision::F32 => HostNumericDataView::F32(&f32_values),
            ProviderPrecision::F64 => HostNumericDataView::F64(&f64_values),
        };
        let handle = provider
            .upload_numeric_produced(
                &HostNumericTensorView {
                    data,
                    shape: &[1, 0],
                    storage: GpuTensorStorage::Real,
                },
                false,
            )
            .expect("empty numeric upload");
        let entry = provider.get_entry_raw(&handle).unwrap();
        let floor = provider.element_size as u64;
        assert_eq!(entry.buffer.size(), floor);
        assert_eq!(entry.allocated_bytes, floor);
        assert!(provider
            .numeric_download_feasibility_exec(&handle, &exact_empty_download_query(&handle))
            .is_supported());
        admit_empty_download_under_zero_staging_and_queue(&provider, &handle);
        assert_eq!(provider.placement_resources().live_bytes - before, floor);
        let downloaded = futures::executor::block_on(provider.download_numeric_exec(&handle))
            .expect("empty numeric readback");
        assert_eq!(downloaded.data.len(), 0);
        assert_eq!(provider.telemetry_snapshot().upload_bytes, 0);
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        let registered = provider
            .with_emission(|emission, _| {
                provider.create_storage_buffer(emission, 0, "registered-empty-readback")
            })
            .unwrap()
            .unwrap();
        let registered = provider
            .register_existing_buffer(registered, vec![1, 0], 0)
            .expect("empty generic registration");
        let registered_entry = provider.get_entry_raw(&registered).unwrap();
        assert_eq!(registered_entry.allocated_bytes, 0);
        assert_eq!(registered.physical().requested_bytes(), floor);
        assert!(provider
            .numeric_download_feasibility_exec(
                &registered,
                &exact_empty_download_query(&registered)
            )
            .is_supported());
        admit_empty_download_under_zero_staging_and_queue(&provider, &registered);
        assert_eq!(
            futures::executor::block_on(provider.download_numeric_exec(&registered))
                .unwrap()
                .data
                .len(),
            0
        );
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        registered.release().unwrap();
        handle.release().unwrap();
        assert_eq!(provider.placement_resources().live_bytes, before);
    }

    #[test]
    #[ignore = "requires provisioned WGPU adapter; run with --ignored"]
    fn readback_decode_error_and_unpolled_drop_preserve_source() {
        let provider =
            WgpuProvider::new(WgpuProviderOptions::default()).expect("required WGPU adapter");
        let handle = provider
            .upload_exec(&HostTensorView {
                data: &[1.25, -3.5],
                shape: &[2, 1],
            })
            .expect("upload");
        let staging = || {
            let entry = provider
                .get_entry_raw(&handle)
                .expect("source remains live");
            provider
                .with_emission(|emission, pending| -> Result<_> {
                    let buffer = runmat_accelerate_api::WgpuBufferOwner::new(
                        emission.device().create_buffer(&wgpu::BufferDescriptor {
                            label: Some("readback-cleanup-regression"),
                            size: entry.allocated_bytes,
                            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        }),
                    );
                    let mut encoder = pending.profiled_encoder(
                        emission,
                        &wgpu::CommandEncoderDescriptor { label: None },
                    );
                    encoder.copy_buffer_to_buffer(
                        &entry.buffer,
                        0,
                        &buffer,
                        0,
                        entry.allocated_bytes,
                    );
                    provider.submit(emission, pending, encoder)?;
                    Ok(buffer)
                })
                .expect("emission access")
                .expect("copy")
        };
        let error = futures::executor::block_on(provider.map_readback_with(
            staging(),
            "injected decode",
            |_| -> Result<()> { Err(anyhow!("decode sentinel")) },
        ))
        .expect_err("decode must remain fallible");
        assert_eq!(error.to_string(), "decode sentinel");
        let unpolled = provider.map_readback_with(staging(), "unpolled", |_| -> Result<()> {
            panic!("unpolled decoder must not execute")
        });
        drop(unpolled);
        #[cfg(panic = "unwind")]
        {
            let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                futures::executor::block_on(provider.map_readback_with(
                    staging(),
                    "decode unwind",
                    |_| -> Result<()> { panic!("decode unwind sentinel") },
                ))
            }));
            assert!(panicked.is_err());
        }
        let host = futures::executor::block_on(provider.download_exec(&handle))
            .expect("source survives both cleanup paths");
        assert_eq!(host.shape, vec![2, 1]);
        assert_eq!(host.data, vec![1.25, -3.5]);
        handle.release().expect("final owner release");
    }

    #[test]
    #[ignore = "requires provisioned WGPU adapter; run with --ignored"]
    fn numeric_readback_decodes_native_storage_without_byte_intermediate() {
        let provider =
            WgpuProvider::new(WgpuProviderOptions::default()).expect("required WGPU adapter");
        for data in [
            HostNumericDataOwned::F32(vec![1.25, -3.5]),
            HostNumericDataOwned::U64(vec![u64::MAX, (1u64 << 53) + 1]),
        ] {
            let host = HostNumericTensorOwned {
                data,
                shape: vec![2, 1, 1],
                storage: GpuTensorStorage::Real,
            };
            let input = provider
                .upload_numeric_exec(&host.as_view())
                .expect("native upload");
            let actual = futures::executor::block_on(provider.download_numeric_exec(&input))
                .expect("typed readback");
            assert_eq!(actual.shape, host.shape);
            match (actual.data, host.data) {
                (HostNumericDataOwned::F32(actual), HostNumericDataOwned::F32(expected)) => {
                    assert_eq!(actual, expected)
                }
                (HostNumericDataOwned::U64(actual), HostNumericDataOwned::U64(expected)) => {
                    assert_eq!(actual, expected)
                }
                other => panic!("native storage changed: {other:?}"),
            }
            input.release().expect("retire uploaded input");
        }
    }

    #[test]
    #[ignore = "requires provisioned WGPU adapter; run with --ignored"]
    fn operations_reject_authentic_handles_owned_by_another_device() {
        let provider =
            WgpuProvider::new(WgpuProviderOptions::default()).expect("required WGPU adapter");
        let other =
            WgpuProvider::new(WgpuProviderOptions::default()).expect("required WGPU adapter");
        let handle = provider
            .upload_exec(&HostTensorView {
                data: &[1.0, 2.0],
                shape: &[2, 1],
            })
            .expect("upload");
        let foreign = other
            .upload_exec(&HostTensorView {
                data: &[7.0],
                shape: &[1, 1],
            })
            .unwrap();
        assert!(provider.get_entry_raw(&foreign).is_err());
        foreign.release().unwrap();
        futures::executor::block_on(provider.download_exec(&handle))
            .expect("the locally owned buffer must remain live");
        handle.release().unwrap();
    }
}
