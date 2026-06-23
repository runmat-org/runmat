use anyhow::{anyhow, ensure, Result};
use bytemuck::{cast_slice, Pod, Zeroable};
use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage, HostTensorView};
use std::sync::Arc;
use wgpu::util::DeviceExt;

use super::backend_shared::gpu_per_buffer_limit_error;
use super::backend_types::WgpuProvider;
use crate::backend::wgpu::shaders::comms::{
    modulate_bits_constellation_shader, modulate_constellation_shader,
};
use crate::backend::wgpu::types::NumericPrecision;

const CPU_COMMS_FAST_PATH_MAX_LOGICAL_LEN: usize = 4096;

struct UploadedTemp<'a> {
    provider: &'a WgpuProvider,
    handle: GpuTensorHandle,
}

impl Drop for UploadedTemp<'_> {
    fn drop(&mut self) {
        self.provider.free_exec(&self.handle).ok();
    }
}

impl WgpuProvider {
    fn upload_complex_interleaved_exec(
        &self,
        data: &[f64],
        shape: Vec<usize>,
        label: &str,
    ) -> Result<GpuTensorHandle> {
        let len = data.len();
        let bytes = (len as u64).saturating_mul(self.element_size as u64);
        if bytes > self.adapter_limits.max_buffer_size {
            return Err(gpu_per_buffer_limit_error(
                label,
                bytes,
                self.adapter_limits.max_buffer_size,
            ));
        }

        let buffer = if data.is_empty() {
            self.create_storage_buffer(0, label)
        } else {
            match self.precision {
                NumericPrecision::F64 => Arc::new(self.device_ref().create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some(label),
                        contents: cast_slice(data),
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::COPY_SRC,
                    },
                )),
                NumericPrecision::F32 => {
                    let data_f32: Vec<f32> = data.iter().map(|value| *value as f32).collect();
                    Arc::new(self.device_ref().create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some(label),
                            contents: cast_slice(&data_f32),
                            usage: wgpu::BufferUsages::STORAGE
                                | wgpu::BufferUsages::COPY_DST
                                | wgpu::BufferUsages::COPY_SRC,
                        },
                    ))
                }
            }
        };
        self.telemetry.record_upload_bytes(bytes);
        Ok(self.register_existing_buffer_with_storage(
            buffer,
            shape,
            len,
            GpuTensorStorage::ComplexInterleaved,
        ))
    }

    pub(crate) async fn modulate_constellation_exec(
        &self,
        request: &runmat_accelerate_api::ProviderModulationRequest<'_>,
    ) -> Result<GpuTensorHandle> {
        ensure!(
            runmat_accelerate_api::handle_storage(request.input) == GpuTensorStorage::Real,
            "modulate_constellation requires a real-valued symbol input"
        );
        ensure!(
            !request.constellation.is_empty() && request.constellation.len().is_multiple_of(2),
            "modulate_constellation requires interleaved real/imag constellation pairs"
        );
        let order = request.constellation.len() / 2;
        ensure!(
            order <= u32::MAX as usize,
            "modulate_constellation: constellation too large"
        );
        let entry = self.get_entry(request.input)?;
        ensure!(
            entry.storage == GpuTensorStorage::Real && entry.precision == self.precision,
            "modulate_constellation: input storage does not match provider precision"
        );
        let logical_len = entry
            .shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| anyhow!("modulate_constellation: input shape overflow"))?;
        ensure!(
            logical_len == entry.len,
            "modulate_constellation: input data length does not match shape"
        );
        ensure!(
            logical_len <= u32::MAX as usize,
            "modulate_constellation: input too large"
        );
        let out_len = logical_len
            .checked_mul(2)
            .ok_or_else(|| anyhow!("modulate_constellation: output length overflow"))?;
        if out_len == 0 {
            let buffer = self.create_storage_buffer(0, "runmat-modulate-constellation-empty");
            let handle = self.register_existing_buffer_with_storage(
                buffer,
                entry.shape,
                0,
                GpuTensorStorage::ComplexInterleaved,
            );
            return Ok(handle);
        }

        if logical_len <= CPU_COMMS_FAST_PATH_MAX_LOGICAL_LEN
            && runmat_accelerate_api::handle_transpose_info(request.input).is_none()
            && std::env::var_os("RUNMAT_WGPU_COMMS_SHADER").is_none()
        {
            let host = self.download_exec(request.input).await?;
            ensure!(
                host.data.len() == logical_len,
                "modulate_constellation: input data length does not match shape"
            );
            let tolerance = match self.precision {
                NumericPrecision::F64 => 1.0e-9,
                NumericPrecision::F32 => 1.0e-5,
            };
            let mut out = Vec::with_capacity(out_len);
            for value in host.data {
                ensure!(
                    value.is_finite(),
                    "modulate_constellation: symbols must be finite integers"
                );
                let rounded = value.round();
                ensure!(
                    (value - rounded).abs() <= tolerance && rounded >= 0.0,
                    "modulate_constellation: symbols must be nonnegative integers"
                );
                let symbol = rounded as usize;
                ensure!(
                    symbol < order,
                    "modulate_constellation: symbols must be in range"
                );
                let point = symbol * 2;
                out.push(request.constellation[point]);
                out.push(request.constellation[point + 1]);
            }
            return self.upload_complex_interleaved_exec(
                &out,
                entry.shape.clone(),
                "runmat-modulate-constellation-cpu-fast-path",
            );
        }

        let table_shape = [request.constellation.len(), 1usize];
        let table = UploadedTemp {
            provider: self,
            handle: self.upload_exec(&HostTensorView {
                data: request.constellation,
                shape: &table_shape,
            })?,
        };
        let table_entry = self.get_entry(&table.handle)?;
        let output =
            self.create_storage_buffer_checked(out_len, "runmat-modulate-constellation")?;
        let error_words = [u32::MAX, 0, 0, 0];
        let error_bytes = cast_slice(&error_words);
        let error_buffer = Arc::new(self.device_ref().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("runmat-modulate-constellation-error"),
                contents: error_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            },
        ));

        let workgroup_size = self.workgroup_config.scalar;
        let shader = modulate_constellation_shader(self.precision, order, workgroup_size);
        let bgl = crate::backend::wgpu::bindings::build_bgl_for_layout_tag(
            self.device_ref(),
            "runmat-modulate-constellation-layout",
        )
        .ok_or_else(|| anyhow!("modulate_constellation: missing bind group layout"))?;
        let pipeline_layout = crate::backend::wgpu::pipelines::create_pipeline_layout(
            self.device_ref(),
            "runmat-modulate-constellation-pipeline-layout",
            &bgl,
        );
        let module = crate::backend::wgpu::pipelines::create_shader_module(
            self.device_ref(),
            "runmat-modulate-constellation-shader",
            &shader,
        );
        let shader_hash = self.compute_pipeline_hash_bytes(
            shader.as_bytes(),
            "runmat-modulate-constellation-layout",
            Some(workgroup_size),
        );
        let pipeline = self.get_or_create_pipeline(
            shader_hash,
            &pipeline_layout,
            &module,
            "runmat-modulate-constellation-pipeline",
            Some(shader.as_bytes()),
            Some("runmat-modulate-constellation-layout"),
            Some(workgroup_size),
        );

        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct Params {
            len: u32,
            _pad0: u32,
            _pad1: u32,
            _pad2: u32,
        }
        let params = Params {
            len: logical_len as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-modulate-constellation-params");
        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-modulate-constellation-bind-group"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: entry.buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: table_entry.buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: error_buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

        let error_size = (std::mem::size_of::<u32>() * 4) as u64;
        let staging = self.device_ref().create_buffer(&wgpu::BufferDescriptor {
            label: Some("runmat-modulate-constellation-error-staging"),
            size: error_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder =
            self.device_ref()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-modulate-constellation-encoder"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-modulate-constellation-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = crate::backend::wgpu::dispatch::common::dispatch_size(
                logical_len as u32,
                workgroup_size,
            );
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(error_buffer.as_ref(), 0, &staging, 0, error_size);
        self.submit(encoder);

        let bytes_result =
            self.map_readback_bytes_sync(staging, error_size, "modulate_constellation");
        let bytes = bytes_result?;
        let words: &[u32] = cast_slice(&bytes);
        let code = match words.first().copied().unwrap_or(u32::MAX) {
            u32::MAX => 0,
            packed => packed >> 30,
        };
        match code {
            0 => {}
            1 => {
                return Err(anyhow!(
                    "modulate_constellation: symbols must be finite integers"
                ))
            }
            2 => return Err(anyhow!("modulate_constellation: symbols must be in range")),
            3 => {
                return Err(anyhow!(
                    "modulate_constellation: symbols must be nonnegative integers"
                ));
            }
            code => {
                return Err(anyhow!(
                    "modulate_constellation: GPU validation error {code}"
                ))
            }
        }

        let handle = self.register_existing_buffer_with_storage(
            output,
            entry.shape,
            out_len,
            GpuTensorStorage::ComplexInterleaved,
        );
        Ok(handle)
    }

    pub(crate) async fn modulate_bits_constellation_exec(
        &self,
        request: &runmat_accelerate_api::ProviderBitModulationRequest<'_>,
    ) -> Result<GpuTensorHandle> {
        ensure!(
            runmat_accelerate_api::handle_storage(request.input) == GpuTensorStorage::Real,
            "modulate_bits_constellation requires a real-valued bit input"
        );
        ensure!(
            !request.constellation.is_empty() && request.constellation.len().is_multiple_of(2),
            "modulate_bits_constellation requires interleaved real/imag constellation pairs"
        );
        ensure!(
            request.input_rows > 0 && request.bits_per_symbol > 0,
            "modulate_bits_constellation: invalid bit grouping"
        );
        ensure!(
            request.input_rows.is_multiple_of(request.bits_per_symbol),
            "modulate_bits_constellation: bit rows must be a multiple of bits_per_symbol"
        );
        let order = request.constellation.len() / 2;
        ensure!(
            request.constellation.len() <= u32::MAX as usize && order <= u32::MAX as usize,
            "modulate_bits_constellation: constellation too large"
        );
        ensure!(
            request.input_rows <= u32::MAX as usize && request.bits_per_symbol <= u32::MAX as usize,
            "modulate_bits_constellation: input too large"
        );
        let entry = self.get_entry(request.input)?;
        ensure!(
            entry.storage == GpuTensorStorage::Real && entry.precision == self.precision,
            "modulate_bits_constellation: input storage does not match provider precision"
        );
        let logical_len = entry
            .shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| anyhow!("modulate_bits_constellation: input shape overflow"))?;
        ensure!(
            logical_len == entry.len,
            "modulate_bits_constellation: input data length does not match shape"
        );
        ensure!(
            entry.shape.first().copied() == Some(request.input_rows),
            "modulate_bits_constellation: input_rows must match the input leading dimension"
        );
        ensure!(
            logical_len <= u32::MAX as usize,
            "modulate_bits_constellation: input too large"
        );
        ensure!(
            logical_len.is_multiple_of(request.input_rows),
            "modulate_bits_constellation: bit input shape is inconsistent"
        );
        let output_rows = request.input_rows / request.bits_per_symbol;
        let channels = logical_len / request.input_rows;
        let output_logical_len = output_rows
            .checked_mul(channels)
            .ok_or_else(|| anyhow!("modulate_bits_constellation: output shape overflow"))?;
        ensure!(
            output_logical_len <= u32::MAX as usize,
            "modulate_bits_constellation: output too large"
        );
        let mut output_shape = entry.shape.clone();
        if output_shape.is_empty() {
            output_shape.push(output_rows);
            output_shape.push(1);
        } else {
            output_shape[0] = output_rows;
        }
        let out_len = output_logical_len
            .checked_mul(2)
            .ok_or_else(|| anyhow!("modulate_bits_constellation: output length overflow"))?;
        ensure!(
            out_len <= u32::MAX as usize,
            "modulate_bits_constellation: output too large"
        );
        if out_len == 0 {
            let buffer = self.create_storage_buffer(0, "runmat-modulate-bits-constellation-empty");
            let handle = self.register_existing_buffer_with_storage(
                buffer,
                output_shape,
                0,
                GpuTensorStorage::ComplexInterleaved,
            );
            return Ok(handle);
        }

        if logical_len <= CPU_COMMS_FAST_PATH_MAX_LOGICAL_LEN
            && runmat_accelerate_api::handle_transpose_info(request.input).is_none()
            && std::env::var_os("RUNMAT_WGPU_COMMS_SHADER").is_none()
        {
            let host = self.download_exec(request.input).await?;
            ensure!(
                host.data.len() == logical_len,
                "modulate_bits_constellation: input data length does not match shape"
            );
            let tolerance = match self.precision {
                NumericPrecision::F64 => 1.0e-9,
                NumericPrecision::F32 => 1.0e-6,
            };
            let mut out = Vec::with_capacity(out_len);
            for channel in 0..channels {
                let channel_offset = channel * request.input_rows;
                for group in 0..output_rows {
                    let mut symbol = 0usize;
                    for bit_idx in 0..request.bits_per_symbol {
                        let value =
                            host.data[channel_offset + group * request.bits_per_symbol + bit_idx];
                        ensure!(
                            value.is_finite(),
                            "modulate_bits_constellation: bits must be finite"
                        );
                        let rounded = value.round();
                        ensure!(
                            (value - rounded).abs() <= tolerance
                                && (rounded == 0.0 || rounded == 1.0),
                            "modulate_bits_constellation: bits must be 0 or 1"
                        );
                        symbol = (symbol << 1) | rounded as usize;
                    }
                    ensure!(
                        symbol < order,
                        "modulate_bits_constellation: symbols must be in range"
                    );
                    let point = symbol * 2;
                    out.push(request.constellation[point]);
                    out.push(request.constellation[point + 1]);
                }
            }
            return self.upload_complex_interleaved_exec(
                &out,
                output_shape.clone(),
                "runmat-modulate-bits-constellation-cpu-fast-path",
            );
        }

        let table_shape = [request.constellation.len(), 1usize];
        let table = UploadedTemp {
            provider: self,
            handle: self.upload_exec(&HostTensorView {
                data: request.constellation,
                shape: &table_shape,
            })?,
        };
        let table_entry = self.get_entry(&table.handle)?;
        let output =
            self.create_storage_buffer_checked(out_len, "runmat-modulate-bits-constellation")?;
        let error_words = [u32::MAX, 0, 0, 0];
        let error_bytes = cast_slice(&error_words);
        let error_buffer = Arc::new(self.device_ref().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("runmat-modulate-bits-constellation-error"),
                contents: error_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            },
        ));

        let workgroup_size = self.workgroup_config.scalar;
        let shader = modulate_bits_constellation_shader(self.precision, order, workgroup_size);
        let bgl = crate::backend::wgpu::bindings::build_bgl_for_layout_tag(
            self.device_ref(),
            "runmat-modulate-constellation-layout",
        )
        .ok_or_else(|| anyhow!("modulate_bits_constellation: missing bind group layout"))?;
        let pipeline_layout = crate::backend::wgpu::pipelines::create_pipeline_layout(
            self.device_ref(),
            "runmat-modulate-bits-constellation-pipeline-layout",
            &bgl,
        );
        let module = crate::backend::wgpu::pipelines::create_shader_module(
            self.device_ref(),
            "runmat-modulate-bits-constellation-shader",
            &shader,
        );
        let shader_hash = self.compute_pipeline_hash_bytes(
            shader.as_bytes(),
            "runmat-modulate-constellation-layout",
            Some(workgroup_size),
        );
        let pipeline = self.get_or_create_pipeline(
            shader_hash,
            &pipeline_layout,
            &module,
            "runmat-modulate-bits-constellation-pipeline",
            Some(shader.as_bytes()),
            Some("runmat-modulate-constellation-layout"),
            Some(workgroup_size),
        );

        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct Params {
            output_len: u32,
            input_rows: u32,
            output_rows: u32,
            bits_per_symbol: u32,
        }
        let params = Params {
            output_len: output_logical_len as u32,
            input_rows: request.input_rows as u32,
            output_rows: output_rows as u32,
            bits_per_symbol: request.bits_per_symbol as u32,
        };
        let params_buffer =
            self.uniform_buffer(&params, "runmat-modulate-bits-constellation-params");
        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-modulate-bits-constellation-bind-group"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: entry.buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: table_entry.buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: error_buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

        let error_size = (std::mem::size_of::<u32>() * 4) as u64;
        let staging = self.device_ref().create_buffer(&wgpu::BufferDescriptor {
            label: Some("runmat-modulate-bits-constellation-error-staging"),
            size: error_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder =
            self.device_ref()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-modulate-bits-constellation-encoder"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-modulate-bits-constellation-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = crate::backend::wgpu::dispatch::common::dispatch_size(
                output_logical_len as u32,
                workgroup_size,
            );
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(error_buffer.as_ref(), 0, &staging, 0, error_size);
        self.submit(encoder);

        let bytes_result =
            self.map_readback_bytes_sync(staging, error_size, "modulate_bits_constellation");
        let bytes = bytes_result?;
        let words: &[u32] = cast_slice(&bytes);
        let code = match words.first().copied().unwrap_or(u32::MAX) {
            u32::MAX => 0,
            packed => packed >> 30,
        };
        match code {
            0 => {}
            1 => return Err(anyhow!("modulate_bits_constellation: bits must be finite")),
            2 => return Err(anyhow!("modulate_bits_constellation: bits must be 0 or 1")),
            3 => {
                return Err(anyhow!(
                    "modulate_bits_constellation: symbols must be in range"
                ))
            }
            code => {
                return Err(anyhow!(
                    "modulate_bits_constellation: GPU validation error {code}"
                ))
            }
        }

        let handle = self.register_existing_buffer_with_storage(
            output,
            output_shape,
            out_len,
            GpuTensorStorage::ComplexInterleaved,
        );
        Ok(handle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::wgpu::provider::{register_wgpu_provider, WgpuProviderOptions};
    use runmat_accelerate_api::{AccelProvider, ProviderModulationRequest};

    #[test]
    fn modulate_constellation_provider_matches_table() {
        let Ok(provider) = register_wgpu_provider(WgpuProviderOptions::default()) else {
            return;
        };
        let symbols = [0.0, 1.0, 2.0, 3.0];
        let shape = [1usize, 4usize];
        let input = provider
            .upload(&HostTensorView {
                data: &symbols,
                shape: &shape,
            })
            .expect("upload symbols");
        let constellation = [1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0];
        let output =
            pollster::block_on(provider.modulate_constellation(ProviderModulationRequest {
                input: &input,
                constellation: &constellation,
            }))
            .expect("modulate");
        assert_eq!(
            runmat_accelerate_api::handle_storage(&output),
            GpuTensorStorage::ComplexInterleaved
        );
        let gathered = pollster::block_on(provider.download(&output)).expect("download output");
        assert_eq!(gathered.shape, shape);
        assert_eq!(gathered.data.len(), constellation.len());
        for (actual, expected) in gathered.data.iter().zip(constellation.iter()) {
            assert!((actual - expected).abs() < 1.0e-12);
        }
        provider.free(&input).ok();
        provider.free(&output).ok();
    }

    #[test]
    fn modulate_constellation_provider_rejects_invalid_symbols() {
        let Ok(provider) = register_wgpu_provider(WgpuProviderOptions::default()) else {
            return;
        };
        let symbols = [0.0, 1.01];
        let shape = [1usize, 2usize];
        let input = provider
            .upload(&HostTensorView {
                data: &symbols,
                shape: &shape,
            })
            .expect("upload symbols");
        let constellation = [1.0, 0.0, -1.0, 0.0];
        let err = pollster::block_on(provider.modulate_constellation(ProviderModulationRequest {
            input: &input,
            constellation: &constellation,
        }))
        .expect_err("invalid symbol should fail");
        assert!(err.to_string().contains("nonnegative integers"), "{err}");
        provider.free(&input).ok();
    }

    #[test]
    fn modulate_bits_constellation_provider_groups_bit_rows() {
        let Ok(provider) = register_wgpu_provider(WgpuProviderOptions::default()) else {
            return;
        };
        let bits = [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
        let shape = [2usize, 4usize];
        let input = provider
            .upload(&HostTensorView {
                data: &bits,
                shape: &shape,
            })
            .expect("upload bits");
        let constellation = [1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0];
        let output = pollster::block_on(provider.modulate_bits_constellation(
            runmat_accelerate_api::ProviderBitModulationRequest {
                input: &input,
                input_rows: 2,
                bits_per_symbol: 2,
                constellation: &constellation,
            },
        ))
        .expect("modulate bits");
        assert_eq!(
            runmat_accelerate_api::handle_storage(&output),
            GpuTensorStorage::ComplexInterleaved
        );
        let gathered = pollster::block_on(provider.download(&output)).expect("download output");
        assert_eq!(gathered.shape, vec![1, 4]);
        assert_eq!(gathered.data.len(), constellation.len());
        for (idx, (actual, expected)) in gathered.data.iter().zip(constellation.iter()).enumerate()
        {
            assert!(
                (actual - expected).abs() < 1.0e-6,
                "constellation lane {idx}: actual={actual} expected={expected}"
            );
        }
        provider.free(&input).ok();
        provider.free(&output).ok();
    }
}
