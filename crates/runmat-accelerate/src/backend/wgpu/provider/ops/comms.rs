use anyhow::{anyhow, ensure, Result};
use bytemuck::{cast_slice, Pod, Zeroable};
use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage, HostTensorView};

use super::backend_shared::gpu_per_buffer_limit_error;
use super::backend_types::WgpuProvider;
use crate::backend::wgpu::shaders::comms::{
    modulate_bits_constellation_shader, modulate_constellation_shader,
};
use crate::backend::wgpu::types::NumericPrecision;

const CPU_COMMS_FAST_PATH_MAX_LOGICAL_LEN: usize = 4096;

#[path = "comms/planned_input.rs"]
mod planned_input;
#[path = "comms/validated_route.rs"]
mod validated_route;
#[cfg(test)]
#[path = "comms/validated_tests.rs"]
mod validated_tests;

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

        let buffer = self.with_emission(|emission, _| -> Result<_> {
            let buffer = if data.is_empty() {
                self.create_storage_buffer(emission, 0, label)?
            } else {
                match self.precision {
                    NumericPrecision::F64 => runmat_accelerate_api::WgpuBufferOwner::new(
                        emission.create_initialized_buffer(&wgpu::util::BufferInitDescriptor {
                            label: Some(label),
                            contents: cast_slice(data),
                            usage: wgpu::BufferUsages::STORAGE
                                | wgpu::BufferUsages::COPY_DST
                                | wgpu::BufferUsages::COPY_SRC,
                        }),
                    ),
                    NumericPrecision::F32 => {
                        let data_f32: Vec<f32> = data.iter().map(|value| *value as f32).collect();
                        runmat_accelerate_api::WgpuBufferOwner::new(
                            emission.create_initialized_buffer(&wgpu::util::BufferInitDescriptor {
                                label: Some(label),
                                contents: cast_slice(&data_f32),
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
        self.register_existing_buffer_with_storage(
            buffer,
            shape,
            len,
            GpuTensorStorage::ComplexInterleaved,
        )
    }

    pub(crate) async fn modulate_constellation_exec(
        &self,
        request: &runmat_accelerate_api::ProviderModulationRequest<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let planned_input::SymbolPlan {
            entry,
            logical_len,
            out_len,
            order,
        } = planned_input::SymbolPlan::new(self, request)?;
        if out_len == 0 {
            return (|| -> Result<_> {
                let buffer = self.with_emission(|emission, _| {
                    self.create_storage_buffer(emission, 0, "runmat-modulate-constellation-empty")
                })??;
                let handle = self.register_existing_buffer_with_storage(
                    buffer,
                    entry.shape,
                    0,
                    GpuTensorStorage::ComplexInterleaved,
                )?;
                Ok(handle)
            })()
            .map_err(planned_input::unproven);
        }

        if logical_len <= CPU_COMMS_FAST_PATH_MAX_LOGICAL_LEN
            && runmat_accelerate_api::handle_transpose_info(request.input).is_none()
            && std::env::var_os("RUNMAT_WGPU_COMMS_SHADER").is_none()
        {
            let host = self
                .download_exec(request.input)
                .await
                .map_err(planned_input::unproven)?;
            return (|| -> Result<_> {
                ensure!(
                    host.data.len() == logical_len,
                    "modulate_constellation: input data length does not match shape"
                );
                let mut out = Vec::with_capacity(out_len);
                for value in host.data {
                    ensure!(
                        value.is_finite(),
                        "modulate_constellation: symbols must be finite integers"
                    );
                    ensure!(
                        value >= 0.0 && value.fract() == 0.0,
                        "modulate_constellation: symbols must be nonnegative integers"
                    );
                    let symbol = value as usize;
                    ensure!(
                        symbol < order,
                        "modulate_constellation: symbols must be in range"
                    );
                    let point = symbol * 2;
                    out.push(request.constellation[point]);
                    out.push(request.constellation[point + 1]);
                }
                self.upload_complex_interleaved_exec(
                    &out,
                    entry.shape.clone(),
                    "runmat-modulate-constellation-cpu-fast-path",
                )
            })()
            .map_err(planned_input::unproven);
        }

        self.require_ieee_words(self.precision)?;
        let workgroup_size = self.workgroup_config.scalar;
        let recipe = self.plan_modulation_pipeline(
            modulate_constellation_shader(self.precision, order, workgroup_size),
            validated_route::ModulationWork {
                input: &entry,
                table_words: request.constellation.len(),
                output_words: out_len,
                dispatch_len: logical_len,
                workgroup: workgroup_size,
            },
        )?;
        let validated = self
            .prepare_validated_modulation_pipeline(
                &recipe,
                "runmat-modulate-constellation-pipeline",
            )
            .await?;
        validated
            .require_modulation(&recipe)
            .map_err(planned_input::unproven)?;
        let live = self
            .get_entry(request.input)
            .map_err(planned_input::unproven)?;
        if !runmat_accelerate_api::WgpuBufferOwner::ptr_eq(&live.buffer, &entry.buffer) {
            return Err(planned_input::unproven(anyhow!(
                "modulate_constellation: input changed during preparation"
            )));
        }
        let table_shape = [request.constellation.len(), 1usize];
        let table = self
            .upload_exec(&HostTensorView {
                data: request.constellation,
                shape: &table_shape,
            })
            .map_err(planned_input::unproven)?;
        let table_entry = self.get_entry(&table).map_err(planned_input::unproven)?;
        let (output, staging) = self
            .with_emission(|emission, pending| -> Result<_> {
                let live = self.get_entry(request.input)?;
                ensure!(
                    runmat_accelerate_api::WgpuBufferOwner::ptr_eq(&live.buffer, &entry.buffer),
                    "modulate_constellation: input changed before emission"
                );
                let output = self.create_storage_buffer_checked(
                    emission,
                    out_len,
                    "runmat-modulate-constellation",
                )?;
                let error_words = [u32::MAX, 0, 0, 0];
                let error_bytes = cast_slice(&error_words);
                let error_buffer = runmat_accelerate_api::WgpuBufferOwner::new(
                    emission.create_initialized_buffer(&wgpu::util::BufferInitDescriptor {
                        label: Some("runmat-modulate-constellation-error"),
                        contents: error_bytes,
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                    }),
                );

                let bgl = validated.layout.clone();
                let pipeline = validated.pipeline.clone();

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
                let params_buffer =
                    self.uniform_buffer(emission, &params, "runmat-modulate-constellation-params")?;
                let bind_group = emission
                    .device()
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
                let staging = runmat_accelerate_api::WgpuBufferOwner::new(
                    emission.device().create_buffer(&wgpu::BufferDescriptor {
                        label: Some("runmat-modulate-constellation-error-staging"),
                        size: error_size,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
                );
                let mut encoder = pending.profiled_encoder(
                    emission,
                    &wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-modulate-constellation-encoder"),
                    },
                );
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
                self.submit(emission, pending, encoder)?;
                Ok((output, staging))
            })
            .map_err(planned_input::unproven)?
            .map_err(planned_input::unproven)?;

        let code = self
            .map_readback_with(
                staging,
                "modulate_constellation",
                validated_route::decode_error_code,
            )
            .await
            .map_err(planned_input::unproven)?;
        (|| -> Result<_> {
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
            )?;
            Ok(handle)
        })()
        .map_err(planned_input::unproven)
    }

    pub(crate) async fn modulate_bits_constellation_exec(
        &self,
        request: &runmat_accelerate_api::ProviderBitModulationRequest<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let planned_input::BitPlan {
            entry,
            logical_len,
            output_logical_len,
            output_shape,
            out_len,
            order,
            output_rows,
            channels,
        } = planned_input::BitPlan::new(self, request)?;
        validated_route::validate_bit_group_width(request.bits_per_symbol)?;
        if out_len == 0 {
            return (|| -> Result<_> {
                let buffer = self.with_emission(|emission, _| {
                    self.create_storage_buffer(
                        emission,
                        0,
                        "runmat-modulate-bits-constellation-empty",
                    )
                })??;
                let handle = self.register_existing_buffer_with_storage(
                    buffer,
                    output_shape,
                    0,
                    GpuTensorStorage::ComplexInterleaved,
                )?;
                Ok(handle)
            })()
            .map_err(planned_input::unproven);
        }

        if logical_len <= CPU_COMMS_FAST_PATH_MAX_LOGICAL_LEN
            && runmat_accelerate_api::handle_transpose_info(request.input).is_none()
            && std::env::var_os("RUNMAT_WGPU_COMMS_SHADER").is_none()
        {
            let host = self
                .download_exec(request.input)
                .await
                .map_err(planned_input::unproven)?;
            return (|| -> Result<_> {
                ensure!(
                    host.data.len() == logical_len,
                    "modulate_bits_constellation: input data length does not match shape"
                );
                let mut out = Vec::with_capacity(out_len);
                for channel in 0..channels {
                    let channel_offset = channel * request.input_rows;
                    for group in 0..output_rows {
                        let mut symbol = 0usize;
                        for bit_idx in 0..request.bits_per_symbol {
                            let value = host.data
                                [channel_offset + group * request.bits_per_symbol + bit_idx];
                            ensure!(
                                value.is_finite(),
                                "modulate_bits_constellation: bits must be finite"
                            );
                            ensure!(
                                value == 0.0 || value == 1.0,
                                "modulate_bits_constellation: bits must be 0 or 1"
                            );
                            symbol = (symbol << 1) | value as usize;
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
                self.upload_complex_interleaved_exec(
                    &out,
                    output_shape.clone(),
                    "runmat-modulate-bits-constellation-cpu-fast-path",
                )
            })()
            .map_err(planned_input::unproven);
        }

        self.require_ieee_words(self.precision)?;
        let workgroup_size = self.workgroup_config.scalar;
        let recipe = self.plan_modulation_pipeline(
            modulate_bits_constellation_shader(self.precision, order, workgroup_size),
            validated_route::ModulationWork {
                input: &entry,
                table_words: request.constellation.len(),
                output_words: out_len,
                dispatch_len: output_logical_len,
                workgroup: workgroup_size,
            },
        )?;
        let validated = self
            .prepare_validated_modulation_pipeline(
                &recipe,
                "runmat-modulate-bits-constellation-pipeline",
            )
            .await?;
        validated
            .require_modulation(&recipe)
            .map_err(planned_input::unproven)?;
        let live = self
            .get_entry(request.input)
            .map_err(planned_input::unproven)?;
        if !runmat_accelerate_api::WgpuBufferOwner::ptr_eq(&live.buffer, &entry.buffer) {
            return Err(planned_input::unproven(anyhow!(
                "modulate_bits_constellation: input changed during preparation"
            )));
        }
        let table_shape = [request.constellation.len(), 1usize];
        let table = self
            .upload_exec(&HostTensorView {
                data: request.constellation,
                shape: &table_shape,
            })
            .map_err(planned_input::unproven)?;
        let table_entry = self.get_entry(&table).map_err(planned_input::unproven)?;
        let (output, staging) = self
            .with_emission(|emission, pending| -> Result<_> {
                let live = self.get_entry(request.input)?;
                ensure!(
                    runmat_accelerate_api::WgpuBufferOwner::ptr_eq(&live.buffer, &entry.buffer),
                    "modulate_bits_constellation: input changed before emission"
                );
                let output = self.create_storage_buffer_checked(
                    emission,
                    out_len,
                    "runmat-modulate-bits-constellation",
                )?;
                let error_words = [u32::MAX, 0, 0, 0];
                let error_bytes = cast_slice(&error_words);
                let error_buffer = runmat_accelerate_api::WgpuBufferOwner::new(
                    emission.create_initialized_buffer(&wgpu::util::BufferInitDescriptor {
                        label: Some("runmat-modulate-bits-constellation-error"),
                        contents: error_bytes,
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                    }),
                );

                let bgl = validated.layout.clone();
                let pipeline = validated.pipeline.clone();

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
                let params_buffer = self.uniform_buffer(
                    emission,
                    &params,
                    "runmat-modulate-bits-constellation-params",
                )?;
                let bind_group = emission
                    .device()
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
                let staging = runmat_accelerate_api::WgpuBufferOwner::new(
                    emission.device().create_buffer(&wgpu::BufferDescriptor {
                        label: Some("runmat-modulate-bits-constellation-error-staging"),
                        size: error_size,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
                );
                let mut encoder = pending.profiled_encoder(
                    emission,
                    &wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-modulate-bits-constellation-encoder"),
                    },
                );
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
                self.submit(emission, pending, encoder)?;
                Ok((output, staging))
            })
            .map_err(planned_input::unproven)?
            .map_err(planned_input::unproven)?;

        let code = self
            .map_readback_with(
                staging,
                "modulate_bits_constellation",
                validated_route::decode_error_code,
            )
            .await
            .map_err(planned_input::unproven)?;
        (|| -> Result<_> {
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
            )?;
            Ok(handle)
        })()
        .map_err(planned_input::unproven)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::wgpu::provider::{register_wgpu_provider, WgpuProviderOptions};
    use runmat_accelerate_api::{AccelProvider, ProviderModulationRequest, ProviderPrecision};

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
        input.release().unwrap();
        output.release().unwrap();
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
        input.release().unwrap();
    }

    #[test]
    fn modulation_provider_rejects_representable_near_integer_values() {
        let Ok(provider) = register_wgpu_provider(WgpuProviderOptions::default()) else {
            return;
        };
        let near_one = match provider.precision() {
            ProviderPrecision::F32 => 1.0 + 2.0f64.powi(-20),
            ProviderPrecision::F64 => 1.0 + 2.0f64.powi(-34),
        };
        let constellation = [1.0, 0.0, -1.0, 0.0];
        let input = provider
            .upload(&HostTensorView {
                data: &[near_one],
                shape: &[1, 1],
            })
            .expect("upload near-integer input");
        let symbol_error =
            pollster::block_on(provider.modulate_constellation(ProviderModulationRequest {
                input: &input,
                constellation: &constellation,
            }))
            .expect_err("fractional symbol must not round to one");
        assert!(symbol_error.to_string().contains("nonnegative integers"));

        let bit_error = pollster::block_on(provider.modulate_bits_constellation(
            runmat_accelerate_api::ProviderBitModulationRequest {
                input: &input,
                input_rows: 1,
                bits_per_symbol: 1,
                constellation: &constellation,
            },
        ))
        .expect_err("fractional bit must not round to one");
        assert!(bit_error.to_string().contains("bits must be 0 or 1"));
        input.release().unwrap();
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
        input.release().unwrap();
        output.release().unwrap();
    }
}
