use anyhow::{anyhow, ensure, Result};
use bytemuck::{cast_slice, Pod, Zeroable};
use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage, HostTensorView};
use std::sync::Arc;
use wgpu::util::DeviceExt;

use super::backend_types::WgpuProvider;
use crate::backend::wgpu::shaders::comms::modulate_constellation_shader;

impl WgpuProvider {
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

        let table_shape = [request.constellation.len(), 1usize];
        let table = self.upload_exec(&HostTensorView {
            data: request.constellation,
            shape: &table_shape,
        })?;
        let table_entry = self.get_entry(&table)?;
        let output =
            self.create_storage_buffer_checked(out_len, "runmat-modulate-constellation")?;
        let error_bytes = [0u8; std::mem::size_of::<u32>() * 4];
        let error_buffer = Arc::new(self.device_ref().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("runmat-modulate-constellation-error"),
                contents: &error_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            },
        ));

        let workgroup_size = crate::backend::wgpu::config::WORKGROUP_SIZE;
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
        self.free_exec(&table).ok();
        let bytes = bytes_result?;
        let words: &[u32] = cast_slice(&bytes);
        match words.first().copied().unwrap_or(0) {
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
        let symbols = [0.0, 1.000001];
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
}
