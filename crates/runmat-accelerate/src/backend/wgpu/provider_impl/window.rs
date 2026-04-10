use super::*;

#[derive(Clone, Copy)]
pub(crate) enum WindowKind {
    Hann = 0,
    Hamming = 1,
    Blackman = 2,
}

impl WgpuProvider {
    pub(crate) fn window_exec(
        &self,
        kind: WindowKind,
        len: usize,
        periodic: bool,
    ) -> Result<GpuTensorHandle> {
        if len > u32::MAX as usize {
            return Err(anyhow!("window: length exceeds GPU limits"));
        }
        let shape_vec = vec![len, 1];
        let out_buffer = self.create_storage_buffer_checked(len, "runmat-window-out")?;
        if len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, shape_vec, 0));
        }

        let logical_u32 = len as u32;
        let total_u32 = if periodic {
            logical_u32 + 1
        } else {
            logical_u32
        };
        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;
        let mut offset = 0usize;
        while offset < len {
            let chunk_len = (len - offset).min(chunk_capacity).max(1);
            let params_buffer = match self.precision {
                NumericPrecision::F64 => {
                    let params = crate::backend::wgpu::params::WindowParamsF64 {
                        len: logical_u32,
                        total: total_u32,
                        chunk: chunk_len as u32,
                        offset: offset as u32,
                        kind: kind as u32,
                        _pad0: 0,
                        _pad1: 0,
                        _pad2: 0,
                    };
                    self.uniform_buffer(&params, "runmat-window-params-f64")
                }
                NumericPrecision::F32 => {
                    let params = crate::backend::wgpu::params::WindowParamsF32 {
                        len: logical_u32,
                        total: total_u32,
                        chunk: chunk_len as u32,
                        offset: offset as u32,
                        kind: kind as u32,
                        _pad0: 0,
                        _pad1: 0,
                        _pad2: 0,
                    };
                    self.uniform_buffer(&params, "runmat-window-params-f32")
                }
            };

            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-window-bind"),
                    layout: &self.pipelines.window.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: out_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });

            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_len as u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::creation::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.window.pipeline,
                &bind_group,
                workgroups,
                "runmat-window-encoder",
                "runmat-window-pass",
            );
            offset += chunk_len;
        }

        Ok(self.register_existing_buffer(out_buffer, shape_vec, len))
    }
}
