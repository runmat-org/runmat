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
        let (logical_u32, total_u32) = window_lengths(len, periodic)?;
        let shape_vec = vec![len, 1];
        if len == 1 {
            let one = [1.0f64];
            return self.upload(&HostTensorView {
                data: &one,
                shape: &[1, 1],
            });
        }
        let out_buffer = self.create_storage_buffer_checked(len, "runmat-window-out")?;
        if len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, shape_vec, 0));
        }

        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;
        let mut offset = 0usize;
        while offset < len {
            let chunk_len = (len - offset).min(chunk_capacity).max(1);
            let params = crate::backend::wgpu::params::WindowParams {
                len: logical_u32,
                total: total_u32,
                chunk: chunk_len as u32,
                offset: offset as u32,
                kind: kind as u32,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };
            let params_buffer = self.uniform_buffer(&params, "runmat-window-params");

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

fn window_lengths(len: usize, periodic: bool) -> Result<(u32, u32)> {
    if len > u32::MAX as usize || (periodic && len == u32::MAX as usize) {
        return Err(anyhow!("window: length exceeds GPU limits"));
    }
    let logical_u32 = len as u32;
    let total_u32 = if periodic {
        logical_u32 + 1
    } else {
        logical_u32
    };
    Ok((logical_u32, total_u32))
}

#[cfg(test)]
mod tests {
    use super::window_lengths;

    #[test]
    fn periodic_window_rejects_u32_max_len() {
        let err = window_lengths(u32::MAX as usize, true).expect_err("expected overflow guard");
        assert!(err.to_string().contains("length exceeds GPU limits"));
    }

    #[test]
    fn symmetric_window_allows_u32_max_len() {
        let (logical, total) = window_lengths(u32::MAX as usize, false).expect("symmetric max len");
        assert_eq!(logical, u32::MAX);
        assert_eq!(total, u32::MAX);
    }
}
