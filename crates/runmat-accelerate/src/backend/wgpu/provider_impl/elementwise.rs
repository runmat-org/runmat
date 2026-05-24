use super::*;

impl WgpuProvider {
    pub(crate) fn elem_eq_exec(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        let entry_a = self.get_entry(a)?;
        let entry_b = self.get_entry(b)?;
        if entry_a.shape != entry_b.shape {
            return Err(anyhow!("elem_eq: shape mismatch between inputs"));
        }
        let len = entry_a.len;
        let handle = if len == 0 {
            let out = self.create_storage_buffer(0, "runmat-elem-eq-empty");
            self.register_existing_buffer(out, entry_a.shape, 0)
        } else {
            let shader = match self.precision {
                NumericPrecision::F64 => ELEM_EQ_SHADER_F64,
                NumericPrecision::F32 => ELEM_EQ_SHADER_F32,
            };
            self.fused_elementwise(shader, &[a.clone(), b.clone()], &entry_a.shape, len)?
        };
        runmat_accelerate_api::set_handle_logical(&handle, true);
        Ok(handle)
    }

    pub(crate) fn elem_ne_exec(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        let entry_a = self.get_entry(a)?;
        let entry_b = self.get_entry(b)?;
        if entry_a.shape != entry_b.shape {
            return Err(anyhow!("elem_ne: shape mismatch between inputs"));
        }
        let len = entry_a.len;
        let handle = if len == 0 {
            let out = self.create_storage_buffer(0, "runmat-elem-ne-empty");
            self.register_existing_buffer(out, entry_a.shape, 0)
        } else {
            let shader = match self.precision {
                NumericPrecision::F64 => ELEM_NE_SHADER_F64,
                NumericPrecision::F32 => ELEM_NE_SHADER_F32,
            };
            self.fused_elementwise(shader, &[a.clone(), b.clone()], &entry_a.shape, len)?
        };
        runmat_accelerate_api::set_handle_logical(&handle, true);
        Ok(handle)
    }
    pub(crate) fn elem_lt_exec(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        let entry_a = self.get_entry(a)?;
        let entry_b = self.get_entry(b)?;
        if entry_a.shape != entry_b.shape {
            return Err(anyhow!("elem_lt: shape mismatch between inputs"));
        }
        let len = entry_a.len;
        let handle = if len == 0 {
            let out = self.create_storage_buffer(0, "runmat-elem-lt-empty");
            self.register_existing_buffer(out, entry_a.shape, 0)
        } else {
            let shader = match self.precision {
                NumericPrecision::F64 => ELEM_LT_SHADER_F64,
                NumericPrecision::F32 => ELEM_LT_SHADER_F32,
            };
            self.fused_elementwise(shader, &[a.clone(), b.clone()], &entry_a.shape, len)?
        };
        runmat_accelerate_api::set_handle_logical(&handle, true);
        Ok(handle)
    }

    pub(crate) fn elem_le_exec(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        let entry_a = self.get_entry(a)?;
        let entry_b = self.get_entry(b)?;
        if entry_a.shape != entry_b.shape {
            return Err(anyhow!("elem_le: shape mismatch between inputs"));
        }
        let len = entry_a.len;
        let handle = if len == 0 {
            let out = self.create_storage_buffer(0, "runmat-elem-le-empty");
            self.register_existing_buffer(out, entry_a.shape, 0)
        } else {
            let shader = match self.precision {
                NumericPrecision::F64 => ELEM_LE_SHADER_F64,
                NumericPrecision::F32 => ELEM_LE_SHADER_F32,
            };
            self.fused_elementwise(shader, &[a.clone(), b.clone()], &entry_a.shape, len)?
        };
        runmat_accelerate_api::set_handle_logical(&handle, true);
        Ok(handle)
    }

    pub(crate) fn elem_gt_exec(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        let entry_a = self.get_entry(a)?;
        let entry_b = self.get_entry(b)?;
        if entry_a.shape != entry_b.shape {
            return Err(anyhow!("elem_gt: shape mismatch between inputs"));
        }
        let len = entry_a.len;
        let handle = if len == 0 {
            let out = self.create_storage_buffer(0, "runmat-elem-gt-empty");
            self.register_existing_buffer(out, entry_a.shape, 0)
        } else {
            let shader = match self.precision {
                NumericPrecision::F64 => ELEM_GT_SHADER_F64,
                NumericPrecision::F32 => ELEM_GT_SHADER_F32,
            };
            self.fused_elementwise(shader, &[a.clone(), b.clone()], &entry_a.shape, len)?
        };
        runmat_accelerate_api::set_handle_logical(&handle, true);
        Ok(handle)
    }

    pub(crate) fn elem_ge_exec(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        let entry_a = self.get_entry(a)?;
        let entry_b = self.get_entry(b)?;
        if entry_a.shape != entry_b.shape {
            return Err(anyhow!("elem_ge: shape mismatch between inputs"));
        }
        let len = entry_a.len;
        let handle = if len == 0 {
            let out = self.create_storage_buffer(0, "runmat-elem-ge-empty");
            self.register_existing_buffer(out, entry_a.shape, 0)
        } else {
            let shader = match self.precision {
                NumericPrecision::F64 => ELEM_GE_SHADER_F64,
                NumericPrecision::F32 => ELEM_GE_SHADER_F32,
            };
            self.fused_elementwise(shader, &[a.clone(), b.clone()], &entry_a.shape, len)?
        };
        runmat_accelerate_api::set_handle_logical(&handle, true);
        Ok(handle)
    }

    pub(crate) fn logical_and_exec(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        let entry_a = self.get_entry(a)?;
        let entry_b = self.get_entry(b)?;
        if entry_a.shape != entry_b.shape {
            return Err(anyhow!("logical_and: shape mismatch between inputs"));
        }
        let len = entry_a.len;
        let handle = if len == 0 {
            let out = self.create_storage_buffer(0, "runmat-logical-and-empty");
            self.register_existing_buffer(out, entry_a.shape, 0)
        } else {
            let shader = match self.precision {
                NumericPrecision::F64 => LOGICAL_AND_SHADER_F64,
                NumericPrecision::F32 => LOGICAL_AND_SHADER_F32,
            };
            self.fused_elementwise(shader, &[a.clone(), b.clone()], &entry_a.shape, len)?
        };
        runmat_accelerate_api::set_handle_logical(&handle, true);
        Ok(handle)
    }

    pub(crate) fn logical_or_exec(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        let entry_a = self.get_entry(a)?;
        let entry_b = self.get_entry(b)?;
        if entry_a.shape != entry_b.shape {
            return Err(anyhow!("logical_or: shape mismatch between inputs"));
        }
        let len = entry_a.len;
        let handle = if len == 0 {
            let out = self.create_storage_buffer(0, "runmat-logical-or-empty");
            self.register_existing_buffer(out, entry_a.shape, 0)
        } else {
            let shader = match self.precision {
                NumericPrecision::F64 => LOGICAL_OR_SHADER_F64,
                NumericPrecision::F32 => LOGICAL_OR_SHADER_F32,
            };
            self.fused_elementwise(shader, &[a.clone(), b.clone()], &entry_a.shape, len)?
        };
        runmat_accelerate_api::set_handle_logical(&handle, true);
        Ok(handle)
    }
    pub(crate) fn logical_xor_exec(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        let entry_a = self.get_entry(a)?;
        let entry_b = self.get_entry(b)?;
        if entry_a.shape != entry_b.shape {
            return Err(anyhow!("logical_xor: shape mismatch between inputs"));
        }
        let len = entry_a.len;
        let handle = if len == 0 {
            let out = self.create_storage_buffer(0, "runmat-logical-xor-empty");
            self.register_existing_buffer(out, entry_a.shape, 0)
        } else {
            let shader = match self.precision {
                NumericPrecision::F64 => LOGICAL_XOR_SHADER_F64,
                NumericPrecision::F32 => LOGICAL_XOR_SHADER_F32,
            };
            self.fused_elementwise(shader, &[a.clone(), b.clone()], &entry_a.shape, len)?
        };
        runmat_accelerate_api::set_handle_logical(&handle, true);
        Ok(handle)
    }

    pub(crate) fn logical_not_exec(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        let len = entry.len;
        let handle = if len == 0 {
            let out = self.create_storage_buffer(0, "runmat-logical-not-empty");
            self.register_existing_buffer(out, entry.shape, 0)
        } else {
            let shader = match self.precision {
                NumericPrecision::F64 => LOGICAL_NOT_SHADER_F64,
                NumericPrecision::F32 => LOGICAL_NOT_SHADER_F32,
            };
            self.fused_elementwise(shader, std::slice::from_ref(a), &entry.shape, len)?
        };
        runmat_accelerate_api::set_handle_logical(&handle, true);
        Ok(handle)
    }

    pub(crate) fn logical_isfinite_exec(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        let len = entry.len;
        let handle = if len == 0 {
            let out = self.create_storage_buffer(0, "runmat-logical-isfinite-empty");
            self.register_existing_buffer(out, entry.shape, 0)
        } else {
            let shader = match self.precision {
                NumericPrecision::F64 => LOGICAL_ISFINITE_SHADER_F64,
                NumericPrecision::F32 => LOGICAL_ISFINITE_SHADER_F32,
            };
            self.fused_elementwise(shader, std::slice::from_ref(a), &entry.shape, len)?
        };
        runmat_accelerate_api::set_handle_logical(&handle, true);
        Ok(handle)
    }

    pub(crate) fn logical_isnan_exec(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        let len = entry.len;
        let handle = if len == 0 {
            let out = self.create_storage_buffer(0, "runmat-logical-isnan-empty");
            self.register_existing_buffer(out, entry.shape, 0)
        } else {
            let shader = match self.precision {
                NumericPrecision::F64 => LOGICAL_ISNAN_SHADER_F64,
                NumericPrecision::F32 => LOGICAL_ISNAN_SHADER_F32,
            };
            self.fused_elementwise(shader, std::slice::from_ref(a), &entry.shape, len)?
        };
        runmat_accelerate_api::set_handle_logical(&handle, true);
        Ok(handle)
    }

    pub(crate) fn logical_isinf_exec(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        let len = entry.len;
        let handle = if len == 0 {
            let out = self.create_storage_buffer(0, "runmat-logical-isinf-empty");
            self.register_existing_buffer(out, entry.shape, 0)
        } else {
            let shader = match self.precision {
                NumericPrecision::F64 => LOGICAL_ISINF_SHADER_F64,
                NumericPrecision::F32 => LOGICAL_ISINF_SHADER_F32,
            };
            self.fused_elementwise(shader, std::slice::from_ref(a), &entry.shape, len)?
        };
        runmat_accelerate_api::set_handle_logical(&handle, true);
        Ok(handle)
    }
    pub(crate) fn unary_op_exec(
        &self,
        op: crate::backend::wgpu::types::UnaryOpCode,
        a: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        if std::env::var("RUNMAT_DISABLE_UNARY").is_ok() {
            return Err(anyhow!("unary ops disabled via RUNMAT_DISABLE_UNARY"));
        }
        let entry_a = self.get_entry(a)?;
        let len = entry_a.len;
        let out_buffer = self.create_storage_buffer_checked(len, "runmat-unary-out")?;
        if len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, entry_a.shape, entry_a.len));
        }
        if len > (u32::MAX as usize) {
            return Err(gpu_dispatch_length_limit_error("unary_op", len));
        }
        let start = Instant::now();
        {
            let mut enc =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-unary-noop"),
                    });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-unary-noop-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.unary.pipeline);
            drop(pass);
            self.submit(enc);
        }
        self.device_ref().poll(wgpu::Maintain::Poll);
        {
            let enc = self
                .device_ref()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-unary-flush-gap"),
                });
            self.submit(enc);
        }
        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;
        let mut offset = 0usize;
        while offset < len {
            let remaining = len - offset;
            let chunk_len = remaining.min(chunk_capacity);
            let params = crate::backend::wgpu::params::LenOpParams {
                len: chunk_len as u32,
                op: op as u32,
                offset: offset as u32,
                total: len as u32,
            };
            let params_buffer = self.kernel_resources.uniform_buffer(
                self.device_ref(),
                UniformBufferKey::LenOpParams,
                std::mem::size_of::<crate::backend::wgpu::params::LenOpParams>() as u64,
                "runmat-unary-params",
            );
            self.queue
                .write_buffer(params_buffer.as_ref(), 0, bytes_of(&params));
            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-unary-bind"),
                    layout: &self.pipelines.unary.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry_a.buffer.as_ref().as_entire_binding(),
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
            let groups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_len as u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::elementwise::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.unary.pipeline,
                &bind_group,
                groups,
            );
            offset += chunk_len;
        }
        let handle = self.register_existing_buffer(out_buffer, entry_a.shape, len);
        self.telemetry
            .record_fused_elementwise_duration(start.elapsed());
        Ok(handle)
    }
    pub(crate) fn scalar_op_exec(
        &self,
        op: crate::backend::wgpu::types::ScalarOpCode,
        a: &GpuTensorHandle,
        scalar: f64,
    ) -> Result<GpuTensorHandle> {
        let entry_a = self.get_entry(a)?;
        let len = entry_a.len;
        let out_buffer = self.create_storage_buffer_checked(len, "runmat-scalar-out")?;
        if len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, entry_a.shape, entry_a.len));
        }
        if len > (u32::MAX as usize) {
            return Err(gpu_dispatch_length_limit_error("scalar_op", len));
        }
        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;
        let mut offset = 0usize;
        let start = Instant::now();
        while offset < len {
            let remaining = len - offset;
            let chunk_len = remaining.min(chunk_capacity);
            let params_buffer = match self.precision() {
                runmat_accelerate_api::ProviderPrecision::F64 => {
                    let params = crate::backend::wgpu::params::ScalarParamsF64 {
                        len: chunk_len as u32,
                        op: op as u32,
                        offset: offset as u32,
                        total: len as u32,
                        scalar,
                        _pad_scalar: 0.0,
                        _pad_tail: 0.0,
                        _pad_tail2: 0.0,
                        _pad_tail3: 0.0,
                        _pad_tail4: 0.0,
                    };
                    let buf = self.kernel_resources.uniform_buffer(
                        self.device_ref(),
                        UniformBufferKey::ScalarParamsF64,
                        std::mem::size_of::<crate::backend::wgpu::params::ScalarParamsF64>() as u64,
                        "runmat-scalar-params-f64",
                    );
                    self.queue.write_buffer(buf.as_ref(), 0, bytes_of(&params));
                    buf
                }
                _ => {
                    let params = crate::backend::wgpu::params::ScalarParamsF32 {
                        len: chunk_len as u32,
                        op: op as u32,
                        offset: offset as u32,
                        total: len as u32,
                        scalar: scalar as f32,
                        _pad_scalar: [0.0; 3],
                        _pad_tail: [0.0; 4],
                        _pad_tail2: [0.0; 4],
                    };
                    let buf = self.kernel_resources.uniform_buffer(
                        self.device_ref(),
                        UniformBufferKey::ScalarParamsF32,
                        std::mem::size_of::<crate::backend::wgpu::params::ScalarParamsF32>() as u64,
                        "runmat-scalar-params-f32",
                    );
                    self.queue.write_buffer(buf.as_ref(), 0, bytes_of(&params));
                    buf
                }
            };
            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-scalar-bind"),
                    layout: &self.pipelines.scalar.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry_a.buffer.as_ref().as_entire_binding(),
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
            let groups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_len as u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::elementwise::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.scalar.pipeline,
                &bind_group,
                groups,
            );
            offset += chunk_len;
        }
        let handle = self.register_existing_buffer(out_buffer, entry_a.shape, len);
        self.telemetry
            .record_fused_elementwise_duration(start.elapsed());
        Ok(handle)
    }

    pub(crate) fn binary_op_exec(
        &self,
        op: crate::backend::wgpu::types::BinaryOpCode,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        if std::env::var("RUNMAT_DISABLE_BINARY").is_ok() {
            return Err(anyhow!("binary ops disabled via RUNMAT_DISABLE_BINARY"));
        }
        let entry_a = self.get_entry(a)?;
        let entry_b = self.get_entry(b)?;
        if entry_a.shape != entry_b.shape {
            // Attempt general N-D broadcasted binary op
            return self.binary_op_broadcast_exec(op, a, b);
        }
        let len = entry_a.len;
        if len == 0 {
            let out_buffer = self.create_storage_buffer(0, "runmat-binary-out");
            return Ok(self.register_existing_buffer(out_buffer, entry_a.shape, entry_a.len));
        }
        if len > (u32::MAX as usize) {
            return Err(gpu_dispatch_length_limit_error("binary_op", len));
        }
        let start = Instant::now();
        {
            let mut enc =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-binary-noop"),
                    });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-binary-noop-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.binary.pipeline);
            drop(pass);
            self.submit(enc);
        }
        self.device_ref().poll(wgpu::Maintain::Poll);
        {
            let enc = self
                .device_ref()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-binary-flush-gap"),
                });
            self.submit(enc);
        }
        let out_buffer = self.create_storage_buffer_checked(len, "runmat-binary-out")?;
        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;
        let mut offset = 0usize;
        while offset < len {
            let remaining = len - offset;
            let chunk_len = remaining.min(chunk_capacity);
            let params = crate::backend::wgpu::params::LenOpParams {
                len: chunk_len as u32,
                op: op as u32,
                offset: offset as u32,
                total: len as u32,
            };
            let params_buffer = self.kernel_resources.uniform_buffer(
                self.device_ref(),
                UniformBufferKey::LenOpParams,
                std::mem::size_of::<crate::backend::wgpu::params::LenOpParams>() as u64,
                "runmat-binary-params",
            );
            self.queue
                .write_buffer(params_buffer.as_ref(), 0, bytes_of(&params));
            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-binary-bind"),
                    layout: &self.pipelines.binary.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry_a.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: entry_b.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: out_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });
            let groups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_len as u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::elementwise::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.binary.pipeline,
                &bind_group,
                groups,
            );
            offset += chunk_len;
        }
        let handle = self.register_existing_buffer(out_buffer, entry_a.shape, len);
        if let Some(info) = runmat_accelerate_api::handle_transpose_info(a) {
            runmat_accelerate_api::record_handle_transpose(&handle, info.base_rows, info.base_cols);
        }
        self.telemetry
            .record_fused_elementwise_duration(start.elapsed());
        Ok(handle)
    }
    fn binary_op_broadcast_exec(
        &self,
        op: crate::backend::wgpu::types::BinaryOpCode,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        use crate::backend::wgpu::params::{AlignedU32, BinaryBroadcastParams, BCAST_MAX_RANK};
        let entry_a = self.get_entry(a)?;
        let entry_b = self.get_entry(b)?;
        // Compute broadcasted output shape
        let mut shape_a = entry_a.shape.clone();
        let mut shape_b = entry_b.shape.clone();
        let rank = shape_a.len().max(shape_b.len());
        if rank > BCAST_MAX_RANK {
            return Err(anyhow!("broadcast rank exceeds limit"));
        }
        if shape_a.len() < rank {
            let pad = rank - shape_a.len();
            let mut v = vec![1usize; pad];
            v.extend_from_slice(&shape_a);
            shape_a = v;
        }
        if shape_b.len() < rank {
            let pad = rank - shape_b.len();
            let mut v = vec![1usize; pad];
            v.extend_from_slice(&shape_b);
            shape_b = v;
        }
        let mut out_shape: Vec<usize> = vec![1; rank];
        for i in 0..rank {
            let da = shape_a[i];
            let db = shape_b[i];
            if da == db {
                out_shape[i] = da;
            } else if da == 1 {
                out_shape[i] = db;
            } else if db == 1 {
                out_shape[i] = da;
            } else {
                return Err(anyhow!("shape mismatch for broadcast"));
            }
        }
        let len: usize = out_shape
            .iter()
            .copied()
            .fold(1usize, |a, b| a.saturating_mul(b));
        if len == 0 {
            let out_buffer = self.create_storage_buffer(0, "runmat-binary-bcast-out");
            return Ok(self.register_existing_buffer(out_buffer, out_shape, 0));
        }
        if len > (u32::MAX as usize) {
            return Err(gpu_dispatch_length_limit_error("binary_op_broadcast", len));
        }
        // Compute strides (column-major)
        let mut stride_a: Vec<u32> = vec![0; rank];
        let mut stride_b: Vec<u32> = vec![0; rank];
        let mut s: u64 = 1;
        for i in 0..rank {
            stride_a[i] = if shape_a[i] == 1 { 0 } else { s as u32 };
            s = s.saturating_mul(shape_a[i] as u64);
        }
        s = 1;
        for i in 0..rank {
            stride_b[i] = if shape_b[i] == 1 { 0 } else { s as u32 };
            s = s.saturating_mul(shape_b[i] as u64);
        }

        // Create output buffer
        let out_buffer = self.create_storage_buffer_checked(len, "runmat-binary-bcast-out")?;
        // Prepare params buffer and bind group once; update params per chunk
        let params_size = std::mem::size_of::<BinaryBroadcastParams>() as u64;
        let params_buffer = self.kernel_resources.uniform_buffer(
            self.device_ref(),
            UniformBufferKey::BinaryBroadcastParams,
            params_size,
            "runmat-binary-bcast-params",
        );
        let bind_group_layout = &self.pipelines.binary_broadcast.layout;
        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-binary-bcast-bind"),
                layout: bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.get_entry(a)?.buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.get_entry(b)?.buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out_buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

        // Dispatch in chunks
        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;
        let mut offset = 0usize;
        let start = Instant::now();
        while offset < len {
            let remaining = len - offset;
            let chunk_len = remaining.min(chunk_capacity);
            // Pack params
            let mut params = BinaryBroadcastParams {
                len: chunk_len as u32,
                offset: offset as u32,
                rank: rank as u32,
                op: op as u32,
                out_shape: [AlignedU32::new(0); BCAST_MAX_RANK],
                a_shape: [AlignedU32::new(0); BCAST_MAX_RANK],
                b_shape: [AlignedU32::new(0); BCAST_MAX_RANK],
                a_strides: [AlignedU32::new(0); BCAST_MAX_RANK],
                b_strides: [AlignedU32::new(0); BCAST_MAX_RANK],
            };
            for i in 0..rank {
                params.out_shape[i] = AlignedU32::new(out_shape[i] as u32);
                params.a_shape[i] = AlignedU32::new(shape_a[i] as u32);
                params.b_shape[i] = AlignedU32::new(shape_b[i] as u32);
                params.a_strides[i] = AlignedU32::new(stride_a[i]);
                params.b_strides[i] = AlignedU32::new(stride_b[i]);
            }
            self.queue_ref()
                .write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));
            let groups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_len as u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::elementwise::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.binary_broadcast.pipeline,
                &bind_group,
                groups,
            );
            offset += chunk_len;
        }
        let handle = self.register_existing_buffer(out_buffer, out_shape, len);
        self.telemetry
            .record_fused_elementwise_duration(start.elapsed());
        Ok(handle)
    }

}
