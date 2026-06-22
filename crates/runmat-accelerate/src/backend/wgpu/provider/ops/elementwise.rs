use anyhow::{anyhow, ensure, Result};
use bytemuck::bytes_of;
use runmat_accelerate_api::GpuTensorHandle;
use runmat_time::Instant;
use std::sync::Arc;

use super::backend_shared::{
    checked_binding_count, gpu_dispatch_length_limit_error, validate_compute_binding_counts,
};
use super::backend_types::WgpuProvider;
use crate::backend::wgpu::residency::BufferUsageClass;
use crate::backend::wgpu::resources::UniformBufferKey;
use crate::backend::wgpu::shaders::elementwise::{complex_unary_shader, ComplexUnaryOp};
use crate::backend::wgpu::shaders::logical::{
    ELEM_EQ_SHADER_F32, ELEM_EQ_SHADER_F64, ELEM_GE_SHADER_F32, ELEM_GE_SHADER_F64,
    ELEM_GT_SHADER_F32, ELEM_GT_SHADER_F64, ELEM_LE_SHADER_F32, ELEM_LE_SHADER_F64,
    ELEM_LT_SHADER_F32, ELEM_LT_SHADER_F64, ELEM_NE_SHADER_F32, ELEM_NE_SHADER_F64,
    LOGICAL_AND_SHADER_F32, LOGICAL_AND_SHADER_F64, LOGICAL_ISFINITE_SHADER_F32,
    LOGICAL_ISFINITE_SHADER_F64, LOGICAL_ISINF_SHADER_F32, LOGICAL_ISINF_SHADER_F64,
    LOGICAL_ISNAN_SHADER_F32, LOGICAL_ISNAN_SHADER_F64, LOGICAL_NOT_SHADER_F32,
    LOGICAL_NOT_SHADER_F64, LOGICAL_OR_SHADER_F32, LOGICAL_OR_SHADER_F64, LOGICAL_XOR_SHADER_F32,
    LOGICAL_XOR_SHADER_F64,
};
use crate::backend::wgpu::types::NumericPrecision;

impl WgpuProvider {
    pub(crate) fn logical_islogical_exec(&self, a: &GpuTensorHandle) -> Result<bool> {
        let _ = self.get_entry(a)?;
        Ok(runmat_accelerate_api::handle_is_logical(a))
    }

    pub(crate) fn logical_isreal_exec(&self, a: &GpuTensorHandle) -> Result<bool> {
        let _ = self.get_entry(a)?;
        Ok(runmat_accelerate_api::handle_storage(a)
            != runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved)
    }

    pub(crate) fn unary_double_exec(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        if self.precision != NumericPrecision::F64 {
            return Err(anyhow!(
                "wgpu provider: shader-f64 unavailable; cannot materialise double precision"
            ));
        }
        let entry = self.get_entry(a)?;
        Ok(self.register_existing_buffer(entry.buffer, entry.shape, entry.len))
    }

    pub(crate) fn unary_real_exec(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        if runmat_accelerate_api::handle_storage(a)
            != runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        {
            return Ok(self.register_existing_buffer(entry.buffer, entry.shape, entry.len));
        }
        self.complex_unary_exec(a, ComplexUnaryOp::Real, false)
    }

    pub(crate) fn unary_imag_exec(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        if runmat_accelerate_api::handle_storage(a)
            != runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        {
            return Ok(self.fill_exec(&entry.shape, 0.0)?);
        }
        self.complex_unary_exec(a, ComplexUnaryOp::Imag, false)
    }

    pub(crate) fn unary_abs_exec(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        if runmat_accelerate_api::handle_storage(a)
            != runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        {
            return self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Abs, a);
        }
        self.complex_unary_exec(a, ComplexUnaryOp::Abs, false)
    }

    pub(crate) fn unary_conj_exec(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        if runmat_accelerate_api::handle_storage(a)
            != runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        {
            return self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Conj, a);
        }
        self.complex_unary_exec(a, ComplexUnaryOp::Conj, true)
    }

    fn complex_unary_exec(
        &self,
        a: &GpuTensorHandle,
        op: ComplexUnaryOp,
        output_complex: bool,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        ensure!(
            runmat_accelerate_api::handle_storage(a)
                == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
            "complex unary operation requires complex-interleaved input"
        );
        ensure!(
            entry.len % 2 == 0,
            "complex unary operation requires even interleaved buffer length"
        );
        let len = if output_complex {
            entry.len
        } else {
            entry.len / 2
        };
        let shader = complex_unary_shader(op, self.precision);
        let out = self.fused_elementwise_with_telemetry_exec(
            &shader,
            std::slice::from_ref(a),
            &entry.shape,
            len,
        )?;
        if output_complex {
            runmat_accelerate_api::set_handle_storage(
                &out,
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
            );
        }
        Ok(out)
    }

    pub(crate) fn unary_pow2_exec(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let out = self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Pow2, a)?;
        // Record squared->base mapping for later reduction fusion (moments reuse).
        if let Ok(mut map) = self.pow2_of.lock() {
            map.insert(out.buffer_id, a.buffer_id);
        }
        Ok(out)
    }

    pub(crate) fn pow2_scale_exec(
        &self,
        mantissa: &GpuTensorHandle,
        exponent: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        ensure!(
            mantissa.shape == exponent.shape,
            "pow2_scale requires matching shapes"
        );
        let pow = self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Pow2, exponent)?;
        let result = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Mul,
            mantissa,
            &pow,
        );
        let _ = self.free_exec(&pow);
        result
    }

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
            self.fused_elementwise_with_telemetry_exec(
                shader,
                &[a.clone(), b.clone()],
                &entry_a.shape,
                len,
            )?
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
            self.fused_elementwise_with_telemetry_exec(
                shader,
                &[a.clone(), b.clone()],
                &entry_a.shape,
                len,
            )?
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
            self.fused_elementwise_with_telemetry_exec(
                shader,
                &[a.clone(), b.clone()],
                &entry_a.shape,
                len,
            )?
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
            self.fused_elementwise_with_telemetry_exec(
                shader,
                &[a.clone(), b.clone()],
                &entry_a.shape,
                len,
            )?
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
            self.fused_elementwise_with_telemetry_exec(
                shader,
                &[a.clone(), b.clone()],
                &entry_a.shape,
                len,
            )?
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
            self.fused_elementwise_with_telemetry_exec(
                shader,
                &[a.clone(), b.clone()],
                &entry_a.shape,
                len,
            )?
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
            self.fused_elementwise_with_telemetry_exec(
                shader,
                &[a.clone(), b.clone()],
                &entry_a.shape,
                len,
            )?
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
            self.fused_elementwise_with_telemetry_exec(
                shader,
                &[a.clone(), b.clone()],
                &entry_a.shape,
                len,
            )?
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
            self.fused_elementwise_with_telemetry_exec(
                shader,
                &[a.clone(), b.clone()],
                &entry_a.shape,
                len,
            )?
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
            self.fused_elementwise_with_telemetry_exec(
                shader,
                std::slice::from_ref(a),
                &entry.shape,
                len,
            )?
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
            self.fused_elementwise_with_telemetry_exec(
                shader,
                std::slice::from_ref(a),
                &entry.shape,
                len,
            )?
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
            self.fused_elementwise_with_telemetry_exec(
                shader,
                std::slice::from_ref(a),
                &entry.shape,
                len,
            )?
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
            self.fused_elementwise_with_telemetry_exec(
                shader,
                std::slice::from_ref(a),
                &entry.shape,
                len,
            )?
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
            let params_buffer = match self.provider_precision_exec() {
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

    pub(crate) fn fused_elementwise_exec(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        len: usize,
    ) -> Result<GpuTensorHandle> {
        if inputs.is_empty() {
            return Err(anyhow!("fused_elementwise: no inputs"));
        }
        if len > u32::MAX as usize {
            return Err(anyhow!("fused_elementwise: tensor too large"));
        }
        let storage_bindings = checked_binding_count("fused_elementwise", inputs.len(), 1)?;
        let total_bindings = checked_binding_count("fused_elementwise", inputs.len(), 2)?;
        validate_compute_binding_counts(
            "fused_elementwise",
            storage_bindings,
            total_bindings,
            &self.adapter_limits,
        )?;
        let entries = inputs
            .iter()
            .map(|h| self.get_entry(h))
            .collect::<Result<Vec<_>>>()?;
        let (output_buffer, _) = self.create_storage_buffer_for_usage(
            BufferUsageClass::FusionOut,
            len,
            "runmat-fusion-output",
        );
        let bind_group_layout = self.cached_fusion_bind_group_layout(inputs.len());
        let pipeline_layout = crate::backend::wgpu::pipelines::create_pipeline_layout(
            self.device_ref(),
            "runmat-fusion-pipeline-layout",
            bind_group_layout.as_ref(),
        );
        let layout_tag = {
            let mut tag = String::from("runmat-fusion-layout-");
            tag.push_str(&inputs.len().to_string());
            tag
        };
        let shader_hash = self.compute_pipeline_hash_bytes(
            shader.as_bytes(),
            &layout_tag,
            Some(crate::backend::wgpu::config::effective_workgroup_size()),
        );
        let module = crate::backend::wgpu::pipelines::create_shader_module(
            self.device_ref(),
            "runmat-fusion-shader",
            shader,
        );
        let pipeline = self.get_or_create_pipeline(
            shader_hash,
            &pipeline_layout,
            &module,
            "runmat-fusion-pipeline",
            Some(shader.as_bytes()),
            Some(&layout_tag),
            Some(crate::backend::wgpu::config::effective_workgroup_size()),
        );
        // Warmup once
        crate::backend::wgpu::dispatch::elementwise::warmup_noop(
            self.device_ref(),
            self.queue_ref(),
            &pipeline,
        );
        self.device_ref().poll(wgpu::Maintain::Poll);

        let broadcast_mode = shader.contains("out_shape") || shader.contains("a_shape");

        struct BroadcastUniformState {
            buffer: Arc<wgpu::Buffer>,
            template: Vec<u8>,
        }

        impl BroadcastUniformState {
            fn update(&mut self, queue: &wgpu::Queue, len: u32, offset: u32) {
                self.template[..4].copy_from_slice(&len.to_ne_bytes());
                self.template[4..8].copy_from_slice(&offset.to_ne_bytes());
                queue.write_buffer(self.buffer.as_ref(), 0, &self.template);
            }
        }

        enum FusionUniformState {
            Broadcast(BroadcastUniformState),
            Simple { buffer: Arc<wgpu::Buffer> },
        }

        let uniform_state = if broadcast_mode {
            let rank = output_shape.len();
            let max_rank = crate::backend::wgpu::params::BCAST_MAX_RANK;
            let mut bytes: Vec<u8> = Vec::with_capacity(4 * 4 + (max_rank * 4 * 4));
            let write_u32 = |buf: &mut Vec<u8>, v: u32| buf.extend_from_slice(&v.to_ne_bytes());
            write_u32(&mut bytes, 0); // len placeholder
            write_u32(&mut bytes, 0); // offset placeholder
            write_u32(&mut bytes, rank as u32);
            write_u32(&mut bytes, 0);
            let write_packed_array = |buf: &mut Vec<u8>, vals: &[u32]| {
                for &val in vals.iter() {
                    write_u32(buf, val);
                    write_u32(buf, 0);
                    write_u32(buf, 0);
                    write_u32(buf, 0);
                }
                for _ in vals.len()..max_rank {
                    write_u32(buf, 0);
                    write_u32(buf, 0);
                    write_u32(buf, 0);
                    write_u32(buf, 0);
                }
            };
            let out_shape_u32: Vec<u32> = output_shape.iter().map(|&d| d as u32).collect();
            write_packed_array(&mut bytes, &out_shape_u32);
            for entry in &entries {
                let mut shape = entry.shape.clone();
                if shape.len() < rank {
                    let pad = rank - shape.len();
                    let mut v = vec![1usize; pad];
                    v.extend_from_slice(&shape);
                    shape = v;
                }
                let shape_u32: Vec<u32> = shape.iter().map(|&d| d as u32).collect();
                write_packed_array(&mut bytes, &shape_u32);
                let mut strides: Vec<u32> = vec![0; rank];
                let mut s: u64 = 1;
                for i in 0..rank {
                    strides[i] = if shape[i] == 1 { 0 } else { s as u32 };
                    s = s.saturating_mul(shape[i] as u64);
                }
                write_packed_array(&mut bytes, &strides);
            }
            let buffer = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("runmat-fusion-params"),
                size: bytes.len() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.queue.write_buffer(buffer.as_ref(), 0, &bytes);
            let uniform_buffer = buffer.clone();
            let state = BroadcastUniformState {
                buffer,
                template: bytes,
            };
            (FusionUniformState::Broadcast(state), uniform_buffer)
        } else {
            let buffer = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("runmat-fusion-params"),
                size: std::mem::size_of::<crate::backend::wgpu::params::FusionParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            (
                FusionUniformState::Simple {
                    buffer: buffer.clone(),
                },
                buffer,
            )
        };

        let (mut uniform_state, uniform_buffer) = uniform_state;

        let mut bind_entries = Vec::with_capacity(total_bindings);
        for (idx, entry) in entries.iter().enumerate() {
            bind_entries.push(wgpu::BindGroupEntry {
                binding: idx as u32,
                resource: entry.buffer.as_ref().as_entire_binding(),
            });
        }
        bind_entries.push(wgpu::BindGroupEntry {
            binding: inputs.len() as u32,
            resource: output_buffer.as_ref().as_entire_binding(),
        });
        bind_entries.push(wgpu::BindGroupEntry {
            binding: storage_bindings as u32,
            resource: uniform_buffer.as_ref().as_entire_binding(),
        });

        let bind_group =
            self.bind_group_cache
                .get_or_create(bind_group_layout.as_ref(), &bind_entries, || {
                    Arc::new(
                        self.device_ref()
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("runmat-fusion-bind-group"),
                                layout: bind_group_layout.as_ref(),
                                entries: &bind_entries,
                            }),
                    )
                });

        let output_ptr = output_buffer.as_ref() as *const wgpu::Buffer as usize;
        let input_ids: Vec<u64> = inputs.iter().map(|h| h.buffer_id).collect();
        log::trace!(
            "fusion elementwise begin len={} out_ptr=0x{:x} inputs={:?}",
            len,
            output_ptr,
            input_ids
        );
        // Dispatch in chunks to satisfy 65535 limit
        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;
        let mut offset_elems = 0usize;
        let mut chunk_index = 0usize;
        let mut last_submission_id = None;
        while offset_elems < len {
            let remaining = len - offset_elems;
            let chunk_len = remaining.min(chunk_capacity);

            log::trace!(
                "fusion elementwise chunk start out_ptr=0x{:x} chunk_len={} offset={} chunk_index={}",
                output_ptr,
                chunk_len,
                offset_elems,
                chunk_index
            );
            match &mut uniform_state {
                FusionUniformState::Broadcast(state) => {
                    state.update(self.queue_ref(), chunk_len as u32, offset_elems as u32)
                }
                FusionUniformState::Simple { buffer } => {
                    let params = crate::backend::wgpu::params::FusionParams {
                        len: chunk_len as u32,
                        offset: offset_elems as u32,
                        _pad1: 0,
                        _pad2: 0,
                    };
                    self.queue
                        .write_buffer(buffer.as_ref(), 0, bytes_of(&params));
                }
            }

            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_len as u32,
                crate::backend::wgpu::config::effective_workgroup_size(),
            );
            let mut enc =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-fusion-elementwise-encoder"),
                    });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("runmat-fusion-elementwise-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, bind_group.as_ref(), &[]);
                if workgroups > 0 {
                    pass.dispatch_workgroups(workgroups, 1, 1);
                }
            }
            let submission_id = self.submit(enc);
            last_submission_id = Some(submission_id);
            log::trace!(
                "fusion elementwise chunk complete out_ptr=0x{:x} chunk_len={} offset={} submission_id={}",
                output_ptr,
                chunk_len,
                offset_elems,
                submission_id
            );
            offset_elems += chunk_len;
            chunk_index += 1;
        }
        let handle = self.register_existing_buffer_with_usage(
            output_buffer,
            output_shape.to_vec(),
            len,
            BufferUsageClass::FusionOut,
        );
        log::trace!(
            "fusion elementwise complete buffer_id={} out_ptr=0x{:x} len={} chunks={} last_submission_id={:?}",
            handle.buffer_id,
            output_ptr,
            len,
            chunk_index,
            last_submission_id
        );
        if let Some(submission_id) = last_submission_id {
            self.record_buffer_submission(handle.buffer_id, submission_id);
        }
        Ok(handle)
    }
    pub(crate) fn fused_elementwise_multi_exec(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        len: usize,
        num_outputs: usize,
    ) -> Result<Vec<GpuTensorHandle>> {
        if inputs.is_empty() {
            return Err(anyhow!("fused_elementwise_multi: no inputs"));
        }
        if num_outputs == 0 {
            return Err(anyhow!("fused_elementwise_multi: num_outputs is zero"));
        }
        if len > u32::MAX as usize {
            return Err(anyhow!("fused_elementwise_multi: tensor too large"));
        }
        let storage_bindings =
            checked_binding_count("fused_elementwise_multi", inputs.len(), num_outputs)?;
        let total_bindings = checked_binding_count("fused_elementwise_multi", storage_bindings, 1)?;
        validate_compute_binding_counts(
            "fused_elementwise_multi",
            storage_bindings,
            total_bindings,
            &self.adapter_limits,
        )?;

        let entries = inputs
            .iter()
            .map(|h| self.get_entry(h))
            .collect::<Result<Vec<_>>>()?;

        let output_buffers: Vec<_> = (0..num_outputs)
            .map(|_| {
                self.create_storage_buffer_for_usage(
                    BufferUsageClass::FusionOut,
                    len,
                    "runmat-fusion-multi-output",
                )
            })
            .collect();

        let bind_group_layout = {
            let key = format!(
                "runmat-fusion-multi-layout-{}-{}",
                inputs.len(),
                num_outputs
            );
            self.cached_bind_group_layout(&key, |device| {
                crate::backend::wgpu::bindings::build_fusion_multi_bgl(
                    device,
                    inputs.len(),
                    num_outputs,
                )
            })
        };
        let pipeline_layout = crate::backend::wgpu::pipelines::create_pipeline_layout(
            self.device_ref(),
            "runmat-fusion-multi-pipeline-layout",
            bind_group_layout.as_ref(),
        );
        let layout_tag = format!(
            "runmat-fusion-multi-layout-{}-{}",
            inputs.len(),
            num_outputs
        );
        let shader_hash = self.compute_pipeline_hash_bytes(
            shader.as_bytes(),
            &layout_tag,
            Some(crate::backend::wgpu::config::effective_workgroup_size()),
        );
        let module = crate::backend::wgpu::pipelines::create_shader_module(
            self.device_ref(),
            "runmat-fusion-multi-shader",
            shader,
        );
        let pipeline = self.get_or_create_pipeline(
            shader_hash,
            &pipeline_layout,
            &module,
            "runmat-fusion-multi-pipeline",
            Some(shader.as_bytes()),
            Some(&layout_tag),
            Some(crate::backend::wgpu::config::effective_workgroup_size()),
        );
        crate::backend::wgpu::dispatch::elementwise::warmup_noop(
            self.device_ref(),
            self.queue_ref(),
            &pipeline,
        );
        self.device_ref().poll(wgpu::Maintain::Poll);

        struct BroadcastUniformState {
            buffer: Arc<wgpu::Buffer>,
            template: Vec<u8>,
        }
        impl BroadcastUniformState {
            fn update(&mut self, queue: &wgpu::Queue, len: u32, offset: u32) {
                self.template[..4].copy_from_slice(&len.to_ne_bytes());
                self.template[4..8].copy_from_slice(&offset.to_ne_bytes());
                queue.write_buffer(self.buffer.as_ref(), 0, &self.template);
            }
        }

        let rank = output_shape.len();
        let max_rank = crate::backend::wgpu::params::BCAST_MAX_RANK;
        let mut bytes: Vec<u8> = Vec::with_capacity(4 * 4 + (max_rank * 4 * 4));
        let write_u32 = |buf: &mut Vec<u8>, v: u32| buf.extend_from_slice(&v.to_ne_bytes());
        write_u32(&mut bytes, 0); // len placeholder
        write_u32(&mut bytes, 0); // offset placeholder
        write_u32(&mut bytes, rank as u32);
        write_u32(&mut bytes, 0);
        let write_packed_array = |buf: &mut Vec<u8>, vals: &[u32]| {
            for &val in vals.iter() {
                write_u32(buf, val);
                write_u32(buf, 0);
                write_u32(buf, 0);
                write_u32(buf, 0);
            }
            for _ in vals.len()..max_rank {
                write_u32(buf, 0);
                write_u32(buf, 0);
                write_u32(buf, 0);
                write_u32(buf, 0);
            }
        };
        let out_shape_u32: Vec<u32> = output_shape.iter().map(|&d| d as u32).collect();
        write_packed_array(&mut bytes, &out_shape_u32);
        for entry in &entries {
            let mut shape = entry.shape.clone();
            if shape.len() < rank {
                let pad = rank - shape.len();
                let mut v = vec![1usize; pad];
                v.extend_from_slice(&shape);
                shape = v;
            }
            let shape_u32: Vec<u32> = shape.iter().map(|&d| d as u32).collect();
            write_packed_array(&mut bytes, &shape_u32);
            let mut strides: Vec<u32> = vec![0; rank];
            let mut s: u64 = 1;
            for i in 0..rank {
                strides[i] = if shape[i] == 1 { 0 } else { s as u32 };
                s = s.saturating_mul(shape[i] as u64);
            }
            write_packed_array(&mut bytes, &strides);
        }
        let uniform_buffer = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("runmat-fusion-multi-params"),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.queue.write_buffer(uniform_buffer.as_ref(), 0, &bytes);

        let mut uniform_state = BroadcastUniformState {
            buffer: uniform_buffer.clone(),
            template: bytes,
        };

        let mut bind_entries = Vec::with_capacity(total_bindings);
        for (idx, entry) in entries.iter().enumerate() {
            bind_entries.push(wgpu::BindGroupEntry {
                binding: idx as u32,
                resource: entry.buffer.as_ref().as_entire_binding(),
            });
        }
        for (k, (out_buf, _)) in output_buffers.iter().enumerate() {
            bind_entries.push(wgpu::BindGroupEntry {
                binding: (inputs.len() + k) as u32,
                resource: out_buf.as_ref().as_entire_binding(),
            });
        }
        bind_entries.push(wgpu::BindGroupEntry {
            binding: storage_bindings as u32,
            resource: uniform_buffer.as_ref().as_entire_binding(),
        });

        let bind_group =
            self.bind_group_cache
                .get_or_create(bind_group_layout.as_ref(), &bind_entries, || {
                    Arc::new(
                        self.device_ref()
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("runmat-fusion-multi-bind-group"),
                                layout: bind_group_layout.as_ref(),
                                entries: &bind_entries,
                            }),
                    )
                });

        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;
        let mut offset_elems = 0usize;
        let mut last_submission_id = None;
        while offset_elems < len {
            let remaining = len - offset_elems;
            let chunk_len = remaining.min(chunk_capacity);
            uniform_state.update(self.queue_ref(), chunk_len as u32, offset_elems as u32);
            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_len as u32,
                crate::backend::wgpu::config::effective_workgroup_size(),
            );
            let mut enc =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-fusion-multi-elementwise-encoder"),
                    });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("runmat-fusion-multi-elementwise-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, bind_group.as_ref(), &[]);
                if workgroups > 0 {
                    pass.dispatch_workgroups(workgroups, 1, 1);
                }
            }
            let submission_id = self.submit(enc);
            last_submission_id = Some(submission_id);
            offset_elems += chunk_len;
        }

        let mut handles = Vec::with_capacity(num_outputs);
        for (out_buf, _) in output_buffers {
            let handle = self.register_existing_buffer_with_usage(
                out_buf,
                output_shape.to_vec(),
                len,
                BufferUsageClass::FusionOut,
            );
            if let Some(submission_id) = last_submission_id {
                self.record_buffer_submission(handle.buffer_id, submission_id);
            }
            handles.push(handle);
        }
        Ok(handles)
    }
}
