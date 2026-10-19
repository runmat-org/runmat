//! Complex binary execution and native-precision admission.

use super::super::backend_types::{BufferEntry, NumericBufferRegistration};
use super::*;
use crate::backend::wgpu::shaders::elementwise::{
    complex_binary_broadcast_shader, complex_binary_shader, ComplexBinaryOp,
};

impl WgpuProvider {
    fn complex_division_entry(&self, handle: &GpuTensorHandle) -> Result<BufferEntry> {
        let entry = self.get_entry_raw(handle)?;
        let expected_type = match entry.precision {
            NumericPrecision::F32 => runmat_accelerate_api::NumericElementType::F32,
            NumericPrecision::F64 => runmat_accelerate_api::NumericElementType::F64,
        };
        if entry.integer_type().is_some()
            || entry.element_type != expected_type
            || (entry.precision != self.precision
                && !(self.precision == NumericPrecision::F64
                    && entry.precision == NumericPrecision::F32))
        {
            return Err(unsupported_provider_operation(
                "complex division requires an available native floating precision",
            ));
        }
        ensure!(
            handle.descriptor().element_type == Some(entry.element_type)
                && handle.descriptor().storage == Some(entry.storage)
                && runmat_accelerate_api::handle_storage(handle) == entry.storage
                && !runmat_accelerate_api::handle_is_logical(handle),
            "complex division physical view does not match its authenticated descriptor"
        );
        Ok(entry)
    }

    pub(super) fn complex_division_inputs(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
    ) -> Result<(BufferEntry, BufferEntry)> {
        let entry_a = self.complex_division_entry(a)?;
        let entry_b = self.complex_division_entry(b)?;
        if entry_a.precision != entry_b.precision {
            return Err(unsupported_provider_operation(
                "complex division requires matching native operand precisions",
            ));
        }
        Ok((entry_a, entry_b))
    }

    pub(super) fn complex_binary_op_exec(
        &self,
        op: crate::backend::wgpu::types::BinaryOpCode,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
        entry_a: &BufferEntry,
        entry_b: &BufferEntry,
    ) -> Result<GpuTensorHandle> {
        ensure!(
            entry_a.shape == entry_b.shape,
            "complex binary operation requires matching logical shapes"
        );
        let complex_op = ComplexBinaryOp::try_from_binary_op(op)
            .ok_or_else(|| anyhow!("binary operation is not supported for complex GPU tensors"))?;
        let lhs_complex = self.effective_storage_for_entry(a, entry_a)
            == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved;
        let rhs_complex = self.effective_storage_for_entry(b, entry_b)
            == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved;
        let logical_len = entry_a
            .shape
            .iter()
            .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
            .ok_or_else(|| anyhow!("complex binary operation output length overflow"))?;
        ensure!(
            !lhs_complex || entry_a.len == logical_len.saturating_mul(2),
            "complex lhs storage length does not match logical shape"
        );
        ensure!(
            lhs_complex || entry_a.len == logical_len,
            "real lhs storage length does not match logical shape"
        );
        ensure!(
            !rhs_complex || entry_b.len == logical_len.saturating_mul(2),
            "complex rhs storage length does not match logical shape"
        );
        ensure!(
            rhs_complex || entry_b.len == logical_len,
            "real rhs storage length does not match logical shape"
        );
        let out_len = logical_len
            .checked_mul(2)
            .ok_or_else(|| anyhow!("complex binary operation output length overflow"))?;
        let precision = entry_a.precision;
        let division = if complex_op == ComplexBinaryOp::Div && out_len != 0 {
            self.require_ieee_words(precision)?;
            Some(ElementwiseInvocationGeometry::complex_elements(
                logical_len,
                precision,
                FusionUniformAbi::Simple,
            )?)
        } else {
            None
        };
        let handle = if out_len == 0 {
            let buffer = self.with_emission(|emission, _| {
                if complex_op == ComplexBinaryOp::Div {
                    self.create_storage_buffer_checked_with_element_size(
                        emission,
                        0,
                        "runmat-complex-binary-out",
                        BufferUsageClass::Generic,
                        match precision {
                            NumericPrecision::F32 => 4,
                            NumericPrecision::F64 => 8,
                        },
                    )
                } else {
                    self.create_storage_buffer(emission, 0, "runmat-complex-binary-out")
                }
            })??;
            if complex_op == ComplexBinaryOp::Div {
                self.register_numeric_buffer(
                    buffer.clone(),
                    NumericBufferRegistration {
                        shape: entry_a.shape.clone(),
                        len: 0,
                        physical_element_type: match precision {
                            NumericPrecision::F32 => runmat_accelerate_api::NumericElementType::F32,
                            NumericPrecision::F64 => runmat_accelerate_api::NumericElementType::F64,
                        },
                        storage: GpuTensorStorage::ComplexInterleaved,
                        allocated_bytes: buffer.size(),
                        usage: BufferUsageClass::Generic,
                    },
                )?
            } else {
                self.register_existing_buffer_with_storage(
                    buffer,
                    entry_a.shape.clone(),
                    0,
                    GpuTensorStorage::ComplexInterleaved,
                )?
            }
        } else {
            let shader = complex_binary_shader(complex_op, precision, lhs_complex, rhs_complex);
            if let Some(division) = division {
                let start = Instant::now();
                let output = self.complex_division_dispatch_exec(
                    &shader,
                    &[a.clone(), b.clone()],
                    &entry_a.shape,
                    out_len,
                    division,
                    precision,
                )?;
                self.telemetry
                    .record_fused_elementwise_duration(start.elapsed());
                output
            } else {
                self.fused_elementwise_with_telemetry_and_storage_exec(
                    &shader,
                    &[a.clone(), b.clone()],
                    &entry_a.shape,
                    out_len,
                    GpuTensorStorage::ComplexInterleaved,
                )?
            }
        };
        Ok(handle)
    }

    pub(super) fn complex_binary_broadcast_exec(
        &self,
        op: crate::backend::wgpu::types::BinaryOpCode,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
        entry_a: &BufferEntry,
        entry_b: &BufferEntry,
    ) -> Result<GpuTensorHandle> {
        let complex_op = ComplexBinaryOp::try_from_binary_op(op)
            .ok_or_else(|| anyhow!("binary operation is not supported for complex GPU tensors"))?;
        let lhs_complex = self.effective_storage_for_entry(a, entry_a)
            == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved;
        let rhs_complex = self.effective_storage_for_entry(b, entry_b)
            == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved;

        let (shape_a, shape_b, out_shape) =
            complex_broadcast::admit_shapes(&entry_a.shape, &entry_b.shape)?;

        let logical_len = out_shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| anyhow!("complex broadcast output length overflow"))?;
        let a_logical_len = shape_a
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| anyhow!("complex broadcast lhs length overflow"))?;
        let b_logical_len = shape_b
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| anyhow!("complex broadcast rhs length overflow"))?;
        let expected_a_len = a_logical_len
            .checked_mul(if lhs_complex { 2 } else { 1 })
            .ok_or_else(|| anyhow!("complex broadcast lhs length overflow"))?;
        let expected_b_len = b_logical_len
            .checked_mul(if rhs_complex { 2 } else { 1 })
            .ok_or_else(|| anyhow!("complex broadcast rhs length overflow"))?;
        ensure!(
            entry_a.len == expected_a_len,
            "complex broadcast lhs storage length does not match logical shape"
        );
        ensure!(
            entry_b.len == expected_b_len,
            "complex broadcast rhs storage length does not match logical shape"
        );

        let out_len = logical_len
            .checked_mul(2)
            .ok_or_else(|| anyhow!("complex broadcast output length overflow"))?;
        let precision = entry_a.precision;
        let division = if complex_op == ComplexBinaryOp::Div && out_len != 0 {
            self.require_ieee_words(precision)?;
            Some(ElementwiseInvocationGeometry::complex_elements(
                logical_len,
                precision,
                FusionUniformAbi::Broadcast,
            )?)
        } else {
            None
        };
        if out_len == 0 {
            let out_buffer = self.with_emission(|emission, _| {
                if complex_op == ComplexBinaryOp::Div {
                    self.create_storage_buffer_checked_with_element_size(
                        emission,
                        0,
                        "runmat-complex-bcast-out",
                        BufferUsageClass::Generic,
                        match precision {
                            NumericPrecision::F32 => 4,
                            NumericPrecision::F64 => 8,
                        },
                    )
                } else {
                    self.create_storage_buffer(emission, 0, "runmat-complex-bcast-out")
                }
            })??;
            if complex_op == ComplexBinaryOp::Div {
                return self.register_numeric_buffer(
                    out_buffer.clone(),
                    NumericBufferRegistration {
                        shape: out_shape,
                        len: 0,
                        physical_element_type: match precision {
                            NumericPrecision::F32 => runmat_accelerate_api::NumericElementType::F32,
                            NumericPrecision::F64 => runmat_accelerate_api::NumericElementType::F64,
                        },
                        storage: GpuTensorStorage::ComplexInterleaved,
                        allocated_bytes: out_buffer.size(),
                        usage: BufferUsageClass::Generic,
                    },
                );
            }
            return self.register_existing_buffer_with_storage(
                out_buffer,
                out_shape,
                0,
                GpuTensorStorage::ComplexInterleaved,
            );
        }
        if out_len > u32::MAX as usize {
            return Err(gpu_dispatch_length_limit_error(
                "complex_binary_broadcast",
                out_len,
            ));
        }

        let shader =
            complex_binary_broadcast_shader(complex_op, precision, lhs_complex, rhs_complex);
        let handle = if let Some(division) = division {
            self.complex_division_dispatch_exec(
                &shader,
                &[a.clone(), b.clone()],
                &out_shape,
                out_len,
                division,
                precision,
            )?
        } else {
            self.fused_elementwise_exec(
                &shader,
                &[a.clone(), b.clone()],
                &out_shape,
                out_len,
                GpuTensorStorage::ComplexInterleaved,
            )?
        };
        Ok(handle)
    }

    fn complex_division_dispatch_exec(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        output_lanes: usize,
        invocation_geometry: ElementwiseInvocationGeometry,
        precision: NumericPrecision,
    ) -> Result<GpuTensorHandle> {
        self.fused_elementwise_typed_exec_recorded(
            shader,
            inputs,
            FusedElementwiseWork {
                output_shape,
                len: output_lanes,
                invocation_geometry,
                representation: FusedElementwiseRepresentation {
                    input_precision: precision,
                    output_precision: precision,
                    output_storage: GpuTensorStorage::ComplexInterleaved,
                    logical_output: false,
                },
                uniform_abi: None,
            },
            |_| Ok(()),
            SingleFusionEvidence::Legacy {
                observed: None,
                prepared: None,
            },
            None,
        )
    }
}
