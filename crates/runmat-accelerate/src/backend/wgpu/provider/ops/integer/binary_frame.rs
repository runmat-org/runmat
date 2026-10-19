//! One frozen physical admission for native integer binary kernels.
use super::*;
use crate::backend::wgpu::pipelines::FrozenLinearLaunch;
use runmat_accelerate_api::unsupported_provider_operation;

#[path = "binary_frame/geometry.rs"]
mod geometry;
#[path = "binary_frame/feasibility.rs"]
mod feasibility;
pub(super) use feasibility::integer_division_feasibility;

#[derive(Clone, Copy)]
pub(super) enum IntegerBinaryOperation {
    Ordinary(crate::backend::wgpu::types::BinaryOpCode),
    Remainder,
    Modulus,
}

impl IntegerBinaryOperation {
    fn kind(self) -> Result<super::super::validated_fusion::IntegerBinaryPipelineKind> {
        use super::super::validated_fusion::IntegerBinaryPipelineKind;
        use crate::backend::wgpu::types::BinaryOpCode;
        Ok(match self {
            Self::Ordinary(BinaryOpCode::Min | BinaryOpCode::Max) => {
                IntegerBinaryPipelineKind::MinMax
            }
            Self::Ordinary(
                BinaryOpCode::Add
                | BinaryOpCode::Sub
                | BinaryOpCode::Mul
                | BinaryOpCode::Div
                | BinaryOpCode::Pow,
            )
            | Self::Remainder
            | Self::Modulus => IntegerBinaryPipelineKind::Arithmetic,
            _ => return Err(anyhow!("unsupported native integer binary operation")),
        })
    }

    pub(super) fn shader_control(self) -> u32 {
        use crate::backend::wgpu::types::BinaryOpCode;
        match self {
            Self::Ordinary(BinaryOpCode::Add) => 0,
            Self::Ordinary(BinaryOpCode::Sub) => 1,
            Self::Ordinary(BinaryOpCode::Mul) => 2,
            Self::Ordinary(BinaryOpCode::Div) => 3,
            Self::Ordinary(BinaryOpCode::Pow) => 4,
            Self::Remainder => 5,
            Self::Modulus => 6,
            Self::Ordinary(BinaryOpCode::Min) => 1,
            Self::Ordinary(BinaryOpCode::Max) => 0,
            _ => unreachable!("integer binary operation was checked by frame admission"),
        }
    }
}

pub(super) struct IntegerBinaryFrame {
    pub(super) lhs: super::super::backend_types::BufferEntry,
    pub(super) rhs: super::super::backend_types::BufferEntry,
    pub(super) integer_type: IntegerElementType,
    pub(super) broadcast: IntegerBroadcastPlan,
    pub(super) output_words: usize,
    pub(super) output_bytes: u64,
    pub(super) launch: FrozenLinearLaunch,
    pub(super) chunk_capacity: usize,
    pub(super) recipe: super::super::pipeline_recipe::PipelineRecipe,
    pub(super) kind: super::super::validated_fusion::IntegerBinaryPipelineKind,
    pub(super) operation: IntegerBinaryOperation,
}

impl IntegerBinaryFrame {
    pub(super) fn plan(
        provider: &WgpuProvider,
        operation_name: &str,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
        operation: IntegerBinaryOperation,
    ) -> Result<Self> {
        let kind = operation.kind()?;
        let lhs_entry = provider.get_entry_raw(lhs)?;
        let rhs_entry = provider.get_entry_raw(rhs)?;
        ensure!(
            !runmat_accelerate_api::handle_is_logical(lhs)
                && !runmat_accelerate_api::handle_is_logical(rhs)
                && !provider.allocations.requires_logical_encoding(lhs)?
                && !provider.allocations.requires_logical_encoding(rhs)?,
            "{operation_name}: logical producer or view is not a numeric integer input"
        );
        let integer_type = lhs_entry
            .integer_type()
            .ok_or_else(|| anyhow!("{operation_name}: expected native integer gpuArray input"))?;
        ensure!(
            rhs_entry.integer_type() == Some(integer_type),
            "{operation_name}: integer operands must have the same class"
        );
        ensure!(
            lhs_entry.storage == GpuTensorStorage::Real
                && rhs_entry.storage == GpuTensorStorage::Real,
            "{operation_name}: complex integer gpuArray binary operation is not supported"
        );
        let required_words = |len| {
            integer_word_count(integer_type, len)?
                .checked_mul(std::mem::size_of::<u32>())
                .ok_or_else(|| anyhow!("{operation_name}: physical word extent overflow"))
        };
        let lhs_bytes = required_words(lhs_entry.len)? as u64;
        let rhs_bytes = required_words(rhs_entry.len)? as u64;
        ensure!(
            lhs_entry.allocated_bytes >= lhs_bytes
                && rhs_entry.allocated_bytes >= rhs_bytes
                && lhs_entry.buffer.size() >= lhs_bytes
                && rhs_entry.buffer.size() >= rhs_bytes,
            "{operation_name}: physical integer input extent is shorter than its words"
        );
        let geometry = geometry::plan(
            &provider.adapter_limits,
            provider.linear_launch,
            operation_name,
            integer_type,
            geometry::IntegerBinaryInputGeometry {
                shape: &lhs_entry.shape,
                len: lhs_entry.len,
                binding_bytes: lhs_entry.buffer.size(),
            },
            geometry::IntegerBinaryInputGeometry {
                shape: &rhs_entry.shape,
                len: rhs_entry.len,
                binding_bytes: rhs_entry.buffer.size(),
            },
        )?;
        let geometry::IntegerBinaryGeometry {
            broadcast,
            output_words,
            output_bytes,
            launch,
            chunk_capacity,
        } = geometry;
        let recipe = super::super::pipeline_recipe::PipelineRecipe {
            shader: kind.shader(launch.workgroup()),
            layout: kind.layout_tag().into(),
            workgroup: launch.workgroup(),
        };
        Ok(Self {
            lhs: lhs_entry,
            rhs: rhs_entry,
            integer_type,
            broadcast,
            output_words,
            output_bytes,
            launch,
            chunk_capacity,
            recipe,
            kind,
            operation,
        })
    }

    pub(super) fn require_live_inputs(
        &self,
        provider: &WgpuProvider,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
    ) -> Result<()> {
        let live_lhs = provider.get_entry_raw(lhs)?;
        let live_rhs = provider.get_entry_raw(rhs)?;
        ensure!(
            runmat_accelerate_api::WgpuBufferOwner::ptr_eq(&live_lhs.buffer, &self.lhs.buffer)
                && runmat_accelerate_api::WgpuBufferOwner::ptr_eq(
                    &live_rhs.buffer,
                    &self.rhs.buffer
                )
                && live_lhs.integer_type() == Some(self.integer_type)
                && live_rhs.integer_type() == Some(self.integer_type)
                && live_lhs.shape == self.lhs.shape
                && live_rhs.shape == self.rhs.shape
                && live_lhs.len == self.lhs.len
                && live_rhs.len == self.rhs.len
                && live_lhs.storage == GpuTensorStorage::Real
                && live_rhs.storage == GpuTensorStorage::Real
                && live_lhs.allocated_bytes >= self.lhs.allocated_bytes
                && live_rhs.allocated_bytes >= self.rhs.allocated_bytes
                && live_lhs.buffer.size() == self.lhs.buffer.size()
                && live_rhs.buffer.size() == self.rhs.buffer.size(),
            "integer binary: borrowed operands changed during preparation"
        );
        ensure!(
            !runmat_accelerate_api::handle_is_logical(lhs)
                && !runmat_accelerate_api::handle_is_logical(rhs)
                && !provider.allocations.requires_logical_encoding(lhs)?
                && !provider.allocations.requires_logical_encoding(rhs)?,
            "integer binary: borrowed operands changed logical producer or view"
        );
        Ok(())
    }
}

#[cfg(test)]
#[path = "binary_frame/tests.rs"]
mod tests;
