//! Frozen physical work for the existing ordinary single-output fusion runner.
//! This is not a source theorem or a whole-operation lifetime/peak receipt.
use super::super::pipeline_recipe::{FusionPipelineLayout, PipelineRecipe};
use super::*;
use crate::backend::wgpu::pipelines::FrozenLinearLaunch;
use runmat_accelerate_api::ProviderRealElementwiseObserved;

/// Selected by the canonical shader generator, never by WGSL text inspection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::backend::wgpu::provider::backend) enum FusionUniformAbi {
    Simple,
    Broadcast,
}

impl FusionUniformAbi {
    pub(super) fn requested_bytes(self, inputs: usize) -> Result<u64> {
        match self {
            Self::Simple => {
                Ok(std::mem::size_of::<crate::backend::wgpu::params::FusionParams>() as u64)
            }
            Self::Broadcast => crate::backend::wgpu_geometry::fusion_broadcast_uniform_bytes(
                inputs,
                crate::backend::wgpu::params::BCAST_MAX_RANK,
            )
            .ok_or_else(|| anyhow!("fused_elementwise: uniform size overflows")),
        }
    }
}

pub(super) fn validate_output_words(
    shape: &[usize],
    words: usize,
    representation: &FusedElementwiseRepresentation,
) -> Result<()> {
    let elements = if shape.contains(&0) {
        0
    } else {
        shape
            .iter()
            .try_fold(1usize, |count, &extent| count.checked_mul(extent))
            .ok_or_else(|| anyhow!("fused_elementwise: output shape overflows"))?
    };
    let lanes = match representation.output_storage {
        GpuTensorStorage::Real => 1,
        GpuTensorStorage::ComplexInterleaved => 2,
    };
    ensure!(
        !representation.logical_output || lanes == 1,
        "fused_elementwise: logical output cannot have complex storage"
    );
    ensure!(
        elements.checked_mul(lanes) == Some(words),
        "fused_elementwise: shape and physical output words disagree"
    );
    Ok(())
}

pub(in crate::backend::wgpu::provider::backend) struct FrozenFusionRecipe {
    pipeline: PipelineRecipe,
    launch: FrozenLinearLaunch,
    shape: Vec<usize>,
    len: usize,
    input_precision: NumericPrecision,
    output_precision: NumericPrecision,
    output_storage: GpuTensorStorage,
    logical_output: bool,
    uniform_abi: FusionUniformAbi,
    output_bytes: u64,
    uniform_bytes: u64,
    dispatches: u32,
    queue_submissions: u32,
}

impl FrozenFusionRecipe {
    pub(in crate::backend::wgpu::provider::backend) fn plan(
        shader: &str,
        inputs: usize,
        work: &FusedElementwiseWork<'_>,
        launch: FrozenLinearLaunch,
        limits: &wgpu::Limits,
    ) -> Result<Self> {
        let shape = work.output_shape;
        let len = work.len;
        let representation = &work.representation;
        ensure!(
            matches!(
                work.invocation_geometry,
                ElementwiseInvocationGeometry::OutputLanes
            ),
            "prepared fusion recipes require output-lane invocation geometry"
        );
        let uniform_abi = work
            .uniform_abi
            .ok_or_else(|| anyhow!("unprepared compatibility shader has no physical recipe"))?;
        validate_output_words(shape, len, representation)?;
        ensure!(inputs != 0, "fused_elementwise: no inputs");
        ensure!(
            len <= u32::MAX as usize,
            "fused_elementwise: tensor too large"
        );
        let storage = checked_binding_count("fused_elementwise", inputs, 1)?;
        let total = checked_binding_count("fused_elementwise", inputs, 2)?;
        validate_compute_binding_counts("fused_elementwise", storage, total, limits)?;
        ensure!(
            limits.max_uniform_buffers_per_shader_stage >= 1,
            "fused_elementwise: uniform binding unavailable"
        );
        ensure!(
            launch.workgroup() <= limits.max_compute_workgroup_size_x
                && launch.workgroup() <= limits.max_compute_invocations_per_workgroup,
            "fused_elementwise: workgroup exceeds enabled limits"
        );
        let capacity = launch
            .chunk_capacity()
            .ok_or_else(|| anyhow!("fused_elementwise: launch capacity overflows"))?;
        let dispatches = u32::try_from(len.div_ceil(capacity))
            .map_err(|_| anyhow!("fused_elementwise: dispatch count overflows"))?;
        let queue_submissions = dispatches
            .checked_add(1)
            .ok_or_else(|| anyhow!("fused_elementwise: submission count overflows"))?;
        // Verify a full chunk against the frozen enabled dispatch limit.
        let groups = launch
            .dispatch_groups(len.min(capacity))
            .ok_or_else(|| anyhow!("fused_elementwise: dispatch exceeds frozen limits"))?;
        ensure!(
            groups <= limits.max_compute_workgroups_per_dimension,
            "fused_elementwise: dispatch exceeds enabled limits"
        );
        let pipeline = PipelineRecipe::fusion(
            shader,
            inputs,
            FusionPipelineLayout::Single,
            launch.workgroup(),
        );
        match uniform_abi {
            FusionUniformAbi::Simple => {}
            FusionUniformAbi::Broadcast => ensure!(
                shape.len() <= crate::backend::wgpu::params::BCAST_MAX_RANK
                    && shape.iter().all(|&extent| u32::try_from(extent).is_ok()),
                "fused_elementwise: broadcast shape exceeds shader ABI"
            ),
        }
        let uniform_bytes = uniform_abi.requested_bytes(inputs)?;
        ensure!(
            uniform_bytes <= u64::from(limits.max_uniform_buffer_binding_size)
                && uniform_bytes <= limits.max_buffer_size,
            "fused_elementwise: uniform exceeds enabled limits"
        );
        let element_size = match representation.output_precision {
            NumericPrecision::F32 => 4,
            NumericPrecision::F64 => 8,
        };
        let output_bytes =
            crate::backend::wgpu::residency::BufferResidency::requested_bytes(len, element_size)
                .ok_or_else(|| anyhow!("fused_elementwise: output extent overflows"))?;
        ensure!(
            output_bytes <= limits.max_buffer_size
                && output_bytes <= u64::from(limits.max_storage_buffer_binding_size),
            "fused_elementwise: output exceeds enabled limits"
        );
        Ok(Self {
            pipeline,
            launch,
            shape: shape.to_vec(),
            len,
            input_precision: representation.input_precision,
            output_precision: representation.output_precision,
            output_storage: representation.output_storage,
            logical_output: representation.logical_output,
            uniform_abi,
            output_bytes,
            uniform_bytes,
            dispatches,
            queue_submissions,
        })
    }

    pub(in crate::backend::wgpu::provider::backend) fn pipeline(&self) -> &PipelineRecipe {
        &self.pipeline
    }
    pub(in crate::backend::wgpu::provider::backend) fn launch(&self) -> FrozenLinearLaunch {
        self.launch
    }
    pub(super) fn uniform_abi(&self) -> FusionUniformAbi {
        self.uniform_abi
    }
    pub(in crate::backend::wgpu::provider::backend) fn expected(
        &self,
    ) -> ProviderRealElementwiseObserved {
        ProviderRealElementwiseObserved {
            output_buffer_bytes: Some(self.output_bytes),
            uniform_allocations: 1,
            uniform_buffer_bytes: self.uniform_bytes,
            dispatches: self.dispatches,
            queue_submissions: self.queue_submissions,
        }
    }
    pub(super) fn same_work(&self, other: &Self) -> bool {
        self.pipeline.shader == other.pipeline.shader
            && self.pipeline.layout == other.pipeline.layout
            && self.pipeline.workgroup == other.pipeline.workgroup
            && self.launch == other.launch
            && self.shape == other.shape
            && self.len == other.len
            && self.input_precision == other.input_precision
            && self.output_precision == other.output_precision
            && self.output_storage == other.output_storage
            && self.logical_output == other.logical_output
            && self.uniform_abi == other.uniform_abi
            && self.expected() == other.expected()
    }
    pub(super) fn require(&self, observed: &ProviderRealElementwiseObserved) -> Result<()> {
        ensure!(
            *observed == self.expected(),
            "ordinary fusion observations differ from the frozen physical recipe"
        );
        Ok(())
    }
}

#[cfg(test)]
#[path = "fusion_recipe/tests.rs"]
mod tests;
#[cfg(test)]
#[path = "fusion_recipe/abi_tests.rs"]
mod abi_tests;
