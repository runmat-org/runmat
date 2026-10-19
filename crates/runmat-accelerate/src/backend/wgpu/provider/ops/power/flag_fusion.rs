//! Full-flag emission prerequisite, not a complete POWER domain-probe receipt.
use super::super::elementwise::{
    ElementwiseInvocationGeometry, FrozenFusionRecipe, FusedElementwiseRepresentation,
    FusedElementwiseWork, FusionUniformAbi,
};
use super::*;
use crate::backend::wgpu::pipelines::FrozenLinearLaunch;
use crate::backend::wgpu::shaders::power::classification_shader;
use runmat_accelerate_api::{
    allocation::{AllocationIdentity, TensorViewIdentity},
    ProviderRealElementwiseObserved,
};

pub(super) struct PowerFlagFusionRecipe {
    input_precision: NumericPrecision,
    flag_precision: NumericPrecision,
    shape: Vec<usize>,
    len: usize,
    fusion: FrozenFusionRecipe,
}

pub(super) struct PreparedPowerFlagFusion {
    owner_device_id: u32,
    source_views: [TensorViewIdentity; 2],
    recipe: PowerFlagFusionRecipe,
}

impl PreparedPowerFlagFusion {
    pub(super) fn pipeline_key(&self) -> u64 {
        self.recipe.pipeline_key()
    }
    pub(super) fn physical_identity(&self) -> Result<runmat_execution::Digest> {
        self.recipe.physical_identity()
    }
    pub(super) fn resource_work(&self) -> ProviderRealElementwiseObserved {
        self.recipe.resource_work()
    }
}

impl PowerFlagFusionRecipe {
    pub(super) fn pipeline_key(&self) -> u64 {
        self.fusion.pipeline().key()
    }
    pub(super) fn physical_identity(&self) -> Result<runmat_execution::Digest> {
        let pipeline = self.fusion.pipeline();
        let launch = self.fusion.launch();
        let work = self.resource_work();
        Ok(runmat_execution::Digest::sha256_serialized(&(
            "wgpu-power-flag-physical-v1",
            &pipeline.shader,
            &pipeline.layout,
            pipeline.workgroup,
            launch.workgroup(),
            launch.chunk_capacity(),
            &self.shape,
            self.len,
            work.output_buffer_bytes,
            work.uniform_allocations,
            work.uniform_buffer_bytes,
            work.dispatches,
            work.queue_submissions,
        ))?)
    }

    pub(super) fn resource_work(&self) -> ProviderRealElementwiseObserved {
        self.fusion.expected()
    }
}

impl PowerFlagFusionRecipe {
    pub(super) fn plan(
        input_precision: NumericPrecision,
        flag_precision: NumericPrecision,
        shape: &[usize],
        launch: FrozenLinearLaunch,
        limits: &wgpu::Limits,
        enabled_features: wgpu::Features,
    ) -> Result<Self> {
        ensure!(
            input_precision == flag_precision
                || (input_precision == NumericPrecision::F32
                    && flag_precision == NumericPrecision::F64),
            "power classification requires exact floating input conversion"
        );
        ensure!(
            flag_precision != NumericPrecision::F64
                || enabled_features
                    .contains(wgpu::Features::SHADER_F64 | wgpu::Features::SHADER_INT64),
            "power flag shader IEEE capabilities unavailable"
        );
        let len = shape
            .iter()
            .try_fold(1usize, |count, &extent| count.checked_mul(extent))
            .ok_or_else(|| anyhow!("power flag shape overflows"))?;
        ensure!(len != 0, "empty power has no flag-emission route");
        let representation = flag_representation(input_precision, flag_precision);
        let fusion = FrozenFusionRecipe::plan(
            &classification_shader(input_precision, flag_precision),
            2,
            &FusedElementwiseWork {
                output_shape: shape,
                len,
                invocation_geometry: ElementwiseInvocationGeometry::OutputLanes,
                representation,
                uniform_abi: Some(FusionUniformAbi::Broadcast),
            },
            launch,
            limits,
        )?;
        Ok(Self {
            input_precision,
            flag_precision,
            shape: shape.to_vec(),
            len,
            fusion,
        })
    }
}

fn flag_representation(
    input_precision: NumericPrecision,
    flag_precision: NumericPrecision,
) -> FusedElementwiseRepresentation {
    FusedElementwiseRepresentation {
        input_precision,
        output_precision: flag_precision,
        output_storage: GpuTensorStorage::Real,
        logical_output: false,
    }
}

/// Privately issued after owner execution/validation. Passive optional fence
/// tickets do not themselves supply peak or reduction/readback authority.
pub(super) struct PowerFlagFusionReceipt {
    source_views: [TensorViewIdentity; 2],
    output_allocation: AllocationIdentity,
    output_view: TensorViewIdentity,
    shape: Vec<usize>,
    flag_precision: NumericPrecision,
    resources: ProviderRealElementwiseObserved,
    tickets: Vec<crate::backend::wgpu::dispatch::observation::ProbeFenceTicket>,
}

impl PowerFlagFusionReceipt {
    #[cfg(test)]
    pub(super) fn tickets(
        &self,
    ) -> &[crate::backend::wgpu::dispatch::observation::ProbeFenceTicket] {
        &self.tickets
    }

    pub(super) fn require_output(&self, output: &GpuTensorHandle) -> Result<()> {
        output.view().validate(output.physical())?;
        output.allocation_lease().validate_live()?;
        ensure!(
            output.allocation_identity() == self.output_allocation
                && output.view_identity() == self.output_view
                && output.shape() == self.shape
                && output.physical().requested_bytes()
                    == self
                        .resources
                        .output_buffer_bytes
                        .ok_or_else(|| anyhow!("power flag allocation observation missing"))?
                && output.physical().storage() == GpuTensorStorage::Real
                && output.view().layout() == runmat_accelerate_api::GpuTensorLayout::Contiguous
                && !output.view().is_logical()
                && output.physical().element_type()
                    == match self.flag_precision {
                        NumericPrecision::F32 => runmat_accelerate_api::NumericElementType::F32,
                        NumericPrecision::F64 => runmat_accelerate_api::NumericElementType::F64,
                    }
                && runmat_accelerate_api::handle_precision(output)
                    == Some(match self.flag_precision {
                        NumericPrecision::F32 => runmat_accelerate_api::ProviderPrecision::F32,
                        NumericPrecision::F64 => runmat_accelerate_api::ProviderPrecision::F64,
                    })
                && self
                    .source_views
                    .iter()
                    .all(|source| source.allocation() != self.output_allocation),
            "power flag output differs from its observed fresh allocation"
        );
        ensure!(
            usize::try_from(self.resources.queue_submissions).ok() == Some(self.tickets.len()),
            "POWER flag submission ticket count mismatch"
        );
        Ok(())
    }
}

impl WgpuProvider {
    pub(super) fn prepare_power_flag_fusion_exec(
        &self,
        base: &GpuTensorHandle,
        exponent: &GpuTensorHandle,
        shape: &[usize],
        len: usize,
        input_precision: NumericPrecision,
    ) -> Result<PreparedPowerFlagFusion> {
        let (actual_shape, actual_len) =
            self.admitted_power_shape(base, exponent, input_precision)?;
        ensure!(
            actual_shape == shape && actual_len == len,
            "power flag source geometry changed"
        );
        self.require_ieee_words(self.precision)?;
        let recipe = PowerFlagFusionRecipe::plan(
            input_precision,
            self.precision,
            shape,
            self.linear_launch,
            &self.adapter_limits,
            self.enabled_features,
        )?;
        Ok(PreparedPowerFlagFusion {
            owner_device_id: self.device_id(),
            source_views: [base.view_identity(), exponent.view_identity()],
            recipe,
        })
    }

    #[cfg(test)]
    pub(super) fn execute_power_flag_fusion_exec(
        &self,
        base: &GpuTensorHandle,
        exponent: &GpuTensorHandle,
        shape: &[usize],
        len: usize,
        input_precision: NumericPrecision,
    ) -> Result<(GpuTensorHandle, PowerFlagFusionReceipt)> {
        let prepared =
            self.prepare_power_flag_fusion_exec(base, exponent, shape, len, input_precision)?;
        self.execute_power_flag_fusion_prepared_exec(base, exponent, prepared)
    }

    pub(super) fn execute_power_flag_fusion_prepared_exec(
        &self,
        base: &GpuTensorHandle,
        exponent: &GpuTensorHandle,
        prepared: PreparedPowerFlagFusion,
    ) -> Result<(GpuTensorHandle, PowerFlagFusionReceipt)> {
        let PreparedPowerFlagFusion {
            owner_device_id,
            source_views,
            recipe,
        } = prepared;
        let (actual_shape, actual_len) =
            self.admitted_power_shape(base, exponent, recipe.input_precision)?;
        ensure!(
            owner_device_id == self.device_id()
                && source_views == [base.view_identity(), exponent.view_identity()]
                && actual_shape == recipe.shape
                && actual_len == recipe.len,
            "prepared POWER flag source/provider changed"
        );
        self.require_ieee_words(self.precision)?;
        let (output, resources, tickets) = self
            .fused_elementwise_prepared_probe_with_telemetry_exec(
                &classification_shader(recipe.input_precision, recipe.flag_precision),
                &[base.clone(), exponent.clone()],
                FusedElementwiseWork {
                    output_shape: &recipe.shape,
                    len: recipe.len,
                    invocation_geometry: ElementwiseInvocationGeometry::OutputLanes,
                    uniform_abi: Some(FusionUniformAbi::Broadcast),
                    representation: flag_representation(
                        recipe.input_precision,
                        recipe.flag_precision,
                    ),
                },
                &recipe.fusion,
            )?;
        let receipt = PowerFlagFusionReceipt {
            source_views,
            output_allocation: output.allocation_identity(),
            output_view: output.view_identity(),
            shape: recipe.shape,
            flag_precision: recipe.flag_precision,
            resources,
            tickets,
        };
        receipt.require_output(&output)?;
        Ok((output, receipt))
    }
}

#[cfg(test)]
#[path = "flag_fusion/tests.rs"]
mod tests;
