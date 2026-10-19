//! Physical projection and consumption of the existing paired POWER shader.
use super::super::elementwise::{
    ElementwiseInvocationGeometry, FrozenFusionRecipe, FusedElementwiseRepresentation,
    FusedElementwiseWork, FusionUniformAbi,
};
use super::*;
use runmat_accelerate_api::{
    power_service::{PowerEvaluationPhysicalProposal, PowerEvaluationRecipe},
    ProviderElementType, ProviderExecutedRealElementwise, ProviderExecutionError,
    ProviderExecutionResult, ProviderLayout, ProviderRealElementwiseOutputRoute,
    ProviderRealElementwisePhysicalProposal, ProviderRepresentation, ProviderResidency,
    ProviderStorage,
};

pub(super) struct PreparedPowerEvaluation {
    shape: Vec<usize>,
    words: usize,
    representation: FusedElementwiseRepresentation,
    shader: String,
    fusion: FrozenFusionRecipe,
}

impl WgpuProvider {
    fn prepare_power_evaluation_exec(
        &self,
        base: &GpuTensorHandle,
        exponent: &GpuTensorHandle,
        domain: PowerResultDomain,
    ) -> Result<Option<PreparedPowerEvaluation>> {
        let input_precision = self.real_floating_entry(base)?.precision;
        self.real_floating_entry(exponent)?;
        if input_precision != self.precision
            && !(input_precision == NumericPrecision::F32
                && self.precision == NumericPrecision::F64)
        {
            return Ok(None);
        }
        let (shape, len) = self.admitted_power_shape(base, exponent, input_precision)?;
        self.prepare_power_evaluation_geometry(input_precision, shape, len, domain)
    }

    pub(super) fn prepare_power_evaluation_geometry(
        &self,
        input_precision: NumericPrecision,
        shape: Vec<usize>,
        len: usize,
        domain: PowerResultDomain,
    ) -> Result<Option<PreparedPowerEvaluation>> {
        if len == 0 {
            return Ok(None);
        }
        let (width, storage) = match domain {
            PowerResultDomain::Real => (1, GpuTensorStorage::Real),
            PowerResultDomain::Complex => (2, GpuTensorStorage::ComplexInterleaved),
        };
        let words = len
            .checked_mul(width)
            .ok_or_else(|| anyhow!("POWER evaluation word count overflow"))?;
        self.require_ieee_words(input_precision)?;
        let representation = FusedElementwiseRepresentation {
            input_precision,
            output_precision: input_precision,
            output_storage: storage,
            logical_output: false,
        };
        let shader = shader(input_precision, PowerShader::Evaluate(domain));
        let fusion = FrozenFusionRecipe::plan(
            &shader,
            2,
            &FusedElementwiseWork {
                output_shape: &shape,
                len: words,
                invocation_geometry: ElementwiseInvocationGeometry::OutputLanes,
                representation,
                uniform_abi: Some(FusionUniformAbi::Broadcast),
            },
            self.linear_launch,
            &self.adapter_limits,
        )?;
        Ok(Some(PreparedPowerEvaluation {
            shape,
            words,
            representation,
            shader,
            fusion,
        }))
    }

    pub(crate) fn propose_power_evaluation_exec(
        &self,
        base: &GpuTensorHandle,
        exponent: &GpuTensorHandle,
        domain: PowerResultDomain,
    ) -> ProviderExecutionResult<Option<PowerEvaluationPhysicalProposal>> {
        self.prepare_power_evaluation_exec(base, exponent, domain)?
            .map(|prepared| prepared.physical())
            .transpose()
            .map_err(Into::into)
    }

    pub(crate) fn execute_power_evaluation_recipe_exec(
        &self,
        recipe: &PowerEvaluationRecipe,
    ) -> ProviderExecutionResult<ProviderExecutedRealElementwise> {
        self.execute_power_evaluation_recipe_admitted_exec(recipe, None)
    }

    pub(crate) fn execute_power_evaluation_recipe_admitted_exec(
        &self,
        recipe: &PowerEvaluationRecipe,
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> ProviderExecutionResult<ProviderExecutedRealElementwise> {
        if !recipe.probe().is_owned_by(self) {
            return Err(anyhow!("POWER evaluation provider changed").into());
        }
        let [base, exponent] = recipe.probe().sources();
        let prepared = self
            .prepare_power_evaluation_exec(base, exponent, recipe.domain())?
            .ok_or_else(|| anyhow!("POWER evaluation physical route unavailable"))?;
        if prepared.physical()? != *recipe.physical() {
            return Err(anyhow!("POWER evaluation physical recipe changed").into());
        }
        let (output, observed, _) = self
            .fused_elementwise_prepared_probe_with_telemetry_admitted_exec(
                &prepared.shader,
                &[base.clone(), exponent.clone()],
                FusedElementwiseWork {
                    output_shape: &prepared.shape,
                    len: prepared.words,
                    invocation_geometry: ElementwiseInvocationGeometry::OutputLanes,
                    representation: prepared.representation,
                    uniform_abi: Some(FusionUniformAbi::Broadcast),
                },
                &prepared.fusion,
                admission,
            )
            .map_err(|cause| ProviderExecutionError::from(cause).with_unproven_effects())?;
        Ok(ProviderExecutedRealElementwise { output, observed })
    }
}

impl PreparedPowerEvaluation {
    pub(super) fn physical(&self) -> Result<PowerEvaluationPhysicalProposal> {
        let resources = self.fusion.expected();
        let launch = self.fusion.launch();
        let pipeline = self.fusion.pipeline();
        let element_type = match (
            self.representation.output_precision,
            self.representation.output_storage,
        ) {
            (NumericPrecision::F32, GpuTensorStorage::Real) => ProviderElementType::F32,
            (NumericPrecision::F64, GpuTensorStorage::Real) => ProviderElementType::F64,
            (NumericPrecision::F32, GpuTensorStorage::ComplexInterleaved) => {
                ProviderElementType::ComplexF32
            }
            (NumericPrecision::F64, GpuTensorStorage::ComplexInterleaved) => {
                ProviderElementType::ComplexF64
            }
        };
        Ok(PowerEvaluationPhysicalProposal {
            output: ProviderRepresentation {
                element_type,
                storage: match self.representation.output_storage {
                    GpuTensorStorage::Real => ProviderStorage::DenseReal,
                    GpuTensorStorage::ComplexInterleaved => {
                        ProviderStorage::DenseComplexInterleaved
                    }
                },
                layout: ProviderLayout::ColumnMajorContiguous,
                shape: self
                    .shape
                    .iter()
                    .copied()
                    .map(u64::try_from)
                    .collect::<Result<Vec<_>, _>>()?,
                residency: ProviderResidency::Device,
            },
            work: ProviderRealElementwisePhysicalProposal {
                output_route: ProviderRealElementwiseOutputRoute::FreshAllocation,
                output_buffer_bytes: resources
                    .output_buffer_bytes
                    .ok_or_else(|| anyhow!("POWER evaluation output extent missing"))?,
                workgroup_size: launch.workgroup(),
                chunk_capacity: u64::try_from(
                    launch
                        .chunk_capacity()
                        .ok_or_else(|| anyhow!("POWER evaluation launch capacity overflow"))?,
                )?,
                uniform_allocations: resources.uniform_allocations,
                uniform_buffer_bytes: resources.uniform_buffer_bytes,
                dispatches: resources.dispatches,
                queue_submissions: resources.queue_submissions,
                pipeline_key: Some(pipeline.key()),
            },
            shader_identity: runmat_execution::Digest::sha256_serialized(&(
                "wgpu-power-evaluation-shader-v1",
                &pipeline.shader,
                &pipeline.layout,
                pipeline.workgroup,
            ))?,
        })
    }
}
