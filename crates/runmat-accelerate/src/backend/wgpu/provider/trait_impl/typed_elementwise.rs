//! Explicit ordinary floating-data ABI; no source-name or shader-text inference.
use super::*;
use runmat_accelerate_api::{
    ProviderElementType, ProviderExecutionError, ProviderExecutionResult,
    ProviderFusedElementwiseRequest,
};

impl WgpuProvider {
    fn validate_typed_elementwise(
        &self,
        request: &ProviderFusedElementwiseRequest<'_>,
    ) -> ProviderExecutionResult<()> {
        if request.len == 0
            || !runmat_accelerate_api::AccelProvider::supports_fused_elementwise_abi(
                self,
                request.abi,
            )
            || request
                .inputs
                .iter()
                .any(runmat_accelerate_api::handle_is_logical)
            || request
                .output_classes
                .iter()
                .any(|class| *class != ProviderElementType::from(request.abi.precision()))
        {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "typed fusion requires supported uniform real floating data",
            )
            .into());
        }
        self.require_selected_shader_requirements(request.requirements)?;
        Ok(())
    }
    pub(super) async fn execute_typed_elementwise(
        &self,
        request: ProviderFusedElementwiseRequest<'_>,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        self.validate_typed_elementwise(&request)?;
        let shader = request.shader;
        let inputs = request.inputs;
        let shape = request.output_shape;
        let len = request.len;
        let frame = self
            .plan_single_fusion(request)
            .map_err(ProviderExecutionError::from)?;
        let validated = self
            .prepare_validated_fusion_pipeline(
                frame.recipe.pipeline(),
                inputs.len(),
                super::super::pipeline_recipe::FusionPipelineLayout::Single,
            )
            .await?;
        let observed = ElementwiseKernelObservation::begin(len);
        let output = self
            .fused_elementwise_validated_with_telemetry_exec(
                shader, inputs, shape, len, &frame, &validated,
            )
            .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
        if let Some(observed) = observed {
            observed.submitted(self.device_id_exec());
        }
        Ok(output)
    }
    pub(super) async fn execute_typed_elementwise_multi(
        &self,
        request: ProviderFusedElementwiseRequest<'_>,
    ) -> ProviderExecutionResult<Vec<GpuTensorHandle>> {
        self.validate_typed_elementwise(&request)?;
        let shader = request.shader;
        let inputs = request.inputs;
        let outputs = request.output_classes.len();
        let len = request.len;
        let frame = self
            .plan_multi_fusion(request)
            .map_err(ProviderExecutionError::from)?;
        let validated = self
            .prepare_validated_fusion_pipeline(
                &frame.recipe,
                inputs.len(),
                super::super::pipeline_recipe::FusionPipelineLayout::Multiple(outputs),
            )
            .await?;
        let observed = ElementwiseKernelObservation::begin(len);
        let outputs = self
            .fused_elementwise_multi_validated_with_telemetry_exec(
                shader, inputs, &frame, &validated,
            )
            .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
        if let Some(observed) = observed {
            observed.submitted(self.device_id_exec());
        }
        Ok(outputs)
    }
}
