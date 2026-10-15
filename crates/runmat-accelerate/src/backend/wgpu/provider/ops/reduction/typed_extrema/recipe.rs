//! Execute the admitted global tree through the same physical emitter.
use super::*;
use runmat_accelerate_api::{
    ProviderExecutedGlobalExtrema, ProviderExtremaKind, ProviderGlobalExtremaObserved,
    ProviderGlobalExtremaRecipe,
};
use std::cell::RefCell;

impl WgpuProvider {
    pub(crate) async fn execute_global_extrema_recipe_exec(
        &self,
        input: &GpuTensorHandle,
        recipe: &ProviderGlobalExtremaRecipe,
    ) -> ProviderExecutionResult<ProviderExecutedGlobalExtrema> {
        if !recipe.is_owned_by(self)
            || recipe.source().view_identity() != input.view_identity()
            || self.propose_global_extrema_exec(recipe.request())?.as_ref()
                != Some(recipe.physical())
        {
            return Err(ProviderExecutionError::from(anyhow!(
                "global extrema recipe is not current for this source"
            )));
        }
        let frame = ExtremaInputFrame::global(self, input)?;
        let selected = if let Some(pipeline) = frame.recipe.as_ref() {
            Some(
                self.prepare_validated_reduction_extrema_pipeline(pipeline, frame.kind)
                    .await?,
            )
        } else {
            None
        };
        frame
            .revalidate(self, input)
            .map_err(ProviderExecutionError::from)?;
        if self.propose_global_extrema_exec(recipe.request())?.as_ref() != Some(recipe.physical()) {
            return Err(ProviderExecutionError::from(anyhow!(
                "global extrema physical route changed during preparation"
            )));
        }
        if let (Some(pipeline), Some(selected)) = (frame.recipe.as_ref(), selected.as_ref()) {
            selected
                .require_reduction_extrema(pipeline, frame.kind)
                .map_err(ProviderExecutionError::from)?;
        }
        let allocated = Cell::new(false);
        let observed = RefCell::new(ProviderGlobalExtremaObserved::default());
        let output = self
            .reduce_global_recorded_exec(
                input,
                match recipe.request().kind {
                    ProviderExtremaKind::Minimum => {
                        crate::backend::wgpu::types::GlobalReduceOp::Min
                    }
                    ProviderExtremaKind::Maximum => {
                        crate::backend::wgpu::types::GlobalReduceOp::Max
                    }
                },
                super::super::phase_context::SelectedEmission {
                    selected: selected.as_deref(),
                    effect: Some(&allocated),
                    after_allocation: None,
                },
                Some(&observed),
            )
            .map_err(|cause| terminal_error(cause, allocated.get()))?;
        Ok(ProviderExecutedGlobalExtrema {
            output,
            observed: observed.into_inner(),
        })
    }
}
