//! Opaque selected stage continuation. Only a source-bound physical candidate
//! chosen by the canonical placement owner can reach its first provider hook.

use super::*;
use crate::fusion::image_reduction::{ImageSourcePublication, ImageSourceTranscript};
use crate::fusion::{FusionGroupPlan, FusionPattern};
use crate::graph::AccelGraph;
use crate::placement::LocalPlacementOutcome;
use std::sync::Arc;

pub struct ImageVmCheckpoint {
    pub source_output: u64,
    pub handle: GpuTensorHandle,
}

pub struct ImageVmStages {
    prepared: PreparedImageStages,
}

/// Source-bound inputs and invocation rights for one selected VM region.
pub struct ImageVmPreparation<'a> {
    pub graph: &'a AccelGraph,
    pub plan: &'a FusionGroupPlan,
    pub source: Arc<ImageSourceTranscript>,
    pub inputs: &'a [Value],
    pub publications: &'a [ImageSourcePublication],
    pub runtime: &'a runmat_runtime::context::RuntimeContext,
    pub placement: Option<crate::placement::PlacementCorrelationId>,
    pub decline: Option<&'a runmat_runtime::context::VmFusionDeclineRight>,
}

impl ImageVmStages {
    /// A structural miss declines before effects; an operational canonical
    /// resolver or physical-owner failure retains its typed cause.
    pub async fn prepare(request: ImageVmPreparation<'_>) -> ProviderExecutionResult<Option<Self>> {
        let ImageVmPreparation {
            graph,
            plan,
            source,
            inputs,
            publications,
            runtime,
            placement,
            decline,
        } = request;
        let Some(FusionPattern::ImageNormalize(pattern)) = plan.group.pattern.as_ref() else {
            return Ok(None);
        };
        let Some(position) = plan.inputs.iter().position(|id| *id == pattern.input) else {
            return Ok(None);
        };
        let Some(Value::GpuTensor(input)) = inputs.get(position) else {
            return Ok(None);
        };
        let Some(provider) = exact_provider_for_handle(input) else {
            return Err(ProviderExecutionError::from(anyhow!(
                "selected image input has no registered exact provider"
            )));
        };
        let invocation = match ImageSourceInvocation::bind_selected(
            source,
            graph,
            plan,
            inputs,
            ProviderPrecision::F32,
            placement,
        )
        .await
        {
            Ok(invocation) => invocation,
            Err(
                crate::fusion::image_reduction::ImageSourceAdmissionFailure::CanonicalExecution(
                    error,
                ),
            ) => return Err(error),
            Err(_) => return Ok(None),
        };
        let Some(invocation) = invocation.bind_publications(graph, publications) else {
            return Ok(None);
        };
        let Some(prepared) = PreparedImageStages::prepare(provider, input, &invocation)? else {
            return Ok(None);
        };
        match prepared.select(&inputs[position], runtime, decline)? {
            LocalPlacementOutcome::Selected {
                selected,
                provider_index: Some(0),
                ..
            } if selected.kind.is_provider() => Ok(Some(Self { prepared })),
            LocalPlacementOutcome::ProviderRejected { .. } => Ok(None),
            _ => Err(ProviderExecutionError::from(anyhow!(
                "selected image placement returned a different physical route"
            ))),
        }
    }

    fn revalidate(&self) -> ProviderExecutionResult<()> {
        if self.prepared.selected_owner_is_live()? {
            Ok(())
        } else {
            Err(ProviderExecutionError::from(anyhow!(
                "selected image provider owner or device liveness changed"
            ))
            .after_effect())
        }
    }

    pub async fn mean(&mut self) -> ProviderExecutionResult<GpuTensorHandle> {
        if !self
            .prepared
            .matches_pre_effect_snapshot(self.prepared.session.provider)?
        {
            return Err(ProviderExecutionError::from(anyhow!(
                "selected image provider resource snapshot changed before dispatch"
            )));
        }
        self.prepared
            .session_mut()
            .mean()
            .await
            .map(|value| value.0)
    }

    pub async fn sigma(&mut self) -> ProviderExecutionResult<GpuTensorHandle> {
        self.revalidate()
            .map_err(ProviderExecutionError::after_effect)?;
        self.prepared
            .session_mut()
            .sigma()
            .await
            .map(|value| value.0)
    }

    pub async fn next_output_checkpoint(&mut self) -> ProviderExecutionResult<ImageVmCheckpoint> {
        self.revalidate()
            .map_err(ProviderExecutionError::after_effect)?;
        self.prepared
            .session_mut()
            .next_output_checkpoint()
            .await
            .map(|checkpoint| ImageVmCheckpoint {
                source_output: checkpoint.source_output,
                handle: checkpoint.handle,
            })
    }

    pub fn acknowledge_output_checkpoint(
        &mut self,
        source_output: u64,
    ) -> ProviderExecutionResult<()> {
        self.prepared
            .session_mut()
            .acknowledge_output_checkpoint(source_output)
    }
}
