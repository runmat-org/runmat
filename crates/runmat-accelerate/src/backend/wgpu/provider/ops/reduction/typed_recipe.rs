//! Executable typed reduction route for prepared resident inputs.
use super::*;
use crate::backend::wgpu::residency::BufferUsageClass;
use runmat_accelerate_api::{
    reduction::{ReductionStrategy, ReductionSubmission},
    NumericElementType, ProviderExecutionError, ProviderExecutionResult,
    ProviderShaderRequirements,
};

mod phase;
pub(super) mod preflight;
mod publication;
mod stage;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod lifecycle_tests;

use phase::{NoopObserver, ReductionObserver};
use preflight::physical;
use stage::{Stage, StageInvocation};

impl WgpuProvider {
    pub(crate) fn submit_reduction_recipe_exec(
        &self,
        submission: &ReductionSubmission<'_>,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        // This hook owns one checked fused-reduction submission. A preceding
        // upload or nested preparation cannot certify its selected kernel.
        let operation = runmat_accelerate_api::ProviderOperationIdentity::new(
            crate::placement::fusion_operation_token(&crate::fusion::FusionKind::Reduction),
        );
        let observed = runmat_accelerate_api::observation::current_provider_observation()
            .filter(|recorder| recorder.expects_operation(&operation))
            .map(|recorder| {
                let checkpoint = recorder.submission_checkpoint();
                (recorder, checkpoint)
            });
        let result = self.submit_reduction_recipe_observed(submission, &mut NoopObserver)?;
        if let Some((recorder, checkpoint)) = observed {
            recorder.confirm_submitted_provider_execution_after(
                self.device_id(),
                &operation,
                checkpoint,
            );
        }
        Ok(result)
    }

    fn submit_reduction_recipe_observed<O: ReductionObserver>(
        &self,
        submission: &ReductionSubmission<'_>,
        observer: &mut O,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        let selection = submission.selection();
        let recipe = selection.recipe();
        let d = recipe.descriptor();
        let inputs = self.admit_reduction_recipe(submission)?;
        let output_len = d.geometry.slices as usize;
        let output_type = physical(d.output_precision);
        let mut output_owned = false;
        let mut submit_attempted = false;
        let result = self.with_emission(|emission, pending| -> Result<_> {
            let output = self.create_storage_buffer_checked_with_element_size(
                emission,
                output_len,
                "typed-reduction-output",
                BufferUsageClass::FusionOut,
                output_type.element_size(),
            )?;
            output_owned = true;
            observer.output_acquired(&output)?;
            ensure!(
                output.size() == d.work.output_requested_bytes,
                "typed reduction output extent mismatch"
            );
            ensure!(
                inputs
                    .iter()
                    .all(|b| !std::ptr::eq(b.as_ref(), output.as_ref())),
                "typed reduction output aliases input"
            );
            let submission_id = match selection.strategy() {
                ReductionStrategy::SinglePass => {
                    let id = self.emit_reduction_stage(
                        emission,
                        pending,
                        StageInvocation {
                            recipe,
                            stage: Stage::Single,
                            inputs: &inputs,
                            output: &output,
                        },
                        &mut submit_attempted,
                    )?;
                    observer.submitted(Stage::Single, id)?;
                    id
                }
                ReductionStrategy::TwoPass => {
                    let partial = self.create_storage_buffer_checked_with_element_size(
                        emission,
                        d.geometry.partial_elements as usize,
                        "typed-reduction-partial",
                        BufferUsageClass::FusionOut,
                        d.partial.stride as usize,
                    )?;
                    observer.partial_acquired(&partial)?;
                    ensure!(
                        partial.size() == d.work.partial_requested_bytes,
                        "typed reduction partial extent mismatch"
                    );
                    ensure!(
                        !std::ptr::eq(partial.as_ref(), output.as_ref())
                            && inputs
                                .iter()
                                .all(|b| !std::ptr::eq(b.as_ref(), partial.as_ref())),
                        "typed reduction partial aliases live input or output"
                    );
                    // The same admitted producer inputs feed pass 1. The final
                    // stage sees only the private typed partial. Keep that owner
                    // alive through both queue submissions; WGPU retains queued
                    // resource references after this closure drops it.
                    let first_id = self.emit_reduction_stage(
                        emission,
                        pending,
                        StageInvocation {
                            recipe,
                            stage: Stage::First,
                            inputs: &inputs,
                            output: &partial,
                        },
                        &mut submit_attempted,
                    )?;
                    observer.submitted(Stage::First, first_id)?;
                    let final_id = self.emit_reduction_stage(
                        emission,
                        pending,
                        StageInvocation {
                            recipe,
                            stage: Stage::Final,
                            inputs: std::slice::from_ref(&partial),
                            output: &output,
                        },
                        &mut submit_attempted,
                    )?;
                    observer.submitted(Stage::Final, final_id)?;
                    final_id
                }
            };
            Ok((output, submission_id))
        });
        let (output, submission_id) = match result {
            Ok(Ok(output)) => output,
            Ok(Err(error)) | Err(error) => {
                let error = ProviderExecutionError::from(error);
                return Err(if output_owned || submit_attempted {
                    error.after_effect()
                } else {
                    error.with_unproven_effects()
                });
            }
        };
        self.publish_reduction_output(
            output,
            d.output_shape.clone(),
            output_len,
            output_type,
            submission_id,
        )
    }
}
