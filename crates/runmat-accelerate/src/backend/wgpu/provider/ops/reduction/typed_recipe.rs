//! Executable typed reduction route for prepared resident inputs.
use super::*;
use crate::backend::wgpu::dispatch::observation::{
    SubmittedReductionInput, SubmittedReductionInputs, SubmittedReductionResources,
};
use crate::backend::wgpu::residency::BufferUsageClass;
use runmat_accelerate_api::{
    reduction::{ReductionStrategy, ReductionSubmission},
    NumericElementType, ProviderExecutionError, ProviderExecutionResult,
    ProviderShaderRequirements,
};

#[cfg(all(test, not(target_arch = "wasm32")))]
mod lifecycle_tests;
mod phase;
pub(super) mod preflight;
mod publication;
mod stage;
mod tag;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

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
            // The original Runtime already reserved this registration's exact
            // record/lease/view headroom before input or output effects. Move
            // the exact-rank shape into that same admitted record below.
            let output_shape = runmat_value::exact_capacity_vec_from_slice(&d.output_shape)?;
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
                    .all(|b| !std::ptr::eq(b.buffer.as_ref(), output.as_ref())),
                "typed reduction output aliases input"
            );
            let published = self.register_reduction_output(
                output.clone(),
                output_shape,
                output_len,
                output_type,
                submission.output_registration_metadata(),
            )?;
            let submission_id = match selection.strategy() {
                ReductionStrategy::SinglePass => {
                    let id = self.emit_reduction_stage(
                        emission,
                        pending,
                        StageInvocation {
                            recipe,
                            stage: Stage::Single,
                            host_metadata: submission.host_metadata(),
                            resources: SubmittedReductionResources::producer(
                                inputs,
                                &output,
                                Some(&published),
                            )?,
                        },
                        &mut submit_attempted,
                        observer,
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
                                .all(|b| !std::ptr::eq(b.buffer.as_ref(), partial.as_ref())),
                        "typed reduction partial aliases live input or output"
                    );
                    // Pass1 owns the original input vector and the exact raw
                    // partial through its real completion. Pass2 separately
                    // retains that partial and final output lease; no temporary
                    // tensor lease or duplicate allocation charge is invented.
                    let first_id = self.emit_reduction_stage(
                        emission,
                        pending,
                        StageInvocation {
                            recipe,
                            stage: Stage::First,
                            host_metadata: submission.host_metadata(),
                            resources: SubmittedReductionResources::producer(
                                inputs, &partial, None,
                            )?,
                        },
                        &mut submit_attempted,
                        observer,
                    )?;
                    observer.submitted(Stage::First, first_id)?;
                    let final_id = self.emit_reduction_stage(
                        emission,
                        pending,
                        StageInvocation {
                            recipe,
                            stage: Stage::Final,
                            host_metadata: submission.host_metadata(),
                            resources: SubmittedReductionResources::final_stage(
                                &partial, &output, &published,
                            )?,
                        },
                        &mut submit_attempted,
                        observer,
                    )?;
                    observer.submitted(Stage::Final, final_id)?;
                    final_id
                }
            };
            Ok((published, submission_id))
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
        self.publish_reduction_output(output, submission_id)
    }
}
