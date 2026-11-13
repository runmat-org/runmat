//! Held source-to-provider join for the three ordinary real binary calls.
//! Every recipe is derived before the first stage effect. It does not grant
//! whole-image placement, numerical equivalence, or VM publication authority.

use super::*;
use crate::fusion::image_reduction::{ImageSelectedBinaryCall, ImageSelectedBinaryCalls};
#[cfg(test)]
use runmat_accelerate_api::ProviderRealBinaryObserved;
use runmat_accelerate_api::{
    handle_precision, ProviderElementType, ProviderImageOutputCall, ProviderImageOutputHook,
    ProviderImageOutputOperand, ProviderRealBinaryInput, ProviderRealBinaryRecipe,
    ProviderRealBinaryRequest, ProviderRepresentation,
};
use runmat_execution::Digest;
use std::cell::Cell;
#[cfg(test)]
use std::cell::RefCell;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BinaryStage {
    SigmaCenter,
    OutputSub,
    OutputDiv,
}

#[cfg(test)]
#[derive(Clone)]
pub(super) struct SelectedBinaryReceipt {
    pub selected_identity: Digest,
    pub source_inputs: [u64; 2],
    pub source_output: u64,
    pub recipe_identity: Digest,
    pub observed: ProviderRealBinaryObserved,
}

struct PlannedCall {
    source: ImageSelectedBinaryCall,
    recipe: ProviderRealBinaryRecipe,
    #[cfg(test)]
    identity: Digest,
}

pub(super) struct SelectedBinaryPlans {
    semantic_identity: Digest,
    calls: [PlannedCall; 3],
    next: Cell<usize>,
    #[cfg(test)]
    observed: RefCell<Vec<SelectedBinaryReceipt>>,
}

impl SelectedBinaryPlans {
    pub(super) fn registered_output_count(&self) -> usize {
        self.calls.len()
    }
    pub(super) fn physical_upper_bound(
        &self,
    ) -> Option<super::resources::ImageStagePhysicalUpperBound> {
        let mut bound = super::resources::ImageStagePhysicalUpperBound::default();
        for call in &self.calls {
            let physical = call.recipe.physical();
            bound.add(
                physical.requested_incremental_peak_bytes()?,
                physical.dispatches,
                Some(physical.queue_submissions),
            )?;
        }
        Some(bound)
    }
}

fn operand(representation: ProviderRepresentation, buffer_bytes: u64) -> ProviderRealBinaryInput {
    ProviderRealBinaryInput {
        representation,
        buffer_bytes,
    }
}

fn plan_call(
    provider: &'static dyn AccelProvider,
    source: &ImageSelectedBinaryCall,
    lhs: ProviderRealBinaryInput,
    rhs: ProviderRealBinaryInput,
) -> ProviderExecutionResult<Option<PlannedCall>> {
    let matches_fact =
        |input: &ProviderRealBinaryInput, precision: ProviderPrecision, shape: &[u64]| {
            input.representation.element_type
                == match precision {
                    ProviderPrecision::F32 => ProviderElementType::F32,
                    ProviderPrecision::F64 => ProviderElementType::F64,
                }
                && input.representation.shape.as_slice() == shape
        };
    let input_facts = source.input_facts();
    let lhs_fact = &input_facts[0];
    let rhs_fact = &input_facts[1];
    if !matches_fact(&lhs, lhs_fact.precision(), lhs_fact.shape())
        || !matches_fact(&rhs, rhs_fact.precision(), rhs_fact.shape())
    {
        return Ok(None);
    }
    let Some(recipe) = ProviderRealBinaryRecipe::plan(
        provider,
        ProviderRealBinaryRequest {
            operation: source.operation(),
            lhs,
            rhs,
        },
    )?
    else {
        return Ok(None);
    };
    let output = &recipe.physical().output;
    let expected_type = match source.output_precision() {
        ProviderPrecision::F32 => ProviderElementType::F32,
        ProviderPrecision::F64 => ProviderElementType::F64,
    };
    if output.element_type != expected_type || output.shape.as_slice() != source.output_shape() {
        return Ok(None);
    }
    #[cfg(test)]
    let identity = recipe.identity().map_err(ProviderExecutionError::from)?;
    Ok(Some(PlannedCall {
        source: source.clone(),
        recipe,
        #[cfg(test)]
        identity,
    }))
}

pub(super) fn plan_source(
    provider: &'static dyn AccelProvider,
    input_rep: ProviderRepresentation,
    input_bytes: u64,
    source: &ImageSelectedBinaryCalls,
    powers: &power::ImagePowerPlans,
    means: &ordered_mean::ImageOrderedMeanPlans,
    program: &ProviderImageStageWorkload,
) -> ProviderExecutionResult<Option<SelectedBinaryPlans>> {
    let image = operand(input_rep, input_bytes);
    let Some((mean_rep, mean_bytes)) = means.public_mean_carrier() else {
        return Ok(None);
    };
    let Some((sigma_rep, sigma_bytes)) = means.public_sigma_carrier() else {
        return Ok(None);
    };
    let Some(center) = plan_call(
        provider,
        source.sigma_center(),
        image.clone(),
        operand(mean_rep.clone(), mean_bytes),
    )?
    else {
        return Ok(None);
    };
    if center.recipe.physical().output != *powers.square.resident_representation() {
        return Ok(None);
    }
    let Some(output_sub) = plan_call(
        provider,
        source.output_sub(),
        image.clone(),
        operand(mean_rep, mean_bytes),
    )?
    else {
        return Ok(None);
    };
    let Some(output_div) = plan_call(
        provider,
        source.output_div(),
        operand(
            output_sub.recipe.physical().output.clone(),
            output_sub.recipe.physical().output_buffer_bytes,
        ),
        operand(sigma_rep, sigma_bytes),
    )?
    else {
        return Ok(None);
    };
    let Some(output_program) = program.output_program.as_ref() else {
        return Ok(None);
    };
    let suffix = output_program
        .calls
        .iter()
        .filter(|call| {
            matches!(
                call.hook,
                ProviderImageOutputHook::ElemSub | ProviderImageOutputHook::ElemDiv,
            )
        })
        .collect::<Vec<_>>();
    if suffix.len() != 2
        || suffix[0].source_output != output_sub.source.output()
        || suffix[1].source_output != output_div.source.output()
        || suffix[0].operands.as_slice()
            != [
                ProviderImageOutputOperand::Input,
                ProviderImageOutputOperand::Mean,
            ]
        || suffix[1].operands.as_slice()
            != [
                ProviderImageOutputOperand::Produced(output_sub.source.output()),
                ProviderImageOutputOperand::Sigma,
            ]
    {
        return Ok(None);
    }
    Ok(Some(SelectedBinaryPlans {
        semantic_identity: source.semantic_identity(),
        calls: [center, output_sub, output_div],
        next: Cell::new(0),
        #[cfg(test)]
        observed: RefCell::new(Vec::new()),
    }))
}

pub(super) fn plan(
    provider: &'static dyn AccelProvider,
    image_input: &GpuTensorHandle,
    source: &ImageSelectedBinaryCalls,
    powers: &power::ImagePowerPlans,
    means: &ordered_mean::ImageOrderedMeanPlans,
    program: &ProviderImageStageWorkload,
) -> ProviderExecutionResult<Option<SelectedBinaryPlans>> {
    let Some(input_precision) = handle_precision(image_input) else {
        return Ok(None);
    };
    let Some(input_rep) = ordered_mean::representation(input_precision, image_input.shape()) else {
        return Ok(None);
    };
    plan_source(
        provider,
        input_rep,
        image_input.physical().requested_bytes(),
        source,
        powers,
        means,
        program,
    )
}

impl SelectedBinaryPlans {
    pub(super) fn semantic_identity(&self) -> Digest {
        self.semantic_identity
    }

    /// Exact provider-owned future output, not an inferred logical byte count.
    pub(super) fn output_div_carrier(&self) -> (ProviderRepresentation, u64) {
        let physical = self.calls[2].recipe.physical();
        (physical.output.clone(), physical.output_buffer_bytes)
    }

    fn index(stage: BinaryStage) -> usize {
        match stage {
            BinaryStage::SigmaCenter => 0,
            BinaryStage::OutputSub => 1,
            BinaryStage::OutputDiv => 2,
        }
    }

    pub(super) fn matches_sigma(&self, program: &ProviderImageStageWorkload) -> bool {
        let Some(moments) = program.moment_program.as_ref() else {
            return false;
        };
        let Some(powers) = program.selected_powers.as_ref() else {
            return false;
        };
        moments.mu.first().is_some_and(|first| {
            moments.mu.last().is_some_and(|last| {
                self.calls[0].source.output() == u64::from(powers.square.base)
                    && self.calls[0].source.inputs() == [first.source_input, last.public_output]
            })
        })
    }

    pub(super) fn matches_output(
        &self,
        stage: BinaryStage,
        wire: &ProviderImageOutputCall,
    ) -> bool {
        let source = &self.calls[Self::index(stage)].source;
        wire.source_output == source.output()
            && wire.output_shape == source.output_shape()
            && wire.output_precision == source.output_precision()
            && wire.operands.as_slice()
                == match stage {
                    BinaryStage::OutputSub => [
                        ProviderImageOutputOperand::Input,
                        ProviderImageOutputOperand::Mean,
                    ],
                    BinaryStage::OutputDiv => [
                        ProviderImageOutputOperand::Produced(self.calls[1].source.output()),
                        ProviderImageOutputOperand::Sigma,
                    ],
                    BinaryStage::SigmaCenter => return false,
                }
            && matches!(
                (stage, wire.hook),
                (BinaryStage::OutputSub, ProviderImageOutputHook::ElemSub)
                    | (BinaryStage::OutputDiv, ProviderImageOutputHook::ElemDiv)
            )
    }

    pub(super) async fn execute(
        &self,
        stage: BinaryStage,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
        protected: &[&GpuTensorHandle],
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        let index = Self::index(stage);
        if self.next.get() != index {
            return Err(ProviderExecutionError::from(anyhow!(
                "selected binary call order or receipt changed"
            ))
            .after_effect());
        }
        let planned = &self.calls[index];
        // Advance before the first await: a dropped future cannot replay an
        // operation whose submission status is unknown.
        self.next.set(index + 1);
        let executed = planned
            .recipe
            .execute_protected_admitted(lhs, rhs, protected, admission)
            .await
            .map_err(ProviderExecutionError::after_effect)?;
        #[cfg(test)]
        self.observed.borrow_mut().push(SelectedBinaryReceipt {
            selected_identity: self.semantic_identity,
            source_inputs: planned.source.inputs(),
            source_output: planned.source.output(),
            recipe_identity: planned.identity,
            observed: executed.observed,
        });
        Ok(executed.output)
    }

    #[cfg(test)]
    pub(super) fn receipts(&self) -> Vec<SelectedBinaryReceipt> {
        self.observed.borrow().clone()
    }

    #[cfg(test)]
    pub(super) fn recipe_identities(&self) -> [Digest; 3] {
        self.calls.each_ref().map(|call| call.identity)
    }

    #[cfg(test)]
    pub(super) fn proposals(
        &self,
    ) -> [runmat_accelerate_api::ProviderRealBinaryPhysicalProposal; 3] {
        self.calls
            .each_ref()
            .map(|call| call.recipe.physical().clone())
    }
}
