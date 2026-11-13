//! Held source-bound ordinary scalar/cast physical execution. No selected
//! arithmetic or whole-image placement authority is minted by this module.

use super::*;
use crate::fusion::image_reduction::{ImageSelectedRealCall, ImageSelectedRealCalls};
#[cfg(test)]
use runmat_accelerate_api::ProviderRealElementwiseObserved;
use runmat_accelerate_api::{
    scalar_program::ScalarLiteral, ProviderElementType, ProviderImageOutputCall,
    ProviderImageOutputOperand, ProviderRealElementwiseOutputRoute, ProviderRealElementwiseRecipe,
    ProviderRealElementwiseRequest, ProviderRepresentation,
};
#[cfg(test)]
use runmat_builtins::BuiltinBindingIdentity;
#[cfg(test)]
use runmat_execution::Digest;
use std::cell::Cell;
#[cfg(test)]
use std::cell::RefCell;

#[cfg(test)]
#[derive(Clone)]
pub(super) struct SelectedRealReceipt {
    pub selected_identity: Digest,
    pub source_input: u64,
    pub source_output: u64,
    pub source_control: Option<u64>,
    pub selected_binding: Option<BuiltinBindingIdentity>,
    pub recipe_identity: Digest,
    pub observed: ProviderRealElementwiseObserved,
}

pub(super) struct PlannedCall {
    source: ImageSelectedRealCall,
    recipe: ProviderRealElementwiseRecipe,
    #[cfg(test)]
    identity: Digest,
}

pub(super) struct SelectedRealPlans {
    #[cfg(test)]
    semantic_identity: Digest,
    calls: Vec<PlannedCall>,
    next: Cell<usize>,
    #[cfg(test)]
    observed: RefCell<Vec<SelectedRealReceipt>>,
}

impl SelectedRealPlans {
    pub(super) fn registered_output_count(&self) -> usize {
        self.calls
            .iter()
            .filter(|call| {
                call.recipe.physical().output_route
                    == ProviderRealElementwiseOutputRoute::FreshAllocation
            })
            .count()
    }

    pub(super) fn alias_view_count(&self) -> usize {
        self.calls
            .iter()
            .filter(|call| {
                call.recipe.physical().output_route
                    == ProviderRealElementwiseOutputRoute::SameClassFloatingCastAlias
            })
            .count()
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

pub(super) struct SelectedRealOutput {
    pub output: GpuTensorHandle,
    /// Minted only by a successfully completed and observed shared recipe.
    pub route: ProviderRealElementwiseOutputRoute,
}

fn element(precision: ProviderPrecision) -> ProviderElementType {
    match precision {
        ProviderPrecision::F32 => ProviderElementType::F32,
        ProviderPrecision::F64 => ProviderElementType::F64,
    }
}

fn matches_fact(representation: &ProviderRepresentation, source: &ImageSelectedRealCall) -> bool {
    representation.element_type == element(source.input_fact().precision())
        && representation.shape == source.input_fact().shape()
}

pub(super) fn plan(
    provider: &'static dyn AccelProvider,
    source: &ImageSelectedRealCalls,
    binary: &selected_binary::SelectedBinaryPlans,
    powers: &power::ImagePowerPlans,
    program: &ProviderImageStageWorkload,
) -> ProviderExecutionResult<Option<SelectedRealPlans>> {
    if source.semantic_identity() != binary.semantic_identity() {
        return Ok(None);
    }
    let Some(output_program) = program.output_program.as_ref() else {
        return Ok(None);
    };
    let Some((sub, rest)) = output_program.calls.split_first() else {
        return Ok(None);
    };
    let Some((div, suffix)) = rest.split_first() else {
        return Ok(None);
    };
    let Some(selected_powers) = program.selected_powers.as_ref() else {
        return Ok(None);
    };
    if !program.validate()
        || !binary.matches_output(selected_binary::BinaryStage::OutputSub, sub)
        || !binary.matches_output(selected_binary::BinaryStage::OutputDiv, div)
        || sub.checkpoint
        || div.checkpoint
    {
        return Ok(None);
    }
    let (mut representation, mut bytes) = binary.output_div_carrier();
    let mut current_source = div.source_output;
    let mut planned = Vec::with_capacity(source.calls().len());
    let mut power_seen = false;
    for wire in suffix {
        if wire.hook == runmat_accelerate_api::ProviderImageOutputHook::ElemPow {
            let expected_class = match selected_powers.final_power.output_class {
                runmat_types::NumericClass::Single => ProviderPrecision::F32,
                runmat_types::NumericClass::Double => ProviderPrecision::F64,
                _ => return Ok(None),
            };
            let Some(output_bytes) = powers.final_power.output_requested_bytes() else {
                return Ok(None);
            };
            if power_seen
                || wire.source_output != u64::from(selected_powers.final_power.output)
                || wire.operands.first().copied()
                    != Some(ProviderImageOutputOperand::Produced(current_source))
                || wire.output_precision != expected_class
            {
                return Ok(None);
            }
            representation = ProviderRepresentation {
                element_type: element(wire.output_precision),
                storage: runmat_accelerate_api::ProviderStorage::DenseReal,
                layout: runmat_accelerate_api::ProviderLayout::ColumnMajorContiguous,
                shape: wire.output_shape.to_vec(),
                residency: runmat_accelerate_api::ProviderResidency::Device,
            };
            if representation
                .checked_byte_len()
                .is_none_or(|minimum| minimum > output_bytes)
            {
                return Ok(None);
            }
            bytes = output_bytes;
            current_source = wire.source_output;
            power_seen = true;
            continue;
        }
        let Some(call) = source.calls().get(planned.len()) else {
            return Ok(None);
        };
        if call.input() != current_source || !matches_fact(&representation, call) {
            return Ok(None);
        }
        let Some(operation) = call.operation_for(&representation) else {
            return Ok(None);
        };
        let Some(recipe) = ProviderRealElementwiseRecipe::plan(
            provider,
            ProviderRealElementwiseRequest {
                operation,
                source: representation,
                source_buffer_bytes: bytes,
            },
        )?
        else {
            return Ok(None);
        };
        if recipe.output().element_type != element(call.output_fact().precision())
            || recipe.output().shape != call.output_shape()
        {
            return Ok(None);
        }
        representation = recipe.output().clone();
        bytes = recipe.physical().output_buffer_bytes;
        current_source = call.output();
        #[cfg(test)]
        let identity = recipe.identity().map_err(ProviderExecutionError::from)?;
        planned.push(PlannedCall {
            source: call.clone(),
            recipe,
            #[cfg(test)]
            identity,
        });
    }
    if !power_seen || planned.len() != source.calls().len() {
        return Ok(None);
    }
    let selected = SelectedRealPlans {
        #[cfg(test)]
        semantic_identity: source.semantic_identity(),
        calls: planned,
        next: Cell::new(0),
        #[cfg(test)]
        observed: RefCell::new(Vec::new()),
    };
    if suffix
        .iter()
        .filter(|call| call.hook != runmat_accelerate_api::ProviderImageOutputHook::ElemPow)
        .enumerate()
        .any(|(index, call)| !selected.matches_output(index, call))
    {
        return Ok(None);
    }
    Ok(Some(selected))
}

impl SelectedRealPlans {
    pub(super) fn matches_output(&self, index: usize, wire: &ProviderImageOutputCall) -> bool {
        let Some(source) = self.calls.get(index).map(|call| &call.source) else {
            return false;
        };
        let control = source.control().map(|control| match control {
            ScalarLiteral::F32Bits(bits) => ProviderImageOutputOperand::ScalarF32Bits(bits),
            ScalarLiteral::F64Bits(bits) => ProviderImageOutputOperand::ScalarF64Bits(bits),
        });
        wire.hook == source.hook()
            && wire.source_output == source.output()
            && wire.output_shape == source.output_shape()
            && wire.output_precision == source.output_fact().precision()
            && wire.operands.first().copied()
                == Some(ProviderImageOutputOperand::Produced(source.input()))
            && wire.operands.get(1).copied() == control
    }

    pub(super) async fn execute(
        &self,
        wire: &ProviderImageOutputCall,
        source: &GpuTensorHandle,
        protected: &[&GpuTensorHandle],
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> ProviderExecutionResult<SelectedRealOutput> {
        let index = self.next.get();
        if !self.matches_output(index, wire) {
            return Err(ProviderExecutionError::from(anyhow!(
                "selected real output call order or source changed"
            ))
            .after_effect());
        }
        let planned = &self.calls[index];
        // Begin before awaiting: cancelled/failed execution cannot replay a
        // provider operation whose submission status is unknown.
        self.next.set(index + 1);
        let executed = planned
            .recipe
            .execute_protected_admitted(source, protected, admission)
            .await
            .map_err(ProviderExecutionError::after_effect)?;
        #[cfg(test)]
        self.observed.borrow_mut().push(SelectedRealReceipt {
            selected_identity: self.semantic_identity,
            source_input: planned.source.input(),
            source_output: planned.source.output(),
            source_control: planned.source.control_source(),
            selected_binding: planned.source.binding(),
            recipe_identity: planned.identity,
            observed: executed.observed,
        });
        Ok(SelectedRealOutput {
            output: executed.output,
            route: planned.recipe.physical().output_route,
        })
    }

    #[cfg(test)]
    pub(super) fn receipts(&self) -> Vec<SelectedRealReceipt> {
        self.observed.borrow().clone()
    }

    #[cfg(test)]
    pub(super) fn recipe_identities(&self) -> [Digest; 5] {
        self.calls
            .iter()
            .map(|call| call.identity)
            .collect::<Vec<_>>()
            .try_into()
            .expect("five-call selected source fixture")
    }
}
