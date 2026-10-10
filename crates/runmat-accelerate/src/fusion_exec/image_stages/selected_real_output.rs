//! Held source-bound ordinary scalar/cast physical execution. No selected
//! arithmetic or whole-image placement authority is minted by this module.

use super::*;
use crate::fusion::image_reduction::{ImageSelectedRealCall, ImageSelectedRealCalls};
use runmat_accelerate_api::{
    scalar_program::ScalarLiteral, ProviderElementType, ProviderImageOutputCall,
    ProviderImageOutputOperand, ProviderRealElementwiseObserved,
    ProviderRealElementwiseOperation as Operation, ProviderRealElementwiseOutputRoute,
    ProviderRealElementwiseRecipe, ProviderRealElementwiseRequest, ProviderRepresentation,
};
use runmat_builtins::BuiltinBindingIdentity;
use runmat_execution::Digest;
use std::cell::{Cell, RefCell};

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

struct PlannedCall {
    source: ImageSelectedRealCall,
    recipe: ProviderRealElementwiseRecipe,
    identity: Digest,
}

pub(super) struct SelectedRealPlans {
    semantic_identity: Digest,
    calls: [PlannedCall; 5],
    next: Cell<usize>,
    observed: RefCell<Vec<SelectedRealReceipt>>,
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
    program: &ProviderImageStageWorkload,
) -> ProviderExecutionResult<Option<SelectedRealPlans>> {
    if source.semantic_identity() != binary.semantic_identity() {
        return Ok(None);
    }
    let Some(output_program) = program.output_program.as_ref() else {
        return Ok(None);
    };
    let (mut representation, mut bytes) = binary.output_div_carrier();
    let mut planned = Vec::with_capacity(source.calls().len());
    for call in source.calls() {
        if !matches_fact(&representation, call) {
            return Ok(None);
        }
        let Some(recipe) = ProviderRealElementwiseRecipe::plan(
            provider,
            ProviderRealElementwiseRequest {
                operation: call.operation(),
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
        let identity = recipe.identity().map_err(ProviderExecutionError::from)?;
        planned.push(PlannedCall {
            source: call.clone(),
            recipe,
            identity,
        });
    }
    let Ok(calls) = planned.try_into() else {
        return Ok(None);
    };
    let selected = SelectedRealPlans {
        semantic_identity: source.semantic_identity(),
        calls,
        next: Cell::new(0),
        observed: RefCell::new(Vec::new()),
    };
    // A selected session may execute no unplanned legacy hook. The complete
    // source-ordered program is exactly two binary calls, five scalar/cast
    // calls, and the final POWER. Filtering by hook family would allow an
    // interleaved ElemMul/ElemAdd after earlier stage effects.
    let [sub, div, mul, add, preclamp, max, cast, power] = output_program.calls.as_slice() else {
        return Ok(None);
    };
    let Some(powers) = program.selected_powers.as_ref() else {
        return Ok(None);
    };
    let real = [mul, add, preclamp, max, cast];
    if !program.validate()
        || !binary.matches_output(selected_binary::BinaryStage::OutputSub, sub)
        || !binary.matches_output(selected_binary::BinaryStage::OutputDiv, div)
        || real
            .iter()
            .enumerate()
            .any(|(index, call)| !selected.matches_output(index, call))
        || power.hook != runmat_accelerate_api::ProviderImageOutputHook::ElemPow
        || power.source_output != u64::from(powers.final_power.output)
        || [sub, div, mul, add, cast]
            .iter()
            .any(|call| call.checkpoint)
        || !preclamp.checkpoint
        || !max.checkpoint
        || !power.checkpoint
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
        let control = match source.operation() {
            Operation::ScalarAdd { control, .. }
            | Operation::ScalarMul { control }
            | Operation::ScalarMax { control } => Some(match control {
                ScalarLiteral::F32Bits(bits) => ProviderImageOutputOperand::ScalarF32Bits(bits),
                ScalarLiteral::F64Bits(bits) => ProviderImageOutputOperand::ScalarF64Bits(bits),
            }),
            Operation::FloatingCast { .. } => None,
            Operation::Sqrt => return false,
        };
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
            .execute_protected(source, protected)
            .await
            .map_err(ProviderExecutionError::after_effect)?;
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
        self.calls.each_ref().map(|call| call.identity)
    }
}
