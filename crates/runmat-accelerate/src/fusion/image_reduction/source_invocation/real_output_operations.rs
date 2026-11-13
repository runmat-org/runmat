//! Exact held scalar/cast suffix authority from the same selected graph walk.
//! These facts do not certify numerical equivalence or provider resources.

use super::*;
use runmat_accelerate_api::{
    scalar_program::ScalarLiteral, ProviderElementType, ProviderImageOutputHook,
    ProviderRealElementwiseOperation, ProviderRepresentation, ProviderStorage,
};
#[cfg(test)]
use runmat_builtins::BuiltinBindingIdentity;
use runmat_builtins::MAX_CATALOG_ENTRY;
#[cfg(test)]
use std::sync::Arc;

use super::binary_operations::CanonicalRealValue;
use runmat_types::{
    NumericClass, OperatorKind, RealFloatingAddMode, RealFloatingRecipe, StorageFact,
};

#[derive(Clone, Copy, Debug)]
enum SelectedRealOperation {
    /// Source identity and classes are sealed here. The Add computation mode
    /// is selected only after the preceding physical recipe supplies DenseReal.
    ScalarAdd(ScalarLiteral),
    ScalarMul(ScalarLiteral),
    ScalarMax(ScalarLiteral),
    FloatingCast(ProviderPrecision),
}

impl SelectedRealOperation {
    fn control(self) -> Option<ScalarLiteral> {
        match self {
            Self::ScalarAdd(control) | Self::ScalarMul(control) | Self::ScalarMax(control) => {
                Some(control)
            }
            Self::FloatingCast(_) => None,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ImageSelectedRealCall {
    output: u64,
    input: u64,
    #[cfg(test)]
    control_source: Option<u64>,
    #[cfg(test)]
    binding: Option<BuiltinBindingIdentity>,
    hook: ProviderImageOutputHook,
    operation: SelectedRealOperation,
    input_fact: CanonicalRealValue,
    output_fact: CanonicalRealValue,
    output_shape: [u64; 3],
}

#[derive(Clone, Debug)]
pub(crate) struct ImageSelectedRealCalls {
    semantic_identity: Digest,
    calls: Vec<ImageSelectedRealCall>,
}

#[derive(Clone, Copy, Debug)]
enum RealSourceFailure {
    MissingStep,
    SourceIdentity,
    SourceOperation,
    MissingInput,
    InputFact,
    OutputFact,
    OutputGeometry,
    ScalarInputs,
    ScalarControl,
    ControlFact,
    ControlGeometry,
    ControlPrecision,
    ArithmeticPrecision,
    AddMode,
    MaxBinding,
    CastSource,
    CastTarget,
    UnsupportedOperation,
    OperationParity,
}

fn typed_scalar(operand: ProviderImageOutputOperand) -> Option<ScalarLiteral> {
    match operand {
        ProviderImageOutputOperand::ScalarF32Bits(bits) => Some(ScalarLiteral::F32Bits(bits)),
        ProviderImageOutputOperand::ScalarF64Bits(bits) => Some(ScalarLiteral::F64Bits(bits)),
        _ => None,
    }
}

fn matching_class(control: ScalarLiteral, class: ProviderPrecision) -> bool {
    matches!(
        (control, class),
        (ScalarLiteral::F32Bits(_), ProviderPrecision::F32)
            | (ScalarLiteral::F64Bits(_), ProviderPrecision::F64)
    )
}

fn source_call(
    invocation: &ImageSourceInvocation,
    step: &super::super::source_output_hooks::SelectedRealStep,
) -> Result<ImageSelectedRealCall, RealSourceFailure> {
    let selected = invocation
        .source
        .steps
        .iter()
        .find(|selected| selected.output == step.output)
        .ok_or(RealSourceFailure::MissingStep)?;
    if selected.inputs != step.inputs || selected.selected_binding != step.binding {
        return Err(RealSourceFailure::SourceIdentity);
    }
    let same_operation = match step.hook {
        ProviderImageOutputHook::ScalarMul => matches!(
            &selected.label,
            AccelNodeLabel::Primitive(crate::graph::PrimitiveOp::Mul)
        ),
        ProviderImageOutputHook::ScalarAdd => matches!(
            &selected.label,
            AccelNodeLabel::Primitive(crate::graph::PrimitiveOp::Add)
        ),
        ProviderImageOutputHook::ScalarMax
        | ProviderImageOutputHook::CastSingle
        | ProviderImageOutputHook::CastDouble => {
            matches!(&selected.label, AccelNodeLabel::Builtin { .. })
        }
        _ => false,
    };
    if !same_operation {
        return Err(RealSourceFailure::SourceOperation);
    }
    let input = *step.inputs.first().ok_or(RealSourceFailure::MissingInput)?;
    let input_source = invocation
        .source_fact(input)
        .ok_or(RealSourceFailure::InputFact)?;
    let output_source = invocation
        .source_fact(step.output)
        .ok_or(RealSourceFailure::OutputFact)?;
    let represented_storage = |storage| {
        matches!(
            storage,
            StorageFact::Scalar | StorageFact::Dense | StorageFact::Unknown
        )
    };
    if !represented_storage(input_source.storage) {
        return Err(RealSourceFailure::InputFact);
    }
    if !represented_storage(output_source.storage) {
        return Err(RealSourceFailure::OutputFact);
    }
    let input_fact =
        CanonicalRealValue::from_fact(input_source).ok_or(RealSourceFailure::InputFact)?;
    let output_fact =
        CanonicalRealValue::from_fact(output_source).ok_or(RealSourceFailure::OutputFact)?;
    if output_fact.shape() != step.output_shape
        || output_fact.precision() != step.output_precision
        || input_fact.shape() != step.output_shape
    {
        return Err(RealSourceFailure::OutputGeometry);
    }
    let (operation, control_source) = match step.hook {
        ProviderImageOutputHook::ScalarMul
        | ProviderImageOutputHook::ScalarAdd
        | ProviderImageOutputHook::ScalarMax => {
            let [source, control_id]: [ValueId; 2] = step
                .inputs
                .as_slice()
                .try_into()
                .map_err(|_| RealSourceFailure::ScalarInputs)?;
            if source != input || step.operands.len() != 2 {
                return Err(RealSourceFailure::ScalarInputs);
            }
            let control = typed_scalar(step.operands[1]).ok_or(RealSourceFailure::ScalarControl)?;
            let control_source = invocation
                .source_fact(control_id)
                .ok_or(RealSourceFailure::ControlFact)?;
            if matches!(
                control_source.storage,
                StorageFact::Sparse | StorageFact::Opaque
            ) {
                return Err(RealSourceFailure::ControlFact);
            }
            if step.hook == ProviderImageOutputHook::ScalarAdd
                && !matches!(
                    control_source.storage,
                    StorageFact::Scalar | StorageFact::Dense
                )
            {
                return Err(RealSourceFailure::AddMode);
            }
            let control_fact = CanonicalRealValue::from_fact(control_source)
                .ok_or(RealSourceFailure::ControlFact)?;
            if control_fact
                .shape()
                .iter()
                .try_fold(1_u64, |a, b| a.checked_mul(*b))
                .ok_or(RealSourceFailure::ControlGeometry)?
                != 1
            {
                return Err(RealSourceFailure::ControlGeometry);
            }
            if !matching_class(control, control_fact.precision()) {
                return Err(RealSourceFailure::ControlPrecision);
            }
            if input_fact.precision() != output_fact.precision() {
                return Err(RealSourceFailure::ArithmeticPrecision);
            }
            let operation = match step.hook {
                ProviderImageOutputHook::ScalarMul if step.binding.is_none() => {
                    SelectedRealOperation::ScalarMul(control)
                }
                ProviderImageOutputHook::ScalarAdd if step.binding.is_none() => {
                    SelectedRealOperation::ScalarAdd(control)
                }
                ProviderImageOutputHook::ScalarMax => {
                    let binding = step.binding.ok_or(RealSourceFailure::MaxBinding)?;
                    if !MAX_CATALOG_ENTRY
                        .bindings
                        .iter()
                        .any(|variant| MAX_CATALOG_ENTRY.binding_identity(variant) == binding)
                    {
                        return Err(RealSourceFailure::MaxBinding);
                    }
                    SelectedRealOperation::ScalarMax(control)
                }
                _ => return Err(RealSourceFailure::UnsupportedOperation),
            };
            (operation, Some(u64::from(control_id)))
        }
        ProviderImageOutputHook::CastSingle | ProviderImageOutputHook::CastDouble => {
            if step.inputs.as_slice() != [input]
                || step.operands.len() != 1
                || super::super::source_moment_publications::conversion_entry(selected).is_none()
            {
                return Err(RealSourceFailure::CastSource);
            }
            let target = match step.hook {
                ProviderImageOutputHook::CastSingle => ProviderPrecision::F32,
                ProviderImageOutputHook::CastDouble => ProviderPrecision::F64,
                _ => return Err(RealSourceFailure::UnsupportedOperation),
            };
            if target != output_fact.precision() {
                return Err(RealSourceFailure::CastTarget);
            }
            (SelectedRealOperation::FloatingCast(target), None)
        }
        _ => return Err(RealSourceFailure::UnsupportedOperation),
    };
    if matches!(operation, SelectedRealOperation::FloatingCast(_)) != control_source.is_none() {
        return Err(RealSourceFailure::OperationParity);
    }
    Ok(ImageSelectedRealCall {
        output: u64::from(step.output),
        input: u64::from(input),
        #[cfg(test)]
        control_source,
        #[cfg(test)]
        binding: step.binding,
        hook: step.hook,
        operation,
        input_fact,
        output_fact,
        output_shape: step.output_shape,
    })
}

impl ImageSourceInvocation {
    pub(crate) fn selected_real_output_calls(&self) -> Option<ImageSelectedRealCalls> {
        let stage = self.selected_stage.as_ref()?;
        let binary = self.selected_binary_calls()?;
        let program = self.output_program.as_ref()?;
        let calls: Vec<ImageSelectedRealCall> = stage
            .output_real
            .iter()
            .map(|step| match source_call(self, step) {
                Ok(call) => Some(call),
                Err(reason) => {
                    log::debug!(
                        target: "runmat_accelerate::fusion_exec",
                        "selected image real output declined: source output={} hook={:?} reason={reason:?}; inputs={:?} operands={:?} output_shape={:?} output_precision={:?}; input_fact={:?} control_fact={:?} output_fact={:?}",
                        step.output,
                        step.hook,
                        step.inputs,
                        step.operands,
                        step.output_shape,
                        step.output_precision,
                        step.inputs.first().and_then(|id| self.source_fact(*id)),
                        step.inputs.get(1).and_then(|id| self.source_fact(*id)),
                        self.source_fact(step.output)
                    );
                    None
                }
            })
            .collect::<Option<Vec<_>>>()?;
        let final_power = self
            .requested_staged_workload()?
            .selected_powers
            .as_ref()?
            .final_power
            .clone();
        if calls.first().is_some_and(|first| {
            first.input != binary.output_div().output()
                && !(u64::from(final_power.base) == binary.output_div().output()
                    && first.input == u64::from(final_power.output))
        }) || calls.windows(2).any(|pair| {
            pair[1].input != pair[0].output
                && !(pair[0].output == u64::from(final_power.base)
                    && pair[1].input == u64::from(final_power.output))
        }) {
            log::debug!(
                target: "runmat_accelerate::fusion_exec",
                "selected image real output declined: producer chain, calls={calls:?}, final power={final_power:?}"
            );
            return None;
        }
        let wire = program
            .calls
            .iter()
            .filter(|call| {
                matches!(
                    call.hook,
                    ProviderImageOutputHook::ScalarMul
                        | ProviderImageOutputHook::ScalarAdd
                        | ProviderImageOutputHook::ScalarMax
                        | ProviderImageOutputHook::CastSingle
                        | ProviderImageOutputHook::CastDouble
                )
            })
            .collect::<Vec<_>>();
        if wire.len() != calls.len()
            || wire.iter().zip(&calls).any(|(wire, source)| {
                wire.source_output != source.output
                    || wire.hook != source.hook
                    || wire.output_precision != source.output_fact.precision()
                    || wire.output_shape != source.output_shape
                    || wire.operands.first().copied()
                        != Some(ProviderImageOutputOperand::Produced(source.input))
                    || wire.operands.get(1).copied()
                        != match source.operation.control() {
                            Some(control) => Some(match control {
                                ScalarLiteral::F32Bits(bits) => {
                                    ProviderImageOutputOperand::ScalarF32Bits(bits)
                                }
                                ScalarLiteral::F64Bits(bits) => {
                                    ProviderImageOutputOperand::ScalarF64Bits(bits)
                                }
                            }),
                            None => None,
                        }
            })
        {
            log::debug!(
                target: "runmat_accelerate::fusion_exec",
                "selected image real output declined: wire projection, wire={wire:?}, calls={calls:?}"
            );
            return None;
        }
        Some(ImageSelectedRealCalls {
            semantic_identity: self.semantic_identity,
            calls,
        })
    }

    #[cfg(test)]
    pub(crate) fn with_real_source_input_for_test(
        &self,
        output: ValueId,
        replacement: ValueId,
    ) -> Self {
        let mut changed = self.clone();
        let step = Arc::make_mut(&mut changed.source)
            .steps
            .iter_mut()
            .find(|step| step.output == output)
            .expect("selected test source output");
        *step.inputs.get_mut(1).expect("selected scalar control") = replacement;
        changed
    }

    #[cfg(test)]
    pub(crate) fn with_real_source_binding_for_test(
        &self,
        output: ValueId,
        binding: BuiltinBindingIdentity,
    ) -> Self {
        let mut changed = self.clone();
        Arc::make_mut(&mut changed.source)
            .steps
            .iter_mut()
            .find(|step| step.output == output)
            .expect("selected cast output")
            .selected_binding = Some(binding);
        changed
    }

    #[cfg(test)]
    pub(crate) fn with_real_wire_control_for_test(
        &self,
        output: u64,
        control: ProviderImageOutputOperand,
    ) -> Self {
        let mut changed = self.clone();
        let wire = changed
            .output_program
            .as_mut()
            .expect("selected output program")
            .calls
            .iter_mut()
            .find(|wire| wire.source_output == output)
            .expect("selected wire output");
        *wire.operands.get_mut(1).expect("selected scalar control") = control;
        changed
    }

    #[cfg(test)]
    pub(crate) fn with_real_output_storage_for_test(
        &self,
        output: ValueId,
        storage: StorageFact,
    ) -> Self {
        let mut changed = self.clone();
        let index = changed
            .facts
            .binary_search_by_key(&output, |(id, _)| *id)
            .expect("selected real source output fact");
        changed.facts[index].1.storage = storage;
        changed
    }
}

impl ImageSelectedRealCall {
    pub(crate) fn output(&self) -> u64 {
        self.output
    }
    pub(crate) fn input(&self) -> u64 {
        self.input
    }
    #[cfg(test)]
    pub(crate) fn control_source(&self) -> Option<u64> {
        self.control_source
    }
    #[cfg(test)]
    pub(crate) fn binding(&self) -> Option<BuiltinBindingIdentity> {
        self.binding
    }
    pub(crate) fn hook(&self) -> ProviderImageOutputHook {
        self.hook
    }
    pub(crate) fn operation_for(
        &self,
        representation: &ProviderRepresentation,
    ) -> Option<ProviderRealElementwiseOperation> {
        match self.operation {
            SelectedRealOperation::ScalarMul(control) => {
                Some(ProviderRealElementwiseOperation::ScalarMul { control })
            }
            SelectedRealOperation::ScalarMax(control) => {
                Some(ProviderRealElementwiseOperation::ScalarMax { control })
            }
            SelectedRealOperation::FloatingCast(target) => {
                Some(ProviderRealElementwiseOperation::FloatingCast { target })
            }
            SelectedRealOperation::ScalarAdd(control) => {
                if representation.storage != ProviderStorage::DenseReal
                    || representation.element_type
                        != match self.input_fact.precision() {
                            ProviderPrecision::F32 => ProviderElementType::F32,
                            ProviderPrecision::F64 => ProviderElementType::F64,
                        }
                    || representation.shape.as_slice() != self.input_fact.shape()
                {
                    return None;
                }
                let left = match self.input_fact.precision() {
                    ProviderPrecision::F32 => NumericClass::Single,
                    ProviderPrecision::F64 => NumericClass::Double,
                };
                let right = match control {
                    ScalarLiteral::F32Bits(_) => NumericClass::Single,
                    ScalarLiteral::F64Bits(_) => NumericClass::Double,
                };
                let recipe = RealFloatingRecipe::for_classes(OperatorKind::Add, left, right)?;
                let mode = RealFloatingAddMode::from_recipe(recipe)?;
                Some(ProviderRealElementwiseOperation::ScalarAdd { control, mode })
            }
        }
    }

    pub(crate) fn control(&self) -> Option<ScalarLiteral> {
        self.operation.control()
    }
    pub(crate) fn input_fact(&self) -> &CanonicalRealValue {
        &self.input_fact
    }
    pub(crate) fn output_fact(&self) -> &CanonicalRealValue {
        &self.output_fact
    }
    pub(crate) fn output_shape(&self) -> [u64; 3] {
        self.output_shape
    }
}

impl ImageSelectedRealCalls {
    pub(crate) fn semantic_identity(&self) -> Digest {
        self.semantic_identity
    }
    pub(crate) fn calls(&self) -> &[ImageSelectedRealCall] {
        &self.calls
    }
}
