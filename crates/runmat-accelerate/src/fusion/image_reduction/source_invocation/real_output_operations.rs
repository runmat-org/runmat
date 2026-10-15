//! Exact held scalar/cast suffix authority from the same selected graph walk.
//! These facts do not certify numerical equivalence or provider resources.

use super::*;
use runmat_accelerate_api::{
    scalar_program::ScalarLiteral, ProviderImageOutputHook, ProviderRealElementwiseOperation,
};
use runmat_builtins::{BuiltinBindingIdentity, MAX_CATALOG_ENTRY};

use super::binary_operations::CanonicalRealValue;

#[derive(Clone, Debug)]
pub(crate) struct ImageSelectedRealCall {
    output: u64,
    input: u64,
    control_source: Option<u64>,
    binding: Option<BuiltinBindingIdentity>,
    hook: ProviderImageOutputHook,
    operation: ProviderRealElementwiseOperation,
    input_fact: CanonicalRealValue,
    output_fact: CanonicalRealValue,
    output_shape: [u64; 3],
}

#[derive(Clone, Debug)]
pub(crate) struct ImageSelectedRealCalls {
    semantic_identity: Digest,
    calls: [ImageSelectedRealCall; 5],
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
) -> Option<ImageSelectedRealCall> {
    let selected = invocation
        .source
        .steps
        .iter()
        .find(|selected| selected.output == step.output)?;
    if selected.inputs != step.inputs || selected.selected_binding != step.binding {
        return None;
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
        return None;
    }
    let input = *step.inputs.first()?;
    let input_fact = CanonicalRealValue::from_fact(invocation.source_fact(input)?)?;
    let output_fact = CanonicalRealValue::from_fact(invocation.source_fact(step.output)?)?;
    if output_fact.shape() != step.output_shape
        || output_fact.precision() != step.output_precision
        || input_fact.shape() != step.output_shape
    {
        return None;
    }
    let (operation, control_source) = match step.hook {
        ProviderImageOutputHook::ScalarMul
        | ProviderImageOutputHook::ScalarAdd
        | ProviderImageOutputHook::ScalarMax => {
            let [source, control_id]: [ValueId; 2] = step.inputs.as_slice().try_into().ok()?;
            if source != input || step.operands.len() != 2 {
                return None;
            }
            let control = typed_scalar(step.operands[1])?;
            let control_fact = CanonicalRealValue::from_fact(invocation.source_fact(control_id)?)?;
            if control_fact
                .shape()
                .iter()
                .try_fold(1_u64, |a, b| a.checked_mul(*b))?
                != 1
                || !matching_class(control, control_fact.precision())
                || input_fact.precision() != output_fact.precision()
            {
                return None;
            }
            let operation = match step.hook {
                ProviderImageOutputHook::ScalarMul if step.binding.is_none() => {
                    ProviderRealElementwiseOperation::ScalarMul { control }
                }
                ProviderImageOutputHook::ScalarAdd if step.binding.is_none() => {
                    let mode = runmat_types::real_floating_add_mode(
                        invocation.source_fact(input)?,
                        invocation.source_fact(control_id)?,
                    )?;
                    ProviderRealElementwiseOperation::ScalarAdd { control, mode }
                }
                ProviderImageOutputHook::ScalarMax => {
                    let binding = step.binding?;
                    if !MAX_CATALOG_ENTRY
                        .bindings
                        .iter()
                        .any(|variant| MAX_CATALOG_ENTRY.binding_identity(variant) == binding)
                    {
                        return None;
                    }
                    ProviderRealElementwiseOperation::ScalarMax { control }
                }
                _ => return None,
            };
            (operation, Some(u64::from(control_id)))
        }
        ProviderImageOutputHook::CastSingle | ProviderImageOutputHook::CastDouble => {
            if step.inputs.as_slice() != [input]
                || step.operands.len() != 1
                || super::super::source_moment_publications::conversion_entry(selected).is_none()
            {
                return None;
            }
            let target = match step.hook {
                ProviderImageOutputHook::CastSingle => ProviderPrecision::F32,
                ProviderImageOutputHook::CastDouble => ProviderPrecision::F64,
                _ => return None,
            };
            if target != output_fact.precision() {
                return None;
            }
            (
                ProviderRealElementwiseOperation::FloatingCast { target },
                None,
            )
        }
        _ => return None,
    };
    if matches!(
        operation,
        ProviderRealElementwiseOperation::FloatingCast { .. }
    ) != control_source.is_none()
    {
        return None;
    }
    Some(ImageSelectedRealCall {
        output: u64::from(step.output),
        input: u64::from(input),
        control_source,
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
        let calls: [ImageSelectedRealCall; 5] = stage
            .output_real
            .iter()
            .map(|step| source_call(self, step))
            .collect::<Option<Vec<_>>>()?
            .try_into()
            .ok()?;
        if calls.each_ref().map(ImageSelectedRealCall::hook)
            != [
                ProviderImageOutputHook::ScalarMul,
                ProviderImageOutputHook::ScalarAdd,
                ProviderImageOutputHook::CastSingle,
                ProviderImageOutputHook::ScalarMax,
                ProviderImageOutputHook::CastSingle,
            ]
            || calls[0].input != binary.output_div().output()
            || calls[1..]
                .iter()
                .zip(&calls[..4])
                .any(|(current, previous)| current.input != previous.output)
        {
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
                        != match source.operation {
                            ProviderRealElementwiseOperation::ScalarAdd { control, .. }
                            | ProviderRealElementwiseOperation::ScalarMul { control }
                            | ProviderRealElementwiseOperation::ScalarMax { control } => {
                                Some(match control {
                                    ScalarLiteral::F32Bits(bits) => {
                                        ProviderImageOutputOperand::ScalarF32Bits(bits)
                                    }
                                    ScalarLiteral::F64Bits(bits) => {
                                        ProviderImageOutputOperand::ScalarF64Bits(bits)
                                    }
                                })
                            }
                            ProviderRealElementwiseOperation::FloatingCast { .. } => None,
                            ProviderRealElementwiseOperation::Sqrt
                            | ProviderRealElementwiseOperation::UnaryMath(_)
                            | ProviderRealElementwiseOperation::ExactSingleBasePower { .. } => {
                                return true
                            }
                        }
            })
        {
            return None;
        }
        Some(ImageSelectedRealCalls {
            semantic_identity: self.semantic_identity,
            calls,
        })
    }

    pub(crate) fn with_real_source_input_for_test(
        &self,
        output: ValueId,
        replacement: ValueId,
    ) -> Self {
        let mut changed = self.clone();
        let step = changed
            .source
            .steps
            .iter_mut()
            .find(|step| step.output == output)
            .expect("selected test source output");
        *step.inputs.get_mut(1).expect("selected scalar control") = replacement;
        changed
    }

    pub(crate) fn with_real_source_binding_for_test(
        &self,
        output: ValueId,
        binding: BuiltinBindingIdentity,
    ) -> Self {
        let mut changed = self.clone();
        changed
            .source
            .steps
            .iter_mut()
            .find(|step| step.output == output)
            .expect("selected cast output")
            .selected_binding = Some(binding);
        changed
    }

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
}

impl ImageSelectedRealCall {
    pub(crate) fn output(&self) -> u64 {
        self.output
    }
    pub(crate) fn input(&self) -> u64 {
        self.input
    }
    pub(crate) fn control_source(&self) -> Option<u64> {
        self.control_source
    }
    pub(crate) fn binding(&self) -> Option<BuiltinBindingIdentity> {
        self.binding
    }
    pub(crate) fn hook(&self) -> ProviderImageOutputHook {
        self.hook
    }
    pub(crate) fn operation(&self) -> ProviderRealElementwiseOperation {
        self.operation
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
    pub(crate) fn calls(&self) -> &[ImageSelectedRealCall; 5] {
        &self.calls
    }
}
