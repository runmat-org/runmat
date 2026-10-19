//! Direct hook request projected from the already bound, source-ordered image
//! transcript. This is not a proof that the provider hooks reproduce ordinary
//! arithmetic; only the exact selected operation/control order is recorded.

use runmat_accelerate_api::{
    ProviderImageOutputCall, ProviderImageOutputHook, ProviderImageOutputOperand,
    ProviderImageOutputProgram,
};
use runmat_builtins::{DOUBLE_CATALOG_ENTRY, MAX_CATALOG_ENTRY, SINGLE_CATALOG_ENTRY};
use runmat_types::{DimensionFact, NumericClass, NumericDomain, ShapeFact};
use runmat_value::{NumericScalar, Value};

use crate::fusion::{FusionGroupPlan, FusionPattern};
use crate::graph::{AccelNodeLabel, PrimitiveOp, ValueId};

use super::source_invocation::ImageValuePositions;
use super::ImageSourceInvocation;

pub(super) struct ProjectedOutputProgram {
    pub program: ProviderImageOutputProgram,
    pub binary_steps: Vec<SelectedBinaryStep>,
    pub real_steps: Vec<SelectedRealStep>,
}

/// A nonserializable companion minted in the same source walk as the public
/// hook request. The public Input/Mean/Sigma categories do not carry ValueIds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SelectedBinaryStep {
    pub output: ValueId,
    pub inputs: [ValueId; 2],
    pub hook: ProviderImageOutputHook,
    pub output_shape: [u64; 3],
    pub output_precision: runmat_accelerate_api::ProviderPrecision,
}

/// The same graph walk also seals each scalar or cast's exact producer,
/// control source, and catalog revision. Wire hook kinds alone have none of
/// this authority.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SelectedRealStep {
    pub output: ValueId,
    pub inputs: Vec<ValueId>,
    pub hook: ProviderImageOutputHook,
    pub operands: Vec<ProviderImageOutputOperand>,
    pub binding: Option<runmat_builtins::BuiltinBindingIdentity>,
    pub output_shape: [u64; 3],
    pub output_precision: runmat_accelerate_api::ProviderPrecision,
}

pub(super) fn scalar(value: &Value) -> Option<ProviderImageOutputOperand> {
    let numeric = match value {
        Value::Num(value) => NumericScalar::F64(*value),
        Value::Tensor(tensor) if tensor.len() == 1 => tensor.numeric_value_at(0)?,
        _ => return None,
    };
    match numeric {
        NumericScalar::F64(value) => {
            Some(ProviderImageOutputOperand::ScalarF64Bits(value.to_bits()))
        }
        NumericScalar::F32(value) => {
            Some(ProviderImageOutputOperand::ScalarF32Bits(value.to_bits()))
        }
        _ => None,
    }
}

impl ImageSourceInvocation {
    /// A candidate direct-hook request for the exact output suffix after the
    /// selected sigma producer. Only source operations and actual request
    /// values are used; captured plan constants cannot shadow live inputs.
    /// No arithmetic, ownership, or resource admission is minted here.
    pub(super) fn project_output_program(
        &self,
        plan: &FusionGroupPlan,
        actual_inputs: &[Value],
    ) -> Option<ProjectedOutputProgram> {
        let FusionPattern::ImageNormalize(pattern) = plan.group.pattern.as_ref()? else {
            return None;
        };
        let routes = pattern.reduction_routes.as_ref()?;
        let producers = routes.stage_producers()?;
        let moments = self.moment_program.as_ref()?;
        let public_mean = u32::try_from(moments.mu.last()?.public_output).ok()?;
        let public_sigma = u32::try_from(moments.sigma.public_output).ok()?;
        if plan.inputs.len() != actual_inputs.len()
            || self.source.checkpoints.first() != Some(&producers.mu)
            || self.source.checkpoints.get(1) != Some(&producers.sigma)
        {
            return None;
        }
        let live = ImageValuePositions::new(&plan.inputs)?;
        let input = routes.mu.input();
        let sigma_position = self
            .source
            .steps
            .iter()
            .position(|step| step.output == public_sigma)?;
        if self.source.checkpoints.len() <= 2 {
            return None;
        }
        let mut produced = Vec::new();
        produced.try_reserve_exact(self.source.steps.len()).ok()?;
        let mut calls = Vec::new();
        let mut binary_steps = Vec::new();
        let mut real_steps = Vec::new();
        for step in &self.source.steps[sigma_position + 1..] {
            let operand = |id: ValueId, produced: &[ValueId]| {
                if id == input {
                    Some(ProviderImageOutputOperand::Input)
                } else if id == public_mean {
                    Some(ProviderImageOutputOperand::Mean)
                } else if id == public_sigma {
                    Some(ProviderImageOutputOperand::Sigma)
                } else if produced.binary_search(&id).is_ok() {
                    Some(ProviderImageOutputOperand::Produced(u64::from(id)))
                } else {
                    scalar(
                        live.get(&id)
                            .and_then(|position| actual_inputs.get(*position))
                            .or_else(|| plan.const_values.get(&id))?,
                    )
                }
            };
            let operands = step
                .inputs
                .iter()
                .map(|&id| operand(id, &produced))
                .collect::<Option<Vec<_>>>()?;
            let hook = match (&step.label, step.selected_binding) {
                (AccelNodeLabel::Primitive(PrimitiveOp::Sub), None) => {
                    ProviderImageOutputHook::ElemSub
                }
                (AccelNodeLabel::Primitive(PrimitiveOp::ElemDiv), None) => {
                    ProviderImageOutputHook::ElemDiv
                }
                (AccelNodeLabel::Primitive(PrimitiveOp::ElemPow), None) => {
                    ProviderImageOutputHook::ElemPow
                }
                (AccelNodeLabel::Primitive(PrimitiveOp::ElemMul), None) => {
                    ProviderImageOutputHook::ElemMul
                }
                (AccelNodeLabel::Primitive(PrimitiveOp::Add), None)
                    if operands.get(1).is_some_and(|operand| {
                        matches!(
                            operand,
                            ProviderImageOutputOperand::ScalarF32Bits(_)
                                | ProviderImageOutputOperand::ScalarF64Bits(_)
                        )
                    }) =>
                {
                    ProviderImageOutputHook::ScalarAdd
                }
                (AccelNodeLabel::Primitive(PrimitiveOp::Add), None) => {
                    ProviderImageOutputHook::ElemAdd
                }
                (AccelNodeLabel::Primitive(PrimitiveOp::Mul), None) => {
                    ProviderImageOutputHook::ScalarMul
                }
                (AccelNodeLabel::Builtin { .. }, Some(binding))
                    if binding.builtin == MAX_CATALOG_ENTRY.identity =>
                {
                    ProviderImageOutputHook::ScalarMax
                }
                (AccelNodeLabel::Builtin { .. }, Some(binding))
                    if binding.builtin == SINGLE_CATALOG_ENTRY.identity =>
                {
                    ProviderImageOutputHook::CastSingle
                }
                (AccelNodeLabel::Builtin { .. }, Some(binding))
                    if binding.builtin == DOUBLE_CATALOG_ENTRY.identity =>
                {
                    ProviderImageOutputHook::CastDouble
                }
                _ => return None,
            };
            let fact = self.source_fact(step.output)?;
            let numeric = fact.numeric()?;
            if numeric.domain != NumericDomain::Real {
                return None;
            }
            let output_precision = match numeric.class {
                NumericClass::Single => runmat_accelerate_api::ProviderPrecision::F32,
                NumericClass::Double => runmat_accelerate_api::ProviderPrecision::F64,
                _ => return None,
            };
            let ShapeFact::Shaped { dims } = &fact.shape else {
                return None;
            };
            let [DimensionFact::Known(batch), DimensionFact::Known(height), DimensionFact::Known(width)] =
                dims.as_slice()
            else {
                return None;
            };
            let output_shape = [
                u64::try_from(*batch).ok()?,
                u64::try_from(*height).ok()?,
                u64::try_from(*width).ok()?,
            ];
            if matches!(
                hook,
                ProviderImageOutputHook::ElemSub | ProviderImageOutputHook::ElemDiv
            ) {
                let [lhs, rhs]: [ValueId; 2] = step.inputs.as_slice().try_into().ok()?;
                binary_steps.push(SelectedBinaryStep {
                    output: step.output,
                    inputs: [lhs, rhs],
                    hook,
                    output_shape,
                    output_precision,
                });
            }
            if matches!(
                hook,
                ProviderImageOutputHook::ScalarMul
                    | ProviderImageOutputHook::ScalarAdd
                    | ProviderImageOutputHook::ScalarMax
                    | ProviderImageOutputHook::CastSingle
                    | ProviderImageOutputHook::CastDouble
            ) {
                real_steps.push(SelectedRealStep {
                    output: step.output,
                    inputs: step.inputs.clone(),
                    hook,
                    operands: operands.clone(),
                    binding: step.selected_binding,
                    output_shape,
                    output_precision,
                });
            }
            calls.push(ProviderImageOutputCall {
                source_output: u64::from(step.output),
                hook,
                operands,
                output_shape,
                output_precision,
                checkpoint: self.source.checkpoints[2..].contains(&step.output),
            });
            let index = produced.binary_search(&step.output).err()?;
            produced.insert(index, step.output);
        }
        // A typed output program needs a terminal checkpoint to validate,
        // but the graph alone cannot name the selected VM store. This final
        // marker is only a pre-publication projection: bind_publications
        // replaces every marker with exact PC-ordered source stores before a
        // production arithmetic proof or physical route can be minted.
        calls.last_mut()?.checkpoint = true;
        let program = ProviderImageOutputProgram { calls };
        program.validate().then_some(ProjectedOutputProgram {
            program,
            binary_steps,
            real_steps,
        })
    }
}
