//! Held, request-local source authority for the three resident real binary calls.
//! This records canonical facts and IDs; it does not certify provider storage,
//! numerical equivalence, or whole-image physical admission.

use super::*;
use runmat_accelerate_api::{
    ProviderImageOutputHook, ProviderPrecision, ProviderRealBinaryOperation,
};
use runmat_types::{NumericClass, NumericDomain};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ImageSelectedBinaryCall {
    output: u64,
    inputs: [u64; 2],
    input_facts: [CanonicalRealValue; 2],
    operation: ProviderRealBinaryOperation,
    output_precision: ProviderPrecision,
    output_shape: [u64; 3],
}

/// Semantic class/shape only: this never claims physical dense storage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CanonicalRealValue {
    precision: ProviderPrecision,
    shape: Vec<u64>,
}

impl CanonicalRealValue {
    pub(super) fn from_fact(fact: &ValueFact) -> Option<Self> {
        let numeric = fact.numeric()?;
        if numeric.domain != NumericDomain::Real {
            return None;
        }
        let precision = match numeric.class {
            NumericClass::Single => ProviderPrecision::F32,
            NumericClass::Double => ProviderPrecision::F64,
            _ => return None,
        };
        let shape = fact
            .shape
            .known_dims()?
            .into_iter()
            .map(|dim| u64::try_from(dim?).ok())
            .collect::<Option<Vec<_>>>()?;
        Some(Self { precision, shape })
    }
    pub(crate) fn precision(&self) -> ProviderPrecision {
        self.precision
    }
    pub(crate) fn shape(&self) -> &[u64] {
        &self.shape
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ImageSelectedBinaryCalls {
    semantic_identity: Digest,
    sigma_center: ImageSelectedBinaryCall,
    output_sub: ImageSelectedBinaryCall,
    output_div: ImageSelectedBinaryCall,
}

impl ImageSelectedBinaryCall {
    fn from_step(
        invocation: &ImageSourceInvocation,
        step: &super::super::source_output_hooks::SelectedBinaryStep,
    ) -> Option<Self> {
        let output = CanonicalRealValue::from_fact(invocation.source_fact(step.output)?)?;
        if output.precision != step.output_precision || output.shape.as_slice() != step.output_shape
        {
            return None;
        }
        let operation = match step.hook {
            ProviderImageOutputHook::ElemSub => ProviderRealBinaryOperation::Sub,
            ProviderImageOutputHook::ElemDiv => ProviderRealBinaryOperation::Div,
            _ => return None,
        };
        Self::from_parts(
            invocation,
            step.output,
            step.inputs,
            operation,
            step.output_precision,
            step.output_shape,
        )
    }

    fn from_parts(
        invocation: &ImageSourceInvocation,
        output: ValueId,
        inputs: [ValueId; 2],
        operation: ProviderRealBinaryOperation,
        output_precision: ProviderPrecision,
        output_shape: [u64; 3],
    ) -> Option<Self> {
        let input_facts = inputs
            .map(|id| CanonicalRealValue::from_fact(invocation.source_fact(id)?))
            .into_iter()
            .collect::<Option<Vec<_>>>()?
            .try_into()
            .ok()?;
        Some(Self {
            output: u64::from(output),
            inputs: inputs.map(u64::from),
            input_facts,
            operation,
            output_precision,
            output_shape,
        })
    }

    pub(crate) fn output(&self) -> u64 {
        self.output
    }
    pub(crate) fn inputs(&self) -> [u64; 2] {
        self.inputs
    }
    pub(crate) fn operation(&self) -> ProviderRealBinaryOperation {
        self.operation
    }
    pub(crate) fn output_precision(&self) -> ProviderPrecision {
        self.output_precision
    }
    pub(crate) fn output_shape(&self) -> [u64; 3] {
        self.output_shape
    }
    pub(crate) fn input_facts(&self) -> &[CanonicalRealValue; 2] {
        &self.input_facts
    }
}

impl ImageSourceInvocation {
    pub(crate) fn selected_binary_calls(&self) -> Option<ImageSelectedBinaryCalls> {
        let stage = self.selected_stage.as_ref()?;
        let moments = self.moment_program.as_ref()?;
        let output_program = self.output_program.as_ref()?;
        let input_id = moments.mu.first()?.source_input;
        let mean_id = moments.mu.last()?.public_output;
        let sigma_id = moments.sigma.public_output;
        let center_id = u64::from(stage.source.square.base);
        let center = self
            .source
            .steps
            .iter()
            .find(|step| u64::from(step.output) == center_id)?;
        if !matches!(
            &center.label,
            AccelNodeLabel::Primitive(crate::graph::PrimitiveOp::Sub)
        ) || center.selected_binding.is_some()
            || center.inputs.as_slice()
                != [
                    ValueId::try_from(input_id).ok()?,
                    ValueId::try_from(mean_id).ok()?,
                ]
            || self.source_fact(center.output)?.numeric()?.domain != NumericDomain::Real
        {
            return None;
        }
        let center_fact = self.source_fact(center.output)?;
        let center_numeric = center_fact.numeric()?;
        let center_precision = match center_numeric.class {
            NumericClass::Single => ProviderPrecision::F32,
            NumericClass::Double => ProviderPrecision::F64,
            _ => return None,
        };
        let center_shape: [u64; 3] = center_fact
            .shape
            .known_dims()?
            .into_iter()
            .map(|dim| u64::try_from(dim?).ok())
            .collect::<Option<Vec<_>>>()?
            .try_into()
            .ok()?;
        let sigma_center = ImageSelectedBinaryCall::from_parts(
            self,
            center.output,
            [
                ValueId::try_from(input_id).ok()?,
                ValueId::try_from(mean_id).ok()?,
            ],
            ProviderRealBinaryOperation::Sub,
            center_precision,
            center_shape,
        )?;
        if stage.output_binary.len() != 2 {
            return None;
        }
        let output_sub = ImageSelectedBinaryCall::from_step(self, &stage.output_binary[0])?;
        let output_div = ImageSelectedBinaryCall::from_step(self, &stage.output_binary[1])?;
        if output_sub.operation != ProviderRealBinaryOperation::Sub
            || output_sub.inputs != [input_id, mean_id]
            || output_div.operation != ProviderRealBinaryOperation::Div
            || output_div.inputs != [output_sub.output, sigma_id]
        {
            return None;
        }
        let suffix = output_program
            .calls
            .iter()
            .filter(|call| {
                matches!(
                    call.hook,
                    ProviderImageOutputHook::ElemSub | ProviderImageOutputHook::ElemDiv
                )
            })
            .collect::<Vec<_>>();
        if suffix.len() != 2
            || suffix
                .iter()
                .zip([&output_sub, &output_div])
                .any(|(wire, source)| {
                    wire.source_output != source.output
                        || wire.output_shape != source.output_shape
                        || wire.output_precision != source.output_precision
                })
        {
            return None;
        }
        Some(ImageSelectedBinaryCalls {
            semantic_identity: self.semantic_identity,
            sigma_center,
            output_sub,
            output_div,
        })
    }
}

impl ImageSelectedBinaryCalls {
    pub(crate) fn semantic_identity(&self) -> Digest {
        self.semantic_identity
    }
    pub(crate) fn sigma_center(&self) -> &ImageSelectedBinaryCall {
        &self.sigma_center
    }
    pub(crate) fn output_sub(&self) -> &ImageSelectedBinaryCall {
        &self.output_sub
    }
    pub(crate) fn output_div(&self) -> &ImageSelectedBinaryCall {
        &self.output_div
    }
}
