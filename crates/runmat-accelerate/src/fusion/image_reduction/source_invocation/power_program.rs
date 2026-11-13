//! Exact selected-source projection of the two canonical POWER receipts.
//! No legacy F32 workload field participates in this conversion.

use super::*;
use runmat_accelerate_api::{
    PowerScalarControl, ProviderImagePowerCall, ProviderImagePowerProgram,
    ProviderOperationIdentity, RealPowerOperand,
};
use runmat_types::NumericDomain;

/// Bind a published final exponent from this invocation's actual scalar
/// carrier. The selected POWER receipt later verifies the same ID, class,
/// bits, and source operation; a compiler fold never substitutes for it.
pub(super) fn bound_final_exponent(
    id: ValueId,
    represented: f64,
    inputs: &[Value],
    positions: &ImageValuePositions,
) -> Option<(ValueId, PowerScalarControl)> {
    let actual = inputs.get(*positions.get(&id)?)?;
    let control = match super::super::source_output_hooks::scalar(actual)? {
        ProviderImageOutputOperand::ScalarF32Bits(bits) => PowerScalarControl::SingleBits(bits),
        ProviderImageOutputOperand::ScalarF64Bits(bits) => PowerScalarControl::DoubleBits(bits),
        _ => return None,
    };
    (represented.to_bits() == control.value().to_bits()).then_some((id, control))
}

impl ImageSourceInvocation {
    pub(super) fn project_power_program(
        &self,
        source: &ImageMeanAndPowerSource,
    ) -> Option<ProviderImagePowerProgram> {
        if self.selected_powers.len() != 2 {
            return None;
        }
        let square = self.project_power_call(&source.square)?;
        let final_power = self.project_power_call(&source.final_power)?;
        let program = ProviderImagePowerProgram {
            revision: ProviderImagePowerProgram::REVISION,
            square,
            final_power,
        };
        program.validate().then_some(program)
    }

    fn project_power_call(
        &self,
        source: &super::super::ImageSourcePower,
    ) -> Option<ProviderImagePowerCall> {
        let step = self
            .source
            .steps
            .iter()
            .find(|step| step.node == source.node)?;
        if step.inputs.as_slice() != [source.base, source.exponent_value] {
            return None;
        }
        let selected = self.selected_power(step.output)?;
        let receipt = selected.arithmetic;
        let control = receipt.exponent_scalar?;
        if selected.output != step.output
            || receipt.base_id != source.base
            || receipt.exponent_id != source.exponent_value
            || receipt.binding.builtin != runmat_builtins::POWER_CATALOG_ENTRY.identity
            || control.value().to_bits() != source.represented_exponent.to_bits()
        {
            return None;
        }
        for (id, class) in [
            (source.base, receipt.base_class),
            (source.exponent_value, receipt.exponent_class),
            (step.output, receipt.output_class),
        ] {
            let numeric = self.source_fact(id)?.numeric()?;
            if numeric.domain != NumericDomain::Real || numeric.class != class {
                return None;
            }
        }
        let exponent_shape = self
            .source_fact(source.exponent_value)?
            .shape
            .known_dims()?
            .into_iter()
            .map(|extent| u64::try_from(extent?).ok())
            .collect::<Option<Vec<_>>>()?;
        let call = ProviderImagePowerCall {
            output: step.output,
            base: source.base,
            exponent: source.exponent_value,
            base_class: receipt.base_class,
            exponent_class: receipt.exponent_class,
            control,
            exponent_shape,
            resident_operand: RealPowerOperand::Base,
            mode: receipt.mode,
            output_class: receipt.output_class,
            operation: ProviderOperationIdentity::new(receipt.binding.builtin.name),
            binding_variant: receipt.binding.variant.into(),
            semantic_revision: receipt.revision,
        };
        call.validate().then_some(call)
    }
}

#[cfg(test)]
#[path = "power_program_tests.rs"]
mod tests;
