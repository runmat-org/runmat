//! Versioned source POWER calls for the held selected image program. These
//! logical facts are not a provider route, resource ticket, or domain proof.

use runmat_types::NumericClass;
use serde::{Deserialize, Serialize};

use crate::{
    PowerScalarControl, ProviderOperationIdentity, RealPowerArithmeticMode, RealPowerOperand,
};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderImagePowerCall {
    pub output: u32,
    pub base: u32,
    pub exponent: u32,
    pub base_class: NumericClass,
    pub exponent_class: NumericClass,
    pub control: PowerScalarControl,
    /// Exact represented source geometry; a scalar payload need not have rank two.
    pub exponent_shape: Vec<u64>,
    pub resident_operand: RealPowerOperand,
    pub mode: RealPowerArithmeticMode,
    pub output_class: NumericClass,
    pub operation: ProviderOperationIdentity,
    pub binding_variant: String,
    pub semantic_revision: u32,
}

impl ProviderImagePowerCall {
    pub fn validate(&self) -> bool {
        self.output != self.base
            && self.output != self.exponent
            && self.base != self.exponent
            && self.resident_operand == RealPowerOperand::Base
            && self.exponent_class
                == match self.control.precision() {
                    crate::ProviderPrecision::F32 => NumericClass::Single,
                    crate::ProviderPrecision::F64 => NumericClass::Double,
                }
            && self.exponent_shape.len() >= 2
            && self
                .exponent_shape
                .iter()
                .try_fold(1_u64, |count, &extent| count.checked_mul(extent))
                == Some(1)
            && RealPowerArithmeticMode::for_classes(self.base_class, self.exponent_class)
                == Some(self.mode)
            && self.mode.output_class() == self.output_class
            && !self.operation.0.is_empty()
            && !self.binding_variant.is_empty()
            && self.semantic_revision > 0
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderImagePowerProgram {
    pub revision: u32,
    pub square: ProviderImagePowerCall,
    pub final_power: ProviderImagePowerCall,
}

impl ProviderImagePowerProgram {
    pub const REVISION: u32 = 1;

    pub fn validate(&self) -> bool {
        self.revision == Self::REVISION
            && self.square.validate()
            && self.final_power.validate()
            && self.square.output != self.final_power.output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn original_control_class_and_mode_are_not_interchangeable() {
        let square = ProviderImagePowerCall {
            output: 22,
            base: 20,
            exponent: 21,
            base_class: NumericClass::Single,
            exponent_class: NumericClass::Double,
            control: PowerScalarControl::DoubleBits(2.0_f64.to_bits()),
            exponent_shape: vec![1, 1],
            resident_operand: RealPowerOperand::Base,
            mode: RealPowerArithmeticMode::MixedDoubleThenSingle {
                single_operand: RealPowerOperand::Base,
            },
            output_class: NumericClass::Single,
            operation: ProviderOperationIdentity::new("power"),
            binding_variant: "default".into(),
            semantic_revision: 6,
        };
        let gamma = ProviderImagePowerCall {
            output: 39,
            base: 37,
            exponent: 38,
            base_class: NumericClass::Single,
            exponent_class: NumericClass::Single,
            control: PowerScalarControl::SingleBits(1.8_f32.to_bits()),
            exponent_shape: vec![1, 1],
            resident_operand: RealPowerOperand::Base,
            mode: RealPowerArithmeticMode::NativeSingle,
            output_class: NumericClass::Single,
            operation: ProviderOperationIdentity::new("power"),
            binding_variant: "default".into(),
            semantic_revision: 6,
        };
        let program = ProviderImagePowerProgram {
            revision: ProviderImagePowerProgram::REVISION,
            square: square.clone(),
            final_power: gamma.clone(),
        };
        assert!(program.validate());
        let mut changed = program.clone();
        changed.square.control = PowerScalarControl::SingleBits(2.0_f32.to_bits());
        assert!(!changed.validate());
        let mut changed = program.clone();
        changed.square.exponent_shape = vec![1, 1, 1];
        assert!(
            changed.validate(),
            "exact singleton rank remains representable"
        );
        changed.square.exponent_shape = vec![1, 2];
        assert!(!changed.validate());
        changed.square.exponent_shape = vec![0, 1];
        assert!(!changed.validate());
        let mut changed = program.clone();
        changed.final_power.mode = RealPowerArithmeticMode::NativeDouble;
        assert!(!changed.validate());
        let mut changed = program.clone();
        changed.square.semantic_revision = 0;
        assert!(!changed.validate());
        let mut changed = program.clone();
        changed.square.binding_variant.clear();
        assert!(!changed.validate());
        let mut changed = program.clone();
        changed.final_power.base = changed.final_power.output;
        assert!(!changed.validate());
        let mut changed = program.clone();
        changed.final_power = ProviderImagePowerCall {
            exponent_class: NumericClass::Double,
            control: PowerScalarControl::DoubleBits(f64::from(1.8_f32).to_bits()),
            mode: RealPowerArithmeticMode::MixedDoubleThenSingle {
                single_operand: RealPowerOperand::Base,
            },
            ..gamma
        };
        assert!(changed.validate(), "Double gamma is a separate mixed route");
    }
}
