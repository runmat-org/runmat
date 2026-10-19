use super::*;
use crate::fusion::image_reduction::{ImageMeanProviderPlan, ImageSourcePower, ImageSourceStep};
use crate::graph::{AccelNodeLabel, InstrSpan, PrimitiveOp};
use runmat_accelerate_api::{PowerScalarControl, RealPowerArithmeticMode, RealPowerOperand};
use runmat_runtime::builtins::common::spec::FusionPowerArithmeticReceipt;
use runmat_types::NumericClass;
use runmat_value::Tensor;

fn fixture() -> (ImageSourceInvocation, ImageMeanAndPowerSource) {
    let source = ImageMeanAndPowerSource {
        mu: ImageMeanProviderPlan { calls: vec![] },
        variance: ImageMeanProviderPlan { calls: vec![] },
        square: ImageSourcePower {
            node: 1,
            base: 20,
            exponent_value: 21,
            represented_exponent: 2.0,
        },
        final_power: ImageSourcePower {
            node: 2,
            base: 37,
            exponent_value: 38,
            represented_exponent: f64::from(1.8_f32),
        },
    };
    let step = |node, inputs, output| ImageSourceStep {
        node,
        pc: InstrSpan {
            start: usize::try_from(node).unwrap(),
            end: usize::try_from(node).unwrap(),
        },
        label: AccelNodeLabel::Primitive(PrimitiveOp::ElemPow),
        inputs,
        output,
        selected_binding: None,
    };
    let single = runmat_runtime::value_fact::value_fact(&Value::Tensor(
        Tensor::from_f32(vec![1.0], vec![1, 1]).unwrap(),
    ));
    let double = runmat_runtime::value_fact::value_fact(&Value::Num(2.0));
    let binding = runmat_builtins::BuiltinBindingIdentity {
        builtin: runmat_builtins::POWER_CATALOG_ENTRY.identity,
        variant: "default",
    };
    let selected =
        |output, base_id, exponent_id, exponent_class, exponent_scalar, mode| ImageSelectedPower {
            output,
            arithmetic: FusionPowerArithmeticReceipt {
                binding,
                revision: 6,
                base_id,
                exponent_id,
                base_class: NumericClass::Single,
                exponent_class,
                exponent_scalar: Some(exponent_scalar),
                mode,
                output_class: NumericClass::Single,
            },
        };
    let invocation = ImageSourceInvocation {
        source: std::sync::Arc::new(ImageSourceTranscript {
            leaves: vec![],
            steps: vec![step(1, vec![20, 21], 22), step(2, vec![37, 38], 39)],
            checkpoints: vec![],
        }),
        facts: vec![
            (20, single.clone()),
            (21, double),
            (22, single.clone()),
            (37, single.clone()),
            (38, single.clone()),
            (39, single),
        ],
        semantic_identity: Digest::sha256("two exact POWER receipts"),
        output_program: None,
        moment_program: None,
        workload: None,
        selected_stage: None,
        selected_powers: vec![
            selected(
                22,
                20,
                21,
                NumericClass::Double,
                PowerScalarControl::DoubleBits(2.0_f64.to_bits()),
                RealPowerArithmeticMode::MixedDoubleThenSingle {
                    single_operand: RealPowerOperand::Base,
                },
            ),
            selected(
                39,
                37,
                38,
                NumericClass::Single,
                PowerScalarControl::SingleBits(1.8_f32.to_bits()),
                RealPowerArithmeticMode::NativeSingle,
            ),
        ],
        probe_observed: false,
        publications_bound: false,
    };
    (invocation, source)
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn source_ids_original_bits_classes_and_modes_are_joined() {
    let (invocation, source) = fixture();
    let program = invocation.project_power_program(&source).unwrap();
    assert!(program.validate());
    assert_eq!(
        program.square.control,
        PowerScalarControl::DoubleBits(2.0_f64.to_bits())
    );
    assert_eq!(
        program.final_power.mode,
        RealPowerArithmeticMode::NativeSingle
    );
    for change in [0, 1, 2] {
        let mut altered = invocation.clone();
        match change {
            0 => altered.selected_powers[0].arithmetic.base_id = 37,
            1 => {
                altered.selected_powers[0].arithmetic.exponent_scalar =
                    Some(PowerScalarControl::SingleBits(2.0_f32.to_bits()))
            }
            _ => altered.selected_powers[0].arithmetic.mode = RealPowerArithmeticMode::NativeSingle,
        }
        assert!(altered.project_power_program(&source).is_none());
    }
    let mut source_changed = source.clone();
    source_changed.square.represented_exponent = 4.0;
    assert!(invocation.project_power_program(&source_changed).is_none());
    let mut class_changed = invocation.clone();
    let single_fact = class_changed.facts[4].1.clone();
    class_changed.facts[1].1 = single_fact;
    assert!(class_changed.project_power_program(&source).is_none());
    let mut ranked = invocation.clone();
    ranked.facts[1].1 = runmat_runtime::value_fact::value_fact(&Value::Tensor(
        Tensor::new(vec![2.0], vec![1, 1, 1]).unwrap(),
    ));
    let ranked_program = ranked.project_power_program(&source).unwrap();
    assert_eq!(ranked_program.square.exponent_shape, [1, 1, 1]);
    ranked.facts[1].1 = runmat_runtime::value_fact::value_fact(&Value::Tensor(
        Tensor::new(vec![2.0, 2.0], vec![1, 2]).unwrap(),
    ));
    assert!(ranked.project_power_program(&source).is_none());

    let mut double_gamma = invocation;
    let double_fact = double_gamma.facts[1].1.clone();
    double_gamma.facts[4].1 = double_fact;
    double_gamma.selected_powers[1].arithmetic.exponent_class = NumericClass::Double;
    double_gamma.selected_powers[1].arithmetic.exponent_scalar =
        Some(PowerScalarControl::DoubleBits(f64::from(1.8_f32).to_bits()));
    double_gamma.selected_powers[1].arithmetic.mode =
        RealPowerArithmeticMode::MixedDoubleThenSingle {
            single_operand: RealPowerOperand::Base,
        };
    let double = double_gamma.project_power_program(&source).unwrap();
    assert_eq!(
        double.final_power.mode,
        RealPowerArithmeticMode::MixedDoubleThenSingle {
            single_operand: RealPowerOperand::Base
        }
    );
}
