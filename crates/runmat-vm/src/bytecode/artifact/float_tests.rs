use super::*;
use crate::bytecode::instr::{BytecodeClassProperty, PropertyDefaultLiteral};
use crate::bytecode::program::FunctionArgumentValidation;
use crate::{Bytecode, FunctionBytecode, FunctionRegistry, Instr};
use runmat_hir::FunctionId;
use runmat_types::{
    FunctionArgDefaultValue, FunctionArgRangeInclusivity, FunctionArgValidationLiteral,
    FunctionArgValidator,
};
use std::collections::HashMap;

const FLOAT_BITS: &[u64] = &[
    0,
    0x8000_0000_0000_0000,
    1,
    0x3ff0_0000_0000_0000,
    0x7fef_ffff_ffff_ffff,
    0x7ff0_0000_0000_0000,
    0xfff0_0000_0000_0000,
    0x7ff8_0000_0000_0042,
    0xfff8_0000_0000_0042,
    0x7ff0_0000_0000_0001,
];

fn instructions() -> Vec<Instr> {
    FLOAT_BITS
        .iter()
        .flat_map(|bits| {
            let value = f64::from_bits(*bits);
            [Instr::LoadConst(value), Instr::LoadComplex(value, value)]
        })
        .collect()
}

fn assert_instruction_bits(instructions: &[Instr]) {
    assert_eq!(instructions.len(), FLOAT_BITS.len() * 2);
    for (pair, bits) in instructions.chunks_exact(2).zip(FLOAT_BITS) {
        let Instr::LoadConst(real) = pair[0] else {
            panic!("real literal")
        };
        let Instr::LoadComplex(real_part, imaginary_part) = pair[1] else {
            panic!("complex literal")
        };
        assert_eq!(real.to_bits(), *bits);
        assert_eq!(real_part.to_bits(), *bits);
        assert_eq!(imaginary_part.to_bits(), *bits);
    }
}

#[test]
fn script_envelope_preserves_all_float_literal_bits() {
    let script = Bytecode::with_instructions(instructions(), 0);
    let encoded = encode_interpreter_script_v2(&script).unwrap();
    let decoded = decode_interpreter_script_v2(&encoded).unwrap();
    assert_instruction_bits(&decoded.instructions);
    assert_eq!(encode_interpreter_script_v2(&decoded).unwrap(), encoded);
}

#[test]
fn finite_instruction_json_remains_legacy_compatible() {
    for value in [0.0, -0.0, 1.25, f64::MIN_POSITIVE, f64::MAX] {
        let legacy = format!(
            "{{\"LoadConst\":{}}}",
            serde_json::to_string(&value).unwrap()
        );
        assert_eq!(
            serde_json::to_string(&Instr::LoadConst(value)).unwrap(),
            legacy
        );
        let decoded: Instr = serde_json::from_str(&legacy).unwrap();
        let Instr::LoadConst(actual) = decoded else {
            panic!("constant")
        };
        assert_eq!(actual.to_bits(), value.to_bits());
    }
}

#[test]
fn registry_capture_preserves_defaults_validation_bounds_and_class_defaults() {
    let function = FunctionBytecode {
        function: FunctionId(1),
        display_name: "nonfinite".into(),
        instructions: instructions(),
        argument_validations: FLOAT_BITS
            .iter()
            .map(|bits| {
                let value = f64::from_bits(*bits);
                FunctionArgumentValidation {
                    input_slot: 0,
                    size: None,
                    class_name: None,
                    default_value: Some(FunctionArgDefaultValue::Number(value)),
                    validators: vec![
                        FunctionArgValidator::Member(vec![FunctionArgValidationLiteral::Number(
                            value,
                        )]),
                        FunctionArgValidator::InRange(
                            value,
                            value,
                            FunctionArgRangeInclusivity::CLOSED,
                        ),
                        FunctionArgValidator::GreaterThanOrEqual(value),
                        FunctionArgValidator::LessThanOrEqual(value),
                        FunctionArgValidator::GreaterThan(value),
                        FunctionArgValidator::LessThan(value),
                    ],
                }
            })
            .collect(),
        ..FunctionBytecode::default()
    };
    let mut registry = FunctionRegistry::new(HashMap::from([(FunctionId(1), function)]));
    registry
        .functions
        .get_mut(&FunctionId(1))
        .unwrap()
        .instructions
        .push(Instr::RegisterClass {
            name: runmat_types::ClassIdentity::new("FloatDefaults").unwrap(),
            super_class: None,
            is_sealed: false,
            is_abstract: false,
            methods: vec![],
            enumerations: vec![],
            events: vec![],
            properties: FLOAT_BITS
                .iter()
                .enumerate()
                .map(|(index, bits)| BytecodeClassProperty {
                    name: runmat_types::MemberName(format!("value{index}")),
                    is_static: false,
                    is_constant: false,
                    is_dependent: false,
                    default_literal: Some(PropertyDefaultLiteral::Num(f64::from_bits(*bits))),
                    get_access: runmat_types::MemberAccess::Public,
                    set_access: runmat_types::MemberAccess::Public,
                })
                .collect(),
        });
    let encoded = encode_interpreter_program_v2(&registry).unwrap();
    let decoded = decode_interpreter_program_v2(&encoded).unwrap();
    let function = &decoded.functions[&FunctionId(1)];
    assert_instruction_bits(&function.instructions[..FLOAT_BITS.len() * 2]);
    for (validation, bits) in function.argument_validations.iter().zip(FLOAT_BITS) {
        let Some(FunctionArgDefaultValue::Number(value)) = validation.default_value else {
            panic!("default")
        };
        assert_eq!(value.to_bits(), *bits);
        for validator in &validation.validators {
            let values = match validator {
                FunctionArgValidator::Member(values) => values
                    .iter()
                    .map(|value| {
                        let FunctionArgValidationLiteral::Number(value) = value else {
                            panic!("member")
                        };
                        *value
                    })
                    .collect::<Vec<_>>(),
                FunctionArgValidator::InRange(left, right, _) => vec![*left, *right],
                FunctionArgValidator::GreaterThanOrEqual(value)
                | FunctionArgValidator::LessThanOrEqual(value)
                | FunctionArgValidator::GreaterThan(value)
                | FunctionArgValidator::LessThan(value) => vec![*value],
                _ => panic!("unexpected validator"),
            };
            assert!(values.iter().all(|value| value.to_bits() == *bits));
        }
    }
    let Instr::RegisterClass { properties, .. } = function.instructions.last().unwrap() else {
        panic!("class")
    };
    for (property, bits) in properties.iter().zip(FLOAT_BITS) {
        let Some(PropertyDefaultLiteral::Num(value)) = property.default_literal else {
            panic!("property")
        };
        assert_eq!(value.to_bits(), *bits);
    }
    assert_eq!(encode_interpreter_program_v2(&decoded).unwrap(), encoded);
}

#[test]
fn float_literal_envelopes_reject_null_and_noncanonical_bit_tags() {
    let script = Bytecode::with_instructions(vec![Instr::LoadConst(f64::INFINITY)], 0);
    let encoded = encode_interpreter_script_v2(&script).unwrap();
    let boundary = encoded.windows(2).position(|pair| pair == b"\n\n").unwrap() + 2;
    let body: serde_json::Value = serde_json::from_slice(&encoded[boundary..]).unwrap();
    for invalid in [
        serde_json::Value::Null,
        serde_json::json!({"f64_bits": "7FF0000000000000"}),
        serde_json::json!({"f64_bits": "7ff000000000000"}),
        serde_json::json!({"f64_bits": "3ff0000000000000"}),
        serde_json::json!({"f64_bits": "7ff0000000000000", "extra": true}),
    ] {
        let mut changed = body.clone();
        changed["instructions"][0]["LoadConst"] = invalid;
        let mut bytes = encoded[..boundary].to_vec();
        bytes.extend(serde_json::to_vec(&changed).unwrap());
        bytes.push(b'\n');
        assert!(decode_interpreter_script_v2(&bytes).is_err());
    }
}
