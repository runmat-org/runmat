use super::*;
use runmat_types::OperatorKind;

fn source() -> Digest {
    Digest::sha256(b"selected canonical binding and facts")
}

fn add(literal: ScalarLiteral) -> Result<ScalarProgram> {
    ScalarProgram::new(
        source(),
        1,
        vec![
            ScalarNode::Input {
                id: 0,
                class: literal.class(),
            },
            ScalarNode::Literal {
                id: 1,
                value: literal,
            },
            ScalarNode::Apply {
                id: 2,
                source_index: 0,
                class: literal.class(),
                operation: ScalarOperation::Operator(OperatorKind::Add),
                operands: vec![0, 1],
            },
        ],
        2,
    )
}

#[test]
fn exact_literal_bits_are_identified_and_unknown_revisions_rejected() {
    let positive = add(ScalarLiteral::F32Bits(0.0f32.to_bits())).unwrap();
    let negative = add(ScalarLiteral::F32Bits((-0.0f32).to_bits())).unwrap();
    assert_ne!(positive.identity(), negative.identity());
    let nan_a = add(ScalarLiteral::F64Bits(0x7ff8_0000_0000_0001)).unwrap();
    let nan_b = add(ScalarLiteral::F64Bits(0x7ff8_0000_0000_0002)).unwrap();
    assert_ne!(nan_a.identity(), nan_b.identity());
    let mut revised = positive.wire().clone();
    revised.semantic_revision += 1;
    assert!(ScalarProgram::from_wire(revised).is_err());
    assert!(positive.require_rederived_identity(&positive).is_ok());
    let mut counterfeit = positive.wire().clone();
    counterfeit.nodes[1] = ScalarNode::Literal {
        id: 1,
        value: ScalarLiteral::F32Bits((-0.0f32).to_bits()),
    };
    assert!(ScalarProgram::from_wire(counterfeit)
        .unwrap()
        .require_rederived_identity(&positive)
        .is_err());
    let encoded = serde_json::to_vec(positive.wire()).unwrap();
    let decoded: ScalarProgramWire = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(
        positive.identity(),
        ScalarProgram::from_wire(decoded).unwrap().identity()
    );
}

#[test]
fn topology_arity_class_and_purity_fail_closed_after_deserialization() {
    let valid = add(ScalarLiteral::F32Bits(1.0f32.to_bits())).unwrap();
    let mut wire = valid.wire().clone();
    wire.nodes.swap(0, 2);
    assert!(ScalarProgram::from_wire(wire).is_err());

    let mut wire = valid.wire().clone();
    wire.nodes[1] = ScalarNode::Input {
        id: 0,
        class: NumericClass::Single,
    };
    assert!(ScalarProgram::from_wire(wire).is_err());

    let mut wire = valid.wire().clone();
    let ScalarNode::Apply { operands, .. } = &mut wire.nodes[2] else {
        unreachable!()
    };
    operands.pop();
    assert!(ScalarProgram::from_wire(wire).is_err());

    let mut wire = valid.wire().clone();
    let ScalarNode::Apply { class, .. } = &mut wire.nodes[2] else {
        unreachable!()
    };
    *class = NumericClass::Double;
    assert!(ScalarProgram::from_wire(wire).is_err());

    let mut wire = valid.wire().clone();
    let ScalarNode::Apply { operation, .. } = &mut wire.nodes[2] else {
        unreachable!()
    };
    *operation = ScalarOperation::Operator(OperatorKind::MatrixMultiply);
    assert!(ScalarProgram::from_wire(wire).is_err());

    let mut wire = valid.wire().clone();
    wire.nodes.insert(
        2,
        ScalarNode::Literal {
            id: 7,
            value: ScalarLiteral::F32Bits(2.0f32.to_bits()),
        },
    );
    assert!(ScalarProgram::from_wire(wire).is_err());

    let mut wire = valid.wire().clone();
    wire.nodes[0] = ScalarNode::Input {
        id: 0,
        class: NumericClass::Int32,
    };
    assert!(ScalarProgram::from_wire(wire).is_err());
}

#[test]
fn cast_is_explicit_and_must_publish_the_declared_class() {
    let rounded = ScalarProgram::new(
        source(),
        1,
        vec![
            ScalarNode::Literal {
                id: 0,
                value: ScalarLiteral::F64Bits(16_777_217.0f64.to_bits()),
            },
            ScalarNode::Apply {
                id: 1,
                source_index: 0,
                class: NumericClass::Single,
                operation: ScalarOperation::Cast(NumericClass::Single),
                operands: vec![0],
            },
        ],
        1,
    )
    .unwrap();
    assert!(matches!(
        &rounded.wire().nodes[0],
        ScalarNode::Literal {
            value: ScalarLiteral::F64Bits(_),
            ..
        }
    ));
    assert_ne!(16_777_217.0f64 as f32 as f64, 16_777_217.0);
    let mut wrong_class = rounded.wire().clone();
    let ScalarNode::Apply { class, .. } = &mut wrong_class.nodes[1] else {
        unreachable!()
    };
    *class = NumericClass::Double;
    assert!(ScalarProgram::from_wire(wrong_class).is_err());
}
