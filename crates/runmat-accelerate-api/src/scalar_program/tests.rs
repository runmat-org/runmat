use super::*;

#[test]
fn unary_math_wire_preserves_class_and_rejects_old_or_malformed_programs() {
    for class in [NumericClass::Single, NumericClass::Double] {
        let program = ScalarProgram::new(
            source(),
            SCALAR_PROGRAM_REVISION,
            vec![
                ScalarNode::Input { id: 0, class },
                ScalarNode::Apply {
                    id: 1,
                    source_index: 0,
                    class,
                    operation: ScalarOperation::UnaryMath(runmat_types::ScalarUnaryMath::Sin),
                    operands: vec![0],
                },
            ],
            1,
        )
        .unwrap();
        let encoded = serde_json::to_vec(program.wire()).unwrap();
        let decoded = ScalarProgram::from_wire(serde_json::from_slice(&encoded).unwrap()).unwrap();
        assert_eq!(decoded.identity(), program.identity());
        let mut old = program.wire().clone();
        old.semantic_revision = 4;
        assert!(ScalarProgram::from_wire(old).is_err());
        let mut wrong = program.wire().clone();
        let ScalarNode::Apply { class: output, .. } = &mut wrong.nodes[1] else {
            unreachable!()
        };
        *output = if class == NumericClass::Single {
            NumericClass::Double
        } else {
            NumericClass::Single
        };
        assert!(ScalarProgram::from_wire(wrong).is_err());
        let mut arity = program.wire().clone();
        let ScalarNode::Apply { operands, .. } = &mut arity.nodes[1] else {
            unreachable!()
        };
        operands.push(0);
        assert!(ScalarProgram::from_wire(arity).is_err());
    }
}

#[test]
fn exact_product_control_preserves_original_words_and_omits_no_data_use() {
    use runmat_types::{OperatorKind, RealFloatingRecipe};
    for scalar in [4.0_f64, -4.0, 0.0, -0.0, f64::from(f32::from_bits(1))] {
        let exact = ExactBinary32Product::for_double_bits(scalar.to_bits()).unwrap();
        assert_eq!(exact.original_double_bits(), scalar.to_bits());
        assert_eq!(
            exact.evaluate(-0.0).to_bits(),
            ((-0.0_f64 * scalar) as f32).to_bits()
        );
        let recipe = RealFloatingRecipe::for_classes(
            OperatorKind::ElementwiseMultiply,
            NumericClass::Single,
            NumericClass::Double,
        )
        .unwrap();
        let mut nodes = vec![
            ScalarNode::Input {
                id: 0,
                class: NumericClass::Single,
            },
            ScalarNode::Literal {
                id: 1,
                value: ScalarLiteral::F64Bits(scalar.to_bits()),
            },
            ScalarNode::Apply {
                id: 2,
                source_index: 0,
                class: NumericClass::Single,
                operation: ScalarOperation::RealArithmetic(recipe),
                operands: vec![0, 1],
            },
        ];
        let program =
            ScalarProgram::new(source(), SCALAR_PROGRAM_REVISION, nodes.clone(), 2).unwrap();
        assert_eq!(program.metadata_control_ids(), HashSet::from([1]));
        assert_eq!(program.exact_product(2), Some((0, 1, exact)));
        nodes.push(ScalarNode::Apply {
            id: 3,
            source_index: 1,
            class: NumericClass::Single,
            operation: ScalarOperation::RealArithmetic(
                RealFloatingRecipe::for_classes(
                    OperatorKind::Add,
                    NumericClass::Single,
                    NumericClass::Double,
                )
                .unwrap(),
            ),
            operands: vec![2, 1],
        });
        assert!(
            ScalarProgram::new(source(), SCALAR_PROGRAM_REVISION, nodes, 3)
                .unwrap()
                .metadata_control_ids()
                .is_empty()
        );
    }
    for scalar in [
        0.1_f64,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        f64::from_bits(4.0_f64.to_bits() + 1),
    ] {
        assert!(ExactBinary32Product::for_double_bits(scalar.to_bits()).is_none());
    }
}

#[test]
fn certified_power_wire_retains_double_bits_and_rejects_nonliteral_or_wider_controls() {
    for exponent in [0.0_f64, -0.0, 1.0, 2.0] {
        let nodes = vec![
            ScalarNode::Input {
                id: 0,
                class: NumericClass::Single,
            },
            ScalarNode::Literal {
                id: 1,
                value: ScalarLiteral::F64Bits(exponent.to_bits()),
            },
            ScalarNode::Apply {
                id: 2,
                source_index: 0,
                class: NumericClass::Single,
                operation: ScalarOperation::Operator(runmat_types::OperatorKind::ElementwisePower),
                operands: vec![0, 1],
            },
        ];
        let program = ScalarProgram::new(
            Digest::sha256(b"captured-power"),
            SCALAR_PROGRAM_REVISION,
            nodes,
            2,
        )
        .unwrap();
        assert_eq!(program.metadata_control_ids(), HashSet::from([1]));
        let decoded = ScalarProgram::from_wire(
            serde_json::from_slice(&serde_json::to_vec(program.wire()).unwrap()).unwrap(),
        )
        .unwrap();
        decoded.require_rederived_identity(&program).unwrap();
        for invalid in [
            ScalarNode::Input {
                id: 1,
                class: NumericClass::Double,
            },
            ScalarNode::Literal {
                id: 1,
                value: ScalarLiteral::F64Bits(3.0_f64.to_bits()),
            },
            ScalarNode::Literal {
                id: 1,
                value: ScalarLiteral::F32Bits(2.0_f32.to_bits()),
            },
        ] {
            let mut wire = program.wire().clone();
            wire.nodes[1] = invalid;
            assert!(ScalarProgram::from_wire(wire).is_err());
        }
        let mut old = program.wire().clone();
        old.semantic_revision = 2;
        assert!(ScalarProgram::from_wire(old).is_err());
    }
}
use runmat_types::OperatorKind;

fn source() -> Digest {
    Digest::sha256(b"selected canonical binding and facts")
}

fn add(literal: ScalarLiteral) -> Result<ScalarProgram> {
    ScalarProgram::new(
        source(),
        SCALAR_PROGRAM_REVISION,
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
        SCALAR_PROGRAM_REVISION,
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

#[test]
fn mixed_add_wire_requires_exact_classes_and_new_revision() {
    let mixed = ScalarProgram::new(
        source(),
        SCALAR_PROGRAM_REVISION,
        vec![
            ScalarNode::Input {
                id: 0,
                class: NumericClass::Single,
            },
            ScalarNode::Literal {
                id: 1,
                value: ScalarLiteral::F64Bits((f64::from(f32::EPSILON) / 4.0).to_bits()),
            },
            ScalarNode::Apply {
                id: 2,
                source_index: 0,
                class: NumericClass::Single,
                operation: ScalarOperation::RealArithmetic(
                    runmat_types::RealFloatingRecipe::for_classes(
                        runmat_types::OperatorKind::Add,
                        NumericClass::Single,
                        NumericClass::Double,
                    )
                    .unwrap(),
                ),
                operands: vec![0, 1],
            },
        ],
        2,
    )
    .unwrap();
    let encoded = serde_json::to_vec(mixed.wire()).unwrap();
    // Computation precision is authenticated, not an arbitrary decoded hint.
    let mut forged = serde_json::to_value(mixed.wire()).unwrap();
    forged["nodes"][2]["Apply"]["operation"]["RealArithmetic"]["computation_class"] =
        serde_json::to_value(NumericClass::Single).unwrap();
    let forged: ScalarProgramWire = serde_json::from_value(forged).unwrap();
    assert!(ScalarProgram::from_wire(forged).is_err());
    let decoded: ScalarProgramWire = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(
        ScalarProgram::from_wire(decoded).unwrap().identity(),
        mixed.identity()
    );
    let mut old = mixed.wire().clone();
    old.semantic_revision = 3;
    assert!(ScalarProgram::from_wire(old).is_err());
    let mut wrong_result = mixed.wire().clone();
    let ScalarNode::Apply { class, .. } = &mut wrong_result.nodes[2] else {
        unreachable!()
    };
    *class = NumericClass::Double;
    assert!(ScalarProgram::from_wire(wrong_result).is_err());
    let mut wrong_pair = mixed.wire().clone();
    wrong_pair.nodes[1] = ScalarNode::Literal {
        id: 1,
        value: ScalarLiteral::F32Bits(0.0f32.to_bits()),
    };
    assert!(ScalarProgram::from_wire(wrong_pair).is_err());
    let mut counterfeit = mixed.wire().clone();
    let ScalarNode::Apply { operation, .. } = &mut counterfeit.nodes[2] else {
        unreachable!()
    };
    *operation = ScalarOperation::Cast(NumericClass::Single);
    assert!(ScalarProgram::from_wire(counterfeit).is_err());
}
