use super::*;
mod iterative;
use crate::{infer_catalog_call, BuiltinDocumentationAuthority};
use runmat_types::{
    CallRequest, CallableFact, LiteralContext, LiteralValue, NumericClass, NumericDomain,
    NumericFact, OutputSelection, RequestedOutputCount, ResidencyFact, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

fn matrix(rows: usize, columns: usize) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(rows), Some(columns)]),
        StorageFact::Dense,
    )
}

mod sparse;

fn factor_request_with_options(
    rows: usize,
    columns: usize,
    class: NumericClass,
    outputs: usize,
    options: &[&str],
) -> CallRequest {
    let mut request = request(rows, columns, outputs);
    request.arguments[0].kind = ValueKindFact::Numeric(NumericFact {
        class,
        domain: NumericDomain::Real,
    });
    request.literals.literal_args.push(LiteralValue::Unknown);
    for option in options {
        request
            .arguments
            .push(ValueFact::scalar(ValueKindFact::Character));
        request
            .literals
            .literal_args
            .push(LiteralValue::Character((*option).to_owned()));
    }
    request
}

#[test]
fn factor_literal_options_follow_runtime_normalization_and_last_selector() {
    for (entry, options, output, expected) in [
        (
            &LU_CATALOG_ENTRY,
            vec![" VECTOR "],
            2,
            ShapeFact::from(vec![Some(3), Some(1)]),
        ),
        (
            &QR_CATALOG_ENTRY,
            vec!["matrix", " VECTOR "],
            2,
            ShapeFact::from(vec![Some(2), Some(1)]),
        ),
        (
            &SVD_CATALOG_ENTRY,
            vec!["matrix", " VECTOR "],
            1,
            ShapeFact::from(vec![Some(2), Some(1)]),
        ),
        (
            &EIG_CATALOG_ENTRY,
            vec!["vector", " MATRIX "],
            1,
            ShapeFact::from(vec![Some(3), Some(3)]),
        ),
    ] {
        let columns = if entry.identity.name == "eig" { 3 } else { 2 };
        let inferred = infer_catalog_call(
            entry,
            &factor_request_with_options(3, columns, NumericClass::Double, 3, &options),
        );
        assert!(
            inferred.diagnostics.is_empty(),
            "{}: {:?}",
            entry.identity.name,
            inferred.diagnostics
        );
        assert_eq!(
            inferred.outputs[output].shape, expected,
            "{}",
            entry.identity.name
        );
    }
}

#[test]
fn qr_diagnoses_only_the_final_explicit_permutation_option() {
    let mut overridden = factor_request_with_options(3, 2, NumericClass::Double, 2, &["vector"]);
    overridden
        .arguments
        .push(ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })));
    overridden
        .literals
        .literal_args
        .push(LiteralValue::Number(0.0));
    let inferred = infer_catalog_call(&QR_CATALOG_ENTRY, &overridden);
    assert!(
        inferred.diagnostics.is_empty(),
        "{:?}",
        inferred.diagnostics
    );

    overridden.arguments.swap(1, 2);
    overridden.literals.literal_args.swap(1, 2);
    let inferred = infer_catalog_call(&QR_CATALOG_ENTRY, &overridden);
    assert!(inferred
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == QR_ERROR_INVALID_ARGUMENT.code));
}

#[test]
fn rectangular_lu_and_host_factor_classes_match_runtime_constructors() {
    let lu = infer_catalog_call(
        &LU_CATALOG_ENTRY,
        &factor_request_with_options(3, 2, NumericClass::Single, 3, &[]),
    );
    assert_eq!(lu.outputs[0].shape, ShapeFact::from(vec![Some(3), Some(3)]));
    assert_eq!(lu.outputs[1].shape, ShapeFact::from(vec![Some(3), Some(2)]));
    assert!(
        matches!(&lu.outputs[2].kind, ValueKindFact::Numeric(n) if n.class == NumericClass::Single && n.domain == NumericDomain::Real)
    );

    let lu_vector = infer_catalog_call(
        &LU_CATALOG_ENTRY,
        &factor_request_with_options(3, 2, NumericClass::Single, 3, &["vector"]),
    );
    assert!(
        matches!(&lu_vector.outputs[2].kind, ValueKindFact::Numeric(n) if n.class == NumericClass::Double)
    );

    for (entry, expected_class) in [
        (&QR_CATALOG_ENTRY, NumericClass::Single),
        (&SVD_CATALOG_ENTRY, NumericClass::Double),
    ] {
        let inferred = infer_catalog_call(
            entry,
            &factor_request_with_options(3, 2, NumericClass::Single, 2, &[]),
        );
        for output in &inferred.outputs[..2] {
            assert!(
                matches!(&output.kind, ValueKindFact::Numeric(n) if n.class == expected_class),
                "{}: {output:?}",
                entry.identity.name
            );
        }
    }

    let empty_svd = infer_catalog_call(&SVD_CATALOG_ENTRY, &request(0, 3, 1));
    assert_eq!(
        empty_svd.outputs[0].shape,
        ShapeFact::from(vec![Some(0), Some(0)])
    );
    let scalar_svd = infer_catalog_call(&SVD_CATALOG_ENTRY, &request(1, 3, 1));
    assert_eq!(scalar_svd.outputs[0].shape, ShapeFact::Scalar);
    assert_eq!(scalar_svd.outputs[0].storage, StorageFact::Scalar);
}

#[test]
fn pcg_operator_forms_preserve_rhs_facts_and_diagnose_unsupported_inputs() {
    // An operator handle's scalar shape and callable facts must not become
    // the solution's numeric type or extent. Both entry forms return b's
    // precision/domain, with real residual history in that precision.
    let callable = ValueFact::scalar(ValueKindFact::Callable(CallableFact {
        identity: None,
        capabilities: Default::default(),
        parameters: Vec::new(),
        parameters_complete: false,
        outputs: Vec::new(),
        outputs_complete: false,
        variadic_inputs: false,
        variadic_outputs: false,
        captures: Vec::new(),
        captures_complete: false,
    }));
    for class in [NumericClass::Single, NumericClass::Double] {
        for domain in [NumericDomain::Real, NumericDomain::Complex] {
            let numeric = ValueKindFact::Numeric(NumericFact { class, domain });
            let mut coefficient = matrix(3, 3);
            coefficient.kind = numeric.clone();
            let mut rhs = matrix(3, 1);
            rhs.kind = numeric.clone();
            for operator in [coefficient, callable.clone()] {
                let inferred = infer_catalog_call(
                    &PCG_CATALOG_ENTRY,
                    &CallRequest {
                        arguments: vec![operator, rhs.clone()],
                        literals: LiteralContext::default(),
                        outputs: OutputSelection::new(RequestedOutputCount::Exactly(5)),
                    },
                );
                assert!(
                    inferred.diagnostics.is_empty(),
                    "{:?}",
                    inferred.diagnostics
                );
                assert_eq!(inferred.outputs[0].kind, numeric);
                assert_eq!(inferred.outputs[0].shape, rhs.shape);
                assert_eq!(
                    inferred.outputs[4].kind,
                    ValueKindFact::Numeric(NumericFact {
                        class,
                        domain: NumericDomain::Real
                    }),
                );
            }
        }
    }
    let mut mixed = request(3, 3, 1);
    mixed.arguments.push(matrix(3, 1));
    for numeric in [
        NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        },
        NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Complex,
        },
    ] {
        mixed.arguments[1].kind = ValueKindFact::Numeric(numeric);
        assert!(infer_catalog_call(&PCG_CATALOG_ENTRY, &mixed)
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM.PCG.UNSUPPORTED_INPUT"));
    }
    mixed.arguments[1] = matrix(3, 1);
    for residency in [
        ResidencyFact::Device { provider: None },
        ResidencyFact::Remote { pool: None },
    ] {
        for index in 0..2 {
            mixed.arguments[index].residency = residency.clone();
            assert!(infer_catalog_call(&PCG_CATALOG_ENTRY, &mixed)
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == "RM.PCG.UNSUPPORTED_INPUT"));
            mixed.arguments[index].residency = ResidencyFact::Host;
        }
    }
}

fn request(rows: usize, columns: usize, count: usize) -> CallRequest {
    CallRequest {
        arguments: vec![matrix(rows, columns)],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
    }
}

#[test]
fn all_six_factors_own_catalog_contracts_and_examples() {
    for entry in [
        &CHOL_CATALOG_ENTRY,
        &EIG_CATALOG_ENTRY,
        &EIGS_CATALOG_ENTRY,
        &LU_CATALOG_ENTRY,
        &QR_CATALOG_ENTRY,
        &SVD_CATALOG_ENTRY,
    ] {
        assert_eq!(
            entry.documentation.authority,
            BuiltinDocumentationAuthority::Catalog
        );
        assert!(!entry.documentation.examples.is_empty());
        assert!(entry.documentation.examples.iter().all(|example| {
            matches!(
                example.verification,
                crate::BuiltinExampleVerification::Assertions { .. }
            )
        }));
    }
}

#[test]
fn output_count_and_matrix_shapes_are_identity_owned() {
    let chol = infer_catalog_call(&CHOL_CATALOG_ENTRY, &request(3, 3, 2));
    assert!(chol.diagnostics.is_empty());
    assert_eq!(
        chol.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(3)])
    );
    assert_eq!(chol.outputs[1].shape, ShapeFact::Scalar);

    let eig_values = infer_catalog_call(&EIG_CATALOG_ENTRY, &request(3, 3, 1));
    let eig_vectors = infer_catalog_call(&EIG_CATALOG_ENTRY, &request(3, 3, 2));
    assert_eq!(
        eig_values.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(1)])
    );
    assert_eq!(
        eig_vectors.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(3)])
    );

    let svd_values = infer_catalog_call(&SVD_CATALOG_ENTRY, &request(4, 2, 1));
    assert_eq!(
        svd_values.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(1)])
    );
}

#[test]
fn square_requirement_does_not_leak_to_rectangular_factors() {
    for (entry, code) in [
        (&CHOL_CATALOG_ENTRY, CHOL_ERROR_INVALID_INPUT.code),
        (&EIG_CATALOG_ENTRY, EIG_ERROR_INVALID_INPUT.code),
        (&EIGS_CATALOG_ENTRY, EIGS_ERROR_INVALID_INPUT.code),
    ] {
        let inferred = infer_catalog_call(entry, &request(2, 3, 1));
        assert!(inferred
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == code));
    }
    for entry in [&LU_CATALOG_ENTRY, &QR_CATALOG_ENTRY, &SVD_CATALOG_ENTRY] {
        let inferred = infer_catalog_call(entry, &request(2, 3, 1));
        assert!(inferred.diagnostics.is_empty());
    }
}

#[test]
fn schur_dynamic_mode_and_mixed_factor_domains_do_not_prove_real_outputs() {
    let mut dynamic_mode = request(2, 2, 2);
    dynamic_mode
        .arguments
        .push(ValueFact::scalar(ValueKindFact::Character));
    dynamic_mode.literals.literal_args = vec![LiteralValue::Unknown, LiteralValue::Unknown];
    let inferred = infer_catalog_call(&SCHUR_CATALOG_ENTRY, &dynamic_mode);
    assert!(inferred.diagnostics.is_empty());
    assert!(matches!(inferred.outputs[0].kind, ValueKindFact::Unknown));
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(2)])
    );

    dynamic_mode.literals.literal_args[1] = LiteralValue::Character(" complex ".to_owned());
    let inferred = infer_catalog_call(&SCHUR_CATALOG_ENTRY, &dynamic_mode);
    assert!(
        matches!(&inferred.outputs[0].kind, ValueKindFact::Numeric(n) if n.domain == NumericDomain::Complex)
    );

    let mut reorder = request(2, 2, 2);
    reorder.arguments[0].kind = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Complex,
    });
    reorder.arguments.push(matrix(2, 2));
    reorder.arguments.push(ValueFact::unknown(
        runmat_types::DynamicReason::RuntimeValue,
    ));
    let inferred = infer_catalog_call(&ORDSCHUR_CATALOG_ENTRY, &reorder);
    assert!(
        matches!(&inferred.outputs[0].kind, ValueKindFact::Numeric(n) if n.domain == NumericDomain::Complex)
    );
}

#[test]
fn generalized_schur_facts_follow_mode_pair_class_and_retained_block_shapes() {
    let mut qz = request(3, 3, 6);
    qz.arguments.push(matrix(3, 3));
    let default = infer_catalog_call(&QZ_CATALOG_ENTRY, &qz);
    assert!(default.diagnostics.is_empty());
    assert_eq!(default.outputs.len(), 6);
    assert!(default.outputs.iter().all(|output| {
        output.shape == ShapeFact::from(vec![Some(3), Some(3)])
            && matches!(&output.kind, ValueKindFact::Numeric(n) if n.class == NumericClass::Double && n.domain == NumericDomain::Complex)
    }));

    qz.arguments
        .push(ValueFact::scalar(ValueKindFact::Character));
    qz.literals.literal_args = vec![
        LiteralValue::Unknown,
        LiteralValue::Unknown,
        LiteralValue::Character(" REAL ".to_owned()),
    ];
    let real = infer_catalog_call(&QZ_CATALOG_ENTRY, &qz);
    assert!(real.diagnostics.is_empty());
    assert!(
        matches!(&real.outputs[4].kind, ValueKindFact::Numeric(n) if n.domain == NumericDomain::Real)
    );

    qz.arguments[0].kind = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Single,
        domain: NumericDomain::Complex,
    });
    let invalid_real = infer_catalog_call(&QZ_CATALOG_ENTRY, &qz);
    assert!(invalid_real
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM.QZ.INVALID_INPUT"));

    let mut generalized = request(3, 3, 1);
    generalized.arguments.push(matrix(3, 3));
    let spectrum = infer_catalog_call(&ORDEIG_CATALOG_ENTRY, &generalized);
    assert!(spectrum.diagnostics.is_empty());
    assert_eq!(
        spectrum.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(1)])
    );
    assert!(matches!(&spectrum.outputs[0].kind, ValueKindFact::Unknown));

    let mut reorder = request(3, 3, 4);
    reorder
        .arguments
        .extend([matrix(3, 3), matrix(3, 3), matrix(3, 3)]);
    reorder.arguments.push(ValueFact::unknown(
        runmat_types::DynamicReason::RuntimeValue,
    ));
    let inferred = infer_catalog_call(&ORDQZ_CATALOG_ENTRY, &reorder);
    assert!(inferred.diagnostics.is_empty());
    assert!(inferred.outputs.iter().all(|output| {
        output.shape == ShapeFact::from(vec![Some(3), Some(3)])
            && matches!(&output.kind, ValueKindFact::Numeric(n) if n.domain == NumericDomain::Real)
    }));
}
