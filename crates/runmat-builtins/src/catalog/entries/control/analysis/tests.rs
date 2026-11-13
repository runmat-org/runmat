use super::{DAMP_CATALOG_ENTRY, DCGAIN_CATALOG_ENTRY, ISSTABLE_CATALOG_ENTRY};
use runmat_types::{
    CallRequest, ClassId, LiteralContext, ObjectFact, OutputSelection, RequestedOutputCount,
    ShapeFact, StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

#[test]
fn public_control_extensions_include_only_reachable_model_property_gates() {
    const TF: &[&str] = &[
        "tf-integer-coefficients",
        "tf-logical-coefficients",
        "tf-integer-sample-time",
    ];
    const TF_SS: &[&str] = &[
        "tf-integer-coefficients",
        "tf-logical-coefficients",
        "tf-integer-sample-time",
        "ss-integer-matrix-input",
        "ss-integer-sample-time",
    ];
    for name in [
        "tf.plus",
        "tf.minus",
        "tf.uplus",
        "tf.uminus",
        "tf.times",
        "tf.mtimes",
        "tf.rdivide",
        "tf.mrdivide",
        "tf.ldivide",
        "tf.mldivide",
        "tf.power",
        "tf.mpower",
        "zero",
        "dcgain",
        "isstable",
    ] {
        let actual = crate::builtin_extensions_by_name(name)
            .expect("canonical method or analysis identity")
            .iter()
            .map(|extension| extension.id)
            .collect::<Vec<_>>();
        assert_eq!(actual.as_slice(), TF, "{name}");
    }
    for name in ["pole", "damp", "pzmap"] {
        let actual = crate::builtin_extensions_by_name(name)
            .expect("canonical analysis identity")
            .iter()
            .map(|extension| extension.id)
            .collect::<Vec<_>>();
        assert_eq!(actual.as_slice(), TF_SS, "{name}");
    }
    for (name, own, model) in [
        (
            "feedback",
            &[
                "feedback-single-system",
                "feedback-scalar-forward-system",
                "feedback-scalar-scalar-systems",
                "feedback-integer-scalar-gain",
                "feedback-integer-sign",
                "feedback-logical-numeric-input",
                "feedback-resident-numeric-input",
                "feedback-single-numeric-input",
            ] as &[_],
            TF,
        ),
        ("rlocus", &["rlocus-integer-gain"] as &[_], TF_SS),
        ("step", &["step-integer-time"] as &[_], TF),
        (
            "stepinfo",
            &[
                "stepinfo-integer-sampled-data",
                "stepinfo-integer-threshold",
            ] as &[_],
            TF,
        ),
    ] {
        let actual = crate::builtin_extensions_by_name(name)
            .expect("canonical control identity")
            .iter()
            .map(|extension| extension.id)
            .collect::<Vec<_>>();
        let expected = own
            .iter()
            .copied()
            .chain(model.iter().copied())
            .collect::<Vec<_>>();
        assert_eq!(actual, expected, "{name}");
    }
    // These consumers gate numeric model metadata under their own identity
    // before entering the shared tf parser; constructor-only options must not
    // leak into their public metadata either.
    for (name, expected) in [
        (
            "impulse",
            &[
                "impulse-integer-numeric-role",
                "impulse-logical-numeric-role",
                "impulse-single-numeric-role",
                "impulse-resident-input",
            ] as &[_],
        ),
        (
            "nyquist",
            &[
                "nyquist-integer-input",
                "nyquist-logical-input",
                "nyquist-resident-input",
            ] as &[_],
        ),
    ] {
        let actual = crate::builtin_extensions_by_name(name)
            .expect("canonical response identity")
            .iter()
            .map(|extension| extension.id)
            .collect::<Vec<_>>();
        assert_eq!(actual.as_slice(), expected, "{name}");
    }
}

#[test]
fn model_admission_distinguishes_unknown_identity_from_known_other_class() {
    for entry in [
        &DAMP_CATALOG_ENTRY,
        &DCGAIN_CATALOG_ENTRY,
        &ISSTABLE_CATALOG_ENTRY,
    ] {
        for class in [None, Some(ClassId(7))] {
            let request = CallRequest {
                arguments: vec![ValueFact::proven(
                    ValueKindFact::Object(ObjectFact {
                        class,
                        runtime_class: None,
                        properties: BTreeMap::new(),
                        properties_complete: false,
                        handle_semantics: None,
                    }),
                    ShapeFact::Scalar,
                    StorageFact::Opaque,
                )],
                literals: LiteralContext::default(),
                outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
            };
            let result = crate::infer_catalog_call(entry, &request);
            assert_eq!(
                result.diagnostics.is_empty(),
                class.is_none(),
                "{}: {:?}",
                entry.identity.name,
                result.diagnostics,
            );
        }
    }
}
