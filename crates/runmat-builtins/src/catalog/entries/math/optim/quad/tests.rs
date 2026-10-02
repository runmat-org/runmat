use super::*;

#[test]
fn descriptor_retains_legacy_public_forms() {
    let labels: Vec<_> = QUAD_DESCRIPTOR
        .signatures
        .iter()
        .map(|form| form.label)
        .collect();
    assert_eq!(
        labels,
        [
            "q = quad(fun, a, b)",
            "q = quad(fun, a, b, tol, trace, p1, p2, ...)",
            "[q, fcnt] = quad(fun, a, b)",
            "[q, fcnt] = quad(fun, a, b, tol, trace, p1, p2, ...)",
        ]
    );
    assert_eq!(QUAD_EXTENSIONS.len(), 5);
    assert_eq!(QUAD_INTEGER_CAPABILITIES.len(), 5);
}

#[test]
fn all_visible_examples_have_independent_assertions() {
    assert_eq!(documentation::DOCUMENTATION.examples.len(), 3);
    for example in documentation::DOCUMENTATION.examples {
        assert!(example.display_output.is_some());
        assert!(matches!(
            example.verification,
            crate::BuiltinExampleVerification::Assertions { .. }
        ));
    }
}
