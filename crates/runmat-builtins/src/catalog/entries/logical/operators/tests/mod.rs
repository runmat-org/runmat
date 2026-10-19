use crate::{
    BuiltinExampleHarness, BuiltinExtensionMode, BuiltinInferenceRule, LogicalElementwiseRule,
    LogicalInferenceRule,
};

use super::super::*;

#[test]
fn operator_entries_own_complete_typed_contracts_and_examples() {
    let expected = [
        (&AND_CATALOG_ENTRY, 7),
        (&OR_CATALOG_ENTRY, 7),
        (&XOR_CATALOG_ENTRY, 7),
        (&NOT_CATALOG_ENTRY, 8),
    ];
    for (entry, example_count) in expected {
        assert_eq!(entry.documentation.examples.len(), example_count);
        assert!(entry.documentation.example_exemption.is_none());
        assert!(entry
            .documentation
            .examples
            .iter()
            .all(|example| !example.id.is_empty()));
        assert!(entry
            .documentation
            .examples
            .iter()
            .any(|example| example.harness == BuiltinExampleHarness::Wgpu));
    }

    assert_eq!(
        AND_CATALOG_ENTRY.contract.inference_rule,
        BuiltinInferenceRule::Logical(LogicalInferenceRule::Elementwise(
            LogicalElementwiseRule::Binary(crate::LogicalBinaryOperator::And)
        ))
    );
    assert_eq!(
        NOT_CATALOG_ENTRY.contract.inference_rule,
        BuiltinInferenceRule::Logical(LogicalInferenceRule::Elementwise(
            LogicalElementwiseRule::Unary(crate::LogicalUnaryOperator::Not)
        ))
    );
}

#[test]
fn compatibility_extensions_match_each_operator_contract() {
    assert_eq!(AND_EXTENSIONS.len(), 2);
    assert_eq!(OR_EXTENSIONS.len(), 2);
    assert_eq!(XOR_EXTENSIONS.len(), 1);
    assert!(NOT_CATALOG_ENTRY.extensions.is_empty());
    assert!(AND_EXTENSIONS
        .iter()
        .chain(OR_EXTENSIONS.iter())
        .chain(XOR_EXTENSIONS.iter())
        .all(|extension| extension.mode == BuiltinExtensionMode::RunMatOnly));
    assert!(XOR_EXTENSIONS
        .iter()
        .all(|extension| !extension.id.contains("character")));
}
