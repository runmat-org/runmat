use super::*;
use crate::{
    BuiltinContractMaturity, BuiltinDocumentationAuthority, BuiltinDocumentationStatus,
    BuiltinInferenceRule, BuiltinIntegerAuditKind, BuiltinOutputMode,
};
use runmat_types::{
    CallInference, CallRequest, DynamicReason, LiteralContext, OutputSelection,
    RequestedOutputCount, ShapeFact, ValueFact,
};

fn request(argument_count: usize, output_count: usize) -> CallRequest {
    CallRequest {
        arguments: (0..argument_count)
            .map(|_| ValueFact::unknown(DynamicReason::RuntimeValue))
            .collect(),
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(output_count)),
    }
}

fn infer(entry: &BuiltinCatalogEntry, request: &CallRequest) -> CallInference {
    match entry.contract.inference_rule {
        BuiltinInferenceRule::Identity(rule) => rule.infer(request, entry),
        other => panic!("expected identity inference, got {other:?}"),
    }
}

#[test]
fn family_entries_remain_incomplete_until_runtime_contracts_are_qualified() {
    assert_eq!(ENTRIES.len(), 5);
    for entry in ENTRIES {
        assert_eq!(entry.contract.maturity, BuiltinContractMaturity::Incomplete);
        assert_eq!(
            entry.documentation.authority,
            BuiltinDocumentationAuthority::Catalog
        );
        assert_eq!(
            entry.documentation.status,
            Some(BuiltinDocumentationStatus::Stable)
        );
        assert_eq!(entry.placement, PLACEMENT);
        assert_eq!(entry.link, LINK);
        assert!(entry
            .contract
            .effects
            .contains(&runmat_types::EffectKind::MaySuspend));
        assert!(entry
            .contract
            .effects
            .contains(&runmat_types::EffectKind::MayThrow));
    }
}

#[test]
fn documentation_conserves_legacy_sidecar_inventory() {
    assert_eq!(SAVEAS_CATALOG_ENTRY.documentation.examples.len(), 2);
    assert_eq!(SAVEAS_CATALOG_ENTRY.documentation.links.len(), 3);
    assert_eq!(SAVEFIG_CATALOG_ENTRY.documentation.examples.len(), 1);
    assert_eq!(SAVEFIG_CATALOG_ENTRY.documentation.links.len(), 3);
    assert_eq!(OPENFIG_CATALOG_ENTRY.documentation.examples.len(), 1);
    assert_eq!(OPENFIG_CATALOG_ENTRY.documentation.links.len(), 2);
    assert_eq!(HGSAVE_CATALOG_ENTRY.documentation.examples.len(), 1);
    assert_eq!(HGSAVE_CATALOG_ENTRY.documentation.links.len(), 2);
    assert_eq!(HGLOAD_CATALOG_ENTRY.documentation.examples.len(), 1);
    assert_eq!(HGLOAD_CATALOG_ENTRY.documentation.links.len(), 2);

    for entry in ENTRIES {
        assert!(!entry.documentation.keywords.is_empty());
        assert!(!entry.documentation.sections.is_empty());
        assert!(!entry.documentation.evidence.implementation.is_empty());
        assert!(!entry.documentation.evidence.verification.is_empty());
        assert!(entry.documentation.example_exemption.is_none());
    }
}

#[test]
fn descriptors_conserve_documented_form_inventory() {
    assert_eq!(SAVEAS_DESCRIPTOR.signatures.len(), 3);
    assert_eq!(SAVEFIG_DESCRIPTOR.signatures.len(), 4);
    assert_eq!(OPENFIG_DESCRIPTOR.signatures.len(), 4);
    assert_eq!(HGSAVE_DESCRIPTOR.signatures.len(), 4);
    assert_eq!(HGLOAD_DESCRIPTOR.signatures.len(), 2);
    assert_eq!(
        HGLOAD_DESCRIPTOR.output_mode,
        BuiltinOutputMode::ByRequestedOutputCount
    );

    for audit in [
        SAVEAS_INTEGER_AUDIT,
        SAVEFIG_INTEGER_AUDIT,
        OPENFIG_INTEGER_AUDIT,
        HGSAVE_INTEGER_AUDIT,
        HGLOAD_INTEGER_AUDIT,
    ] {
        assert_eq!(audit.kind, BuiltinIntegerAuditKind::NotApplicable);
    }
}

#[test]
fn inference_preserves_each_declared_output_contract() {
    let cases = [
        (&SAVEAS_CATALOG_ENTRY, 2, 1),
        (&SAVEFIG_CATALOG_ENTRY, 1, 1),
        (&OPENFIG_CATALOG_ENTRY, 1, 1),
        (&HGSAVE_CATALOG_ENTRY, 1, 1),
        (&HGLOAD_CATALOG_ENTRY, 1, 2),
    ];

    for (entry, arguments, outputs) in cases {
        let inferred = infer(entry, &request(arguments, outputs));
        assert!(inferred.diagnostics.is_empty(), "{}", entry.identity.name);
        assert_eq!(inferred.outputs.len(), outputs, "{}", entry.identity.name);
    }
}

#[test]
fn load_inference_preserves_scalar_or_row_vector_handles() {
    for entry in [&OPENFIG_CATALOG_ENTRY, &HGLOAD_CATALOG_ENTRY] {
        let inferred = infer(entry, &request(1, 1));
        assert!(inferred.diagnostics.is_empty());
        assert_eq!(
            inferred.outputs[0].shape,
            ShapeFact::from(vec![Some(1), None])
        );
    }
}

#[test]
fn inference_rejects_out_of_range_arity() {
    for entry in ENTRIES {
        let inferred = infer(entry, &request(4, 1));
        assert!(!inferred.diagnostics.is_empty(), "{}", entry.identity.name);
    }
}
