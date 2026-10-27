use super::*;
use crate::*;
use std::collections::BTreeSet;

#[test]
fn frame_reference_and_dof_bindings_have_unique_complete_canonical_owners() {
    let entries: Vec<_> = frame::ENTRIES
        .iter()
        .chain(reference_point::ENTRIES)
        .chain(dof::ENTRIES)
        .copied()
        .collect();
    assert_eq!(entries.len(), 16);
    assert_eq!(validate_builtin_catalog(&entries), Vec::new());
    let names: BTreeSet<_> = entries.iter().map(|entry| entry.identity.name).collect();
    assert_eq!(names.len(), 16);
    for entry in entries {
        assert_eq!(entry.bindings, REQUIRED_DEFAULT_BINDING.as_slice());
        assert_eq!(entry.descriptor.output_mode, BuiltinOutputMode::Fixed);
        assert_eq!(
            entry.descriptor.completion_policy,
            BuiltinCompletionPolicy::Public
        );
        assert_eq!(
            entry
                .descriptor
                .errors
                .iter()
                .map(|error| (error.code, error.identifier, error.when, error.message))
                .collect::<Vec<_>>(),
            crate::fea_contract::descriptors::ERRORS
                .iter()
                .map(|error| (error.code, error.identifier, error.when, error.message))
                .collect::<Vec<_>>()
        );
        assert!(entry.integer_capabilities.is_empty());
        assert_eq!(
            entry.integer_audit,
            Some(&constraint_support::FEA_TYPED_CONSTRAINT_INTEGER_AUDIT)
        );
        assert_eq!(entry.contract.compatibility, BuiltinCompatibility::Matlab);
        assert_eq!(
            entry.contract.async_behavior,
            BuiltinAsyncBehavior::MaySuspend
        );
        assert_eq!(entry.contract.purity, BuiltinPurity::Impure);
        assert_eq!(entry.contract.maturity, BuiltinContractMaturity::Incomplete);
        assert!(entry
            .contract
            .effects
            .contains(&runmat_types::EffectKind::Unknown));
        assert_eq!(
            entry.documentation.authority,
            BuiltinDocumentationAuthority::Catalog
        );
        assert_eq!(entry.documentation.examples.len(), 1);
        assert!(entry.documentation.example_exemption.is_none());
        assert!(entry.documentation.evidence.notes.is_empty());
    }
}

#[test]
fn support_descriptors_preserve_original_parameter_and_signature_labels() {
    let cartesian = FEA_COORDINATE_FRAME_CARTESIAN_CATALOG_ENTRY.descriptor;
    assert_eq!(
        cartesian.signatures[0].label,
        "frame = fea.CoordinateFrame.cartesian(id, origin, xAxis, yAxis)"
    );
    assert_eq!(
        cartesian.signatures[0]
            .inputs
            .iter()
            .map(|input| input.name)
            .collect::<Vec<_>>(),
        ["id", "origin", "primaryAxis", "secondaryAxis"]
    );
    assert!(FEA_COORDINATE_FRAME_GLOBAL_CATALOG_ENTRY
        .descriptor
        .signatures[0]
        .inputs
        .is_empty());
    let reference = FEA_REFERENCE_POINT_FROM_COORDINATES_CATALOG_ENTRY.descriptor;
    assert_eq!(
        reference.signatures[0].label,
        "point = fea.ReferencePoint.fromCoordinates(id, position)"
    );
    assert_eq!(
        reference.signatures[0]
            .inputs
            .iter()
            .map(|input| input.name)
            .collect::<Vec<_>>(),
        ["id", "position"]
    );
    assert_eq!(
        FEA_STRUCTURAL_DOF_UX_CATALOG_ENTRY.descriptor.signatures[0].label,
        "dof = fea.StructuralDof.component()"
    );
    assert_eq!(
        FEA_STRUCTURAL_DOF_VALUE_TRANSLATION_CATALOG_ENTRY
            .descriptor
            .signatures[0]
            .label,
        "entry = fea.StructuralDofValue.component(dof, value)"
    );
    assert_eq!(
        FEA_STRUCTURAL_DOF_VALUE_ROTATION_CATALOG_ENTRY
            .descriptor
            .signatures[0]
            .inputs
            .iter()
            .map(|input| input.name)
            .collect::<Vec<_>>(),
        ["dof", "value"]
    );
}
