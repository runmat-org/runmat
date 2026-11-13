use super::*;
use runmat_execution::value::{artifact_object_value_id, PublishedObjectCatalog, ValueRefKind};

fn reference() -> ValueRef {
    let logical_digest = Digest::sha256("immutable root bytes");
    let encryption_context = Digest::sha256("artifact encryption");
    let authorization_scope = "analysis:root-contract".to_string();
    ValueRef {
        schema_version: runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_VERSION,
        id: artifact_object_value_id(logical_digest, encryption_context, &authorization_scope),
        logical_digest,
        encoded_length: 20,
        media_type: "application/vnd.runmat.analysis-object-manifest".into(),
        value_schema: "runmat.analysis-object-manifest/v1".into(),
        encryption_context,
        kind: ValueRefKind::ResultObject,
        authorization_scope,
        resident_fence: None,
    }
}

#[test]
fn shared_root_roles_retain_portable_authority_and_reject_placeholder_or_foreign_expectations() {
    let logical = Digest::sha256("original logical authority");
    let payload = Digest::sha256("original payload");
    for kind in [
        AnalysisNumericalObjectKind::Field,
        AnalysisNumericalObjectKind::Observation,
        AnalysisNumericalObjectKind::ObservationHistoryCheckpoint,
        AnalysisNumericalObjectKind::FrequencySamplePublication,
        AnalysisNumericalObjectKind::SolverResult,
        AnalysisNumericalObjectKind::NumericalReceipt,
        AnalysisNumericalObjectKind::ResourceUsage,
    ] {
        let root = AnalysisArtifactRoot::new(reference(), kind, logical, payload).unwrap();
        PublishedObjectCatalog::new(vec![root.reference.clone()], vec![root.reference.clone()])
            .unwrap();
        root.validate_role(kind, logical, "expected").unwrap();
        assert!(root
            .validate_role(
                kind,
                Digest::sha256("foreign logical authority"),
                "expected"
            )
            .is_err());
        let cold: AnalysisArtifactRoot =
            serde_json::from_value(serde_json::to_value(&root).unwrap()).unwrap();
        assert_eq!(cold, root);
        cold.validate_role(kind, logical, "expected").unwrap();
        for placeholder in [Digest::sha256([]), Digest::from_bytes([0; 32])] {
            assert!(AnalysisArtifactRoot::new(reference(), kind, placeholder, payload).is_err());
            assert!(AnalysisArtifactRoot::new(reference(), kind, logical, placeholder).is_err());
            let mut wire = serde_json::to_value(&root).unwrap();
            wire["expected_payload_digest"] = serde_json::to_value(placeholder).unwrap();
            let forged: AnalysisArtifactRoot = serde_json::from_value(wire).unwrap();
            assert!(forged.validate_role(kind, logical, "expected").is_err());
        }
        let foreign_kind = if kind == AnalysisNumericalObjectKind::Field {
            AnalysisNumericalObjectKind::Observation
        } else {
            AnalysisNumericalObjectKind::Field
        };
        assert!(cold
            .validate_role(foreign_kind, logical, "expected")
            .is_err());
    }
}
