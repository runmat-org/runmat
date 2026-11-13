use super::*;

#[test]
fn shared_compiled_clone_preserves_definition_and_cold_validation_under_bounded_inventory() {
    let model = crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
    let mesh = crate::tests::single_tetrahedron_solver_mesh_for_model(&model);
    let compiled = CompiledAnalysisStudy {
        module_path: "tests/shared-definition.fea".into(),
        study_id: "shared-definition".into(),
        study_inputs: Vec::new(),
        outputs: Vec::new(),
        assessments: Vec::new(),
        properties: super::super::compile_study_properties(&model, &mesh).unwrap(),
        model,
        solver_mesh_digest: mesh.canonical_digest,
    };
    let identity = compiled.definition_digest().unwrap();
    let wire = serde_json::to_vec(&compiled).unwrap();
    let limits = CanonicalLimits::new(16 << 20, 1_000_000, 65_536, 64);
    let bytes = compiled
        .shared_clone_storage_bytes_estimate(limits)
        .unwrap();
    assert!(bytes >= std::mem::size_of::<CompiledAnalysisStudy>() as u64);
    assert!(compiled
        .shared_clone_storage_bytes_estimate(CanonicalLimits::new(1, 1_000_000, 65_536, 64),)
        .is_err());
    let retained = compiled.clone();
    drop(compiled);
    assert_eq!(retained.definition_digest().unwrap(), identity);
    assert_eq!(serde_json::to_vec(&retained).unwrap(), wire);
    retained.validate_against(&mesh).unwrap();
    let cold: CompiledAnalysisStudy = serde_json::from_slice(&wire).unwrap();
    cold.validate_against(&mesh).unwrap();
    assert_eq!(cold.definition_digest().unwrap(), identity);
}
