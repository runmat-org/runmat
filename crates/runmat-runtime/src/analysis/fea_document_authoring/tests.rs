use super::*;
use crate::analysis::contracts::{AnalysisModelTemplate, ANALYSIS_MODEL_TEMPLATE_CATALOG};
use runmat_analysis_core::{
    AnalysisProcedure, AnalysisSequence, CoupledPhysicsFamily, ProcedureStep,
};
use runmat_meshing_core::ElementOrder;
use serde_json::Value as JsonValue;
use serde_yaml::Value as YamlValue;

fn input(template: AnalysisModelTemplate) -> JsonValue {
    let step = match template {
        AnalysisModelTemplate::LinearStaticStructural => ProcedureStep::structural_static("solve"),
        AnalysisModelTemplate::StructuralHarmonic => ProcedureStep::structural_harmonic("solve"),
        AnalysisModelTemplate::ThermoMechanicalCoupled => ProcedureStep::coupled(
            "solve",
            CoupledPhysicsFamily::Thermomechanical,
            AnalysisProcedure::Static,
        ),
        AnalysisModelTemplate::ThermalSteady => ProcedureStep::thermal_steady("solve"),
        AnalysisModelTemplate::ThermalTransient => ProcedureStep::thermal_transient("solve"),
        AnalysisModelTemplate::ModalStructural => ProcedureStep::structural_modal("solve"),
        AnalysisModelTemplate::AcousticHarmonic => ProcedureStep::acoustic_harmonic("solve"),
        AnalysisModelTemplate::TransientStructural => ProcedureStep::structural_transient("solve"),
        AnalysisModelTemplate::NonlinearStructural => ProcedureStep::structural_nonlinear("solve"),
        AnalysisModelTemplate::ElectromagneticHarmonic => {
            ProcedureStep::electromagnetic_harmonic("solve")
        }
        AnalysisModelTemplate::CfdSteadyState => ProcedureStep::fluid_static("solve"),
        AnalysisModelTemplate::CfdTransient => ProcedureStep::fluid_transient("solve"),
        AnalysisModelTemplate::ChtCoupled => ProcedureStep::coupled(
            "solve",
            CoupledPhysicsFamily::ConjugateHeatTransfer,
            AnalysisProcedure::Static,
        ),
        AnalysisModelTemplate::FsiCoupled => ProcedureStep::coupled(
            "solve",
            CoupledPhysicsFamily::FluidStructureInteraction,
            AnalysisProcedure::TransientResponse,
        ),
    };
    serde_json::json!({
        "geometry": { "path": "part.stl", "units": "meter" },
        "model": { "source": { "inline": {} } },
        "sequence": AnalysisSequence::from_steps("study.sequence", vec![step])
    })
}

#[test]
fn operations_are_canonical() {
    for name in FEA_STUDY_DOCUMENT_OPERATION_NAMES {
        assert_eq!(
            name.parse::<FeaStudyDocumentOperation>().unwrap().as_str(),
            *name
        );
    }
    for name in [
        "add_driving_condition",
        "set_outputs",
        "set_steps",
        "set_loads",
    ] {
        assert!(name.parse::<FeaStudyDocumentOperation>().is_err());
    }
}

#[test]
fn create_covers_templates_without_retired_sections() {
    for entry in ANALYSIS_MODEL_TEMPLATE_CATALOG {
        let output = apply_fea_study_document_operation(
            "create",
            "/p/model.fea",
            None,
            input(entry.template),
        )
        .unwrap();
        let document: YamlValue = serde_yaml::from_str(&output.source).unwrap();
        assert!(yaml::yaml_at(&document, &["sequence", "steps"])
            .and_then(YamlValue::as_sequence)
            .is_some_and(|steps| steps.len() == 1));
        for key in ["boundary_conditions", "loads", "outputs"] {
            assert!(yaml::yaml_at(&document, &[key]).is_none());
        }
        assert!(yaml::yaml_at(&document, &["run"]).is_none());
        let execution = yaml::yaml_at(&document, &["execution"])
            .and_then(YamlValue::as_sequence)
            .expect("canonical execution list");
        assert_eq!(execution.len(), 1);
        assert_eq!(
            yaml::scalar(&execution[0], &["requirement"]).as_deref(),
            Some("production")
        );
        assert!(yaml::scalar(&execution[0], &["solver"]).is_some());
        assert!(yaml::yaml_at(&execution[0], &["options"])
            .and_then(YamlValue::as_mapping)
            .is_some());
        assert_eq!(
            output.result.summary.model_source.as_deref(),
            Some("inline")
        );
        assert!(output.result.summary.primary_procedure.is_some());
    }
}

#[test]
fn canonical_edits_are_validated_and_summarized() {
    let created = apply_fea_study_document_operation(
        "create",
        "/p/a.fea",
        None,
        input(AnalysisModelTemplate::LinearStaticStructural),
    )
    .unwrap();
    let region = apply_fea_study_document_operation(
        "add_region",
        "/p/a.fea",
        Some(&created.source),
        serde_json::json!({ "region_id": "solid", "selector": { "id": "body_1" } }),
    )
    .unwrap();
    assert_eq!(
        region.result.regions[0].selector,
        Some(FeaRegionSelector::Id("body_1".into()))
    );
    let material = apply_fea_study_document_operation(
        "add_material",
        "/p/a.fea",
        Some(&region.source),
        serde_json::json!({
            "material_id": "steel",
            "name": "Steel",
            "mechanical": {
                "youngs_modulus_pa": 2e11,
                "poisson_ratio": 0.3,
                "density_kg_per_m3": 7850.0
            }
        }),
    )
    .unwrap();
    let assignment = apply_fea_study_document_operation(
        "assign_material",
        "/p/a.fea",
        Some(&material.source),
        serde_json::json!({
            "target": { "solid": { "region": "solid" } },
            "material": "steel",
            "confidence": "verified"
        }),
    )
    .unwrap();
    assert_eq!(assignment.result.counts.material_assignments, 1);
    assert!(assignment.result.readiness.ready_to_solve);
}

#[test]
fn composed_sources_edit_only_the_explicit_inline_overlay() {
    let mut create = input(AnalysisModelTemplate::LinearStaticStructural);
    create["model"]["source"] = serde_json::json!({
        "compose": [
            { "file": { "path": "models/shared.fea" } },
            { "inline": {} }
        ]
    });
    let created = apply_fea_study_document_operation("create", "/p/a.fea", None, create).unwrap();
    assert_eq!(
        created.result.summary.model_source.as_deref(),
        Some("compose")
    );
    assert!(!created.result.readiness.ready_to_solve);
    assert!(created
        .result
        .readiness
        .blockers
        .iter()
        .any(|blocker| blocker.contains("runmat check")));

    let material = apply_fea_study_document_operation(
        "add_material",
        "/p/a.fea",
        Some(&created.source),
        serde_json::json!({
            "material_id": "steel",
            "mechanical": {
                "youngs_modulus_pa": 2e11,
                "poisson_ratio": 0.3,
                "density_kg_per_m3": 7850.0
            }
        }),
    )
    .unwrap();
    let document: YamlValue = serde_yaml::from_str(&material.source).unwrap();
    let composed = yaml::yaml_at(&document, &["model", "source", "compose"])
        .and_then(YamlValue::as_sequence)
        .expect("composed model sources");
    assert!(yaml::yaml_at(&composed[1], &["inline", "materials", "steel"]).is_some());
}

#[test]
fn region_authoring_rejects_legacy_string_selectors() {
    let created = apply_fea_study_document_operation(
        "create",
        "/p/a.fea",
        None,
        input(AnalysisModelTemplate::LinearStaticStructural),
    )
    .unwrap();

    for selector in [
        serde_json::json!("body_1"),
        serde_json::json!("region:body_1"),
        serde_json::json!({ "region": "body_1" }),
    ] {
        let error = apply_fea_study_document_operation(
            "add_region",
            "/p/a.fea",
            Some(&created.source),
            serde_json::json!({ "region_id": "solid", "selector": selector }),
        )
        .expect_err("legacy selector syntax must be rejected");
        assert!(error.contains("selector") || error.contains("add_region"));
    }
}

#[test]
fn execution_authoring_rejects_untyped_or_unknown_fields() {
    let cases = [
        serde_json::json!({ "step": "solve", "unexpected": true }),
        serde_json::json!({ "step": "solve", "options": {} }),
        serde_json::json!({
            "step": "solve",
            "solver": "structural_linear",
            "options": { "unexpected": true }
        }),
    ];

    for execution in cases {
        let mut create = input(AnalysisModelTemplate::LinearStaticStructural);
        create["execution"] = serde_json::json!([execution]);
        apply_fea_study_document_operation("create", "/p/a.fea", None, create)
            .expect_err("invalid execution authoring must be rejected");
    }
}

#[test]
fn mesh_has_only_canonical_settings() {
    let created = apply_fea_study_document_operation(
        "create",
        "/p/a.fea",
        None,
        input(AnalysisModelTemplate::LinearStaticStructural),
    )
    .unwrap();
    let output = apply_fea_study_document_operation(
        "set_mesh",
        "/p/a.fea",
        Some(&created.source),
        serde_json::json!({
            "element_order": "tet10",
            "maximum_elements": 10,
            "target_edge_length_m": 0.2,
            "maximum_chordal_deviation_m": 0.01,
            "maximum_grading_ratio": 1.2,
            "deterministic_seed": 7
        }),
    )
    .unwrap();
    assert_eq!(
        output.result.summary.mesh.unwrap().element_order,
        ElementOrder::Tet10
    );
    assert!(!output.source.contains("profile:"));
}

#[test]
fn random_excitation_authoring_round_trips_and_removes_by_typed_id() {
    const TRIANGLE_STL: &str = "solid triangle\n  facet normal 0 0 1\n    outer loop\n      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n    endloop\n  endfacet\nendsolid triangle\n";
    let directory = tempfile::tempdir().expect("temporary study directory");
    std::fs::write(directory.path().join("part.stl"), TRIANGLE_STL)
        .expect("write geometry fixture");
    let path = directory.path().join("random-response.fea");
    let created = apply_fea_study_document_operation(
        "create",
        path.to_string_lossy().as_ref(),
        None,
        input(AnalysisModelTemplate::ModalStructural),
    )
    .expect("create modal structural study");
    let excitation = serde_json::json!({
        "excitation_id": "broadband_base",
        "frequencies_hz": [0.1, 10.0, 100.0],
        "channels": [{
            "channel_id": "base_x",
            "quantity": "base_acceleration",
            "direction": {
                "frame": "global",
                "vector": [1.0, 0.0, 0.0]
            }
        }],
        "spectral_density": [
            { "values": [{ "real": 1.0e-4, "imaginary": 0.0 }] },
            { "values": [{ "real": 2.0e-4, "imaginary": 0.0 }] },
            { "values": [{ "real": 1.0e-4, "imaginary": 0.0 }] }
        ],
        "convention": "one_sided_per_hertz",
        "interpolation": "linear_frequency",
        "extrapolation": "reject",
        "evidence": {
            "max_absolute_density_error": 0.0,
            "equivalent_degrees_of_freedom": null
        },
        "provenance_digest": "sha256:document-authoring-random-response"
    });
    let added = apply_fea_study_document_operation(
        "add_random_excitation",
        path.to_string_lossy().as_ref(),
        Some(&created.source),
        excitation.clone(),
    )
    .expect("add random excitation");
    assert_eq!(added.result.counts.random_excitations, 1);
    assert_eq!(added.result.random_excitations[0].id, "broadband_base");
    let resolved = pollster::block_on(super::super::fea_document::parse_and_resolve_fea_document(
        &added.source,
        directory.path(),
    ))
    .expect("authored random excitation must pass the execution parser");
    let super::super::fea_document::FeaResolvedDocument::Study(study) = resolved else {
        panic!("authoring must preserve a study document");
    };
    assert_eq!(study.model.unwrap().random_excitations.len(), 1);

    let mut updated_excitation = excitation;
    updated_excitation["spectral_density"][1]["values"][0]["real"] = serde_json::json!(3.0e-4);
    let updated = apply_fea_study_document_operation(
        "update_random_excitation",
        path.to_string_lossy().as_ref(),
        Some(&added.source),
        updated_excitation,
    )
    .expect("update random excitation");
    assert_eq!(updated.result.counts.random_excitations, 1);

    let removed = apply_fea_study_document_operation(
        "remove_random_excitation",
        path.to_string_lossy().as_ref(),
        Some(&updated.source),
        serde_json::json!({ "id": "broadband_base" }),
    )
    .expect("remove random excitation");
    assert_eq!(removed.result.counts.random_excitations, 0);
    assert!(removed.result.random_excitations.is_empty());
}

#[test]
fn created_source_is_accepted_by_the_execution_parser() {
    const TRIANGLE_STL: &str = "solid triangle\n  facet normal 0 0 1\n    outer loop\n      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n    endloop\n  endfacet\nendsolid triangle\n";
    let directory = tempfile::tempdir().expect("temporary study directory");
    std::fs::write(directory.path().join("part.stl"), TRIANGLE_STL)
        .expect("write geometry fixture");
    for entry in ANALYSIS_MODEL_TEMPLATE_CATALOG {
        let created = apply_fea_study_document_operation(
            "create",
            directory
                .path()
                .join(format!("{}.fea", entry.template.as_snake_case()))
                .to_string_lossy()
                .as_ref(),
            None,
            input(entry.template),
        )
        .expect("create canonical study");
        assert!(
            !created.source.contains('!'),
            "canonical authoring must not emit Rust-specific YAML enum tags"
        );

        pollster::block_on(super::super::fea_document::parse_and_resolve_fea_document(
            &created.source,
            directory.path(),
        ))
        .unwrap_or_else(|error| {
            panic!(
                "template `{}` must parse: {error}",
                entry.template.as_snake_case()
            )
        });
    }
}
