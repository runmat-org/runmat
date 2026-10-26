pub mod support;
use runmat_analysis_core::{ElectromagneticDomain, MagnetostaticFormulation};
use runmat_analysis_language::*;
use support::{source_module, TestHost};

const SOURCE: &str = r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
}
mesh part on geometry.part {
    tetrahedra { order = linear
        target_size = 0.5 m
        maximum_deviation = 0.01 m }
}
model magnetic {
    electromagnetic { equation = magnetostatics }
}
study response {
    geometry = geometry.part
    mesh = mesh.part
    model = model.magnetic
    analysis potential { electromagnetic static {} }
}
"#;

fn check(source: &str) -> CheckedFeaProgram {
    FeaCompiler::new(&TestHost::default()).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    })
}

#[test]
fn magnetic_vector_potential_intent_survives_static_study_realization() {
    let checked = check(SOURCE);
    assert!(checked.is_valid(), "{:#?}", checked.diagnostics);
    let resolved = checked
        .resolve_study("project/main.fea", "response")
        .unwrap();
    let realized = resolved
        .study
        .realize_with_models(&resolved.models, &resolved.geometry, &[])
        .unwrap();
    assert_eq!(
        realized.model.electromagnetic,
        Some(ElectromagneticDomain::Magnetostatic {
            formulation: MagnetostaticFormulation::MagneticVectorPotential,
        })
    );
}

#[test]
fn magnetic_domain_rejects_unknown_frequency_and_conflicting_equations() {
    for replacement in [
        "electromagnetic { equation = unknown }",
        "electromagnetic { equation = magnetostatics\nreference_frequency = 0 Hz }",
        "electromagnetic { equation = magnetostatics\nreference_frequency = 60 Hz }",
        "electromagnetic { equation = magnetostatics\nequation = electrostatics }",
        "electromagnetic { equation = magnetostatics } electromagnetic { equation = electrostatics }",
    ] {
        let checked = check(&SOURCE.replace("electromagnetic { equation = magnetostatics }", replacement));
        assert!(!checked.is_valid(), "accepted {replacement}");
    }
}

#[test]
fn included_static_electrical_domain_cannot_replace_magnetic_intent() {
    let source = SOURCE.replace(
        "model magnetic {",
        "model electrical { electromagnetic { equation = electrostatics } }\nmodel magnetic {\ninclude model.electrical",
    );
    let checked = check(&source);
    // Source checking and study binding preserve both declarations. The actual
    // solver-model realization owns physical domain conflict rejection.
    assert!(checked.is_valid(), "{:#?}", checked.diagnostics);
    let resolved = checked
        .resolve_study("project/main.fea", "response")
        .unwrap();
    let study = resolved
        .study
        .realize_with_models(&resolved.models, &resolved.geometry, &[])
        .unwrap();
    let error = runmat_analysis_core::realize_analysis_model(
        runmat_analysis_core::AnalysisModelRealizationInput {
            study: &study,
            geometry_id: "part",
            geometry_revision: 1,
            units: runmat_types::physical::CoordinateUnits::canonical(),
            regions: &Default::default(),
            materials: &resolved.material_catalog,
        },
    )
    .unwrap_err();
    assert_eq!(
        error,
        runmat_analysis_core::AnalysisModelRealizationError::ConflictingDomain("electromagnetic")
    );
}
