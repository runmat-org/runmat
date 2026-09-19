use std::fs;

use runmat_analysis_language::{
    GeometryInspectionRequest, GeometryInspectionSource, LogicalSourcePath,
};
use tempfile::tempdir;

use super::{check_fea_path_in_root, FilesystemFeaHost};

#[test]
fn filesystem_host_checks_a_complete_constructed_study() {
    let temporary = tempdir().unwrap();
    let source = r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
}
mesh volume on geometry.part {
    tetrahedra {
        order = linear
        target_size = 0.1 m
        maximum_deviation = 0.01 m
    }
}
model structure {
}
study static {
    geometry = geometry.part
    mesh = mesh.volume
    model = model.structure
    analysis solve {
        structural static {
        }
    }
    output displacement = analysis.solve.field.node.displacement
}
"#;
    let path = temporary.path().join("main.fea");
    fs::write(&path, source).unwrap();

    let program = check_fea_path_in_root(temporary.path(), &path).unwrap();

    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    assert_eq!(program.geometries.len(), 1);
    assert_eq!(program.meshes.len(), 1);
    assert_eq!(program.models.len(), 1);
    assert_eq!(program.studies.len(), 1);
}

#[test]
fn filesystem_host_confines_imports_to_the_declared_root() {
    let temporary = tempdir().unwrap();
    let source_root = temporary.path().join("project");
    fs::create_dir(&source_root).unwrap();
    let path = source_root.join("main.fea");
    fs::write(
        &path,
        "import model escaped from \"../outside.fea\"\nmodel local {}\n",
    )
    .unwrap();

    let program = check_fea_path_in_root(&source_root, &path).unwrap();

    assert!(!program.is_valid());
    assert!(program
        .bound
        .graph
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "fea.module.invalid_path"));
}

#[test]
fn filesystem_host_retains_normalized_ap242_exchange_facts() {
    let temporary = tempdir().unwrap();
    let geometry_path = temporary.path().join("part.step");
    fs::write(
        &geometry_path,
        include_bytes!(
            "../../../../runmat-geometry/io/tests/fixtures/ap242_exchange_metadata.step"
        ),
    )
    .unwrap();
    let host = FilesystemFeaHost::new(temporary.path()).unwrap();
    let exchange = host
        .inspect(&GeometryInspectionRequest {
            referrer: LogicalSourcePath::new("project/main.fea").unwrap(),
            logical_path: LogicalSourcePath::new("project/part.step").unwrap(),
            expected_content_digest: None,
        })
        .unwrap()
        .exchange
        .unwrap();

    assert_eq!(exchange.parameters.len(), 1);
    assert_eq!(exchange.materials.len(), 1);
    assert_eq!(exchange.engineering_properties.len(), 1);
    assert_eq!(
        exchange.parameters[0].persistent_id.as_deref(),
        Some("parameter-wall-thickness")
    );
    assert_eq!(
        exchange.materials[0].persistent_id.as_deref(),
        Some("material-structural-steel")
    );
}

#[test]
fn checked_study_realizes_shared_case_membership_without_duplicate_actions() {
    let temporary = tempdir().unwrap();
    let source = r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
    region body : solid {
        select result
        expect exactly_one
    }
    region support : face {
        select result
        expect one_or_more
    }
}
mesh volume on geometry.part {
    tetrahedra {
        order = linear
        target_size = 0.1 m
        maximum_deviation = 0.01 m
    }
}
material steel {
    mass_density = 7850 kg/m^3
    mechanical.linear_elastic {
        youngs_modulus = 200 GPa
        poisson_ratio = 0.3
    }
}
model structure {
    assign material.steel to geometry.part.region.body
    constraint base {
        structural.fixed { target = geometry.part.region.support }
    }
    load downward {
        force on geometry.part.region.support = [0, -1, 0] kN
    }
    load_case service { include load.downward }
    load_case proof { include load.downward }
}
study static {
    geometry = geometry.part
    mesh = mesh.volume
    model = model.structure
    analysis solve { structural static {} }
    output displacement = analysis.solve.field.node.displacement
}
"#;
    let path = temporary.path().join("main.fea");
    fs::write(&path, source).unwrap();
    let program = check_fea_path_in_root(temporary.path(), &path).unwrap();
    assert!(program.is_valid(), "{:#?}", program.diagnostics);

    let resolved = program.resolve_study("project/main.fea", "static").unwrap();
    assert_eq!(resolved.mesh.mesh_id, "volume");
    assert!(resolved.execution_policy.is_none());
    let realized = resolved
        .study
        .realize_with_models(&resolved.models, &resolved.geometry, &[])
        .unwrap();
    let materials = resolved.material_catalog;
    let missing_material = runmat_analysis_core::realize_analysis_model(
        runmat_analysis_core::AnalysisModelRealizationInput {
            study: &realized,
            geometry_id: "materialized-part",
            geometry_revision: 7,
            units: runmat_geometry_core::UnitSystem::Meter,
            materials: &Default::default(),
        },
    )
    .unwrap_err();
    assert!(matches!(
        missing_material,
        runmat_analysis_core::AnalysisModelRealizationError::MissingMaterial { .. }
    ));

    let mut property_bearing = realized.clone();
    property_bearing.model.sections.push(
        runmat_analysis_core::AuthoredSectionDefinition::RectangularBeam(
            runmat_analysis_core::RectangularBeamSection {
                section_id: "unsupported-unassigned-section".into(),
                width_m: 0.02,
                height_m: 0.04,
            },
        ),
    );
    let unsupported_property = runmat_analysis_core::realize_analysis_model(
        runmat_analysis_core::AnalysisModelRealizationInput {
            study: &property_bearing,
            geometry_id: "materialized-part",
            geometry_revision: 7,
            units: runmat_geometry_core::UnitSystem::Meter,
            materials: &materials,
        },
    )
    .unwrap_err();
    assert!(matches!(
        unsupported_property,
        runmat_analysis_core::AnalysisModelRealizationError::UnsupportedPropertyRealization(_)
    ));

    let mut wrong_geometry = realized.clone();
    let runmat_analysis_core::AuthoredPropertyTarget::Solid(
        runmat_analysis_core::AuthoredRegionSource::Geometry(region),
    ) = &mut wrong_geometry.model.material_assignments[0].target
    else {
        panic!("fixture material should target its solid geometry region");
    };
    region.geometry_id = "other-part".into();
    let wrong_geometry_error = runmat_analysis_core::realize_analysis_model(
        runmat_analysis_core::AnalysisModelRealizationInput {
            study: &wrong_geometry,
            geometry_id: "materialized-part",
            geometry_revision: 7,
            units: runmat_geometry_core::UnitSystem::Meter,
            materials: &materials,
        },
    )
    .unwrap_err();
    assert!(matches!(
        wrong_geometry_error,
        runmat_analysis_core::AnalysisModelRealizationError::RegionFromWrongGeometry { .. }
    ));

    let model = runmat_analysis_core::realize_analysis_model(
        runmat_analysis_core::AnalysisModelRealizationInput {
            study: &realized,
            geometry_id: "materialized-part",
            geometry_revision: 7,
            units: runmat_geometry_core::UnitSystem::Meter,
            materials: &materials,
        },
    )
    .unwrap();

    assert_eq!(model.materials.len(), 1);
    assert_eq!(model.material_assignments.len(), 1);
    assert_eq!(model.constraints.len(), 1);
    assert_eq!(model.excitations.len(), 1);
    assert_eq!(model.excitation_cases.len(), 2);
    assert!(model.excitation_cases.iter().all(
        |case| case.action_ids == [runmat_analysis_core::ExcitationId("load.downward".into())]
    ));
    assert_eq!(model.sequence.steps[0].observations.len(), 1);
    runmat_analysis_core::validate_model(&model).unwrap();
}
