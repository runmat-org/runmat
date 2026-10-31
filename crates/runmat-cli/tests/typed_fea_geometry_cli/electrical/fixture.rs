use super::*;
use runmat_geometry_ops::{
    GeometryFeatureProvider, GeometryFeatureProviderSelection, ResolvedGeometryFeatureOperation,
};
use runmat_meshing_core::{PersistentEntityId, SolverMeshArtifact};

pub(super) fn prepare(
    project: &TempDir,
    quadratic: bool,
    equation: &str,
    gauge: f64,
    voltage: f64,
) -> [PersistentEntityId; 2] {
    fs::create_dir(project.path().join("src")).unwrap();
    fs::write(project.path().join("runmat.toml"),
        "[package]\nname = \"electrical-prism\"\n[sources]\nroots = [\"src\"]\n[runtime.fea]\nartifact_root = \"artifacts\"\nstudy_artifact_root = \"study-artifacts\"\n").unwrap();
    let provider = runmat_geometry_io::feature::OcctExactFeatureProvider::new(
        runmat_geometry_io::exact::ExactCadImportOptions::default(),
        runmat_geometry_io::GeometryImportContext::new(),
    );
    let geometry = provider
        .execute(
            "unit-box",
            &ResolvedGeometryFeatureOperation::Box {
                width_m: runmat_types::physical::Length::new(1.0),
                depth_m: runmat_types::physical::Length::new(1.0),
                height_m: runmat_types::physical::Length::new(1.0),
            },
            &GeometryFeatureProviderSelection::None,
            &[],
            None,
        )
        .unwrap()
        .geometry;
    fs::write(
        project.path().join("src/unit-box.brep"),
        &geometry.representation,
    )
    .unwrap();
    let imported = runmat_geometry_io::exact::import_exact_cad(
        "unit-box.brep",
        &geometry.representation,
        runmat_geometry_io::GeometryFormat::Brep,
        &runmat_geometry_io::exact::ExactCadImportOptions {
            source_units: Some(runmat_geometry_core::LengthUnit::Meter),
            ..Default::default()
        },
        &runmat_geometry_io::GeometryImportContext::new(),
    )
    .unwrap();
    let source = STUDY
        .replace("ORDER", if quadratic { "quadratic" } else { "linear" })
        .replace("EQUATION", equation)
        .replace("LOW_V", &gauge.to_string())
        .replace("HIGH_V", &(gauge + voltage).to_string());
    let source = source
        .replace(
            "FIRST_FACE_ID",
            &imported.source_face_ids[&0]
                .analysis_selection_id()
                .unwrap(),
        )
        .replace(
            "SECOND_FACE_ID",
            &imported.source_face_ids[&1]
                .analysis_selection_id()
                .unwrap(),
        );
    fs::write(project.path().join("src/main.fea"), source).unwrap();
    [
        imported.source_face_ids[&0].clone(),
        imported.source_face_ids[&1].clone(),
    ]
}

pub(super) fn verify_faces(mesh: &SolverMeshArtifact, ids: &[PersistentEntityId; 2]) {
    let projection = runmat_analysis_fea::topology::AnalysisProjection::derive(mesh).unwrap();
    for (ordinal, id) in ids.iter().enumerate() {
        let normal = if ordinal == 0 { -1.0 } else { 1.0 };
        let mut area = 0.0;
        let mut count = 0;
        for face in &projection.boundary_faces {
            if !face.provenance.contains(id) {
                continue;
            }
            count += 1;
            for index in &face.node_indices {
                assert!(
                    (projection.nodes[*index].coordinates_m.canonical_value()[0] - ordinal as f64)
                        .abs()
                        < 1e-12
                );
            }
            assert!(normal * face.oriented_area_vector_m2[0] > 0.0);
            area += face.oriented_area_vector_m2[0].abs();
        }
        assert!(count > 0);
        assert!((area - 1.0).abs() < 1e-12, "face={ordinal} area={area}");
    }
}

const STUDY: &str = r#"
geometry part {
    file = "./unit-box.brep"
    units = m
    region body : solid { select result
        expect exactly_one }
    region low : face { select id "FIRST_FACE_ID"
        expect exactly_one }
    region high : face { select id "SECOND_FACE_ID"
        expect exactly_one }
}
mesh volume on geometry.part {
    tetrahedra { order = ORDER
        target_size = 0.5 m
        maximum_deviation = 0.02 m }
}
material dielectric {
    electrical {
        reference_temperature = 293.15 K
        conductivity = 3 A/V/m
        resistive_heating_coefficient = 0 K^-1
        relative_permittivity = 4
        relative_permeability = 1
    }
}
model electric {
    electromagnetic { equation = EQUATION }
    assign material.dielectric to geometry.part.region.body
    constraint low {
        electrical.potential { target = geometry.part.region.low
            value = LOW_V V }
    }
    constraint high {
        electrical.potential { target = geometry.part.region.high
            value = HIGH_V V }
    }
}
study response {
    geometry = geometry.part
    mesh = mesh.volume
    model = model.electric
    analysis potential {
        electromagnetic static {}
        accuracy { algebraic_relative_tolerance = 1e-11 }
    }
}
"#;
