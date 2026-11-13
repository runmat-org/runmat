use super::*;
use runmat_geometry_ops::{
    GeometryFeatureProvider, GeometryFeatureProviderSelection, ResolvedGeometryFeatureOperation,
};
use runmat_meshing_core::{PersistentEntityId, SolverMeshArtifact};

pub(super) fn prepare(
    project: &TempDir,
    drive: Drive,
    quadratic: bool,
    size: f64,
    generated: bool,
) -> [PersistentEntityId; 2] {
    fs::create_dir(project.path().join("src")).unwrap();
    fs::write(project.path().join("runmat.toml"),
        "[package]\nname = \"matched-duct\"\n[sources]\nroots = [\"src\"]\n[runtime.fea]\nartifact_root = \"artifacts\"\nstudy_artifact_root = \"study-artifacts\"\n").unwrap();
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
    let (quantity, action, expression, explicit) = match drive {
        Drive::NormalVelocity => (
            "velocity",
            "normal_velocity",
            "-(1 + 0.4i) * (0 * frequency + 1) / 6",
            EXPLICIT,
        ),
        Drive::IncidentPressure => (
            "pressure",
            "incident_pressure",
            "complex(1 + 0 * frequency, 0.4 + 0 * frequency)",
            INCIDENT_EXPLICIT,
        ),
        Drive::VolumeVelocity => (
            "volume_flow_rate",
            "volume_velocity",
            "-(1 + 0.4i) * (0 * frequency + 1) / 6",
            VOLUME_EXPLICIT,
        ),
    };
    fs::write(project.path().join("src/drive.m"), format!(
        "function value = drive(frequency)\narguments\nfrequency (:,1) double\nend\nvalue = {expression};\nend\n")).unwrap();
    let source = STUDY
        .replace("ORDER", if quadratic { "quadratic" } else { "linear" })
        .replace("SIZE", &size.to_string())
        .replace("QUANTITY", quantity)
        .replace("ACTION", action)
        .replace("SOURCE", if generated { GENERATED } else { explicit });
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

pub(super) fn verify_faces(mesh: &SolverMeshArtifact, identities: &[PersistentEntityId; 2]) {
    let projection = runmat_analysis_fea::topology::AnalysisProjection::derive(mesh).unwrap();
    for (ordinal, identity) in identities.iter().enumerate() {
        let x = ordinal as f64;
        let normal = if ordinal == 0 { -1.0 } else { 1.0 };
        let mut area = 0.0;
        let mut count = 0;
        for face in &projection.boundary_faces {
            if !face.provenance.contains(identity) {
                continue;
            }
            count += 1;
            for index in &face.node_indices {
                assert!(
                    (projection.nodes[*index].coordinates_m.canonical_value()[0] - x).abs() < 1e-12
                );
            }
            assert!(normal * face.oriented_area_vector_m2[0] > 0.0);
            area += face.oriented_area_vector_m2[0].abs();
        }
        assert!(count > 0, "authored face ordinal {ordinal} missing");
        assert!((area - 1.0).abs() < 1e-12, "ordinal={ordinal} area={area}");
    }
}

pub(super) fn verify_medium(
    model: &runmat_analysis_core::AnalysisModel,
    mesh: &SolverMeshArtifact,
) {
    let properties =
        runmat_analysis_fea::properties::compile_solid_property_map(model, mesh).unwrap();
    properties.validate_integrity().unwrap();
    assert_eq!(
        properties.solid_materials().len(),
        mesh.topology.volume_elements.len()
    );
    for element in &mesh.topology.volume_elements {
        let owner = properties.solid_material(element.stable_identity).unwrap();
        assert_eq!(owner.element_id, element.element_id);
        assert_eq!(
            owner
                .material
                .mass_density_kg_per_m3
                .map(|quantity| quantity.canonical_value()),
            Some(3.0)
        );
        let acoustic = owner.material.acoustic.as_ref().unwrap();
        assert_eq!(acoustic.speed_of_sound_m_per_s.canonical_value(), 2.0);
        assert_eq!(acoustic.damping_ratio.canonical_value(), 0.0);
    }
    assert_eq!(model.boundaries.len(), 1);
    let runmat_analysis_core::BoundaryOperator::AcousticImpedance { impedance } =
        &model.boundaries[0].operator
    else {
        panic!("expected matched impedance, not radiation");
    };
    assert_eq!(impedance.canonical_value(), 6.0);
}

pub(super) fn verify_drive(
    model: &runmat_analysis_core::AnalysisModel,
    drive: Drive,
    generated: bool,
) {
    use runmat_analysis_core::{ExcitationAction, PhysicalSignal, SignalProvenance};
    assert_eq!(model.excitations.len(), 1);
    assert_eq!(model.signals.len(), 1);
    let provenance = match (drive, &model.excitations[0], &model.signals[0]) {
        (
            Drive::NormalVelocity,
            ExcitationAction::NormalVelocity(_),
            PhysicalSignal::Velocity(s),
        ) => &s.provenance,
        (
            Drive::IncidentPressure,
            ExcitationAction::AcousticIncidentPressure(_),
            PhysicalSignal::Pressure(s),
        ) => &s.provenance,
        (
            Drive::VolumeVelocity,
            ExcitationAction::VolumeVelocity(action),
            PhysicalSignal::VolumeFlowRate(s),
        ) => {
            assert_eq!(
                action.distribution,
                runmat_analysis_core::ExcitationDistribution::PreserveResultant
            );
            assert!(action.projection.preserve_resultant && !action.projection.preserve_moment);
            &s.provenance
        }
        _ => panic!("source must retain its exact typed acoustic action and quantity"),
    };
    assert_eq!(
        matches!(provenance, SignalProvenance::Generated(_)),
        generated
    );
    assert!(model.signals[0].generator().is_none());
}

const GENERATED: &str = r#"
generated by function.drive {
    arguments {}
    sample_points = [0.2 Hz, 0.4 Hz]
    precision = complex128
    interpolation = linear
    extrapolation = reject
}
"#;
const EXPLICIT: &str = r#"
samples = [
    { at = 0.2 Hz, value = { real = -0.16666666666666666 m/s, imaginary = -0.06666666666666667 m/s } },
    { at = 0.4 Hz, value = { real = -0.16666666666666666 m/s, imaginary = -0.06666666666666667 m/s } }
]
interpolation = linear
extrapolation = reject
"#;
const INCIDENT_EXPLICIT: &str = r#"
samples = [
    { at = 0.2 Hz, value = { real = 1 Pa, imaginary = 0.4 Pa } },
    { at = 0.4 Hz, value = { real = 1 Pa, imaginary = 0.4 Pa } }
]
interpolation = linear
extrapolation = reject
"#;
const VOLUME_EXPLICIT: &str = r#"
samples = [
    { at = 0.2 Hz, value = { real = -0.16666666666666666 m^3/s, imaginary = -0.06666666666666667 m^3/s } },
    { at = 0.4 Hz, value = { real = -0.16666666666666666 m^3/s, imaginary = -0.06666666666666667 m^3/s } }
]
interpolation = linear
extrapolation = reject
"#;
const STUDY: &str = r#"
import function drive from "./drive.m"
geometry part {
    file = "./unit-box.brep"
    units = m
    region body : solid {
        select result
        expect exactly_one
    }
    region inlet : face {
        select id "FIRST_FACE_ID"
        expect exactly_one
    }
    region outlet : face {
        select id "SECOND_FACE_ID"
        expect exactly_one
    }
}
mesh volume on geometry.part {
    tetrahedra {
        order = ORDER
        target_size = SIZE m
        maximum_deviation = 0.02 m
    }
}
material fluid {
    mass_density = 3 kg/m^3
    acoustic {
        speed_of_sound = 2 m/s
        damping_ratio = 0
    }
}
model duct {
    assign material.fluid to geometry.part.region.body
    load termination {
        acoustic.impedance {
            target = geometry.part.region.outlet
            impedance = 6 Pa*s/m
        }
    }
    signal drive : QUANTITY over frequency {
        SOURCE
    }
    excitation inlet {
        ACTION {
            target = geometry.part.region.inlet
            signal = signal.drive
        }
    }
    load_case driven {
        include excitation.inlet
    }
}
study response {
    geometry = geometry.part
    mesh = mesh.volume
    model = model.duct
    analysis sound {
        acoustic harmonic {
            frequencies = [0.2 Hz]
        }
        accuracy { algebraic_relative_tolerance = 1e-9 }
    }
}
"#;
