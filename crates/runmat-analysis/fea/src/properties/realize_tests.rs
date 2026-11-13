use runmat_analysis_core::{
    AuthoredOrientationDefinition, AuthoredPropertyDefinitionRef, AuthoredSectionDefinition,
    CartesianOrientation, HomogeneousShellReferenceSurface, HomogeneousShellSection,
    PropertyRegionTarget, RealizedAnalysisModel, RealizedModelProperties,
    RealizedOrientationDefinition, RealizedPropertyAssignment, RealizedSectionDefinition,
    RectangularBeamSection,
};
use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};
use runmat_meshing_core::{ElementOrder, StableDigest};

use crate::fixtures::{fixture_model, FixtureId};

use super::realize::realize_formulation_properties;

#[test]
fn exact_edge_and_face_properties_generate_attached_formulation_topology() {
    let mut mesh =
        runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let edge = entity(PersistentEntityKind::Edge, "rail-edge");
    let face = entity(PersistentEntityKind::Face, "skin-face");
    mesh.topology.boundary_edges[0].provenance = vec![edge.clone()];
    mesh.topology.boundary_faces[0].provenance = vec![face.clone()];
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();

    let beam_id = definition("rail-section");
    let shell_id = definition("skin-section");
    let orientation_id = definition("rail-axes");
    let mut beam_target =
        PropertyRegionTarget::beam(runmat_meshing_core::fixtures::fixture_edge("rail"));
    beam_target.bind_resolved_entities(vec![edge]).unwrap();
    let mut shell_target =
        PropertyRegionTarget::shell(runmat_meshing_core::fixtures::fixture_face("skin"));
    shell_target.bind_resolved_entities(vec![face]).unwrap();
    let properties = RealizedModelProperties {
        sections: vec![
            RealizedSectionDefinition {
                definition_id: beam_id.clone(),
                definition: AuthoredSectionDefinition::RectangularBeam(RectangularBeamSection {
                    section_id: "rail-section".into(),
                    width_m: runmat_types::physical::Quantity::<
                        runmat_types::physical::kind::Length,
                        _,
                    >::new(0.02),
                    height_m: runmat_types::physical::Quantity::<
                        runmat_types::physical::kind::Length,
                        _,
                    >::new(0.04),
                }),
            },
            RealizedSectionDefinition {
                definition_id: shell_id.clone(),
                definition: AuthoredSectionDefinition::HomogeneousShell(HomogeneousShellSection {
                    section_id: "skin-section".into(),
                    thickness_m: runmat_types::physical::Quantity::<
                        runmat_types::physical::kind::Length,
                        _,
                    >::new(0.002),
                    reference_surface: HomogeneousShellReferenceSurface::MidSurface,
                    shear_correction: 5.0 / 6.0,
                }),
            },
        ],
        orientations: vec![RealizedOrientationDefinition {
            definition_id: orientation_id.clone(),
            definition: AuthoredOrientationDefinition::Cartesian(CartesianOrientation {
                orientation_id: "rail-axes".into(),
                origin_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new([0.0; 3]),
                x_axis: [1.0, 0.0, 0.0],
                y_axis: [0.0, 1.0, 0.0],
            }),
        }],
        layups: Vec::new(),
        assignments: vec![
            RealizedPropertyAssignment::Section {
                section: beam_id.clone(),
                target: beam_target.clone(),
            },
            RealizedPropertyAssignment::Orientation {
                orientation: orientation_id.clone(),
                target: beam_target.clone(),
            },
            RealizedPropertyAssignment::Section {
                section: shell_id.clone(),
                target: shell_target.clone(),
            },
        ],
    };
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.structural = None;
    let mut beam_material = model.material_assignments[0].clone();
    beam_material.target = beam_target.clone();
    let mut shell_material = model.material_assignments[0].clone();
    shell_material.target = shell_target;
    model.material_assignments.push(beam_material);
    model.material_assignments.push(shell_material);

    let mut overlapping = properties.clone();
    overlapping
        .assignments
        .push(RealizedPropertyAssignment::Section {
            section: beam_id,
            target: beam_target.clone(),
        });
    let error = realize_formulation_properties(
        RealizedAnalysisModel {
            model: model.clone(),
            properties: overlapping,
        },
        &mesh,
    )
    .expect_err("overlapping section ownership must fail closed");
    assert!(matches!(
        error,
        super::realize::PropertyRealizationError::SectionCoverage { count: 2, .. }
    ));

    let mut offset = properties.clone();
    let AuthoredSectionDefinition::HomogeneousShell(shell) = &mut offset.sections[1].definition
    else {
        panic!("fixture shell section changed kind");
    };
    shell.reference_surface = HomogeneousShellReferenceSurface::OffsetFromMidSurface(0.0005);
    let offset_model = realize_formulation_properties(
        RealizedAnalysisModel {
            model: model.clone(),
            properties: offset,
        },
        &mesh,
    )
    .expect("homogeneous shell offset must be retained");
    let runmat_analysis_core::ShellSectionModel::Homogeneous(offset_section) =
        &offset_model.structural.as_ref().unwrap().shell_sections[0]
    else {
        panic!("offset shell must remain homogeneous")
    };
    assert_eq!(
        offset_section.reference_surface,
        HomogeneousShellReferenceSurface::OffsetFromMidSurface(0.0005)
    );

    let mut degenerate = properties.clone();
    let AuthoredOrientationDefinition::Cartesian(orientation) =
        &mut degenerate.orientations[0].definition
    else {
        panic!("fixture orientation changed kind");
    };
    let edge_nodes = &mesh.topology.boundary_edges[0].node_ids;
    let from = mesh
        .topology
        .nodes
        .iter()
        .find(|node| node.node_id == edge_nodes[0])
        .expect("fixture edge start");
    let to = mesh
        .topology
        .nodes
        .iter()
        .find(|node| node.node_id == edge_nodes[1])
        .expect("fixture edge end");
    orientation.y_axis = std::array::from_fn(|index| {
        to.coordinates_m.canonical_value()[index] - from.coordinates_m.canonical_value()[index]
    });
    let error = realize_formulation_properties(
        RealizedAnalysisModel {
            model: model.clone(),
            properties: degenerate,
        },
        &mesh,
    )
    .expect_err("parallel beam orientation must fail closed");
    assert!(matches!(
        error,
        super::realize::PropertyRealizationError::DegenerateOrientation(_)
    ));

    let model = realize_formulation_properties(RealizedAnalysisModel { model, properties }, &mesh)
        .expect("geometry properties should realize");
    let structural = model
        .structural
        .as_ref()
        .expect("derived structural topology");
    assert_eq!(structural.elements.len(), 2);
    assert_eq!(structural.beam_sections.len(), 1);
    assert_eq!(structural.shell_sections.len(), 1);
    let runmat_analysis_core::ShellSectionModel::Homogeneous(shell_section) =
        &structural.shell_sections[0]
    else {
        panic!("fixture should realize a homogeneous shell section")
    };
    assert_eq!(
        shell_section.reference_surface,
        HomogeneousShellReferenceSurface::MidSurface
    );
    assert!(structural.nodes.iter().all(|node| matches!(
        node.solid_attachment,
        Some(runmat_analysis_core::ConformalSolidAttachment::SolverNode { .. })
    )));
    assert_eq!(
        structural.beam_sections[0].area_m2.canonical_value(),
        0.0008
    );
    let compiled = super::compile::compile_property_map(&model, Some(&mesh))
        .expect("derived elements should consume the canonical material compiler");
    assert_eq!(compiled.beam_properties().len(), 1);
    assert_eq!(compiled.shell_properties().len(), 1);
}

fn definition(property_id: &str) -> AuthoredPropertyDefinitionRef {
    AuthoredPropertyDefinitionRef {
        module_path: "project/main.fea".into(),
        model_id: "structure".into(),
        property_id: property_id.into(),
    }
}

fn entity(kind: PersistentEntityKind, source_topology_id: &str) -> PersistentEntityId {
    PersistentEntityId::new(kind, source_topology_id.into(), vec!["root".into()])
}
