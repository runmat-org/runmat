use super::*;
use crate::assembly::dofs::StructuralNodeDofSet;
use crate::excitation::{
    ComplexComponent, ComplexVector3, ProjectionSample, RealizedExcitationPolicy,
};
use runmat_analysis_core::{
    CoordinateFrame, ExcitationDistribution, ExcitationId, ExcitationKinematics,
    PhysicalQuantityKind, Position, ProjectionRequirement, RegionTarget, SignalId,
};
use runmat_meshing_core::{BoundaryTriangleOrder, CanonicalMeshingContract};

mod nodal_body_density;

fn apply_solver_mesh_structural_loads(
    excitations: &RealizedExcitations,
    mesh: &SolverMeshArtifact,
    constraint_plan: &CanonicalStructuralConstraintPlan,
    layout: &StructuralDofLayout,
    rhs: &mut [f64],
    strict_region_mapping: bool,
) -> Result<StructuralSourceProjection, SolverMeshRegionMappingError> {
    let projection = AnalysisProjection::derive(mesh).map_err(|error| {
        SolverMeshRegionMappingError::InvalidAnalysisProjection(error.to_string())
    })?;
    apply_solver_mesh_structural_loads_on_projection(
        excitations,
        mesh,
        &projection,
        constraint_plan,
        layout,
        rhs,
        strict_region_mapping,
    )
}
fn mesh(quadratic: bool, scale: f64) -> SolverMeshArtifact {
    let mut mesh = if quadratic {
        crate::topology::test_meshes::quadratic_slab_mesh(2)
    } else {
        crate::topology::test_meshes::slab_mesh(2)
    };
    // Two unequal strips of widths .2 and .8. The piecewise affine map preserves straight
    // Tri6/Tet10 midsides, but equal-node loading is not the consistent surface measure.
    for node in &mut mesh.topology.nodes {
        let y = node.coordinates_m.canonical_value()[1];
        {
            let quantity_change = if y <= 0.5 {
                0.4 * y
            } else {
                0.2 + 1.6 * (y - 0.5)
            };
            let quantity_index = 1;
            let quantity_slot = &mut node.coordinates_m;
            let mut quantity_coordinates = quantity_slot.canonical_value();
            quantity_coordinates[quantity_index] = quantity_change;
            *quantity_slot = runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(quantity_coordinates);
        };
        node.coordinates_m =
            runmat_types::physical::Quantity::<runmat_types::physical::kind::Position, _>::new(
                node.coordinates_m
                    .canonical_value()
                    .map(|coordinate| scale * coordinate),
            );
    }
    mesh.seal_canonical_digest().unwrap();
    mesh.validate_canonical().unwrap();
    mesh
}

fn force() -> RealizedExcitation {
    RealizedExcitation {
        excitation_id: ExcitationId("face-resultant".into()),
        case_ids: Vec::new(),
        signal_id: SignalId("complex-force".into()),
        quantity: PhysicalQuantityKind::Force,
        sample: ProjectionSample::Static,
        target: RealizedExcitationTarget::Region(RegionTarget::boundary(
            runmat_meshing_core::fixtures::fixture_face("slab_right"),
        )),
        kind: RealizedExcitationKind::Force(ComplexVector3 {
            real: [5.0, -7.0, 11.0],
            imaginary: [-3.0, 2.0, 4.0],
        }),
        policy: RealizedExcitationPolicy {
            frame: CoordinateFrame::Global,
            distribution: ExcitationDistribution::PreserveResultant,
            kinematics: ExcitationKinematics::FixedFrame,
            projection: ProjectionRequirement {
                preserve_resultant: true,
                preserve_moment: true,
                relative_tolerance: 1e-12,
            },
        },
    }
}

fn constraints(mesh: &SolverMeshArtifact) -> CanonicalStructuralConstraintPlan {
    let mut model =
        crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
    model.constraints.clear();
    CanonicalStructuralConstraintPlan::compile_for_model(&model, Some(mesh)).unwrap()
}

fn layout(mesh: &SolverMeshArtifact) -> StructuralDofLayout {
    StructuralDofLayout::from_node_sets(vec![
        StructuralNodeDofSet::translational();
        mesh.topology.nodes.len()
    ])
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn independent_weights(mesh: &SolverMeshArtifact) -> Vec<f64> {
    let projection = AnalysisProjection::derive(mesh).unwrap();
    let mut weights = vec![0.0; projection.nodes.len()];
    let mut area = 0.0;
    for face in projection
        .boundary_faces_matching(&runmat_meshing_core::fixtures::fixture_face("slab_right"))
    {
        let [a, b, c] = [0, 1, 2].map(|corner| {
            projection.nodes[face.node_indices[corner]]
                .coordinates_m
                .canonical_value()
        });
        let ab = std::array::from_fn(|axis| b[axis] - a[axis]);
        let ac = std::array::from_fn(|axis| c[axis] - a[axis]);
        let vector = cross(ab, ac);
        let triangle_area = 0.5 * vector.iter().map(|value| value * value).sum::<f64>().sqrt();
        area += triangle_area;
        // Independent simplex moments: integral lambda_i=A/3, lambda_i²=A/6,
        // lambda_i lambda_j=A/12. Therefore P2 vertex weight2*A/6−A/3=0,
        // edge weight4*A/12=A/3. No production boundary quadrature supplies the reference.
        let nodes = match face.order {
            BoundaryTriangleOrder::Tri3 => &face.node_indices[..3],
            BoundaryTriangleOrder::Tri6 => &face.node_indices[3..],
        };
        for &node in nodes {
            weights[node] += triangle_area / 3.0;
        }
    }
    assert!(area > 0.0);
    weights.iter_mut().for_each(|weight| *weight /= area);
    weights
}

#[test]
fn face_resultant_uses_consistent_nonuniform_tri3_and_tri6_surface_measure() {
    for quadratic in [false, true] {
        let mut unscaled = Vec::new();
        for scale in [1.0, 2.0] {
            let mesh = mesh(quadratic, scale);
            let plan = constraints(&mesh);
            let layout = layout(&mesh);
            let weights = independent_weights(&mesh);
            let mut components = Vec::new();
            for component in [ComplexComponent::Real, ComplexComponent::Imaginary] {
                let load = force();
                let RealizedExcitationKind::Force(value) = load.kind else {
                    unreachable!()
                };
                let expected = match component {
                    ComplexComponent::Real => value.real,
                    ComplexComponent::Imaginary => value.imaginary,
                };
                let excitations = RealizedExcitations::new(vec![load]).component(component);
                let mut rhs = vec![0.0; layout.total_dof_count()];
                let projected = apply_solver_mesh_structural_loads(
                    &excitations,
                    &mesh,
                    &plan,
                    &layout,
                    &mut rhs,
                    true,
                )
                .unwrap();
                assert_eq!(projected.evidence.len(), 1);
                let evidence = &projected.evidence[0];
                evidence.validate().unwrap();
                assert!(evidence.accepted);
                let crate::excitation::SpatialProjectionBalance::VectorResultant {
                    requested_resultant,
                    requested_moment,
                    ..
                } = evidence.balance
                else {
                    panic!("face force must retain vector balance evidence");
                };
                assert_eq!(requested_resultant.real, expected);
                assert_eq!(requested_moment.real, [0.0; 3]);
                assert_eq!(
                    evidence.integration_rule,
                    SpatialIntegrationRule::BoundaryDunavantSeven
                );
                let centroid = [scale, scale / 2.0, scale / 2.0];
                for (actual, expected) in evidence.reference_point_m.iter().zip(centroid) {
                    assert!((actual - expected).abs() < 1e-12);
                }
                let mut resultant = [0.0; 3];
                let mut moment = [0.0; 3];
                for ((node, &weight), nodal) in mesh
                    .topology
                    .nodes
                    .iter()
                    .zip(&weights)
                    .zip(rhs.chunks_exact(3))
                {
                    let force = [nodal[0], nodal[1], nodal[2]];
                    let nodal_moment = cross(node.coordinates_m.canonical_value(), force);
                    for axis in 0..3 {
                        assert!((force[axis] - weight * expected[axis]).abs() < 1e-11,
                            "quadratic={quadratic},scale={scale},node={:?}, weight={weight}, force={force:?}", node.coordinates_m.canonical_value());
                        resultant[axis] += force[axis];
                        moment[axis] += nodal_moment[axis];
                    }
                }
                let expected_moment = cross(centroid, expected);
                for axis in 0..3 {
                    assert!((resultant[axis] - expected[axis]).abs() < 1e-11);
                    assert!((moment[axis] - expected_moment[axis]).abs() < 1e-11);
                }
                components.push(rhs);
            }
            if scale == 1.0 {
                unscaled = components;
            } else {
                for (before, after) in unscaled.iter().flatten().zip(components.iter().flatten()) {
                    assert!(
                        (before - after).abs() < 1e-12,
                        "area scaling changed total-force projection"
                    );
                }
            }
        }
    }
}

#[test]
fn reference_point_force_stays_generalized_and_does_not_load_surface_nodes() {
    let mesh = mesh(true, 1.0);
    let mut plan = constraints(&mesh);
    let point = [2.0, 3.0, 4.0];
    plan.reference_points
        .push(crate::constraints::CanonicalStructuralReferencePoint {
            identity: runmat_execution::Digest::sha256(b"point-test"),
            reference_point_id: "remote".into(),
            position_meters: point,
        });
    let layout = layout(&mesh).with_reference_points(1);
    let mut load = force();
    load.target = RealizedExcitationTarget::ReferencePoint {
        reference_point_id: "remote".into(),
        position: Position::new(point),
    };
    let RealizedExcitationKind::Force(value) = load.kind else {
        unreachable!()
    };
    let mut rhs = vec![0.0; layout.total_dof_count()];
    let projected = apply_solver_mesh_structural_loads(
        &RealizedExcitations::new(vec![load]),
        &mesh,
        &plan,
        &layout,
        &mut rhs,
        true,
    )
    .unwrap();
    assert!(rhs[..mesh.topology.nodes.len() * 3]
        .iter()
        .all(|value| *value == 0.0));
    assert_eq!(&rhs[mesh.topology.nodes.len() * 3..][..3], &value.real);
    assert!(rhs[mesh.topology.nodes.len() * 3 + 3..]
        .iter()
        .all(|value| *value == 0.0));
    assert_eq!(
        projected.evidence[0].integration_rule,
        SpatialIntegrationRule::ReferencePointGeneralized
    );
    assert!(projected.evidence[0].accepted);
}
