//! Closed-form affine triangle integrals, independent of production quadrature.
use super::*;
use crate::{
    excitation::{ProjectionSample, RealizedExcitationPolicy},
    properties::compile_solid_property_map,
    topology::{
        test_meshes::{quadratic_slab_mesh, slab_mesh},
        AnalysisProjection,
    },
};
use runmat_analysis_core::{
    CoordinateFrame, ExcitationId, PhysicalQuantityKind, ProjectionRequirement,
    PropertyRegionTarget, SignalId, SignalSampleApproach,
};

fn assembly(quadratic: bool) -> AcousticAssembly {
    let mesh = if quadratic {
        quadratic_slab_mesh(2)
    } else {
        slab_mesh(2)
    };
    let mut model = crate::physics::acoustic::canonical::tests::model();
    model.material_assignments[0].target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("slab"));
    let properties = compile_solid_property_map(&model, &mesh).unwrap();
    AcousticAssembly::assemble(
        AnalysisProjection::derive(&mesh).unwrap(),
        &properties,
        runmat_types::physical::AngularFrequency::new(1.0),
    )
    .unwrap()
}

fn load(flow: ComplexScalar) -> RealizedExcitation {
    RealizedExcitation {
        excitation_id: ExcitationId("total_flow".into()),
        case_ids: vec![],
        signal_id: SignalId("flow".into()),
        quantity: PhysicalQuantityKind::VolumeFlowRate,
        sample: ProjectionSample::FrequencyHertz {
            value: 1.0,
            approach: SignalSampleApproach::At,
        },
        target: RealizedExcitationTarget::Region(RegionTarget::boundary(
            runmat_meshing_core::fixtures::fixture_face("slab_left"),
        )),
        kind: RealizedExcitationKind::VolumeVelocity(flow),
        policy: RealizedExcitationPolicy {
            frame: CoordinateFrame::BoundaryNormalTangent,
            distribution: ExcitationDistribution::PreserveResultant,
            kinematics: ExcitationKinematics::FixedFrame,
            projection: ProjectionRequirement {
                preserve_resultant: true,
                preserve_moment: false,
                relative_tolerance: 1e-10,
            },
        },
    }
}

fn triangle_area(assembly: &AcousticAssembly, face: &ProjectedBoundaryFace) -> f64 {
    let corners: [[f64; 3]; 3] = std::array::from_fn(|i| {
        assembly.projection.nodes[face.node_indices[i]]
            .coordinates_m
            .canonical_value()
    });
    let a: [f64; 3] = std::array::from_fn(|i| corners[1][i] - corners[0][i]);
    let b: [f64; 3] = std::array::from_fn(|i| corners[2][i] - corners[0][i]);
    let cross = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    0.5 * cross.into_iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[test]
fn total_flow_normalizes_once_over_union_with_exact_tri3_tri6_moments() {
    for quadratic in [false, true] {
        for both_sides in [false, true] {
            let mut a = assembly(quadratic);
            let mut faces = a
                .faces(&RegionTarget::boundary(
                    runmat_meshing_core::fixtures::fixture_face("slab_left"),
                ))
                .unwrap();
            if both_sides {
                faces.extend(
                    a.faces(&RegionTarget::boundary(
                        runmat_meshing_core::fixtures::fixture_face("slab_right"),
                    ))
                    .unwrap(),
                );
            }
            let area = faces
                .iter()
                .map(|face| triangle_area(&a, face))
                .sum::<f64>();
            assert!((area - if both_sides { 2.0 } else { 1.0 }).abs() < 1e-14);
            let mut expected = vec![0.0; a.projection.nodes.len()];
            for face in &faces {
                let face_area = triangle_area(&a, face);
                for (i, &node) in face.node_indices.iter().enumerate() {
                    // Tri3: integral N=A/3. Tri6: corners zero, edge mids A/3.
                    if !quadratic || i >= 3 {
                        expected[node] += face_area / (3.0 * area);
                    }
                }
            }
            let flow = ComplexScalar {
                real: 0.7,
                imaginary: -0.2,
            };
            a.add_volume_velocity(&load(flow), flow, faces).unwrap();
            for (source, weight) in a.source.canonical_values().iter().zip(expected) {
                assert!(
                    (source.real - a.derivative_rate.canonical_value() * flow.imaginary * weight)
                        .abs()
                        < 1e-13
                );
                assert!(
                    (source.imaginary + a.derivative_rate.canonical_value() * flow.real * weight)
                        .abs()
                        < 1e-13
                );
            }
            assert_eq!(a.source_evidence.len(), 1);
            assert!(a.source_evidence[0].accepted);
            assert_eq!(
                a.source_evidence[0].quantity,
                PhysicalQuantityKind::VolumeFlowRate
            );
            assert!(a.source_evidence[0].resultant_relative_error() < 1e-13);
        }
    }
}

#[test]
fn total_flow_rejects_empty_duplicate_and_invalid_ownership_or_distribution() {
    let flow = ComplexScalar::real(1.0);
    let a = assembly(false);
    let face = a
        .faces(&RegionTarget::boundary(
            runmat_meshing_core::fixtures::fixture_face("slab_left"),
        ))
        .unwrap()
        .remove(0);
    let mut invalid_patches = vec![vec![], vec![face.clone(), face.clone()]];
    for owners in [vec![], vec![0, 0], vec![usize::MAX]] {
        let mut invalid = face.clone();
        invalid.adjacent_element_indices = owners;
        invalid_patches.push(vec![invalid]);
    }
    let mut interior = face.clone();
    interior.role = BoundaryFaceRole::ConformalInterface;
    invalid_patches.push(vec![interior]);
    for faces in invalid_patches {
        assert!(assembly(false)
            .add_volume_velocity(&load(flow), flow, faces)
            .is_err());
    }
    let mut invalid = load(flow);
    invalid.policy.distribution = ExcitationDistribution::UniformKinematic;
    assert!(assembly(false)
        .add_volume_velocity(&invalid, flow, vec![face])
        .is_err());
}
