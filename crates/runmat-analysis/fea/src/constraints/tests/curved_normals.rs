use super::*;

#[test]
fn curved_tri6_normal_displacement_uses_local_midside_tangent() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
    // Face 0 is the z=0 triangle. Bowing edge 1-2 toward positive z by h
    // yields the exact area vector (0, 2h, 1) at its midside node. The
    // outward normal is its negative; the corner plane alone remains z=0.
    let bow_m = 0.1;
    mesh.topology.nodes[4].coordinates_m[2] = bow_m;
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();
    model.geometry_revision = mesh.geometry.geometry_revision as u32;
    model.constraints = vec![AnalysisConstraint::Structural(StructuralConstraint {
        constraint_id: "curved_normal".into(),
        frame: CoordinateFrame::BoundaryNormalTangent,
        kind: StructuralConstraintKind::NormalDisplacement {
            target: RegionTarget::boundary("face:0"),
            value: PrescribedValue::Constant(Displacement::new(0.002)),
        },
    })];

    let plan = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();
    assert_eq!(plan.rows.len(), 6);
    let midside = mesh.topology.nodes[4].stable_identity;
    let row = plan
        .rows
        .iter()
        .find(|row| row.terms[0].owner == CanonicalStructuralDofOwner::StructuralNode(midside))
        .unwrap();
    let local = row_vector(row);
    let norm = (1.0 + 4.0 * bow_m * bow_m).sqrt();
    for (actual, expected) in local
        .into_iter()
        .zip([0.0, -2.0 * bow_m / norm, -1.0 / norm])
    {
        assert!(
            (actual - expected).abs() <= 1.0e-12,
            "actual={actual} expected={expected}"
        );
    }
    assert_eq!(row.right_hand_side, 0.002);

    model.constraints = vec![AnalysisConstraint::Structural(StructuralConstraint {
        constraint_id: "curved_symmetry".into(),
        frame: CoordinateFrame::BoundaryNormalTangent,
        kind: StructuralConstraintKind::Symmetry {
            target: RegionTarget::boundary("face:0"),
        },
    })];
    let symmetry = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();
    let symmetry_row = symmetry
        .rows
        .iter()
        .find(|row| row.terms[0].owner == CanonicalStructuralDofOwner::StructuralNode(midside))
        .unwrap();
    for (actual, expected) in row_vector(symmetry_row).into_iter().zip(local) {
        assert!((actual - expected).abs() <= 1.0e-12);
    }

    model.constraints = vec![AnalysisConstraint::Structural(StructuralConstraint {
        constraint_id: "curved_antisymmetry".into(),
        frame: CoordinateFrame::BoundaryNormalTangent,
        kind: StructuralConstraintKind::Antisymmetry {
            target: RegionTarget::boundary("face:0"),
        },
    })];
    let antisymmetry = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();
    let tangents = antisymmetry
        .rows
        .iter()
        .filter(|row| row.terms[0].owner == CanonicalStructuralDofOwner::StructuralNode(midside))
        .collect::<Vec<_>>();
    assert_eq!(tangents.len(), 2);
    for tangent in tangents {
        let dot = row_vector(tangent)
            .into_iter()
            .zip(local)
            .map(|(component, normal)| component * normal)
            .sum::<f64>();
        assert!(dot.abs() <= 1.0e-12);
    }
}
