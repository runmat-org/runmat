//! Analytic physical integrals on grouped, exact original exterior authority.
use super::*;
use runmat_analysis_core::{
    AnalysisBoundary, BoundaryOperator, FluidConstraint, FluidConstraintKind, PrescribedValue,
    Pressure, RegionTarget, Velocity,
};

// witness: FEA-FLUID-RECOVERY-001/balance @
//   incompressible_laminar_steady/canonical_tetrahedra/numerical_kernel;
//   boundary: Independent constant translation exact group signs/pressure forces; no general
//   wall-shear convergence.
#[test]
fn grouped_original_translation_reports_signed_mass_and_pressure_forces() {
    for reference in [0.0, 101325.0] {
        let mut f = case(2, ReferenceCase::Couette, false);
        let mut walls = RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
            "inlet_and_walls",
        ));
        let entities = f
            .constraints
            .iter()
            .flat_map(|c| {
                let FluidConstraintKind::PrescribedVelocity { target, .. } = &c.kind else {
                    unreachable!()
                };
                target.resolved_entities().iter().cloned()
            })
            .collect();
        walls.bind_resolved_entities(entities).unwrap();
        let mut outlet =
            RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("outlet"));
        outlet
            .bind_resolved_entities(
                f.operators
                    .iter()
                    .flat_map(|b| b.target.resolved_entities().iter().cloned())
                    .collect(),
            )
            .unwrap();
        let exact = [0.001, 0.0, 0.0];
        let density = 7.0;
        f.material.density = runmat_types::physical::Quantity::new(density);
        f.constraints = vec![FluidConstraint {
            constraint_id: "signed_inlet_and_walls".into(),
            kind: FluidConstraintKind::PrescribedVelocity {
                target: walls,
                value: PrescribedValue::Constant(exact.map(Velocity::new)),
            },
        }];
        f.operators = vec![AnalysisBoundary {
            boundary_id: "signed_outlet".into(),
            target: outlet,
            operator: BoundaryOperator::FluidOutletPressure {
                value: PrescribedValue::Constant(Pressure::new(reference)),
            },
        }];
        let p = CanonicalFluidProblem::prepare(f.input()).unwrap();
        let q = f
            .topology
            .faces
            .iter()
            .map(|face| geometry_ops::dot(exact, face.owner_area_vector_m2))
            .collect::<Vec<_>>();
        let r = recovery::evaluate(
            &p,
            &vec![exact; f.topology.cells.len()],
            &vec![0.0; f.topology.cells.len()],
            &q,
        )
        .unwrap();
        assert!(r.accepted(1e-8));
        let reports = &r.boundary_balances;
        assert_eq!(reports.len(), 2);
        assert_eq!(reports[0].authority_id, "signed_inlet_and_walls");
        assert_eq!(reports[1].authority_id, "signed_outlet");
        for (report, sign, area) in [(&reports[0], -1.0, 5.0), (&reports[1], 1.0, 1.0)] {
            assert!((report.area_m2.canonical_value() - area).abs() < 1e-14);
            assert!(
                (report.volume_flow_rate_m3_per_s.canonical_value() - sign * exact[0]).abs()
                    < 1e-15
            );
            assert!(
                (report.mass_flow_rate_kg_per_s.canonical_value() - density * sign * exact[0])
                    .abs()
                    < 1e-15
            );
            assert!(
                norm(sub(
                    report.advective_momentum_n.canonical_value(),
                    [density * sign * exact[0].powi(2), 0.0, 0.0]
                )) < 1e-18
            );
            assert_eq!(report.viscous_traction_n.canonical_value(), [0.0; 3]);
            assert_eq!(
                report.pressure_fluctuation_traction_n.canonical_value(),
                [0.0; 3]
            );
            assert!(
                norm(sub(
                    report.pressure_traction_n.canonical_value(),
                    [-sign * reference, 0.0, 0.0]
                )) < 1e-10
            );
        }
        assert_eq!(
            reports.iter().map(|r| r.face_count).sum::<u64>(),
            f.topology
                .faces
                .iter()
                .filter(|face| face.neighbor.is_none())
                .count() as u64
        );
    }
}

// witness: FEA-FLUID-RECOVERY-001/fields @
//   incompressible_laminar_steady/canonical_tetrahedra/numerical_kernel;
//   boundary: Couette boundary shear/source attribution preserved under cold recovery; one fixture
//   only.
#[test]
fn original_couette_boundary_shear_and_cold_attribution_are_preserved() {
    let f = case(2, ReferenceCase::Couette, false);
    let p = CanonicalFluidProblem::prepare(f.input()).unwrap();
    let result = p.solve().unwrap();
    let cold: CanonicalFluidResult =
        serde_json::from_slice(&serde_json::to_vec(&result).unwrap()).unwrap();
    assert_eq!(p.recover(&cold).unwrap(), result.residual);
    let mut top = 0.0;
    let mut bottom = 0.0;
    for report in &result.residual.boundary_balances {
        let constraint = f
            .constraints
            .iter()
            .find(|c| c.constraint_id == report.authority_id);
        let Some(constraint) = constraint else {
            continue;
        };
        let FluidConstraintKind::PrescribedVelocity { target, .. } = &constraint.kind else {
            unreachable!()
        };
        let face = f
            .source
            .boundary_faces
            .iter()
            .find(|face| {
                face.provenance
                    .iter()
                    .any(|entity| target.matches_persistent_entity(entity))
            })
            .unwrap();
        if (face.centroid_m[1] - 1.0).abs() < 1e-12 {
            top += report.viscous_traction_n.canonical_value()[0];
        }
        if face.centroid_m[1].abs() < 1e-12 {
            bottom += report.viscous_traction_n.canonical_value()[0];
        }
    }
    assert!((top - 1.0).abs() < 0.25 && (bottom + 1.0).abs() < 0.25);
    let mut changed = cold.clone();
    changed.residual.boundary_balances[0].authority_id = "foreign".into();
    assert!(p.recover(&changed).is_err());
    let mut changed = cold;
    changed.residual.boundary_balances.reverse();
    assert!(p.recover(&changed).is_err());
}
