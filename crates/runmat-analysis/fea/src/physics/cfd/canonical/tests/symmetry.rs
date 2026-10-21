//! Shear-free impermeable planes: plane Poiseuille flow between no-slip walls with
//! symmetry planes on the spanwise faces. The reference u = G y (1 − y) / (2µ) is a
//! continuum expression, never a residual-generated forcing.
use super::*;
use runmat_analysis_core::{
    AnalysisBoundary, BoundaryOperator, FluidConstraint, FluidConstraintKind, PrescribedValue,
    Pressure, RegionTarget, Velocity,
};

fn plane_poiseuille(y: f64) -> f64 {
    0.5 * y * (1.0 - y)
}

fn target(id: &str, entities: Vec<runmat_meshing_core::PersistentEntityId>) -> RegionTarget {
    let mut target = RegionTarget::boundary(id);
    target.bind_resolved_entities(entities).unwrap();
    target
}

fn symmetry_case(n: usize, kind: fn(RegionTarget) -> FluidConstraintKind) -> fixture::Fixture {
    let mut f = case(n, ReferenceCase::Duct, false);
    let mut inlet = Vec::new();
    let mut walls = Vec::new();
    let mut planes = Vec::new();
    let mut outlet = Vec::new();
    for face in &f.source.boundary_faces {
        let x = face.centroid_m;
        let entities = face.provenance.clone();
        if (x[0] - 1.0).abs() < 1e-12 {
            outlet.extend(entities);
        } else if x[0].abs() < 1e-12 {
            inlet.push((x, entities));
        } else if x[1].abs() < 1e-12 || (x[1] - 1.0).abs() < 1e-12 {
            walls.extend(entities);
        } else {
            planes.extend(entities);
        }
    }
    let mut constraints = inlet
        .into_iter()
        .enumerate()
        .map(|(i, (x, entities))| FluidConstraint {
            constraint_id: format!("inlet-{i}"),
            kind: FluidConstraintKind::PrescribedVelocity {
                target: target(&format!("inlet-{i}"), entities),
                value: PrescribedValue::Constant(
                    [plane_poiseuille(x[1]), 0.0, 0.0].map(Velocity::new),
                ),
            },
        })
        .collect::<Vec<_>>();
    constraints.push(FluidConstraint {
        constraint_id: "walls".into(),
        kind: FluidConstraintKind::NoSlipWall {
            target: target("walls", walls),
        },
    });
    constraints.push(FluidConstraint {
        constraint_id: "spanwise".into(),
        kind: kind(target("spanwise", planes)),
    });
    f.constraints = constraints;
    f.operators = vec![AnalysisBoundary {
        boundary_id: "outlet".into(),
        target: target("outlet", outlet),
        operator: BoundaryOperator::FluidOutletPressure {
            value: PrescribedValue::Constant(Pressure::new(0.0)),
        },
    }];
    f
}

fn error(f: &fixture::Fixture, r: &CanonicalFluidResult) -> (f64, f64) {
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    let mut spanwise = 0.0f64;
    for (c, u) in f.topology.cells.iter().zip(&r.velocity_m_per_s) {
        let exact = [plane_poiseuille(c.centroid_m[1]), 0.0, 0.0];
        numerator += c.volume_m3 * norm(sub(*u, exact)).powi(2);
        denominator += c.volume_m3 * norm(exact).powi(2);
        spanwise = spanwise.max(u[2].abs());
    }
    ((numerator / denominator).sqrt(), spanwise)
}

#[test]
fn symmetry_planes_reproduce_plane_poiseuille_flow_under_refinement() {
    let mut previous = f64::INFINITY;
    for n in [2, 3, 4] {
        let f = symmetry_case(n, |target| FluidConstraintKind::Symmetry { target });
        let p = CanonicalFluidProblem::prepare(f.input()).unwrap();
        let r = p.solve().unwrap();
        p.recover(&r).unwrap();
        let (relative, spanwise) = error(&f, &r);
        assert!(relative < previous, "n={n}: {relative} !< {previous}");
        assert!(relative < 0.2, "n={n}: {relative}");
        assert!(spanwise < 1e-6, "n={n}: spanwise velocity {spanwise}");
        for (i, face) in f.topology.faces.iter().enumerate() {
            if matches!(p.boundaries[i], boundary::FaceBoundary::Symmetry { .. }) {
                assert_eq!(r.face_volume_flux_m3_per_s[i], 0.0);
                let shear = r.residual.face_forces.viscous_traction_n[i];
                let normal = geometry_ops::scale(face.owner_area_vector_m2, 1.0 / face.area_m2);
                let tangential = sub(
                    shear,
                    geometry_ops::scale(normal, geometry_ops::dot(shear, normal)),
                );
                assert!(norm(tangential) < 1e-12, "symmetry plane carries shear");
            }
        }
        let balance = r
            .residual
            .boundary_balances
            .iter()
            .find(|b| b.authority_id == "spanwise")
            .unwrap();
        assert_eq!(balance.volume_flow_rate_m3_per_s, 0.0);
        previous = relative;
    }
}

#[test]
fn slip_walls_share_the_symmetry_plane_law() {
    let symmetric = symmetry_case(2, |target| FluidConstraintKind::Symmetry { target });
    let slip = symmetry_case(2, |target| FluidConstraintKind::SlipWall { target });
    let a = CanonicalFluidProblem::prepare(symmetric.input())
        .unwrap()
        .solve()
        .unwrap();
    let b = CanonicalFluidProblem::prepare(slip.input())
        .unwrap()
        .solve()
        .unwrap();
    assert_eq!(a.velocity_m_per_s, b.velocity_m_per_s);
    assert_eq!(a.pressure_pa, b.pressure_pa);
}

#[test]
fn retained_symmetry_flux_substitution_is_rejected() {
    let f = symmetry_case(2, |target| FluidConstraintKind::Symmetry { target });
    let p = CanonicalFluidProblem::prepare(f.input()).unwrap();
    let mut r = p.solve().unwrap();
    let face = p
        .boundaries
        .iter()
        .position(|b| matches!(b, boundary::FaceBoundary::Symmetry { .. }))
        .unwrap();
    r.face_volume_flux_m3_per_s[face] = 1e-9;
    assert!(p.recover(&r).is_err());
}
