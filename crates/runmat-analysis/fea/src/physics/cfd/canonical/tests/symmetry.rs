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
    symmetry_case_on(fixture::case(n, ReferenceCase::Duct, false), kind)
}

fn symmetry_case_on(
    mut f: fixture::Fixture,
    kind: fn(RegionTarget) -> FluidConstraintKind,
) -> fixture::Fixture {
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
    // Tetrahedral cells are not mirror-symmetric about the spanwise mid-plane, so the
    // spanwise velocity is a discretization error: small against the peak 0.125 m/s and
    // shrinking under refinement, never an exact zero.
    let mut previous = f64::INFINITY;
    let mut spanwise_history = Vec::new();
    for n in [2, 3, 4] {
        let f = symmetry_case(n, |target| FluidConstraintKind::Symmetry { target });
        let p = CanonicalFluidProblem::prepare(f.input()).unwrap();
        let r = p.solve().unwrap();
        p.recover(&r).unwrap();
        let (relative, spanwise) = error(&f, &r);
        assert!(relative < previous, "n={n}: {relative} !< {previous}");
        assert!(relative < 0.2, "n={n}: {relative}");
        assert!(
            spanwise < 0.05 * 0.125,
            "n={n}: spanwise velocity {spanwise}"
        );
        spanwise_history.push(spanwise);
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
    assert!(
        spanwise_history[2] < spanwise_history[0],
        "spanwise velocity does not shrink under refinement: {spanwise_history:?}"
    );
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

// witness: FEA-FLUID-INPUTS-001/boundaries @
//   incompressible_laminar_steady/canonical_tetrahedra/numerical_kernel;
//   coverage: partial; boundary: a closed component bounded by velocity walls and symmetry planes
//   (a spanwise slab of a driven cavity) takes its pressure level from the authored gauge and
//   reproduces Couette flow; symmetry planes carry no volume flux.
#[test]
fn closed_component_with_symmetry_planes_reproduces_couette_flow() {
    let mut f = case(3, ReferenceCase::Duct, false);
    let (mut lid, mut planes) = (Vec::new(), Vec::new());
    for face in &f.source.boundary_faces {
        let x = face.centroid_m;
        let entities = face.provenance.clone();
        if (x[1] - 1.0).abs() < 1e-12 {
            lid.extend(entities);
        } else if x[2].abs() < 1e-12 || (x[2] - 1.0).abs() < 1e-12 {
            planes.extend(entities);
        }
    }
    // Couette flow u = y between a stationary floor and a lid moving at 1 m/s, with the end
    // walls moving with the exact profile so the closed component is consistent.
    let mut constraints = vec![
        FluidConstraint {
            constraint_id: "lid".into(),
            kind: FluidConstraintKind::PrescribedVelocity {
                target: target("lid", lid.clone()),
                value: PrescribedValue::Constant([1.0, 0.0, 0.0].map(Velocity::new)),
            },
        },
        FluidConstraint {
            constraint_id: "spanwise".into(),
            kind: FluidConstraintKind::Symmetry {
                target: target("spanwise", planes),
            },
        },
        FluidConstraint {
            constraint_id: "gauge".into(),
            kind: FluidConstraintKind::PressureReference {
                target: target("gauge", lid),
                value: PrescribedValue::Constant(Pressure::new(0.0)),
            },
        },
    ];
    let mut by_height = std::collections::BTreeMap::<i64, Vec<_>>::new();
    for face in &f.source.boundary_faces {
        let x = face.centroid_m;
        let interior_side = (x[1] - 1.0).abs() >= 1e-12
            && x[2].abs() >= 1e-12
            && (x[2] - 1.0).abs() >= 1e-12;
        if interior_side {
            by_height
                .entry((x[1] * 1e9).round() as i64)
                .or_default()
                .extend(face.provenance.clone());
        }
    }
    for (i, (height, entities)) in by_height.into_iter().enumerate() {
        let y = height as f64 * 1e-9;
        constraints.push(FluidConstraint {
            constraint_id: format!("end-{i}"),
            kind: FluidConstraintKind::PrescribedVelocity {
                target: target(&format!("end-{i}"), entities),
                value: PrescribedValue::Constant([y, 0.0, 0.0].map(Velocity::new)),
            },
        });
    }
    f.constraints = constraints;
    f.operators = Vec::new();
    let p = CanonicalFluidProblem::prepare(f.input()).unwrap();
    let r = p.solve().unwrap();
    let mut worst = 0.0f64;
    for (c, u) in f.topology.cells.iter().zip(&r.velocity_m_per_s) {
        worst = worst.max(norm(sub(*u, [c.centroid_m[1], 0.0, 0.0])));
    }
    assert!(worst < 0.05, "Couette error {worst}");
}

// witness: FEA-FLUID-SOLVERS-001/coupled_newton_continuation @
//   incompressible_laminar_steady/tet4/numerical_kernel; coverage: partial;
//   boundary: thin cells on a symmetry plane (spanwise grading z⁴, first layer 1/256 of the
//   span) converge in at most five coupled steps with the assembled momentum Jacobian and
//   reproduce plane Poiseuille flow; continuation and line search are not covered.
#[test]
fn thin_cells_on_symmetry_planes_converge_promptly() {
    let f = symmetry_case_on(
        fixture::case_mapped(4, ReferenceCase::Duct, false, |[x, y, z]| [x, y, z.powi(4)]),
        |target| FluidConstraintKind::Symmetry { target },
    );
    let p = CanonicalFluidProblem::prepare(f.input()).unwrap();
    let r = p.solve().unwrap();
    assert!(r.outer_iterations <= 5, "outer iterations {}", r.outer_iterations);
    let (relative, _) = error(&f, &r);
    assert!(relative < 0.1, "relative L2 error {relative}");
}

// witness: FEA-FLUID-SOLVERS-001/block_linear_solvers @
//   incompressible_laminar_steady/tet4/numerical_kernel; coverage: partial;
//   boundary: the assembled two-ring momentum Jacobian reproduces the matrix-free momentum
//   rows to roundoff on a graded slab with symmetry planes, a pressure outlet, inflow and
//   outflow faces, for a nonuniform velocity and change.
#[test]
fn assembled_momentum_block_matches_the_matrix_free_rows() {
    let f = symmetry_case_on(
        fixture::case_mapped(3, ReferenceCase::Duct, false, |[x, y, z]| [x, y, z.powi(4)]),
        |target| FluidConstraintKind::Symmetry { target },
    );
    let p = CanonicalFluidProblem::prepare(f.input()).unwrap();
    let n = p.topology.cells.len();
    let field = |c: &crate::topology::FiniteVolumeCell, a: f64| {
        let [x, y, z] = c.centroid_m;
        [a * (1.0 + x * y), 0.3 * a * (z - y * y), 0.2 * a * (x - z)]
    };
    let origin = p.topology.cells.iter().map(|c| field(c, 0.7)).collect::<Vec<_>>();
    let change = p.topology.cells.iter().map(|c| field(c, -0.4)).collect::<Vec<_>>();
    let pressure = vec![0.0; n];
    let q = super::super::pressure::h_flux(&p, &origin, &vec![[0.0; 3]; n]).unwrap();
    let (assembled, matrix_free) =
        super::super::coupled::assembled_and_matrix_free_rows(&p, &origin, &pressure, &q, &change)
            .unwrap();
    let scale = matrix_free.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let worst = assembled
        .iter()
        .zip(&matrix_free)
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    assert!(worst <= 1e-12 * scale, "difference {worst} against scale {scale}");
}

/// Solver-speed benchmark on the graded symmetry slab (not a validation test):
/// `RUNMAT_CFD_BENCH_N=<n> RUNMAT_CFD_BENCH_THREADS=<t> cargo test -p runmat-analysis-fea --lib
/// graded_slab_solver_benchmark -- --ignored --nocapture`. The slab has about 6·n³ cells.
#[test]
#[ignore = "benchmark"]
fn graded_slab_solver_benchmark() {
    let n: usize = std::env::var("RUNMAT_CFD_BENCH_N").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let threads: usize =
        std::env::var("RUNMAT_CFD_BENCH_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let start = std::time::Instant::now();
    let f = symmetry_case_on(
        fixture::case_mapped(n, ReferenceCase::Duct, false, |[x, y, z]| [x, y, z.powi(4)]),
        |target| FluidConstraintKind::Symmetry { target },
    );
    let mut f = f;
    // Benchmark budgets: the study converges in a handful of coupled steps.
    f.policy.maximum_outer_iterations = 50;
    f.policy.maximum_resident_bytes = 24_000_000_000;
    f.policy.maximum_operator_work = u64::MAX / 4;
    let p = CanonicalFluidProblem::prepare(f.input()).unwrap();
    let prepare = start.elapsed().as_secs_f64();
    let cells = p.topology.cells.len();
    let start = std::time::Instant::now();
    let r = runmat_cpu_execution::with_cpu_pool(threads, || p.solve()).unwrap().unwrap();
    let solve = start.elapsed().as_secs_f64();
    let (relative, _) = error(&f, &r);
    println!(
        "{{\"bench\":\"graded_slab\",\"n\":{n},\"cells\":{cells},\"threads\":{threads},\"prepare_s\":{prepare:.2},\"solve_s\":{solve:.2},\"outer_iterations\":{},\"linear_iterations\":{},\"relative_l2_error\":{relative:.3e}}}",
        r.outer_iterations, r.linear_iterations
    );
}
