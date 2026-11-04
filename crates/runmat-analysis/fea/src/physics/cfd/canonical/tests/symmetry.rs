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
    let mut target = RegionTarget::boundary(
        runmat_meshing_core::fixtures::fixture_boundary_selection(&(id)),
    );
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
    for (c, u) in f
        .topology
        .cells
        .iter()
        .zip(r.velocity_m_per_s.canonical_values())
    {
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
                assert_eq!(r.face_volume_flux_m3_per_s.canonical_values()[i], 0.0);
                let shear = r.residual.face_forces.viscous_traction_n.canonical_values()[i];
                let normal = geometry_ops::scale(
                    face.owner_area_vector_m2,
                    1.0 / face.area_m2.canonical_value(),
                );
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
        assert_eq!(
            balance.volume_flow_rate_m3_per_s,
            runmat_types::physical::Quantity::new(0.0)
        );
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
    let mut q = r.face_volume_flux_m3_per_s.canonical_values().clone();
    q[face] = 1e-9;
    r.face_volume_flux_m3_per_s = runmat_types::physical::Quantity::new(q);
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
        let interior_side =
            (x[1] - 1.0).abs() >= 1e-12 && x[2].abs() >= 1e-12 && (x[2] - 1.0).abs() >= 1e-12;
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
    for (c, u) in f
        .topology
        .cells
        .iter()
        .zip(r.velocity_m_per_s.canonical_values())
    {
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
    assert!(
        r.outer_iterations <= 5,
        "outer iterations {}",
        r.outer_iterations
    );
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
    let origin = p
        .topology
        .cells
        .iter()
        .map(|c| field(c, 0.7))
        .collect::<Vec<_>>();
    let change = p
        .topology
        .cells
        .iter()
        .map(|c| field(c, -0.4))
        .collect::<Vec<_>>();
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
    assert!(
        worst <= 1e-12 * scale,
        "difference {worst} against scale {scale}"
    );
}

/// Solver-speed benchmark on the graded symmetry slab (not a validation test):
/// `RUNMAT_CFD_BENCH_N=<n> RUNMAT_CFD_BENCH_THREADS=<t> cargo test -p runmat-analysis-fea --lib
/// graded_slab_solver_benchmark -- --ignored --nocapture`. The slab has about 6·n³ cells.
#[test]
#[ignore = "benchmark"]
fn graded_slab_solver_benchmark() {
    let n: usize = std::env::var("RUNMAT_CFD_BENCH_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let threads: usize = std::env::var("RUNMAT_CFD_BENCH_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let start = std::time::Instant::now();
    let f = symmetry_case_on(
        fixture::case_mapped(n, ReferenceCase::Duct, false, |[x, y, z]| [x, y, z.powi(4)]),
        |target| FluidConstraintKind::Symmetry { target },
    );
    let mut f = f;
    // Benchmark budgets: the study converges in a handful of coupled steps.
    f.policy.maximum_outer_iterations = 50;
    f.policy.maximum_resident_bytes = Some(24_000_000_000);
    f.policy.maximum_operator_work = None;
    let p = CanonicalFluidProblem::prepare(f.input()).unwrap();
    let prepare = start.elapsed().as_secs_f64();
    let cells = p.topology.cells.len();
    let start = std::time::Instant::now();
    let r = runmat_cpu_execution::with_cpu_pool(threads, || p.solve())
        .unwrap()
        .unwrap();
    let solve = start.elapsed().as_secs_f64();
    let (relative, _) = error(&f, &r);
    println!(
        "{{\"bench\":\"graded_slab\",\"n\":{n},\"cells\":{cells},\"threads\":{threads},\"prepare_s\":{prepare:.2},\"solve_s\":{solve:.2},\"outer_iterations\":{},\"linear_iterations\":{},\"relative_l2_error\":{relative:.3e}}}",
        r.outer_iterations, r.linear_iterations
    );
}

/// Transient solver-speed benchmark (not a validation test): the symmetry-plane channel started
/// from rest by backward-Euler intervals with the default fluid policy. The interval inertia
/// ratio ρh²/(µΔt) defaults to 1.6, that of the coarse channel-startup study.
/// `RUNMAT_CFD_BENCH_N=<n> RUNMAT_CFD_BENCH_THREADS=<t> RUNMAT_CFD_BENCH_STEPS=<s>
/// [RUNMAT_CFD_BENCH_DT=<s>] cargo test --release -p runmat-analysis-fea --lib
/// startup_slab_solver_benchmark -- --ignored --nocapture`. The slab has about 6·n³ cells.
#[test]
#[ignore = "benchmark"]
fn startup_slab_solver_benchmark() {
    let var = |name: &str| std::env::var(name).ok();
    let n: usize = var("RUNMAT_CFD_BENCH_N")
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let threads: usize = var("RUNMAT_CFD_BENCH_THREADS")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let steps: usize = var("RUNMAT_CFD_BENCH_STEPS")
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let h = 1.0 / n as f64;
    let dt: f64 = var("RUNMAT_CFD_BENCH_DT")
        .and_then(|v| v.parse().ok())
        .unwrap_or(h * h / 1.6);
    let mut f = symmetry_case_on(fixture::case(n, ReferenceCase::Duct, false), |target| {
        FluidConstraintKind::Symmetry { target }
    });
    f.policy = CanonicalFluidPolicy {
        maximum_resident_bytes: Some(24_000_000_000),
        ..CanonicalFluidPolicy::default()
    };
    let p = CanonicalFluidProblem::prepare(f.input()).unwrap();
    let cells = p.topology.cells.len();
    let mut u = vec![[0.0; 3]; cells];
    let mut pressure = vec![0.0; cells];
    let mut q = vec![0.0; p.topology.faces.len()];
    let started = std::time::Instant::now();
    for step in 0..steps {
        let previous = super::super::transient::CanonicalFluidFields {
            pressure_reference_pa: runmat_types::physical::Quantity::new(
                p.pressure_reference_pa.canonical_values(),
            ),
            velocity_m_per_s: runmat_types::physical::Quantity::new(&u[..]),
            pressure_fluctuation_pa: runmat_types::physical::Quantity::new(&pressure[..]),
            face_volume_flux_m3_per_s: runmat_types::physical::Quantity::new(&q[..]),
        };
        let interval = p
            .prepare_backward_euler(step as f64 * dt, (step + 1) as f64 * dt, previous)
            .unwrap();
        let t = std::time::Instant::now();
        let r = runmat_cpu_execution::with_cpu_pool(threads, || interval.solve())
            .unwrap()
            .unwrap();
        println!(
            "{{\"bench\":\"startup_slab\",\"n\":{n},\"cells\":{cells},\"threads\":{threads},\"step\":{step},\"dt\":{dt:.3e},\"solve_s\":{:.3},\"outer_iterations\":{},\"linear_iterations\":{},\"momentum_relative\":{:.3e},\"continuity_relative\":{:.3e}}}",
            t.elapsed().as_secs_f64(), r.outer_iterations, r.linear_iterations,
            r.residual.momentum_relative, r.residual.continuity_relative
        );
        u = r.velocity_m_per_s.canonical_values().clone();
        pressure = r.pressure_fluctuation_pa.canonical_values().clone();
        q = r.face_volume_flux_m3_per_s.canonical_values().clone();
    }
    let mean_u = u.iter().map(|v| v[0]).sum::<f64>() / cells as f64;
    println!(
        "{{\"bench\":\"startup_slab_total\",\"n\":{n},\"threads\":{threads},\"steps\":{steps},\"solve_s\":{:.3},\"mean_ux\":{mean_u:.9e}}}",
        started.elapsed().as_secs_f64()
    );
}

/// Schur scaling of the SIMPLE sweep (q-fluid-16 item 6): the Rhie–Chow coefficients carry the
/// momentum relaxation, rau = α V / a_P, while the coupled momentum block is unrelaxed, which
/// raised the question whether the sweep's pressure operator is half the true one. It is not:
/// the discrete flux itself carries α, so on cell-scale modes L(rf) is exactly the operator of
/// the sweep's own correction, and on the lowest mode the viscous term already exceeds the
/// exact Schur complement (measured 0.32 and 0.84 for exact/sweep at n = 4). Rayleigh
/// quotients of the exact Schur complement against the sweep's pressure operator, on the
/// converged symmetry-plane channel, for the lowest pressure mode cos(πx/2) (zero at the outlet, flat at the inlet) and a
/// cell-alternating one.
#[test]
#[ignore = "fluid-18 WIP calibration probe: on the rough mode the exact Schur complement exceeds the sweep operator (exact/sweep 1.15) under the transplanted viscous Schur weight 0.5; owned by the fluid lane"]
fn sweep_pressure_operator_against_the_exact_schur_complement() {
    let f = symmetry_case(4, |target| FluidConstraintKind::Symmetry { target });
    let p = CanonicalFluidProblem::prepare(f.input()).unwrap();
    let r = p.solve().unwrap();
    let u = r.velocity_m_per_s.canonical_values().clone();
    let pressure = r.pressure_fluctuation_pa.canonical_values().clone();
    let q = r.face_volume_flux_m3_per_s.canonical_values().clone();
    let smooth = p
        .topology
        .cells
        .iter()
        .map(|c| (0.5 * std::f64::consts::PI * c.centroid_m[0]).cos())
        .collect::<Vec<_>>();
    let mut state = 0x9e3779b97f4a7c15u64;
    let rough = (0..u.len())
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            if state & 1 == 0 {
                1.0
            } else {
                -1.0
            }
        })
        .collect::<Vec<_>>();
    for (name, z) in [("smooth", smooth), ("rough", rough)] {
        let (exact, simple, sweep) =
            super::super::coupled::schur_rayleigh_quotients(&p, &u, &pressure, &q, &z).unwrap();
        println!(
            "schur {name}: exact {exact:.6e} simple {simple:.6e} sweep {sweep:.6e} exact/sweep {:.4} simple/sweep {:.4} relaxation {}",
            exact / sweep,
            simple / sweep,
            p.policy.momentum_relaxation
        );
        assert!(exact > 0.0 && simple > 0.0 && sweep > 0.0);
        // Refutes a factor 2 from the relaxation in rau: the sweep's operator is not below the
        // exact Schur complement on either mode (doubling it would halve both quotients).
        assert!(exact / sweep < 1.0, "{name}: exact/sweep {}", exact / sweep);
        if name == "rough" {
            // On cell-scale modes the Rhie–Chow Laplacian L(rf) is SIMPLE's own operator.
            assert!(
                (0.6..1.05).contains(&(simple / sweep)),
                "simple/sweep {}",
                simple / sweep
            );
            assert!(exact / sweep > 0.6, "exact/sweep {}", exact / sweep);
        }
    }
}

/// The plane-channel study geometry on a persisted study mesh (`RUNMAT_CFD_BENCH_MESH`, a
/// `.solver-mesh.cbor`): inlet at min x with a uniform velocity, outlet at max x, no-slip walls at
/// min/max y, symmetry planes elsewhere; ρ = 1000, µ = 1 as the channel studies.
fn persisted_channel_case(path: &str, inlet_speed: f64) -> fixture::Fixture {
    use runmat_meshing_core::{CanonicalMeshingContract, PersistentEntityId, PersistentEntityKind};
    let bytes = std::fs::read(path).unwrap();
    let mut mesh = runmat_meshing_core::SolverMeshArtifact::canonical_decode(&bytes).unwrap();
    for face in &mut mesh.topology.boundary_faces {
        face.provenance = vec![PersistentEntityId::new(
            PersistentEntityKind::Face,
            format!("original-reference-face-{}", face.face_id),
            vec!["root".into()],
        )];
    }
    mesh.seal_canonical_digest().unwrap();
    let source = crate::topology::AnalysisProjection::derive(&mesh).unwrap();
    // `RUNMAT_CFD_BENCH_REGION=<k>` keeps the k-th volume region only (a conjugate model's
    // coolant); every face of it other than the inlet and outlet planes is then a no-slip wall.
    let region = std::env::var("RUNMAT_CFD_BENCH_REGION").ok().map(|k| {
        let mut regions = source
            .elements
            .iter()
            .map(|e| e.region_id.clone())
            .collect::<Vec<_>>();
        regions.sort();
        regions.dedup();
        regions[k.parse::<usize>().unwrap()].clone()
    });
    let topology = match &region {
        Some(region) => crate::topology::FiniteVolumeTopology::derive_selected(
            &source,
            fixture::geometry_policy(),
            crate::topology::FiniteVolumeCellSelection::regions(vec![region.clone()]).unwrap(),
        ),
        None => crate::topology::FiniteVolumeTopology::derive(&source, fixture::geometry_policy()),
    }
    .unwrap();
    let mut bx = [f64::MAX, f64::MIN];
    let mut by = [f64::MAX, f64::MIN];
    let exterior = topology
        .faces
        .iter()
        .filter(|f| f.neighbor.is_none())
        .filter_map(|f| f.boundary_source.as_ref())
        .map(|b| b.projection_boundary_index)
        .collect::<Vec<_>>();
    for &i in &exterior {
        let x = source.boundary_faces[i].centroid_m;
        bx = [bx[0].min(x[0]), bx[1].max(x[0])];
        by = [by[0].min(x[1]), by[1].max(x[1])];
    }
    let near = |a: f64, b: f64| (a - b).abs() < 1e-9;
    let (mut inlet, mut outlet, mut walls, mut planes) = (vec![], vec![], vec![], vec![]);
    for &i in &exterior {
        let face = &source.boundary_faces[i];
        let x = face.centroid_m;
        let entities = face.provenance.clone();
        if near(x[0], bx[1]) {
            outlet.extend(entities);
        } else if near(x[0], bx[0]) {
            inlet.extend(entities);
        } else if region.is_some() || near(x[1], by[0]) || near(x[1], by[1]) {
            walls.extend(entities);
        } else {
            planes.extend(entities);
        }
    }
    let mut material = fixture::material();
    material.density = runmat_analysis_core::MassDensity::new(1000.0);
    fixture::Fixture {
        mesh,
        source,
        topology,
        constraints: vec![
            FluidConstraint {
                constraint_id: "inlet".into(),
                kind: FluidConstraintKind::PrescribedVelocity {
                    target: target("inlet", inlet),
                    value: PrescribedValue::Constant([inlet_speed, 0.0, 0.0].map(Velocity::new)),
                },
            },
            FluidConstraint {
                constraint_id: "walls".into(),
                kind: FluidConstraintKind::NoSlipWall {
                    target: target("walls", walls),
                },
            },
        ]
        .into_iter()
        .chain((!planes.is_empty()).then(|| FluidConstraint {
            constraint_id: "spanwise".into(),
            kind: FluidConstraintKind::Symmetry {
                target: target("spanwise", planes),
            },
        }))
        .collect(),
        operators: vec![AnalysisBoundary {
            boundary_id: "outlet".into(),
            target: target("outlet", outlet),
            operator: BoundaryOperator::FluidOutletPressure {
                value: PrescribedValue::Constant(Pressure::new(0.0)),
            },
        }],
        signals: Vec::new(),
        actions: Vec::new(),
        material,
        policy: CanonicalFluidPolicy {
            maximum_resident_bytes: Some(24_000_000_000),
            ..CanonicalFluidPolicy::default()
        },
        reference: ReferenceCase::Duct,
        rotated: false,
    }
}

/// Solver-speed benchmark on a persisted study mesh (not a validation test): the channel started
/// impulsively from rest by backward-Euler intervals of `RUNMAT_CFD_BENCH_DT` (default 0.25 s,
/// the startup study's) with the default fluid policy. `RUNMAT_CFD_BENCH_MESH=<cbor>
/// RUNMAT_CFD_BENCH_THREADS=<t> RUNMAT_CFD_BENCH_STEPS=<s> [RUNMAT_CFD_BENCH_STEADY=1]`.
#[test]
#[ignore = "benchmark"]
fn persisted_channel_solver_benchmark() {
    let var = |name: &str| std::env::var(name).ok();
    let path = var("RUNMAT_CFD_BENCH_MESH").expect("RUNMAT_CFD_BENCH_MESH");
    let threads: usize = var("RUNMAT_CFD_BENCH_THREADS")
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);
    let steps: usize = var("RUNMAT_CFD_BENCH_STEPS")
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    let dt: f64 = var("RUNMAT_CFD_BENCH_DT")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.25);
    let speed: f64 = var("RUNMAT_CFD_BENCH_U")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.005);
    let mut f = persisted_channel_case(&path, speed);
    if let Some(restart) = var("RUNMAT_CFD_BENCH_RESTART").and_then(|v| v.parse().ok()) {
        f.policy.gmres_restart = restart;
    }
    // A memory probe needs only the first coupled steps: their workspaces are the peak.
    let probe = var("RUNMAT_CFD_BENCH_OUTER").and_then(|v| v.parse().ok());
    if let Some(outer) = probe {
        f.policy.maximum_outer_iterations = outer;
    }
    let p = CanonicalFluidProblem::prepare(f.input()).unwrap();
    let cells = p.topology.cells.len();
    let reservation = p.reservation();
    println!(
        "{{\"bench\":\"persisted_reservation\",\"cells\":{cells},\"faces\":{},\"resident_bytes_upper_bound\":{},\"verification_resident_bytes_upper_bound\":{},\"momentum_jacobian_entries\":{}}}",
        p.topology.faces.len(),
        reservation.resident_bytes_upper_bound,
        reservation.verification_resident_bytes_upper_bound,
        p.momentum_jacobian.entries()
    );
    let whole =
        crate::topology::FiniteVolumeTopology::reserve_mesh(&f.mesh, fixture::geometry_policy())
            .unwrap();
    println!(
        "{{\"bench\":\"persisted_preparation\",\"mesh_cells\":{},\"whole_mesh_geometry_scratch\":{},\"projection_estimate\":{}}}",
        whole.cell_count,
        whole.scratch_bytes,
        crate::topology::AnalysisProjection::derivation_resident_bytes_estimate(&f.mesh).unwrap()
    );
    if probe.is_some() {
        let t = std::time::Instant::now();
        let r = runmat_cpu_execution::with_cpu_pool(threads, || p.solve()).unwrap();
        println!(
            "{{\"bench\":\"persisted_probe\",\"cells\":{cells},\"solve_s\":{:.3},\"result\":\"{}\"}}",
            t.elapsed().as_secs_f64(),
            r.map_or_else(|e| e.to_string().chars().take(80).collect::<String>(), |_| "converged".into())
        );
        return;
    }
    if var("RUNMAT_CFD_BENCH_STEADY").is_some() {
        let t = std::time::Instant::now();
        let r = runmat_cpu_execution::with_cpu_pool(threads, || p.solve())
            .unwrap()
            .unwrap();
        let gradient = channel_pressure_gradient(&p, &r);
        println!(
            "{{\"bench\":\"persisted_steady\",\"cells\":{cells},\"threads\":{threads},\"solve_s\":{:.3},\"outer_iterations\":{},\"linear_iterations\":{},\"minus_dpdx\":{gradient:.6}}}",
            t.elapsed().as_secs_f64(), r.outer_iterations, r.linear_iterations
        );
        return;
    }
    let mut u = vec![[0.0; 3]; cells];
    let mut pressure = vec![0.0; cells];
    let mut q = vec![0.0; p.topology.faces.len()];
    let started = std::time::Instant::now();
    for step in 0..steps {
        let previous = super::super::transient::CanonicalFluidFields {
            pressure_reference_pa: runmat_types::physical::Quantity::new(
                p.pressure_reference_pa.canonical_values(),
            ),
            velocity_m_per_s: runmat_types::physical::Quantity::new(&u[..]),
            pressure_fluctuation_pa: runmat_types::physical::Quantity::new(&pressure[..]),
            face_volume_flux_m3_per_s: runmat_types::physical::Quantity::new(&q[..]),
        };
        let interval = p
            .prepare_backward_euler(step as f64 * dt, (step + 1) as f64 * dt, previous)
            .unwrap();
        let t = std::time::Instant::now();
        let r = runmat_cpu_execution::with_cpu_pool(threads, || interval.solve())
            .unwrap()
            .unwrap();
        println!(
            "{{\"bench\":\"persisted_startup\",\"cells\":{cells},\"threads\":{threads},\"step\":{step},\"solve_s\":{:.3},\"outer_iterations\":{},\"linear_iterations\":{},\"minus_dpdx\":{:.6}}}",
            t.elapsed().as_secs_f64(), r.outer_iterations, r.linear_iterations,
            channel_pressure_gradient(&p, &r)
        );
        u = r.velocity_m_per_s.canonical_values().clone();
        pressure = r.pressure_fluctuation_pa.canonical_values().clone();
        q = r.face_volume_flux_m3_per_s.canonical_values().clone();
    }
    println!(
        "{{\"bench\":\"persisted_startup_total\",\"cells\":{cells},\"threads\":{threads},\"steps\":{steps},\"solve_s\":{:.3}}}",
        started.elapsed().as_secs_f64()
    );
}

/// −dp/dx from a least-squares line through cell pressures in the middle half of the channel.
fn channel_pressure_gradient(p: &CanonicalFluidProblem<'_>, r: &CanonicalFluidResult) -> f64 {
    let xs = p
        .topology
        .cells
        .iter()
        .map(|c| c.centroid_m[0])
        .collect::<Vec<_>>();
    let (lo, hi) = xs
        .iter()
        .fold((f64::MAX, f64::MIN), |(a, b), x| (a.min(*x), b.max(*x)));
    let (l, h) = (lo + 0.25 * (hi - lo), lo + 0.75 * (hi - lo));
    let pairs = xs
        .iter()
        .zip(r.pressure_pa.canonical_values())
        .filter(|(x, _)| (l..h).contains(*x))
        .map(|(x, p)| (*x, *p))
        .collect::<Vec<_>>();
    let n = pairs.len() as f64;
    let mx = pairs.iter().map(|v| v.0).sum::<f64>() / n;
    let mp = pairs.iter().map(|v| v.1).sum::<f64>() / n;
    let sxy = pairs.iter().map(|v| (v.0 - mx) * (v.1 - mp)).sum::<f64>();
    let sxx = pairs.iter().map(|v| (v.0 - mx).powi(2)).sum::<f64>();
    -sxy / sxx
}
