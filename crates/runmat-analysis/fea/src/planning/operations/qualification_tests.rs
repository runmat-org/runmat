//! Calibration of the operations estimate against the usage ledger of real solves.
use crate::adaptation::structural_recovery::qualification_tests::mesh::structured_cantilever;
use crate::contracts::LinearStaticSolveOptions;
use crate::fixtures::{fixture_model, FixtureId};
use crate::solve::preconditioner::SpdPreconditionerKind;
use crate::contracts::ComputeBackend;
use runmat_meshing_core::{ElementOrder, SolverMeshArtifact};
use runmat_simulation_usage::{CompletedUsage, OperationCount, UsageContext, UsageLedger};
use std::sync::Arc;

fn measured_static(
    mesh: &SolverMeshArtifact,
    preconditioner: SpdPreconditionerKind,
) -> (CompletedUsage, u32) {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let limit = OperationCount::from_operations(u64::MAX / 4);
    let ledger = Arc::new(UsageLedger::new(limit, limit).unwrap());
    let context = UsageContext::new(Arc::clone(&ledger));
    let result = context
        .run_sync(|| {
            crate::run_linear_static_with_options(
                &model,
                ComputeBackend::Cpu,
                LinearStaticSolveOptions {
                    solver_mesh: Some(mesh.clone()),
                    require_solver_mesh_for_solid: true,
                    preconditioner_kind: preconditioner,
                    ..LinearStaticSolveOptions::default()
                },
            )
        })
        .unwrap_or_else(|error| panic!("the cantilever solves: {error}"));
    let iterations = result
        .structural_static_convergence
        .as_ref()
        .map_or(0, |convergence| convergence.iterations);
    (ledger.snapshot().unwrap().completed, iterations)
}

/// The structured Tet10 cantilever with its top face (y = 1) bulged: midside nodes on that face
/// move outward, so the elements behind it are curved and integrate on tensor Gauss rules.
pub(super) fn curved_top_cantilever(divisions: usize) -> SolverMeshArtifact {
    let mut mesh = structured_cantilever(divisions, ElementOrder::Tet10);
    let corners = (divisions + 1).pow(3) as u64;
    for node in &mut mesh.topology.nodes {
        let mut position = node.coordinates_m.canonical_value();
        if node.node_id > corners && (position[1] - 1.0).abs() < 1.0e-12 {
            let [x, _, z] = position;
            position[1] += 0.2 / divisions as f64 * (x * (1.0 - x) + z * (1.0 - z));
            node.coordinates_m = runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(position);
        }
    }
    mesh.canonical_digest = runmat_meshing_core::StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();
    runmat_meshing_core::CanonicalMeshingContract::validate_canonical(&mesh).unwrap();
    mesh
}

#[test]
#[ignore]
fn report_measured_curved_operations() {
    for divisions in [2usize, 4, 6] {
        let mesh = curved_top_cantilever(divisions);
        let curved = mesh
            .topology
            .volume_elements
            .iter()
            .filter(|element| {
                let nodes: [[f64; 3]; 10] = std::array::from_fn(|index| {
                    mesh.topology.nodes[element.node_ids[index] as usize - 1]
                        .coordinates_m
                        .canonical_value()
                });
                !runmat_meshing_core::tetrahedron10_is_affine(&nodes)
            })
            .count();
        for (name, preconditioner) in [
            ("factor", SpdPreconditionerKind::Auto),
            ("multigrid", SpdPreconditionerKind::Amg),
        ] {
            let (usage, iterations) = measured_static(&mesh, preconditioner);
            eprintln!(
                "G87 curved div={divisions} path={name} curved={curved} elements={} iterations={iterations} assembly={} factorization={} solve={} other={}",
                mesh.topology.volume_elements.len(),
                usage.assembly,
                usage.factorization,
                usage.solve,
                usage.other,
            );
        }
    }
}

#[test]
#[ignore]
fn report_measured_static_operations() {
    for (order, divisions) in [
        (ElementOrder::Tet4, [4usize, 6, 8, 12]),
        (ElementOrder::Tet10, [2, 3, 4, 6]),
    ] {
        for divisions in divisions {
            let mesh = structured_cantilever(divisions, order);
            for (name, preconditioner) in [
                ("factor", SpdPreconditionerKind::Auto),
                ("multigrid", SpdPreconditionerKind::Amg),
            ] {
                if name == "factor" && order == ElementOrder::Tet4 && divisions >= 8 {
                    continue; // G100: the factor path is unavailable on these meshes.
                }
                let (usage, iterations) = measured_static(&mesh, preconditioner);
                eprintln!(
                    "G87 {order:?} div={divisions} path={name} nodes={} elements={} boundary={} iterations={iterations} meshing={} assembly={} factorization={} solve={} other={}",
                    mesh.topology.nodes.len(),
                    mesh.topology.volume_elements.len(),
                    mesh.topology.boundary_faces.len(),
                    usage.meshing,
                    usage.assembly,
                    usage.factorization,
                    usage.solve,
                    usage.other,
                );
            }
        }
    }
}

/// The admitted plan of the cantilever solve on `path`, as the solve derives it.
fn static_plan(
    mesh: &SolverMeshArtifact,
    path: crate::planning::resource::StaticSolvePath,
) -> crate::planning::resource::FeaResourcePlan {
    use crate::planning::resource::{
        build_static_resource_plan_selecting, LocalExecutionPlan, StaticResourcePreflight,
        StaticSolvePath, StaticSolveSelection,
    };
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let constraints =
        crate::constraints::CanonicalStructuralConstraintPlan::compile(&model, mesh).unwrap();
    let fields = crate::post::fields::structural_static_observable_field_definitions().unwrap();
    let observations = crate::results::observation::resolve_structural_observation_plan(
        &model.sequence.steps[0],
        &model,
        Some(mesh),
        &constraints,
        &fields,
        crate::results::observation::ObservationLimits::default(),
    )
    .unwrap();
    build_static_resource_plan_selecting(
        StaticResourcePreflight {
            compilation: None,
            prepared_inputs: None,
            model: &model,
            step: &model.sequence.steps[0],
            backend: ComputeBackend::Cpu,
            preconditioner: match path {
                StaticSolvePath::ProvingFactorization => SpdPreconditionerKind::Auto,
                StaticSolvePath::ElasticRigidityMultigrid => SpdPreconditionerKind::Amg,
            },
            observation_execution:
                crate::results::observation::ObservationExecutionPolicy::serial(),
            mesh: Some(mesh),
            constraints: &constraints,
            observations: &observations,
            accelerator_requests: &[],
            local_execution: LocalExecutionPlan::serial(),
        },
        StaticSolveSelection::Recorded(path),
    )
    .expect("the cantilever plan is admitted")
}

/// Every category's estimate covers the work the ledger measured. Assembly, the direct solve and
/// recovery stay within twice it (plus a small absolute allowance for categories of a few
/// thousand operations). The multigrid setup bounds its coarse factor by the coarsest level's
/// dense size, and its solve covers [`super::MULTIGRID_ITERATIONS_COVERED`] iterations, so those
/// two are checked per iteration by the caller instead.
fn assert_calibrated(
    label: &str,
    measured: CompletedUsage,
    estimate: &super::FeaOperationEstimate,
    bounded: &[&str],
) {
    for (category, measured, estimated) in [
        ("meshing", measured.meshing, estimate.meshing),
        ("assembly", measured.assembly, estimate.assembly),
        ("factorization", measured.factorization, estimate.factorization),
        ("solve", measured.solve, estimate.solve),
        ("other", measured.other, estimate.other),
    ] {
        let (measured, estimated) = (measured.operations(), estimated.operations());
        eprintln!(
            "G87 calibration {label} {category}: measured {measured} estimate {estimated} ratio {:.2}",
            estimated as f64 / measured.max(1) as f64
        );
        assert!(
            measured <= estimated,
            "{label} {category}: measured {measured} exceeds the estimate {estimated}"
        );
        assert!(
            !bounded.contains(&category) || estimated <= 2 * measured + 200_000,
            "{label} {category}: estimate {estimated} exceeds twice the measured {measured}"
        );
    }
}

fn calibrate(label: &str, mesh: &SolverMeshArtifact, factor: bool) {
    use crate::planning::resource::StaticSolvePath;
    let paths = [
        (
            StaticSolvePath::ProvingFactorization,
            SpdPreconditionerKind::Auto,
        ),
        (
            StaticSolvePath::ElasticRigidityMultigrid,
            SpdPreconditionerKind::Amg,
        ),
    ];
    for (path, preconditioner) in paths.into_iter().skip(usize::from(!factor)) {
        let plan = static_plan(mesh, path);
        let estimate = super::estimate_static_solid_operations_from_plan(&plan, mesh)
            .unwrap()
            .expect("a static solid plan has an estimate");
        let (measured, iterations) = measured_static(mesh, preconditioner);
        let label = format!("{label} {path:?}");
        if path == StaticSolvePath::ElasticRigidityMultigrid {
            assert!(iterations <= super::MULTIGRID_ITERATIONS_COVERED);
            // Per iteration, the covered solve stays within twice the measured one.
            let covered = u64::from(super::MULTIGRID_ITERATIONS_COVERED);
            let per_iteration = measured.solve.operations() / u64::from(iterations.max(1));
            let estimated = estimate.solve.operations() / covered;
            eprintln!("G87 calibration {label} solve per iteration: measured {per_iteration} estimate {estimated}");
            assert!(estimated <= 2 * per_iteration + 200_000, "{label}: {estimated} per iteration");
            assert_calibrated(&label, measured, &estimate, &["assembly", "other"]);
        } else {
            assert_calibrated(
                &label,
                measured,
                &estimate,
                &["assembly", "factorization", "solve", "other"],
            );
        }
    }
}

#[test]
fn static_tet4_estimate_covers_measured_work_within_twice() {
    // The proving factorization is unavailable on structured Tet4 cantilevers from eight
    // divisions (G100: exact fill exceeds the envelope); those calibrate the multigrid path.
    for (divisions, factor) in [(4, true), (6, true), (12, false)] {
        calibrate(
            &format!("tet4 div={divisions}"),
            &structured_cantilever(divisions, ElementOrder::Tet4),
            factor,
        );
    }
}

#[test]
fn static_tet10_estimate_covers_measured_work_within_twice() {
    for divisions in [3, 5] {
        calibrate(
            &format!("tet10 div={divisions}"),
            &structured_cantilever(divisions, ElementOrder::Tet10),
            true,
        );
    }
}

#[test]
fn curved_tet10_estimate_covers_measured_work_within_twice() {
    for divisions in [4, 6] {
        calibrate(
            &format!("curved tet10 div={divisions}"),
            &curved_top_cantilever(divisions),
            true,
        );
    }
}
