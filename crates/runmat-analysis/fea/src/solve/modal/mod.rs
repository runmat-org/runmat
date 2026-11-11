use crate::{
    assembly::AssemblySummary,
    contracts::{SolverMethod, SolverPreconditioner},
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
    planning::resource::LocalExecutionPlan,
    planning::resource::SubspaceSolvePath,
    solve::backend::kind::LinearAlgebraBackendKind,
    ComputeBackend,
};

mod capacity;
mod diagnostics;
mod math;
mod truncation;

use diagnostics::push_modal_quality_diagnostics;
mod preparation;
pub use preparation::solve_modal_system;
pub(crate) use preparation::ModalSolveInput;
mod result;
pub use result::ModalSolveResult;
use truncation::resolve_truncation_boundary;

pub(crate) const MODAL_BOUNDARY_PROBE_WIDTH: usize = 4;
/// Block width of the modal shift-invert Lanczos.
pub(crate) const MODAL_LANCZOS_BLOCK_SIZE: usize = runmat_sparse_solvers::eigen::DEFAULT_BLOCK_SIZE;
/// Modes past the last requested one that the admitted subspace keeps room for when they share
/// its eigenvalue. A Lanczos block of `p` vectors resolves an eigenvalue of multiplicity up to
/// `p`, so a resolved cluster at the truncation boundary adds at most `p − 1` modes.
pub(crate) const MODAL_BOUNDARY_CLUSTER_ALLOWANCE: usize = MODAL_LANCZOS_BLOCK_SIZE - 1;

mod options;
use options::keeps_whole_translational_nodes;
pub(crate) use options::{modal_eigen_options, modal_lobpcg_options};

pub(crate) const MODAL_MASS_ORTHOGONALIZATION_PASSES: usize = 2;
/// (frequency in Hz, expanded shape, relative residual, backward error).
type ModalCandidate = (f64, Vec<f64>, f64, f64);

pub(crate) fn solve_modal_system_reusing(
    input: ModalSolveInput<'_>,
    reuse: &mut crate::algebra::LinearPreparationReuse,
) -> ModalSolveResult {
    let ModalSolveInput {
        summary,
        requested_mode_count,
        eigenspace_relative_tolerance,
        convergence,
        backend,
        path,
        node_coordinates_m,
    } = input;
    let (solver_method, path_preconditioner) = match path {
        SubspaceSolvePath::ShiftInvertFactorization => (
            SolverMethod::ShiftInvertBlockLanczos,
            SolverPreconditioner::SparseCholesky,
        ),
        SubspaceSolvePath::PreconditionedLobpcg => (
            SolverMethod::PreconditionedLobpcg,
            SolverPreconditioner::Amg,
        ),
    };
    if summary.dof_count == 0 || requested_mode_count == 0 {
        return ModalSolveResult {
            converged: false,
            frequencies_hz: Vec::new(),
            mode_shapes: Vec::new(),
            residual_norms: Vec::new(),
            backward_errors: Vec::new(),
            diagnostics: vec![FeaDiagnostic {
                code: "FEA_MODAL_EMPTY_SYSTEM".to_string(),
                severity: FeaDiagnosticSeverity::Warning,
                message:
                    "modal solve skipped because assembled system has zero DOFs or requested mode_count is zero"
                        .to_string(),
            }],
            solver_method: SolverMethod::ShiftInvertBlockLanczos,
            solver_backend: LinearAlgebraBackendKind::CpuReference,
            preconditioner: SolverPreconditioner::None,
            solver_host_sync_count: 0,
            device_apply_k_count: 0,
            device_apply_k_attempt_count: 0,
            accelerator_evidence: Vec::new(),
            available_mode_count: summary.dof_count,
            boundary_probe_count: 0,
            boundary_probe_frequency_hz: None,
            boundary_unresolved_at_capacity: None,
        };
    }

    let prepared_modal_result = match summary.constraint_matrix.as_ref() {
        Some(constraints)
            if constraints
                .right_hand_side
                .iter()
                .all(|value| *value == 0.0) =>
        {
            constraints.affine_reduction().and_then(|reduction| {
                summary
                    .operator
                    .prepare_modal_with_reduction(reduction)
                    .map_err(|error| {
                        crate::constraints::ConstraintAlgebraError::InvalidMatrix(error.to_string())
                    })
            })
        }
        Some(_) => Err(crate::constraints::ConstraintAlgebraError::InvalidMatrix(
            "modal constraints must be homogeneous".into(),
        )),
        None => summary.operator.prepare_modal().map_err(|error| {
            crate::constraints::ConstraintAlgebraError::InvalidMatrix(error.to_string())
        }),
    };
    let prepared_modal = match prepared_modal_result {
        Ok(prepared) => prepared,
        Err(error) => {
            return ModalSolveResult {
                converged: false,
                frequencies_hz: Vec::new(),
                mode_shapes: Vec::new(),
                residual_norms: Vec::new(),
                backward_errors: Vec::new(),
                diagnostics: vec![FeaDiagnostic {
                    code: "FEA_MODAL_PREPARATION_FAILED".to_string(),
                    severity: FeaDiagnosticSeverity::Error,
                    message: format!("modal free-DOF reduction failed: {error:?}"),
                }],
                solver_method: SolverMethod::ShiftInvertBlockLanczos,
                solver_backend: LinearAlgebraBackendKind::CpuReference,
                preconditioner: SolverPreconditioner::None,
                solver_host_sync_count: 0,
                device_apply_k_count: 0,
                device_apply_k_attempt_count: 0,
                accelerator_evidence: Vec::new(),
                available_mode_count: summary.dof_count,
                boundary_probe_count: 0,
                boundary_probe_frequency_hz: None,
                boundary_unresolved_at_capacity: None,
            };
        }
    };
    let reduced_operator = match prepared_modal.as_assembled() {
        Ok(operator) => operator,
        Err(error) => {
            return ModalSolveResult {
                converged: false,
                frequencies_hz: Vec::new(),
                mode_shapes: Vec::new(),
                residual_norms: Vec::new(),
                backward_errors: Vec::new(),
                diagnostics: vec![FeaDiagnostic {
                    code: "FEA_MODAL_PREPARATION_FAILED".to_string(),
                    severity: FeaDiagnosticSeverity::Error,
                    message: format!("modal reduced system validation failed: {error:?}"),
                }],
                solver_method: SolverMethod::ShiftInvertBlockLanczos,
                solver_backend: LinearAlgebraBackendKind::CpuReference,
                preconditioner: SolverPreconditioner::None,
                solver_host_sync_count: 0,
                device_apply_k_count: 0,
                device_apply_k_attempt_count: 0,
                accelerator_evidence: Vec::new(),
                available_mode_count: summary.dof_count,
                boundary_probe_count: 0,
                boundary_probe_frequency_hz: None,
                boundary_unresolved_at_capacity: None,
            };
        }
    };
    // The factor-free path proves coercivity by its own inertia count, so it never factors K.
    let prepared = match path {
        SubspaceSolvePath::PreconditionedLobpcg => None,
        SubspaceSolvePath::ShiftInvertFactorization => {
            match reduced_operator.prepare_linear_reusing(reuse) {
                Ok(prepared) => Some(prepared),
                Err(error) => {
                    return ModalSolveResult {
                        converged: false,
                        frequencies_hz: Vec::new(),
                        mode_shapes: Vec::new(),
                        residual_norms: Vec::new(),
                        backward_errors: Vec::new(),
                        diagnostics: vec![FeaDiagnostic {
                            code: "FEA_MODAL_SOLVER_ADMISSION_FAILED".into(),
                            severity: FeaDiagnosticSeverity::Error,
                            message: format!(
                                "modal stiffness admission failed on the constraint space: {error}"
                            ),
                        }],
                        solver_method,
                        solver_backend: LinearAlgebraBackendKind::CpuReference,
                        preconditioner: SolverPreconditioner::None,
                        solver_host_sync_count: 0,
                        device_apply_k_count: 0,
                        device_apply_k_attempt_count: 0,
                        accelerator_evidence: Vec::new(),
                        available_mode_count: reduced_operator.dof_count(),
                        boundary_probe_count: 0,
                        boundary_probe_frequency_hz: None,
                        boundary_unresolved_at_capacity: None,
                    }
                }
            }
        }
    };
    let mut linear_diagnostics = Vec::new();
    // Strict placement fails closed at the modal entry; here the shared admission only supplies
    // the fallback warning for a GPU request.
    if let Ok(admission) = crate::contracts::accelerator_admission::admit_accelerator_with_policy(
        crate::contracts::accelerator_admission::AcceleratorFamily::StructuralModal,
        backend,
        crate::contracts::accelerator_admission::AcceleratorAdmissionPolicy::FallbackToCpu,
    ) {
        linear_diagnostics.extend(admission.fallback);
    }
    let unconstrained_count = reduced_operator.dof_count();
    let target_mode_count = requested_mode_count.min(unconstrained_count);
    let maximum_subspace_dimension =
        match convergence.admitted_subspace_dimension(unconstrained_count, target_mode_count) {
            Ok(capacity) => capacity,
            Err(error) => return capacity::failure(error, unconstrained_count),
        };
    // The Cholesky factor that proved K coercive is the σ = 0 shift-invert operator.
    let mut factor = prepared
        .as_ref()
        .and_then(|prepared| prepared.admissibility())
        .and_then(|evidence| evidence.factor())
        .map(|factor| factor.0.clone());
    // The LOBPCG preconditioner, built once for every search of this solve.
    let mut multigrid = None;
    if path == SubspaceSolvePath::PreconditionedLobpcg {
        // A translational solid space is node-major with three DOFs per node. When the
        // constraint reduction keeps whole nodes, multigrid aggregates the nodal 3×3 blocks, as
        // the plan's multigrid estimate (operator complexity of a rigid-body hierarchy) assumes.
        let free_dofs = prepared_modal.free_dofs();
        let block_size = if keeps_whole_translational_nodes(summary, free_dofs) {
            3
        } else {
            1
        };
        // With the node positions, the nodal blocks aggregate the six rigid-body modes of the
        // free nodes, which bending-dominated solids need for multigrid to converge quickly.
        let near_nullspace = node_coordinates_m
            .filter(|coordinates| {
                block_size == 3 && coordinates.len() == summary.structural_node_count
            })
            .map(|coordinates| {
                let free_coordinates = free_dofs
                    .chunks_exact(3)
                    .flat_map(|triple| coordinates[triple[0] / 3])
                    .collect::<Vec<f64>>();
                runmat_sparse_solvers::precond::rigid_body_modes(&free_coordinates, 3)
            });
        match crate::solve::shared_sparse::with_plan(LocalExecutionPlan::serial(), || {
            runmat_sparse_solvers::precond::Amg::new(
                crate::solve::shared_sparse::CsrArrays::of(
                    reduced_operator.operators().stiffness(),
                )
                .view(),
                runmat_sparse_solvers::precond::AmgOptions {
                    block_size,
                    near_nullspace,
                    strength_threshold: if block_size > 1 { 0.02 } else { 0.0 },
                    ..Default::default()
                },
                crate::solve::shared_sparse::control(),
            )
        }) {
            Ok(amg) => multigrid = Some(amg),
            Err(error) => {
                linear_diagnostics.push(FeaDiagnostic {
                    code: "FEA_MODAL_SOLVER_ADMISSION_FAILED".into(),
                    severity: FeaDiagnosticSeverity::Error,
                    message: format!("modal multigrid preconditioner setup failed: {error}"),
                });
            }
        }
    }
    let mut last_values: Vec<f64> = Vec::new();
    let stiffness =
        crate::solve::shared_sparse::CsrArrays::of(reduced_operator.operators().stiffness());
    let mass = crate::solve::shared_sparse::CsrArrays::of(reduced_operator.operators().mass());
    let mut modes: Vec<ModalCandidate> = Vec::with_capacity(target_mode_count.saturating_add(4));
    let mut boundary_probe_count = 0usize;
    let mut complete_boundary_eigenspace = false;
    let mut capacity_reached = false;
    let mut boundary_probe_frequency_hz = None;
    let mut subspace_dimension = target_mode_count
        .saturating_add(MODAL_BOUNDARY_PROBE_WIDTH)
        .min(maximum_subspace_dimension);
    let mut subspace_iteration_count = 0usize;
    let mut subspace_maximum_residual = f64::INFINITY;
    let mut subspace_maximum_relative_residual = f64::INFINITY;
    let mut subspace_maximum_backward_error = f64::INFINITY;
    let mut subspace_converged = false;
    let mut sturm = None;
    let tolerance = convergence.algebraic_tolerance();
    let admitted = path == SubspaceSolvePath::ShiftInvertFactorization || multigrid.is_some();
    while admitted && subspace_dimension > 0 && !crate::is_fea_cancelled() {
        let options = modal_eigen_options(subspace_dimension, convergence);
        let solved = crate::solve::shared_sparse::with_plan(LocalExecutionPlan::serial(), || {
            let (k, m) = (stiffness.view(), mass.view());
            let control = crate::solve::shared_sparse::control();
            match (path, factor.as_deref(), multigrid.as_ref()) {
                (SubspaceSolvePath::PreconditionedLobpcg, _, Some(amg)) => {
                    runmat_sparse_solvers::eigen::lobpcg(
                        k,
                        m,
                        Some(amg),
                        &modal_lobpcg_options(subspace_dimension, convergence).lobpcg,
                        control,
                    )
                }
                (_, Some(solver), _) => runmat_sparse_solvers::eigen::modal_with_factor(
                    k, m, solver, 0.0, &options, control,
                ),
                (_, None, _) => runmat_sparse_solvers::eigen::modal(
                    k,
                    m,
                    &runmat_sparse_solvers::eigen::EigenOptions {
                        sturm_check: false,
                        ..options
                    },
                    control,
                ),
            }
        });
        let result = match solved {
            Ok(result) => result,
            Err(error) => {
                linear_diagnostics.push(FeaDiagnostic {
                    code: "FEA_MODAL_EIGENSOLVER_FAILED".into(),
                    severity: FeaDiagnosticSeverity::Error,
                    message: error.to_string(),
                });
                break;
            }
        };
        subspace_iteration_count = subspace_iteration_count.saturating_add(result.restarts.max(1));
        if result.backward_errors.len() != result.residuals.len() {
            linear_diagnostics.push(FeaDiagnostic {
                code: "FEA_MODAL_EIGENSOLVER_FAILED".into(),
                severity: FeaDiagnosticSeverity::Error,
                message: "eigensolver returned no backward error for its pairs".into(),
            });
            break;
        }
        subspace_maximum_relative_residual = result.residuals.iter().copied().fold(0.0, f64::max);
        subspace_maximum_backward_error =
            result.backward_errors.iter().copied().fold(0.0, f64::max);
        subspace_maximum_residual = convergence.gated_residual(
            subspace_maximum_relative_residual,
            subspace_maximum_backward_error,
        );
        subspace_converged = result.converged && subspace_maximum_residual <= tolerance;
        last_values.clone_from(&result.values);
        let n = unconstrained_count;
        let mut candidates = Vec::with_capacity(result.values.len());
        for (index, eigenvalue) in result.values.iter().copied().enumerate() {
            let vector = &result.vectors[index * n..(index + 1) * n];
            let Ok(expanded) = prepared_modal.expand(vector) else {
                candidates.clear();
                break;
            };
            candidates.push((
                eigenvalue.max(0.0).sqrt() / (2.0 * std::f64::consts::PI),
                expanded,
                result.residuals[index],
                result.backward_errors[index],
            ));
        }
        let exhausted = subspace_dimension == unconstrained_count;
        if !subspace_converged {
            modes = candidates;
            break;
        }
        if let Some(boundary) = resolve_truncation_boundary(
            &candidates,
            target_mode_count,
            eigenspace_relative_tolerance,
            exhausted,
        ) {
            boundary_probe_count = boundary.probe_count;
            boundary_probe_frequency_hz = boundary.probe_frequency_hz;
            modes = boundary.retained;
            complete_boundary_eigenspace = true;
            break;
        }
        if exhausted || subspace_dimension == maximum_subspace_dimension {
            capacity_reached = !exhausted;
            modes = candidates;
            break;
        }
        subspace_dimension = subspace_dimension
            .saturating_add(MODAL_BOUNDARY_PROBE_WIDTH)
            .min(maximum_subspace_dimension);
    }
    // Completeness proof, after every factor and preconditioner is released: the shift-invert
    // factor and the Sturm count never coexist, and the LOBPCG path first proves K positive
    // definite with a factor-free inertia count at shift 0.
    drop(multigrid);
    let _ = factor.take();
    drop(prepared);
    reuse.release_factor();
    if subspace_converged && !last_values.is_empty() && !crate::is_fea_cancelled() {
        let proved = crate::solve::shared_sparse::with_plan(LocalExecutionPlan::serial(), || {
            modal_completeness_proof(
                stiffness.view(),
                mass.view(),
                &last_values,
                path,
                convergence,
                subspace_dimension,
            )
        });
        match proved {
            Ok(check) => sturm = Some(check),
            Err(error) => {
                subspace_converged = false;
                linear_diagnostics.push(FeaDiagnostic {
                    code: match error {
                        ModalProofError::Coercivity(_) => "FEA_MODAL_SOLVER_ADMISSION_FAILED",
                        ModalProofError::Inertia(_) => "FEA_MODAL_STURM_CHECK",
                    }
                    .into(),
                    severity: FeaDiagnosticSeverity::Error,
                    message: error.to_string(),
                });
            }
        }
    }
    if let Some(sturm) = sturm {
        linear_diagnostics.push(FeaDiagnostic {
            code: "FEA_MODAL_STURM_CHECK".into(),
            severity: if sturm.complete() {
                FeaDiagnosticSeverity::Info
            } else {
                FeaDiagnosticSeverity::Error
            },
            message: format!(
                "eigenvalues_below_shift={} returned_below_shift={} shift={} complete={}",
                sturm.counted,
                sturm.returned,
                sturm.shift,
                sturm.complete()
            ),
        });
        if !sturm.complete() {
            subspace_converged = false;
        }
    }
    let solver_backend = LinearAlgebraBackendKind::CpuReference;
    let (solver_host_sync_count, device_apply_k_count, device_apply_k_attempt_count) =
        (0u32, 0u32, 0u32);
    let fallback_apply_count = 0u32;
    let accelerator_evidence = Vec::new();
    let unconstrained: Vec<usize> = (0..unconstrained_count).collect();

    modes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut frequencies_hz = Vec::with_capacity(modes.len());
    let mut mode_shapes = Vec::with_capacity(modes.len());
    let mut residual_norms = Vec::with_capacity(modes.len());
    let mut backward_errors = Vec::with_capacity(modes.len());
    for (freq_hz, shape, residual, backward_error) in modes {
        frequencies_hz.push(freq_hz);
        mode_shapes.push(shape);
        residual_norms.push(residual);
        backward_errors.push(backward_error);
    }

    let converged = subspace_converged
        && !crate::is_fea_cancelled()
        && frequencies_hz.len() >= target_mode_count
        && complete_boundary_eigenspace;
    let mut diagnostics = vec![FeaDiagnostic {
        code: "FEA_MODAL_METHOD".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: match path {
            SubspaceSolvePath::ShiftInvertFactorization => {
                "solver=shift_invert_block_lanczos shift=0 factor=sparse_cholesky sturm_check=factor_free_inertia"
            }
            SubspaceSolvePath::PreconditionedLobpcg => {
                "solver=block_lobpcg preconditioner=smoothed_aggregation_amg coercivity=factor_free_inertia sturm_check=factor_free_inertia"
            }
        }
        .to_string(),
    }];
    diagnostics.extend(linear_diagnostics);
    diagnostics.push(FeaDiagnostic {
        code: "FEA_MODAL_CONVERGENCE".to_string(),
        severity: if converged {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "mode_count_requested={} mode_count_solved={} subspace_dimension={} iteration_count={} measure={} tolerance={} maximum_gated_residual={} maximum_backward_error={} maximum_relative_residual={} converged={}",
            requested_mode_count,
            frequencies_hz.len(),
            subspace_dimension,
            subspace_iteration_count,
            convergence.algebraic_measure().label(),
            tolerance,
            subspace_maximum_residual,
            subspace_maximum_backward_error,
            subspace_maximum_relative_residual,
            converged
        ),
    });
    if !frequencies_hz.is_empty() {
        diagnostics.push(FeaDiagnostic {
            code: "FEA_MODAL_ACCURACY".into(),
            severity: FeaDiagnosticSeverity::Info,
            message: format!(
                "measure={} tolerance={} modes=[{}]",
                convergence.algebraic_measure().label(),
                tolerance,
                frequencies_hz
                    .iter()
                    .zip(&backward_errors)
                    .zip(&residual_norms)
                    .enumerate()
                    .map(|(index, ((frequency, backward), relative))| format!(
                        "{{mode={} frequency_hz={frequency} backward_error={backward:e} relative_residual={relative:e}}}",
                        index + 1
                    ))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        });
    }

    if !complete_boundary_eigenspace
        && subspace_dimension == maximum_subspace_dimension
        && maximum_subspace_dimension < unconstrained.len()
    {
        diagnostics.push(FeaDiagnostic {
            code: "FEA_MODAL_SUBSPACE_CAPACITY".into(),
            severity: FeaDiagnosticSeverity::Error,
            message: format!(
                "modal truncation boundary remains unproven at admitted subspace capacity: requested={target_mode_count} capacity={maximum_subspace_dimension} available={}",
                unconstrained.len(),
            ),
        });
    }
    push_modal_quality_diagnostics(
        &mut diagnostics,
        &summary.operator,
        &frequencies_hz,
        &mode_shapes,
        &residual_norms,
        &backward_errors,
        convergence,
    );
    diagnostics.push(FeaDiagnostic {
        code: "FEA_MODAL_COST".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "subspace_iteration_count={} fallback_apply_count={} device_apply_k_count={} device_apply_k_attempt_count={} host_sync_count={}",
            subspace_iteration_count,
            fallback_apply_count,
            device_apply_k_count,
            device_apply_k_attempt_count,
            solver_host_sync_count,
        ),
    });

    ModalSolveResult {
        converged,
        frequencies_hz,
        mode_shapes,
        residual_norms,
        backward_errors,
        diagnostics,
        solver_method,
        solver_backend,
        preconditioner: path_preconditioner,
        solver_host_sync_count,
        device_apply_k_count,
        device_apply_k_attempt_count,
        accelerator_evidence,
        available_mode_count: unconstrained.len(),
        boundary_probe_count,
        boundary_probe_frequency_hz,
        boundary_unresolved_at_capacity: capacity_reached.then_some(maximum_subspace_dimension),
    }
}

#[derive(Debug, thiserror::Error)]
enum ModalProofError {
    #[error("the factor-free inertia count does not prove the stiffness positive definite on the constraint space: {0}")]
    Coercivity(String),
    #[error("the factor-free Sturm inertia count failed: {0}")]
    Inertia(String),
}

/// The proof that no mode below the reported ones was missed: one factor-free inertia analysis
/// of the reduced pencil, counted at the Sturm probe above the highest returned value. The
/// LOBPCG path first counts at shift 0 and requires no negative or zero pivot, which proves K
/// positive definite in place of the Cholesky proof it never ran.
fn modal_completeness_proof(
    stiffness: runmat_sparse_solvers::csr::CsrRef<'_, f64>,
    mass: runmat_sparse_solvers::csr::CsrRef<'_, f64>,
    values: &[f64],
    path: SubspaceSolvePath,
    convergence: crate::pipeline::modal_governance::ModalConvergencePolicy,
    subspace_dimension: usize,
) -> Result<runmat_sparse_solvers::eigen::SturmCheck, ModalProofError> {
    let control = crate::solve::shared_sparse::control();
    let options = modal_eigen_options(subspace_dimension, convergence);
    let inertia = runmat_sparse_solvers::eigen::PencilInertia::analyze(
        stiffness,
        mass,
        &options.ordering,
        control,
    )
    .map_err(|error| ModalProofError::Inertia(error.to_string()))?;
    if path == SubspaceSolvePath::PreconditionedLobpcg {
        let at_zero = inertia
            .count(0.0, control)
            .map_err(|error| ModalProofError::Coercivity(error.to_string()))?;
        if at_zero.negative != 0 || at_zero.zero != 0 {
            return Err(ModalProofError::Coercivity(format!(
                "{} negative and {} zero pivots at shift 0",
                at_zero.negative, at_zero.zero
            )));
        }
    }
    runmat_sparse_solvers::eigen::sturm_check_with(&inertia, values, options.tolerance, control)
        .map_err(|error| ModalProofError::Inertia(error.to_string()))
}
