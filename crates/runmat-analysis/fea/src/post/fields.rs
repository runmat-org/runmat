use crate::algebra::{capacity::measured, coercivity::NumericalCapacityError};
use runmat_analysis_core::AnalysisField;
use runmat_simulation_usage::{AsyncCapacityOwner, LedgerError, OperationCount, UsageCategory};

use crate::contracts::{
    FEA_FIELD_STRUCTURAL_DISPLACEMENT, FEA_FIELD_STRUCTURAL_EQUATION_SCALE,
    FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRAIN, FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS,
    FEA_FIELD_STRUCTURAL_NODAL_STRESS, FEA_FIELD_STRUCTURAL_NODAL_VON_MISES,
    FEA_FIELD_STRUCTURAL_REACTION_FORCE, FEA_FIELD_STRUCTURAL_REACTION_MOMENT,
    FEA_FIELD_STRUCTURAL_RESIDUAL_NORM, FEA_FIELD_STRUCTURAL_ROTATION, FEA_FIELD_STRUCTURAL_STRAIN,
    FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY, FEA_FIELD_STRUCTURAL_STRESS,
    FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY, FEA_FIELD_STRUCTURAL_VON_MISES,
};
use crate::{
    assembly::AssemblySummary, planning::resource::LocalExecutionPlan,
    post::error::StructuralFieldRecoveryError, solve::linear::LinearSolveResult,
};

mod catalog;
mod edge;
mod formulation;
mod scalars;
mod state;

pub use catalog::{
    structural_observable_field_definitions, structural_retained_field_inventory,
    structural_static_observable_field_definitions, take_structural_observable_fields,
    StructuralLinearFieldQuantity,
};
use edge::edge_strain_tensor;
use scalars::{recover_strain_energy_density, recover_stress, recover_von_mises};
use state::{
    recover_displacement, recover_reaction_force, recover_reaction_moment, recover_rotation,
    rotation_shape, structural_displacement_node_count,
};

const VECTOR_COMPONENT_COUNT: usize = 3;
const TENSOR_COMPONENT_COUNT: usize = 6;

fn immediate<T>(
    future: impl std::future::Future<
        Output = Result<
            T,
            NumericalCapacityError<StructuralFieldRecoveryError, std::convert::Infallible>,
        >,
    >,
) -> Result<T, StructuralFieldRecoveryError> {
    use futures::FutureExt;
    future
        .now_or_never()
        .expect("native field recovery capacity is immediately ready")
        .map_err(|error| match error {
            NumericalCapacityError::Numerical(error) => error,
            NumericalCapacityError::Admission(never) => match never {},
        })
}

fn count(length: usize) -> Result<OperationCount, StructuralFieldRecoveryError> {
    Ok(OperationCount::from_operations(
        u64::try_from(length).map_err(|_| LedgerError::CountOverflow)?,
    ))
}

fn tensor_work(
    values: &[f64],
    per_tensor: u64,
) -> Result<OperationCount, StructuralFieldRecoveryError> {
    count(values.len() / TENSOR_COMPONENT_COUNT)?
        .checked_mul(per_tensor)
        .map_err(LedgerError::from)
        .map_err(StructuralFieldRecoveryError::from)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DeformationRecoveryDetail {
    Complete,
    TransientHistory,
}

pub fn recover_result_fields(
    summary: &AssemblySummary,
    solve_result: &LinearSolveResult,
    local_execution: LocalExecutionPlan,
) -> Result<Vec<AnalysisField>, StructuralFieldRecoveryError> {
    immediate(recover_result_fields_with_capacity::<
        std::convert::Infallible,
    >(summary, solve_result, local_execution, None))
}

pub(crate) async fn recover_result_fields_with_capacity<A>(
    summary: &AssemblySummary,
    solve_result: &LinearSolveResult,
    local_execution: LocalExecutionPlan,
    owner: Option<&dyn AsyncCapacityOwner<A>>,
) -> Result<Vec<AnalysisField>, NumericalCapacityError<StructuralFieldRecoveryError, A>> {
    if solve_result.solution.is_empty()
        || solve_result.solution.iter().any(|value| !value.is_finite())
    {
        return Ok(empty_structural_fields());
    }

    let node_count = structural_displacement_node_count(summary, summary.dof_count.max(3));
    let physical_operator =
        crate::assembly::physical_fields::StructuralPhysicalOperator::from_assembly(summary)
            .map_err(
                |error| StructuralFieldRecoveryError::InvalidPhysicalLayout {
                    reason: format!("{error:?}"),
                },
            )?;
    let physical_state = summary
        .state_field(0, &solve_result.solution)
        .map_err(
            |error| StructuralFieldRecoveryError::InvalidPhysicalLayout {
                reason: format!("{error:?}"),
            },
        )?;
    let mut spmv_work = OperationCount::ZERO;
    let matrix = summary.operator.operators().stiffness();
    for row in 0..matrix.rows() {
        spmv_work = spmv_work
            .checked_add(
                OperationCount::direct_real_dot(
                    count(matrix.pattern().row_range(row).len())?.operations(),
                )
                .map_err(LedgerError::from)
                .map_err(StructuralFieldRecoveryError::from)?,
            )
            .map_err(LedgerError::from)
            .map_err(StructuralFieldRecoveryError::from)?;
    }
    let internal_force = measured(owner, UsageCategory::Solve, spmv_work, || {
        physical_operator
            .apply(
                crate::assembly::physical_fields::StructuralOperatorRole::Stiffness,
                &physical_state,
            )
            .map_err(
                |error| StructuralFieldRecoveryError::InvalidPhysicalLayout {
                    reason: format!("{error:?}"),
                },
            )
    })
    .await?;
    let (reaction_values, reaction_moment_values) = measured(
        owner,
        UsageCategory::Solve,
        state::reaction_reference_work(summary, node_count)?,
        || {
            Ok::<_, StructuralFieldRecoveryError>(internal_force.with_canonical_storage(|values| {
                (
                    recover_reaction_force(summary, values, node_count),
                    recover_reaction_moment(summary, values),
                )
            }))
        },
    )
    .await?;
    let residual_metrics =
        state::recover_residual_metrics_with_capacity(summary, &solve_result.solution, owner)
            .await?;
    let mut fields = recover_deformation_result_fields_with_capacity(
        summary,
        &solve_result.solution,
        local_execution,
        DeformationRecoveryDetail::Complete,
        owner,
    )
    .await?;
    let strain_energy = if summary.structural_solid_recovery.iter().any(|element| {
        element.thermal_eigenstrain.is_some() || element.solved_thermal_quadrature.is_some()
    }) {
        super::thermoelastic_energy::total_elastic_energy_with_capacity(summary, &fields, owner)
            .await?
    } else {
        let work =
            OperationCount::direct_real_dot(count(solve_result.solution.len())?.operations())
                .and_then(|work| work.checked_add(OperationCount::from_operations(1)))
                .map_err(LedgerError::from)
                .map_err(StructuralFieldRecoveryError::from)?;
        measured(owner, UsageCategory::Solve, work, || {
            physical_operator
                .elastic_energy(&physical_state, &internal_force)
                .map_err(
                    |error| StructuralFieldRecoveryError::InvalidPhysicalLayout {
                        reason: format!("{error:?}"),
                    },
                )
                .map(|energy| energy.canonical_value())
        })
        .await?
    };
    fields.extend([
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_REACTION_FORCE,
            vec![node_count, VECTOR_COMPONENT_COUNT],
            reaction_values,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_REACTION_MOMENT,
            rotation_shape(summary),
            reaction_moment_values,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY,
            vec![1],
            vec![strain_energy],
        ),
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_RESIDUAL_NORM,
            vec![1],
            vec![residual_metrics.normalized_residual_norm],
        ),
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_EQUATION_SCALE,
            vec![1],
            vec![residual_metrics.equation_scale],
        ),
    ]);
    Ok(fields)
}

/// Recovers only deformation-dependent physical fields from a structural state vector.
///
/// Modal shapes have no static load equilibrium, so reaction and residual fields would be
/// misleading. This shared path keeps their strain, stress, energy-density, beam, and shell
/// recovery identical to linear-static recovery without manufacturing equilibrium quantities.
pub fn recover_deformation_result_fields(
    summary: &AssemblySummary,
    solution: &[f64],
) -> Result<Vec<AnalysisField>, StructuralFieldRecoveryError> {
    recover_deformation_result_fields_with_plan(
        summary,
        solution,
        LocalExecutionPlan::serial(),
        DeformationRecoveryDetail::Complete,
    )
}

/// Intermediate transient states retain element averages, not integration-point tensors.
/// The final accepted state still uses complete recovery for ordinary result fields.
pub(crate) fn recover_transient_history_deformation_fields(
    summary: &AssemblySummary,
    solution: &[f64],
) -> Result<Vec<AnalysisField>, StructuralFieldRecoveryError> {
    recover_deformation_result_fields_with_plan(
        summary,
        solution,
        LocalExecutionPlan::serial(),
        DeformationRecoveryDetail::TransientHistory,
    )
}

fn recover_deformation_result_fields_with_plan(
    summary: &AssemblySummary,
    solution: &[f64],
    local_execution: LocalExecutionPlan,
    detail: DeformationRecoveryDetail,
) -> Result<Vec<AnalysisField>, StructuralFieldRecoveryError> {
    immediate(recover_deformation_result_fields_with_capacity::<
        std::convert::Infallible,
    >(summary, solution, local_execution, detail, None))
}

async fn recover_deformation_result_fields_with_capacity<A>(
    summary: &AssemblySummary,
    solution: &[f64],
    local_execution: LocalExecutionPlan,
    detail: DeformationRecoveryDetail,
    owner: Option<&dyn AsyncCapacityOwner<A>>,
) -> Result<Vec<AnalysisField>, NumericalCapacityError<StructuralFieldRecoveryError, A>> {
    if solution.is_empty() || solution.iter().any(|value| !value.is_finite()) {
        return Ok(Vec::new());
    }
    if solution.len() != summary.dof_count {
        return Err(StructuralFieldRecoveryError::DisplacementShapeMismatch {
            expected: summary.dof_count,
            actual: solution.len(),
        }
        .into());
    }
    let node_count = structural_displacement_node_count(summary, summary.dof_count.max(3));
    let displacement_values = recover_displacement(summary, solution, node_count);
    let continuum_fields = if publishes_continuum_fields(summary) {
        let strain_recovery = recover_structural_strain_with_capacity(
            summary,
            &displacement_values,
            local_execution,
            detail == DeformationRecoveryDetail::Complete,
            owner,
        )
        .await?;
        let element_count = strain_recovery.element_count;
        let strain_values = strain_recovery.values;
        let stress_values = match strain_recovery.stress {
            Some(stress) => stress,
            None => {
                measured(
                    owner,
                    UsageCategory::Solve,
                    tensor_work(&strain_values, 17)?,
                    || {
                        Ok::<_, StructuralFieldRecoveryError>(recover_stress(
                            &strain_values,
                            summary.structural_material,
                        ))
                    },
                )
                .await?
            }
        };
        let von_mises_values = measured(
            owner,
            UsageCategory::Solve,
            tensor_work(&stress_values, 17)?,
            || Ok::<_, StructuralFieldRecoveryError>(recover_von_mises(&stress_values)),
        )
        .await?;
        // Nodal stress is recovered as a tensor from each element's own field at its nodes,
        // then averaged; nodal von Mises is computed from that averaged tensor.
        let nodal_stress_values = if detail == DeformationRecoveryDetail::Complete {
            super::volume_weighted::recover_volume_weighted_nodal_tensor_with_capacity(
                &summary.structural_solid_recovery,
                &strain_recovery.integration_point_stress,
                node_count,
                owner,
            )
            .await?
        } else {
            Vec::new()
        };
        let nodal_von_mises_values = measured(
            owner,
            UsageCategory::Solve,
            tensor_work(&nodal_stress_values, 17)?,
            || Ok::<_, StructuralFieldRecoveryError>(recover_von_mises(&nodal_stress_values)),
        )
        .await?;
        let strain_energy_density_values = match strain_recovery.strain_energy_density {
            Some(energy) => energy,
            None => {
                measured(
                    owner,
                    UsageCategory::Solve,
                    tensor_work(&strain_values, 12)?,
                    || {
                        Ok::<_, StructuralFieldRecoveryError>(recover_strain_energy_density(
                            &strain_values,
                            &stress_values,
                        ))
                    },
                )
                .await?
            }
        };
        let integration_point_strain = strain_recovery.integration_point_strain;
        let integration_point_stress = strain_recovery.integration_point_stress;
        Some((
            element_count,
            strain_values,
            stress_values,
            von_mises_values,
            nodal_von_mises_values,
            nodal_stress_values,
            strain_energy_density_values,
            integration_point_strain,
            integration_point_stress,
        ))
    } else {
        None
    };

    let mut fields = vec![
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_DISPLACEMENT,
            vec![node_count, VECTOR_COMPONENT_COUNT],
            displacement_values,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_ROTATION,
            rotation_shape(summary),
            recover_rotation(summary, solution),
        ),
    ];
    if let Some((
        element_count,
        strain_values,
        stress_values,
        von_mises_values,
        nodal_von_mises_values,
        nodal_stress_values,
        strain_energy_density_values,
        integration_point_strain,
        integration_point_stress,
    )) = continuum_fields
    {
        fields.extend([
            AnalysisField::host_f64(
                FEA_FIELD_STRUCTURAL_VON_MISES,
                vec![element_count],
                von_mises_values,
            ),
            AnalysisField::host_f64(
                FEA_FIELD_STRUCTURAL_STRAIN,
                vec![element_count, TENSOR_COMPONENT_COUNT],
                strain_values,
            ),
            AnalysisField::host_f64(
                FEA_FIELD_STRUCTURAL_STRESS,
                vec![element_count, TENSOR_COMPONENT_COUNT],
                stress_values,
            ),
            AnalysisField::host_f64(
                FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY,
                vec![element_count],
                strain_energy_density_values,
            ),
        ]);
        if detail == DeformationRecoveryDetail::Complete {
            fields.extend([
                AnalysisField::host_f64(
                    FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRAIN,
                    vec![element_count * 4, TENSOR_COMPONENT_COUNT],
                    integration_point_strain,
                ),
                AnalysisField::host_f64(
                    FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS,
                    vec![element_count * 4, TENSOR_COMPONENT_COUNT],
                    integration_point_stress,
                ),
                AnalysisField::host_f64(
                    FEA_FIELD_STRUCTURAL_NODAL_VON_MISES,
                    vec![node_count],
                    nodal_von_mises_values,
                ),
                AnalysisField::host_f64(
                    FEA_FIELD_STRUCTURAL_NODAL_STRESS,
                    vec![node_count, TENSOR_COMPONENT_COUNT],
                    nodal_stress_values,
                ),
            ]);
        }
    }
    if let Some(beam_fields) = formulation::recover_beam_result_fields_with_capacity(
        summary,
        solution,
        local_execution,
        owner,
    )
    .await?
    {
        fields.extend(beam_fields);
    }
    if let Some(shell_fields) = formulation::recover_shell_result_fields_with_capacity(
        summary,
        solution,
        local_execution,
        owner,
    )
    .await?
    {
        fields.extend(shell_fields);
    }
    Ok(fields)
}

fn publishes_continuum_fields(summary: &AssemblySummary) -> bool {
    !summary.structural_solid_recovery.is_empty()
}

/// Moves formulation-owned spatial fields with stable physical semantics into the observation
/// domain.
///
/// Diagnostics, global summaries, derived display conveniences, and unsupported beam/shell
/// resultants remain in `fields`; they are not mislabeled merely to make them observable.
/// Constraint-group reactions also remain here because their canonical values are recovered from
/// [`crate::constraints::CanonicalStructuralConstraintResult`], which preserves authored
/// constraint ownership when constraints share nodes or degrees of freedom.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StructuralFieldRecoveryMetrics {
    pub active_stiffness_edge_count: usize,
    pub constrained_edge_count: usize,
    pub recovery_element_count: usize,
    pub solver_mesh_node_count: usize,
    pub solver_mesh_element_count: usize,
    pub max_edge_displacement_jump: f64,
    pub max_edge_strain_norm: f64,
    pub mean_edge_stiffness_ratio: f64,
    pub mean_edge_length_m: f64,
    pub strain_component_coverage_ratio: f64,
    pub element_geometry_node_count: usize,
    pub element_geometry_edge_count: usize,
    pub element_geometry_coverage_ratio: f64,
    pub basis: &'static str,
}

pub fn structural_field_recovery_metrics(
    summary: &AssemblySummary,
    displacement: &[f64],
) -> Result<StructuralFieldRecoveryMetrics, StructuralFieldRecoveryError> {
    let strain_recovery = recover_structural_strain(summary, displacement)?;
    Ok(recovery_metrics_from_strain(
        summary,
        displacement,
        &strain_recovery.values,
        strain_recovery.element_count,
        strain_recovery.basis,
    ))
}

/// Uses the completed physical field rather than running element recovery again.
pub(crate) fn structural_field_recovery_metrics_from_fields(
    summary: &AssemblySummary,
    displacement: &[f64],
    fields: &[AnalysisField],
) -> Result<StructuralFieldRecoveryMetrics, StructuralFieldRecoveryError> {
    if summary.structural_solid_recovery.is_empty() {
        let edges = structural_recovery_edges(summary);
        return Ok(recovery_metrics_from_strain(
            summary,
            displacement,
            &recover_edge_strain(displacement, &edges),
            edges.len().max(1),
            StructuralRecoveryBasis::OperatorConnectivity.as_str(),
        ));
    }
    let strain = fields
        .iter()
        .find(|field| field.field_id == FEA_FIELD_STRUCTURAL_STRAIN)
        .and_then(AnalysisField::as_host_f64)
        .ok_or_else(|| StructuralFieldRecoveryError::InvalidPhysicalLayout {
            reason: "completed structural strain field is absent".into(),
        })?;
    let expected = summary.structural_solid_recovery.len() * TENSOR_COMPONENT_COUNT;
    if strain.len() != expected {
        return Err(StructuralFieldRecoveryError::ElementValueCountMismatch {
            expected,
            actual: strain.len(),
        });
    }
    Ok(recovery_metrics_from_strain(
        summary,
        displacement,
        strain,
        summary.structural_solid_recovery.len(),
        solid_recovery_basis(summary),
    ))
}

fn recovery_metrics_from_strain(
    summary: &AssemblySummary,
    displacement: &[f64],
    strain: &[f64],
    element_count: usize,
    basis: &'static str,
) -> StructuralFieldRecoveryMetrics {
    let recovery_edges = structural_recovery_edges(summary);
    let active_stiffness_edge_count = recovery_edges.len();
    let constrained_edge_count = constrained_recovery_edge_count(summary);
    let mut max_edge_displacement_jump = 0.0_f64;
    let mut max_edge_strain_norm = 0.0_f64;
    let mut stiffness_ratio_sum = 0.0_f64;
    let mut edge_length_sum = 0.0_f64;
    for edge in &recovery_edges {
        let jump = displacement
            .get(edge.to_dof)
            .zip(displacement.get(edge.from_dof))
            .map(|(right, left)| (right - left).abs())
            .unwrap_or(0.0);
        max_edge_displacement_jump = max_edge_displacement_jump.max(jump);
        stiffness_ratio_sum += edge.stiffness_ratio;
        edge_length_sum += edge.edge_length_m;
    }
    for strain_tensor in strain.chunks_exact(TENSOR_COMPONENT_COUNT) {
        let strain_norm = strain_tensor
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        max_edge_strain_norm = max_edge_strain_norm.max(strain_norm);
    }
    let recovery_edge_count = recovery_edges.len();
    let mean_edge_stiffness_ratio = if recovery_edge_count == 0 {
        0.0
    } else {
        stiffness_ratio_sum / recovery_edge_count as f64
    };
    let mean_edge_length_m = if recovery_edge_count == 0 {
        0.0
    } else {
        edge_length_sum / recovery_edge_count as f64
    };
    let strain_component_coverage_ratio =
        strain_component_coverage_ratio(displacement, &recovery_edges);
    let solver_mesh_element_count = summary.structural_solid_recovery.len();
    let solver_mesh_node_count = if solver_mesh_element_count == 0 {
        0
    } else {
        let mut nodes = summary
            .structural_solid_recovery
            .iter()
            .flat_map(|element| element.geometry.node_indices().iter().copied())
            .collect::<Vec<_>>();
        nodes.sort_unstable();
        nodes.dedup();
        nodes.len()
    };

    StructuralFieldRecoveryMetrics {
        active_stiffness_edge_count,
        constrained_edge_count,
        recovery_element_count: element_count,
        solver_mesh_node_count,
        solver_mesh_element_count,
        max_edge_displacement_jump,
        max_edge_strain_norm,
        mean_edge_stiffness_ratio,
        mean_edge_length_m,
        strain_component_coverage_ratio,
        element_geometry_node_count: solver_mesh_node_count,
        element_geometry_edge_count: solver_mesh_element_count.saturating_mul(6),
        element_geometry_coverage_ratio: f64::from(solver_mesh_element_count > 0),
        basis,
    }
}

fn solid_recovery_basis(summary: &AssemblySummary) -> &'static str {
    if summary.structural_solid_recovery.iter().all(|element| {
        matches!(
            element.geometry,
            crate::assembly::SolidRecoveryGeometry::Tetrahedron4 { .. }
        )
    }) {
        "solid_tetrahedron4_constant_strain"
    } else {
        "solid_tetrahedron10_integration_point_projection"
    }
}

struct StructuralStrainRecovery {
    values: Vec<f64>,
    stress: Option<Vec<f64>>,
    strain_energy_density: Option<Vec<f64>>,
    integration_point_strain: Vec<f64>,
    integration_point_stress: Vec<f64>,
    element_count: usize,
    basis: &'static str,
}

fn empty_structural_fields() -> Vec<AnalysisField> {
    vec![
        AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_DISPLACEMENT, vec![0, 3], Vec::new()),
        AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_ROTATION, vec![0, 3], Vec::new()),
        AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_VON_MISES, vec![0], Vec::new()),
        AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_STRAIN, vec![0, 6], Vec::new()),
        AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_STRESS, vec![0, 6], Vec::new()),
        AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_NODAL_VON_MISES, vec![0], Vec::new()),
        AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_NODAL_STRESS, vec![0, 6], Vec::new()),
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY,
            vec![0],
            Vec::new(),
        ),
        AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_REACTION_FORCE, vec![0, 3], Vec::new()),
        AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_REACTION_MOMENT, vec![0, 3], Vec::new()),
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY,
            vec![0],
            Vec::new(),
        ),
        AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_RESIDUAL_NORM, vec![0], Vec::new()),
        AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_EQUATION_SCALE, vec![0], Vec::new()),
    ]
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct StructuralRecoveryEdge {
    from_dof: usize,
    to_dof: usize,
    component: usize,
    hop: usize,
    edge_length_m: f64,
    stiffness_ratio: f64,
    basis: StructuralRecoveryBasis,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StructuralRecoveryBasis {
    OperatorConnectivity,
}

impl StructuralRecoveryBasis {
    const fn as_str(self) -> &'static str {
        match self {
            Self::OperatorConnectivity => "operator_connectivity",
        }
    }
}

fn structural_recovery_edges(summary: &AssemblySummary) -> Vec<StructuralRecoveryEdge> {
    let matrix = summary.operator.operators().stiffness();
    let constrained = summary.operator.constrained_dofs();
    let mut edges = Vec::new();
    for from_dof in 0..matrix.rows() {
        for entry in matrix.pattern().row_range(from_dof) {
            let to_dof = matrix.pattern().column(entry);
            let stiffness = matrix.values()[entry].abs();
            if to_dof <= from_dof
                || stiffness == 0.0
                || constrained[from_dof]
                || constrained[to_dof]
            {
                continue;
            }
            let left_diag = matrix.get(from_dof, from_dof);
            let right_diag = matrix.get(to_dof, to_dof);
            let diag_scale = (0.5 * (left_diag + right_diag)).abs().max(1.0);
            edges.push(StructuralRecoveryEdge {
                from_dof,
                to_dof,
                component: from_dof % VECTOR_COMPONENT_COUNT,
                hop: to_dof - from_dof,
                edge_length_m: 1.0,
                stiffness_ratio: stiffness / diag_scale,
                basis: StructuralRecoveryBasis::OperatorConnectivity,
            });
        }
    }
    edges
}

fn constrained_recovery_edge_count(summary: &AssemblySummary) -> usize {
    structural_recovery_edges_including_constraints(summary)
        .into_iter()
        .filter(|(left, right)| {
            summary.operator.constrained_dofs()[*left]
                || summary.operator.constrained_dofs()[*right]
        })
        .count()
}

fn structural_recovery_edges_including_constraints(
    summary: &AssemblySummary,
) -> Vec<(usize, usize)> {
    let matrix = summary.operator.operators().stiffness();
    let mut edges = Vec::new();
    for row in 0..matrix.rows() {
        for entry in matrix.pattern().row_range(row) {
            let column = matrix.pattern().column(entry);
            if column > row && matrix.values()[entry] != 0.0 {
                edges.push((row, column));
            }
        }
    }
    edges
}

fn recover_structural_strain(
    summary: &AssemblySummary,
    displacement: &[f64],
) -> Result<StructuralStrainRecovery, StructuralFieldRecoveryError> {
    recover_structural_strain_with_plan(summary, displacement, LocalExecutionPlan::serial())
}

fn recover_structural_strain_with_plan(
    summary: &AssemblySummary,
    displacement: &[f64],
    local_execution: LocalExecutionPlan,
) -> Result<StructuralStrainRecovery, StructuralFieldRecoveryError> {
    recover_structural_strain_with_detail(summary, displacement, local_execution, true)
}

fn recover_structural_strain_with_detail(
    summary: &AssemblySummary,
    displacement: &[f64],
    local_execution: LocalExecutionPlan,
    retain_integration_points: bool,
) -> Result<StructuralStrainRecovery, StructuralFieldRecoveryError> {
    immediate(recover_structural_strain_with_capacity::<
        std::convert::Infallible,
    >(
        summary,
        displacement,
        local_execution,
        retain_integration_points,
        None,
    ))
}

async fn recover_structural_strain_with_capacity<A>(
    summary: &AssemblySummary,
    displacement: &[f64],
    local_execution: LocalExecutionPlan,
    retain_integration_points: bool,
    owner: Option<&dyn AsyncCapacityOwner<A>>,
) -> Result<StructuralStrainRecovery, NumericalCapacityError<StructuralFieldRecoveryError, A>> {
    if !summary.structural_solid_recovery.is_empty() {
        let recovered = super::solid::recover_solid_fields_with_plan_and_capacity(
            &summary.structural_solid_recovery,
            displacement,
            local_execution,
            retain_integration_points,
            owner,
        )
        .await?;
        let basis = solid_recovery_basis(summary);
        return Ok(StructuralStrainRecovery {
            values: recovered.strain,
            stress: Some(recovered.stress),
            strain_energy_density: Some(recovered.strain_energy_density),
            integration_point_strain: recovered.integration_point_strain,
            integration_point_stress: recovered.integration_point_stress,
            element_count: summary.structural_solid_recovery.len().max(1),
            basis,
        });
    }

    let recovery_edges = structural_recovery_edges(summary);
    Ok(StructuralStrainRecovery {
        values: recover_edge_strain(displacement, &recovery_edges),
        stress: None,
        strain_energy_density: None,
        integration_point_strain: Vec::new(),
        integration_point_stress: Vec::new(),
        element_count: recovery_edges.len().max(1),
        basis: recovery_edges
            .first()
            .map(|edge| edge.basis.as_str())
            .unwrap_or(StructuralRecoveryBasis::OperatorConnectivity.as_str()),
    })
}

fn recover_edge_strain(
    displacement: &[f64],
    recovery_edges: &[StructuralRecoveryEdge],
) -> Vec<f64> {
    let element_count = recovery_edges.len().max(1);
    let mut strain = vec![0.0; element_count * TENSOR_COMPONENT_COUNT];
    for (element_index, edge) in recovery_edges.iter().enumerate() {
        let edge_strain = edge_strain_tensor(displacement, edge);
        let base = element_index * TENSOR_COMPONENT_COUNT;
        strain[base..base + TENSOR_COMPONENT_COUNT].copy_from_slice(&edge_strain);
    }
    strain
}

fn strain_component_coverage_ratio(
    displacement: &[f64],
    recovery_edges: &[StructuralRecoveryEdge],
) -> f64 {
    let expected_component_count = recovery_edges.len() * TENSOR_COMPONENT_COUNT;
    if expected_component_count == 0 {
        return 0.0;
    }
    let active_component_count = recovery_edges
        .iter()
        .map(|edge| {
            edge_strain_tensor(displacement, edge)
                .iter()
                .filter(|value| value.abs() > 0.0)
                .count()
        })
        .sum::<usize>();
    active_component_count as f64 / expected_component_count as f64
}

#[cfg(test)]
mod tests;
