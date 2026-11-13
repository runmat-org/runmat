//! Shared authoritative solid load and essential-constraint lowering.

use super::super::*;

/// The time-invariant spatial authority of solid load lowering: the structural node space and
/// the analysis projection of one admitted mesh. Derived once per assembly (and once per
/// prepared transient basis), never per load or per time sample.
pub(in crate::assembly) struct SolidLoadDomain {
    pub topology: crate::physics::structural::topology::node_space::CanonicalStructuralNodeSpace,
    /// A derivation failure is reported against the first load that needs the projection,
    /// exactly as when each load derived it.
    pub projection: Result<crate::topology::AnalysisProjection, String>,
}

impl SolidLoadDomain {
    pub(in crate::assembly) fn body_load_work_upper_bound(
        &self,
        excitations: &RealizedExcitations,
    ) -> Result<runmat_simulation_usage::OperationCount, LinearAssemblyError> {
        let Ok(projection) = &self.projection else {
            return Ok(runmat_simulation_usage::OperationCount::ZERO);
        };
        excitations.iter().try_fold(
            runmat_simulation_usage::OperationCount::ZERO,
            |total, load| {
                let work =
                    crate::assembly::solver_solid_boundary::body_load_quadrature_work_upper_bound(
                        projection, load,
                    )
                    .map_err(|cause| {
                        LinearAssemblyError::Structural(StructuralAssemblyError::Accounting(cause))
                    })?;
                total.checked_add(work).map_err(|cause| {
                    LinearAssemblyError::Structural(StructuralAssemblyError::Accounting(
                        cause.into(),
                    ))
                })
            },
        )
    }
    pub(in crate::assembly) fn projection_work_upper_bound(
        mesh: &SolverMeshArtifact,
    ) -> Result<runmat_simulation_usage::OperationCount, LinearAssemblyError> {
        crate::topology::AnalysisProjection::derivation_work_upper_bound(mesh)
            .map_err(projection_error)
    }

    pub(in crate::assembly) fn derive(
        mesh: &SolverMeshArtifact,
    ) -> Result<Self, LinearAssemblyError> {
        let invalid = |error: String| {
            LinearAssemblyError::Structural(StructuralAssemblyError::InvalidLoadPlan(error))
        };
        Ok(Self {
            topology: crate::physics::structural::topology::node_space::CanonicalStructuralNodeSpace::derive_solid(mesh)
                .map_err(|error| invalid(error.to_string()))?,
            projection: match crate::topology::AnalysisProjection::derive(mesh) {
                Ok(projection) => Ok(projection),
                Err(error @ crate::topology::AnalysisProjectionError::Accounting(_))
                | Err(error @ crate::topology::AnalysisProjectionError::Cancelled) => {
                    return Err(projection_error(error));
                }
                Err(error) => Err(error.to_string()),
            },
        })
    }
}

fn projection_error(error: crate::topology::AnalysisProjectionError) -> LinearAssemblyError {
    let structural = match error {
        crate::topology::AnalysisProjectionError::Accounting(cause) => {
            StructuralAssemblyError::Accounting(cause)
        }
        crate::topology::AnalysisProjectionError::Cancelled => StructuralAssemblyError::Cancelled,
        other => StructuralAssemblyError::InvalidLoadPlan(other.to_string()),
    };
    LinearAssemblyError::Structural(structural)
}

pub(in crate::assembly) struct SolidForcing {
    pub rhs: Vec<f64>,
    pub physical_load: Vec<f64>,
    pub constrained: Vec<bool>,
    pub constraint_matrix: Option<ResolvedConstraintMatrix>,
    pub affine_constraint_reduction: Option<AffineConstraintReduction>,
    pub moment_projection: Vec<MomentProjectionSummary>,
    pub projection_evidence: Vec<crate::excitation::SpatialProjectionEvidence>,
    pub plan_identity: Option<runmat_execution::Digest>,
}

pub(in crate::assembly) fn lower(
    excitations: &RealizedExcitations,
    mesh: &SolverMeshArtifact,
    constraint_plan: &CanonicalStructuralConstraintPlan,
    structural_dof_layout: &StructuralDofLayout,
    domain: &SolidLoadDomain,
) -> Result<SolidForcing, LinearAssemblyError> {
    let mut rhs = vec![0.0; structural_dof_layout.total_dof_count()];
    let mut constrained = vec![false; rhs.len()];
    let mut constraint_matrix = None;
    let mut affine_constraint_reduction = None;
    let topology = &domain.topology;
    let source_projection = compile_solid_load_plan(
        excitations,
        mesh,
        &domain.projection,
        constraint_plan,
        topology,
        structural_dof_layout,
    )
    .map_err(LinearAssemblyError::from)?;
    apply_structural_load_plan(
        &source_projection.plan,
        &topology,
        &constraint_plan.reference_points,
        structural_dof_layout,
        &mut rhs,
    )
    .map_err(|error| {
        LinearAssemblyError::Structural(StructuralAssemblyError::InvalidLoadPlan(error.to_string()))
    })?;
    let structural_moment_projection = source_projection.moment_projection;
    let spatial_projection_evidence = source_projection
        .plan
        .entries
        .iter()
        .flat_map(|entry| entry.evidence.iter().cloned())
        .collect();
    let structural_load_plan_identity = Some(source_projection.plan.plan_digest);
    let physical_load = rhs.clone();
    constrained.fill(false);
    if constraint_plan.enforcement != crate::constraints::ConstraintEnforcement::DirectElimination {
        let matrix = resolve_solid_constraint_matrix(constraint_plan, mesh, structural_dof_layout)
            .map_err(LinearAssemblyError::ConstraintAlgebra)?;
        if constraint_plan.enforcement
            != crate::constraints::ConstraintEnforcement::LagrangeMultiplier
        {
            affine_constraint_reduction = Some(
                matrix
                    .affine_reduction()
                    .map_err(LinearAssemblyError::ConstraintAlgebra)?,
            );
        }
        constraint_matrix = Some(matrix);
    } else {
        let mut direct_constraints = DirectConstraintAssignments::new(structural_dof_layout);
        lower_solid_direct_constraints(constraint_plan, mesh, &mut direct_constraints)
            .map_err(LinearAssemblyError::Structural)?;
        constrained = direct_constraints
            .apply(&mut rhs)
            .map_err(LinearAssemblyError::Structural)?;
    }
    Ok(SolidForcing {
        rhs,
        physical_load,
        constrained,
        constraint_matrix,
        affine_constraint_reduction,
        moment_projection: structural_moment_projection,
        projection_evidence: spatial_projection_evidence,
        plan_identity: structural_load_plan_identity,
    })
}
