//! One borrowed compilation pipeline for solid, explicit and conformal mixed constraints.
//! Family lowering owns equations; this owner accumulates and finalizes the canonical plan once.

use runmat_analysis_core::{AnalysisModel, BoundaryTransferImplementation, StructuralConstraint};
use runmat_meshing_core::SolverMeshArtifact;

use super::{
    compile::canonical_reference_points,
    contract::*,
    enforcement,
    plan::{authored_constraints_digest, source_digest, structural_constraints},
    row,
};

mod admission;
mod lowering_profile;
mod storage;
use super::workspace::{add, slots};
use admission::CompilationAdmission;
pub use admission::{ConstraintCompilationPhase, ConstraintCompilationWorkspace};

/// Original compilation and its construction inventory stay paired and immutable.
/// The inventory is requested storage, not a numerical certificate or cold artifact.
///
/// Callers cannot attach a foreign plan to another construction's inventory:
/// ```compile_fail
/// use runmat_analysis_fea::constraints::{ConstraintCompilation, CanonicalStructuralConstraintPlan};
/// fn substitute(compiled: &mut ConstraintCompilation, foreign: CanonicalStructuralConstraintPlan) {
///     compiled.plan = foreign;
/// }
/// ```
#[derive(Debug)]
pub struct ConstraintCompilation<'a> {
    model: &'a AnalysisModel,
    mesh: Option<&'a SolverMeshArtifact>,
    plan: CanonicalStructuralConstraintPlan,
    workspace: ConstraintCompilationWorkspace,
}
impl ConstraintCompilation<'_> {
    /// Construction scope only; no memory address enters a physical or resource identity.
    pub(crate) fn require_original_inputs(
        &self,
        model: &AnalysisModel,
        mesh: Option<&SolverMeshArtifact>,
    ) -> Result<(), ConstraintPlanError> {
        let same_mesh = match (self.mesh, mesh) {
            (None, None) => true,
            (Some(a), Some(b)) => std::ptr::eq(a, b),
            _ => false,
        };
        if !std::ptr::eq(self.model, model) || !same_mesh {
            return Err(ConstraintPlanError::Invalid(
                "compilation inventory belongs to different borrowed inputs".into(),
            ));
        }
        Ok(())
    }
    pub fn plan(&self) -> &CanonicalStructuralConstraintPlan {
        &self.plan
    }
    pub fn workspace(&self) -> &ConstraintCompilationWorkspace {
        &self.workspace
    }
    /// Consume the carrier when only the original physical plan is needed.
    pub fn into_plan(self) -> CanonicalStructuralConstraintPlan {
        self.plan
    }
}
mod preparation;
mod topology;
pub use preparation::{
    constraint_preparation_workspace_profile, ConstraintPreparationWorkspaceProfile,
};
mod workspace;
use topology::CompilationTopology;
use workspace::AccumulationWorkspace;
pub use workspace::{
    constraint_group_construction_workspace_profile, ConstraintGroupConstructionWorkspaceProfile,
};

pub(super) struct CompiledConstraint {
    pub(super) targets: Vec<ResolvedConstraintTarget>,
    pub(super) rows: Vec<CanonicalStructuralConstraintRow>,
    pub(super) pairings: Vec<ConstraintPairingEvidence>,
    pub(super) coupling: Option<ConstraintCouplingEvidence>,
}

impl CompiledConstraint {
    pub(super) fn rows(
        targets: Vec<ResolvedConstraintTarget>,
        rows: Vec<CanonicalStructuralConstraintRow>,
    ) -> Self {
        Self {
            targets,
            rows,
            pairings: Vec::new(),
            coupling: None,
        }
    }

    pub(super) fn coupling(
        target: ResolvedConstraintTarget,
        rows: Vec<CanonicalStructuralConstraintRow>,
        coupling: ConstraintCouplingEvidence,
    ) -> Result<Self, ConstraintPlanError> {
        Ok(Self {
            targets: super::lowering::targets([target])?,
            rows,
            pairings: Vec::new(),
            coupling: Some(coupling),
        })
    }
}

pub(super) fn compile<'a>(
    model: &'a AnalysisModel,
    mesh: Option<&'a SolverMeshArtifact>,
    maximum_workspace_bytes: u64,
) -> Result<ConstraintCompilation<'a>, ConstraintPlanError> {
    super::workspace::check_cancelled()?;
    let preparation = constraint_preparation_workspace_profile(model, mesh)?;
    let count = super::plan::constraint_authored_workspace_profile(model)?.constraint_count;
    let mut accumulation = AccumulationWorkspace::new(count)?;
    let mut admission = CompilationAdmission::new(maximum_workspace_bytes);
    admission.require(
        ConstraintCompilationPhase::Preparation,
        accumulation.retained_bytes(),
        preparation.total_requested_bytes,
    )?;
    // Retain conservative owner inventories for authored/topology/reference state.
    // Target preparation is added separately at each actual occurrence's lifetime.
    let baseline = add(
        add(preparation.authored_bytes, preparation.topology_bytes)?,
        preparation.reference_point_bytes,
    )?;
    let authored = structural_constraints(model)?;
    validate_unique_ids(&authored)?;
    let topology = CompilationTopology::derive(model, mesh)?;
    let reference_points = canonical_reference_points(&authored)?;
    let authored_constraints_digest = authored_constraints_digest(&authored)?;
    let mut groups = super::workspace::reserve(authored.len())?;
    let mut rows = Vec::new();
    let mut enforcement_mode = ConstraintEnforcement::DirectElimination;

    for constraint in authored {
        runmat_analysis_core::validate_coordinate_frame(&constraint.frame).map_err(|_| {
            ConstraintPlanError::DegenerateFrame {
                constraint_id: constraint.constraint_id.clone(),
            }
        })?;
        let retained = add(
            add(baseline, accumulation.retained_bytes())?,
            slots::<CanonicalStructuralConstraintRow>(rows.capacity())?,
        )?;
        let (compiled, implementation) =
            topology.lower_admitted(model, constraint, &mut admission, retained)?;
        let lowered_bytes = storage::lowered_bytes(&compiled)?;
        let CompiledConstraint {
            mut targets,
            rows: compiled_rows,
            pairings,
            coupling,
        } = compiled;
        let group_workspace = constraint_group_construction_workspace_profile(
            constraint,
            targets.len(),
            coupling.is_some(),
        )?;
        admission.require(
            ConstraintCompilationPhase::Group,
            add(retained, lowered_bytes)?,
            group_workspace.total_requested_bytes,
        )?;
        super::workspace::sort_stable_by(&mut targets, |a, b| {
            a.selection_digest.cmp(&b.selection_digest)
        })?;
        targets.dedup_by_key(|target| target.selection_digest);
        enforcement_mode = enforcement::combine(
            enforcement_mode,
            enforcement::select(&constraint.kind, &compiled_rows),
        );
        let row_count = compiled_rows.len();
        let group = group(
            constraint,
            implementation,
            targets,
            pairings,
            coupling,
            row_count,
        )?;
        let accumulated = accumulation.advance(
            &group,
            &compiled_rows,
            compiled_rows.capacity(),
            rows.len(),
            rows.capacity(),
        )?;
        admission.require(
            ConstraintCompilationPhase::Accumulation,
            baseline,
            accumulated,
        )?;
        rows.try_reserve_exact(row_count).map_err(allocation)?;
        rows.extend(compiled_rows);
        groups.push(group);
    }

    let retained = add(
        add(baseline, accumulation.retained_bytes())?,
        slots::<CanonicalStructuralConstraintRow>(rows.capacity())?,
    )?;
    // The profile's connectivity graph allocates too; admit its count-only envelope first.
    let graph_bytes = super::rank::constraint_rank_graph_workspace_bytes(&rows)?;
    admission.require(
        ConstraintCompilationPhase::RankProfile,
        retained,
        graph_bytes,
    )?;
    let row_count = rows.len();
    let mut plan = CanonicalStructuralConstraintPlan {
        schema_version: 1,
        topology: topology.identity(),
        authored_constraints_digest,
        enforcement: enforcement_mode,
        reference_points,
        evidence: ConstraintPlanEvidence {
            authored_constraint_count: groups.len(),
            active_row_count: row_count,
            redundant_row_count: row_count,
            conflicting_row_count: 0,
            rank: row_count,
        },
        groups,
        rows,
    };
    // Preliminary evidence only bounds inventory; it is replaced before validation/return.
    let finalization = super::plan::constraint_plan_finalization_workspace_profile(&plan)?;
    admission.require(
        ConstraintCompilationPhase::Finalization,
        retained,
        finalization.total_requested_bytes,
    )?;
    super::workspace::sort_stable_by(&mut plan.groups, |a, b| {
        a.constraint_id.cmp(&b.constraint_id)
    })?;
    super::workspace::sort_stable_by(&mut plan.rows, row::compare_rows)?;
    let (active, redundant) = row::classify(&mut plan.rows)?;
    plan.evidence.active_row_count = active;
    plan.evidence.redundant_row_count = redundant;
    plan.evidence.rank = active;
    plan.validate()?;
    Ok(ConstraintCompilation {
        model,
        mesh,
        plan,
        workspace: admission.finish(),
    })
}

fn group(
    constraint: &StructuralConstraint,
    implementation: BoundaryTransferImplementation,
    targets: Vec<ResolvedConstraintTarget>,
    pairings: Vec<ConstraintPairingEvidence>,
    coupling: Option<ConstraintCouplingEvidence>,
    row_count: usize,
) -> Result<ConstraintGroupEvidence, ConstraintPlanError> {
    let mut reference_point_identities =
        super::workspace::reserve(usize::from(coupling.is_some()))?;
    reference_point_identities.extend(coupling.iter().map(|value| value.reference_point_identity));
    Ok(ConstraintGroupEvidence {
        constraint_id: super::lowering::owned_id(&constraint.constraint_id)?,
        source_digest: source_digest(constraint)?,
        implementation,
        targets,
        reference_point_identities,
        coupling,
        pairings,
        row_count,
    })
}

fn validate_unique_ids(constraints: &[&StructuralConstraint]) -> Result<(), ConstraintPlanError> {
    let mut ids = super::workspace::reserve(constraints.len())?;
    ids.extend(
        constraints
            .iter()
            .enumerate()
            .map(|(ordinal, c)| (c.constraint_id.as_str(), ordinal)),
    );
    ids.sort_unstable();
    if let Some(pair) = ids
        .windows(2)
        .filter(|pair| pair[0].0 == pair[1].0)
        .min_by_key(|pair| pair[1].1)
    {
        return Err(ConstraintPlanError::DuplicateConstraintId {
            constraint_id: pair[1].0.into(),
        });
    }

    Ok(())
}

fn allocation(_: std::collections::TryReserveError) -> ConstraintPlanError {
    ConstraintPlanError::Invalid("constraint plan accumulation exceeds addressable storage".into())
}
