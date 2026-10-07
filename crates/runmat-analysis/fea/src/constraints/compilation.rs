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

mod topology;
use topology::CompilationTopology;

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
    ) -> Self {
        Self {
            targets: vec![target],
            rows,
            pairings: Vec::new(),
            coupling: Some(coupling),
        }
    }
}

pub(super) fn compile(
    model: &AnalysisModel,
    mesh: Option<&SolverMeshArtifact>,
) -> Result<CanonicalStructuralConstraintPlan, ConstraintPlanError> {
    super::workspace::check_cancelled()?;
    let _authored_profile = super::plan::constraint_authored_workspace_profile(model)?;
    let authored = structural_constraints(model)?;
    validate_unique_ids(&authored)?;
    let topology = CompilationTopology::derive(model, mesh)?;
    let reference_points = canonical_reference_points(&authored)?;
    let authored_constraints_digest = authored_constraints_digest(&authored)?;
    let mut groups = Vec::new();
    groups
        .try_reserve_exact(authored.len())
        .map_err(allocation)?;
    let mut rows = Vec::new();
    let mut enforcement_mode = ConstraintEnforcement::DirectElimination;

    for constraint in authored {
        runmat_analysis_core::validate_coordinate_frame(&constraint.frame).map_err(|_| {
            ConstraintPlanError::DegenerateFrame {
                constraint_id: constraint.constraint_id.clone(),
            }
        })?;
        let (compiled, implementation) = topology.lower(model, constraint)?;
        let CompiledConstraint {
            mut targets,
            rows: compiled_rows,
            pairings,
            coupling,
        } = compiled;
        super::workspace::sort_stable_by(&mut targets, |a, b| {
            a.selection_digest.cmp(&b.selection_digest)
        })?;
        targets.dedup_by_key(|target| target.selection_digest);
        enforcement_mode = enforcement::combine(
            enforcement_mode,
            enforcement::select(&constraint.kind, &compiled_rows),
        );
        let row_count = compiled_rows.len();
        // This fallible accumulation is not a pre-expansion workspace admission claim.
        // Per-constraint target/row construction still needs its owner-side profile.
        rows.try_reserve_exact(row_count).map_err(allocation)?;
        rows.extend(compiled_rows);
        groups.push(group(
            constraint,
            implementation,
            targets,
            pairings,
            coupling,
            row_count,
        )?);
    }

    super::workspace::sort_stable_by(&mut groups, |a, b| a.constraint_id.cmp(&b.constraint_id))?;
    super::workspace::sort_stable_by(&mut rows, row::compare_rows)?;
    let (active_row_count, redundant_row_count) = row::classify(&mut rows)?;
    let plan = CanonicalStructuralConstraintPlan {
        schema_version: 1,
        topology: topology.identity(),
        authored_constraints_digest,
        enforcement: enforcement_mode,
        reference_points,
        evidence: ConstraintPlanEvidence {
            authored_constraint_count: groups.len(),
            active_row_count,
            redundant_row_count,
            conflicting_row_count: 0,
            rank: active_row_count,
        },
        groups,
        rows,
    };
    plan.validate()?;
    Ok(plan)
}

fn group(
    constraint: &StructuralConstraint,
    implementation: BoundaryTransferImplementation,
    targets: Vec<ResolvedConstraintTarget>,
    pairings: Vec<ConstraintPairingEvidence>,
    coupling: Option<ConstraintCouplingEvidence>,
    row_count: usize,
) -> Result<ConstraintGroupEvidence, ConstraintPlanError> {
    Ok(ConstraintGroupEvidence {
        constraint_id: constraint.constraint_id.clone(),
        source_digest: source_digest(constraint)?,
        implementation,
        targets,
        reference_point_identities: coupling
            .iter()
            .map(|value| value.reference_point_identity)
            .collect(),
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
