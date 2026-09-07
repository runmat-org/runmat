//! Canonical, topology-resolved constraint plans.
//!
//! Authored constraints belong to `runmat-analysis-core`. This module owns the one-way
//! projection from that physical intent onto solver topology. Assembly consumes the resulting
//! plan; it must not reinterpret authored targets or coordinate frames.

use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_core::{
    AnalysisModel, CoordinateFrame, PrescribedValue, StructuralConstraintKind,
};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::{SolverMeshArtifact, StableDigest};
use serde::Serialize;

use super::contract::*;
use super::target::{resolve_solid_target, validate_resolved_target};

const PLAN_PREFIX: &[u8] = b"runmat-analysis-structural-constraint-plan/v1\0";
const PLAN_DOMAIN: &str = "structural-constraint-plan";
const ROW_DOMAIN: &str = "structural-constraint-row";
const SOURCE_DOMAIN: &str = "structural-constraint-source";
const PLAN_LIMITS: CanonicalLimits = CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 65_536, 64);

impl CanonicalStructuralConstraintPlan {
    pub fn compile(
        model: &AnalysisModel,
        mesh: &SolverMeshArtifact,
    ) -> Result<Self, ConstraintPlanError> {
        mesh.validate()
            .map_err(|error| ConstraintPlanError::Invalid(error.to_string()))?;
        let authored = structural_constraints(model);
        let mut ids = BTreeSet::new();
        for constraint in &authored {
            if !ids.insert(constraint.constraint_id.as_str()) {
                return Err(ConstraintPlanError::DuplicateConstraintId {
                    constraint_id: constraint.constraint_id.clone(),
                });
            }
        }
        let authored_constraints_digest = authored_constraints_digest(&authored)?;
        let mut groups = Vec::with_capacity(authored.len());
        let mut rows = Vec::new();

        for constraint in authored {
            if constraint.frame != CoordinateFrame::Global {
                return Err(ConstraintPlanError::UnsupportedFrame {
                    constraint_id: constraint.constraint_id.clone(),
                    frame: constraint.frame.context_label(),
                });
            }
            let (target, prescribed) = match &constraint.kind {
                StructuralConstraintKind::Fixed { target } => (target, PrescribedRows::Fixed),
                StructuralConstraintKind::Prescribed(values) => {
                    let target = &values.target;
                    if !values.rotations.is_empty() {
                        return Err(ConstraintPlanError::UnavailableRotationalDof {
                            constraint_id: constraint.constraint_id.clone(),
                        });
                    }
                    let prescribed_rows = values
                        .translations
                        .iter()
                        .map(|value| {
                            let constant = study_constant_displacement(
                                &value.value,
                                model,
                                &constraint.constraint_id,
                            )?;
                            Ok((CanonicalStructuralDof::from(value.dof), constant))
                        })
                        .collect::<Result<Vec<_>, ConstraintPlanError>>()?;
                    (target, PrescribedRows::Values(prescribed_rows))
                }
                kind => {
                    return Err(ConstraintPlanError::UnsupportedKind {
                        constraint_id: constraint.constraint_id.clone(),
                        kind: structural_kind_name(kind),
                    });
                }
            };
            let resolved = resolve_solid_target(mesh, target, &constraint.constraint_id)?;
            let source_digest = source_digest(constraint)?;
            let first_row = rows.len();
            for node_identity in &resolved.matched_node_identities {
                match &prescribed {
                    PrescribedRows::Fixed => {
                        for dof in CanonicalStructuralDof::TRANSLATIONS {
                            rows.push(new_row(
                                &constraint.constraint_id,
                                *node_identity,
                                dof,
                                0.0,
                            )?);
                        }
                    }
                    PrescribedRows::Values(values) => {
                        for &(dof, value) in values {
                            rows.push(new_row(
                                &constraint.constraint_id,
                                *node_identity,
                                dof,
                                value,
                            )?);
                        }
                    }
                }
            }
            groups.push(ConstraintGroupEvidence {
                constraint_id: constraint.constraint_id.clone(),
                source_digest,
                target: resolved,
                enforcement: ConstraintEnforcement::DirectElimination,
                row_count: rows.len() - first_row,
            });
        }

        groups.sort_by(|left, right| left.constraint_id.cmp(&right.constraint_id));
        rows.sort_by_key(|row| (row.node_identity, row.dof, row.constraint_id.clone()));
        let (active_row_count, redundant_row_count) = analyze_rows(&rows)?;
        let plan = Self {
            schema_version: 1,
            topology_digest: mesh.canonical_digest,
            geometry_source_digest: mesh.geometry.source_digest,
            geometry_revision: mesh.geometry.geometry_revision,
            persistent_mapping_version: mesh.geometry.persistent_mapping_version,
            authored_constraints_digest,
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

    pub fn validate(&self) -> Result<(), ConstraintPlanError> {
        if self.schema_version != 1
            || self.topology_digest == StableDigest::ZERO
            || self.geometry_source_digest == StableDigest::ZERO
            || self.geometry_revision == 0
            || self.persistent_mapping_version == 0
            || self.evidence.authored_constraint_count != self.groups.len()
            || self.evidence.active_row_count + self.evidence.redundant_row_count != self.rows.len()
            || self.evidence.rank != self.evidence.active_row_count
            || self.evidence.conflicting_row_count != 0
        {
            return Err(ConstraintPlanError::Invalid(
                "schema, topology, geometry, or evidence is inconsistent".into(),
            ));
        }
        if !self
            .groups
            .windows(2)
            .all(|pair| pair[0].constraint_id < pair[1].constraint_id)
            || !self
                .rows
                .windows(2)
                .all(|pair| row_key(&pair[0]) < row_key(&pair[1]))
        {
            return Err(ConstraintPlanError::Invalid(
                "groups and rows must be uniquely ordered".into(),
            ));
        }
        for group in &self.groups {
            if group.constraint_id.is_empty()
                || group.enforcement != ConstraintEnforcement::DirectElimination
                || group.row_count
                    != self
                        .rows
                        .iter()
                        .filter(|row| row.constraint_id == group.constraint_id)
                        .count()
            {
                return Err(ConstraintPlanError::Invalid(
                    "group evidence differs from its canonical rows".into(),
                ));
            }
            validate_resolved_target(&group.constraint_id, &group.target)?;
        }
        for row in &self.rows {
            if row.constraint_id.is_empty()
                || row.node_identity == StableDigest::ZERO
                || row.coefficient != 1.0
                || !row.right_hand_side.is_finite()
                || row.reaction_recovery != ReactionRecovery::UnconstrainedEquilibriumResidual
                || row.identity != row_identity(row)?
            {
                return Err(ConstraintPlanError::Invalid(
                    "constraint row identity or numerical content is invalid".into(),
                ));
            }
        }
        let (active, redundant) = analyze_rows(&self.rows)?;
        if active != self.evidence.active_row_count
            || redundant != self.evidence.redundant_row_count
        {
            return Err(ConstraintPlanError::Invalid(
                "constraint rank evidence differs from its canonical rows".into(),
            ));
        }
        Ok(())
    }

    pub fn validate_against(
        &self,
        model: &AnalysisModel,
        mesh: &SolverMeshArtifact,
    ) -> Result<(), ConstraintPlanError> {
        self.validate()?;
        if self.topology_digest != mesh.canonical_digest
            || self.geometry_source_digest != mesh.geometry.source_digest
            || self.geometry_revision != mesh.geometry.geometry_revision
            || self.persistent_mapping_version != mesh.geometry.persistent_mapping_version
        {
            return Err(ConstraintPlanError::TopologyMismatch);
        }
        let authored = structural_constraints(model);
        if self.authored_constraints_digest != authored_constraints_digest(&authored)? {
            return Err(ConstraintPlanError::AuthoredConstraintsMismatch);
        }
        let expected = Self::compile(model, mesh)?;
        if *self != expected {
            return Err(ConstraintPlanError::Invalid(
                "constraint plan differs from canonical compilation".into(),
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ConstraintPlanError> {
        self.validate()?;
        encode_contract(PLAN_PREFIX, PLAN_DOMAIN, self, PLAN_LIMITS)
            .map_err(|error| ConstraintPlanError::Encoding(error.to_string()))
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, ConstraintPlanError> {
        let plan = decode_contract(PLAN_PREFIX, PLAN_DOMAIN, bytes, PLAN_LIMITS)
            .map_err(|error| ConstraintPlanError::Encoding(error.to_string()))?;
        Self::validate(&plan)?;
        Ok(plan)
    }

    pub fn content_digest(&self) -> Result<Digest, ConstraintPlanError> {
        Ok(Digest::sha256(&self.canonical_bytes()?))
    }
}

enum PrescribedRows {
    Fixed,
    Values(Vec<(CanonicalStructuralDof, f64)>),
}

fn structural_constraints(
    model: &AnalysisModel,
) -> Vec<&runmat_analysis_core::StructuralConstraint> {
    model
        .constraints
        .iter()
        .filter_map(|constraint| constraint.structural())
        .collect()
}

fn authored_constraints_digest(
    constraints: &[&runmat_analysis_core::StructuralConstraint],
) -> Result<Digest, ConstraintPlanError> {
    let mut sources = constraints
        .iter()
        .map(|constraint| source_digest(constraint))
        .collect::<Result<Vec<_>, _>>()?;
    sources.sort_unstable();
    canonical_digest(SOURCE_DOMAIN, &sources)
}

fn source_digest(
    constraint: &runmat_analysis_core::StructuralConstraint,
) -> Result<Digest, ConstraintPlanError> {
    canonical_digest(SOURCE_DOMAIN, constraint)
}

fn study_constant_displacement(
    value: &PrescribedValue<runmat_analysis_core::Displacement>,
    model: &AnalysisModel,
    constraint_id: &str,
) -> Result<f64, ConstraintPlanError> {
    let value = value
        .study_constant(&model.signals)
        .ok_or_else(|| ConstraintPlanError::ScheduledValue {
            constraint_id: constraint_id.into(),
        })?
        .meters;
    if !value.is_finite() {
        return Err(ConstraintPlanError::NonFiniteValue {
            constraint_id: constraint_id.into(),
        });
    }
    Ok(value)
}

fn new_row(
    constraint_id: &str,
    node_identity: StableDigest,
    dof: CanonicalStructuralDof,
    right_hand_side: f64,
) -> Result<CanonicalStructuralConstraintRow, ConstraintPlanError> {
    let mut row = CanonicalStructuralConstraintRow {
        identity: Digest::from_bytes([0; 32]),
        constraint_id: constraint_id.into(),
        node_identity,
        dof,
        coefficient: 1.0,
        right_hand_side,
        reaction_recovery: ReactionRecovery::UnconstrainedEquilibriumResidual,
    };
    row.identity = row_identity(&row)?;
    Ok(row)
}

fn row_identity(row: &CanonicalStructuralConstraintRow) -> Result<Digest, ConstraintPlanError> {
    canonical_digest(
        ROW_DOMAIN,
        &(
            &row.constraint_id,
            row.node_identity,
            row.dof,
            row.coefficient,
            row.right_hand_side,
            row.reaction_recovery,
        ),
    )
}

fn row_key(row: &CanonicalStructuralConstraintRow) -> (StableDigest, CanonicalStructuralDof, &str) {
    (row.node_identity, row.dof, row.constraint_id.as_str())
}

fn analyze_rows(
    rows: &[CanonicalStructuralConstraintRow],
) -> Result<(usize, usize), ConstraintPlanError> {
    let mut owners = BTreeMap::<_, (&str, f64)>::new();
    let mut redundant = 0;
    for row in rows {
        let key = (row.node_identity, row.dof);
        if let Some((first, value)) = owners.get(&key).copied() {
            if value != row.right_hand_side {
                return Err(ConstraintPlanError::ConflictingRow {
                    first_constraint_id: first.into(),
                    second_constraint_id: row.constraint_id.clone(),
                });
            }
            redundant += 1;
        } else {
            owners.insert(key, (row.constraint_id.as_str(), row.right_hand_side));
        }
    }
    Ok((owners.len(), redundant))
}

pub(super) fn canonical_digest<T: Serialize>(
    domain: &'static str,
    value: &T,
) -> Result<Digest, ConstraintPlanError> {
    encode_contract(PLAN_PREFIX, domain, value, PLAN_LIMITS)
        .map(|bytes| Digest::sha256(&bytes))
        .map_err(|error| ConstraintPlanError::Encoding(error.to_string()))
}

fn structural_kind_name(kind: &StructuralConstraintKind) -> &'static str {
    match kind {
        StructuralConstraintKind::Fixed { .. } => "fixed",
        StructuralConstraintKind::Prescribed(_) => "prescribed",
        StructuralConstraintKind::NormalDisplacement { .. } => "normal_displacement",
        StructuralConstraintKind::Symmetry { .. } => "symmetry",
        StructuralConstraintKind::Antisymmetry { .. } => "antisymmetry",
        StructuralConstraintKind::RigidCoupling(_) => "rigid_coupling",
        StructuralConstraintKind::DistributingCoupling(_) => "distributing_coupling",
        StructuralConstraintKind::Tie(_) => "tie",
        StructuralConstraintKind::Periodic(_) => "periodic",
        StructuralConstraintKind::Equation(_) => "equation",
    }
}
