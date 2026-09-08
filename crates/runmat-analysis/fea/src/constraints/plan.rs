//! Canonical, topology-resolved constraint plans.
//!
//! Authored constraints belong to `runmat-analysis-core`. This module owns the one-way
//! projection from that physical intent onto solver topology. Assembly consumes the resulting
//! plan; it must not reinterpret authored targets or coordinate frames.

use std::collections::BTreeSet;

use runmat_analysis_core::{
    AnalysisModel, CoordinateFrame, PrescribedValue, StructuralConstraintKind,
};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::{SolverMeshArtifact, StableDigest};
use serde::Serialize;

use super::contract::*;
use super::rank;
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
            let (mut targets, compiled_rows, enforcement) = match &constraint.kind {
                StructuralConstraintKind::Fixed { target } => {
                    let resolved = resolve_solid_target(mesh, target, &constraint.constraint_id)?;
                    let compiled_rows = resolved
                        .matched_node_identities
                        .iter()
                        .flat_map(|node_identity| {
                            CanonicalStructuralDof::TRANSLATIONS.map(|dof| {
                                new_row(&constraint.constraint_id, *node_identity, dof, 1.0, 0.0)
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    (
                        vec![resolved],
                        compiled_rows,
                        ConstraintEnforcement::DirectElimination,
                    )
                }
                StructuralConstraintKind::Prescribed(values) => {
                    if !values.rotations.is_empty() {
                        return Err(ConstraintPlanError::UnavailableRotationalDof {
                            constraint_id: constraint.constraint_id.clone(),
                        });
                    }
                    let prescribed = values
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
                    let resolved =
                        resolve_solid_target(mesh, &values.target, &constraint.constraint_id)?;
                    let compiled_rows = resolved
                        .matched_node_identities
                        .iter()
                        .flat_map(|node_identity| {
                            prescribed.iter().map(move |&(dof, value)| {
                                new_row(&constraint.constraint_id, *node_identity, dof, 1.0, value)
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    (
                        vec![resolved],
                        compiled_rows,
                        ConstraintEnforcement::DirectElimination,
                    )
                }
                StructuralConstraintKind::Equation(equation) => {
                    if equation.terms.is_empty() {
                        return Err(ConstraintPlanError::Invalid(format!(
                            "constraint `{}` has no equation terms",
                            constraint.constraint_id
                        )));
                    }
                    let mut targets = Vec::with_capacity(equation.terms.len());
                    let mut terms = Vec::new();
                    for term in &equation.terms {
                        if !term.coefficient.is_finite() || term.coefficient == 0.0 {
                            return Err(ConstraintPlanError::Invalid(format!(
                                "constraint `{}` contains a zero or non-finite equation coefficient",
                                constraint.constraint_id
                            )));
                        }
                        let resolved =
                            resolve_solid_target(mesh, &term.target, &constraint.constraint_id)?;
                        terms.extend(resolved.matched_node_identities.iter().map(
                            |node_identity| CanonicalStructuralConstraintTerm {
                                node_identity: *node_identity,
                                dof: term.dof.into(),
                                coefficient: term.coefficient,
                            },
                        ));
                        targets.push(resolved);
                    }
                    let right_hand_side = equation.right_hand_side.meters;
                    if !right_hand_side.is_finite() {
                        return Err(ConstraintPlanError::NonFiniteValue {
                            constraint_id: constraint.constraint_id.clone(),
                        });
                    }
                    let row = new_sparse_row(&constraint.constraint_id, terms, right_hand_side)?;
                    (targets, vec![row], ConstraintEnforcement::AffineReduction)
                }
                kind => {
                    return Err(ConstraintPlanError::UnsupportedKind {
                        constraint_id: constraint.constraint_id.clone(),
                        kind: structural_kind_name(kind),
                    });
                }
            };
            targets.sort_by_key(|target| target.selection_digest);
            targets.dedup_by_key(|target| target.selection_digest);
            let source_digest = source_digest(constraint)?;
            let first_row = rows.len();
            rows.extend(compiled_rows);
            groups.push(ConstraintGroupEvidence {
                constraint_id: constraint.constraint_id.clone(),
                source_digest,
                targets,
                enforcement,
                row_count: rows.len() - first_row,
            });
        }

        groups.sort_by(|left, right| left.constraint_id.cmp(&right.constraint_id));
        rows.sort_by(|left, right| row_key(left).cmp(&row_key(right)));
        let (active_row_count, redundant_row_count) = classify_rows(&mut rows)?;
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
            let group_rows = self
                .rows
                .iter()
                .filter(|row| row.constraint_id == group.constraint_id)
                .collect::<Vec<_>>();
            if group.constraint_id.is_empty()
                || group.targets.is_empty()
                || group.row_count != group_rows.len()
                || !group
                    .targets
                    .windows(2)
                    .all(|pair| pair[0].selection_digest < pair[1].selection_digest)
                || !enforcement_matches_rows(group.enforcement, &group_rows)
            {
                return Err(ConstraintPlanError::Invalid(
                    "group evidence differs from its canonical rows".into(),
                ));
            }
            for target in &group.targets {
                validate_resolved_target(&group.constraint_id, target)?;
            }
        }
        for row in &self.rows {
            if row.constraint_id.is_empty()
                || row.terms.is_empty()
                || !row.right_hand_side.is_finite()
                || row.identity != row_identity(row)?
            {
                return Err(ConstraintPlanError::Invalid(
                    "constraint row identity or numerical content is invalid".into(),
                ));
            }
            if row.terms.iter().any(|term| {
                term.node_identity == StableDigest::ZERO
                    || !term.coefficient.is_finite()
                    || term.coefficient == 0.0
            }) || !row
                .terms
                .windows(2)
                .all(|pair| term_key(&pair[0]) < term_key(&pair[1]))
            {
                return Err(ConstraintPlanError::Invalid(
                    "constraint row terms are invalid or not uniquely ordered".into(),
                ));
            }
        }
        let (active, redundant) = validate_row_classification(&self.rows)?;
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

fn enforcement_matches_rows(
    enforcement: ConstraintEnforcement,
    rows: &[&CanonicalStructuralConstraintRow],
) -> bool {
    match enforcement {
        ConstraintEnforcement::DirectElimination => {
            !rows.is_empty() && rows.iter().all(|row| row.terms.len() == 1)
        }
        ConstraintEnforcement::AffineReduction => rows.len() == 1 && rows[0].terms.len() > 1,
        ConstraintEnforcement::LagrangeMultiplier | ConstraintEnforcement::NullSpaceReduction => {
            false
        }
    }
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
    if let StructuralConstraintKind::Equation(equation) = &constraint.kind {
        let mut terms = equation.terms.clone();
        terms.sort_by(|left, right| {
            authored_target_key(&left.target)
                .cmp(&authored_target_key(&right.target))
                .then_with(|| left.dof.cmp(&right.dof))
                .then_with(|| left.coefficient.total_cmp(&right.coefficient))
        });
        canonical_digest(
            SOURCE_DOMAIN,
            &(
                &constraint.constraint_id,
                &constraint.frame,
                "equation",
                terms,
                equation.right_hand_side,
            ),
        )
    } else {
        canonical_digest(SOURCE_DOMAIN, constraint)
    }
}

fn authored_target_key(target: &runmat_analysis_core::RegionTarget) -> (&str, u8, u8, u8) {
    use runmat_analysis_core::{EntityDimension, ExpectedTopology, TargetCardinality};
    let dimension = match target.entity_dimension {
        EntityDimension::Point => 0,
        EntityDimension::Edge => 1,
        EntityDimension::Face => 2,
        EntityDimension::Volume => 3,
    };
    let topology = match target.expected_topology {
        ExpectedTopology::GeometryEntity => 0,
        ExpectedTopology::MeshNode => 1,
        ExpectedTopology::MeshElement => 2,
        ExpectedTopology::BoundaryFace => 3,
        ExpectedTopology::Cell => 4,
    };
    let cardinality = match target.cardinality {
        TargetCardinality::One => 0,
        TargetCardinality::OneOrMore => 1,
    };
    (&target.region_id, dimension, topology, cardinality)
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
    coefficient: f64,
    right_hand_side: f64,
) -> Result<CanonicalStructuralConstraintRow, ConstraintPlanError> {
    new_sparse_row(
        constraint_id,
        vec![CanonicalStructuralConstraintTerm {
            node_identity,
            dof,
            coefficient,
        }],
        right_hand_side,
    )
}

fn new_sparse_row(
    constraint_id: &str,
    mut terms: Vec<CanonicalStructuralConstraintTerm>,
    right_hand_side: f64,
) -> Result<CanonicalStructuralConstraintRow, ConstraintPlanError> {
    terms.sort_by_key(term_key);
    let mut combined: Vec<CanonicalStructuralConstraintTerm> = Vec::with_capacity(terms.len());
    for term in terms {
        if let Some(previous) = combined
            .last_mut()
            .filter(|previous| term_key(previous) == term_key(&term))
        {
            previous.coefficient += term.coefficient;
        } else {
            combined.push(term);
        }
    }
    combined.retain(|term| term.coefficient != 0.0);
    if combined.is_empty() || !right_hand_side.is_finite() {
        return Err(ConstraintPlanError::Invalid(format!(
            "constraint `{constraint_id}` compiles to an empty or non-finite equation"
        )));
    }
    let mut row = CanonicalStructuralConstraintRow {
        identity: Digest::from_bytes([0; 32]),
        constraint_id: constraint_id.into(),
        terms: combined,
        right_hand_side,
        reaction_recovery: ReactionRecovery::ConstraintMultiplier,
    };
    if row.terms.len() == 1 {
        row.reaction_recovery = ReactionRecovery::UnconstrainedEquilibriumResidual;
    }
    row.identity = row_identity(&row)?;
    Ok(row)
}

pub(super) fn row_identity(
    row: &CanonicalStructuralConstraintRow,
) -> Result<Digest, ConstraintPlanError> {
    canonical_digest(
        ROW_DOMAIN,
        &(
            &row.constraint_id,
            &row.terms,
            row.right_hand_side,
            row.reaction_recovery,
        ),
    )
}

fn term_key(term: &CanonicalStructuralConstraintTerm) -> (StableDigest, CanonicalStructuralDof) {
    (term.node_identity, term.dof)
}

fn row_key(
    row: &CanonicalStructuralConstraintRow,
) -> (&str, StableDigest, CanonicalStructuralDof, usize) {
    let first = &row.terms[0];
    (
        row.constraint_id.as_str(),
        first.node_identity,
        first.dof,
        row.terms.len(),
    )
}

fn classify_rows(
    rows: &mut [CanonicalStructuralConstraintRow],
) -> Result<(usize, usize), ConstraintPlanError> {
    let result = rank::classify(rows)?;
    for row in rows {
        row.identity = row_identity(row)?;
    }
    Ok(result)
}

fn validate_row_classification(
    rows: &[CanonicalStructuralConstraintRow],
) -> Result<(usize, usize), ConstraintPlanError> {
    let expected = rows
        .iter()
        .map(|row| {
            let mut row = row.clone();
            row.reaction_recovery = if row.terms.len() == 1 {
                ReactionRecovery::UnconstrainedEquilibriumResidual
            } else {
                ReactionRecovery::ConstraintMultiplier
            };
            Ok(row)
        })
        .collect::<Result<Vec<_>, ConstraintPlanError>>()?;
    let mut expected = expected;
    let result = rank::classify(&mut expected)?;
    for (actual, expected) in rows.iter().zip(expected) {
        if actual.reaction_recovery != expected.reaction_recovery {
            return Err(ConstraintPlanError::Invalid(
                "constraint row rank classification is inconsistent".into(),
            ));
        }
    }
    Ok(result)
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
