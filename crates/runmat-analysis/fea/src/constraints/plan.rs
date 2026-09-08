//! Canonical, topology-resolved constraint plans.
//!
//! Authored constraints belong to `runmat-analysis-core`. This module owns the one-way
//! projection from that physical intent onto solver topology. Assembly consumes the resulting
//! plan; it must not reinterpret authored targets or coordinate frames.

use std::cmp::Ordering;
use std::collections::BTreeSet;

use runmat_analysis_core::{AnalysisModel, StructuralConstraintKind};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::{SolverMeshArtifact, StableDigest};
use serde::Serialize;

use super::contract::*;
use super::row::{
    compare_rows, enforcement_matches_rows, row_identity, term_key, validate_row_classification,
};
use super::target::validate_resolved_target;

const PLAN_PREFIX: &[u8] = b"runmat-analysis-structural-constraint-plan/v1\0";
const PLAN_DOMAIN: &str = "structural-constraint-plan";
const SOURCE_DOMAIN: &str = "structural-constraint-source";
const PLAN_LIMITS: CanonicalLimits = CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 65_536, 64);

impl CanonicalStructuralConstraintPlan {
    pub fn compile(
        model: &AnalysisModel,
        mesh: &SolverMeshArtifact,
    ) -> Result<Self, ConstraintPlanError> {
        super::compile::compile(model, mesh)
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
            .reference_points
            .windows(2)
            .all(|pair| pair[0].identity < pair[1].identity)
            || self.reference_points.iter().any(|point| {
                point.reference_point_id.trim().is_empty()
                    || point.position_meters.iter().any(|value| !value.is_finite())
                    || match reference_point_identity(
                        &point.reference_point_id,
                        point.position_meters,
                    ) {
                        Ok(identity) => identity != point.identity,
                        Err(_) => true,
                    }
            })
        {
            return Err(ConstraintPlanError::Invalid(
                "canonical reference-point evidence is invalid".into(),
            ));
        }
        let reference_point_identities = self
            .reference_points
            .iter()
            .map(|point| point.identity)
            .collect::<BTreeSet<_>>();
        if !self
            .groups
            .windows(2)
            .all(|pair| pair[0].constraint_id < pair[1].constraint_id)
            || !self
                .rows
                .windows(2)
                .all(|pair| compare_rows(&pair[0], &pair[1]) == Ordering::Less)
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
                || !group
                    .reference_point_identities
                    .windows(2)
                    .all(|pair| pair[0] < pair[1])
                || group
                    .reference_point_identities
                    .iter()
                    .any(|identity| !reference_point_identities.contains(identity))
                || match &group.coupling {
                    Some(coupling) => {
                        group.reference_point_identities.as_slice()
                            != [coupling.reference_point_identity]
                            || coupling.target_weights.is_empty()
                            || !coupling.characteristic_length_meters.is_finite()
                            || coupling.characteristic_length_meters < 0.0
                            || (coupling.kind == ConstraintCouplingKind::Distributing
                                && coupling.characteristic_length_meters <= 1.0e-12)
                            || !coupling
                                .target_weights
                                .windows(2)
                                .all(|pair| pair[0].node_identity < pair[1].node_identity)
                            || coupling.target_weights.iter().any(|weight| {
                                weight.node_identity == StableDigest::ZERO
                                    || !weight.weight.is_finite()
                                    || weight.weight <= 0.0
                                    || !group.targets.iter().any(|target| {
                                        target
                                            .matched_node_identities
                                            .contains(&weight.node_identity)
                                    })
                            })
                            || (coupling
                                .target_weights
                                .iter()
                                .map(|weight| weight.weight)
                                .sum::<f64>()
                                - 1.0)
                                .abs()
                                > 1.0e-10
                    }
                    None => !group.reference_point_identities.is_empty(),
                }
                || !group
                    .pairings
                    .windows(2)
                    .all(|pair| pair[0].secondary_node_identity < pair[1].secondary_node_identity)
                || group.pairings.iter().any(|pairing| {
                    pairing.secondary_node_identity == StableDigest::ZERO
                        || !pairing.geometric_error_meters.is_finite()
                        || pairing.geometric_error_meters < 0.0
                        || pairing.primary_weights.is_empty()
                        || !pairing.primary_weights.windows(2).all(|pair| {
                            pair[0].primary_node_identity < pair[1].primary_node_identity
                        })
                        || pairing.primary_weights.iter().any(|weight| {
                            weight.primary_node_identity == StableDigest::ZERO
                                || !weight.weight.is_finite()
                                || weight.weight == 0.0
                        })
                        || (pairing
                            .primary_weights
                            .iter()
                            .map(|weight| weight.weight)
                            .sum::<f64>()
                            - 1.0)
                            .abs()
                            > 1.0e-10
                })
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
                let invalid_owner = match term.owner {
                    CanonicalStructuralDofOwner::MeshNode(identity) => {
                        identity == StableDigest::ZERO
                    }
                    CanonicalStructuralDofOwner::ReferencePoint(identity) => {
                        !reference_point_identities.contains(&identity)
                    }
                };
                invalid_owner || !term.coefficient.is_finite() || term.coefficient == 0.0
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

pub(super) fn structural_constraints(
    model: &AnalysisModel,
) -> Vec<&runmat_analysis_core::StructuralConstraint> {
    model
        .constraints
        .iter()
        .filter_map(|constraint| constraint.structural())
        .collect()
}

pub(super) fn authored_constraints_digest(
    constraints: &[&runmat_analysis_core::StructuralConstraint],
) -> Result<Digest, ConstraintPlanError> {
    let mut sources = constraints
        .iter()
        .map(|constraint| source_digest(constraint))
        .collect::<Result<Vec<_>, _>>()?;
    sources.sort_unstable();
    canonical_digest(SOURCE_DOMAIN, &sources)
}

pub(super) fn source_digest(
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

pub(super) fn canonical_digest<T: Serialize>(
    domain: &'static str,
    value: &T,
) -> Result<Digest, ConstraintPlanError> {
    encode_contract(PLAN_PREFIX, domain, value, PLAN_LIMITS)
        .map(|bytes| Digest::sha256(&bytes))
        .map_err(|error| ConstraintPlanError::Encoding(error.to_string()))
}

pub(super) fn reference_point_identity(
    reference_point_id: &str,
    position_meters: [f64; 3],
) -> Result<Digest, ConstraintPlanError> {
    canonical_digest(
        "structural-reference-point",
        &(reference_point_id, position_meters),
    )
}
