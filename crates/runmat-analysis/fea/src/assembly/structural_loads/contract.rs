use std::collections::BTreeSet;

use runmat_analysis_core::{
    CoordinateFrame, ExcitationDistribution, ExcitationKinematics, PhysicalQuantityKind,
};
#[cfg(test)]
use runmat_canonical_codec::decode_contract;
use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    constraints::{CanonicalStructuralDof, CanonicalStructuralDofOwner},
    excitation::{ProjectionSample, RealizedExcitationTarget, SpatialProjectionEvidence},
    physics::structural::topology::CanonicalStructuralTopologyIdentity,
};

const PLAN_PREFIX: &[u8] = b"runmat-fea-structural-load-plan/v1\0";
const PLAN_DOMAIN: &str = "fea-structural-load-plan";
const PLAN_LIMITS: CanonicalLimits = CanonicalLimits::platform_content(1_000_000, 64);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CanonicalStructuralLoadTerm {
    pub owner: CanonicalStructuralDofOwner,
    pub dof: CanonicalStructuralDof,
    pub value: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CanonicalStructuralLoadEntry {
    pub excitation_id: String,
    pub case_ids: Vec<String>,
    pub signal_id: String,
    pub sampled_value_digest: Digest,
    pub sample: ProjectionSample,
    pub target: RealizedExcitationTarget,
    pub quantity: PhysicalQuantityKind,
    pub frame: CoordinateFrame,
    pub distribution: ExcitationDistribution,
    pub kinematics: ExcitationKinematics,
    pub terms: Vec<CanonicalStructuralLoadTerm>,
    pub evidence: Vec<SpatialProjectionEvidence>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CanonicalStructuralLoadPlan {
    pub schema_version: u16,
    pub topology: CanonicalStructuralTopologyIdentity,
    pub entries: Vec<CanonicalStructuralLoadEntry>,
    pub plan_digest: Digest,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub(crate) enum StructuralLoadPlanError {
    #[error("structural load numerical accounting failed: {0}")]
    Accounting(runmat_simulation_usage::LedgerError),
    #[error("structural load projection was cancelled")]
    Cancelled,
    #[error("structural excitation `{excitation_id}` uses unsupported {policy} `{value}`")]
    UnsupportedPolicy {
        excitation_id: String,
        policy: &'static str,
        value: String,
    },
    #[error(
        "structural excitation `{excitation_id}` does not resolve to structural topology: {reason}"
    )]
    UnresolvedTarget {
        excitation_id: String,
        reason: String,
    },
    #[error("structural excitation `{excitation_id}` projection is invalid: {reason}")]
    InvalidProjection {
        excitation_id: String,
        reason: String,
    },
    #[error("structural load plan topology does not match assembly topology")]
    TopologyMismatch,
    #[error("structural load plan is malformed: {0}")]
    Invalid(String),
    #[error("structural load plan identity encoding failed: {0}")]
    Encoding(String),
}

impl CanonicalStructuralLoadPlan {
    pub(crate) fn new(
        topology: CanonicalStructuralTopologyIdentity,
        mut entries: Vec<CanonicalStructuralLoadEntry>,
    ) -> Result<Self, StructuralLoadPlanError> {
        entries.sort_by(|left, right| left.excitation_id.cmp(&right.excitation_id));
        if entries
            .windows(2)
            .any(|pair| pair[0].excitation_id == pair[1].excitation_id)
        {
            return Err(StructuralLoadPlanError::Invalid(
                "excitation ids must be unique".into(),
            ));
        }
        for entry in &mut entries {
            entry.case_ids.sort();
            normalize_terms(&mut entry.terms)?;
            validate_entry(entry)?;
        }
        let plan_digest = digest(&(1_u16, topology, &entries))?;
        let plan = Self {
            schema_version: 1,
            topology,
            entries,
            plan_digest,
        };
        plan.validate()?;
        Ok(plan)
    }

    pub(crate) fn validate(&self) -> Result<(), StructuralLoadPlanError> {
        if self.schema_version != 1 || !self.topology.has_valid_identity() {
            return Err(StructuralLoadPlanError::Invalid(
                "schema version or topology identity is invalid".into(),
            ));
        }
        if self
            .entries
            .windows(2)
            .any(|pair| pair[0].excitation_id >= pair[1].excitation_id)
        {
            return Err(StructuralLoadPlanError::Invalid(
                "entries are not in unique canonical excitation order".into(),
            ));
        }
        for entry in &self.entries {
            validate_entry(entry)?;
            if !terms_are_canonical(&entry.terms) {
                return Err(StructuralLoadPlanError::Invalid(format!(
                    "excitation `{}` terms are not canonical",
                    entry.excitation_id
                )));
            }
        }
        if self.plan_digest != digest(&(self.schema_version, self.topology, &self.entries))? {
            return Err(StructuralLoadPlanError::Invalid(
                "plan digest does not match its contents".into(),
            ));
        }
        Ok(())
    }

    #[cfg(test)]
    fn canonical_bytes(&self) -> Result<Vec<u8>, StructuralLoadPlanError> {
        self.validate()?;
        encode_contract(PLAN_PREFIX, PLAN_DOMAIN, self, PLAN_LIMITS)
            .map_err(|error| StructuralLoadPlanError::Encoding(error.to_string()))
    }

    #[cfg(test)]
    fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, StructuralLoadPlanError> {
        let plan: Self = decode_contract(PLAN_PREFIX, PLAN_DOMAIN, bytes, PLAN_LIMITS)
            .map_err(|error| StructuralLoadPlanError::Encoding(error.to_string()))?;
        plan.validate()?;
        Ok(plan)
    }
}

fn validate_entry(entry: &CanonicalStructuralLoadEntry) -> Result<(), StructuralLoadPlanError> {
    if entry.excitation_id.trim().is_empty()
        || entry.signal_id.trim().is_empty()
        || entry.case_ids.is_empty()
        || entry.case_ids.iter().any(|id| id.trim().is_empty())
        || entry.case_ids.iter().collect::<BTreeSet<_>>().len() != entry.case_ids.len()
        || entry
            .terms
            .iter()
            .any(|term| !term.value.is_finite() || term.value == 0.0)
        || entry.sampled_value_digest == Digest::sha256([])
    {
        return Err(StructuralLoadPlanError::Invalid(format!(
            "excitation `{}` metadata or terms are invalid",
            entry.excitation_id
        )));
    }
    if entry.evidence.is_empty() {
        return Err(StructuralLoadPlanError::InvalidProjection {
            excitation_id: entry.excitation_id.clone(),
            reason: "projection evidence is empty".into(),
        });
    }
    for evidence in &entry.evidence {
        evidence
            .validate()
            .map_err(|error| StructuralLoadPlanError::InvalidProjection {
                excitation_id: entry.excitation_id.clone(),
                reason: error.to_string(),
            })?;
        if evidence.excitation_id != entry.excitation_id
            || evidence.signal_id != entry.signal_id
            || evidence.quantity != entry.quantity
            || evidence.sample != entry.sample
            || !evidence.accepted
        {
            return Err(StructuralLoadPlanError::InvalidProjection {
                excitation_id: entry.excitation_id.clone(),
                reason: "projection evidence is not bound to the entry".into(),
            });
        }
    }
    Ok(())
}

fn normalize_terms(
    terms: &mut Vec<CanonicalStructuralLoadTerm>,
) -> Result<(), StructuralLoadPlanError> {
    terms.sort_by_key(|term| (term.owner, term.dof));
    let mut normalized: Vec<CanonicalStructuralLoadTerm> = Vec::with_capacity(terms.len());
    for term in terms.drain(..) {
        if !term.value.is_finite() {
            return Err(StructuralLoadPlanError::Invalid(
                "load term is non-finite".into(),
            ));
        }
        if let Some(previous) = normalized
            .last_mut()
            .filter(|previous| previous.owner == term.owner && previous.dof == term.dof)
        {
            previous.value += term.value;
        } else {
            normalized.push(term);
        }
    }
    normalized.retain(|term| term.value != 0.0);
    *terms = normalized;
    Ok(())
}

fn terms_are_canonical(terms: &[CanonicalStructuralLoadTerm]) -> bool {
    terms
        .iter()
        .all(|term| term.value.is_finite() && term.value != 0.0)
        && terms
            .windows(2)
            .all(|pair| (pair[0].owner, pair[0].dof) < (pair[1].owner, pair[1].dof))
}

fn digest(value: &impl Serialize) -> Result<Digest, StructuralLoadPlanError> {
    encode_contract(PLAN_PREFIX, PLAN_DOMAIN, value, PLAN_LIMITS)
        .map(Digest::sha256)
        .map_err(|error| StructuralLoadPlanError::Encoding(error.to_string()))
}

#[cfg(test)]
mod tests {
    use runmat_analysis_core::{
        ExcitationDistribution, ExcitationKinematics, PhysicalQuantityKind, ProjectionRequirement,
    };
    use runmat_meshing_core::StableDigest;

    use crate::{
        constraints::{CanonicalStructuralDof, CanonicalStructuralDofOwner},
        excitation::{
            ProjectionSample, ProjectionVector3, SpatialIntegrationRule, SpatialProjectionBalance,
            SpatialProjectionEvidence,
        },
        physics::structural::topology::CanonicalStructuralTopologyIdentity,
    };

    use super::*;

    fn fixture() -> CanonicalStructuralLoadPlan {
        let topology = CanonicalStructuralTopologyIdentity::Solid {
            mesh_digest: StableDigest::from_bytes([7; 32]),
        };
        let evidence = SpatialProjectionEvidence {
            excitation_id: "tip_force".into(),
            signal_id: "tip_force_signal".into(),
            quantity: PhysicalQuantityKind::Force,
            sample: ProjectionSample::Static,
            analysis_projection_digest: topology.discretization_digest(),
            integration_rule: SpatialIntegrationRule::UniformResultant,
            resolved_entity_count: 1,
            resolved_node_count: 1,
            reference_point_m: [0.0; 3],
            balance: SpatialProjectionBalance::VectorResultant {
                requested_resultant: ProjectionVector3 {
                    real: [2.0, 0.0, 0.0],
                    imaginary: [0.0; 3],
                },
                projected_resultant: ProjectionVector3 {
                    real: [2.0, 0.0, 0.0],
                    imaginary: [0.0; 3],
                },
                requested_moment: ProjectionVector3 {
                    real: [0.0; 3],
                    imaginary: [0.0; 3],
                },
                projected_moment: ProjectionVector3 {
                    real: [0.0; 3],
                    imaginary: [0.0; 3],
                },
                resultant_scale: 2.0,
                moment_scale: 0.0,
                resultant_relative_error: 0.0,
                moment_relative_error: 0.0,
            },
            requirements: ProjectionRequirement {
                preserve_resultant: true,
                preserve_moment: true,
                relative_tolerance: 1.0e-12,
            },
            accepted: true,
        };
        CanonicalStructuralLoadPlan::new(
            topology,
            vec![CanonicalStructuralLoadEntry {
                excitation_id: "tip_force".into(),
                case_ids: vec!["static".into()],
                signal_id: "tip_force_signal".into(),
                sampled_value_digest: Digest::sha256(b"sample"),
                sample: ProjectionSample::Static,
                target: RealizedExcitationTarget::WholeModel,
                quantity: PhysicalQuantityKind::Force,
                frame: CoordinateFrame::Global,
                distribution: ExcitationDistribution::PreserveResultant,
                kinematics: ExcitationKinematics::FixedFrame,
                terms: vec![CanonicalStructuralLoadTerm {
                    owner: CanonicalStructuralDofOwner::StructuralNode(StableDigest::from_bytes(
                        [9; 32],
                    )),
                    dof: CanonicalStructuralDof::Translation(
                        runmat_analysis_core::StructuralTranslationDof::Ux,
                    ),
                    value: 2.0,
                }],
                evidence: vec![evidence],
            }],
        )
        .unwrap()
    }

    #[test]
    fn canonical_round_trip_preserves_the_complete_load_contract() {
        let plan = fixture();
        let decoded =
            CanonicalStructuralLoadPlan::from_canonical_bytes(&plan.canonical_bytes().unwrap())
                .unwrap();

        assert_eq!(decoded, plan);
    }

    #[test]
    fn content_tampering_invalidates_the_plan_identity() {
        let mut plan = fixture();
        plan.entries[0].terms[0].value = 3.0;

        assert!(matches!(
            plan.validate(),
            Err(StructuralLoadPlanError::Invalid(message))
                if message.contains("digest")
        ));
    }
}
