//! Canonical, topology-resolved constraint plans.
//!
//! Authored constraints belong to `runmat-analysis-core`. This module owns the one-way
//! projection from that physical intent onto solver topology. Assembly consumes the resulting
//! plan; it must not reinterpret authored targets or coordinate frames.

use std::cmp::Ordering;

use runmat_analysis_core::{AnalysisModel, StructuralConstraintKind};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::{SolverMeshArtifact, StableDigest};
use serde::Serialize;

use super::contract::*;
use super::enforcement;
use super::row::{
    compare_rows, residual_quantity, row_identity, term_key, validate_row_classification,
};
use super::target::validate_resolved_target;
use crate::physics::structural::topology::CanonicalStructuralTopologyIdentity;

mod authored_profile;
mod profile;
mod validation;
pub use authored_profile::{
    constraint_authored_workspace_profile, ConstraintAuthoredWorkspaceProfile,
};
pub use profile::{
    constraint_plan_finalization_workspace_profile, ConstraintPlanFinalizationWorkspaceProfile,
};

pub(super) const PLAN_PREFIX: &[u8] = b"runmat-analysis-structural-constraint-plan/v1\0";
const PROFILE_LIMITS: CanonicalLimits =
    CanonicalLimits::new(usize::MAX, usize::MAX, usize::MAX, 64);
const PLAN_DOMAIN: &str = "structural-constraint-plan";
const SOURCE_DOMAIN: &str = "structural-constraint-source";
pub(super) const PLAN_LIMITS: CanonicalLimits =
    CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 65_536, 64);

impl CanonicalStructuralConstraintPlan {
    pub fn compile_for_model(
        model: &AnalysisModel,
        mesh: Option<&SolverMeshArtifact>,
    ) -> Result<Self, ConstraintPlanError> {
        super::compilation::compile(model, mesh)
    }

    pub fn compile(
        model: &AnalysisModel,
        mesh: &SolverMeshArtifact,
    ) -> Result<Self, ConstraintPlanError> {
        Self::compile_for_model(model, Some(mesh))
    }

    pub fn validate(&self) -> Result<(), ConstraintPlanError> {
        validation::validate(self)
    }

    pub fn validate_against(
        &self,
        model: &AnalysisModel,
        mesh: &SolverMeshArtifact,
    ) -> Result<(), ConstraintPlanError> {
        self.validate()?;
        if self.topology.mesh_digest() != Some(mesh.canonical_digest) {
            return Err(ConstraintPlanError::TopologyMismatch);
        }
        let authored = structural_constraints(model)?;
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

    pub fn validate_against_explicit(
        &self,
        model: &AnalysisModel,
    ) -> Result<(), ConstraintPlanError> {
        self.validate()?;
        let expected_topology = model
            .structural
            .as_ref()
            .ok_or_else(|| {
                ConstraintPlanError::Invalid(
                    "explicit constraint validation requires structural topology".into(),
                )
            })
            .and_then(|structural| {
                crate::physics::structural::topology::explicit_topology_digest(structural)
                    .map_err(|error| ConstraintPlanError::Invalid(error.to_string()))
            })?;
        if self.topology
            != (CanonicalStructuralTopologyIdentity::Explicit {
                structural_topology_digest: expected_topology,
            })
        {
            return Err(ConstraintPlanError::TopologyMismatch);
        }
        let authored = structural_constraints(model)?;
        if self.authored_constraints_digest != authored_constraints_digest(&authored)? {
            return Err(ConstraintPlanError::AuthoredConstraintsMismatch);
        }
        let expected = Self::compile_for_model(model, None)?;
        if *self != expected {
            return Err(ConstraintPlanError::Invalid(
                "constraint plan differs from canonical explicit compilation".into(),
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
) -> Result<Vec<&runmat_analysis_core::StructuralConstraint>, ConstraintPlanError> {
    let count = model
        .constraints
        .iter()
        .filter(|c| c.structural().is_some())
        .count();
    let mut authored = super::workspace::reserve(count)?;
    for constraint in &model.constraints {
        super::workspace::check_cancelled()?;
        if let Some(c) = constraint.structural() {
            authored.push(c);
        }
    }
    Ok(authored)
}

pub(super) fn authored_constraints_digest(
    constraints: &[&runmat_analysis_core::StructuralConstraint],
) -> Result<Digest, ConstraintPlanError> {
    let mut sources = super::workspace::reserve(constraints.len())?;
    for constraint in constraints {
        super::workspace::check_cancelled()?;
        sources.push(source_digest(constraint)?);
    }
    sources.sort_unstable();
    canonical_digest(SOURCE_DOMAIN, &sources)
}

pub(super) fn source_digest(
    constraint: &runmat_analysis_core::StructuralConstraint,
) -> Result<Digest, ConstraintPlanError> {
    if let StructuralConstraintKind::Equation(equation) = &constraint.kind {
        let mut terms = super::workspace::reserve(equation.terms.len())?;
        terms.extend(equation.terms.iter());
        super::workspace::sort_stable_by(&mut terms, |left, right| {
            authored_target_key(&left.target)
                .cmp(&authored_target_key(&right.target))
                .then_with(|| left.dof.cmp(&right.dof))
                .then_with(|| left.coefficient.total_cmp(&right.coefficient))
        })?;
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

pub(super) fn canonical_value_workspace_bytes<T: Serialize + ?Sized>(
    value: &T,
) -> Result<u64, ConstraintPlanError> {
    super::workspace::check_cancelled()?;
    runmat_canonical_codec::serialized_materialization_resident_bytes_estimate(
        value,
        PROFILE_LIMITS,
    )
    .map_err(|e| {
        if crate::is_fea_cancelled() {
            ConstraintPlanError::Cancelled
        } else {
            ConstraintPlanError::Encoding(e.to_string())
        }
    })
}

#[cfg(test)]
mod source_tests;
