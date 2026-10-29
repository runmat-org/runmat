//! Immutable numerical output. Integrity validation does not certify equilibrium.
use super::*;
use crate::{
    pipeline::nonlinear::NonlinearBlockOutput,
    solve::nonlinear::{NonlinearLoadContinuation, NonlinearSolveResult},
};
use runmat_canonical_codec::decode_contract;
use serde::{Deserialize, Serialize};

const PREFIX: &[u8] = b"runmat-nonlinear-solid-block-artifact/v1\0";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NonlinearSolidBlockArtifact {
    pub identity: Digest,
    pub request: NonlinearSolidBlockRequest,
    pub result: NonlinearSolveResult,
    pub checkpoint: NonlinearLoadContinuation,
    pub complete: bool,
}

impl NonlinearSolidBlockArtifact {
    pub fn source_result_identity(
        &self,
    ) -> AnalysisExecutionResult<super::super::SolverResultIdentity> {
        self.validate()?;
        super::super::SolverResultIdentity::new(self.request.workload, self.identity)
    }
    pub(super) fn new(
        request: NonlinearSolidBlockRequest,
        block: NonlinearBlockOutput,
    ) -> AnalysisExecutionResult<Self> {
        let mut value = Self {
            identity: Digest::sha256([]),
            request,
            result: block.result,
            checkpoint: block.checkpoint,
            complete: block.complete,
        };
        value.identity = value.derive_identity()?;
        value.validate()?;
        Ok(value)
    }
    fn derive_identity(&self) -> AnalysisExecutionResult<Digest> {
        digest(
            "nonlinear-solid-block-artifact",
            &(&self.request, &self.result, &self.checkpoint, self.complete),
        )
    }
    pub fn last_output_ordinal(&self) -> AnalysisExecutionResult<u64> {
        let count = u64::try_from(self.result.load_factors.len()).map_err(invalid)?;
        self.request
            .first_output_ordinal
            .checked_add(
                count
                    .checked_sub(1)
                    .ok_or_else(|| invalid("empty nonlinear block"))?,
            )
            .ok_or_else(|| invalid("nonlinear output ordinal overflow"))
    }
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        self.request.validate()?;
        self.checkpoint.validate_integrity().map_err(invalid)?;
        let r = &self.result;
        let n = r.load_factors.len();
        let element_count = r
            .accepted_state
            .point_history()
            .map(|(element, _, _)| element)
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        let tensor_count = element_count
            .checked_mul(6)
            .ok_or_else(|| invalid("nonlinear tensor inventory overflow"))?;
        let dofs = r.accepted_state.displacement().len();
        if element_count == 0
            || r.initial_displacement.len() != dofs
            || r.initial_equivalent_plastic_strain.len() != element_count
            || [&r.displacement_snapshots, &r.internal_force_snapshots]
                .into_iter()
                .flatten()
                .any(|v| v.len() != dofs)
            || [
                &r.stress_snapshots,
                &r.total_strain_snapshots,
                &r.plastic_strain_snapshots,
            ]
            .into_iter()
            .flatten()
            .any(|v| v.len() != tensor_count)
            || [
                &r.von_mises_snapshots,
                &r.equivalent_plastic_strain_snapshots,
            ]
            .into_iter()
            .flatten()
            .any(|v| v.len() != element_count)
        {
            return Err(invalid(
                "original nonlinear field shape differs from point/displacement inventory",
            ));
        }
        let lengths = [
            r.displacement_snapshots.len(),
            r.internal_force_snapshots.len(),
            r.stress_snapshots.len(),
            r.total_strain_snapshots.len(),
            r.von_mises_snapshots.len(),
            r.plastic_strain_snapshots.len(),
            r.equivalent_plastic_strain_snapshots.len(),
            r.residual_norms.len(),
            r.increment_norms.len(),
            r.relative_increment_norms.len(),
            r.iteration_counts.len(),
        ];
        if n == 0
            || r.residual_norms
                .iter()
                .any(|x| *x > self.checkpoint.maximum_relative_residual())
            || r.relative_increment_norms
                .iter()
                .any(|x| *x > self.checkpoint.maximum_relative_increment())
            || n > self.request.accepted_block_limit
            || lengths.iter().any(|len| *len != n)
            // Contact history is complete for a contact run and absent otherwise.
            || r.contact_pressure_snapshots.len() != r.contact_gap_snapshots.len()
            || (!r.contact_pressure_snapshots.is_empty() && r.contact_pressure_snapshots.len() != n)
            || r.total_increments != n
            || r.converged_increments != n
            || r.accepted_state != *self.checkpoint.accepted_state()
            || self.complete != self.checkpoint.complete()
            || u64::try_from(self.checkpoint.work().accepted_increments()).map_err(invalid)?
                != self.last_output_ordinal()?
            || !r.initial_load_factor.is_finite()
            || r.load_factors.iter().any(|x| !x.is_finite())
            || r.load_factors[0] <= r.initial_load_factor
            || r.load_factors.windows(2).any(|w| w[1] <= w[0])
            || r.load_factors[n - 1].to_bits() != r.accepted_state.load_factor().to_bits()
            || r.displacement_snapshots[n - 1] != r.accepted_state.displacement()
            || r.initial_displacement
                .iter()
                .chain(&r.initial_equivalent_plastic_strain)
                .chain(&r.initial_contact_pressure)
                .chain(&r.initial_contact_gap)
                .any(|v| !v.is_finite())
            || [
                &r.displacement_snapshots,
                &r.internal_force_snapshots,
                &r.stress_snapshots,
                &r.total_strain_snapshots,
                &r.von_mises_snapshots,
                &r.plastic_strain_snapshots,
                &r.equivalent_plastic_strain_snapshots,
                &r.contact_pressure_snapshots,
                &r.contact_gap_snapshots,
            ]
            .into_iter()
            .flatten()
            .flatten()
            .any(|v| !v.is_finite())
            || r.residual_norms
                .iter()
                .chain(&r.increment_norms)
                .chain(&r.relative_increment_norms)
                .any(|v| !v.is_finite() || *v < 0.0)
            || self.identity != self.derive_identity()?
        {
            return Err(invalid(
                "malformed original nonlinear block inventory or identity",
            ));
        }
        Ok(())
    }
    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            PREFIX,
            "nonlinear-solid-block-artifact",
            self,
            LIMITS,
        )?)
    }
    /// Expected identity must come from the authenticated immutable input closure.
    pub fn from_canonical_bytes(
        bytes: &[u8],
        expected_identity: Digest,
    ) -> AnalysisExecutionResult<Self> {
        let value: Self = decode_contract(PREFIX, "nonlinear-solid-block-artifact", bytes, LIMITS)?;
        value.validate()?;
        if value.identity != expected_identity {
            return Err(invalid(
                "nonlinear block differs from authenticated identity",
            ));
        }
        Ok(value)
    }
}
