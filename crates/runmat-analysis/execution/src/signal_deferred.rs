//! Frozen callable intent is not an executable, topology-bound materialization.
use std::collections::BTreeSet;

use runmat_analysis_core::{GeneratorArgument, PhysicalSignal, SignalShape};
use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::{Digest, ExecutableUnitEnvelope, ProgramRevision};
use runmat_execution_artifact::BundleCallable;
use runmat_meshing_core::SolverMeshArtifact;
use runmat_types::CapabilityRequirement;
use serde::{Deserialize, Serialize};

use crate::{
    plan_signal_generator_chunks, prepare_signal_materialization, AdmittedSignalGenerator,
    AnalysisBridgeError, AnalysisBridgeResult, PreparedSignalMaterialization,
    SignalMaterializationExpectation, SignalTopologyResolution,
};

/// A named generated field admitted before meshing. It contains no chunks or mesh authority.
/// Imported values must match the externally expected frozen intent, not merely self-validate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenDeferredSignal {
    authored: PhysicalSignal,
    admitted: AdmittedSignalGenerator,
    executable: ExecutableUnitEnvelope,
    available_capabilities: BTreeSet<CapabilityRequirement>,
    maximum_inline_result_bytes: usize,
}

impl FrozenDeferredSignal {
    pub fn prepare(
        authored: PhysicalSignal,
        revision: ProgramRevision,
        callable: BundleCallable,
        executable: ExecutableUnitEnvelope,
        available_capabilities: BTreeSet<CapabilityRequirement>,
        maximum_inline_result_bytes: usize,
    ) -> AnalysisBridgeResult<Self> {
        let admitted = crate::callable::executable::admit_named_signal_generator_from_executable(
            &authored,
            revision,
            callable,
            &executable,
            &available_capabilities,
        )?;
        let frozen = Self {
            authored,
            admitted,
            executable,
            available_capabilities,
            maximum_inline_result_bytes,
        };
        frozen.validate()?;
        Ok(frozen)
    }

    pub fn descriptor(
        &self,
    ) -> AnalysisBridgeResult<runmat_execution_artifact::ProgramExecutionDescriptor> {
        self.validate()?;
        crate::prepare_generator_descriptor(&self.admitted, self.executable.clone())
    }

    pub fn validate(&self) -> AnalysisBridgeResult<()> {
        if !matches!(
            self.authored.shape(),
            SignalShape::DeferredTopologyField { .. }
        ) {
            return Err(invalid(
                "frozen deferred input requires unresolved authored topology",
            ));
        }
        let generator = self
            .authored
            .generator()
            .ok_or_else(|| invalid("deferred input has no generator"))?;
        if generator
            .invocation
            .arguments
            .iter()
            .any(|argument| matches!(argument, GeneratorArgument::Artifact(_)))
        {
            return Err(invalid(
                "deferred artifact arguments require an admitted immutable input closure",
            ));
        }
        let limit =
            usize::try_from(runmat_execution::value::ValueLimits::default().max_inline_bytes)
                .map_err(|_| invalid("inline result bound exceeds this host"))?;
        if self.maximum_inline_result_bytes <= crate::signal_materialization::RESULT_ENVELOPE_BYTES
            || self.maximum_inline_result_bytes > limit
        {
            return Err(invalid(
                "deferred inline result bound is outside the portable envelope",
            ));
        }
        // Compare against the original frozen binding before recovering facts
        // from the candidate envelope; never let re-admission replace it.
        crate::prepare_generator_descriptor(&self.admitted, self.executable.clone())?;
        let rederived = crate::callable::executable::admit_named_signal_generator_from_executable(
            &self.authored,
            self.admitted.program_revision.clone(),
            self.admitted.callable.clone(),
            &self.executable,
            &self.available_capabilities,
        )?;
        if rederived != self.admitted {
            return Err(AnalysisBridgeError::Identity(
                "deferred admission differs from frozen executable facts",
            ));
        }
        Ok(())
    }

    pub fn identity(&self) -> AnalysisBridgeResult<Digest> {
        self.validate()?;
        Ok(Digest::sha256(encode_contract(
            b"runmat-analysis-deferred-signal\0",
            "deferred-signal-intent",
            &(
                &self.authored,
                &self.admitted,
                self.executable.cache_key()?,
                &self.available_capabilities,
                self.maximum_inline_result_bytes,
            ),
            CanonicalLimits::new(8 * 1024 * 1024, 65_536, 4096, 32),
        )?))
    }

    /// A transported contract must agree with the independently retained frozen authority.
    pub fn validate_for(&self, expected: &Self) -> AnalysisBridgeResult<()> {
        self.validate()?;
        expected.validate()?;
        if self != expected {
            return Err(AnalysisBridgeError::Identity(
                "transported deferred input differs from expected frozen intent",
            ));
        }
        Ok(())
    }

    pub fn authored(&self) -> &PhysicalSignal {
        &self.authored
    }
    pub fn admitted(&self) -> &AdmittedSignalGenerator {
        &self.admitted
    }

    /// Only this transition creates executable chunks, using authenticated domain cardinality.
    pub fn resolve(
        &self,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisBridgeResult<ResolvedDeferredSignalMaterialization> {
        self.validate()?;
        let resolution = SignalTopologyResolution::derive(&self.authored, mesh)?;
        let admitted = self.resolved_admission(&resolution)?;
        let plan = plan_signal_generator_chunks(
            resolution.resolved(),
            &admitted,
            Some(resolution.binding().entity_count()),
            self.maximum_inline_result_bytes,
            |_| Err(invalid("deferred artifact input closure is not admitted")),
        )?;
        let prepared = prepare_signal_materialization(
            resolution.resolved().clone(),
            admitted,
            self.executable.clone(),
            plan.chunks,
            Vec::new(),
            plan.input_admissions,
            Some(mesh),
        )?;
        Ok(ResolvedDeferredSignalMaterialization {
            resolution,
            prepared,
        })
    }

    fn resolved_admission(
        &self,
        resolution: &SignalTopologyResolution,
    ) -> AnalysisBridgeResult<AdmittedSignalGenerator> {
        let admitted = crate::callable::executable::admit_named_signal_generator_from_executable(
            resolution.resolved(),
            self.admitted.program_revision.clone(),
            self.admitted.callable.clone(),
            &self.executable,
            &self.available_capabilities,
        )?;
        if admitted.generator_identity != self.admitted.generator_identity
            || admitted.provenance != self.admitted.provenance
            || admitted.frozen_inputs != self.admitted.frozen_inputs
            || admitted.materialization_identity == self.admitted.materialization_identity
        {
            return Err(AnalysisBridgeError::Identity("mesh resolution changed frozen invocation or retained unresolved materialization identity"));
        }
        Ok(admitted)
    }

    /// Cold validation uses original expected intent plus the authenticated returned mesh.
    /// The returned resolved expectation is not allowed to authorize its own source.
    pub fn validate_resolved_expectation(
        &self,
        expectation: &SignalMaterializationExpectation,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisBridgeResult<()> {
        self.validate()?;
        let resolution = SignalTopologyResolution::derive(&self.authored, mesh)?;
        expectation.validate_for_mesh(mesh)?;
        if expectation.authored != *resolution.resolved()
            || expectation.topology_binding.as_ref() != Some(resolution.binding())
            || expectation.admitted != self.resolved_admission(&resolution)?
        {
            return Err(AnalysisBridgeError::Identity(
                "resolved expectation differs from expected frozen intent and mesh",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedDeferredSignalMaterialization {
    resolution: SignalTopologyResolution,
    prepared: PreparedSignalMaterialization,
}

impl ResolvedDeferredSignalMaterialization {
    pub fn resolution(&self) -> &SignalTopologyResolution {
        &self.resolution
    }
    pub fn prepared(&self) -> &PreparedSignalMaterialization {
        &self.prepared
    }
    pub fn into_prepared(self) -> PreparedSignalMaterialization {
        self.prepared
    }
}

fn invalid(detail: &str) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(detail.into())
}

#[cfg(test)]
mod tests;
