//! Canonical metadata authority transported independently from selected numerical partitions.
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::SolverMeshArtifact;
use serde::{Deserialize, Serialize};

use super::resolve::{verify_authored_observations, verify_structural_observation_plan};
use super::{ObservationFieldSelection, ResolvedObservationPlan};
use crate::execution::{
    AnalysisExecutionError, AnalysisExecutionResult, FieldRecoveryRequest,
    ThermalSolverResultManifest, ThermalWorkload,
};
use crate::results::field::RecoveredFieldArtifactManifest;

mod entry_history;
mod sampled;
mod selection;
mod volume_mesh;
pub use selection::ObservationSelectionCoordinate;
pub use volume_mesh::VolumeMeshObservationAuthority;

const PREFIX: &[u8] = b"runmat-observation-field-catalog/v1\0";
const LIMITS: CanonicalLimits = crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

/// Producer metadata only: no recovered-field arrays or thermal snapshot arrays are retained.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", deny_unknown_fields)]
pub enum ObservationFieldCatalogSource {
    SampledRecovery {
        recovery_request: Box<FieldRecoveryRequest>,
        manifest: Box<RecoveredFieldArtifactManifest>,
        sampling: Box<crate::results::field::SampledVolumeFieldTopology>,
    },
    Structural {
        recovery_request: Box<FieldRecoveryRequest>,
        manifest: Box<RecoveredFieldArtifactManifest>,
    },
    /// Complete, ordered source authority for a history that spans the entries of a sampled
    /// study (one frequency of a sweep, for example). Children are the entries' individual
    /// single-sample catalogs of one kind (structural or volume-mesh recoveries), in increasing
    /// sample order; nested history catalogs are inadmissible.
    EntryHistory {
        samples: Vec<ObservationFieldCatalog>,
    },
    Thermal {
        manifest: Box<ThermalSolverResultManifest>,
    },
    /// Native node/element fields on the canonical solver mesh, for any physics family.
    /// The producing authored step is carried and bound to the request's producer, so
    /// plans re-resolve without decoding the family-specific workload.
    VolumeMeshRecovery {
        recovery_request: Box<FieldRecoveryRequest>,
        manifest: Box<RecoveredFieldArtifactManifest>,
        sequence: crate::execution::SequenceIdentity,
        step: Box<runmat_analysis_core::ProcedureStep>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "CatalogWire")]
pub struct ObservationFieldCatalog {
    schema_version: u16,
    identity: Digest,
    source: ObservationFieldCatalogSource,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CatalogWire {
    schema_version: u16,
    identity: Digest,
    source: ObservationFieldCatalogSource,
}

impl TryFrom<CatalogWire> for ObservationFieldCatalog {
    type Error = AnalysisExecutionError;
    fn try_from(wire: CatalogWire) -> Result<Self, Self::Error> {
        let catalog = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            source: wire.source,
        };
        catalog.validate()?;
        Ok(catalog)
    }
}

fn invalid(error: impl std::fmt::Display) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(format!("observation field catalog: {error}"))
}

impl ObservationFieldCatalog {
    /// Reconstruct structural targets from the producing authored model and exact topology.
    pub fn validate_structural_plan(
        &self,
        plan: &ResolvedObservationPlan,
        model: &runmat_analysis_core::AnalysisModel,
        mesh: Option<&SolverMeshArtifact>,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        let ObservationFieldCatalogSource::Structural {
            recovery_request,
            manifest,
        } = &self.source
        else {
            return Err(invalid(
                "structural plan requires a structural recovery catalog",
            ));
        };
        let step = model
            .sequence
            .steps
            .iter()
            .find(|step| step.step_id == plan.step_id)
            .ok_or_else(|| invalid("structural authored step absent"))?;
        let sequence = crate::execution::SequenceIdentity::from_solver_contract(&model.sequence)?;
        verify_authored_observations(step, plan).map_err(invalid)?;
        let producer = crate::execution::StepIdentity::from_solver_contract(sequence, step)?;
        let constraints =
            crate::constraints::CanonicalStructuralConstraintPlan::compile_for_model(model, mesh)
                .map_err(invalid)?;
        if producer != recovery_request.producer
            || recovery_request.topology
                != (crate::execution::FieldTopologyIdentity::Structural {
                    topology: constraints.topology,
                })
            || manifest.topology_digest != constraints.topology.discretization_digest()
        {
            return Err(invalid(
                "structural catalog differs from authored producer or canonical topology",
            ));
        }
        let mut seen = std::collections::BTreeSet::new();
        let definitions = manifest
            .fields
            .iter()
            .filter(|entry| seen.insert(entry.definition.identity()))
            .map(|entry| entry.definition.clone())
            .collect::<Vec<_>>();
        verify_structural_observation_plan(step, model, mesh, &constraints, &definitions, plan)
            .map_err(invalid)
    }
    /// Reconstruct the plan from producer metadata and the actual mesh, not asserted indices.
    pub fn validate_thermal_plan(
        &self,
        plan: &ResolvedObservationPlan,
        workload: &ThermalWorkload,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        let ObservationFieldCatalogSource::Thermal { manifest } = &self.source else {
            return Err(invalid("thermal plan requires a thermal catalog"));
        };
        super::source::ThermalFieldSourceAuthority::new(workload, manifest, mesh)?
            .validate_plan(plan)
    }

    pub fn structural(
        recovery_request: FieldRecoveryRequest,
        manifest: RecoveredFieldArtifactManifest,
    ) -> AnalysisExecutionResult<Self> {
        Self::new(ObservationFieldCatalogSource::Structural {
            recovery_request: Box::new(recovery_request),
            manifest: Box::new(manifest),
        })
    }

    pub fn thermal(manifest: ThermalSolverResultManifest) -> AnalysisExecutionResult<Self> {
        Self::new(ObservationFieldCatalogSource::Thermal {
            manifest: Box::new(manifest),
        })
    }

    fn new(source: ObservationFieldCatalogSource) -> AnalysisExecutionResult<Self> {
        let mut catalog = Self {
            schema_version: 1,
            identity: Digest::sha256([]),
            source,
        };
        catalog.identity = catalog.derive_identity()?;
        catalog.validate()?;
        Ok(catalog)
    }

    pub const fn identity(&self) -> Digest {
        self.identity
    }
    pub const fn source(&self) -> &ObservationFieldCatalogSource {
        &self.source
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        match &self.source {
            ObservationFieldCatalogSource::SampledRecovery {
                recovery_request,
                manifest,
                sampling,
            } => {
                super::source::validate_sampled_authority(recovery_request, manifest, sampling)?;
            }
            ObservationFieldCatalogSource::Structural {
                recovery_request,
                manifest,
            } => {
                recovery_request.validate()?;
                manifest.validate().map_err(invalid)?;
                if manifest.recovery_request != recovery_request.identity()
                    || manifest.source != recovery_request.source
                    || manifest.topology_digest != recovery_request.topology.discretization_digest()
                {
                    return Err(invalid(
                        "recovered-field manifest differs from producing request",
                    ));
                }
            }
            ObservationFieldCatalogSource::Thermal { manifest } => manifest.validate()?,
            ObservationFieldCatalogSource::VolumeMeshRecovery {
                recovery_request,
                manifest,
                sequence,
                step,
            } => {
                super::source::validate_volume_mesh_authority(recovery_request, manifest)?;
                if crate::execution::StepIdentity::from_solver_contract(*sequence, step)?
                    != recovery_request.producer
                {
                    return Err(invalid(
                        "volume-mesh catalog step differs from its recovery producer",
                    ));
                }
            }
            ObservationFieldCatalogSource::EntryHistory { samples } => {
                entry_history::validate_samples(samples)?;
            }
        }
        if self.schema_version != 1 || self.identity != self.derive_identity()? {
            return Err(invalid("unsupported schema or catalog identity mismatch"));
        }
        Ok(())
    }

    pub fn validate_structural_selection(
        &self,
        selection: &ObservationFieldSelection,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        match &self.source {
            ObservationFieldCatalogSource::Structural {
                recovery_request,
                manifest,
            } => selection.validate_structural(recovery_request, manifest),
            ObservationFieldCatalogSource::Thermal { .. }
            | ObservationFieldCatalogSource::EntryHistory { .. }
            | ObservationFieldCatalogSource::VolumeMeshRecovery { .. }
            | ObservationFieldCatalogSource::SampledRecovery { .. } => {
                Err(invalid("expected structural catalog"))
            }
        }
    }

    pub fn validate_thermal_selection(
        &self,
        selection: &ObservationFieldSelection,
        workload: &ThermalWorkload,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        match &self.source {
            ObservationFieldCatalogSource::Thermal { manifest } => {
                selection.validate_thermal(workload, manifest, mesh)
            }
            ObservationFieldCatalogSource::Structural { .. }
            | ObservationFieldCatalogSource::EntryHistory { .. }
            | ObservationFieldCatalogSource::VolumeMeshRecovery { .. }
            | ObservationFieldCatalogSource::SampledRecovery { .. } => {
                Err(invalid("expected thermal catalog"))
            }
        }
    }

    fn derive_identity(&self) -> AnalysisExecutionResult<Digest> {
        Ok(Digest::sha256(encode_contract(
            PREFIX,
            "observation-field-catalog-identity",
            &(self.schema_version, &self.source),
            LIMITS,
        )?))
    }
    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            PREFIX,
            "observation-field-catalog",
            self,
            LIMITS,
        )?)
    }
    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            PREFIX,
            "observation-field-catalog",
            bytes,
            LIMITS,
        )?)
    }
}

#[cfg(test)]
mod tests;
