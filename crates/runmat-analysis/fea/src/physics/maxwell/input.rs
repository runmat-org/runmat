//! Frozen, mesh/sample-bound physical coefficients and impressed volume current.
use super::{invalid, topology, ComplexVector3, MaxwellError};
use crate::results::field::{SampledVolumeFieldTopology, VolumeFieldSamplingRule};
use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::{SolverMeshArtifact, StableDigest};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaxwellVolumeData {
    pub element_identity: StableDigest,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub permeability_h_per_m: runmat_types::physical::MagneticPermeability,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub permittivity_f_per_m: runmat_types::physical::ElectricPermittivity,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub conductivity_s_per_m: runmat_types::physical::ElectricalConductivity,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaxwellCurrentSample {
    pub sample_identity: StableDigest,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub current_density_a_per_m2: runmat_types::physical::Quantity<
        runmat_types::physical::kind::ElectricCurrentDensity,
        ComplexVector3,
    >,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaxwellInput {
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub angular_frequency_rad_per_s:
        runmat_types::physical::Quantity<runmat_types::physical::kind::AngularFrequency, f64>,
    pub sampling: SampledVolumeFieldTopology,
    pub volumes: Vec<MaxwellVolumeData>,
    /// Exact canonical integration-sample order, no omissions/aliases/callbacks.
    pub source_current: Vec<MaxwellCurrentSample>,
}
impl MaxwellInput {
    /// Canonical input identity includes exact sample authority, source, frequency
    /// and laws. Element law storage order is retained, while assignment is by ID.
    /// This is numerical lineage only, not a native task/publication authorization.
    pub fn identity(&self) -> Result<Digest, MaxwellError> {
        let bytes = encode_contract(
            b"runmat-maxwell-physical-input/v1\0",
            "maxwell-physical-input",
            self,
            CanonicalLimits::new(512 * 1024 * 1024, 8_000_000, 65_536, 64),
        )
        .map_err(invalid)?;
        Ok(Digest::sha256(bytes))
    }
    pub fn validate_for_mesh(&self, mesh: &SolverMeshArtifact) -> Result<(), MaxwellError> {
        self.ordered_volumes(mesh).map(|_| ())
    }
    pub(super) fn ordered_volumes(
        &self,
        mesh: &SolverMeshArtifact,
    ) -> Result<Vec<MaxwellVolumeData>, MaxwellError> {
        let omega = self.angular_frequency_rad_per_s.canonical_value();
        if !omega.is_finite() || omega <= 0.0 || !(omega * omega).is_finite() {
            return Err(invalid(
                "frequency must be finite, strictly positive and representable",
            ));
        }
        if self.sampling.rule() != VolumeFieldSamplingRule::TetrahedronKeastFifteen {
            return Err(MaxwellError::Unsupported(
                "Maxwell source requires declared Keast15 volume sampling".into(),
            ));
        }
        self.sampling.validate_for_mesh(mesh).map_err(topology)?;
        if self.source_current.len() != self.sampling.integration_samples().len()
            || self
                .source_current
                .iter()
                .zip(self.sampling.integration_samples())
                .any(|(source, sample)| {
                    source.sample_identity != sample.identity
                        || source
                            .current_density_a_per_m2
                            .canonical_value()
                            .iter()
                            .any(|value| !value.is_finite())
                })
        {
            return Err(invalid(
                "source sample identity/order/cardinality/value differs from canonical topology",
            ));
        }
        let mut inventory = BTreeMap::new();
        for volume in &self.volumes {
            if !volume.permeability_h_per_m.canonical_value().is_finite()
                || volume.permeability_h_per_m.canonical_value() <= 0.0
                || !volume.permittivity_f_per_m.canonical_value().is_finite()
                || volume.permittivity_f_per_m.canonical_value() <= 0.0
                || !volume.conductivity_s_per_m.canonical_value().is_finite()
                || volume.conductivity_s_per_m.canonical_value() < 0.0
                || inventory.insert(volume.element_identity, *volume).is_some()
            {
                return Err(invalid("invalid or duplicate Maxwell element law"));
            }
            if volume.conductivity_s_per_m.canonical_value() == 0.0 {
                return Err(MaxwellError::Unsupported("zero-loss Maxwell regions need independent spectral/nonresonance authority; residual is not uniqueness evidence".into()));
            }
            if !(omega * volume.conductivity_s_per_m.canonical_value()).is_finite()
                || omega * volume.conductivity_s_per_m.canonical_value() == 0.0
                || !(omega * omega * volume.permittivity_f_per_m.canonical_value()).is_finite()
                || omega * omega * volume.permittivity_f_per_m.canonical_value() == 0.0
                || !(1.0 / volume.permeability_h_per_m.canonical_value()).is_finite()
            {
                return Err(invalid(
                    "Maxwell physical coefficients exceed finite numerical envelope",
                ));
            }
        }
        let ordered = self
            .sampling
            .element_identities()
            .iter()
            .map(|identity| {
                inventory
                    .remove(identity)
                    .ok_or_else(|| invalid("missing Maxwell element law"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if !inventory.is_empty() {
            return Err(invalid("foreign Maxwell element identity"));
        }
        Ok(ordered)
    }
}
