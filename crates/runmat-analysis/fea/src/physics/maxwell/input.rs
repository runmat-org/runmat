//! Frozen, mesh/sample-bound physical coefficients and impressed volume current.
use super::{invalid, topology, ComplexPhasor, ComplexVector3, MaxwellError};
use crate::results::field::{SampledVolumeFieldTopology, VolumeFieldSamplingRule};
use runmat_canonical_codec::CanonicalLimits;
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
/// Constant vector carried by one element (unit generator field or unit closure current).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaxwellElementVector {
    pub element_identity: StableDigest,
    pub vector: [f64; 3],
}

/// How a terminal pair drives its conductor: total current or terminal voltage (peak phasors).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum MaxwellDriveExcitation {
    Current { amperes: ComplexPhasor },
    Voltage { volts: ComplexPhasor },
}

/// A solid conductor driven between two terminal faces (A-V formulation with an
/// electrokinetic unit potential). Inside the conductor the total field is
/// E + U g with g = -grad v0, where v0 solves div(sigma grad v0) = 0 with v0 = 1 on the
/// driven terminal and 0 on the return terminal. Current leaving the conductor through a
/// terminal that is not on a grounded boundary continues as a discrete potential flow in
/// the electrically transparent medium (the gap feed), so the impressed current is
/// discretely solenoidal and no artificial charge builds up on the terminals.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaxwellTerminalDrive {
    pub drive_id: String,
    pub excitation: MaxwellDriveExcitation,
    /// DC conductance between the terminals, integral of sigma |grad v0|^2 (S).
    pub conductance_s: f64,
    /// g = -grad v0 per driven-conductor element for a 1 V drive (V/m per V).
    pub unit_generator_field: Vec<MaxwellElementVector>,
    /// Gap-feed closure current density per transparent element for a 1 V drive (A/m^2 per V).
    pub unit_closure_current: Vec<MaxwellElementVector>,
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
    /// Sorted exterior faces with tangential E = 0 (vector-potential ground / PEC).
    /// Every other exterior face carries the natural condition n x H = 0.
    pub grounded_faces: Vec<StableDigest>,
    pub drives: Vec<MaxwellTerminalDrive>,
}
impl MaxwellInput {
    /// Canonical input identity includes exact sample authority, source, frequency
    /// and laws. Element law storage order is retained, while assignment is by ID.
    /// This is numerical lineage only, not a native task/publication authorization.
    pub fn identity(&self) -> Result<Digest, MaxwellError> {
        // Streamed: the mesh-sized input is hashed without materializing its encoding.
        crate::contracts::canonical_digest::canonical_contract_digest(
            b"runmat-maxwell-physical-input/v1\0",
            "maxwell-physical-input",
            self,
            CanonicalLimits::new(0, 0, 65_536, 64),
        )
        .map_err(invalid)
    }
    pub fn validate_for_mesh(&self, mesh: &SolverMeshArtifact) -> Result<(), MaxwellError> {
        self.ordered_volumes(mesh)?;
        if self
            .grounded_faces
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        {
            return Err(invalid("grounded faces must be sorted and unique"));
        }
        let elements: std::collections::BTreeSet<_> =
            self.sampling.element_identities().iter().copied().collect();
        let mut ids = std::collections::BTreeSet::new();
        for drive in &self.drives {
            if drive.drive_id.trim().is_empty()
                || !ids.insert(drive.drive_id.as_str())
                || !drive.conductance_s.is_finite()
                || drive.conductance_s <= 0.0
                || drive.unit_generator_field.is_empty()
                || !match drive.excitation {
                    MaxwellDriveExcitation::Current { amperes } => amperes.is_finite(),
                    MaxwellDriveExcitation::Voltage { volts } => volts.is_finite(),
                }
                || drive
                    .unit_generator_field
                    .iter()
                    .chain(&drive.unit_closure_current)
                    .any(|value| {
                        !elements.contains(&value.element_identity)
                            || value.vector.iter().any(|v| !v.is_finite())
                    })
            {
                return Err(invalid("invalid terminal drive"));
            }
        }
        Ok(())
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
        if self.source_current.len() != self.sampling.integration_sample_count()
            || self
                .source_current
                .iter()
                .enumerate()
                .any(|(index, source)| {
                    self.sampling
                        .integration_sample_identity(index)
                        .map_or(true, |identity| source.sample_identity != identity)
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
            if !(omega * volume.conductivity_s_per_m.canonical_value()).is_finite()
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
