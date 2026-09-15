use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use crate::{
    algebra::{apply_k, apply_m},
    assembly::{dofs::StructuralDofKind, AssemblySummary},
    contracts::FeaRunError,
    AnalysisRunIdentity,
};

use super::modal_governance::{
    build_modal_governance, ModalEigenspaceEvidence, ModalTruncationEvidence, ModalTruncationPolicy,
};

const MODAL_BASIS_PREFIX: &[u8] = b"runmat-fea-modal-basis/v1\0";
const MODAL_BASIS_LIMITS: CanonicalLimits =
    CanonicalLimits::new(512 * 1024 * 1024, 20_000_000, 4096, 16);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ModalBasisIdentity(Digest);

impl ModalBasisIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModalDirectionalMetric {
    participation_factor: f64,
    effective_mass: f64,
    cumulative_effective_mass: f64,
    cumulative_mass_ratio: f64,
}

impl ModalDirectionalMetric {
    pub const fn participation_factor(self) -> f64 {
        self.participation_factor
    }

    pub const fn effective_mass(self) -> f64 {
        self.effective_mass
    }

    pub const fn cumulative_effective_mass(self) -> f64 {
        self.cumulative_effective_mass
    }

    pub const fn cumulative_mass_ratio(self) -> f64 {
        self.cumulative_mass_ratio
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModalBasisMode {
    frequency_hz: f64,
    eigenvalue_rad2_per_s2: f64,
    residual_norm: f64,
    modal_mass: f64,
    modal_stiffness: f64,
    relative_frequency_separation: f64,
    directional: [ModalDirectionalMetric; 3],
    dof_shape: Vec<f64>,
}

impl ModalBasisMode {
    pub const fn frequency_hz(&self) -> f64 {
        self.frequency_hz
    }
    pub const fn eigenvalue_rad2_per_s2(&self) -> f64 {
        self.eigenvalue_rad2_per_s2
    }
    pub const fn residual_norm(&self) -> f64 {
        self.residual_norm
    }
    pub const fn modal_mass(&self) -> f64 {
        self.modal_mass
    }
    pub const fn modal_stiffness(&self) -> f64 {
        self.modal_stiffness
    }
    pub const fn relative_frequency_separation(&self) -> f64 {
        self.relative_frequency_separation
    }
    pub const fn directional(&self) -> &[ModalDirectionalMetric; 3] {
        &self.directional
    }
    pub fn dof_shape(&self) -> &[f64] {
        &self.dof_shape
    }
}

/// Immutable, operator-bound modal basis suitable for later modal response procedures.
///
/// The raw DOF vectors remain authoritative for computation. Nodal translation and rotation are
/// projected separately for results and observations, so rotational DOFs are never folded into
/// displacement fields. Direction order is always global X, Y, Z.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModalBasisArtifact {
    identity: ModalBasisIdentity,
    source_run: AnalysisRunIdentity,
    operator_digest: Digest,
    node_count: usize,
    modes: Vec<ModalBasisMode>,
    total_free_mass: [f64; 3],
    mass_orthogonality: Vec<f64>,
    truncation: ModalTruncationEvidence,
    eigenspaces: Vec<ModalEigenspaceEvidence>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModalBasisArtifactData {
    identity: ModalBasisIdentity,
    source_run: AnalysisRunIdentity,
    operator_digest: Digest,
    node_count: usize,
    modes: Vec<ModalBasisMode>,
    total_free_mass: [f64; 3],
    mass_orthogonality: Vec<f64>,
    truncation: ModalTruncationEvidence,
    eigenspaces: Vec<ModalEigenspaceEvidence>,
}

pub(crate) struct ModalBasisInputs<'a> {
    pub frequencies_hz: &'a [f64],
    pub dof_shapes: &'a [Vec<f64>],
    pub residual_norms: &'a [f64],
    pub truncation_policy: ModalTruncationPolicy,
    pub available_mode_count: usize,
    pub boundary_probe_count: usize,
    pub boundary_probe_frequency_hz: Option<f64>,
}

impl TryFrom<ModalBasisArtifactData> for ModalBasisArtifact {
    type Error = FeaRunError;

    fn try_from(data: ModalBasisArtifactData) -> Result<Self, Self::Error> {
        let artifact = Self {
            identity: data.identity,
            source_run: data.source_run,
            operator_digest: data.operator_digest,
            node_count: data.node_count,
            modes: data.modes,
            total_free_mass: data.total_free_mass,
            mass_orthogonality: data.mass_orthogonality,
            truncation: data.truncation,
            eigenspaces: data.eigenspaces,
        };
        artifact.validate()?;
        Ok(artifact)
    }
}

impl<'de> Deserialize<'de> for ModalBasisArtifact {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        ModalBasisArtifactData::deserialize(deserializer)?
            .try_into()
            .map_err(serde::de::Error::custom)
    }
}

impl ModalBasisArtifact {
    pub(crate) fn new(
        source_run: AnalysisRunIdentity,
        summary: &AssemblySummary,
        inputs: ModalBasisInputs<'_>,
    ) -> Result<Self, FeaRunError> {
        let ModalBasisInputs {
            frequencies_hz,
            dof_shapes,
            residual_norms,
            truncation_policy,
            available_mode_count,
            boundary_probe_count,
            boundary_probe_frequency_hz,
        } = inputs;
        let count = frequencies_hz.len();
        if count == 0 || dof_shapes.len() != count || residual_norms.len() != count {
            return Err(FeaRunError::Assembly(
                "modal basis inputs must be non-empty and have identical mode counts".into(),
            ));
        }
        let operator_digest = crate::assembly::identity::structural_operator_digest(summary)?;
        let influence = directional_influence_vectors(summary);
        let total_free_mass = influence
            .each_ref()
            .map(|vector| dot(vector, &apply_m(&summary.operator, vector)));
        let separation = relative_frequency_separation(frequencies_hz);
        let mut cumulative = [0.0; 3];
        let mut modes = Vec::with_capacity(count);
        for index in 0..count {
            let shape = &dof_shapes[index];
            if shape.len() != summary.dof_count || shape.iter().any(|value| !value.is_finite()) {
                return Err(FeaRunError::Assembly(format!(
                    "modal DOF shape {index} does not match the assembled operator"
                )));
            }
            let mass_applied = apply_m(&summary.operator, shape);
            let stiffness_applied = apply_k(&summary.operator, shape);
            let modal_mass = dot(shape, &mass_applied);
            let modal_stiffness = dot(shape, &stiffness_applied);
            if !modal_mass.is_finite()
                || modal_mass <= 0.0
                || !modal_stiffness.is_finite()
                || modal_stiffness < 0.0
            {
                return Err(FeaRunError::Assembly(format!(
                    "modal mode {index} has invalid physical mass or stiffness"
                )));
            }
            let directional = std::array::from_fn(|axis| {
                let numerator = dot(shape, &apply_m(&summary.operator, &influence[axis]));
                let participation_factor = numerator / modal_mass;
                let effective_mass = numerator * numerator / modal_mass;
                cumulative[axis] += effective_mass;
                ModalDirectionalMetric {
                    participation_factor,
                    effective_mass,
                    cumulative_effective_mass: cumulative[axis],
                    cumulative_mass_ratio: if total_free_mass[axis] > 0.0 {
                        cumulative[axis] / total_free_mass[axis]
                    } else {
                        0.0
                    },
                }
            });
            modes.push(ModalBasisMode {
                frequency_hz: frequencies_hz[index],
                eigenvalue_rad2_per_s2: (2.0 * std::f64::consts::PI * frequencies_hz[index])
                    .powi(2),
                residual_norm: residual_norms[index],
                modal_mass,
                modal_stiffness,
                relative_frequency_separation: separation[index],
                directional,
                dof_shape: shape.clone(),
            });
        }
        let mass_orthogonality: Vec<f64> = dof_shapes
            .iter()
            .flat_map(|left| {
                dof_shapes
                    .iter()
                    .map(|right| dot(left, &apply_m(&summary.operator, right)))
            })
            .collect();
        let (truncation, eigenspaces) = build_modal_governance(
            truncation_policy,
            frequencies_hz,
            residual_norms,
            &mass_orthogonality,
            available_mode_count,
            boundary_probe_count,
            boundary_probe_frequency_hz,
        )?;
        let mut artifact = Self {
            identity: ModalBasisIdentity(Digest::sha256([])),
            source_run,
            operator_digest,
            node_count: summary.structural_dof_layout.node_count(),
            modes,
            total_free_mass,
            mass_orthogonality,
            truncation,
            eigenspaces,
        };
        artifact.identity = artifact.derive_identity()?;
        artifact.validate()?;
        Ok(artifact)
    }

    pub const fn identity(&self) -> ModalBasisIdentity {
        self.identity
    }
    pub fn source_run(&self) -> &AnalysisRunIdentity {
        &self.source_run
    }
    pub const fn operator_digest(&self) -> Digest {
        self.operator_digest
    }
    pub const fn node_count(&self) -> usize {
        self.node_count
    }
    pub fn modes(&self) -> &[ModalBasisMode] {
        &self.modes
    }
    pub const fn total_free_mass(&self) -> [f64; 3] {
        self.total_free_mass
    }
    pub fn mass_orthogonality(&self) -> &[f64] {
        &self.mass_orthogonality
    }
    pub const fn truncation(&self) -> &ModalTruncationEvidence {
        &self.truncation
    }
    pub fn eigenspaces(&self) -> &[ModalEigenspaceEvidence] {
        &self.eigenspaces
    }

    pub fn validate_for(&self, summary: &AssemblySummary) -> Result<(), FeaRunError> {
        self.validate()?;
        if self.operator_digest != crate::assembly::identity::structural_operator_digest(summary)?
            || self
                .modes
                .iter()
                .any(|mode| mode.dof_shape.len() != summary.dof_count)
        {
            return Err(FeaRunError::Assembly(
                "modal basis does not belong to the supplied structural operator".into(),
            ));
        }
        let frequencies = self
            .modes
            .iter()
            .map(|mode| mode.frequency_hz)
            .collect::<Vec<_>>();
        let dof_shapes = self
            .modes
            .iter()
            .map(|mode| mode.dof_shape.clone())
            .collect::<Vec<_>>();
        let residuals = self
            .modes
            .iter()
            .map(|mode| mode.residual_norm)
            .collect::<Vec<_>>();
        let regenerated = Self::new(
            self.source_run.clone(),
            summary,
            ModalBasisInputs {
                frequencies_hz: &frequencies,
                dof_shapes: &dof_shapes,
                residual_norms: &residuals,
                truncation_policy: self.truncation.policy(),
                available_mode_count: self.truncation.available_mode_count(),
                boundary_probe_count: self.truncation.boundary_probe_count(),
                boundary_probe_frequency_hz: self.truncation.boundary_probe_frequency_hz(),
            },
        )?;
        if regenerated.identity != self.identity {
            return Err(FeaRunError::Assembly(
                "modal basis metrics disagree with the supplied structural operator".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn validate(&self) -> Result<(), FeaRunError> {
        let count = self.modes.len();
        let valid_scalar = |value: f64| value.is_finite();
        let frequencies = self
            .modes
            .iter()
            .map(|mode| mode.frequency_hz)
            .collect::<Vec<_>>();
        let residuals = self
            .modes
            .iter()
            .map(|mode| mode.residual_norm)
            .collect::<Vec<_>>();
        let regenerated_governance = build_modal_governance(
            self.truncation.policy(),
            &frequencies,
            &residuals,
            &self.mass_orthogonality,
            self.truncation.available_mode_count(),
            self.truncation.boundary_probe_count(),
            self.truncation.boundary_probe_frequency_hz(),
        );
        if count == 0
            || self.node_count == 0
            || self.mass_orthogonality.len() != count.saturating_mul(count)
            || self.modes.windows(2).any(|pair| {
                pair[0]
                    .frequency_hz
                    .total_cmp(&pair[1].frequency_hz)
                    .is_gt()
            })
            || self
                .modes
                .iter()
                .map(|mode| mode.dof_shape.len())
                .any(|length| length == 0 || length != self.modes[0].dof_shape.len())
            || self
                .total_free_mass
                .iter()
                .any(|value| !valid_scalar(*value) || *value < 0.0)
            || self.modes.iter().any(|mode| {
                !valid_scalar(mode.frequency_hz)
                    || mode.frequency_hz < 0.0
                    || !valid_scalar(mode.eigenvalue_rad2_per_s2)
                    || mode.eigenvalue_rad2_per_s2 < 0.0
                    || !valid_scalar(mode.residual_norm)
                    || mode.residual_norm < 0.0
                    || !valid_scalar(mode.modal_mass)
                    || mode.modal_mass <= 0.0
                    || !valid_scalar(mode.modal_stiffness)
                    || mode.modal_stiffness < 0.0
                    || !valid_scalar(mode.relative_frequency_separation)
                    || mode.relative_frequency_separation < 0.0
                    || mode.dof_shape.is_empty()
                    || mode.dof_shape.iter().any(|value| !valid_scalar(*value))
                    || mode.directional.iter().any(|metric| {
                        [
                            metric.participation_factor,
                            metric.effective_mass,
                            metric.cumulative_effective_mass,
                            metric.cumulative_mass_ratio,
                        ]
                        .iter()
                        .any(|value| !valid_scalar(*value))
                            || metric.effective_mass < 0.0
                            || metric.cumulative_effective_mass < 0.0
                            || metric.cumulative_mass_ratio < 0.0
                    })
            })
            || self
                .mass_orthogonality
                .iter()
                .any(|value| !valid_scalar(*value))
            || self.truncation.retained_mode_count() != count
            || self.eigenspaces.is_empty()
            || regenerated_governance
                .as_ref()
                .map(|(truncation, eigenspaces)| {
                    truncation != &self.truncation || eigenspaces != &self.eigenspaces
                })
                .unwrap_or(true)
            || !directional_metrics_are_consistent(&self.modes, self.total_free_mass)
            || self.identity != self.derive_identity()?
        {
            return Err(FeaRunError::Assembly(
                "modal basis artifact is not canonical".into(),
            ));
        }
        Ok(())
    }

    fn derive_identity(&self) -> Result<ModalBasisIdentity, FeaRunError> {
        encode_contract(
            MODAL_BASIS_PREFIX,
            "modal-basis-artifact",
            &(
                &self.source_run,
                self.operator_digest,
                self.node_count,
                &self.modes,
                self.total_free_mass,
                &self.mass_orthogonality,
                &self.truncation,
                &self.eigenspaces,
            ),
            MODAL_BASIS_LIMITS,
        )
        .map(Digest::sha256)
        .map(ModalBasisIdentity)
        .map_err(|error| FeaRunError::Assembly(error.to_string()))
    }
}

fn directional_influence_vectors(summary: &AssemblySummary) -> [Vec<f64>; 3] {
    std::array::from_fn(|axis| {
        let mut vector = vec![0.0; summary.dof_count];
        for (row, value) in vector.iter_mut().enumerate() {
            if summary
                .operator
                .constrained_dofs()
                .get(row)
                .copied()
                .unwrap_or(true)
            {
                continue;
            }
            let expected = [
                StructuralDofKind::Ux,
                StructuralDofKind::Uy,
                StructuralDofKind::Uz,
            ][axis];
            if summary
                .structural_dof_layout
                .address(row)
                .is_some_and(|address| address.kind == expected)
            {
                *value = 1.0;
            }
        }
        vector
    })
}

fn relative_frequency_separation(frequencies: &[f64]) -> Vec<f64> {
    if frequencies.len() < 2 {
        return vec![1.0; frequencies.len()];
    }
    let mut separation = vec![f64::INFINITY; frequencies.len()];
    for (index, window) in frequencies.windows(2).enumerate() {
        let value =
            (window[1] - window[0]).abs() / window[0].abs().max(window[1].abs()).max(1.0e-12);
        separation[index] = separation[index].min(value);
        separation[index + 1] = separation[index + 1].min(value);
    }
    separation
        .into_iter()
        .map(|value| if value.is_finite() { value } else { 1.0 })
        .collect()
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn directional_metrics_are_consistent(modes: &[ModalBasisMode], total_mass: [f64; 3]) -> bool {
    let mut cumulative = [0.0; 3];
    for mode in modes {
        for axis in 0..3 {
            let metric = mode.directional[axis];
            cumulative[axis] += metric.effective_mass;
            let expected_ratio = if total_mass[axis] > 0.0 {
                cumulative[axis] / total_mass[axis]
            } else {
                0.0
            };
            if metric.cumulative_effective_mass.to_bits() != cumulative[axis].to_bits()
                || metric.cumulative_mass_ratio.to_bits() != expected_ratio.to_bits()
            {
                return false;
            }
        }
    }
    true
}
