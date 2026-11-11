mod accuracy;
mod metrics;
mod coordinates;
pub use coordinates::{ModalBasisCoordinates, ModalShapeLayout};

use runmat_canonical_codec::CanonicalLimits;
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use crate::{
    algebra::{apply_k, apply_m},
    assembly::AssemblySummary,
    contracts::FeaRunError,
    AnalysisRunIdentity,
};

use super::modal_governance::{
    build_modal_governance, ModalEigenspaceEvidence, ModalTruncationEvidence, ModalTruncationPolicy,
};
use super::modal_residual::{build_directional_residual_vectors, ModalResidualVector};
pub(crate) use metrics::directional_influence_vector;
use metrics::{
    directional_influence_vectors, directional_mass, directional_mass_gram_is_canonical,
    directional_metrics_are_consistent, dot, relative_frequency_separation,
    residual_directions_are_canonical, scalar_close,
};

const MODAL_BASIS_PREFIX: &[u8] = b"runmat-fea-modal-basis/v2\0";
const MODAL_BASIS_LIMITS: CanonicalLimits = crate::contracts::canonical_digest::wire_bounded(
    CanonicalLimits::new(512 * 1024 * 1024, 20_000_000, 4096, 16),
);

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
    backward_error: f64,
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
    #[serde(with = "coordinates::shared_wire")]
    coordinates: std::sync::Arc<ModalBasisCoordinates>,
    modes: Vec<ModalBasisMode>,
    residual_vectors: Vec<ModalResidualVector>,
    total_free_mass: [f64; 3],
    directional_mass_gram: [[f64; 3]; 3],
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
    #[serde(with = "coordinates::shared_wire")]
    coordinates: std::sync::Arc<ModalBasisCoordinates>,
    modes: Vec<ModalBasisMode>,
    residual_vectors: Vec<ModalResidualVector>,
    total_free_mass: [f64; 3],
    directional_mass_gram: [[f64; 3]; 3],
    mass_orthogonality: Vec<f64>,
    truncation: ModalTruncationEvidence,
    eigenspaces: Vec<ModalEigenspaceEvidence>,
}

pub(crate) struct ModalBasisInputs<'a> {
    pub frequencies_hz: &'a [f64],
    pub dof_shapes: &'a [Vec<f64>],
    pub residual_norms: &'a [f64],
    pub backward_errors: &'a [f64],
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
            coordinates: data.coordinates,
            modes: data.modes,
            residual_vectors: data.residual_vectors,
            total_free_mass: data.total_free_mass,
            directional_mass_gram: data.directional_mass_gram,
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
            backward_errors,
            truncation_policy,
            available_mode_count,
            boundary_probe_count,
            boundary_probe_frequency_hz,
        } = inputs;
        let count = frequencies_hz.len();
        if count == 0
            || !accuracy::columns_match(count, dof_shapes, residual_norms, backward_errors)
        {
            return Err(FeaRunError::Assembly(
                "modal basis inputs must be non-empty and have identical mode counts".into(),
            ));
        }
        let operator_digest = crate::assembly::identity::structural_operator_digest(summary)?;
        let influence = directional_influence_vectors(summary);
        let mass_applied_influence = influence
            .each_ref()
            .map(|vector| apply_m(&summary.operator, vector));
        let directional_mass_gram = std::array::from_fn(|row| {
            std::array::from_fn(|column| dot(&influence[row], &mass_applied_influence[column]))
        });
        let total_free_mass = std::array::from_fn(|axis| directional_mass_gram[axis][axis]);
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
                backward_error: backward_errors[index],
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
            coordinates: std::sync::Arc::new(ModalBasisCoordinates::for_layout(
                summary.structural_dof_layout.clone(),
            )),
            modes,
            residual_vectors: Vec::new(),
            total_free_mass,
            directional_mass_gram,
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
    pub fn coordinates(&self) -> &ModalBasisCoordinates {
        &self.coordinates
    }
    pub(crate) fn shared_coordinates(&self) -> std::sync::Arc<ModalBasisCoordinates> {
        std::sync::Arc::clone(&self.coordinates)
    }
    pub fn mode_shape_field(
        &self,
        index: usize,
    ) -> Result<runmat_types::physical::QuantityField<ModalShapeLayout<'_>, &[f64]>, FeaRunError>
    {
        let mode = self.modes.get(index).ok_or_else(|| {
            FeaRunError::Assembly("modal coordinate index is out of bounds".into())
        })?;
        self.coordinates
            .shape_field(&mode.dof_shape)
            .map_err(|error| FeaRunError::Assembly(error.into()))
    }
    pub fn modes(&self) -> &[ModalBasisMode] {
        &self.modes
    }
    pub fn residual_vectors(&self) -> &[ModalResidualVector] {
        &self.residual_vectors
    }

    pub(crate) fn with_directional_residual_vectors(
        mut self,
        summary: &AssemblySummary,
        directions: &[[f64; 3]],
    ) -> Result<Self, FeaRunError> {
        self.validate_for(summary)?;
        self.residual_vectors =
            build_directional_residual_vectors(summary, &self.modes, directions)?;
        self.identity = self.derive_identity()?;
        self.validate()?;
        Ok(self)
    }
    pub const fn total_free_mass(&self) -> [f64; 3] {
        self.total_free_mass
    }
    pub const fn directional_mass_gram(&self) -> [[f64; 3]; 3] {
        self.directional_mass_gram
    }
    pub fn total_free_mass_in_direction(&self, direction: [f64; 3]) -> f64 {
        directional_mass(self.directional_mass_gram, direction)
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
        if self.coordinates.full_dof_layout() != &summary.structural_dof_layout
            || self.operator_digest
                != crate::assembly::identity::structural_operator_digest(summary)?
            || self
                .modes
                .iter()
                .any(|mode| mode.dof_shape.len() != summary.dof_count)
        {
            return Err(FeaRunError::Assembly(
                "modal basis does not belong to the supplied structural operator".into(),
            ));
        }
        let dof_shapes = self
            .modes
            .iter()
            .map(|mode| mode.dof_shape.clone())
            .collect::<Vec<_>>();
        let (frequencies, residuals, backward_errors) = accuracy::mode_scalar_columns(&self.modes);
        let regenerated = Self::new(
            self.source_run.clone(),
            summary,
            ModalBasisInputs {
                frequencies_hz: &frequencies,
                dof_shapes: &dof_shapes,
                residual_norms: &residuals,
                backward_errors: &backward_errors,
                truncation_policy: self.truncation.policy(),
                available_mode_count: self.truncation.available_mode_count(),
                boundary_probe_count: self.truncation.boundary_probe_count(),
                boundary_probe_frequency_hz: self.truncation.boundary_probe_frequency_hz(),
            },
        )?;
        let regenerated = if self.residual_vectors.is_empty() {
            regenerated
        } else {
            regenerated.with_directional_residual_vectors(
                summary,
                &self
                    .residual_vectors
                    .iter()
                    .map(ModalResidualVector::direction)
                    .collect::<Vec<_>>(),
            )?
        };
        if regenerated.identity != self.identity {
            return Err(FeaRunError::Assembly(
                "modal basis metrics disagree with the supplied structural operator".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn validate(&self) -> Result<(), FeaRunError> {
        self.coordinates
            .validate()
            .map_err(|error| FeaRunError::Assembly(error.into()))?;
        if self.coordinates.full_dof_layout().node_count() != self.node_count
            || self.modes.iter().any(|mode| {
                mode.dof_shape.len() != self.coordinates.full_dof_layout().total_dof_count()
            })
        {
            return Err(FeaRunError::Assembly(
                "modal transform dimensions disagree with retained full layout".into(),
            ));
        }
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
            || !directional_mass_gram_is_canonical(self.directional_mass_gram, self.total_free_mass)
            || self.modes.iter().any(|mode| {
                !valid_scalar(mode.frequency_hz)
                    || mode.frequency_hz < 0.0
                    || !valid_scalar(mode.eigenvalue_rad2_per_s2)
                    || mode.eigenvalue_rad2_per_s2 < 0.0
                    || !valid_scalar(mode.residual_norm)
                    || mode.residual_norm < 0.0
                    || !(valid_scalar(mode.backward_error) && mode.backward_error >= 0.0)
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
            || self.residual_vectors.len() > 3
            || !residual_directions_are_canonical(&self.residual_vectors)
            || self.residual_vectors.iter().any(|residual| {
                residual.dof_shape().len() != self.modes[0].dof_shape.len()
                    || residual
                        .dof_shape()
                        .iter()
                        .any(|value| !valid_scalar(*value))
                    || !valid_scalar(residual.frequency_hz())
                    || residual.frequency_hz() <= 0.0
                    || !valid_scalar(residual.modal_mass())
                    || residual.modal_mass() <= 0.0
                    || !valid_scalar(residual.modal_stiffness())
                    || residual.modal_stiffness() <= 0.0
                    || !valid_scalar(residual.participation_factor())
                    || !valid_scalar(residual.effective_mass())
                    || residual.effective_mass() < 0.0
                    || !scalar_close(
                        residual.effective_mass(),
                        residual.participation_factor().powi(2) * residual.modal_mass(),
                    )
                    || !scalar_close(
                        residual.frequency_hz(),
                        (residual.modal_stiffness() / residual.modal_mass()).sqrt()
                            / (2.0 * std::f64::consts::PI),
                    )
                    || residual.frequency_hz() <= self.modes[count - 1].frequency_hz()
                    || !valid_scalar(residual.static_equilibrium_residual_ratio())
                    || !(0.0..=1.0e-8).contains(&residual.static_equilibrium_residual_ratio())
                    || !valid_scalar(residual.maximum_mass_orthogonality_ratio())
                    || !(0.0..=1.0e-8).contains(&residual.maximum_mass_orthogonality_ratio())
            })
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
        crate::contracts::canonical_digest::canonical_contract_digest(
            MODAL_BASIS_PREFIX,
            "modal-basis-artifact",
            &(
                &self.source_run,
                self.operator_digest,
                self.node_count,
                self.coordinates.as_ref(),
                &self.modes,
                &self.residual_vectors,
                self.total_free_mass,
                self.directional_mass_gram,
                &self.mass_orthogonality,
                &self.truncation,
                &self.eigenspaces,
            ),
            MODAL_BASIS_LIMITS,
        )
        .map(ModalBasisIdentity)
        .map_err(|error| FeaRunError::Assembly(error.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::directional_mass;

    #[test]
    fn directional_mass_retains_cross_axis_coupling() {
        let gram = [[2.0, 0.5, 0.0], [0.5, 3.0, -0.25], [0.0, -0.25, 4.0]];
        let direction = [0.8, 0.48, 0.36];
        let expected = 2.0 * 0.8_f64.powi(2)
            + 3.0 * 0.48_f64.powi(2)
            + 4.0 * 0.36_f64.powi(2)
            + 2.0 * 0.5 * 0.8 * 0.48
            - 2.0 * 0.25 * 0.48 * 0.36;
        assert!((directional_mass(gram, direction) - expected).abs() <= 1.0e-14);
    }
}
