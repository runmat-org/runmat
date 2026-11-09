use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use super::{
    authored_parameterized_signals::AuthoredParameterizedSignal,
    coordinate_frame::CoordinateFrame,
    excitation::{
        Direction, ExcitationCaseActivation, ExcitationDistribution, ExcitationKinematics,
        ProjectionRequirement,
    },
    model_authoring::AuthoredRegionSource,
    signal::{PhysicalQuantityKind, PhysicalSignal, SignalId, SignalShape},
};
use runmat_geometry_core::GeometryRegionRole;

#[cfg(test)]
mod deferred_topology_tests;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredExcitation {
    pub excitation_id: String,
    pub kind: AuthoredExcitationKind,
    pub target: AuthoredExcitationTarget,
    pub frame: CoordinateFrame,
    pub direction: Option<Direction>,
    pub distribution: ExcitationDistribution,
    pub kinematics: ExcitationKinematics,
    pub signal_id: SignalId,
    pub active_step_ids: Vec<String>,
    pub combination_group: Option<String>,
    pub projection: ProjectionRequirement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthoredExcitationKind {
    Force,
    Moment,
    Traction,
    Pressure,
    BodyForce,
    HeatGeneration,
    HeatFlux,
    TemperatureField,
    AmbientTemperature,
    TerminalVoltage,
    ElectricCurrent,
    ElectricCharge,
    ElectricChargeDensity,
    CurrentDensity,
    /// Vector J in A/m² supported on a solid volume, not the surface-current action.
    MagneticVolumeCurrentDensity,
    MagneticVectorPotential,
    AcousticPressure,
    /// Uniform incoming pressure on a planar exterior acoustic boundary.
    AcousticIncidentPressure,
    NormalVelocity,
    VolumeVelocity,
    InletVelocity,
    MassFlow,
    WallMotion,
}

#[cfg(test)]
mod charge_density_tests;

#[cfg(test)]
mod magnetic_volume_current_tests;

#[cfg(test)]
mod incident_pressure_tests;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthoredExcitationTarget {
    Region(AuthoredRegionSource),
    WholeModel,
    /// The reference point of a rigid or distributing coupling, named by the coupling's
    /// constraint id; a force or moment there is transmitted through the coupling.
    ReferencePoint {
        constraint_id: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredExcitationCase {
    pub case_id: String,
    pub excitation_ids: Vec<String>,
    pub activation: ExcitationCaseActivation,
}

impl AuthoredExcitationKind {
    pub fn quantity(self) -> PhysicalQuantityKind {
        match self {
            Self::Force => PhysicalQuantityKind::Force,
            Self::Moment => PhysicalQuantityKind::Moment,
            Self::Traction => PhysicalQuantityKind::Traction,
            Self::Pressure | Self::AcousticPressure | Self::AcousticIncidentPressure => {
                PhysicalQuantityKind::Pressure
            }
            Self::BodyForce => PhysicalQuantityKind::BodyForceDensity,
            Self::HeatGeneration => PhysicalQuantityKind::HeatGenerationDensity,
            Self::HeatFlux => PhysicalQuantityKind::HeatFlux,
            Self::TemperatureField | Self::AmbientTemperature => PhysicalQuantityKind::Temperature,
            Self::TerminalVoltage => PhysicalQuantityKind::ElectricPotential,
            Self::ElectricCurrent => PhysicalQuantityKind::ElectricCurrent,
            Self::ElectricCharge => PhysicalQuantityKind::ElectricCharge,
            Self::ElectricChargeDensity => PhysicalQuantityKind::ElectricChargeDensity,
            Self::CurrentDensity => PhysicalQuantityKind::ElectricCurrentDensity,
            Self::MagneticVolumeCurrentDensity => PhysicalQuantityKind::ElectricCurrentDensity,
            Self::MagneticVectorPotential => PhysicalQuantityKind::MagneticVectorPotential,
            Self::NormalVelocity | Self::InletVelocity | Self::WallMotion => {
                PhysicalQuantityKind::Velocity
            }
            Self::MassFlow => PhysicalQuantityKind::MassFlowRate,
            Self::VolumeVelocity => PhysicalQuantityKind::VolumeFlowRate,
        }
    }

    /// The signal quantity an excitation of this kind reads under `distribution`: a heat
    /// generation preserving its resultant reads a total heat rate (W), compiled to a uniform
    /// density over the meshed region volume.
    pub fn signal_quantity(self, distribution: ExcitationDistribution) -> PhysicalQuantityKind {
        match (self, distribution) {
            (Self::HeatGeneration, ExcitationDistribution::PreserveResultant) => {
                PhysicalQuantityKind::Power
            }
            _ => self.quantity(),
        }
    }

    pub fn default_distribution(self) -> ExcitationDistribution {
        match self {
            Self::Force
            | Self::Moment
            | Self::ElectricCurrent
            | Self::ElectricCharge
            | Self::MassFlow
            | Self::VolumeVelocity => ExcitationDistribution::PreserveResultant,
            Self::Traction | Self::Pressure | Self::HeatFlux | Self::CurrentDensity => {
                ExcitationDistribution::IntensityPerArea
            }
            Self::BodyForce
            | Self::HeatGeneration
            | Self::ElectricChargeDensity
            | Self::MagneticVolumeCurrentDensity => ExcitationDistribution::IntensityPerVolume,
            Self::TemperatureField
            | Self::AmbientTemperature
            | Self::TerminalVoltage
            | Self::MagneticVectorPotential
            | Self::AcousticPressure
            | Self::AcousticIncidentPressure
            | Self::NormalVelocity
            | Self::InletVelocity
            | Self::WallMotion => ExcitationDistribution::UniformKinematic,
        }
    }

    pub fn preserves_resultant(self) -> bool {
        matches!(
            self,
            Self::Force
                | Self::Moment
                | Self::ElectricCurrent
                | Self::ElectricCharge
                | Self::MassFlow
                | Self::VolumeVelocity
        )
    }

    pub fn preserves_moment(self) -> bool {
        matches!(self, Self::Moment)
    }
}

pub fn validate_authored_excitations(
    excitations: &[AuthoredExcitation],
    cases: &[AuthoredExcitationCase],
    signals: &[PhysicalSignal],
    parameterized_signals: &[AuthoredParameterizedSignal],
) -> Result<(), String> {
    let signals = signals
        .iter()
        .map(|signal| (signal.signal_id().0.as_str(), signal))
        .collect::<BTreeMap<_, _>>();
    let mut ids = BTreeSet::new();
    for excitation in excitations {
        if excitation.excitation_id.trim().is_empty()
            || !ids.insert(excitation.excitation_id.as_str())
        {
            return Err("excitation ids must be non-empty and unique".into());
        }
        validate_excitation(excitation, &signals, parameterized_signals)?;
    }
    let mut case_ids = BTreeSet::new();
    let mut membership = BTreeSet::new();
    for case in cases {
        if case.case_id.trim().is_empty()
            || !case_ids.insert(case.case_id.as_str())
            || case.excitation_ids.is_empty()
        {
            return Err("excitation-case ids and membership must be non-empty and unique".into());
        }
        let mut local = BTreeSet::new();
        for excitation_id in &case.excitation_ids {
            if !ids.contains(excitation_id.as_str()) || !local.insert(excitation_id.as_str()) {
                return Err("excitation cases must reference unique declared excitations".into());
            }
            membership.insert(excitation_id.as_str());
        }
        match &case.activation {
            ExcitationCaseActivation::Steps(steps)
                if steps.is_empty()
                    || steps.iter().any(|step| step.trim().is_empty())
                    || steps.iter().collect::<BTreeSet<_>>().len() != steps.len() =>
            {
                return Err("step-activated excitation cases require unique step ids".into())
            }
            ExcitationCaseActivation::MutuallyExclusiveGroup { group_id }
                if group_id.trim().is_empty() =>
            {
                return Err("mutually exclusive excitation cases require a group id".into())
            }
            _ => {}
        }
    }
    if membership.len() != excitations.len() {
        return Err("every excitation must belong to at least one declared case".into());
    }
    Ok(())
}

fn validate_excitation(
    excitation: &AuthoredExcitation,
    signals: &BTreeMap<&str, &PhysicalSignal>,
    parameterized_signals: &[AuthoredParameterizedSignal],
) -> Result<(), String> {
    let signal = signals.get(excitation.signal_id.0.as_str());
    let parameterized = parameterized_signals
        .iter()
        .find(|signal| signal.signal_id == excitation.signal_id);
    let quantity = signal
        .map(|signal| signal.quantity())
        .or_else(|| parameterized.map(|signal| signal.quantity))
        .ok_or_else(|| "excitation references an unknown signal".to_string())?;
    let shape = signal
        .map(|signal| signal.shape().clone())
        .or_else(|| parameterized.map(AuthoredParameterizedSignal::shape))
        .ok_or_else(|| "excitation references an unknown signal".to_string())?;
    if quantity != excitation.kind.signal_quantity(excitation.distribution) {
        return Err("excitation signal has the wrong physical quantity".into());
    }
    if shape.topology_component_count().is_some()
        && excitation.distribution != ExcitationDistribution::TopologyField
    {
        return Err("topology-field signals require topology-field distribution".into());
    }
    if !valid_shape_and_direction(excitation.kind, &shape, excitation.direction.is_some()) {
        return Err("excitation direction is incompatible with its kind and signal shape".into());
    }
    if let Some(direction) = excitation.direction {
        let norm = direction
            .vector
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        if !norm.is_finite() || norm <= f64::EPSILON || (norm - 1.0).abs() > 1.0e-10 {
            return Err("excitation direction must be a finite unit vector".into());
        }
    }
    if excitation
        .active_step_ids
        .iter()
        .any(|step| step.trim().is_empty())
        || excitation
            .active_step_ids
            .iter()
            .collect::<BTreeSet<_>>()
            .len()
            != excitation.active_step_ids.len()
        || excitation
            .combination_group
            .as_ref()
            .is_some_and(|group| group.trim().is_empty())
        || !valid_distribution(excitation.kind, excitation.distribution)
        || (excitation.kinematics == ExcitationKinematics::Follower
            && !matches!(
                excitation.kind,
                AuthoredExcitationKind::Force
                    | AuthoredExcitationKind::Traction
                    | AuthoredExcitationKind::Pressure
            ))
        || excitation.projection.preserve_resultant != excitation.kind.preserves_resultant()
        || excitation.projection.preserve_moment != excitation.kind.preserves_moment()
        || !excitation.projection.relative_tolerance.is_finite()
        || excitation.projection.relative_tolerance <= 0.0
    {
        return Err("excitation activation, grouping, and projection policy must be valid".into());
    }
    let role = match &excitation.target {
        AuthoredExcitationTarget::Region(source) => Some(source.role()),
        AuthoredExcitationTarget::WholeModel | AuthoredExcitationTarget::ReferencePoint { .. } => {
            None
        }
    };
    let reference_point = matches!(
        excitation.target,
        AuthoredExcitationTarget::ReferencePoint { .. }
    );
    let surface = matches!(role, Some(GeometryRegionRole::Face));
    let volume = matches!(role, Some(GeometryRegionRole::Solid));
    let attachment = matches!(
        role,
        Some(GeometryRegionRole::Vertex | GeometryRegionRole::Edge | GeometryRegionRole::Face)
    );
    let valid_target = match excitation.kind {
        AuthoredExcitationKind::Force | AuthoredExcitationKind::Moment => {
            attachment || reference_point
        }
        AuthoredExcitationKind::BodyForce
        | AuthoredExcitationKind::HeatGeneration
        | AuthoredExcitationKind::ElectricChargeDensity
        | AuthoredExcitationKind::MagneticVolumeCurrentDensity => {
            volume || (role.is_none() && !reference_point)
        }
        AuthoredExcitationKind::TemperatureField => volume,
        _ => surface,
    };
    valid_target
        .then_some(())
        .ok_or_else(|| "excitation target topology is incompatible with its physical kind".into())
}

fn valid_shape_and_direction(
    kind: AuthoredExcitationKind,
    shape: &SignalShape,
    has_direction: bool,
) -> bool {
    if let Some(width) = shape.topology_component_count() {
        return !has_direction
            && match kind {
                AuthoredExcitationKind::Traction
                | AuthoredExcitationKind::BodyForce
                | AuthoredExcitationKind::CurrentDensity
                | AuthoredExcitationKind::MagneticVectorPotential
                | AuthoredExcitationKind::InletVelocity
                | AuthoredExcitationKind::WallMotion => width == 3,
                AuthoredExcitationKind::Pressure
                | AuthoredExcitationKind::HeatGeneration
                | AuthoredExcitationKind::HeatFlux
                | AuthoredExcitationKind::TemperatureField
                | AuthoredExcitationKind::AmbientTemperature
                | AuthoredExcitationKind::TerminalVoltage
                | AuthoredExcitationKind::AcousticPressure
                | AuthoredExcitationKind::NormalVelocity => width == 1,
                _ => false,
            };
    }
    let scalar = shape == &SignalShape::Scalar;
    let vector = shape == &SignalShape::Vector3;
    match kind {
        AuthoredExcitationKind::Force
        | AuthoredExcitationKind::Moment
        | AuthoredExcitationKind::InletVelocity
        | AuthoredExcitationKind::WallMotion => {
            (scalar && has_direction) || (vector && !has_direction)
        }
        AuthoredExcitationKind::Traction
        | AuthoredExcitationKind::BodyForce
        | AuthoredExcitationKind::CurrentDensity
        | AuthoredExcitationKind::MagneticVolumeCurrentDensity
        | AuthoredExcitationKind::MagneticVectorPotential => vector && !has_direction,
        AuthoredExcitationKind::Pressure
        | AuthoredExcitationKind::HeatGeneration
        | AuthoredExcitationKind::HeatFlux
        | AuthoredExcitationKind::TemperatureField
        | AuthoredExcitationKind::AmbientTemperature
        | AuthoredExcitationKind::TerminalVoltage
        | AuthoredExcitationKind::ElectricCurrent
        | AuthoredExcitationKind::ElectricCharge
        | AuthoredExcitationKind::ElectricChargeDensity
        | AuthoredExcitationKind::AcousticPressure
        | AuthoredExcitationKind::AcousticIncidentPressure
        | AuthoredExcitationKind::NormalVelocity
        | AuthoredExcitationKind::VolumeVelocity
        | AuthoredExcitationKind::MassFlow => scalar && !has_direction,
    }
}

fn valid_distribution(kind: AuthoredExcitationKind, distribution: ExcitationDistribution) -> bool {
    distribution == kind.default_distribution()
        || matches!(
            (kind, distribution),
            (
                AuthoredExcitationKind::Force | AuthoredExcitationKind::Moment,
                ExcitationDistribution::Concentrated
            ) | (
                AuthoredExcitationKind::HeatGeneration,
                ExcitationDistribution::PreserveResultant
            ) | (
                AuthoredExcitationKind::Traction
                    | AuthoredExcitationKind::Pressure
                    | AuthoredExcitationKind::BodyForce
                    | AuthoredExcitationKind::HeatGeneration
                    | AuthoredExcitationKind::HeatFlux
                    | AuthoredExcitationKind::TemperatureField
                    | AuthoredExcitationKind::AmbientTemperature
                    | AuthoredExcitationKind::TerminalVoltage
                    | AuthoredExcitationKind::CurrentDensity
                    | AuthoredExcitationKind::MagneticVectorPotential
                    | AuthoredExcitationKind::AcousticPressure
                    | AuthoredExcitationKind::NormalVelocity
                    | AuthoredExcitationKind::InletVelocity
                    | AuthoredExcitationKind::WallMotion,
                ExcitationDistribution::TopologyField
            )
        )
}
