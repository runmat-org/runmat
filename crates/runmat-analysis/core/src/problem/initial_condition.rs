use serde::{Deserialize, Serialize};

use super::{
    coordinate_frame::CoordinateFrame,
    quantity::{
        AngularVelocity, Displacement, ElectricPotential, MagneticVectorPotential, Pressure,
        Temperature, Velocity,
    },
    signal::{SignalId, SignalSampleApproach},
    target::RegionTarget,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InitialCondition {
    Displacement(InitialConditionValue<Displacement>),
    Velocity(InitialConditionValue<Velocity>),
    AngularVelocity(InitialConditionValue<AngularVelocity>),
    Temperature(InitialConditionValue<Temperature>),
    ElectricPotential(InitialConditionValue<ElectricPotential>),
    MagneticVectorPotential(InitialConditionValue<MagneticVectorPotential>),
    AcousticPressure(InitialConditionValue<Pressure>),
    FluidPressure(InitialConditionValue<Pressure>),
    FluidVelocity(InitialConditionValue<Velocity>),
}

impl InitialCondition {
    pub fn field_artifact_ref(&self) -> Option<&InitialFieldArtifactRef> {
        fn field<Q>(value: &InitialConditionValue<Q>) -> Option<&InitialFieldArtifactRef> {
            match &value.value {
                InitialValue::FieldArtifact(reference) => Some(reference),
                _ => None,
            }
        }
        match self {
            Self::Displacement(value) => field(value),
            Self::Velocity(value) => field(value),
            Self::AngularVelocity(value) => field(value),
            Self::Temperature(value) => field(value),
            Self::ElectricPotential(value) => field(value),
            Self::MagneticVectorPotential(value) => field(value),
            Self::AcousticPressure(value) => field(value),
            Self::FluidPressure(value) => field(value),
            Self::FluidVelocity(value) => field(value),
        }
    }

    pub fn initial_condition_id(&self) -> &str {
        match self {
            Self::Displacement(value) => &value.initial_condition_id,
            Self::Velocity(value) => &value.initial_condition_id,
            Self::AngularVelocity(value) => &value.initial_condition_id,
            Self::Temperature(value) => &value.initial_condition_id,
            Self::ElectricPotential(value) => &value.initial_condition_id,
            Self::MagneticVectorPotential(value) => &value.initial_condition_id,
            Self::AcousticPressure(value) => &value.initial_condition_id,
            Self::FluidPressure(value) => &value.initial_condition_id,
            Self::FluidVelocity(value) => &value.initial_condition_id,
        }
    }

    pub fn applies_before_step_id(&self) -> &str {
        match self {
            Self::Displacement(value) => &value.applies_before_step_id,
            Self::Velocity(value) => &value.applies_before_step_id,
            Self::AngularVelocity(value) => &value.applies_before_step_id,
            Self::Temperature(value) => &value.applies_before_step_id,
            Self::ElectricPotential(value) => &value.applies_before_step_id,
            Self::MagneticVectorPotential(value) => &value.applies_before_step_id,
            Self::AcousticPressure(value) => &value.applies_before_step_id,
            Self::FluidPressure(value) => &value.applies_before_step_id,
            Self::FluidVelocity(value) => &value.applies_before_step_id,
        }
    }

    pub fn applies_to_step(&self, step: &crate::ProcedureStep) -> bool {
        self.applies_before_step_id() == step.step_id.as_str()
            && step.object_is_active(&crate::StepObjectRef::InitialCondition(
                self.initial_condition_id().to_string(),
            ))
    }

    pub fn region_target_mut(&mut self) -> Option<&mut RegionTarget> {
        let target = match self {
            Self::Displacement(value) => &mut value.target,
            Self::Velocity(value) => &mut value.target,
            Self::AngularVelocity(value) => &mut value.target,
            Self::Temperature(value) => &mut value.target,
            Self::ElectricPotential(value) => &mut value.target,
            Self::MagneticVectorPotential(value) => &mut value.target,
            Self::AcousticPressure(value) => &mut value.target,
            Self::FluidPressure(value) => &mut value.target,
            Self::FluidVelocity(value) => &mut value.target,
        };
        match target {
            InitialConditionTarget::Region(target) => Some(target),
            InitialConditionTarget::WholeModel => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InitialConditionValue<Q> {
    pub initial_condition_id: String,
    pub target: InitialConditionTarget,
    pub frame: CoordinateFrame,
    pub value: InitialValue<Q>,
    pub applies_before_step_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InitialConditionTarget {
    Region(RegionTarget),
    WholeModel,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InitialValue<Q> {
    Uniform(Q),
    UniformVector3([Q; 3]),
    SignalSample {
        signal_id: SignalId,
        abscissa: f64,
        approach: SignalSampleApproach,
    },
    FieldArtifact(InitialFieldArtifactRef),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InitialFieldArtifactRef {
    pub artifact_id: String,
    pub content_digest: String,
    pub topology_digest: String,
    pub field_id: String,
}
