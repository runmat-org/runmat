use runmat_analysis_core::{ExcitationCaseId, ExcitationId, Position};

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ComplexScalar {
    pub real: f64,
    pub imaginary: f64,
}

impl ComplexScalar {
    pub(crate) const fn real(value: f64) -> Self {
        Self {
            real: value,
            imaginary: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ComplexVector3 {
    pub real: [f64; 3],
    pub imaginary: [f64; 3],
}

impl ComplexVector3 {
    pub(crate) const fn real(value: [f64; 3]) -> Self {
        Self {
            real: value,
            imaginary: [0.0; 3],
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum RealizedExcitationTarget {
    Region(String),
    ReferencePoint {
        reference_point_id: String,
        position: Position,
    },
    Port(String),
    Interface(String),
    WholeModel,
}

impl RealizedExcitationTarget {
    pub(crate) fn solver_region_id(&self) -> Option<&str> {
        match self {
            Self::Region(region_id) => Some(region_id),
            Self::ReferencePoint {
                reference_point_id, ..
            } => Some(reference_point_id),
            Self::Port(_) | Self::Interface(_) | Self::WholeModel => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum RealizedExcitationKind {
    Force(ComplexVector3),
    Moment(ComplexVector3),
    Wrench {
        force: ComplexVector3,
        moment: ComplexVector3,
        point: Position,
    },
    Traction(ComplexVector3),
    Pressure(ComplexScalar),
    BodyForce(ComplexVector3),
    BaseDisplacement(ComplexVector3),
    BaseVelocity(ComplexVector3),
    BaseAcceleration(ComplexVector3),
    HeatGeneration(ComplexScalar),
    HeatFlux(ComplexScalar),
    TemperatureField(ComplexScalar),
    AmbientTemperature(ComplexScalar),
    TerminalVoltage(ComplexScalar),
    ElectricCurrent(ComplexScalar),
    ElectricCharge(ComplexScalar),
    CurrentDensity(ComplexVector3),
    MagneticVectorPotential(ComplexVector3),
    AcousticPressure(ComplexScalar),
    NormalVelocity(ComplexScalar),
    InletVelocity(ComplexVector3),
    MassFlow(ComplexScalar),
    WallMotion(ComplexVector3),
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RealizedExcitation {
    pub excitation_id: ExcitationId,
    pub case_id: ExcitationCaseId,
    pub target: RealizedExcitationTarget,
    pub kind: RealizedExcitationKind,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct RealizedExcitations {
    entries: Vec<RealizedExcitation>,
}

impl RealizedExcitations {
    pub(crate) fn new(entries: Vec<RealizedExcitation>) -> Self {
        Self { entries }
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &RealizedExcitation> {
        self.entries.iter()
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }
}
