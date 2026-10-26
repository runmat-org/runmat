//! One default/range authority for the constructor, static inference and
//! runtime control preparation. Dimension-dependent counts are not evaluated
//! as arbitrary user expressions.
use super::super::OptimOptionField as Field;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OptimPatternDefault {
    Number(f64),
    Logical(bool),
    Text(&'static str),
    DimensionCount(usize),
    Empty,
}
impl OptimPatternDefault {
    pub const fn for_field(field: Field) -> Option<Self> {
        Some(match field {
            Field::Algorithm => Self::Text("classic"),
            Field::Display => Self::Text("final"),
            Field::TolX | Field::FunctionTolerance | Field::TolCon | Field::MeshTolerance => {
                Self::Number(1e-6)
            }
            Field::MaxIter => Self::DimensionCount(100),
            Field::MaxFunEvals => Self::DimensionCount(2000),
            Field::InitialMeshSize => Self::Number(1.0),
            Field::MaxMeshSize | Field::MaxTime => Self::Number(f64::INFINITY),
            Field::MeshContractionFactor => Self::Number(0.5),
            Field::MeshExpansionFactor => Self::Number(2.0),
            Field::MeshRotate => Self::Text("on"),
            Field::AccelerateMesh
            | Field::UseCompletePoll
            | Field::UseCompleteSearch
            | Field::UseVectorized => Self::Logical(false),
            Field::ScaleMesh => Self::Logical(true),
            Field::PollMethod => Self::Text("GPSPositiveBasis2N"),
            Field::PollOrderAlgorithm => Self::Text("Consecutive"),
            Field::SearchFcn | Field::OutputFcn | Field::PlotFcn => Self::Empty,
            Field::TolBind => Self::Number(1e-3),
            Field::InitialPenalty => Self::Number(10.0),
            Field::PenaltyFactor => Self::Number(100.0),
            Field::Cache | Field::UseParallel => Self::Text("off"),
            Field::CacheSize => Self::Number(10_000.0),
            Field::CacheTol => Self::Number(f64::EPSILON),
            Field::PlotInterval => Self::Number(1.0),
            _ => return None,
        })
    }
    pub const fn dimension_expression(field: Field) -> Option<&'static str> {
        match field {
            Field::MaxIter => Some("100*numberOfVariables"),
            Field::MaxFunEvals => Some("2000*numberOfVariables"),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimPatternScalarRule {
    Nonnegative,
    Positive,
    PositiveOrInfinity,
    UnitContraction,
    Expansion,
    AtLeastOne,
    Count { zero: bool },
    Time,
}
impl OptimPatternScalarRule {
    pub const fn for_field(field: Field) -> Option<Self> {
        Some(match field {
            Field::TolX | Field::FunctionTolerance | Field::MeshTolerance | Field::TolBind => {
                Self::Nonnegative
            }
            Field::TolCon | Field::InitialMeshSize | Field::CacheTol => Self::Positive,
            Field::MaxMeshSize => Self::PositiveOrInfinity,
            Field::MeshContractionFactor => Self::UnitContraction,
            Field::MeshExpansionFactor | Field::PenaltyFactor => Self::Expansion,
            Field::InitialPenalty => Self::AtLeastOne,
            Field::MaxIter | Field::MaxFunEvals => Self::Count { zero: true },
            Field::PlotInterval | Field::CacheSize => Self::Count { zero: false },
            Field::MaxTime => Self::Time,
            _ => return None,
        })
    }
    pub fn accepts(self, value: f64) -> bool {
        match self {
            Self::Nonnegative => value.is_finite() && value >= 0.0,
            Self::Positive => value.is_finite() && value > 0.0,
            Self::PositiveOrInfinity => !value.is_nan() && value > 0.0,
            Self::UnitContraction => value.is_finite() && value > 0.0 && value < 1.0,
            Self::Expansion => value.is_finite() && value > 1.0,
            Self::AtLeastOne => value.is_finite() && value >= 1.0,
            Self::Count { zero } => {
                value.is_finite()
                    && value.fract() == 0.0
                    && if zero { value >= 0.0 } else { value > 0.0 }
            }
            Self::Time => !value.is_nan() && value >= 0.0,
        }
    }
}
