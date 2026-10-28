//! Source declarations are evidence. A requested display unit or a numeric
//! fallback is never promoted into file-declared evidence.
use super::{LengthUnit, UnitError};
use serde::{Deserialize, Deserializer, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "tool-schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct LengthUnitDeclaration {
    pub context: String,
    pub source_symbol: String,
    pub unit: LengthUnit,
}
impl LengthUnitDeclaration {
    pub fn validate(&self) -> Result<(), UnitError> {
        if self.context.is_empty()
            || self.context.len() > 256
            || self.context.chars().any(char::is_control)
            || self.source_symbol.is_empty()
            || self.source_symbol.len() > 256
            || self.source_symbol.chars().any(char::is_control)
        {
            return Err(UnitError::InvalidSymbol);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[cfg_attr(feature = "tool-schema", derive(schemars::JsonSchema))]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum LengthUnitEvidence {
    Missing {
        reason: String,
    },
    Declared {
        contexts: Vec<LengthUnitDeclaration>,
    },
    Unsupported {
        contexts: Vec<String>,
        reason: String,
    },
    Conflicting {
        contexts: Vec<LengthUnitDeclaration>,
        reason: String,
    },
    FormatConvention {
        format: String,
        declaration: LengthUnitDeclaration,
    },
    ExplicitFallback {
        declaration: LengthUnitDeclaration,
        source: String,
    },
    /// An authored unit reinterprets coordinates whose file declares a different single length
    /// unit. The file declaration stays as provenance in `declared`.
    AuthoredOverride {
        declaration: LengthUnitDeclaration,
        declared: Vec<LengthUnitDeclaration>,
        source: String,
    },
}
impl Default for LengthUnitEvidence {
    fn default() -> Self {
        Self::Missing {
            reason: "legacy geometry has no preserved source-unit evidence".into(),
        }
    }
}
impl LengthUnitEvidence {
    pub fn explicit_fallback(unit: LengthUnit, source: impl Into<String>) -> Self {
        Self::ExplicitFallback {
            declaration: LengthUnitDeclaration {
                context: "unitless-source".into(),
                source_symbol: unit.symbol().into(),
                unit,
            },
            source: source.into(),
        }
    }
    pub fn uniform_unit(&self) -> Option<LengthUnit> {
        match self {
            Self::Declared { contexts } => {
                let first = contexts.first()?.unit;
                contexts.iter().all(|c| c.unit == first).then_some(first)
            }
            Self::ExplicitFallback { declaration, .. }
            | Self::AuthoredOverride { declaration, .. }
            | Self::FormatConvention { declaration, .. } => Some(declaration.unit),
            Self::Missing { .. } | Self::Unsupported { .. } | Self::Conflicting { .. } => None,
        }
    }
    pub fn physical_scale_resolved(&self) -> bool {
        matches!(self,Self::Declared{contexts}if !contexts.is_empty())
            || matches!(
                self,
                Self::ExplicitFallback { .. }
                    | Self::AuthoredOverride { .. }
                    | Self::FormatConvention { .. }
            )
    }
    pub fn validate(&self) -> Result<(), UnitError> {
        fn label(value: &str, limit: usize) -> bool {
            !value.trim().is_empty() && value.len() <= limit && !value.chars().any(char::is_control)
        }
        if let Self::Conflicting { reason, .. } = self {
            if !label(reason, 1024) {
                return Err(UnitError::InvalidSymbol);
            }
        }
        if let Self::Unsupported { contexts, .. } = self {
            if contexts.len() > 4096 || contexts.iter().any(|context| !label(context, 256)) {
                return Err(UnitError::InvalidSymbol);
            }
        }
        match self {
            Self::Declared { contexts } | Self::Conflicting { contexts, .. } => {
                if contexts.is_empty() || contexts.len() > 4096 {
                    return Err(UnitError::InvalidSymbol);
                }
                for context in contexts {
                    context.validate()?;
                }
                if matches!(self, Self::Declared { .. }) {
                    let mut seen = std::collections::BTreeMap::new();
                    for context in contexts {
                        if seen
                            .insert(&context.context, context.unit)
                            .is_some_and(|old| old != context.unit)
                        {
                            return Err(UnitError::InvalidConversion);
                        }
                    }
                }
            }
            Self::FormatConvention {
                format,
                declaration,
            } => {
                declaration.validate()?;
                if format.is_empty() || format.len() > 128 {
                    return Err(UnitError::InvalidSymbol);
                }
            }
            Self::ExplicitFallback {
                declaration,
                source,
            } => {
                declaration.validate()?;
                if !label(source, 1024) {
                    return Err(UnitError::InvalidSymbol);
                }
            }
            Self::AuthoredOverride {
                declaration,
                declared,
                source,
            } => {
                declaration.validate()?;
                if declared.is_empty() || declared.len() > 4096 || !label(source, 1024) {
                    return Err(UnitError::InvalidSymbol);
                }
                for context in declared {
                    context.validate()?;
                    if context.unit == declaration.unit {
                        return Err(UnitError::InvalidConversion);
                    }
                }
            }
            Self::Missing { reason } | Self::Unsupported { reason, .. } => {
                if !label(reason, 1024) {
                    return Err(UnitError::InvalidSymbol);
                }
            }
        }
        Ok(())
    }
    /// Apply an authored coordinate unit. Authored units are authoritative: a file declaring one
    /// different length unit is reinterpreted and its declaration kept as provenance; a file
    /// declaring the same unit keeps its declaration; a source without a declaration takes the
    /// authored unit as its explicit unit. Missing-declaration sources are returned as
    /// `ExplicitFallback`; callers that require a file declaration (STEP, IGES) refuse that
    /// state. Unsupported or conflicting declarations cannot be reinterpreted and are returned
    /// unchanged as the error.
    pub fn apply_authored(self, unit: LengthUnit, source: &str) -> Result<Self, Box<Self>> {
        let declaration = LengthUnitDeclaration {
            context: "authored geometry units".into(),
            source_symbol: unit.symbol().into(),
            unit,
        };
        match self {
            Self::Declared { contexts } => match contexts.first().map(|c| c.unit) {
                Some(first) if contexts.iter().all(|c| c.unit == first) => {
                    if first == unit {
                        Ok(Self::Declared { contexts })
                    } else {
                        Ok(Self::AuthoredOverride {
                            declaration,
                            declared: contexts,
                            source: source.into(),
                        })
                    }
                }
                _ => Err(Box::new(Self::Declared { contexts })),
            },
            Self::Missing { .. } => Ok(Self::ExplicitFallback {
                declaration,
                source: source.into(),
            }),
            Self::ExplicitFallback { .. } | Self::AuthoredOverride { .. } => Ok(self),
            other => Err(Box::new(other)),
        }
    }
    /// The file-declared unit an authored unit overrode, when it did.
    pub fn overridden_declared_unit(&self) -> Option<LengthUnit> {
        match self {
            Self::AuthoredOverride { declared, .. } => declared.first().map(|c| c.unit),
            _ => None,
        }
    }
    pub fn resolve_fallback(self, unit: Option<LengthUnit>, source: &str) -> Self {
        if !matches!(self, Self::Missing { .. }) {
            return self;
        }
        match unit {
            Some(unit) => Self::ExplicitFallback {
                declaration: LengthUnitDeclaration {
                    context: "unitless-source".into(),
                    source_symbol: "explicit fallback".into(),
                    unit,
                },
                source: source.into(),
            },
            None => self,
        }
    }
}
impl<'de> Deserialize<'de> for LengthUnitEvidence {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
        enum Wire {
            Missing {
                reason: String,
            },
            Declared {
                contexts: Vec<LengthUnitDeclaration>,
            },
            Unsupported {
                contexts: Vec<String>,
                reason: String,
            },
            Conflicting {
                contexts: Vec<LengthUnitDeclaration>,
                reason: String,
            },
            FormatConvention {
                format: String,
                declaration: LengthUnitDeclaration,
            },
            ExplicitFallback {
                declaration: LengthUnitDeclaration,
                source: String,
            },
            AuthoredOverride {
                declaration: LengthUnitDeclaration,
                declared: Vec<LengthUnitDeclaration>,
                source: String,
            },
        }
        let value = match Wire::deserialize(d)? {
            Wire::Missing { reason } => Self::Missing { reason },
            Wire::Declared { contexts } => Self::Declared { contexts },
            Wire::Unsupported { contexts, reason } => Self::Unsupported { contexts, reason },
            Wire::Conflicting { contexts, reason } => Self::Conflicting { contexts, reason },
            Wire::FormatConvention {
                format,
                declaration,
            } => Self::FormatConvention {
                format,
                declaration,
            },
            Wire::ExplicitFallback {
                declaration,
                source,
            } => Self::ExplicitFallback {
                declaration,
                source,
            },
            Wire::AuthoredOverride {
                declaration,
                declared,
                source,
            } => Self::AuthoredOverride {
                declaration,
                declared,
                source,
            },
        };
        value.validate().map_err(serde::de::Error::custom)?;
        Ok(value)
    }
}

mod role {
    pub trait Sealed {}
}
pub trait CoordinateRole: role::Sealed + Copy + std::fmt::Debug + PartialEq + Eq {
    const CANONICAL: bool;
}
pub trait SelectableCoordinateRole: CoordinateRole {}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreviewCoordinates;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelCoordinates;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CanonicalCoordinates;
impl role::Sealed for PreviewCoordinates {}
impl CoordinateRole for PreviewCoordinates {
    const CANONICAL: bool = false;
}
impl SelectableCoordinateRole for PreviewCoordinates {}
impl role::Sealed for KernelCoordinates {}
impl CoordinateRole for KernelCoordinates {
    const CANONICAL: bool = false;
}
impl SelectableCoordinateRole for KernelCoordinates {}
impl role::Sealed for CanonicalCoordinates {}
impl CoordinateRole for CanonicalCoordinates {
    const CANONICAL: bool = true;
}
/// Coordinate roles are distinct types. Canonical analysis coordinates cannot
/// be constructed with a non-SI scale, including through deserialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent, bound = "")]
pub struct CoordinateUnits<R: CoordinateRole> {
    unit: LengthUnit,
    #[serde(skip)]
    role: std::marker::PhantomData<R>,
}
impl<R: SelectableCoordinateRole> CoordinateUnits<R> {
    pub const fn new(unit: LengthUnit) -> Self {
        Self {
            unit,
            role: std::marker::PhantomData,
        }
    }
}
impl CoordinateUnits<CanonicalCoordinates> {
    pub const fn canonical() -> Self {
        Self {
            unit: LengthUnit::Meter,
            role: std::marker::PhantomData,
        }
    }
}
impl Default for CoordinateUnits<CanonicalCoordinates> {
    fn default() -> Self {
        Self::canonical()
    }
}
impl<R: CoordinateRole> CoordinateUnits<R> {
    pub const fn unit(self) -> LengthUnit {
        self.unit
    }
    pub fn meters_per_unit(self) -> f64 {
        self.unit.meters_per_unit()
    }
}
impl<'de, R: CoordinateRole> Deserialize<'de> for CoordinateUnits<R> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let unit = LengthUnit::deserialize(d)?;
        if R::CANONICAL && unit != LengthUnit::Meter {
            return Err(serde::de::Error::custom(
                "canonical solver coordinates must be meters",
            ));
        }
        Ok(Self {
            unit,
            role: std::marker::PhantomData,
        })
    }
}
#[cfg(feature = "tool-schema")]
impl<R: CoordinateRole> schemars::JsonSchema for CoordinateUnits<R> {
    fn schema_name() -> String {
        format!(
            "CoordinateUnits{}",
            std::any::type_name::<R>().rsplit("::").next().unwrap()
        )
    }
    fn json_schema(g: &mut schemars::gen::SchemaGenerator) -> schemars::schema::Schema {
        <LengthUnit as schemars::JsonSchema>::json_schema(g)
    }
}
