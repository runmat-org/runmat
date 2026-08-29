use crate::{
    DistributedValueId, DistributionScheme, LabCount, ObjectFact, ParallelRegionId,
    ProgramFunctionId, QualifiedName, ShapeFact, StorageFact, SymbolName, ValueFact, ValueKindFact,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum DistributedOwner {
    Client(ProgramFunctionId),
    Region(ParallelRegionId),
}

impl DistributedOwner {
    pub const fn function(self) -> ProgramFunctionId {
        match self {
            Self::Client(function) => function,
            Self::Region(region) => region.0.function,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DistributedFact {
    pub id: DistributedValueId,
    pub owner: DistributedOwner,
    pub scheme: Option<DistributionScheme>,
    pub value: Box<ValueFact>,
    pub materializable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompositeFact {
    pub owner: ParallelRegionId,
    pub labs: LabCount,
    pub value: Box<ValueFact>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodistributorClass {
    OneDimensional,
    TwoDimensionalBlockCyclic,
}

impl CodistributorClass {
    pub const fn runtime_name(self) -> &'static str {
        match self {
            Self::OneDimensional => "codistributor1d",
            Self::TwoDimensionalBlockCyclic => "codistributor2dbc",
        }
    }

    pub const fn from_scheme(scheme: &DistributionScheme) -> Option<Self> {
        match scheme {
            DistributionScheme::Block { .. } | DistributionScheme::OneDimensional { .. } => {
                Some(Self::OneDimensional)
            }
            DistributionScheme::TwoDimensionalBlockCyclic { .. } => {
                Some(Self::TwoDimensionalBlockCyclic)
            }
            DistributionScheme::Replicated
            | DistributionScheme::Cyclic { .. }
            | DistributionScheme::Custom { .. } => None,
        }
    }
}

pub fn codistributor_fact(class: Option<CodistributorClass>) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: class
                .map(|class| QualifiedName(vec![SymbolName(class.runtime_name().to_owned())])),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}
