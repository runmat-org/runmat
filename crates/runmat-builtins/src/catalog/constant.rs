use runmat_types::{NumericClass, NumericDomain, NumericFact, ValueFact, ValueKindFact};
use serde::Serialize;

use super::BuiltinCatalogProvenance;

/// Static, target-independent contract for a language constant.
///
/// The runtime registry separately binds each identity to its live `Value`.
/// Static consumers must use this catalog so type/shape analysis does not
/// inspect execution storage or require a runtime registration side effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinConstantCatalogEntry {
    pub name: &'static str,
    pub kind: BuiltinConstantKind,
    pub provenance: BuiltinCatalogProvenance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BuiltinConstantKind {
    RealDouble,
    ComplexDouble,
    Logical,
}

impl BuiltinConstantCatalogEntry {
    pub(crate) const fn new(
        name: &'static str,
        kind: BuiltinConstantKind,
        provenance: BuiltinCatalogProvenance,
    ) -> Self {
        Self {
            name,
            kind,
            provenance,
        }
    }

    pub fn fact(self) -> ValueFact {
        match self.kind {
            BuiltinConstantKind::RealDouble => numeric(NumericDomain::Real),
            BuiltinConstantKind::ComplexDouble => numeric(NumericDomain::Complex),
            BuiltinConstantKind::Logical => ValueFact::scalar(ValueKindFact::Logical),
        }
    }
}

fn numeric(domain: NumericDomain) -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain,
    }))
}
