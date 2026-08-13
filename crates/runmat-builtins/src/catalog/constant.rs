use runmat_types::{NumericClass, NumericDomain, NumericFact, ValueFact, ValueKindFact};
use serde::{Deserialize, Serialize};

/// Static, target-independent contract for a language constant.
///
/// The runtime registry separately binds each identity to its live `Value`.
/// Static consumers must use this catalog so type/shape analysis does not
/// inspect execution storage or require a runtime registration side effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BuiltinConstantCatalogEntry {
    pub name: &'static str,
    pub kind: BuiltinConstantKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BuiltinConstantKind {
    RealDouble,
    ComplexDouble,
    Logical,
}

impl BuiltinConstantCatalogEntry {
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

static CONSTANTS: [BuiltinConstantCatalogEntry; 11] = [
    BuiltinConstantCatalogEntry {
        name: "pi",
        kind: BuiltinConstantKind::RealDouble,
    },
    BuiltinConstantCatalogEntry {
        name: "eps",
        kind: BuiltinConstantKind::RealDouble,
    },
    BuiltinConstantCatalogEntry {
        name: "sqrt2",
        kind: BuiltinConstantKind::RealDouble,
    },
    BuiltinConstantCatalogEntry {
        name: "i",
        kind: BuiltinConstantKind::ComplexDouble,
    },
    BuiltinConstantCatalogEntry {
        name: "j",
        kind: BuiltinConstantKind::ComplexDouble,
    },
    BuiltinConstantCatalogEntry {
        name: "inf",
        kind: BuiltinConstantKind::RealDouble,
    },
    BuiltinConstantCatalogEntry {
        name: "Inf",
        kind: BuiltinConstantKind::RealDouble,
    },
    BuiltinConstantCatalogEntry {
        name: "nan",
        kind: BuiltinConstantKind::RealDouble,
    },
    BuiltinConstantCatalogEntry {
        name: "NaN",
        kind: BuiltinConstantKind::RealDouble,
    },
    BuiltinConstantCatalogEntry {
        name: "true",
        kind: BuiltinConstantKind::Logical,
    },
    BuiltinConstantCatalogEntry {
        name: "false",
        kind: BuiltinConstantKind::Logical,
    },
];

pub fn builtin_constant_catalog_entries() -> &'static [BuiltinConstantCatalogEntry] {
    &CONSTANTS
}

pub fn builtin_constant_catalog_entry_by_name(
    name: &str,
) -> Option<&'static BuiltinConstantCatalogEntry> {
    CONSTANTS.iter().find(|entry| entry.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identities_are_unique_and_facts_cover_the_runtime_constant_set() {
        let names = builtin_constant_catalog_entries()
            .iter()
            .map(|entry| entry.name)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(names.len(), builtin_constant_catalog_entries().len());
        assert!(matches!(
            builtin_constant_catalog_entry_by_name("pi")
                .expect("pi")
                .fact()
                .kind,
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            })
        ));
        assert_eq!(
            builtin_constant_catalog_entry_by_name("true")
                .expect("true")
                .fact()
                .kind,
            ValueKindFact::Logical
        );
        assert!(builtin_constant_catalog_entry_by_name("PI").is_none());
    }
}
