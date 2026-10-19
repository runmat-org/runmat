use runmat_types::{NumericClass, NumericDomain, NumericFact, ValueFact, ValueKindFact};
use serde::Serialize;

use super::{BuiltinCatalogProvenance, BuiltinDocumentation};

/// Static, target-independent contract for a language constant.
///
/// The runtime registry separately binds each identity to its live `Value`.
/// Static consumers must use this catalog so type/shape analysis does not
/// inspect execution storage or require a runtime registration side effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinConstantCatalogEntry {
    pub name: &'static str,
    pub kind: BuiltinConstantKind,
    /// Exact scalar content when the constant has a portable literal value.
    /// The execution fingerprint includes this field; documentation keeps its
    /// existing catalog shape rather than exposing an additional wire field.
    #[serde(skip)]
    pub literal: Option<BuiltinConstantLiteral>,
    pub provenance: BuiltinCatalogProvenance,
    /// Canonical content for a constant-only public identity. Callable forms
    /// and alternate spellings share their canonical identity's documentation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub documentation: Option<BuiltinDocumentation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BuiltinConstantKind {
    RealDouble,
    ComplexDouble,
    Logical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinConstantLiteral {
    Logical(bool),
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
            literal: None,
            provenance,
            documentation: None,
        }
    }

    pub const fn with_documentation(mut self, documentation: BuiltinDocumentation) -> Self {
        self.documentation = Some(documentation);
        self
    }

    pub const fn with_literal(mut self, literal: BuiltinConstantLiteral) -> Self {
        match (self.kind, literal) {
            (BuiltinConstantKind::Logical, BuiltinConstantLiteral::Logical(_)) => {}
            _ => panic!("constant literal does not match its catalog kind"),
        }
        self.literal = Some(literal);
        self
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optional_documentation_preserves_legacy_shape_and_constant_facts() {
        let original = BuiltinConstantCatalogEntry::new(
            "pi",
            BuiltinConstantKind::RealDouble,
            BuiltinCatalogProvenance::new(file!(), module_path!()),
        );
        assert!(original.documentation.is_none());
        assert!(serde_json::to_value(original)
            .unwrap()
            .get("documentation")
            .is_none());
        let documented = original.with_documentation(BuiltinDocumentation::EMPTY);
        assert_eq!(original.fact(), documented.fact());
        assert!(serde_json::to_value(documented)
            .unwrap()
            .get("documentation")
            .is_some());
    }
}
