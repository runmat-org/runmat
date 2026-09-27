//! Syntax provenance is distinct from the implementation selected at execution.
use runmat_builtins::{builtin_catalog_entry_by_name, BuiltinBindingIdentity};
use runmat_types::OperatorKind;
use serde::{Deserialize, Deserializer, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub enum FusionCallOrigin {
    /// Older graphs and unresolved named calls carry no compiled binding proof.
    #[default]
    NamedUnresolved,
    Named {
        binding: BuiltinBindingIdentity,
    },
    Operator(OperatorKind),
}

impl FusionCallOrigin {
    pub fn named(name: &str) -> Self {
        let Some(entry) = builtin_catalog_entry_by_name(name) else {
            return Self::NamedUnresolved;
        };
        match entry.bindings {
            [binding] => Self::Named {
                binding: entry.binding_identity(binding),
            },
            _ => Self::NamedUnresolved,
        }
    }
}

// Catalog identities contain static references. Decode their existing stable
// fields at the serialization boundary and rejoin the catalog, never leak an
// arbitrary wire string into an executable identity.
impl<'de> Deserialize<'de> for FusionCallOrigin {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Catalog {
            name: String,
        }
        #[derive(Deserialize)]
        struct Binding {
            builtin: Catalog,
            variant: String,
        }
        #[derive(Deserialize)]
        enum Wire {
            NamedUnresolved,
            Named { binding: Binding },
            Operator(OperatorKind),
        }
        Ok(match Wire::deserialize(deserializer)? {
            Wire::NamedUnresolved => Self::NamedUnresolved,
            Wire::Operator(operator) => Self::Operator(operator),
            Wire::Named { binding } => {
                let Some(entry) = builtin_catalog_entry_by_name(&binding.builtin.name) else {
                    return Err(<D::Error as serde::de::Error>::custom(
                        "unknown canonical fusion identity",
                    ));
                };
                entry
                    .bindings
                    .iter()
                    .find_map(|candidate| {
                        let identity = entry.binding_identity(candidate);
                        (identity.variant == binding.variant)
                            .then_some(Self::Named { binding: identity })
                    })
                    .ok_or_else(|| {
                        <D::Error as serde::de::Error>::custom("unknown canonical fusion binding")
                    })?
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::AccelNodeLabel;

    #[test]
    fn old_graphs_remain_named_and_current_canonical_identity_round_trips() {
        let old: AccelNodeLabel = serde_json::from_str(r#"{"Builtin":{"name":"mtimes"}}"#).unwrap();
        assert!(matches!(
            old,
            AccelNodeLabel::Builtin {
                origin: FusionCallOrigin::NamedUnresolved,
                ..
            }
        ));
        for origin in [
            FusionCallOrigin::named("mtimes"),
            FusionCallOrigin::Operator(OperatorKind::MatrixMultiply),
        ] {
            assert_ne!(origin, FusionCallOrigin::NamedUnresolved);
            let encoded = serde_json::to_string(&origin).unwrap();
            assert_eq!(
                serde_json::from_str::<FusionCallOrigin>(&encoded).unwrap(),
                origin
            );
        }
        let unknown =
            r#"{"Named":{"binding":{"builtin":{"name":"mtimes"},"variant":"not-a-live-binding"}}}"#;
        assert!(serde_json::from_str::<FusionCallOrigin>(unknown).is_err());
    }
}
