use runmat_types::{CallableIdentity, QualifiedName, SymbolName};

/// Ownership of the existing canonical semantic identity at a call boundary.
/// Owned callers keep their original representation; compiled retained callers
/// share the original immutable source allocation, never a per-call Arc copy.
#[derive(Debug, Clone)]
pub enum CallableIdentityOwner {
    Owned(CallableIdentity),
    Retained(std::sync::Arc<CallableIdentity>),
}

impl CallableIdentityOwner {
    pub fn as_identity(&self) -> &CallableIdentity {
        match self {
            Self::Owned(identity) => identity,
            Self::Retained(identity) => identity.as_ref(),
        }
    }
}

impl From<CallableIdentity> for CallableIdentityOwner {
    fn from(identity: CallableIdentity) -> Self {
        Self::Owned(identity)
    }
}

impl From<std::sync::Arc<CallableIdentity>> for CallableIdentityOwner {
    fn from(identity: std::sync::Arc<CallableIdentity>) -> Self {
        Self::Retained(identity)
    }
}

impl PartialEq for CallableIdentityOwner {
    fn eq(&self, other: &Self) -> bool {
        self.as_identity() == other.as_identity()
    }
}
impl Eq for CallableIdentityOwner {}

impl PartialEq<CallableIdentity> for CallableIdentityOwner {
    fn eq(&self, other: &CallableIdentity) -> bool {
        self.as_identity() == other
    }
}

/// Builds the stable identity used for an externally resolved member.
pub fn external_qualified_identity(base: &str, member: &str) -> CallableIdentity {
    let base_segments = {
        let split = base.split('.').collect::<Vec<_>>();
        if !split.is_empty() && split.iter().all(|segment| !segment.is_empty()) {
            split
                .into_iter()
                .map(|segment| SymbolName(segment.to_string()))
                .collect::<Vec<_>>()
        } else {
            vec![SymbolName(base.to_string())]
        }
    };
    let mut segments = base_segments;
    segments.push(SymbolName(member.to_string()));
    CallableIdentity::ExternalName(QualifiedName(segments))
}

/// Returns a lossless source-facing name only when the identity carries one.
pub fn strict_callable_display_name(identity: &CallableIdentity) -> Option<String> {
    match identity {
        CallableIdentity::BoundFunction(_) | CallableIdentity::AnonymousFunction(_) => None,
        CallableIdentity::ExternalFunction { display_name, .. } => {
            (!display_name.is_empty()).then_some(display_name.clone())
        }
        CallableIdentity::Builtin(id) => (!id.0.is_empty()).then_some(id.0.clone()),
        CallableIdentity::Imported(path) => path.module.display_name(),
        CallableIdentity::Method(id) => (!id.0.is_empty()).then_some(id.0.clone()),
        CallableIdentity::DynamicName(name) => (!name.0.is_empty()).then_some(name.0.clone()),
        CallableIdentity::ExternalName(QualifiedName(segments)) => {
            if segments.is_empty() || segments.iter().any(|segment| segment.0.is_empty()) {
                return None;
            }
            Some(
                segments
                    .iter()
                    .map(|segment| segment.0.as_str())
                    .collect::<Vec<_>>()
                    .join("."),
            )
        }
    }
}

pub fn external_qualified_display_name(base: &str, member: &str) -> String {
    strict_callable_display_name(&external_qualified_identity(base, member))
        .expect("external qualified identity should always have a display name")
}

/// Missing-callable text preserves the source-facing name and the selected
/// typed identity. Formatting a name here never changes dispatch selection.
pub(crate) fn undefined_callable_message(identity: &CallableIdentity) -> String {
    match strict_callable_display_name(identity) {
        Some(name) => format!("Undefined function '{name}' for callable identity {identity:?}"),
        None => format!("Undefined function for callable identity {identity:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ownership_wrapper_retains_original_segments_without_allocating_an_arc_for_owned_calls() {
        let owned = CallableIdentity::DynamicName(SymbolName("plain".into()));
        let CallableIdentity::DynamicName(name) = &owned else {
            unreachable!()
        };
        let original_name = name.0.as_ptr();
        let owned: CallableIdentityOwner = owned.into();
        let CallableIdentityOwner::Owned(CallableIdentity::DynamicName(name)) = &owned else {
            panic!("owned call allocated a retained identity")
        };
        assert_eq!(name.0.as_ptr(), original_name);

        let original = std::sync::Arc::new(external_qualified_identity("pkg.Point", "origin"));
        let CallableIdentity::ExternalName(name) = original.as_ref() else {
            unreachable!()
        };
        let segments = name.0.as_ptr();
        let owner = CallableIdentityOwner::from(std::sync::Arc::clone(&original));
        let retained = owner.clone();
        let CallableIdentityOwner::Retained(retained) = retained else {
            panic!("retained source was copied")
        };
        assert!(std::sync::Arc::ptr_eq(&original, &retained));
        let CallableIdentity::ExternalName(name) = owner.as_identity() else {
            unreachable!()
        };
        assert_eq!(name.0.as_ptr(), segments);
        assert_eq!(std::sync::Arc::strong_count(&original), 3);
        drop(retained);
        drop(owner);
        assert_eq!(std::sync::Arc::strong_count(&original), 1);
    }

    #[test]
    fn qualified_identity_preserves_valid_segments() {
        let CallableIdentity::ExternalName(QualifiedName(segments)) =
            external_qualified_identity("pkg.Point", "origin")
        else {
            panic!("expected external name");
        };
        assert_eq!(
            segments,
            vec![
                SymbolName("pkg".into()),
                SymbolName("Point".into()),
                SymbolName("origin".into())
            ]
        );
    }

    #[test]
    fn malformed_base_remains_one_lossless_segment() {
        let CallableIdentity::ExternalName(QualifiedName(segments)) =
            external_qualified_identity("pkg..Point", "origin")
        else {
            panic!("expected external name");
        };
        assert_eq!(
            segments,
            vec![SymbolName("pkg..Point".into()), SymbolName("origin".into())]
        );
    }
}
