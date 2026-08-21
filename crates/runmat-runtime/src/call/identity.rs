use runmat_types::{CallableIdentity, QualifiedName, SymbolName};

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

#[cfg(test)]
mod tests {
    use super::*;

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
