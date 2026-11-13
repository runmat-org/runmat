//! Cross-document link validation against the complete transition inventory.

use std::collections::BTreeSet;

use crate::{BuiltinDocumentation, BuiltinDocumentationLinkTarget};

pub(super) fn validate_scope<'a>(
    documents: impl IntoIterator<Item = (&'a str, &'a BuiltinDocumentation)>,
    complete_names: &BTreeSet<String>,
) -> Result<(), String> {
    let mut errors = Vec::new();
    for (name, documentation) in documents {
        for related in documentation.related {
            if !complete_names.contains(&related.to_ascii_lowercase()) {
                errors.push(format!(
                    "{name} references unknown related builtin {related}"
                ));
            }
        }
        for link in documentation.links {
            if let BuiltinDocumentationLinkTarget::Builtin(target) = link.target {
                if !complete_names.contains(&target.to_ascii_lowercase()) {
                    errors.push(format!("{name} links to unknown builtin {target}"));
                }
            }
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{public_builtin_catalog_entries, BuiltinDocumentationLink};

    #[test]
    fn collects_related_and_explicit_link_errors_across_the_selected_scope() {
        let mut document = public_builtin_catalog_entries()[0].documentation;
        document.related = &["missing_related"];
        document.links = &[BuiltinDocumentationLink {
            label: "Missing builtin",
            target: BuiltinDocumentationLinkTarget::Builtin("missing_target"),
        }];
        let error = validate_scope(
            [("first", &document), ("second", &document)],
            &BTreeSet::new(),
        )
        .unwrap_err();
        assert_eq!(error.lines().count(), 4);
        for name in ["first", "second"] {
            assert!(error.contains(&format!("{name} references unknown related builtin")));
            assert!(error.contains(&format!("{name} links to unknown builtin")));
        }
    }
}
