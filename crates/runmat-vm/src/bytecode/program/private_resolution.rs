//! Private-name observation from the selected VM definition registry.
use super::FunctionRegistry;
use runmat_hir::FunctionId;
use std::path::Path;

impl FunctionRegistry {
    /// Empty owner scope denotes a loose function-file directory, not global
    /// visibility. The captured source paths belong to selected executable
    /// definitions; no current filesystem search grants private authority.
    pub fn resolve_private_name_for_caller(
        &self,
        caller: FunctionId,
        name: &str,
    ) -> Option<FunctionId> {
        let body = self.get(caller)?;
        if !body.private_owner_scope.is_empty() {
            let selected = self.resolve_name_in_private_scope(&body.private_owner_scope, name)?;
            return self.get_current(selected).map(|_| selected);
        }
        if name.contains('.') {
            return None;
        }
        let caller_file = Path::new(body.source_file.as_deref()?);
        let caller_directory = caller_file.parent()?;
        let lexical_owner = if caller_directory
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.eq_ignore_ascii_case("private"))
        {
            caller_directory.parent()?
        } else {
            caller_directory
        };
        let synthetic_name = format!("__private__.{name}");
        unique_private_candidate(
            &synthetic_name,
            name,
            body.private_source_owner.as_ref(),
            lexical_owner,
            self.functions.values(),
        )
    }
}

fn unique_private_candidate<'a>(
    synthetic_name: &str,
    name: &str,
    captured_owner: Option<&super::PrivateSourceDirectory>,
    lexical_owner: &Path,
    candidates: impl Iterator<Item = &'a super::FunctionBytecode>,
) -> Option<FunctionId> {
    let mut matches = candidates.filter(|candidate| {
        (candidate.display_name == synthetic_name || candidate.display_name == name)
            && candidate.source_file.as_deref().is_some_and(|source| {
                let path = Path::new(source);
                let Some(private_directory) = path.parent() else {
                    return false;
                };
                private_directory
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.eq_ignore_ascii_case("private"))
                    && match (captured_owner, candidate.private_source_owner.as_ref()) {
                        (Some(caller), Some(candidate))
                            if caller.is_known() && candidate.is_known() =>
                        {
                            caller.same_owner(candidate)
                        }
                        _ => private_directory.parent() == Some(lexical_owner),
                    }
            })
    });
    let selected = matches.next()?;
    matches.next().is_none().then_some(selected.function)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::{FunctionBytecode, Instr, PrivateSourceAlias, PrivateSourceDirectory};
    use std::{collections::HashMap, path::Path};

    fn body(id: usize, name: &str, source: Option<&Path>) -> FunctionBytecode {
        FunctionBytecode {
            function: FunctionId(id),
            display_name: name.into(),
            source_file: source.map(|path| path.to_string_lossy().into_owned()),
            ..FunctionBytecode::default()
        }
    }

    #[test]
    fn loose_private_observation_requires_exact_published_caller_directory() {
        let root = std::env::temp_dir().join("runmat-private-resolution-source-fixture");
        let mut functions = HashMap::new();
        for (id, name, path) in [
            (1, "left_entry", root.join("left/left_entry.m")),
            (2, "right_entry", root.join("right/right_entry.m")),
            (3, "other_entry", root.join("other/other_entry.m")),
            (4, "__private__.helper", root.join("left/private/helper.m")),
            (5, "__private__.helper", root.join("right/Private/helper.m")),
        ] {
            functions.insert(FunctionId(id), body(id, name, Some(&path)));
        }
        let registry = FunctionRegistry::new(functions);
        assert_eq!(
            registry.resolve_private_name_for_caller(FunctionId(1), "helper"),
            Some(FunctionId(4))
        );
        assert_eq!(
            registry.resolve_private_name_for_caller(FunctionId(2), "helper"),
            Some(FunctionId(5)),
            "a duplicate leaf in another source root has distinct authority"
        );
        assert_eq!(
            registry.resolve_private_name_for_caller(FunctionId(3), "helper"),
            None,
            "an unrelated source cannot observe a private helper"
        );
        assert_eq!(
            registry.resolve_private_name_for_caller(FunctionId(1), "pkg.helper"),
            None
        );
        assert_eq!(
            registry.resolve_private_name_for_caller(FunctionId(999), "helper"),
            None
        );
        let mut functions = registry.functions.clone();
        functions.insert(FunctionId(6), body(6, "missing_source", None));
        functions.insert(
            FunctionId(7),
            body(
                7,
                "__private__.helper",
                Some(&root.join("left/private/duplicate.m")),
            ),
        );
        let registry = FunctionRegistry::new(functions);
        assert_eq!(
            registry.resolve_private_name_for_caller(FunctionId(6), "helper"),
            None,
            "missing caller provenance grants no private visibility"
        );
        assert_eq!(
            registry.resolve_private_name_for_caller(FunctionId(1), "helper"),
            None,
            "ambiguous published private definitions are not guessed"
        );
    }

    #[test]
    fn retained_caller_source_controls_dynamic_private_name_not_reused_id() {
        let root = std::env::temp_dir().join("runmat-private-resolution-retained-fixture");
        let mut old_caller = body(1, "entry", Some(&root.join("old/entry.m")));
        old_caller.instructions = vec![Instr::CallSemanticFunctionMulti(FunctionId(2), 0, 1)];
        let old_helper = body(
            2,
            "__private__.helper",
            Some(&root.join("old/private/helper.m")),
        );
        let mut registry = FunctionRegistry::new(HashMap::from([
            (FunctionId(1), old_caller),
            (FunctionId(2), old_helper),
        ]));
        let owners = registry
            .definition_dependency_owners(FunctionId(1))
            .expect("old caller retains its private static dependency");
        let issuer = std::sync::Arc::new(());
        let authority = runmat_value::BoundFunctionAuthority::local(
            std::sync::Arc::downgrade(&issuer),
            runmat_value::FunctionDefinitionOwner::issue(&owners[0].1, 1),
            owners[1..]
                .iter()
                .map(|(id, owner)| runmat_value::FunctionDefinitionOwner::issue(owner, id.0))
                .collect(),
            runmat_value::LocalExport::Ephemeral,
        );
        registry.insert_replacing_name(body(1, "entry", Some(&root.join("new/entry.m"))));
        registry.insert_replacing_name(body(
            2,
            "__private__.helper",
            Some(&root.join("new/private/helper.m")),
        ));
        registry.insert_replacing_name(body(3, "helper", Some(&root.join("old/private/helper.m"))));
        let retained = registry
            .owned_view(FunctionId(1), &authority)
            .expect("old caller remains leased");
        assert_eq!(
            retained.get(FunctionId(1)).unwrap().source_file,
            Some(root.join("old/entry.m").to_string_lossy().into_owned())
        );
        assert_eq!(
            retained
                .current_view()
                .get(FunctionId(1))
                .unwrap()
                .source_file,
            Some(root.join("new/entry.m").to_string_lossy().into_owned())
        );
        assert_eq!(
            retained.resolve_private_name_for_caller(FunctionId(1), "helper"),
            Some(FunctionId(3)),
            "the old caller uses its captured directory but observes current published callees"
        );
        assert_eq!(
            retained
                .current_view()
                .resolve_private_name_for_caller(FunctionId(1), "helper"),
            Some(FunctionId(2)),
            "the republished caller uses its new captured directory"
        );
    }

    #[test]
    fn captured_owner_matches_distinct_published_access_spellings() {
        let root = std::env::temp_dir().join("runmat-private-resolution-alias-fixture");
        let mut caller = body(1, "entry", Some(&root.join("alias/entry.m")));
        caller.private_source_owner = Some(PrivateSourceDirectory::new(root.join("real")));
        let mut helper = body(
            2,
            "__private__.helper",
            Some(&root.join("real/private/helper.m")),
        );
        helper.private_source_owner = Some(PrivateSourceDirectory::new(root.join("real")));
        let registry = FunctionRegistry::new(HashMap::from([
            (FunctionId(1), caller),
            (FunctionId(2), helper),
        ]));
        assert_eq!(
            registry.resolve_private_name_for_caller(FunctionId(1), "helper"),
            Some(FunctionId(2)),
            "observation uses publication-time owner identity, not the access-path spelling"
        );
    }

    #[test]
    fn portable_relative_source_paths_recover_private_scope_without_host_key() {
        let aliases = vec![
            PrivateSourceAlias {
                owner_identity: "package".into(),
                relative_directory: "alias".into(),
            },
            PrivateSourceAlias {
                owner_identity: "package".into(),
                relative_directory: "real".into(),
            },
        ];
        let mut caller = body(1, "entry", Some(Path::new("alias/entry.m")));
        caller.private_source_owner = Some(PrivateSourceDirectory::captured(
            Some(std::env::temp_dir().join("host-only-owner")),
            aliases.clone(),
        ));
        let mut helper = body(
            2,
            "__private__.helper",
            Some(Path::new("real/private/helper.m")),
        );
        helper.private_source_owner = Some(PrivateSourceDirectory::captured(
            Some(std::env::temp_dir().join("host-only-owner")),
            aliases,
        ));
        let registry = FunctionRegistry::new(HashMap::from([
            (FunctionId(1), caller),
            (FunctionId(2), helper),
        ]));
        let encoded = serde_json::to_string(&registry).expect("portable registry payload");
        assert!(!encoded.contains("host-only-owner"));
        let decoded: FunctionRegistry = serde_json::from_str(&encoded).expect("registry reload");
        assert_eq!(
            decoded
                .get(FunctionId(1))
                .unwrap()
                .private_source_owner
                .as_ref()
                .unwrap()
                .aliases()
                .len(),
            2,
            "the portable alias partition survives without its host path"
        );
        assert_eq!(
            decoded.resolve_private_name_for_caller(FunctionId(1), "helper"),
            Some(FunctionId(2))
        );
    }
}
