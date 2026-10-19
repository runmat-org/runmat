//! Private-source owner identity captured from one authenticated source map.
use std::path::{Path, PathBuf};

use runmat_vm::PrivateSourceDirectory;

pub(super) struct CapturedPrivateSourceOwners(
    Vec<(runmat_types::ProgramSourceId, PrivateSourceDirectory)>,
);

#[derive(Clone, Eq, Hash, PartialEq)]
enum OwnerGroup {
    Host(PathBuf),
    Logical(runmat_vm::PrivateSourceAlias),
}

impl CapturedPrivateSourceOwners {
    pub(super) fn new(source_map: &crate::ExecutableSourceMap) -> Self {
        let entries = source_map.entries();
        let hosts = entries
            .iter()
            .map(|entry| entry.full_path.as_deref().and_then(captured_host_owner))
            .collect::<Vec<_>>();
        let logical = entries
            .iter()
            .map(|entry| {
                runmat_vm::PrivateSourceAlias::from_source_path(
                    &entry.owner_identity,
                    &entry.relative_path,
                )
            })
            .collect::<Vec<_>>();
        let groups = hosts
            .iter()
            .zip(&logical)
            .map(|(host, alias)| match host {
                Some(path) => OwnerGroup::Host(path.clone()),
                None => OwnerGroup::Logical(alias.clone()),
            })
            .collect::<Vec<_>>();
        let mut aliases_by_group = std::collections::HashMap::<OwnerGroup, Vec<_>>::new();
        for (group, alias) in groups.iter().zip(&logical) {
            aliases_by_group
                .entry(group.clone())
                .or_default()
                .push(alias.clone());
        }
        let owners = groups
            .iter()
            .enumerate()
            .map(|(index, group)| {
                (
                    entries[index].source_id,
                    PrivateSourceDirectory::captured(
                        hosts[index].clone(),
                        aliases_by_group.get(group).cloned().unwrap_or_default(),
                    ),
                )
            })
            .collect();
        Self(owners)
    }

    pub(super) fn get(
        &self,
        source: runmat_vm::BytecodeSourceRef,
    ) -> Option<PrivateSourceDirectory> {
        let ordinal = source.program()?;
        let (selected, owner) = self.0.get(usize::try_from(ordinal.0).ok()?)?;
        (*selected == ordinal).then(|| owner.clone())
    }
}

fn captured_host_owner(source_file: &str) -> Option<PathBuf> {
    let source_directory = Path::new(source_file).parent()?;
    let owner = if source_directory
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.eq_ignore_ascii_case("private"))
    {
        source_directory.parent()?
    } else {
        source_directory
    };
    // Native source admission recognizes symlink-equivalent directories.
    // Capture that equivalence once; browser virtual paths have only logical
    // source-map aliases and never trigger a filesystem operation.
    #[cfg(not(target_arch = "wasm32"))]
    if let Ok(canonical) = std::fs::canonicalize(owner) {
        return Some(canonical);
    }
    Some(owner.to_path_buf())
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn publication_captures_symlink_equivalent_portable_aliases() {
        let temp = tempfile::TempDir::new().expect("fixture root");
        let real = temp.path().join("real");
        std::fs::create_dir_all(real.join("private")).expect("private directory");
        let alias = temp.path().join("alias");
        std::os::unix::fs::symlink(&real, &alias).expect("symlink fixture");
        let source = |id, relative: &str, full: String| crate::SourceMapEntry {
            source_id: runmat_types::ProgramSourceId(id),
            owner_identity: "package".into(),
            relative_path: relative.into(),
            content_revision: runmat_execution::Digest::sha256(b"function y = f, y = 1; end"),
            display_name: relative.into(),
            full_path: Some(full),
            text: "function y = f, y = 1; end".into(),
        };
        let source_map = crate::ExecutableSourceMap::new(vec![
            source(
                0,
                "alias/entry.m",
                alias.join("entry.m").to_string_lossy().into_owned(),
            ),
            source(
                1,
                "real/private/helper.m",
                real.join("private/helper.m").to_string_lossy().into_owned(),
            ),
        ]);
        let captured = CapturedPrivateSourceOwners::new(&source_map);
        let first = captured
            .get(runmat_vm::BytecodeSourceRef::Program(
                runmat_types::ProgramSourceId(0),
            ))
            .unwrap();
        let second = captured
            .get(runmat_vm::BytecodeSourceRef::Program(
                runmat_types::ProgramSourceId(1),
            ))
            .unwrap();
        assert_eq!(first.aliases(), second.aliases());
        assert_eq!(first.aliases().len(), 2);
        assert!(first.same_owner(&second));
        let portable_first: PrivateSourceDirectory =
            serde_json::from_str(&serde_json::to_string(&first).unwrap()).unwrap();
        let portable_second: PrivateSourceDirectory =
            serde_json::from_str(&serde_json::to_string(&second).unwrap()).unwrap();
        assert!(portable_first.same_owner(&portable_second));
    }
}
