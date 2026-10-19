//! Exact issued-source identity for an eval compilation product.

use runmat_runtime::context::RuntimeContext;
use runmat_runtime::source_context::RuntimeSourceLease;
use runmat_types::{ProgramSourceId, SourceId};
use runmat_vm::FunctionRegistry;
use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::{ExecutableSource, ExecutableSourceMap, SourceMapEntry};

pub(super) struct DynamicSourceIdentity {
    pub(super) source: ExecutableSource,
    pub(super) map: ExecutableSourceMap,
}

#[derive(Clone, Debug)]
pub(super) struct ProjectSourceIdentity {
    owner_identity: String,
    relative_path: String,
    content_digest: runmat_package::ContentDigest,
}

/// A compact projection of a validated frozen handoff. The executable source
/// remains selected by exact access path and content, never a display name.
#[derive(Default, Debug)]
pub(super) struct ProjectSourceIdentityIndex {
    by_access_path: BTreeMap<PathBuf, Vec<ProjectSourceIdentity>>,
}

impl ProjectSourceIdentityIndex {
    pub(super) fn from_handoff(handoff: &runmat_package::FrozenProjectHandoff) -> Self {
        let mut by_access_path = BTreeMap::<PathBuf, Vec<ProjectSourceIdentity>>::new();
        for (source, access_path) in handoff.project.all_sources() {
            by_access_path
                .entry(access_path.clone())
                .or_default()
                .push(ProjectSourceIdentity {
                    owner_identity: source.id.package_instance.to_string(),
                    relative_path: source.id.relative_path.to_string(),
                    content_digest: source.id.content_digest.clone(),
                });
        }
        Self { by_access_path }
    }

    fn select(&self, path: &str, text: &str) -> Result<Option<(String, String)>, String> {
        let Some(candidates) = self.by_access_path.get(&PathBuf::from(path)) else {
            return Ok(None);
        };
        let digest = runmat_package::ContentDigest::sha256(text.as_bytes());
        let matches = candidates
            .iter()
            .filter(|candidate| candidate.content_digest == digest)
            .collect::<Vec<_>>();
        let [selected] = matches.as_slice() else {
            return Err("dynamic source does not match one exact frozen-project source".into());
        };
        Ok(Some((
            selected.owner_identity.clone(),
            selected.relative_path.clone(),
        )))
    }
}

/// A selected executable has already bound its primary and companion source
/// table into this invocation. Join a `run` file to that exact issued table;
/// neither its display path nor a later project discovery selects authority.
fn selected_companion_identity(
    runtime: &RuntimeContext,
    path: &str,
    text: &str,
) -> Result<Option<(String, String)>, String> {
    let Some(map) = runtime.portable_program_source_map() else {
        return Ok(None);
    };
    let owner = runmat_runtime::source_context::SelectedProgramSourceOwner::capture(runtime, map)
        .map_err(|_| "selected companion source table has no exact issuing owner")?;
    let candidates = map
        .entries
        .iter()
        .filter(|record| {
            owner
                .issued_source_info_for(record.source_id)
                .and_then(|info| info.fullpath_name.as_deref())
                == Some(path)
        })
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        return Ok(None);
    }
    let [selected] = candidates.as_slice() else {
        return Err("run source has ambiguous selected companion records".into());
    };
    if selected.text != text
        || selected.content_revision != runmat_execution::Digest::sha256(text.as_bytes())
    {
        return Err("run source differs from its selected companion record".into());
    }
    Ok(Some((
        selected.owner_identity.clone(),
        selected.relative_path.clone(),
    )))
}

/// The lexical caller, when present, is selected by the VM frame's typed
/// source ID in the *already bound* program source table. Its authenticated
/// package/directory fields may scope a new eval body; its display path alone
/// may not. A separately issued run/session source keeps its own identity.
pub(super) fn selected_dynamic_source_identity(
    runtime: &RuntimeContext,
    lease: &RuntimeSourceLease,
    caller_source: Option<SourceId>,
    inherited_registry: &FunctionRegistry,
    project_sources: Option<&ProjectSourceIdentityIndex>,
) -> Result<DynamicSourceIdentity, String> {
    let issued = lease.source_info();
    if lease.source_ref().resolve_in(runtime) != Some(lease.source_id()) {
        return Err("dynamic source lease is not issued in this runtime".into());
    }

    let inherited = if issued.fullpath_name.is_none() {
        caller_source
            .map(|source| {
                let map = runtime
                    .portable_program_source_map()
                    .ok_or("lexical caller has no selected portable source table")?;
                let matches = map
                    .entries
                    .iter()
                    .filter(|record| {
                        runtime
                            .program_source_ref(record.source_id)
                            .and_then(|reference| reference.resolve_in(runtime))
                            == Some(source)
                    })
                    .collect::<Vec<_>>();
                let [record] = matches.as_slice() else {
                    return Err("lexical caller has no unique selected source record");
                };
                let owner = runmat_runtime::source_context::SelectedProgramSourceOwner::capture(
                    runtime, map,
                )
                .map_err(|_| "lexical caller has no exact issuing source owner")?;
                if !inherited_registry.selected_source_issuance_agrees_in(runtime, source, &owner) {
                    return Err("lexical caller body disagrees with selected source issuance");
                }
                Ok((record.owner_identity.clone(), record.relative_path.clone()))
            })
            .transpose()?
    } else {
        issued.fullpath_name.as_deref().map_or(Ok(None), |path| {
            // An already-bound unit precedes any later project handoff.
            // A file outside that unit may still use this port's frozen
            // handoff, but cannot replace the unit's selected source.
            let selected = selected_companion_identity(runtime, path, &issued.text)?;
            match selected {
                Some(selected) => Ok(Some(selected)),
                None => {
                    project_sources.map_or(Ok(None), |sources| sources.select(path, &issued.text))
                }
            }
        })?
    };

    let content_revision = runmat_execution::Digest::sha256(issued.text.as_bytes());
    let leaf = format!(
        "issued-{}.m",
        content_revision.to_string().replace(':', "-")
    );
    let (owner_identity, relative_path) = match inherited {
        Some((owner, caller_path)) => {
            let directory = std::path::Path::new(&caller_path)
                .parent()
                .unwrap_or(std::path::Path::new(""));
            (
                owner,
                directory.join(leaf).to_string_lossy().replace('\\', "/"),
            )
        }
        None => (
            "runmat-issued-dynamic-source".into(),
            format!("dynamic/{leaf}"),
        ),
    };
    let text = issued.text.to_string();
    let source = ExecutableSource::new(&owner_identity, &relative_path, &text);
    let map = ExecutableSourceMap::new(vec![SourceMapEntry {
        source_id: ProgramSourceId(0),
        owner_identity,
        relative_path,
        content_revision,
        display_name: issued.name.to_string(),
        full_path: issued.fullpath_name.as_deref().map(str::to_owned),
        text,
    }]);
    Ok(DynamicSourceIdentity { source, map })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;
    use std::sync::Arc;

    use runmat_execution::{Digest, ExecutableSourceMapV2, ExecutableSourceRecordV2};

    #[test]
    fn project_source_requires_exact_access_path_and_frozen_content() {
        let path = PathBuf::from("/selected/tool.m");
        let project_sources = ProjectSourceIdentityIndex {
            by_access_path: BTreeMap::from([(
                path,
                vec![ProjectSourceIdentity {
                    owner_identity: "frozen-package".into(),
                    relative_path: "lib/tool.m".into(),
                    content_digest: runmat_package::ContentDigest::sha256(b"x = 1;"),
                }],
            )]),
        };
        assert_eq!(
            project_sources
                .select("/selected/tool.m", "x = 1;")
                .unwrap(),
            Some(("frozen-package".into(), "lib/tool.m".into()))
        );
        assert!(project_sources
            .select("/selected/tool.m", "x = 2;")
            .is_err());
        assert_eq!(
            project_sources.select("/other/tool.m", "x = 1;").unwrap(),
            None
        );
        let runtime = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let lease = runmat_runtime::source_context::lease_source_context_in_context(
            &runtime,
            "tool.m",
            Some("/selected/tool.m".into()),
            "x = 1;",
        )
        .expect("issue exact project source");
        let selected = selected_dynamic_source_identity(
            &runtime,
            &lease,
            None,
            &FunctionRegistry::default(),
            Some(&project_sources),
        )
        .expect("select issued project source");
        assert_eq!(selected.source.owner_identity, "frozen-package");
        assert!(selected
            .source
            .relative_path
            .starts_with("lib/issued-sha256-"));
    }

    #[test]
    fn dynamic_eval_inherits_only_a_selected_caller_directory() {
        let map = ExecutableSourceMapV2::new(vec![ExecutableSourceRecordV2 {
            source_id: ProgramSourceId(0),
            owner_identity: "captured-package".into(),
            relative_path: "lib/caller.m".into(),
            display_name: "caller.m".into(),
            text: "eval('x = 1;')".into(),
            content_revision: Digest::sha256(b"eval('x = 1;')"),
        }])
        .expect("valid selected source map");
        let runtime = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let runtime = runmat_runtime::source_context::bind_portable_source_map_v2(runtime, &map)
            .expect("bind exact selected caller");
        let caller = runtime
            .program_source_id(ProgramSourceId(0))
            .expect("selected caller source");
        let lease = runmat_runtime::source_context::lease_source_context_in_context(
            &runtime, "<eval>", None, "x = 1;",
        )
        .expect("issue dynamic source");
        let selected = selected_dynamic_source_identity(
            &runtime,
            &lease,
            Some(caller),
            &FunctionRegistry::default(),
            None,
        )
        .expect("derive identity from selected caller");
        assert_eq!(selected.source.owner_identity, "captured-package");
        assert!(selected
            .source
            .relative_path
            .starts_with("lib/issued-sha256-"));
        assert_eq!(selected.map.entries()[0].display_name, "<eval>");

        let foreign = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        assert!(selected_dynamic_source_identity(
            &foreign,
            &lease,
            Some(caller),
            &FunctionRegistry::default(),
            None,
        )
        .is_err());
    }

    #[test]
    fn run_source_uses_the_invocations_exact_companion_record() {
        let base = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let lease = runmat_runtime::source_context::lease_source_context_in_context(
            &base,
            "tool.m",
            Some("/selected/tool.m".into()),
            "captured = 40;",
        )
        .expect("issue selected run source");
        let bind = |owner: &str, text: &str| {
            let map = ExecutableSourceMapV2::new(vec![ExecutableSourceRecordV2 {
                source_id: ProgramSourceId(0),
                owner_identity: owner.into(),
                relative_path: "lib/tool.m".into(),
                display_name: "tool.m".into(),
                text: text.into(),
                content_revision: Digest::sha256(text.as_bytes()),
            }])
            .expect("valid selected map");
            runmat_runtime::source_context::bind_portable_source_map_v2_with_records(
                base.clone(),
                &map,
                [runmat_runtime::source_context::ProgramSourceRecord {
                    ordinal: ProgramSourceId(0),
                    name: Arc::from("tool.m"),
                    fullpath_name: Some(Arc::from("/selected/tool.m")),
                    text: Arc::from(text),
                }],
            )
            .expect("bind selected companion source")
        };
        let old = bind("selected-old", "captured = 40;");
        let newer = bind("selected-new", "captured = 90;");
        let later_handoff = ProjectSourceIdentityIndex {
            by_access_path: BTreeMap::from([(
                PathBuf::from("/selected/tool.m"),
                vec![ProjectSourceIdentity {
                    owner_identity: "later-handoff".into(),
                    relative_path: "new/tool.m".into(),
                    content_digest: runmat_package::ContentDigest::sha256(b"captured = 40;"),
                }],
            )]),
        };
        let selected = selected_dynamic_source_identity(
            &old,
            &lease,
            None,
            &FunctionRegistry::default(),
            Some(&later_handoff),
        )
        .expect("old invocation retains its exact selected catalog");
        assert_eq!(selected.source.owner_identity, "selected-old");
        assert!(selected
            .source
            .relative_path
            .starts_with("lib/issued-sha256-"));
        assert!(selected_dynamic_source_identity(
            &newer,
            &lease,
            None,
            &FunctionRegistry::default(),
            None,
        )
        .is_err());

        let ambiguous = ExecutableSourceMapV2::new(
            ["first", "second"]
                .into_iter()
                .enumerate()
                .map(|(ordinal, owner)| ExecutableSourceRecordV2 {
                    source_id: ProgramSourceId(ordinal as u32),
                    owner_identity: owner.into(),
                    relative_path: format!("{owner}/tool.m"),
                    display_name: "tool.m".into(),
                    text: "captured = 40;".into(),
                    content_revision: Digest::sha256(b"captured = 40;"),
                })
                .collect(),
        )
        .expect("valid distinct portable sources");
        let ambiguous_runtime =
            runmat_runtime::source_context::bind_portable_source_map_v2_with_records(
                base,
                &ambiguous,
                (0..2).map(
                    |ordinal| runmat_runtime::source_context::ProgramSourceRecord {
                        ordinal: ProgramSourceId(ordinal),
                        name: Arc::from("tool.m"),
                        fullpath_name: Some(Arc::from("/selected/tool.m")),
                        text: Arc::from("captured = 40;"),
                    },
                ),
            )
            .expect("bind ambiguous local access path");
        assert!(selected_dynamic_source_identity(
            &ambiguous_runtime,
            &lease,
            None,
            &FunctionRegistry::default(),
            None,
        )
        .is_err());
    }
}
