//! The frozen test run may select source files outside package source roots.
//! This issued overlay binds their exact bytes to the installed project graph
//! without pretending that they are package-owned sources.

use super::{
    is_class_source_body, is_function_source_body, qualify_companion_classdefs,
    qualify_companion_functions, CompanionSourceDiscovery, SourceContextData,
};
use crate::ExecutableSource;
use runmat_test::descriptor::{SourceDescriptor, SourceSpan};
use runmat_test::discovery::{FrozenTestRunSnapshot, RunSourceOrigin};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;

pub(in crate::session) struct CapturedTestSourceAuthority<'a> {
    snapshot: &'a FrozenTestRunSnapshot,
    primary_owner: &'a str,
    primary_relative: &'a str,
    primary_access: Option<String>,
    project_paths: HashSet<String>,
    owner_roots: BTreeMap<String, PathBuf>,
    captured_access: HashMap<String, usize>,
    selected_section: Option<SourceSpan>,
}

impl<'a> CapturedTestSourceAuthority<'a> {
    pub(super) fn new(
        snapshot: &'a FrozenTestRunSnapshot,
        descriptor: &'a SourceDescriptor,
        selected: &ExecutableSource,
        span_only: bool,
        handoff: Option<&runmat_package::FrozenProjectHandoff>,
    ) -> Result<Self, String> {
        snapshot.validate().map_err(|error| error.to_string())?;
        if let Some(handoff) = handoff {
            let installed = handoff.revision();
            if snapshot.program_revision.graph_digest().bytes() != installed.graph_digest.bytes()
                || snapshot.base_source_digest != installed.source_revision.to_string()
            {
                return Err("test source snapshot differs from installed project revision".into());
            }
        }
        if descriptor.owner_identity != selected.owner_identity
            || descriptor.relative_path != selected.relative_path
        {
            return Err("selected test source differs from its frozen descriptor".into());
        }
        let source = snapshot
            .sources
            .iter()
            .find(|source| {
                source.owner_identity == descriptor.owner_identity
                    && source.relative_path == descriptor.relative_path
            })
            .ok_or("selected test source is absent from its frozen run")?;
        if source.content != selected.text {
            return Err("selected test bytes differ from their frozen source".into());
        }
        if span_only
            && source
                .content
                .get(descriptor.span.start_byte as usize..descriptor.span.end_byte as usize)
                .is_none()
        {
            return Err("selected test span is not a frozen UTF-8 source range".into());
        }
        let workspace_root = handoff
            .map(|handoff| handoff.project.workspace_root.clone())
            .unwrap_or_else(|| PathBuf::from("__runmat_frozen_test_source__"));
        let owner_roots = {
            let root = &workspace_root;
            let mut owners = BTreeMap::new();
            for source in &snapshot.sources {
                if !owners.contains_key(&source.owner_identity) {
                    let owner_root = if source.owner_identity == "path:workspace" {
                        root.clone()
                    } else {
                        root.join(".runmat-captured-tests")
                            .join(owners.len().to_string())
                    };
                    owners.insert(source.owner_identity.clone(), owner_root);
                }
            }
            owners
        };
        let primary_access = owner_roots
            .get(&source.owner_identity)
            .map(|root| access_path(root, &source.relative_path));
        let mut project_sources = HashMap::new();
        for (descriptor, path) in handoff
            .into_iter()
            .flat_map(|handoff| handoff.project.all_sources())
        {
            if project_sources
                .insert(
                    path.to_string_lossy().replace('\\', "/"),
                    *descriptor.id.content_digest.bytes(),
                )
                .is_some()
            {
                return Err("captured project has ambiguous source access paths".into());
            }
        }
        let project_paths = project_sources.keys().cloned().collect::<HashSet<_>>();
        let mut captured_access = HashMap::with_capacity(snapshot.sources.len());
        for (index, source) in snapshot.sources.iter().enumerate() {
            let root = owner_roots
                .get(&source.owner_identity)
                .ok_or("frozen test source has no issued access root")?;
            if captured_access
                .insert(access_path(root, &source.relative_path), index)
                .is_some()
            {
                return Err("distinct frozen test sources share one issued access path".into());
            }
        }
        for source in &snapshot.sources {
            if source.origin != RunSourceOrigin::Saved {
                continue;
            }
            let Some(root) = owner_roots.get(&source.owner_identity) else {
                continue;
            };
            let full = access_path(root, &source.relative_path);
            if let Some(expected) = project_sources.get(&full) {
                if expected != runmat_execution::Digest::sha256(source.content.as_bytes()).bytes() {
                    return Err(
                        "frozen saved test companion differs from captured project source".into(),
                    );
                }
            }
        }
        if snapshot.sources.iter().any(|source| {
            source.owner_identity != "path:workspace"
                && owner_roots.get(&source.owner_identity).is_some_and(|root| {
                    project_paths.contains(&access_path(root, &source.relative_path))
                })
        }) {
            return Err("issued test access path collides with captured project source".into());
        }
        Ok(Self {
            snapshot,
            primary_owner: &source.owner_identity,
            primary_relative: &source.relative_path,
            primary_access,
            project_paths,
            owner_roots,
            captured_access,
            selected_section: span_only.then_some(descriptor.span),
        })
    }

    /// Keep the original primary file and its local definitions in the source
    /// table; select executable section statements from the parsed program.
    pub(super) fn select_program(
        &self,
        program: &mut runmat_parser::Program,
        source_len: usize,
    ) -> Result<(), String> {
        let Some(selected) = &self.selected_section else {
            return Ok(());
        };
        let start = selected.start_byte as usize;
        let end = selected.end_byte as usize;
        let authentic = if program.sections.is_empty() {
            start == 0 && end == source_len
        } else {
            program
                .sections
                .iter()
                .any(|section| section.body_span.start == start && section.body_span.end == end)
        };
        if !authentic {
            return Err("selected test section differs from parsed frozen source".into());
        }
        program.body.retain(|statement| {
            let span = statement.span();
            matches!(statement, runmat_parser::Stmt::Function { .. })
                || (span.start >= start && span.end <= end)
        });
        program
            .sections
            .retain(|section| section.body_span.start == start && section.body_span.end == end);
        Ok(())
    }

    pub(super) fn primary_access_path(&self) -> Option<&str> {
        self.primary_access.as_deref()
    }

    pub(super) fn executable_selection(
        &self,
    ) -> Option<crate::executable::ExecutableSourceSelection> {
        self.selected_section.map(|span| {
            crate::executable::ExecutableSourceSelection::ScriptSection {
                start_byte: span.start_byte,
                end_byte: span.end_byte,
            }
        })
    }

    pub(super) fn captured_text_for(&self, path: &std::path::Path) -> Option<&str> {
        let access = path.to_string_lossy().replace('\\', "/");
        self.captured_access
            .get(&access)
            .and_then(|index| self.snapshot.sources.get(*index))
            .map(|source| source.content.as_str())
    }

    pub(super) fn source_identity(
        &self,
        full_path: &str,
        text: &str,
    ) -> Result<Option<(String, String)>, String> {
        if self.project_paths.contains(full_path) {
            // The installed graph owns this location. Captured test bytes may
            // overlay its body, but cannot mint a second package identity.
            return Ok(None);
        }
        let matching = self
            .captured_access
            .get(full_path)
            .and_then(|index| self.snapshot.sources.get(*index));
        let Some(source) = matching else {
            return Ok(None);
        };
        if source.content != text {
            return Err("test companion bytes differ from their frozen source".into());
        }
        Ok(Some((
            source.owner_identity.clone(),
            source.relative_path.clone(),
        )))
    }

    pub(super) fn add_companions(
        &self,
        discovered: &mut CompanionSourceDiscovery,
        compat: runmat_parser::CompatMode,
    ) -> Result<(), String> {
        for source in &self.snapshot.sources {
            if source.owner_identity == self.primary_owner
                && source.relative_path == self.primary_relative
            {
                continue;
            }
            let root = self
                .owner_roots
                .get(&source.owner_identity)
                .ok_or("frozen test companion has no issued source root")?;
            let full_path = access_path(root, &source.relative_path);
            if self.project_paths.contains(&full_path) {
                // Frozen graph composition read these bytes from the captured
                // snapshot through captured_text_for, retaining package scope.
                continue;
            }
            discovered.statements.retain(|statement| {
                statement
                    .source
                    .as_ref()
                    .and_then(|context| context.fullpath_name.as_deref())
                    != Some(full_path.as_str())
            });
            discovered.observed_sources.retain(|observed| {
                observed.access_path.to_string_lossy().replace('\\', "/") != full_path
            });
            if !source.content.contains("function") && !source.content.contains("classdef") {
                continue;
            }
            let program = runmat_parser::parse_with_options(
                &source.content,
                runmat_parser::ParserOptions::new(compat),
            )
            .map_err(|error| format!("frozen test companion {}: {error}", source.relative_path))?;
            let is_class_source = is_class_source_body(&program.body);
            if !is_function_source_body(&program.body) && !is_class_source {
                continue;
            }
            let mut body = program.body;
            if let Some(qualified) = captured_qualified_name(&source.relative_path) {
                if is_class_source {
                    qualify_companion_classdefs(&mut body, &qualified);
                } else {
                    qualify_companion_functions(&mut body, &qualified);
                }
            }
            discovered.extend_body(
                body,
                None,
                Default::default(),
                Some(SourceContextData {
                    display_name: source.relative_path.clone(),
                    fullpath_name: Some(full_path),
                    text: source.content.clone(),
                }),
            );
        }
        Ok(())
    }
}

/// A frozen buffer can be outside the installed source graph, but its
/// `+package`/`@class` directory suffix still supplies language scope. Plain
/// containing directories such as `tests/` are not package components.
fn captured_qualified_name(relative: &str) -> Option<String> {
    let path = std::path::Path::new(relative);
    let mut scope = Vec::new();
    let mut started = false;
    for component in path.parent()?.components() {
        let segment = component.as_os_str().to_str()?;
        if let Some(name) = segment
            .strip_prefix('+')
            .or_else(|| segment.strip_prefix('@'))
        {
            if name.is_empty() {
                return None;
            }
            scope.push(name.to_owned());
            started = true;
        } else if started {
            return None;
        }
    }
    if scope.is_empty() {
        return None;
    }
    scope.push(path.file_stem()?.to_str()?.to_owned());
    Some(scope.join("."))
}

fn access_path(root: &std::path::Path, relative: &str) -> String {
    root.join(relative).to_string_lossy().replace('\\', "/")
}

#[cfg(test)]
mod tests {
    use super::captured_qualified_name;

    #[test]
    fn frozen_companion_scope_ignores_plain_containing_directories() {
        assert_eq!(
            captured_qualified_name("tests/+signals/@Filter/step.m"),
            Some("signals.Filter.step".into())
        );
        assert_eq!(captured_qualified_name("tests/helper.m"), None);
    }
}
