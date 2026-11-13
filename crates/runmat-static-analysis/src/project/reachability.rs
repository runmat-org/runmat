use std::collections::HashSet;
use std::path::{Component, Path, PathBuf};

use runmat_hir::{CallSyntax, HirCallableRef};
use runmat_package::{DiscoveredSourceSymbols, ProjectSymbolDefinition};

use super::ProjectSource;

/// Pure lexical path equivalence; adapters retain filesystem/symlink authority.
/// In particular, never collapse a parent component through a possible symlink.
pub(super) fn path_key(path: &Path) -> PathBuf {
    let mut key = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            other => key.push(other.as_os_str()),
        }
    }
    key
}

pub(super) fn private_visible(caller: &Path, definition: &Path) -> bool {
    let owner = definition.parent().and_then(Path::parent).map(path_key);
    owner.is_some() && caller.parent().map(path_key) == owner
}

pub(super) fn pending_sources(
    lowering: &runmat_hir::LoweringResult,
    mir: &runmat_mir::MirAssembly,
    sources: &[ProjectSource],
    catalog: Option<&DiscoveredSourceSymbols>,
    seen: &HashSet<PathBuf>,
    incomplete: &mut Vec<PathBuf>,
) -> Vec<ProjectSymbolDefinition> {
    let Some(catalog) = catalog else {
        return Vec::new();
    };
    let mut pending = Vec::new();
    let aliases = super::composition::project_symbol_aliases(Some(catalog));
    for call in &lowering.hir_index.calls {
        let bound_elsewhere = matches!(&call.callee,
        HirCallableRef::Function(function) if mir.functions.get(function).is_some_and(|metadata| {
            Some(metadata.source) != call.source
        }));
        let name = match &call.callee {
            HirCallableRef::Imported(path) => path.module.display_name(),
            HirCallableRef::Unresolved(name)
                if matches!(call.syntax, CallSyntax::Plain | CallSyntax::Command) =>
            {
                name.display_name()
            }
            HirCallableRef::Function(_) if bound_elsewhere => call.name.display_name(),
            // A bound/local/builtin target already has its canonical identity;
            // a receiver-dependent method is not a statically selected file.
            _ => None,
        };
        let Some(name) = name else { continue };
        let source = call.source.map_or(0, |source| source.0 as usize);
        let Some(caller) = sources.get(source) else {
            continue;
        };
        // One physical source can have multiple dependency-alias views. A
        // private lookup may use catalog ownership only when every effective
        // composed view agrees; catalog iteration order is never authority.
        let owners = catalog
            .definitions
            .iter()
            .filter(|definition| path_key(&definition.source_path) == path_key(&caller.path))
            .map(|definition| {
                super::composition::function_owner_scope_from_qualified_name(
                    aliases
                        .get(&definition.name)
                        .unwrap_or(&definition.qualified_name),
                )
            })
            .collect::<HashSet<_>>();
        if owners.len() > 1 {
            if !incomplete.contains(&caller.path) {
                incomplete.push(caller.path.clone());
            }
            continue;
        }
        let private_qualified_name = (!name.contains('.'))
            .then(|| {
                owners
                    .into_iter()
                    .next()
                    .filter(|owner| !owner.is_empty())
                    .map(|owner| format!("{owner}.{name}"))
            })
            .flatten();
        let prior_environment_change = mir.bodies.iter().any(|(function, body)| {
            mir.functions.get(function).is_some_and(|metadata| {
                metadata.source == call.source.unwrap_or(runmat_types::ProgramSourceId(0))
            }) && body
                .blocks
                .iter()
                .flat_map(|block| &block.statements)
                .any(|statement| {
                    statement.span.start < call.span.start
                        && matches!(
                            statement.kind,
                            runmat_mir::MirStmtKind::EnvironmentEffect(_)
                        )
                })
        });
        if prior_environment_change {
            continue;
        }
        let candidates = catalog
            .definitions
            .iter()
            .filter(|definition| {
                (definition.name == name
                    || (definition.is_private
                        && private_qualified_name.as_ref()
                            == Some(
                                aliases
                                    .get(&definition.name)
                                    .unwrap_or(&definition.qualified_name),
                            )))
                    && (!bound_elsewhere || definition.is_private)
                    && (!definition.is_private
                        || private_visible(&caller.path, &definition.source_path))
            })
            .collect::<Vec<_>>();
        let has_private = candidates.iter().any(|definition| definition.is_private);
        let mut selected: Option<&ProjectSymbolDefinition> = None;
        let mut ambiguous = false;
        for definition in candidates
            .into_iter()
            .filter(|definition| !has_private || definition.is_private)
        {
            if let Some(previous) = selected {
                if path_key(&previous.source_path) != path_key(&definition.source_path) {
                    ambiguous = true;
                    for path in [&previous.source_path, &definition.source_path] {
                        if !incomplete.contains(path) {
                            incomplete.push(path.clone());
                        }
                    }
                    break;
                }
            }
            selected = Some(definition);
        }
        if let Some(definition) = selected.filter(|_| !ambiguous) {
            let key = path_key(&definition.source_path);
            if !seen.contains(&key)
                && !pending
                    .iter()
                    .any(|queued: &ProjectSymbolDefinition| path_key(&queued.source_path) == key)
            {
                pending.push(definition.clone());
            }
        }
    }
    pending
}
