//! Pure source qualification shared by project checking and execution composition.
//! Filesystem/revision admission and executable source tables remain adapter-owned.
use std::collections::HashMap;

pub fn is_class_source_body(stmts: &[runmat_parser::Stmt]) -> bool {
    let has_classdef = stmts
        .iter()
        .any(|stmt| matches!(stmt, runmat_parser::Stmt::ClassDef { .. }));
    if !has_classdef {
        return false;
    }
    stmts.iter().all(|stmt| {
        matches!(
            stmt,
            runmat_parser::Stmt::ClassDef { .. } | runmat_parser::Stmt::Function { .. }
        )
    })
}

pub fn is_function_source_body(stmts: &[runmat_parser::Stmt]) -> bool {
    !stmts.is_empty()
        && stmts
            .iter()
            .all(|stmt| matches!(stmt, runmat_parser::Stmt::Function { .. }))
}

pub fn qualify_companion_classdefs(stmts: &mut [runmat_parser::Stmt], qualified_name: &str) {
    for stmt in stmts {
        if let runmat_parser::Stmt::ClassDef { name, .. } = stmt {
            if !name.contains('.') {
                *name = qualified_name.to_string();
            }
        }
    }
}

pub fn qualify_companion_functions(stmts: &mut [runmat_parser::Stmt], qualified_name: &str) {
    for stmt in stmts {
        if let runmat_parser::Stmt::Function { name, .. } = stmt {
            if !name.contains('.') {
                *name = qualified_name.to_string();
            }
        }
    }
}

pub fn function_owner_scope_from_qualified_name(qualified_name: &str) -> String {
    qualified_name
        .rsplit_once('.')
        .map(|(owner, _)| owner.to_string())
        .unwrap_or_default()
}

pub fn function_leaf_name(name: &str) -> &str {
    name.rsplit_once('.').map(|(_, leaf)| leaf).unwrap_or(name)
}

pub fn synthetic_private_function_name(owner_scope: &str, leaf_name: &str) -> String {
    if owner_scope.is_empty() {
        format!("__private__.{leaf_name}")
    } else {
        format!("{owner_scope}.__private__.{leaf_name}")
    }
}

pub fn qualify_private_companion_functions(
    stmts: &mut [runmat_parser::Stmt],
    owner_scope: &str,
    primary_visible: bool,
) -> HashMap<String, String> {
    let mut aliases = HashMap::new();
    for stmt in stmts {
        if let runmat_parser::Stmt::Function { name, .. } = stmt {
            let leaf = function_leaf_name(name).to_string();
            let display_name = if primary_visible {
                leaf.clone()
            } else {
                synthetic_private_function_name(owner_scope, &leaf)
            };
            *name = display_name.clone();
            aliases.insert(leaf, display_name);
        }
    }
    aliases
}

pub fn project_symbol_aliases(
    catalog: Option<&runmat_package::DiscoveredSourceSymbols>,
) -> HashMap<String, String> {
    let Some(catalog) = catalog else {
        return HashMap::new();
    };
    catalog
        .definitions
        .iter()
        .filter_map(|definition| {
            let alias = definition.dependency_alias.as_deref()?;
            let original = definition.name.strip_prefix(&format!("{alias}."))?;
            let has_unique_bare_name = catalog.definitions.iter().any(|candidate| {
                candidate.name == original
                    && candidate.dependency_alias.as_deref() == Some(alias)
                    && candidate.source_id == definition.source_id
            });
            Some((
                definition.name.clone(),
                if has_unique_bare_name {
                    original.to_string()
                } else {
                    definition.name.clone()
                },
            ))
        })
        .collect()
}
