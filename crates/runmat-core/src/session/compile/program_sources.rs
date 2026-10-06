use super::{
    executable_source_identity, source_context_function_names_in_statements,
    CompanionSourceDiscovery, SourceContextData,
};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use runmat_types::ProgramSourceId;

/// One source table for the immutable compiler product. These ordinals never
/// borrow authority from the session SourcePool or a runtime source lease.
pub(super) struct ProgramSourceTable {
    pub source_map: crate::ExecutableSourceMap,
    pub statement_sources: Vec<ProgramSourceId>,
    pub function_sources: HashMap<String, ProgramSourceId>,
}

struct CompanionSource {
    context: Arc<SourceContextData>,
    owner_identity: String,
    relative_path: String,
}

impl ProgramSourceTable {
    pub fn build(
        root: &crate::ExecutableSource,
        root_full_path: Option<&str>,
        discovery: &CompanionSourceDiscovery,
        primary_statement_count: usize,
        handoff: Option<&runmat_package::FrozenProjectHandoff>,
    ) -> Result<Self, String> {
        let mut companions = BTreeMap::<(String, String), CompanionSource>::new();
        for discovered in &discovery.statements {
            let Some(context) = &discovered.source else {
                return Err("companion statement is missing source provenance".into());
            };
            let (owner_identity, relative_path) = executable_source_identity(
                root,
                handoff,
                &context.display_name,
                context.fullpath_name.as_deref(),
            );
            if owner_identity == root.owner_identity && relative_path == root.relative_path {
                return Err("companion source resolves to the primary source".into());
            }
            let key = (owner_identity.clone(), relative_path.clone());
            if let Some(previous) = companions.get(&key) {
                if previous.context.text != context.text {
                    return Err("one companion source has conflicting discovered text".into());
                }
            } else {
                companions.insert(
                    key,
                    CompanionSource {
                        context: Arc::clone(context),
                        owner_identity,
                        relative_path,
                    },
                );
            }
        }

        let mut entries = Vec::with_capacity(companions.len() + 1);
        entries.push(crate::SourceMapEntry {
            source_id: 0,
            owner_identity: root.owner_identity.clone(),
            relative_path: root.relative_path.clone(),
            display_name: root.relative_path.clone(),
            full_path: root_full_path.map(ToString::to_string),
            text: root.text.clone(),
        });
        let mut ids = BTreeMap::new();
        for (key, source) in companions {
            let ordinal = u32::try_from(entries.len())
                .map_err(|_| "portable program has too many source units".to_string())?;
            ids.insert(key, ProgramSourceId(ordinal));
            entries.push(crate::SourceMapEntry {
                source_id: ordinal as usize,
                owner_identity: source.owner_identity,
                relative_path: source.relative_path,
                display_name: source.context.display_name.clone(),
                full_path: source.context.fullpath_name.clone(),
                text: source.context.text.clone(),
            });
        }

        let mut statement_sources = vec![ProgramSourceId(0); primary_statement_count];
        let mut function_sources = HashMap::new();
        for discovered in &discovery.statements {
            let context = discovered
                .source
                .as_ref()
                .expect("companion source was validated above");
            let key = executable_source_identity(
                root,
                handoff,
                &context.display_name,
                context.fullpath_name.as_deref(),
            );
            let source_id = ids[&key];
            statement_sources.push(source_id);
            for name in source_context_function_names_in_statements(std::slice::from_ref(
                &discovered.statement,
            )) {
                function_sources.insert(name, source_id);
            }
        }
        Ok(Self {
            source_map: crate::ExecutableSourceMap::new(entries),
            statement_sources,
            function_sources,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::compile::DiscoveredSourceStatement;

    fn discovered(text: &str, name: &str, display: &str) -> DiscoveredSourceStatement {
        let statement = runmat_parser::parse(text)
            .expect("source parses")
            .body
            .into_iter()
            .next()
            .expect("source declares one function");
        DiscoveredSourceStatement {
            statement,
            is_private: false,
            source: Some(Arc::new(SourceContextData {
                display_name: display.into(),
                fullpath_name: Some(format!("/project/{name}")),
                text: text.into(),
            })),
        }
    }

    #[test]
    fn companion_order_does_not_change_source_ordinals_or_map() {
        let root = crate::ExecutableSource::new("package", "main.m", "value = a() + z();");
        let a = discovered("function y = a(); y = 1; end", "a.m", "a.m");
        let z = discovered("function y = z(); y = 2; end", "z.m", "z.m");
        let one = CompanionSourceDiscovery {
            statements: vec![z, a],
            ..Default::default()
        };
        let first = ProgramSourceTable::build(&root, None, &one, 1, None).unwrap();
        assert_eq!(first.function_sources["a"], ProgramSourceId(1));
        assert_eq!(first.function_sources["z"], ProgramSourceId(2));
        assert_eq!(
            first.statement_sources,
            vec![ProgramSourceId(0), ProgramSourceId(2), ProgramSourceId(1)]
        );
        let two = CompanionSourceDiscovery {
            statements: vec![
                discovered("function y = a(); y = 1; end", "a.m", "a.m"),
                discovered("function y = z(); y = 2; end", "z.m", "z.m"),
            ],
            ..Default::default()
        };
        let second = ProgramSourceTable::build(&root, None, &two, 1, None).unwrap();
        assert_eq!(first.source_map, second.source_map);
        assert_eq!(second.statement_sources[0], ProgramSourceId(0));
    }

    #[test]
    fn one_canonical_source_cannot_publish_conflicting_text() {
        let root = crate::ExecutableSource::new("package", "main.m", "x = 1;");
        let discovery = CompanionSourceDiscovery {
            statements: vec![
                discovered("function y = a(); y = 1; end", "a.m", "a.m"),
                discovered("function y = a(); y = 2; end", "a.m", "a.m"),
            ],
            ..Default::default()
        };
        assert!(ProgramSourceTable::build(&root, None, &discovery, 1, None)
            .err()
            .expect("conflicting source must fail")
            .contains("conflicting discovered text"));
    }
}
