use runmat_test::descriptor::{
    FixtureDescriptor, FixtureScope, ProcedureDescriptor, ProcedureKind,
};
use runmat_test::discovery::{DiscoveredSuite, TestDiscovery};
use runmat_test::identity::{FixtureGroupId, FixtureId, SuiteId};

use crate::{DefPath, DefPathSegment, FunctionKind, PackageName, QualifiedName, Span, SymbolName};

use super::{def_path_string, descriptor, function_leaf};
use crate::testing::source::{is_test_name, source_descriptor, source_stem};
use crate::testing::SemanticDiscoveryInput;

pub(super) fn discover(
    input: &SemanticDiscoveryInput<'_>,
    revision_identity: &str,
    discovery: &mut TestDiscovery,
) {
    for source in input.sources {
        let stem = source_stem(source.relative_source_identity);
        if !is_test_name(&stem) {
            continue;
        }
        let Some(module) = source.assembly.modules.first() else {
            continue;
        };
        if !module.script_sections.is_empty() || module.synthetic_entry_function.is_some() {
            let semantic_suite = format!(
                "{}/{}#script",
                source.owner_identity,
                source.relative_source_identity.replace('\\', "/")
            );
            let suite_id = SuiteId::derive(revision_identity, &semantic_suite);
            let group_id = FixtureGroupId::derive(suite_id.as_str(), "script-session");
            let sections = if module.script_sections.is_empty() {
                vec![(
                    1,
                    stem.clone(),
                    Span {
                        start: 0,
                        end: source.source_text.len(),
                    },
                )]
            } else {
                module
                    .script_sections
                    .iter()
                    .map(|section| (section.ordinal, section.title.clone(), section.body_span))
                    .collect()
            };
            let tests = sections
                .into_iter()
                .map(|(ordinal, title, span)| {
                    let semantic_path = DefPath {
                        package: PackageName(source.owner_identity.to_owned()),
                        module: QualifiedName(vec![SymbolName(stem.clone())]),
                        item: vec![DefPathSegment::ScriptSection {
                            ordinal,
                            title: title.clone(),
                        }],
                    };
                    let item_path = def_path_string(&semantic_path);
                    let display_name = if title.is_empty() {
                        format!("{stem}/section-{ordinal}")
                    } else {
                        format!("{stem}/{title}")
                    };
                    descriptor(
                        source,
                        suite_id.clone(),
                        group_id.clone(),
                        "script-section",
                        item_path,
                        display_name,
                        ProcedureKind::ScriptSection,
                        span,
                        Vec::new(),
                        Vec::new(),
                    )
                })
                .collect();
            discovery.suites.push(DiscoveredSuite {
                id: suite_id,
                fixture_group_id: group_id,
                display_name: stem.clone(),
                source: source_descriptor(
                    source,
                    format!("{stem}#script"),
                    Span {
                        start: 0,
                        end: source.source_text.len(),
                    },
                ),
                fixtures: Vec::new(),
                tests,
            });
        }

        let mut functions = source
            .assembly
            .functions
            .iter()
            .filter(|function| {
                let leaf = function_leaf(&function.name.0);
                matches!(function.kind, FunctionKind::Named)
                    && function.enclosing_class.is_none()
                    && is_test_name(leaf)
                    && (function.outputs.is_empty() || !leaf.eq_ignore_ascii_case(&stem))
            })
            .collect::<Vec<_>>();
        if functions.is_empty() {
            continue;
        }
        functions.sort_by(|left, right| left.name.0.cmp(&right.name.0));
        let semantic_suite = format!(
            "{}/{}#functions",
            source.owner_identity,
            source.relative_source_identity.replace('\\', "/")
        );
        let suite_id = SuiteId::derive(revision_identity, &semantic_suite);
        let group_id = FixtureGroupId::derive(suite_id.as_str(), "function-session");
        let tests = functions
            .into_iter()
            .map(|function| {
                let leaf = function_leaf(&function.name.0);
                let path = DefPath {
                    package: PackageName(source.owner_identity.to_owned()),
                    module: QualifiedName(vec![SymbolName(stem.clone())]),
                    item: vec![DefPathSegment::Function(SymbolName(leaf.to_owned()))],
                };
                descriptor(
                    source,
                    suite_id.clone(),
                    group_id.clone(),
                    "function",
                    def_path_string(&path),
                    format!("{stem}/{leaf}"),
                    ProcedureKind::Function,
                    function.span,
                    Vec::new(),
                    Vec::new(),
                )
            })
            .collect();
        let fixtures = function_fixtures(source, suite_id.as_str(), &group_id, &stem);
        discovery.suites.push(DiscoveredSuite {
            id: suite_id,
            fixture_group_id: group_id,
            display_name: stem,
            source: source_descriptor(
                source,
                format!("{}#functions", source.relative_source_identity),
                Span {
                    start: 0,
                    end: source.source_text.len(),
                },
            ),
            fixtures,
            tests,
        });
    }
}

fn function_fixtures(
    source: &crate::testing::SemanticTestSource<'_>,
    suite_identity: &str,
    group_id: &FixtureGroupId,
    stem: &str,
) -> Vec<FixtureDescriptor> {
    let procedure = |name: &str| {
        source.assembly.functions.iter().find_map(|function| {
            let leaf = function_leaf(&function.name.0);
            if !leaf.eq_ignore_ascii_case(name)
                || !matches!(function.kind, FunctionKind::Named)
                || function.enclosing_class.is_some()
            {
                return None;
            }
            let path = DefPath {
                package: PackageName(source.owner_identity.to_owned()),
                module: QualifiedName(vec![SymbolName(stem.to_owned())]),
                item: vec![DefPathSegment::Function(SymbolName(leaf.to_owned()))],
            };
            let semantic_path = def_path_string(&path);
            Some(ProcedureDescriptor {
                semantic_path: semantic_path.clone(),
                display_name: leaf.to_owned(),
                kind: ProcedureKind::Function,
                source: source_descriptor(source, semantic_path, function.span),
            })
        })
    };

    [
        (
            "suite",
            FixtureScope::Suite,
            procedure("setupOnce"),
            procedure("teardownOnce"),
        ),
        (
            "test",
            FixtureScope::Test,
            procedure("setup"),
            procedure("teardown"),
        ),
    ]
    .into_iter()
    .filter_map(|(name, scope, setup, teardown)| {
        if setup.is_none() && teardown.is_none() {
            return None;
        }
        let semantic_identity = format!("{stem}/function-fixture/{name}");
        Some(FixtureDescriptor {
            id: FixtureId::derive(suite_identity, &semantic_identity),
            group_id: group_id.clone(),
            display_name: format!("{name} fixture"),
            scope,
            setup,
            teardown,
            dependencies: Vec::new(),
        })
    })
    .collect()
}
