use super::*;
use crate::session::compile::DiscoveredSourceStatement;

#[test]
fn loose_search_root_identity_is_relocatable_and_distinguishes_equal_leaf_sources() {
    use runmat_runtime::builtins::common::path_search::FileCandidateOrigin;
    let text = "function y = same(); y = 1; end";
    let origin = |ordinal| FileCandidateOrigin::SearchRoot {
        ordinal,
        access_root: std::path::PathBuf::from("/checkout-one/pkg"),
        relative: std::path::PathBuf::from("+pkg/same.m"),
    };
    let first =
        ProgramSourceTable::dynamic_root("/checkout-one/pkg/+pkg/same.m", text, None, &origin(0))
            .unwrap();
    let relocated =
        ProgramSourceTable::dynamic_root("/checkout-two/pkg/+pkg/same.m", text, None, &origin(0))
            .unwrap();
    let next_root =
        ProgramSourceTable::dynamic_root("/checkout-one/other/+pkg/same.m", text, None, &origin(1))
            .unwrap();
    assert_eq!(first, relocated);
    assert_ne!(first, next_root);
    let build = |root: &crate::ExecutableSource, full_path: &str| {
        let map = ProgramSourceTable::build(
            root,
            Some(full_path),
            &CompanionSourceDiscovery::default(),
            1,
            None,
            &[],
        )
        .unwrap()
        .source_map
        .portable_v2()
        .unwrap();
        serde_json::to_vec(&map).unwrap()
    };
    assert_eq!(
        build(&first, "/checkout-one/pkg/+pkg/same.m"),
        build(&relocated, "/checkout-two/pkg/+pkg/same.m")
    );
    assert_ne!(
        build(&first, "/checkout-one/pkg/+pkg/same.m"),
        build(&next_root, "/checkout-one/other/+pkg/same.m")
    );
}

#[test]
fn loose_companion_uses_the_same_captured_search_root_not_cwd_display_text() {
    use runmat_runtime::builtins::common::path_search::FileCandidateOrigin;
    let build = |prefix: &str, companion_prefix: &str, canonical_root: Option<&str>| {
        let access_root = std::path::PathBuf::from(prefix);
        let primary_access = format!("{prefix}/+pkg/main.m");
        let companion_access = std::path::Path::new(companion_prefix)
            .join("+pkg")
            .join("helper.m")
            .to_string_lossy()
            .into_owned();
        let origin = FileCandidateOrigin::SearchRoot {
            ordinal: 2,
            access_root,
            relative: std::path::PathBuf::from("+pkg/main.m"),
        };
        let primary =
            ProgramSourceTable::dynamic_root(&primary_access, "value = helper();", None, &origin)
                .unwrap();
        let mut discovery = CompanionSourceDiscovery::default();
        #[cfg(windows)]
        {
            discovery.canonical_search_root = canonical_root.map(std::path::PathBuf::from);
        }
        #[cfg(not(windows))]
        let _ = canonical_root;
        discovery.statements.push(DiscoveredSourceStatement {
            statement: runmat_parser::parse("function y = helper(); y = 5; end")
                .unwrap()
                .body
                .remove(0),
            is_private: false,
            source: Some(Arc::new(SourceContextData {
                display_name: companion_access.clone(),
                fullpath_name: Some(companion_access),
                text: "function y = helper(); y = 5; end".into(),
            })),
        });
        let table = ProgramSourceTable::build_dynamic(
            &primary,
            &primary_access,
            &discovery,
            1,
            None,
            &origin,
        )?;
        assert_eq!(table.source_map.entries()[1].relative_path, "+pkg/helper.m");
        assert_eq!(
            table.source_map.entries()[1].owner_identity,
            "dynamic-search-root:2"
        );
        Ok::<_, String>(serde_json::to_vec(&table.source_map.portable_v2().unwrap()).unwrap())
    };
    assert_eq!(
        build("/checkout-one", "/checkout-one", None).unwrap(),
        build("/checkout-two", "/checkout-two", None).unwrap()
    );
    assert!(build("/checkout-one", "/different-root", None).is_err());
    #[cfg(windows)]
    {
        assert_eq!(
            build(r"C:\checkout-one", r"C:\checkout-one", None).unwrap(),
            build(
                r"C:\checkout-one",
                r"\\?\C:\checkout-one",
                Some(r"\\?\C:\checkout-one")
            )
            .unwrap()
        );
        assert!(build(
            r"C:\checkout-one",
            r"\\?\C:\different-root",
            Some(r"\\?\C:\checkout-one")
        )
        .is_err());
    }
}

fn runtime() -> runmat_runtime::context::RuntimeContext {
    runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
}

fn lease(
    runtime: &runmat_runtime::context::RuntimeContext,
    source: &SourceContextData,
) -> runmat_runtime::source_context::RuntimeSourceLease {
    let map = runmat_runtime::source_context::ValidatedPortableSourceMap::new(Arc::new(
        runmat_execution::ExecutableSourceMapV2::new(vec![
            runmat_execution::ExecutableSourceRecordV2 {
                source_id: ProgramSourceId(0),
                owner_identity: "interactive".into(),
                relative_path: source.display_name.clone(),
                content_revision: runmat_execution::Digest::sha256(source.text.as_bytes()),
                display_name: source.display_name.clone(),
                text: source.text.clone(),
            },
        ])
        .expect("validated retained source map"),
    ))
    .expect("admitted retained source map");
    runmat_runtime::source_context::lease_portable_source_context_in_context(
        runtime,
        &map.record(ProgramSourceId(0)).expect("retained ordinal"),
        source.fullpath_name.as_deref().map(Arc::<str>::from),
    )
    .expect("captured retained source identity")
}

fn retained(
    runtime: &runmat_runtime::context::RuntimeContext,
    lease: &runmat_runtime::source_context::RuntimeSourceLease,
    source: SourceContextData,
) -> (
    runmat_runtime::source_context::RuntimeSourceRef,
    SourceContextData,
    runmat_runtime::source_context::PortableSourceIdentity,
) {
    let reference = lease.source_ref();
    let identity =
        runmat_runtime::source_context::portable_source_identity_in_context(runtime, reference)
            .expect("retained source owner");
    (reference, source, identity)
}

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
    let first = ProgramSourceTable::build(&root, None, &one, 1, None, &[]).unwrap();
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
    let second = ProgramSourceTable::build(&root, None, &two, 1, None, &[]).unwrap();
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
    assert!(
        ProgramSourceTable::build(&root, None, &discovery, 1, None, &[])
            .err()
            .expect("conflicting source must fail")
            .contains("conflicting discovered text")
    );
}

#[test]
fn retained_definition_sources_use_stable_ordinals_not_runtime_ids() {
    let root = crate::ExecutableSource::new("interactive", "main.m", "value = prior();");
    let prior = SourceContextData {
        display_name: "prior.m".into(),
        fullpath_name: Some("/project/prior.m".into()),
        text: "function y = prior(); y = 4; end".into(),
    };
    let runtime = runtime();
    let first_lease = lease(&runtime, &prior);
    let second_lease = lease(&runtime, &prior);
    let empty = CompanionSourceDiscovery::default();
    let first = ProgramSourceTable::build(
        &root,
        None,
        &empty,
        1,
        None,
        &[retained(&runtime, &first_lease, prior.clone())],
    )
    .unwrap();
    let second = ProgramSourceTable::build(
        &root,
        None,
        &empty,
        1,
        None,
        &[retained(&runtime, &second_lease, prior)],
    )
    .unwrap();
    assert_eq!(first.source_map, second.source_map);
    assert_eq!(
        first.retained_runtime_sources[&first_lease.source_ref()],
        ProgramSourceId(1)
    );
    assert_eq!(
        second.retained_runtime_sources[&second_lease.source_ref()],
        ProgramSourceId(1)
    );
}

#[test]
fn retained_source_keeps_original_owner_across_a_foreign_primary() {
    let runtime = runtime();
    let prior = SourceContextData {
        display_name: "prior.m".into(),
        fullpath_name: None,
        text: "function y = prior(); y = 4; end".into(),
    };
    let lease = lease(&runtime, &prior);
    let table = ProgramSourceTable::build(
        &crate::ExecutableSource::new("new-owner", "main.m", "y = prior();"),
        None,
        &CompanionSourceDiscovery::default(),
        1,
        None,
        &[retained(&runtime, &lease, prior)],
    )
    .expect("retained source table");
    let retained_entry = &table.source_map.entries()[1];
    assert_eq!(retained_entry.owner_identity, "interactive");
    assert_eq!(retained_entry.relative_path, "prior.m");
}

#[test]
fn retained_source_rejects_content_outside_its_issued_owner() {
    let runtime = runtime();
    let prior = SourceContextData {
        display_name: "prior.m".into(),
        fullpath_name: None,
        text: "function y = prior(); y = 4; end".into(),
    };
    let lease = lease(&runtime, &prior);
    let mut altered = prior.clone();
    altered.text = "function y = prior(); y = 5; end".into();
    let result = ProgramSourceTable::build(
        &crate::ExecutableSource::new("new-owner", "main.m", "y = prior();"),
        None,
        &CompanionSourceDiscovery::default(),
        1,
        None,
        &[retained(&runtime, &lease, altered)],
    );
    assert!(matches!(
        result,
        Err(ref error) if error == "retained source text disagrees with its issued portable identity"
    ));
}

#[test]
fn retained_generations_at_one_location_have_explicit_content_revisions() {
    let root = crate::ExecutableSource::new("interactive", "main.m", "value = prior();");
    let source = |text: &str| SourceContextData {
        display_name: "prior.m".into(),
        fullpath_name: Some("/project/prior.m".into()),
        text: text.into(),
    };
    let runtime = runtime();
    let first_source = source("function y = prior(); y = 1; end");
    let second_source = source("function y = prior(); y = 2; end");
    let first_lease = lease(&runtime, &first_source);
    let second_lease = lease(&runtime, &second_source);
    let table = ProgramSourceTable::build(
        &root,
        None,
        &CompanionSourceDiscovery::default(),
        1,
        None,
        &[
            retained(&runtime, &first_lease, first_source),
            retained(&runtime, &second_lease, second_source),
        ],
    )
    .unwrap();
    assert_eq!(table.source_map.entries().len(), 3);
    assert_ne!(
        table.retained_runtime_sources[&first_lease.source_ref()],
        table.retained_runtime_sources[&second_lease.source_ref()]
    );
}

#[test]
fn same_retained_sources_ignore_runtime_issuance_order_but_not_content() {
    let root = crate::ExecutableSource::new("interactive", "main.m", "y = prior();");
    let source = |name: &str, text: &str| SourceContextData {
        display_name: name.into(),
        fullpath_name: Some(format!("/project/{name}")),
        text: text.into(),
    };
    let a = source("a.m", "function y = a(); y = 1; end");
    let b = source("b.m", "function y = b(); y = 2; end");
    let changed_a = source("a.m", "function y = a(); y = 3; end");
    let runtime = runtime();
    let first_a = lease(&runtime, &a);
    let first_b = lease(&runtime, &b);
    let second_b = lease(&runtime, &b);
    let second_a = lease(&runtime, &a);
    let changed = lease(&runtime, &changed_a);
    let build = |retained: &[(
        runmat_runtime::source_context::RuntimeSourceRef,
        SourceContextData,
        runmat_runtime::source_context::PortableSourceIdentity,
    )]| {
        ProgramSourceTable::build(
            &root,
            None,
            &CompanionSourceDiscovery::default(),
            1,
            None,
            retained,
        )
        .unwrap()
        .source_map
    };
    let first = build(&[
        retained(&runtime, &first_a, a.clone()),
        retained(&runtime, &first_b, b.clone()),
    ]);
    let reordered = build(&[
        retained(&runtime, &second_b, b.clone()),
        retained(&runtime, &second_a, a),
    ]);
    assert_eq!(first, reordered);
    let changed = build(&[
        retained(&runtime, &second_b, b),
        retained(&runtime, &changed, changed_a),
    ]);
    assert_ne!(first, changed);
}
