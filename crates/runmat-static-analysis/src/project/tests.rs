use super::*;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};

struct ReadyOnly;
impl Wake for ReadyOnly {
    fn wake(self: Arc<Self>) {}
}

fn check(primary: &str, companions: &[(&str, &str)]) -> ProjectAnalysis {
    let definitions = companions
        .iter()
        .map(|(name, _)| ProjectSymbolDefinition {
            name: (*name).into(),
            qualified_name: (*name).into(),
            source_path: PathBuf::from(format!("/project/{name}.m")),
            package_name: String::new(),
            dependency_alias: None,
            package_instance: None,
            source_id: None,
            is_private: false,
        })
        .collect::<Vec<_>>();
    let catalog = DiscoveredSourceSymbols {
        manifest_path: None,
        project_root: PathBuf::from("/project"),
        graph_digest: None,
        source_revision: None,
        symbols: definitions
            .iter()
            .map(|definition| definition.name.clone())
            .collect(),
        definitions,
    };
    let bytes = companions
        .iter()
        .map(|(name, text)| {
            (
                PathBuf::from(format!("/project/{name}.m")),
                text.to_string(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    check_catalog(primary, &catalog, &bytes).expect("project analysis")
}

fn check_catalog(
    primary: &str,
    catalog: &DiscoveredSourceSymbols,
    bytes: &BTreeMap<PathBuf, String>,
) -> Result<ProjectAnalysis, ProjectAnalysisError> {
    let future = analyze_project_source(
        ProjectSource {
            path: PathBuf::from("/project/driver.m"),
            text: primary.into(),
        },
        FsPathNamespace::VirtualRooted,
        CompatMode::default(),
        Some(catalog),
        |definition| std::future::ready(Ok(bytes[&definition.source_path].clone())),
    );
    ready_project(future)
}

fn ready_project(
    future: impl Future<Output = Result<ProjectAnalysis, ProjectAnalysisError>>,
) -> Result<ProjectAnalysis, ProjectAnalysisError> {
    let mut future = std::pin::pin!(future);
    let waker = Waker::from(Arc::new(ReadyOnly));
    match future.as_mut().poll(&mut Context::from_waker(&waker)) {
        Poll::Ready(result) => result,
        Poll::Pending => panic!("fixture loader is always ready"),
    }
}

#[test]
fn explicit_virtual_namespace_admits_rooted_paths_without_host_absolute_rules() {
    let analyze = |namespace, path: &str| {
        analyze_project_source(
            ProjectSource {
                path: path.into(),
                text: "x=1;".into(),
            },
            namespace,
            CompatMode::default(),
            None,
            |_| std::future::ready(Err("no source reads expected".into())),
        )
    };
    let result = ready_project(analyze(FsPathNamespace::VirtualRooted, "/project/driver.m"))
        .expect("provider-rooted path is independent of the host OS grammar");
    assert!(!result.analysis.has_errors());
    assert!(matches!(
        ready_project(analyze(FsPathNamespace::VirtualRooted, "driver.m")),
        Err(ProjectAnalysisError::InvalidPrimaryPath),
    ));
    #[cfg(target_arch = "wasm32")]
    assert!(matches!(
        ready_project(analyze(FsPathNamespace::NativeOs, "/project/driver.m")),
        Err(ProjectAnalysisError::InvalidPrimaryPath),
    ));

    let catalog = DiscoveredSourceSymbols {
        manifest_path: None,
        project_root: "/project".into(),
        graph_digest: None,
        source_revision: None,
        symbols: HashSet::from(["product".into()]),
        definitions: vec![ProjectSymbolDefinition {
            name: "product".into(),
            qualified_name: "product".into(),
            source_path: "product.m".into(),
            package_name: String::new(),
            dependency_alias: None,
            package_instance: None,
            source_id: None,
            is_private: false,
        }],
    };
    let result = ready_project(analyze_project_source(
        ProjectSource {
            path: "/project/driver.m".into(),
            text: "C=product(1,1);".into(),
        },
        FsPathNamespace::VirtualRooted,
        CompatMode::default(),
        Some(&catalog),
        |_| {
            std::future::ready(Err(
                "relative selection must be rejected before reading".into()
            ))
        },
    ));
    assert!(matches!(result,
        Err(ProjectAnalysisError::InvalidSelectedPath { path }) if path == PathBuf::from("product.m")));
}

const PRODUCT: &str = "function C = product(A, B)\nC = A * B;\nend\n";

#[test]
fn reached_cross_file_shape_failure_is_at_driver_with_related_callee_source() {
    let primary = "A=ones(2,3); B=ones(4,2); C=product(A,B);";
    let result = check(
        primary,
        &[("product", PRODUCT), ("unused", "not valid MATLAB @@@")],
    );
    assert_eq!(
        result.sources.len(),
        2,
        "unreached files must not be read or parsed"
    );
    let diagnostic = result
        .analysis
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code == "RM-TYPE-MATMUL")
        .expect("known incompatible operands must be diagnosed through the call");
    assert_eq!(diagnostic.primary.source, Some(ProgramSourceId(0)));
    assert!(
        primary[diagnostic.primary.span.start..diagnostic.primary.span.end].contains("product")
    );
    assert!(diagnostic.secondary.iter().any(|related| {
        related.source == Some(ProgramSourceId(1))
            && PRODUCT[related.span.start..related.span.end].contains('*')
    }));
    assert_eq!(
        result.source(Some(ProgramSourceId(1))).unwrap().path,
        PathBuf::from("/project/product.m")
    );
    assert!(result.source(Some(ProgramSourceId(99))).is_none());
}

#[test]
fn public_function_contexts_preserve_polymorphic_shapes_and_scalar_multiply() {
    for primary in [
        "C=product(ones(2,3),ones(3,4));",
        "C=product(ones(2,3),2);",
        "C=product(ones(2,3),ones(3,4)); D=product(ones(5,2),ones(2,1));",
        "function C=entry(A,B); C=product(A,B); end",
    ] {
        let result = check(primary, &[("product", PRODUCT)]);
        assert!(
            !result.analysis.has_errors(),
            "{primary}: {:?}",
            result.analysis.diagnostics
        );
        assert_eq!(result.sources.len(), 2);
    }
}

#[test]
fn reachable_wrapper_uses_existing_call_facts_but_recursion_stays_conservative() {
    let wrapper = "function C=wrapper(A,B); C=product(A,B); end";
    let result = check(
        "C=wrapper(ones(2,3),ones(4,2));",
        &[("wrapper", wrapper), ("product", PRODUCT)],
    );
    assert_eq!(result.sources.len(), 3);
    assert!(result.analysis.resolution.iter().any(|resolution| {
        resolution.name == "wrapper" && resolution.source_id == Some(ProgramSourceId(0))
    }));
    assert!(result.analysis.resolution.iter().any(|resolution| {
        resolution.name == "product" && resolution.source_id == Some(ProgramSourceId(1))
    }));
    let resolution_json = serde_json::to_value(&result.analysis.resolution).unwrap();
    assert!(resolution_json
        .as_array()
        .unwrap()
        .iter()
        .any(|resolution| { resolution["name"] == "product" && resolution["source_id"] == 1 }));
    assert!(result.analysis.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "RM-TYPE-MATMUL" && diagnostic.primary.source == Some(ProgramSourceId(0))
    }));
    let recursive = "function C=recursive(A,B); C=recursive(A,B); end";
    let result = check(
        "C=recursive(ones(2,3),ones(4,2));",
        &[("recursive", recursive)],
    );
    assert!(
        !result.analysis.has_errors(),
        "{:?}",
        result.analysis.diagnostics
    );
}

#[test]
fn multiple_file_local_declarations_are_explicitly_partial_not_falsely_rebound() {
    let source = "function C=product(A,B); C=local(A,B); end\nfunction C=local(A,B); C=A*B; end";
    let result = check("C=product(ones(2,3),ones(4,2));", &[("product", source)]);
    assert_eq!(
        result.incomplete_sources,
        [PathBuf::from("/project/product.m")]
    );
    assert_eq!(
        result.analysis.domains.name_resolution,
        AnalysisCompleteness::Partial
    );
    assert_eq!(
        result.analysis.domains.shapes,
        AnalysisCompleteness::Partial
    );
    assert!(
        !result.analysis.has_errors(),
        "no bogus duplicate/guessed alias diagnostic"
    );
}

#[test]
fn package_and_visible_private_sources_keep_their_declared_qualified_identity() {
    for (name, path, private) in [
        ("pkg.product", "/project/+pkg/product.m", false),
        ("product", "/project/private/product.m", true),
    ] {
        let definition = ProjectSymbolDefinition {
            name: name.into(),
            qualified_name: name.into(),
            source_path: path.into(),
            package_name: String::new(),
            dependency_alias: None,
            package_instance: None,
            source_id: None,
            is_private: private,
        };
        let catalog = DiscoveredSourceSymbols {
            manifest_path: None,
            project_root: "/project".into(),
            graph_digest: None,
            source_revision: None,
            symbols: HashSet::from([name.into()]),
            definitions: vec![definition],
        };
        let bytes = BTreeMap::from([(PathBuf::from(path), PRODUCT.into())]);
        let result =
            check_catalog(&format!("C={name}(ones(2,3),ones(4,2));"), &catalog, &bytes).unwrap();
        assert!(
            result.analysis.diagnostics.iter().any(|diagnostic| {
                diagnostic.code == "RM-TYPE-MATMUL"
                    && diagnostic.primary.source == Some(ProgramSourceId(0))
            }),
            "{name}: {:?}",
            result.analysis.diagnostics
        );
        assert_eq!(result.sources[1].path, PathBuf::from(path));
    }
}

#[test]
fn selected_frozen_source_bytes_cannot_silently_differ_from_the_catalog_revision() {
    let path = PathBuf::from("/project/product.m");
    let digest = runmat_package::ContentDigest::sha256(b"fixture-owner");
    let definition = ProjectSymbolDefinition {
        name: "product".into(),
        qualified_name: "product".into(),
        source_path: path.clone(),
        package_name: String::new(),
        dependency_alias: None,
        package_instance: Some(digest.clone()),
        source_id: Some(runmat_package::StableSourceId {
            package_instance: digest,
            relative_path: runmat_package::NormalizedRelativePath::new("product.m").unwrap(),
            content_digest: runmat_package::ContentDigest::sha256(b"other source revision"),
        }),
        is_private: false,
    };
    let catalog = DiscoveredSourceSymbols {
        manifest_path: None,
        project_root: "/project".into(),
        graph_digest: None,
        source_revision: None,
        symbols: HashSet::from(["product".into()]),
        definitions: vec![definition],
    };
    let result = check_catalog(
        "C=product(ones(2,3),ones(4,2));",
        &catalog,
        &BTreeMap::from([(path, PRODUCT.into())]),
    );
    assert!(
        matches!(result, Err(ProjectAnalysisError::RevisionChanged { path }) if path == PathBuf::from("/project/product.m"))
    );
}

#[test]
fn private_alias_scope_cannot_leak_between_unqualified_source_roots() {
    let public = ProjectSymbolDefinition {
        name: "public_helper".into(),
        qualified_name: "public_helper".into(),
        source_path: "/other/public_helper.m".into(),
        package_name: String::new(),
        dependency_alias: None,
        package_instance: None,
        source_id: None,
        is_private: false,
    };
    let mut public_clashing = public.clone();
    public_clashing.name = "clashing".into();
    public_clashing.qualified_name = "clashing".into();
    public_clashing.source_path = "/project/clashing.m".into();
    let private = ProjectSymbolDefinition {
        name: "clashing".into(),
        qualified_name: "clashing".into(),
        source_path: "/other/private/clashing.m".into(),
        package_name: String::new(),
        dependency_alias: None,
        package_instance: None,
        source_id: None,
        is_private: true,
    };
    let bytes = BTreeMap::from([
        (
            public.source_path.clone(),
            "function C=public_helper(A,B); C=clashing(A,B); end".into(),
        ),
        (
            public_clashing.source_path.clone(),
            "function C=clashing(A,B); C=ones(4,2); end".into(),
        ),
        (
            private.source_path.clone(),
            "function C=clashing(A,B); C=ones(3,2); end".into(),
        ),
    ]);
    let catalog = DiscoveredSourceSymbols {
        manifest_path: None,
        project_root: "/project".into(),
        graph_digest: None,
        source_revision: None,
        symbols: HashSet::from(["public_helper".into(), "clashing".into()]),
        definitions: vec![public, public_clashing, private],
    };
    let result = check_catalog(
        "ignored=clashing(1,1); C=public_helper(1,1); Z=ones(2,3)*C;",
        &catalog,
        &bytes,
    )
    .unwrap();
    assert_eq!(
        result.incomplete_sources,
        [PathBuf::from("/other/private/clashing.m")]
    );
    assert_eq!(
        result.analysis.domains.name_resolution,
        AnalysisCompleteness::Partial
    );
    assert!(!result
        .analysis
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TYPE-MATMUL"));
    assert!(result.call_context_precision_withheld);
    assert_eq!(
        result.analysis.mir.as_ref().unwrap().functions.len(),
        1,
        "unsafe assembly must not publish generic wrong-public summary facts either"
    );
}

#[test]
fn multiple_alias_views_of_one_source_cannot_guess_a_private_owner() {
    let path = PathBuf::from("/project/shared/wrapper.m");
    let definitions = ["left", "right"].map(|alias| ProjectSymbolDefinition {
        name: format!("{alias}.pkg.wrapper"),
        qualified_name: "pkg.wrapper".into(),
        source_path: path.clone(),
        package_name: String::new(),
        dependency_alias: Some(alias.into()),
        package_instance: None,
        source_id: None,
        is_private: false,
    });
    let catalog = DiscoveredSourceSymbols {
        manifest_path: None,
        project_root: "/project".into(),
        graph_digest: None,
        source_revision: None,
        symbols: definitions
            .iter()
            .map(|definition| definition.name.clone())
            .collect(),
        definitions: definitions.into(),
    };
    let result = check_catalog(
        "C=right.pkg.wrapper(ones(2,3),ones(4,2));",
        &catalog,
        &BTreeMap::from([(
            path.clone(),
            "function C=wrapper(A,B); C=private_helper(A,B); end".into(),
        )]),
    )
    .unwrap();
    assert_eq!(result.incomplete_sources, [path]);
    assert!(result.call_context_precision_withheld);
    assert_eq!(
        result.analysis.domains.name_resolution,
        AnalysisCompleteness::Partial
    );
    assert!(!result.analysis.has_errors());
    assert_eq!(
        result.analysis.mir.as_ref().unwrap().functions.len(),
        1,
        "ambiguous owner must fall back before publishing cross-file facts"
    );
}
