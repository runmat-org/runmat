use runmat_package::{
    build_path_graph, CanonicalPackageId, ContentDigest, GraphError, HostCapability, PackageAlias,
    PathGraphInput, PathPackageInput, VisibilityResolution,
};
use std::collections::{BTreeMap, BTreeSet};

fn package(identity: &str, local_name: &str, path: &str, version: &str) -> PathPackageInput {
    PathPackageInput {
        package: identity.parse::<CanonicalPackageId>().unwrap(),
        local_name: local_name.to_string(),
        workspace_path: path.parse().unwrap(),
        manifest_digest: ContentDigest::sha256(format!("{path}/runmat.toml")),
        tree_digest: ContentDigest::sha256(format!("{path}/tree")),
        version: Some(version.parse().unwrap()),
        dependencies: BTreeMap::new(),
        required_capabilities: BTreeSet::new(),
        singleton: false,
    }
}

fn graph_input() -> PathGraphInput {
    let mut root = package("workspace:local/application", "application", ".", "1.0.0");
    root.dependencies
        .insert("matrix-v1".parse::<PackageAlias>().unwrap(), "v1".into());
    root.dependencies
        .insert("matrix-v2".parse::<PackageAlias>().unwrap(), "v2".into());
    PathGraphInput {
        root: "root".to_string(),
        packages: BTreeMap::from([
            ("root".to_string(), root),
            (
                "v1".to_string(),
                package("default:runmat/matrix", "matrix", "deps/matrix-v1", "1.0.0"),
            ),
            (
                "v2".to_string(),
                package("default:runmat/matrix", "matrix", "deps/matrix-v2", "2.0.0"),
            ),
        ]),
        host_capabilities: BTreeSet::new(),
    }
}

#[test]
fn deterministic_graph_allows_multiple_safe_instances() {
    let first = build_path_graph(graph_input()).unwrap();
    let second = build_path_graph(graph_input()).unwrap();
    assert_eq!(first, second);
    assert_eq!(
        first
            .instances_of(&"default:runmat/matrix".parse().unwrap())
            .len(),
        2
    );
    let matrix_candidates = first
        .instances_of(&"default:runmat/matrix".parse().unwrap())
        .iter()
        .map(|package| package.instance.identity_digest.clone())
        .collect::<Vec<_>>();
    assert!(matches!(
        first.resolve_visible_candidates(&first.root, matrix_candidates),
        VisibilityResolution::Ambiguous(candidates) if candidates.len() == 2
    ));
}

#[test]
fn root_precedence_and_edge_local_alias_visibility_are_explicit() {
    let graph = build_path_graph(graph_input()).unwrap();
    let root = graph.packages.get(&graph.root).unwrap();
    let dependency = graph
        .dependency(&graph.root, &"matrix-v1".parse().unwrap())
        .unwrap();
    assert_eq!(
        dependency.instance.version.as_ref().unwrap().to_string(),
        "1.0.0"
    );
    assert_eq!(
        graph.resolve_visible_candidates(
            &dependency.instance.identity_digest,
            [root.instance.identity_digest.clone()]
        ),
        VisibilityResolution::Found(graph.root.clone())
    );
}

#[test]
fn singleton_and_capability_failures_are_diagnostic() {
    let mut input = graph_input();
    input.packages.get_mut("v1").unwrap().singleton = true;
    assert!(matches!(
        build_path_graph(input),
        Err(GraphError::Invalid(message)) if message.contains("singleton")
    ));

    let mut input = graph_input();
    input
        .packages
        .get_mut("v2")
        .unwrap()
        .required_capabilities
        .insert(HostCapability::NativeLibrary);
    assert!(matches!(
        build_path_graph(input),
        Err(GraphError::UnavailableCapabilities {
            dependency_path,
            capabilities,
        }) if dependency_path == "root -> matrix-v2" && capabilities == "native-library"
    ));
}
