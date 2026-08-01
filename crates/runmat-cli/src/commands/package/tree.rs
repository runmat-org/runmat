use anyhow::{bail, Result};
use runmat_package::{ContentDigest, PackageGraph};
use std::collections::BTreeSet;

pub(super) fn print_tree(graph: &PackageGraph) {
    print_package(graph, &graph.root, "", true, true, &mut BTreeSet::new());
}

fn print_package(
    graph: &PackageGraph,
    identity: &ContentDigest,
    prefix: &str,
    last: bool,
    root: bool,
    active: &mut BTreeSet<ContentDigest>,
) {
    let package = &graph.packages[identity];
    let connector = if root {
        ""
    } else if last {
        "└── "
    } else {
        "├── "
    };
    println!(
        "{prefix}{connector}{}{}",
        package.instance.package,
        package
            .instance
            .version
            .as_ref()
            .map(|version| format!(" {version}"))
            .unwrap_or_default()
    );
    if !active.insert(identity.clone()) {
        return;
    }
    let edges = graph
        .edges
        .iter()
        .filter(|edge| &edge.from == identity)
        .collect::<Vec<_>>();
    let child_prefix = if root {
        String::new()
    } else if last {
        format!("{prefix}    ")
    } else {
        format!("{prefix}│   ")
    };
    for (index, edge) in edges.iter().enumerate() {
        print_package(
            graph,
            &edge.to,
            &child_prefix,
            index + 1 == edges.len(),
            false,
            active,
        );
    }
    active.remove(identity);
}

pub(super) fn print_why(graph: &PackageGraph, query: &str) -> Result<()> {
    let matches = graph
        .packages
        .iter()
        .filter(|(_, package)| {
            package.local_name == query || package.instance.package.to_string() == query
        })
        .map(|(identity, _)| identity)
        .collect::<Vec<_>>();
    if matches.is_empty() {
        bail!("no resolved package matches `{query}`");
    }
    for target in matches {
        for path in dependency_paths(graph, target) {
            println!("{path}");
        }
    }
    Ok(())
}

fn dependency_paths(
    graph: &PackageGraph,
    target: &ContentDigest,
) -> Vec<runmat_package::DependencyPath> {
    let root_name = graph.packages[&graph.root].local_name.clone();
    let mut found = Vec::new();
    let mut pending = vec![(
        graph.root.clone(),
        Vec::<runmat_package::PackageAlias>::new(),
        BTreeSet::new(),
    )];
    while let Some((current, path, mut visited)) = pending.pop() {
        if !visited.insert(current.clone()) {
            continue;
        }
        for edge in graph.edges.iter().filter(|edge| edge.from == current) {
            let mut next = path.clone();
            next.push(edge.alias.clone());
            if &edge.to == target {
                found.push(runmat_package::DependencyPath {
                    root: root_name.clone(),
                    aliases: next,
                });
            } else {
                pending.push((edge.to.clone(), next, visited.clone()));
            }
        }
    }
    found.sort_by(|left, right| left.aliases.cmp(&right.aliases));
    found
}
