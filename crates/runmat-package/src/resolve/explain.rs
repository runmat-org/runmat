use super::{RequirementPath, Resolution};
use crate::{ContentDigest, PackageAlias};
use std::collections::BTreeSet;

pub fn dependency_tree(resolution: &Resolution, root: &str) -> String {
    let mut output = root.to_string();
    let mut seen = BTreeSet::new();
    for edge in resolution.edges.iter().filter(|edge| edge.from.is_none()) {
        append_tree(
            resolution,
            &edge.to,
            &edge.alias,
            "",
            &mut seen,
            &mut output,
        );
    }
    output.push('\n');
    output
}

fn append_tree(
    resolution: &Resolution,
    identity: &ContentDigest,
    alias: &PackageAlias,
    indent: &str,
    seen: &mut BTreeSet<ContentDigest>,
    output: &mut String,
) {
    let package = &resolution.packages[identity].candidate.instance;
    let version = package
        .version
        .as_ref()
        .map(ToString::to_string)
        .unwrap_or_else(|| "unversioned".to_string());
    let repeated = !seen.insert(identity.clone());
    output.push_str(&format!(
        "\n{indent}+- {alias}: {} {version}{}",
        package.package,
        if repeated { " (*)" } else { "" }
    ));
    if repeated {
        return;
    }
    let next_indent = format!("{indent}   ");
    for edge in resolution
        .edges
        .iter()
        .filter(|edge| edge.from.as_ref() == Some(identity))
    {
        append_tree(
            resolution,
            &edge.to,
            &edge.alias,
            &next_indent,
            seen,
            output,
        );
    }
}

pub fn why(resolution: &Resolution, root: &str, target: &ContentDigest) -> Vec<RequirementPath> {
    let mut paths = Vec::new();
    for edge in resolution.edges.iter().filter(|edge| edge.from.is_none()) {
        let mut aliases = vec![edge.alias.clone()];
        find_paths(
            resolution,
            &edge.to,
            target,
            &mut aliases,
            &mut BTreeSet::new(),
            &mut paths,
            root,
        );
    }
    paths.sort();
    paths.dedup();
    paths
}

fn find_paths(
    resolution: &Resolution,
    current: &ContentDigest,
    target: &ContentDigest,
    aliases: &mut Vec<PackageAlias>,
    active: &mut BTreeSet<ContentDigest>,
    output: &mut Vec<RequirementPath>,
    root: &str,
) {
    if current == target {
        output.push(RequirementPath {
            root: root.to_string(),
            aliases: aliases.clone(),
        });
        return;
    }
    if !active.insert(current.clone()) {
        return;
    }
    for edge in resolution
        .edges
        .iter()
        .filter(|edge| edge.from.as_ref() == Some(current))
    {
        aliases.push(edge.alias.clone());
        find_paths(resolution, &edge.to, target, aliases, active, output, root);
        aliases.pop();
    }
    active.remove(current);
}
