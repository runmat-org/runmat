//! Witnesses are declared at the test that supports an obligation, never in the register.
//!
//! ```text
//! // witness: FEA-XXX-001/obligation_id @ formulation/mesh[|mesh]/execution[, ...];
//! //   boundary: what the assertions support and what they do not.
//! #[test]
//! fn the_test() { .. }
//! ```
//!
//! A tag is a `// witness:` comment line plus its `//` continuation lines, placed directly
//! above the test function (attributes may sit between). Several tags may stack on one test.
//! The scanner is std-only so no crate depends on the qualification crate. A tag declares what
//! assertions support; the evidence report decides pass, fail or open by running the test.
use crate::requirement::{Envelope, ExecutionEnvelope, Formulation, MeshOrder, Requirement};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Target {
    Library,
    Integration(String),
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct TestLocator {
    pub package: String,
    pub target: Target,
    /// Module path inside the target followed by the function name, as libtest lists it.
    pub name: String,
    /// Relative to the repository root.
    pub source: String,
    /// 1-based line of the test function.
    pub line: usize,
}

impl TestLocator {
    pub fn describe(&self) -> String {
        let target = match &self.target {
            Target::Library => "lib".to_string(),
            Target::Integration(name) => format!("test:{name}"),
        };
        format!("{}[{}]::{}", self.package, target, self.name)
    }
}

#[derive(Clone, Debug)]
pub struct Tag {
    pub requirement: String,
    pub obligation: String,
    pub envelopes: Vec<Envelope>,
    pub boundary: String,
    pub test: TestLocator,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Problem {
    pub location: String,
    pub message: String,
}

impl std::fmt::Display for Problem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.location, self.message)
    }
}

pub fn repository_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

/// Parse the joined text of one tag (after `witness:`).
pub fn parse_tag_text(text: &str) -> Result<(String, String, Vec<Envelope>, String), String> {
    let (head, boundary) = text
        .split_once("; boundary:")
        .ok_or("missing `; boundary: <text>`")?;
    let boundary = boundary.trim();
    if boundary.is_empty() {
        return Err("empty boundary".into());
    }
    let (claim, envelopes) = head.split_once('@').ok_or("missing `@ <envelopes>`")?;
    let (requirement, obligation) = claim
        .trim()
        .split_once('/')
        .ok_or("expected REQUIREMENT/obligation before `@`")?;
    let mut cells = Vec::new();
    for part in envelopes.split(',') {
        let part = part.trim();
        if part.is_empty() {
            return Err("empty envelope".into());
        }
        let fields: Vec<&str> = part.split('/').map(str::trim).collect();
        let [formulations, meshes, executions] = fields[..] else {
            return Err(format!("envelope `{part}` is not formulation/mesh/execution"));
        };
        for formulation in formulations.split('|') {
            let formulation = Formulation::parse(formulation)
                .ok_or_else(|| format!("unknown formulation `{formulation}`"))?;
            for mesh in meshes.split('|') {
                let mesh =
                    MeshOrder::parse(mesh).ok_or_else(|| format!("unknown mesh `{mesh}`"))?;
                for execution in executions.split('|') {
                    let execution = ExecutionEnvelope::parse(execution)
                        .ok_or_else(|| format!("unknown execution `{execution}`"))?;
                    let cell = Envelope {
                        formulation,
                        mesh,
                        execution,
                    };
                    if !cells.contains(&cell) {
                        cells.push(cell);
                    }
                }
            }
        }
    }
    Ok((
        requirement.trim().to_string(),
        obligation.trim().to_string(),
        cells,
        boundary.to_string(),
    ))
}

fn package_of(manifest: &Path) -> Option<String> {
    let text = fs::read_to_string(manifest).ok()?;
    let mut in_package = false;
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            in_package = line == "[package]";
        } else if in_package {
            if let Some(rest) = line.strip_prefix("name") {
                let value = rest.trim_start().strip_prefix('=')?.trim();
                return Some(value.trim_matches('"').to_string());
            }
        }
    }
    None
}

fn strip_literals(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let chars: Vec<char> = line.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '/' if chars.get(i + 1) == Some(&'/') => break,
            '"' => {
                i += 1;
                while i < chars.len() && chars[i] != '"' {
                    if chars[i] == '\\' {
                        i += 1;
                    }
                    i += 1;
                }
            }
            '\'' if chars.get(i + 2) == Some(&'\'') => i += 2,
            '\'' if chars.get(i + 1) == Some(&'\\') && chars.get(i + 3) == Some(&'\'') => i += 3,
            c => out.push(c),
        }
        i += 1;
    }
    out
}

fn module_prefix(relative_to_package: &Path) -> Option<(Target, Vec<String>)> {
    let parts: Vec<String> = relative_to_package
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    let stem = |s: &str| s.trim_end_matches(".rs").to_string();
    match parts.first().map(String::as_str) {
        Some("src") => {
            let mut path: Vec<String> = parts[1..].iter().map(|p| stem(p)).collect();
            match path.last().map(String::as_str) {
                Some("lib") if path.len() == 1 => path.clear(),
                Some("mod") => {
                    path.pop();
                }
                Some("main") | None => return None,
                _ => {}
            }
            if path.first().map(String::as_str) == Some("bin") {
                return None;
            }
            Some((Target::Library, path))
        }
        Some("tests") if parts.len() >= 2 => {
            let target = stem(&parts[1]);
            let mut path: Vec<String> = parts[2..].iter().map(|p| stem(p)).collect();
            if path.last().map(String::as_str) == Some("mod") {
                path.pop();
            }
            Some((Target::Integration(target), path))
        }
        _ => None,
    }
}

/// Collect every witness tag under `root`/`crates`. Problems are returned alongside tags so
/// one malformed tag does not hide others.
pub fn scan(root: &Path) -> (Vec<Tag>, Vec<Problem>) {
    let mut tags = Vec::new();
    let mut problems = Vec::new();
    let mut files = Vec::new();
    collect_sources(&root.join("crates"), &mut files);
    files.sort();
    for file in files {
        let Ok(text) = fs::read_to_string(&file) else {
            continue;
        };
        if !text.lines().any(|l| l.trim_start().starts_with("// witness:")) {
            continue;
        }
        let relative = file.strip_prefix(root).unwrap_or(&file).to_path_buf();
        scan_file(root, &file, &relative, &text, &mut tags, &mut problems);
    }
    (tags, problems)
}

fn collect_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if path.is_dir() {
            if name == "target" || name.starts_with("target-") || name.starts_with('.') {
                continue;
            }
            collect_sources(&path, out);
        } else if name.ends_with(".rs") {
            out.push(path);
        }
    }
}

fn scan_file(
    root: &Path,
    file: &Path,
    relative: &Path,
    text: &str,
    tags: &mut Vec<Tag>,
    problems: &mut Vec<Problem>,
) {
    let mut manifest_dir = file.parent();
    let mut package = None;
    while let Some(dir) = manifest_dir {
        if dir.join("Cargo.toml").is_file() {
            package = package_of(&dir.join("Cargo.toml")).map(|p| (p, dir.to_path_buf()));
            if package.is_some() {
                break;
            }
        }
        if dir == root {
            break;
        }
        manifest_dir = dir.parent();
    }
    let display = relative.display().to_string();
    let Some((package, package_dir)) = package else {
        problems.push(Problem {
            location: display,
            message: "no owning Cargo package".into(),
        });
        return;
    };
    let Some((target, base_modules)) = file
        .strip_prefix(&package_dir)
        .ok()
        .and_then(module_prefix)
    else {
        problems.push(Problem {
            location: display,
            message: "witness tags are supported in library sources and tests/ targets only"
                .into(),
        });
        return;
    };
    let lines: Vec<&str> = text.lines().collect();
    // Inline module stack: (name, brace depth at which it opened).
    let mut stack: Vec<(String, i64)> = Vec::new();
    let mut depth: i64 = 0;
    // Pending tag texts for the current comment block, plus attributes seen.
    let mut pending: Vec<(usize, String)> = Vec::new();
    let mut attrs: Vec<String> = Vec::new();
    let mut in_block = false;
    let mut last_comment_was_tag_text = false;
    for (index, raw) in lines.iter().enumerate() {
        let trimmed = raw.trim();
        let location = format!("{}:{}", display, index + 1);
        if let Some(rest) = trimmed.strip_prefix("// witness:") {
            pending.push((index + 1, rest.trim().to_string()));
            in_block = true;
            last_comment_was_tag_text = true;
            continue;
        }
        if in_block && trimmed.starts_with("//") && !trimmed.starts_with("///") && last_comment_was_tag_text {
            let rest = trimmed.trim_start_matches('/').trim();
            if let Some((_, text)) = pending.last_mut() {
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(rest);
            }
            continue;
        }
        if in_block && trimmed.starts_with("#[") {
            attrs.push(trimmed.to_string());
            continue;
        }
        if in_block {
            // Must be the function.
            let signature = trimmed
                .trim_start_matches("pub(crate) ")
                .trim_start_matches("pub(super) ")
                .trim_start_matches("pub ")
                .trim_start_matches("async ");
            let function = signature
                .strip_prefix("fn ")
                .map(|r| r.split(|c: char| !(c.is_alphanumeric() || c == '_')).next().unwrap_or(""));
            match function {
                Some(function) if !function.is_empty() => {
                    let above_is_test = index > 0 && {
                        // attribute directly above the tag block
                        let mut k = pending[0].0 - 1; // 1-based line of first tag -> index of it is k-1
                        let mut found = false;
                        while k > 1 {
                            let prev = lines[k - 2].trim();
                            if prev.starts_with("#[") {
                                found |= is_test_attr(prev);
                                k -= 1;
                            } else {
                                break;
                            }
                        }
                        found
                    };
                    let is_test = above_is_test || attrs.iter().any(|a| is_test_attr(a));
                    let mut modules = base_modules.clone();
                    modules.extend(stack.iter().map(|(n, _)| n.clone()));
                    modules.push(function.to_string());
                    let locator = TestLocator {
                        package: package.clone(),
                        target: target.clone(),
                        name: modules.join("::"),
                        source: display.clone(),
                        line: index + 1,
                    };
                    if !is_test {
                        problems.push(Problem {
                            location: location.clone(),
                            message: format!("`{function}` is not a #[test] function"),
                        });
                    }
                    for (tag_line, text) in pending.drain(..) {
                        let location = format!("{display}:{tag_line}");
                        match parse_tag_text(&text) {
                            Ok((requirement, obligation, envelopes, boundary)) => tags.push(Tag {
                                requirement,
                                obligation,
                                envelopes,
                                boundary,
                                test: locator.clone(),
                            }),
                            Err(message) => problems.push(Problem { location, message }),
                        }
                    }
                }
                _ => {
                    problems.push(Problem {
                        location: format!("{display}:{}", pending[0].0),
                        message: "witness tag is not directly above a function".into(),
                    });
                    pending.clear();
                }
            }
            attrs.clear();
            in_block = false;
            last_comment_was_tag_text = false;
        }
        // Track inline modules and brace depth.
        let code = strip_literals(raw);
        let module_name = code
            .trim_start()
            .trim_start_matches("pub(crate) ")
            .trim_start_matches("pub(super) ")
            .trim_start_matches("pub ")
            .strip_prefix("mod ")
            .and_then(|r| {
                let name: String = r
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                let after = r[name.len()..].trim_start();
                (after.starts_with('{') && !name.is_empty()).then_some(name)
            });
        if let Some(name) = module_name {
            stack.push((name, depth));
        }
        for c in code.chars() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    while stack.last().is_some_and(|(_, d)| *d >= depth) {
                        stack.pop();
                    }
                }
                _ => {}
            }
        }
        // A comment-only or attribute-only line outside a tag block resets nothing.
    }
    if !pending.is_empty() {
        problems.push(Problem {
            location: format!("{display}:{}", pending[0].0),
            message: "witness tag at end of file".into(),
        });
    }
}

fn is_test_attr(attribute: &str) -> bool {
    let inner = attribute.trim_start_matches("#[").trim_end_matches(']').trim();
    inner == "test" || inner.ends_with("::test") || inner.starts_with("test(") || inner.contains("::test(")
}

/// Check tags against the register: the obligation exists, every envelope is required for it,
/// and no (obligation, envelope) boundary is empty. Returns every problem found.
pub fn validate_tags(tags: &[Tag], registry: &[&Requirement]) -> Vec<Problem> {
    let mut problems = Vec::new();
    let by_id: BTreeMap<&str, &Requirement> = registry.iter().map(|r| (r.id, *r)).collect();
    let mut seen = BTreeSet::new();
    for tag in tags {
        let location = format!("{}:{}", tag.test.source, tag.test.line);
        let Some(requirement) = by_id.get(tag.requirement.as_str()) else {
            problems.push(Problem {
                location,
                message: format!("unknown requirement `{}`", tag.requirement),
            });
            continue;
        };
        let Some(obligation) = requirement
            .obligations
            .iter()
            .find(|o| o.id == tag.obligation)
        else {
            problems.push(Problem {
                location,
                message: format!(
                    "requirement {} has no obligation `{}`",
                    tag.requirement, tag.obligation
                ),
            });
            continue;
        };
        for envelope in &tag.envelopes {
            if !obligation.required_envelopes.contains(envelope) {
                problems.push(Problem {
                    location: location.clone(),
                    message: format!(
                        "{}/{} does not require {}/{}/{}",
                        tag.requirement,
                        tag.obligation,
                        envelope.formulation.wire_name(),
                        envelope.mesh.wire_name(),
                        envelope.execution.wire_name()
                    ),
                });
            }
            if !seen.insert(format!(
                "{}/{}/{:?}/{}",
                tag.requirement,
                tag.obligation,
                envelope,
                tag.test.describe()
            )) {
                problems.push(Problem {
                    location: location.clone(),
                    message: "duplicate tag for the same test and cell".into(),
                });
            }
        }
    }
    problems
}
