use crate::{HirProgram, HirStmt, SemanticError};

pub fn collect_imports(prog: &HirProgram) -> Vec<(Vec<String>, bool)> {
    let mut imports = Vec::new();
    for stmt in &prog.body {
        if let HirStmt::Import { path, wildcard, .. } = stmt {
            imports.push((path.clone(), *wildcard));
        }
    }
    imports
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizedImport {
    pub path: String,
    pub wildcard: bool,
    pub unqualified: Option<String>,
}

pub fn normalize_imports(prog: &HirProgram) -> Vec<NormalizedImport> {
    let mut out = Vec::new();
    for stmt in &prog.body {
        if let HirStmt::Import { path, wildcard, .. } = stmt {
            let path_str = path.join(".");
            let last = if *wildcard {
                None
            } else {
                path.last().cloned()
            };
            out.push(NormalizedImport {
                path: path_str,
                wildcard: *wildcard,
                unqualified: last,
            });
        }
    }
    out
}

pub fn validate_imports(prog: &HirProgram) -> Result<(), SemanticError> {
    use std::collections::{HashMap, HashSet};

    let norms = normalize_imports(prog);
    let mut seen_exact: HashSet<(String, bool)> = HashSet::new();
    for n in &norms {
        if !seen_exact.insert((n.path.clone(), n.wildcard)) {
            return Err(SemanticError::new(format!(
                "duplicate import '{}{}'",
                n.path,
                if n.wildcard { ".*" } else { "" }
            )));
        }
    }

    let mut by_name: HashMap<String, Vec<String>> = HashMap::new();
    for n in &norms {
        if !n.wildcard {
            if let Some(uq) = &n.unqualified {
                by_name.entry(uq.clone()).or_default().push(n.path.clone());
            }
        }
    }
    for (uq, sources) in by_name {
        if sources.len() > 1 {
            return Err(SemanticError::new(format!(
                "ambiguous import for '{}': {}",
                uq,
                sources.join(", ")
            )));
        }
    }
    Ok(())
}
