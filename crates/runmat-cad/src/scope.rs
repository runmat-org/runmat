//! Name resolution without imports in a folder (report 12, design B).
//!
//! - A **folder** is the unit of scope: every design file and `.m` file in it is visible by
//!   name, with no import. Subfolders are not; other folders, packages and pinned
//!   cross-partition sources come in through `use`.
//! - **Declarations are kind-qualified** (`material.aluminum`, `part.bracket`): the kind keyword
//!   picks the namespace, so `geometry beam` and `mesh beam` never collide.
//! - **A bare name** resolves, first match wins: locals → the dialect's vocabulary (never
//!   shadowed: [`VALUE_CHOICES`]) → a same-folder `.m` (`private/` included) → project-path
//!   functions in root order → curated runtime builtins. A local that hides a folder function
//!   is reported; `function.n` reaches the function.
//! - **`pkg.fn`** reaches `+pkg/fn.m` on the path (MATLAB packages).
//! - Two declarations of one kind and name in a folder are an error naming both files; `use`
//!   cycles between folders are an error.
//!
//! The resolver works on folder index rows (a header-only scan per file), so resolution
//! touches the referring folder, its `use` targets and the path index, never the tree.

use crate::ids::FilePath;
use crate::value::VALUE_CHOICES;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// One folder's index row.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FolderIndex {
    pub folder: String,
    /// `(kind, name)` → the files declaring it (more than one is a duplicate).
    pub declarations: BTreeMap<(String, String), Vec<FilePath>>,
    /// `.m` stems in the folder.
    pub functions: BTreeSet<String>,
    /// `.m` stems in the folder's `private/`.
    pub private_functions: BTreeSet<String>,
    /// `use` lines of the folder's files (folder scope: one list per folder).
    pub uses: Vec<UseDecl>,
}

/// `use material aluminum from "./materials"`, `use "./materials" as lib`,
/// `use part wing_skeleton from "//wing/skeleton" @ "R3"`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UseDecl {
    /// `Some((kind, name))`: one symbol; `None`: the whole folder under `alias`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub symbol: Option<(String, String)>,
    /// A folder path (project-root relative) or a package name.
    pub from: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alias: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ProjectScope {
    pub folders: BTreeMap<String, FolderIndex>,
    /// Path roots in manifest order (folders whose functions are on the path).
    pub roots: Vec<String>,
    /// `+pkg` folders: package → its functions.
    pub packages: BTreeMap<String, BTreeSet<String>>,
    /// Registry words of the dialect at the file's `std` version.
    pub vocabulary: BTreeSet<String>,
    /// Curated pure runtime builtins callable from design files.
    pub builtins: BTreeSet<String>,
    /// Kind keywords that qualify declarations.
    pub kinds: BTreeSet<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "binding")]
pub enum Binding {
    Local {
        name: String,
    },
    Vocabulary {
        word: String,
    },
    FolderFunction {
        folder: String,
        name: String,
        private: bool,
    },
    PathFunction {
        folder: String,
        name: String,
    },
    PackageFunction {
        package: String,
        name: String,
    },
    Builtin {
        name: String,
    },
    Declaration {
        kind: String,
        name: String,
        file: FilePath,
        via_use: bool,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "problem")]
pub enum ScopeProblem {
    Unresolved {
        name: String,
        candidates: Vec<String>,
    },
    Duplicate {
        kind: String,
        name: String,
        files: Vec<FilePath>,
    },
    UnknownFolder {
        folder: String,
    },
    UseCycle {
        cycle: Vec<String>,
    },
}

/// Information that is not an error.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "hint")]
pub enum ScopeHint {
    /// `design.resolve.local_shadows_function`.
    LocalShadowsFunction { name: String },
    /// A same-folder `.m` shares a vocabulary word: reach it as `function.<name>`.
    VocabularyWins { name: String },
    /// A folder function hides a path function of the same name (MATLAB order).
    FolderBeatsPath { name: String },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Resolution {
    pub binding: Binding,
    #[serde(default)]
    pub hints: Vec<ScopeHint>,
}

impl ProjectScope {
    fn folder(&self, f: &str) -> Result<&FolderIndex, ScopeProblem> {
        self.folders
            .get(f)
            .ok_or_else(|| ScopeProblem::UnknownFolder { folder: f.into() })
    }

    fn path_function(&self, name: &str) -> Option<&str> {
        self.roots
            .iter()
            .find(|r| {
                self.folders
                    .get(*r)
                    .map(|f| f.functions.contains(name))
                    .unwrap_or(false)
            })
            .map(|s| s.as_str())
    }

    fn candidates(&self, folder: &FolderIndex, name: &str) -> Vec<String> {
        let close = |c: &str| c.contains(name) || name.contains(c) || edit_distance(c, name) <= 2;
        let mut out: BTreeSet<String> = BTreeSet::new();
        for f in folder.functions.iter().filter(|c| close(c)) {
            out.insert(f.clone());
        }
        for (k, n) in folder.declarations.keys().filter(|(_, n)| close(n)) {
            out.insert(format!("{}.{}", k, n));
        }
        for v in self.vocabulary.iter().filter(|c| close(c)) {
            out.insert(v.clone());
        }
        out.into_iter().take(8).collect()
    }

    /// A bare single-segment name in `folder`, given the enclosing locals.
    pub fn resolve_bare(
        &self,
        folder: &str,
        locals: &BTreeSet<String>,
        name: &str,
    ) -> Result<Resolution, ScopeProblem> {
        let f = self.folder(folder)?;
        let mut hints = Vec::new();
        let folder_fn = f.functions.contains(name) || f.private_functions.contains(name);
        if locals.contains(name) {
            if folder_fn || self.path_function(name).is_some() {
                hints.push(ScopeHint::LocalShadowsFunction { name: name.into() });
            }
            return Ok(Resolution {
                binding: Binding::Local { name: name.into() },
                hints,
            });
        }
        if !VALUE_CHOICES.vocabulary_shadowable && self.vocabulary.contains(name) {
            if folder_fn {
                hints.push(ScopeHint::VocabularyWins { name: name.into() });
            }
            return Ok(Resolution {
                binding: Binding::Vocabulary { word: name.into() },
                hints,
            });
        }
        if folder_fn {
            if self
                .path_function(name)
                .map(|p| p != folder)
                .unwrap_or(false)
            {
                hints.push(ScopeHint::FolderBeatsPath { name: name.into() });
            }
            return Ok(Resolution {
                binding: Binding::FolderFunction {
                    folder: folder.into(),
                    name: name.into(),
                    private: !f.functions.contains(name),
                },
                hints,
            });
        }
        if VALUE_CHOICES.vocabulary_shadowable && self.vocabulary.contains(name) {
            return Ok(Resolution {
                binding: Binding::Vocabulary { word: name.into() },
                hints,
            });
        }
        if let Some(root) = self.path_function(name) {
            return Ok(Resolution {
                binding: Binding::PathFunction {
                    folder: root.into(),
                    name: name.into(),
                },
                hints,
            });
        }
        if self.builtins.contains(name) {
            return Ok(Resolution {
                binding: Binding::Builtin { name: name.into() },
                hints,
            });
        }
        Err(ScopeProblem::Unresolved {
            name: name.into(),
            candidates: self.candidates(f, name),
        })
    }

    fn declaration_in(
        &self,
        f: &FolderIndex,
        kind: &str,
        name: &str,
        via_use: bool,
    ) -> Option<Result<Resolution, ScopeProblem>> {
        let files = f.declarations.get(&(kind.to_owned(), name.to_owned()))?;
        Some(if files.len() > 1 {
            Err(ScopeProblem::Duplicate {
                kind: kind.into(),
                name: name.into(),
                files: files.clone(),
            })
        } else {
            Ok(Resolution {
                binding: Binding::Declaration {
                    kind: kind.into(),
                    name: name.into(),
                    file: files[0].clone(),
                    via_use,
                },
                hints: Vec::new(),
            })
        })
    }

    /// A qualified name `a.b…` in `folder`.
    pub fn resolve_qualified(
        &self,
        folder: &str,
        segments: &[&str],
    ) -> Result<Resolution, ScopeProblem> {
        let f = self.folder(folder)?;
        let joined = segments.join(".");
        match segments {
            ["function", name, ..] => {
                if f.functions.contains(*name) || f.private_functions.contains(*name) {
                    return Ok(Resolution {
                        binding: Binding::FolderFunction {
                            folder: folder.into(),
                            name: (*name).into(),
                            private: !f.functions.contains(*name),
                        },
                        hints: Vec::new(),
                    });
                }
                if let Some(root) = self.path_function(name) {
                    return Ok(Resolution {
                        binding: Binding::PathFunction {
                            folder: root.into(),
                            name: (*name).into(),
                        },
                        hints: Vec::new(),
                    });
                }
                Err(ScopeProblem::Unresolved {
                    name: joined,
                    candidates: self.candidates(f, name),
                })
            }
            [kind, name, ..] if self.kinds.contains(*kind) => {
                if let Some(r) = self.declaration_in(f, kind, name, false) {
                    return r;
                }
                if VALUE_CHOICES.packages_and_use {
                    for u in &f.uses {
                        match &u.symbol {
                            Some((k, n))
                                if k == kind
                                    && (u.alias.as_deref() == Some(*name)
                                        || (u.alias.is_none() && n == name)) =>
                            {
                                let target = self.folder(&u.from)?;
                                if let Some(r) = self.declaration_in(target, k, n, true) {
                                    return r;
                                }
                            }
                            _ => {}
                        }
                    }
                }
                Err(ScopeProblem::Unresolved {
                    name: joined,
                    candidates: self.candidates(f, name),
                })
            }
            [alias, rest @ ..] if !rest.is_empty() => {
                // A folder alias from `use "<folder>" as <alias>`.
                if VALUE_CHOICES.packages_and_use {
                    if let Some(u) = f
                        .uses
                        .iter()
                        .find(|u| u.symbol.is_none() && u.alias.as_deref() == Some(*alias))
                    {
                        return self.resolve_qualified(&u.from.clone(), rest);
                    }
                    // A MATLAB package function.
                    if let (Some(fns), [name]) = (self.packages.get(*alias), rest) {
                        if fns.contains(*name) {
                            return Ok(Resolution {
                                binding: Binding::PackageFunction {
                                    package: (*alias).into(),
                                    name: (*name).into(),
                                },
                                hints: Vec::new(),
                            });
                        }
                    }
                }
                Err(ScopeProblem::Unresolved {
                    name: joined,
                    candidates: self.candidates(f, alias),
                })
            }
            _ => Err(ScopeProblem::Unresolved {
                name: joined,
                candidates: Vec::new(),
            }),
        }
    }

    /// Every duplicate declaration in the project.
    pub fn duplicates(&self) -> Vec<ScopeProblem> {
        let mut out = Vec::new();
        for f in self.folders.values() {
            for ((k, n), files) in &f.declarations {
                if files.len() > 1 {
                    out.push(ScopeProblem::Duplicate {
                        kind: k.clone(),
                        name: n.clone(),
                        files: files.clone(),
                    });
                }
            }
        }
        out
    }

    /// `use` cycles between folders (the Go rule), each reported once from its least folder.
    pub fn use_cycles(&self) -> Vec<ScopeProblem> {
        let edges: BTreeMap<&str, BTreeSet<&str>> = self
            .folders
            .iter()
            .map(|(k, f)| (k.as_str(), f.uses.iter().map(|u| u.from.as_str()).collect()))
            .collect();
        let mut out = Vec::new();
        let mut reported: BTreeSet<Vec<String>> = BTreeSet::new();
        fn dfs<'a>(
            n: &'a str,
            edges: &BTreeMap<&'a str, BTreeSet<&'a str>>,
            stack: &mut Vec<&'a str>,
            done: &mut BTreeSet<&'a str>,
            found: &mut Vec<Vec<String>>,
        ) {
            if let Some(i) = stack.iter().position(|s| *s == n) {
                found.push(stack[i..].iter().map(|s| s.to_string()).collect());
                return;
            }
            if done.contains(n) {
                return;
            }
            stack.push(n);
            for m in edges.get(n).into_iter().flatten() {
                dfs(m, edges, stack, done, found);
            }
            stack.pop();
            done.insert(n);
        }
        let mut done = BTreeSet::new();
        let mut found = Vec::new();
        for n in edges.keys() {
            dfs(n, &edges, &mut Vec::new(), &mut done, &mut found);
        }
        for mut c in found {
            // Rotate so the least folder leads, for a canonical report.
            let k = c
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.cmp(b.1))
                .map(|(i, _)| i)
                .unwrap_or(0);
            c.rotate_left(k);
            if reported.insert(c.clone()) {
                out.push(ScopeProblem::UseCycle { cycle: c });
            }
        }
        out
    }
}

fn edit_distance(a: &str, b: &str) -> usize {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut cur = vec![i; b.len() + 1];
        for j in 1..=b.len() {
            cur[j] = (prev[j] + 1)
                .min(cur[j - 1] + 1)
                .min(prev[j - 1] + usize::from(a[i - 1] != b[j - 1]));
        }
        prev = cur;
    }
    prev[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scope() -> ProjectScope {
        let mut s = ProjectScope {
            vocabulary: ["line", "extrude", "max"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            builtins: ["interp1"].iter().map(|s| s.to_string()).collect(),
            kinds: ["material", "part", "geometry", "mesh"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            roots: vec!["lib".into()],
            ..Default::default()
        };
        s.packages.insert(
            "aero".into(),
            ["airDensity".to_string()].into_iter().collect(),
        );
        let mut wing = FolderIndex {
            folder: "wing".into(),
            ..Default::default()
        };
        wing.functions.extend([
            "calculateAirDensity".to_string(),
            "line".to_string(),
            "helper".to_string(),
        ]);
        wing.private_functions.insert("secret".into());
        wing.declarations.insert(
            ("material".into(), "calculateAirDensity".into()),
            vec![FilePath::new("wing/m.fea")],
        );
        wing.declarations.insert(
            ("geometry".into(), "beam".into()),
            vec![FilePath::new("wing/a.fea")],
        );
        wing.declarations.insert(
            ("mesh".into(), "beam".into()),
            vec![FilePath::new("wing/a.fea")],
        );
        wing.declarations.insert(
            ("part".into(), "rib".into()),
            vec![
                FilePath::new("wing/rib.part"),
                FilePath::new("wing/rib2.part"),
            ],
        );
        wing.uses.push(UseDecl {
            symbol: Some(("material".into(), "al".into())),
            from: "materials".into(),
            alias: None,
            revision: None,
        });
        wing.uses.push(UseDecl {
            symbol: None,
            from: "materials".into(),
            alias: Some("lib".into()),
            revision: None,
        });
        let mut mats = FolderIndex {
            folder: "materials".into(),
            ..Default::default()
        };
        mats.declarations.insert(
            ("material".into(), "al".into()),
            vec![FilePath::new("materials/al.fea")],
        );
        let mut lib = FolderIndex {
            folder: "lib".into(),
            ..Default::default()
        };
        lib.functions
            .extend(["helper".to_string(), "pathOnly".to_string()]);
        s.folders.insert("wing".into(), wing);
        s.folders.insert("materials".into(), mats);
        s.folders.insert("lib".into(), lib);
        s
    }

    #[test]
    fn bare_names_follow_the_order() {
        let s = scope();
        let none = BTreeSet::new();
        let r = s
            .resolve_bare("wing", &none, "calculateAirDensity")
            .unwrap();
        assert!(matches!(r.binding, Binding::FolderFunction { .. }));
        let r = s.resolve_bare("wing", &none, "line").unwrap();
        assert!(
            matches!(r.binding, Binding::Vocabulary { .. }),
            "vocabulary is not shadowed"
        );
        assert_eq!(
            r.hints,
            vec![ScopeHint::VocabularyWins {
                name: "line".into()
            }]
        );
        let r = s.resolve_bare("wing", &none, "helper").unwrap();
        assert_eq!(
            r.hints,
            vec![ScopeHint::FolderBeatsPath {
                name: "helper".into()
            }]
        );
        assert!(matches!(
            s.resolve_bare("wing", &none, "pathOnly").unwrap().binding,
            Binding::PathFunction { .. }
        ));
        assert!(matches!(
            s.resolve_bare("wing", &none, "secret").unwrap().binding,
            Binding::FolderFunction { private: true, .. }
        ));
        assert!(matches!(
            s.resolve_bare("wing", &none, "interp1").unwrap().binding,
            Binding::Builtin { .. }
        ));
        let locals: BTreeSet<String> = ["helper".to_string()].into_iter().collect();
        let r = s.resolve_bare("wing", &locals, "helper").unwrap();
        assert_eq!(
            r.hints,
            vec![ScopeHint::LocalShadowsFunction {
                name: "helper".into()
            }]
        );
        match s.resolve_bare("wing", &none, "calculateAirDensty") {
            Err(ScopeProblem::Unresolved { candidates, .. }) => {
                assert!(candidates.contains(&"calculateAirDensity".to_string()))
            }
            other => panic!("{:?}", other),
        }
    }

    #[test]
    fn qualified_names_kinds_uses_and_packages() {
        let s = scope();
        let r = s
            .resolve_qualified("wing", &["material", "calculateAirDensity"])
            .unwrap();
        assert!(matches!(r.binding, Binding::Declaration { ref kind, .. } if kind == "material"));
        assert!(matches!(
            s.resolve_qualified("wing", &["function", "calculateAirDensity"])
                .unwrap()
                .binding,
            Binding::FolderFunction { .. }
        ));
        assert!(
            matches!(s.resolve_qualified("wing", &["geometry", "beam"]).unwrap().binding, Binding::Declaration { ref kind, .. } if kind == "geometry")
        );
        assert!(
            matches!(s.resolve_qualified("wing", &["mesh", "beam"]).unwrap().binding, Binding::Declaration { ref kind, .. } if kind == "mesh")
        );
        assert!(matches!(
            s.resolve_qualified("wing", &["material", "al"])
                .unwrap()
                .binding,
            Binding::Declaration { via_use: true, .. }
        ));
        assert!(matches!(
            s.resolve_qualified("wing", &["lib", "material", "al"])
                .unwrap()
                .binding,
            Binding::Declaration { .. }
        ));
        assert!(matches!(
            s.resolve_qualified("wing", &["aero", "airDensity"])
                .unwrap()
                .binding,
            Binding::PackageFunction { .. }
        ));
        assert!(matches!(
            s.resolve_qualified("wing", &["part", "rib"]),
            Err(ScopeProblem::Duplicate { .. })
        ));
        assert_eq!(s.duplicates().len(), 1);
    }

    #[test]
    fn use_cycles_are_found_once() {
        let mut s = scope();
        s.folders.get_mut("materials").unwrap().uses.push(UseDecl {
            symbol: None,
            from: "wing".into(),
            alias: Some("w".into()),
            revision: None,
        });
        let c = s.use_cycles();
        assert_eq!(
            c,
            vec![ScopeProblem::UseCycle {
                cycle: vec!["materials".into(), "wing".into()]
            }]
        );
    }
}
