//! Design files composed as modules: folders are scopes, and designs name each other by
//! kind-qualified symbol (report 12, design B, §3).
//!
//! - **Same folder:** every design and `.m` file in a folder is in scope with no `use`.
//!   `use part seat_frame as left` places the folder's `part seat_frame`.
//! - **Other folders:** `use part seat_frame from "./cabin" as left` names one design in another
//!   folder; a file-level `use "./cabin" as cabin` brings a folder in under an alias, and
//!   `use cabin.part.seat_frame as left` names a design in it.
//! - **Packages:** a `+hardware` folder in the referring folder or any folder above it is the
//!   package `hardware` (`use part rivet from hardware as r`, nested `+a/+b` is `a.b`), as MATLAB
//!   reaches `+pkg/fn.m` as `pkg.fn`.
//! - **Pins:** `use part wing_skeleton from "//wing/skeleton" @ "R3" as skeleton`; `//` is the
//!   project root (the outermost folder holding `runmat.toml`).
//! - A design named by its file path (`from "seat_frame.part"`, `file = "arm.part"`) is an
//!   error that says which symbol to write. Non-design sources (STEP) keep their paths.
//!
//! The model keeps one identity for a definition: its resolved file ([`DefinitionRef`]). Text
//! holds the symbol. [`bind`] turns the symbols a parse lowered into files through a
//! [`ModuleScope`]; [`unbound`] turns files back into symbols, relative to the printing file, so a
//! printed document always reads in the module form.
//!
//! [`DefinitionRef`]: crate::assembly::model::DefinitionRef

use crate::ids::FilePath;
use crate::scope::{Binding, FolderIndex, ProjectScope, ScopeProblem, UseDecl};
use runmat_design_syntax::ast::{Headed, Item, Statement, UseSource};
use runmat_design_syntax::json::Json;
use runmat_design_syntax::{parse_source, Dialect};
use std::collections::{BTreeMap, BTreeSet};

/// The kinds a design declares, with their file extensions.
pub const DESIGN_KINDS: &[(&str, &str)] = &[
    ("part", ".part"),
    ("assembly", ".assembly"),
    ("circuit", ".circuit"),
    ("drawing", ".drawing"),
];

/// The kind a design file's extension declares (`.circuit.layout` is not a definition).
pub fn kind_of_file(path: &str) -> Option<&'static str> {
    DESIGN_KINDS
        .iter()
        .find(|(_, ext)| path.ends_with(ext))
        .map(|(k, _)| *k)
}

/// Whether a `from` path names one design file (the file form) rather than a folder.
pub fn is_file_path(from: &str) -> bool {
    kind_of_file(from).is_some() || from.ends_with(".circuit.layout")
}

/// The folder holding `path`: `""` for a relative top-level file, `"/"` for a file at the root
/// of an absolute workspace (`/airliner.assembly`).
pub fn folder_of(path: &str) -> &str {
    match path.rsplit_once('/') {
        Some(("", _)) if path.starts_with('/') => "/",
        Some((d, _)) => d,
        None => "",
    }
}

/// `a/./b/../c` → `a/c`, keeping a leading `/`.
pub fn normalize(path: &str) -> String {
    let mut out: Vec<&str> = Vec::new();
    for seg in path.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                out.pop();
            }
            s => out.push(s),
        }
    }
    let joined = out.join("/");
    if path.starts_with('/') {
        format!("/{joined}")
    } else {
        joined
    }
}

fn join(folder: &str, rel: &str) -> String {
    if folder.is_empty() {
        normalize(rel)
    } else {
        normalize(&format!("{folder}/{rel}"))
    }
}

fn stem(file: &str) -> &str {
    let name = file.rsplit('/').next().unwrap_or(file);
    name.split('.').next().unwrap_or(name)
}

/// The name standing for a file's single design: its stem, each character a name cannot hold
/// written `_` (`engine-a.part` → `engine_a`).
pub fn stem_name(file: &str) -> String {
    let mut name: String = stem(file)
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if name.chars().next().is_none_or(|c| c.is_ascii_digit()) {
        name.insert(0, '_');
    }
    name
}

/// The ancestors of `folder`, nearest first, ending at the workspace root (`""` or `/`).
pub fn ancestors(folder: &str) -> Vec<String> {
    let mut out = vec![folder.to_string()];
    let mut at = folder.to_string();
    while !at.is_empty() && at != "/" {
        at = match at.rsplit_once('/') {
            Some(("", _)) if at.starts_with('/') => "/".into(),
            Some((d, _)) => d.into(),
            None => String::new(),
        };
        out.push(at.clone());
    }
    out
}

/// The `+a/+b` folder path of the package `a.b`.
fn package_path(package: &str) -> String {
    package
        .split('.')
        .map(|s| format!("+{s}"))
        .collect::<Vec<_>>()
        .join("/")
}

/// The `kind name` pairs a design file declares, and its file-level and occurrence `use` lines:
/// a syntax-only scan, no lowering.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FileHeader {
    pub declarations: Vec<(String, String)>,
    /// File-level `use` lines, `from` as written.
    pub uses: Vec<UseDecl>,
    /// Occurrence `use` lines that name another folder or package (`from` as written).
    pub occurrence_froms: Vec<String>,
}

pub fn scan(path: &str, text: &str) -> FileHeader {
    let mut out = FileHeader::default();
    let Some(dialect) = Dialect::for_path(path) else {
        return out;
    };
    let kind = kind_of_file(path);
    let syntax = parse_source(dialect, text);
    for item in syntax.file().items() {
        match item {
            Item::Declaration(d) => {
                let Some(name) = d.name() else { continue };
                let keyword = d.keyword();
                if Some(keyword.as_str()) == kind {
                    out.declarations.push((keyword.clone(), name.text()));
                }
                for statement in d.block().map(|b| b.statements()).unwrap_or_default() {
                    match &statement {
                        // A part file's further outputs are parts of their own.
                        Statement::Value(v) if keyword == "part" && v.keyword() == "output" => {
                            if let Some(n) = v.name() {
                                let pair = ("part".to_string(), n.text());
                                if !out.declarations.contains(&pair) {
                                    out.declarations.push(pair);
                                }
                            }
                        }
                        Statement::Use(u) => {
                            if let (Some(_), Some(source)) = (u.symbol(), u.source()) {
                                match source {
                                    UseSource::Path(p) if !is_file_path(&p) => {
                                        out.occurrence_froms.push(p)
                                    }
                                    UseSource::Package(s) => out.occurrence_froms.push(s.join(".")),
                                    _ => {}
                                }
                            } else if let Some(UseSource::Package(s)) = u.source() {
                                if s.len() == 3 {
                                    out.occurrence_froms.push(s[0].clone());
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            Item::Use(u) => {
                let from = match u.source() {
                    Some(UseSource::Path(p)) => p,
                    Some(UseSource::Package(s)) => s.join("."),
                    None => continue,
                };
                out.uses.push(UseDecl {
                    symbol: u.symbol().map(|(k, n)| (k.text(), n.text())),
                    from,
                    alias: u.alias().map(|a| a.text()),
                    revision: u.pin(),
                });
            }
            _ => {}
        }
    }
    out
}

/// The folder scope of a set of workspace files: one index row per folder, `.m` functions
/// included, and the packages.
#[derive(Clone, Debug, Default)]
pub struct ModuleScope {
    pub scope: ProjectScope,
    /// The project root (`//`): the outermost folder holding a `runmat.toml`.
    pub root: Option<String>,
    /// Each design file's declared `(kind, name)` pairs.
    pub declared: BTreeMap<String, Vec<(String, String)>>,
    /// Folder-to-folder edges from occurrence `use` lines, for the cycle check.
    occurrence_edges: BTreeMap<String, BTreeSet<String>>,
}

impl ModuleScope {
    /// A scope over `files` (workspace path → text). Text is read for design files only; an
    /// `.m` file contributes its stem.
    pub fn from_files<'a>(files: impl IntoIterator<Item = (&'a str, &'a str)>) -> Self {
        let mut s = ModuleScope::default();
        s.scope.kinds = DESIGN_KINDS.iter().map(|(k, _)| k.to_string()).collect();
        let mut headers: Vec<(String, FileHeader)> = Vec::new();
        let mut roots: Vec<String> = Vec::new();
        for (path, text) in files {
            let folder = folder_of(path).to_string();
            if path.ends_with("/runmat.toml") || path == "runmat.toml" {
                roots.push(folder);
                continue;
            }
            if let Some(m) = path.strip_suffix(".m") {
                let fname = stem(m).to_string();
                let (parent, last) = folder.rsplit_once('/').unwrap_or(("", folder.as_str()));
                if last == "private" {
                    s.row(parent).private_functions.insert(fname);
                } else {
                    s.row(&folder).functions.insert(fname.clone());
                    if last.starts_with('+') {
                        let package: Vec<&str> = folder
                            .split('/')
                            .filter_map(|seg| seg.strip_prefix('+'))
                            .collect();
                        s.scope
                            .packages
                            .entry(package.join("."))
                            .or_default()
                            .insert(fname);
                    }
                }
                continue;
            }
            if kind_of_file(path).is_none() {
                continue;
            }
            let header = scan(path, text);
            for (k, n) in &header.declarations {
                s.row(&folder)
                    .declarations
                    .entry((k.clone(), n.clone()))
                    .or_default()
                    .push(FilePath::new(path));
            }
            s.declared
                .insert(path.to_string(), header.declarations.clone());
            headers.push((path.to_string(), header));
        }
        s.root = roots.into_iter().min_by_key(|r| r.len());
        // `use` targets resolve against their file's folder, once every folder is known.
        for (path, header) in headers {
            let folder = folder_of(&path).to_string();
            for u in header.uses {
                let target = s.module_folder(&folder, &u.from).unwrap_or(u.from.clone());
                s.row(&folder).uses.push(UseDecl { from: target, ..u });
            }
            for from in header.occurrence_froms {
                if let Ok(target) = s.module_folder(&folder, &from) {
                    if target != folder {
                        s.occurrence_edges
                            .entry(folder.clone())
                            .or_default()
                            .insert(target);
                    }
                }
            }
        }
        s
    }

    fn row(&mut self, folder: &str) -> &mut FolderIndex {
        self.scope
            .folders
            .entry(folder.to_string())
            .or_insert_with(|| FolderIndex {
                folder: folder.to_string(),
                ..Default::default()
            })
    }

    /// The folder a `from` names, seen from `folder`: a path (`./cabin`, `../wing`, `//lib`),
    /// a folder alias of `folder`'s `use` lines, or a package.
    pub fn module_folder(&self, folder: &str, from: &str) -> Result<String, String> {
        if let Some(rest) = from.strip_prefix("//") {
            let root = self.root.as_deref().ok_or_else(|| {
                format!("`{from}` is root-relative, and no runmat.toml marks the project root")
            })?;
            return Ok(join(root, rest));
        }
        if from.starts_with('.') || from.starts_with('/') || from.contains('/') {
            return Ok(if from.starts_with('/') {
                normalize(from)
            } else {
                join(folder, from)
            });
        }
        if let Some(u) = self.scope.folders.get(folder).and_then(|f| {
            f.uses
                .iter()
                .find(|u| u.symbol.is_none() && u.alias.as_deref() == Some(from))
        }) {
            return Ok(u.from.clone());
        }
        for at in ancestors(folder) {
            let candidate = join(&at, &package_path(from));
            if self.scope.folders.contains_key(&candidate) {
                return Ok(candidate);
            }
        }
        Err(format!(
            "`{from}` is not a folder alias of this folder or a package (`+{}`) in it or above it",
            from.replace('.', "/+")
        ))
    }

    /// The file declaring `kind name`, seen from the file `doc`, through `from` when given.
    pub fn locate(
        &self,
        doc: &str,
        kind: &str,
        name: &str,
        from: Option<&str>,
    ) -> Result<String, String> {
        let folder = folder_of(doc).to_string();
        let target = match from {
            None => folder.clone(),
            Some(f) => self.module_folder(&folder, f)?,
        };
        if !self.scope.folders.contains_key(&target) {
            return Err(match from {
                None => format!("no `{kind} {name}` in this folder"),
                Some(f) => format!("the folder `{f}` holds no designs (or isn't in the workspace)"),
            });
        }
        match self.scope.resolve_qualified(&target, &[kind, name]) {
            Ok(r) => match r.binding {
                Binding::Declaration { file, .. } => Ok(file.0),
                other => Err(format!("`{kind}.{name}` is {other:?}, not a design")),
            },
            Err(ScopeProblem::Duplicate { files, .. }) => Err(format!(
                "`{kind} {name}` is declared twice in one folder: {}",
                files
                    .iter()
                    .map(|f| f.0.as_str())
                    .collect::<Vec<_>>()
                    .join(" and ")
            )),
            Err(ScopeProblem::Unresolved { candidates, .. }) => {
                let place = match from {
                    None => "this folder".to_string(),
                    Some(f) => format!("`{f}`"),
                };
                let mut m = format!("no `{kind} {name}` in {place}");
                if !candidates.is_empty() {
                    m.push_str(&format!(" (did you mean {}?)", candidates.join(", ")));
                }
                Err(m)
            }
            Err(e) => Err(format!("{e:?}")),
        }
    }

    /// The name a file's design goes by: its declaration's (a file declares one), else its stem.
    pub fn name_of(&self, file: &str) -> String {
        self.declared
            .get(file)
            .and_then(|d| d.first())
            .map(|(_, n)| n.clone())
            .unwrap_or_else(|| stem_name(file))
    }

    /// Duplicate declarations and `use` cycles between folders.
    pub fn problems(&self) -> Vec<ScopeProblem> {
        let mut out = self.scope.duplicates();
        let mut with_edges = self.scope.clone();
        for (from, targets) in &self.occurrence_edges {
            let row = with_edges
                .folders
                .entry(from.clone())
                .or_insert_with(|| FolderIndex {
                    folder: from.clone(),
                    ..Default::default()
                });
            for t in targets {
                row.uses.push(UseDecl {
                    symbol: None,
                    from: t.clone(),
                    alias: None,
                    revision: None,
                });
            }
        }
        out.extend(with_edges.use_cycles());
        out
    }
}

/// Text for a scope problem.
pub fn problem_text(p: &ScopeProblem) -> String {
    match p {
        ScopeProblem::Duplicate { kind, name, files } => format!(
            "`{kind} {name}` is declared twice in one folder: {}",
            files
                .iter()
                .map(|f| f.0.as_str())
                .collect::<Vec<_>>()
                .join(" and ")
        ),
        ScopeProblem::UseCycle { cycle } => format!(
            "folders use each other in a cycle: {} → {}",
            cycle.join(" → "),
            cycle.first().map(String::as_str).unwrap_or("")
        ),
        ScopeProblem::UnknownFolder { folder } => format!("no folder `{folder}`"),
        ScopeProblem::Unresolved { name, .. } => format!("`{name}` doesn't resolve"),
    }
}

// ------------------------------------------------------------------------- bind and unbound

/// A design named by symbol, as a parse lowered it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SymbolRef {
    pub kind: String,
    pub name: String,
    pub from: Option<String>,
    /// The JSON path of the object holding it (`occurrences[2].source.definition`).
    pub at: String,
}

/// The keys whose value is a design reference `{kind, name, from?}`: an occurrence's
/// `definition` and a fastener rule's `family`.
fn is_definition_key(key: &str) -> bool {
    key == "definition" || key == "family"
}

enum Site {
    /// `{kind, name, from?}` → `{kind, file, part?}`.
    Definition,
    /// `{frame = interface, part = n, from?}` → `{frame = interface, file}`.
    Interface,
    /// `{part = n, from?}` (or `assembly = n`) → `{file}`.
    Model,
}

fn site(key: &str, obj: &[(String, Json)]) -> Option<(Site, String, String)> {
    let get = |k: &str| obj.iter().find(|(x, _)| x == k).map(|(_, v)| v);
    if get("file").is_some() {
        return None;
    }
    let text = |v: Option<&Json>| v.and_then(Json::as_str).map(str::to_string);
    if is_definition_key(key) {
        if let (Some(k), Some(n)) = (text(get("kind")), text(get("name"))) {
            return Some((Site::Definition, k, n));
        }
    }
    if text(get("frame")).as_deref() == Some("interface") {
        if let Some(n) = text(get("part")) {
            return Some((Site::Interface, "part".into(), n));
        }
    }
    if key == "model" {
        for (k, _) in DESIGN_KINDS {
            if let Some(n) = text(get(k)) {
                return Some((Site::Model, (*k).into(), n));
            }
        }
    }
    None
}

type ModuleVisitor<'a> = dyn FnMut(&mut Vec<(String, Json)>, Site, String, String, &str) + 'a;

fn walk(json: &mut Json, key: &str, at: &str, visit: &mut ModuleVisitor<'_>) {
    match json {
        Json::Object(obj) => {
            if let Some((s, k, n)) = site(key, obj) {
                visit(obj, s, k, n, at);
                return;
            }
            for (k, v) in obj.iter_mut() {
                let path = if at.is_empty() {
                    k.clone()
                } else {
                    format!("{at}.{k}")
                };
                walk(v, k, &path, visit);
            }
        }
        Json::Array(items) => {
            for (i, v) in items.iter_mut().enumerate() {
                walk(v, key, &format!("{at}[{i}]"), visit);
            }
        }
        _ => {}
    }
}

fn take(obj: &mut Vec<(String, Json)>, key: &str) -> Option<Json> {
    let i = obj.iter().position(|(k, _)| k == key)?;
    Some(obj.remove(i).1)
}

fn put_first(obj: &mut Vec<(String, Json)>, after: Option<&str>, key: &str, value: Json) {
    let at = after
        .and_then(|a| obj.iter().position(|(k, _)| k == a))
        .map_or(0, |i| i + 1);
    obj.insert(at, (key.into(), value));
}

/// The designs a lowered document names by symbol.
pub fn references(json: &Json) -> Vec<SymbolRef> {
    let mut json = json.clone();
    let mut out = Vec::new();
    walk(&mut json, "", "", &mut |obj, _, kind, name, at| {
        let from = obj
            .iter()
            .find(|(k, _)| k == "from")
            .and_then(|(_, v)| v.as_str())
            .map(str::to_string);
        out.push(SymbolRef {
            kind,
            name,
            from,
            at: at.into(),
        });
    });
    out
}

/// Design references written as file paths (`from "plate.part"`, `file = "arm.part"`): the file
/// form is refused (report 12 §3.5). Each is `(json path, message)` with the symbol form to
/// write. STEP and other non-design sources keep their paths.
fn file_form_errors(json: &Json, key: &str, at: &str, errors: &mut Vec<(String, String)>) {
    match json {
        Json::Object(obj) => {
            let interface = obj
                .iter()
                .any(|(k, v)| k == "frame" && v.as_str() == Some("interface"));
            if is_definition_key(key) || key == "model" || interface {
                if let Some(file) = obj
                    .iter()
                    .find(|(k, _)| k == "file")
                    .and_then(|(_, v)| v.as_str())
                {
                    if let Some(kind) = kind_of_file(file) {
                        let name = stem_name(file);
                        let folder = folder_of(file);
                        let from = if folder.is_empty() {
                            String::new()
                        } else {
                            format!(", from = \"./{folder}\"")
                        };
                        let written = if is_definition_key(key) {
                            format!(
                                "`use {kind} {name} as …`{}",
                                if folder.is_empty() {
                                    String::new()
                                } else {
                                    format!(" with `from \"./{folder}\"`")
                                }
                            )
                        } else if interface {
                            format!("`part = {name}{from}`")
                        } else {
                            format!("`model = {{ {kind} = {name}{from} }}`")
                        };
                        errors.push((
                            at.to_string(),
                            format!("name the design: write {written}, not the file \"{file}\""),
                        ));
                        return;
                    }
                }
            }
            for (k, v) in obj.iter() {
                let path = if at.is_empty() {
                    k.clone()
                } else {
                    format!("{at}.{k}")
                };
                file_form_errors(v, k, &path, errors);
            }
        }
        Json::Array(items) => {
            for (i, v) in items.iter().enumerate() {
                file_form_errors(v, key, &format!("{at}[{i}]"), errors);
            }
        }
        _ => {}
    }
}

/// Resolves every symbol in the lowered document `json` of the file `doc` to its file; a
/// design named by its file path is an error. Each failure is
/// `(json path, message)`; a failed symbol stays as written.
pub fn bind(json: &mut Json, doc: &str, scope: &ModuleScope) -> Vec<(String, String)> {
    let mut errors = Vec::new();
    file_form_errors(json, "", "", &mut errors);
    if !errors.is_empty() {
        return errors;
    }
    walk(json, "", "", &mut |obj, site, kind, name, at| {
        let from = obj
            .iter()
            .find(|(k, _)| k == "from")
            .and_then(|(_, v)| v.as_str())
            .map(str::to_string);
        let file = match scope.locate(doc, &kind, &name, from.as_deref()) {
            Ok(f) => f,
            Err(e) => {
                errors.push((at.to_string(), e));
                return;
            }
        };
        take(obj, "from");
        match site {
            Site::Definition => {
                take(obj, "name");
                put_first(obj, Some("kind"), "file", Json::String(file.clone()));
                if name != stem_name(&file) {
                    put_first(obj, Some("file"), "part", Json::String(name));
                }
            }
            Site::Interface => {
                take(obj, "part");
                put_first(obj, Some("frame"), "file", Json::String(file));
            }
            Site::Model => {
                take(obj, &kind);
                put_first(obj, None, "file", Json::String(file));
            }
        }
    });
    errors
}

/// How the file `doc` names the folder holding `file`: `None` for its own folder, a package
/// (`hardware`) for a `+pkg` folder in it or above it, else a relative path (`./cabin`).
pub fn module_text(doc: &str, file: &str) -> Option<String> {
    let here = folder_of(doc);
    let there = folder_of(file);
    if here == there {
        return None;
    }
    // A package: `+a/+b` directly under an ancestor of `here`.
    let segs: Vec<&str> = there.split('/').collect();
    let first_pkg = segs.iter().position(|s| s.starts_with('+'));
    if let Some(i) = first_pkg {
        if segs[i..].iter().all(|s| s.starts_with('+')) {
            let base = if i == 1 && there.starts_with('/') {
                "/".to_string()
            } else {
                segs[..i].join("/")
            };
            if ancestors(here).contains(&base) {
                return Some(
                    segs[i..]
                        .iter()
                        .map(|s| &s[1..])
                        .collect::<Vec<_>>()
                        .join("."),
                );
            }
        }
    }
    let a: Vec<&str> = here.split('/').filter(|s| !s.is_empty()).collect();
    let b: Vec<&str> = there.split('/').filter(|s| !s.is_empty()).collect();
    let common = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
    let mut parts: Vec<String> = Vec::new();
    if common == a.len() {
        parts.push(".".into());
    }
    parts.extend(std::iter::repeat_n("..".to_string(), a.len() - common));
    parts.extend(b[common..].iter().map(|s| s.to_string()));
    Some(parts.join("/"))
}

/// The symbol form of a value that names a design by file, for printing in the file `doc`;
/// `None` when the value names no design file (a STEP path keeps its path). The inverse of
/// [`bind`]:
/// - a `definition` (`{kind, file, part?}`) → `{kind, name, from?}`;
/// - an interface frame (`{frame = interface, file, …}`) → `{frame = interface, part, from?, …}`;
/// - a drawing `model` (`{file, …}`) → `{<kind> = name, from?, …}`.
pub fn unbound(key: &str, value: &Json, doc: &str) -> Option<Json> {
    let Json::Object(obj) = value else {
        return None;
    };
    let file = obj
        .iter()
        .find(|(k, _)| k == "file")
        .and_then(|(_, v)| v.as_str())?
        .to_string();
    let kind = kind_of_file(&file)?;
    let mut obj = obj.clone();
    let interface = obj
        .iter()
        .any(|(k, v)| k == "frame" && v.as_str() == Some("interface"));
    let (name_key, after, name) = if is_definition_key(key) {
        let name = take(&mut obj, "part")
            .and_then(|p| p.as_str().map(str::to_string))
            .unwrap_or_else(|| stem_name(&file));
        ("name", Some("kind"), name)
    } else if interface {
        ("part", Some("frame"), stem_name(&file))
    } else if key == "model" {
        (kind, None, stem_name(&file))
    } else {
        return None;
    };
    take(&mut obj, "file");
    put_first(&mut obj, after, name_key, Json::String(name));
    if let Some(f) = module_text(doc, &file) {
        put_first(&mut obj, Some(name_key), "from", Json::String(f));
    }
    Some(Json::Object(obj))
}

/// [`unbound`] applied at every depth of `value` (held under `key`).
pub fn unbound_tree(key: &str, value: &Json, doc: &str) -> Json {
    if let Some(symbolic) = unbound(key, value, doc) {
        return symbolic;
    }
    match value {
        Json::Object(obj) => Json::Object(
            obj.iter()
                .map(|(k, v)| (k.clone(), unbound_tree(k, v, doc)))
                .collect(),
        ),
        Json::Array(items) => {
            Json::Array(items.iter().map(|v| unbound_tree(key, v, doc)).collect())
        }
        other => other.clone(),
    }
}

// ----------------------------------------------------------------------------- dependencies

/// What a design file needs beyond itself before it can bind: each folder whose designs it can
/// name (written `<folder>/`, for the host to list) and each design file it names by path.
pub fn dependencies(path: &str, text: &str) -> Vec<String> {
    let mut out: BTreeSet<String> = BTreeSet::new();
    let folder = folder_of(path).to_string();
    let header = scan(path, text);
    let Some(dialect) = Dialect::for_path(path) else {
        return Vec::new();
    };
    let syntax = parse_source(dialect, text);
    let source = syntax.source();
    // Same-folder symbols, folder sources and packages: the host lists those folders.
    let mut froms: Vec<String> = header.occurrence_froms.clone();
    froms.extend(header.uses.iter().map(|u| u.from.clone()));
    let mut same_folder = false;
    for node in syntax.syntax().descendants() {
        use runmat_design_syntax::tree::NodeKind;
        match node.kind() {
            NodeKind::UseDeclaration => {
                if let Some(u) = runmat_design_syntax::ast::UseDeclaration::cast(node.clone()) {
                    if u.is_same_folder() {
                        same_folder = true;
                    }
                }
            }
            NodeKind::StringLiteral => {
                let span = node.trimmed_span();
                if let Some(inner) = source
                    .get(span.start..span.end)
                    .and_then(|t| t.strip_prefix('"'))
                    .and_then(|t| t.strip_suffix('"'))
                {
                    if is_file_path(inner) && !inner.ends_with(".circuit.layout") {
                        out.insert(join(&folder, inner));
                    } else if inner.starts_with('.') && !inner.contains(char::is_whitespace) {
                        froms.push(inner.to_string());
                    }
                }
            }
            _ => {}
        }
    }
    // Record forms (`part = airliner_skeleton`, `model = { part = arm }`) name the folder too.
    if !same_folder {
        same_folder =
            text.contains("part =") || text.contains("assembly =") || text.contains("circuit =");
    }
    if same_folder {
        out.insert(if folder.ends_with('/') {
            folder.clone()
        } else {
            format!("{folder}/")
        });
    }
    for from in froms {
        out.extend(from_listings(path, &from));
    }
    out.into_iter().collect()
}

/// The folder listings (`<folder>/`) a `from` names, seen from the file `doc`: the folder of a
/// path (`./cabin`, `../wing`), or each `+package` folder a package can be in, from `doc`'s
/// folder up to the workspace root. An alias names a folder the file's own `use` line lists; a
/// file path and a root path (`//lib`) name none.
pub fn from_listings(doc: &str, from: &str) -> Vec<String> {
    let folder = folder_of(doc);
    let listing = |f: String| if f.ends_with('/') { f } else { format!("{f}/") };
    if is_file_path(from) || from.starts_with("//") {
        return Vec::new();
    }
    if from.starts_with('.') || from.contains('/') {
        return vec![listing(join(folder, from))];
    }
    ancestors(folder)
        .iter()
        .map(|at| listing(join(at, &package_path(from))))
        .collect()
}

#[cfg(test)]
#[path = "modules_tests.rs"]
mod tests;
