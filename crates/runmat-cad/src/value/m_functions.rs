//! `.m` functions as design values (report 12 §2.3–2.6).
//!
//! A design value `base = camBaseRadius(lift = input.lift, rise = 120 deg) as mm` calls the
//! `.m` function `camBaseRadius`:
//! - **Resolution** goes through the folder scope ([`ModuleScope`]): a bare name is a
//!   same-folder `.m` (its `private/` included), then a project-path function; `pkg.fn` is
//!   `+pkg/fn.m` in the referring folder or a folder above it; `alias.fn` is a function of a
//!   folder brought in with `use "<folder>" as alias`.
//! - **Arguments** cross as canonical SI doubles in the order the `function` line declares
//!   them (today's runtime ABI, until the runtime's quantity value lands). Named arguments map to
//!   the declared names; positional ones fill the remaining slots in order.
//! - **The result** is a plain number, so a dimensioned target writes its unit at the call
//!   (`as mm`); a bare result meets only a dimensionless target.
//! - **Determinism:** a function whose text calls an effectful builtin (the clock, files,
//!   network, unseeded randomness) must be marked `volatile` at the use site. A deterministic
//!   call is also evaluated twice on first use, and two results that differ bit for bit fail
//!   as `design.value.nondeterministic`.
//! - **Cache:** results are kept by the cache key ([`dynamic_cache_key`]): the function file's
//!   digest, the digests of the folder functions it calls (directly or through each other), the
//!   SI arguments bit for bit and the runtime version.
//!
//! The host runs the code through an [`MRunner`]: the native tools run it in a RunMat session,
//! and the Desktop in its runtime.

use super::{
    dynamic_cache_key, DynamicCall, DynamicContract, DynamicOutput, DynamicValueEvaluator,
    Quantity, Randomness,
};
use crate::modules::{folder_of, ModuleScope};
use crate::scope::Binding;
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

/// Runs one `.m` function on SI double arguments.
pub trait MRunner {
    /// Calls `function`, defined in `sources[0]`. The rest of `sources` are the `.m` files it
    /// can call without a `use` (its folder and `private/`). Returns the result's elements
    /// (one for a scalar).
    fn call(
        &self,
        sources: &[(String, String)],
        function: &str,
        arguments: &[f64],
    ) -> Result<Vec<f64>, String>;
    /// The runtime version, part of every cache key.
    fn runtime_version(&self) -> String;
}

/// A resolved `.m` function: its file and the files it reaches.
#[derive(Clone, Debug, PartialEq)]
pub struct MFunction {
    /// The workspace path of the function's file.
    pub file: String,
    /// The function name inside the file (the file's stem).
    pub name: String,
    /// The parameter names of its `function` line, in order.
    pub parameters: Vec<String>,
}

/// Evaluates design values that call `.m` functions, for one design file.
pub struct MFunctions<'a> {
    pub scope: &'a ModuleScope,
    /// Workspace path → text, `.m` files included.
    pub files: &'a BTreeMap<String, String>,
    /// The design file whose values are evaluated.
    pub document: String,
    pub runner: &'a dyn MRunner,
    cache: RefCell<BTreeMap<String, Result<f64, String>>>,
}

impl<'a> MFunctions<'a> {
    pub fn new(
        scope: &'a ModuleScope,
        files: &'a BTreeMap<String, String>,
        document: impl Into<String>,
        runner: &'a dyn MRunner,
    ) -> Self {
        MFunctions {
            scope,
            files,
            document: document.into(),
            runner,
            cache: RefCell::new(BTreeMap::new()),
        }
    }

    /// The `.m` file `function` names, seen from the design file's folder.
    pub fn resolve(&self, function: &str) -> Result<MFunction, String> {
        let folder = folder_of(&self.document).to_string();
        let segments: Vec<&str> = function.split('.').collect();
        let resolution = if segments.len() == 1 {
            self.scope
                .scope
                .resolve_bare(&folder, &BTreeSet::new(), function)
        } else {
            self.scope.scope.resolve_qualified(&folder, &segments)
        }
        .map_err(|p| crate::modules::problem_text(&p))?;
        let file = match resolution.binding {
            Binding::FolderFunction {
                folder,
                name,
                private,
            } => join(&folder, &format!("{}{name}.m", if private { "private/" } else { "" })),
            Binding::PathFunction { folder, name } => join(&folder, &format!("{name}.m")),
            Binding::PackageFunction { package, name } => self
                .package_file(&folder, &package, &name)
                .ok_or_else(|| format!("no `+{}` folder above `{folder}` holds `{name}.m`", package.replace('.', "/+")))?,
            other => return Err(format!("`{function}` is {}, not an `.m` function", binding_text(&other))),
        };
        let text = self
            .files
            .get(&file)
            .ok_or_else(|| format!("`{file}` is not in the workspace"))?;
        let name = stem(&file).to_string();
        let parameters = parameters_of(text, &name)
            .ok_or_else(|| format!("`{file}` has no `function … = {name}(…)` line"))?;
        Ok(MFunction {
            file,
            name,
            parameters,
        })
    }

    fn package_file(&self, folder: &str, package: &str, name: &str) -> Option<String> {
        let rel = package
            .split('.')
            .map(|p| format!("+{p}"))
            .collect::<Vec<_>>()
            .join("/");
        let mut at = Some(folder.to_string());
        while let Some(dir) = at {
            let file = join(&dir, &format!("{rel}/{name}.m"));
            if self.files.contains_key(&file) {
                return Some(file);
            }
            at = (!dir.is_empty() && dir != "/").then(|| folder_of(&dir).to_string());
        }
        None
    }

    /// The function's file first, then the `.m` files of its folder and `private/`.
    fn sources(&self, f: &MFunction) -> Vec<(String, String)> {
        let folder = folder_of(&f.file);
        let base = folder.strip_suffix("/private").unwrap_or(folder);
        let mut out = vec![(f.file.clone(), self.files[&f.file].clone())];
        for (path, text) in self.files {
            let dir = folder_of(path);
            if path != &f.file && path.ends_with(".m") && (dir == base || dir == join(base, "private")) {
                out.push((path.clone(), text.clone()));
            }
        }
        out
    }

    /// SI arguments in declared order.
    fn ordered(&self, f: &MFunction, arguments: &[(String, Quantity)]) -> Result<Vec<f64>, String> {
        let mut slots: Vec<Option<f64>> = vec![None; f.parameters.len()];
        for (name, q) in arguments.iter().filter(|(n, _)| !n.is_empty()) {
            let at = f.parameters.iter().position(|p| p == name).ok_or_else(|| {
                format!("`{}` takes ({}), not `{name}`", f.name, f.parameters.join(", "))
            })?;
            slots[at] = Some(q.si);
        }
        let mut free = slots.iter_mut().filter(|s| s.is_none());
        for (_, q) in arguments.iter().filter(|(n, _)| n.is_empty()) {
            *free
                .next()
                .ok_or_else(|| format!("`{}` takes {} arguments", f.name, f.parameters.len()))? = Some(q.si);
        }
        // Trailing parameters may stay unset (MATLAB's nargin); a gap before a set one may not.
        let last = slots.iter().rposition(Option::is_some).map_or(0, |i| i + 1);
        slots[..last]
            .iter()
            .zip(&f.parameters)
            .map(|(s, p)| s.ok_or_else(|| format!("`{}` needs `{p}`", f.name)))
            .collect()
    }
}

impl DynamicValueEvaluator for MFunctions<'_> {
    fn evaluate(&self, call: &DynamicCall, args: &[(String, Quantity)]) -> Result<Quantity, String> {
        match self.evaluate_output(call, args)? {
            DynamicOutput::Quantity { value } => Ok(value),
            DynamicOutput::Double { value } => Ok(Quantity::dimensionless(value)),
        }
    }

    fn digest(&self, call: &DynamicCall) -> String {
        self.contract(call).function_digest
    }

    fn contract(&self, call: &DynamicCall) -> DynamicContract {
        let resolved = self.resolve(&call.function);
        let (function_digest, callees, effect) = match &resolved {
            Ok(f) => {
                let sources = reachable(self.sources(f));
                let callees = sources[1..].iter().map(|(p, t)| digest(p, t)).collect();
                let effect = sources.iter().find_map(|(p, t)| effect_of(t).map(|e| format!("{e} in `{p}`")));
                (digest(&f.file, &sources[0].1), callees, effect)
            }
            // Resolution fails again in evaluate, with its message.
            Err(_) => (format!("unresolved:{}", call.function), Vec::new(), None),
        };
        DynamicContract {
            function_digest,
            callees,
            reads: Vec::new(),
            runtime_version: self.runner.runtime_version(),
            effect,
            randomness: Randomness::None,
        }
    }

    fn evaluate_output(&self, call: &DynamicCall, args: &[(String, Quantity)]) -> Result<DynamicOutput, String> {
        let f = self.resolve(&call.function)?;
        let ordered = self.ordered(&f, args)?;
        let key = dynamic_cache_key(call, args, &self.contract(call));
        if let Some(hit) = self.cache.borrow().get(&key) {
            return hit.clone().map(|value| DynamicOutput::Double { value });
        }
        let sources = self.sources(&f);
        let run = || -> Result<f64, String> {
            match self.runner.call(&sources, &f.name, &ordered)?.as_slice() {
                [v] => Ok(*v),
                many => Err(format!(
                    "`{}` returned {} numbers; a design value takes one",
                    f.name,
                    many.len()
                )),
            }
        };
        let result = run().and_then(|first| {
            if call.volatile {
                return Ok(first);
            }
            let second = run()?;
            if first.to_bits() == second.to_bits() {
                Ok(first)
            } else {
                Err(format!(
                    "design.value.nondeterministic: `{}` returned {first} and then {second} for the same arguments",
                    f.name
                ))
            }
        });
        if !call.volatile {
            self.cache.borrow_mut().insert(key, result.clone());
        }
        result.map(|value| DynamicOutput::Double { value })
    }
}

/// The function's file and the folder functions it calls, directly or through each other: the
/// callees whose digests and effects count.
fn reachable(sources: Vec<(String, String)>) -> Vec<(String, String)> {
    let mut taken = vec![false; sources.len()];
    taken[0] = true;
    let mut queue = vec![0];
    while let Some(i) = queue.pop() {
        let words = words_of(&sources[i].1);
        for (j, (path, _)) in sources.iter().enumerate() {
            if !taken[j] && words.contains(stem(path)) {
                taken[j] = true;
                queue.push(j);
            }
        }
    }
    sources.into_iter().zip(taken).filter_map(|(s, t)| t.then_some(s)).collect()
}

fn words_of(text: &str) -> BTreeSet<String> {
    text.lines()
        .flat_map(|l| {
            strip_comment_and_strings(l)
                .split(|c: char| !(c.is_alphanumeric() || c == '_'))
                .filter(|w| !w.is_empty())
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .collect()
}

/// The parameter names of `name`'s `function` line.
pub fn parameters_of(text: &str, name: &str) -> Option<Vec<String>> {
    text.lines().find_map(|line| {
        let rest = line.trim().strip_prefix("function")?;
        let rest = rest.trim_start();
        let after = rest.split_once('=').map_or(rest, |(_, r)| r).trim_start();
        let after = after.strip_prefix(name)?;
        let after = after.trim_start();
        if after.is_empty() || after.starts_with('%') {
            return Some(Vec::new());
        }
        let inner = after.strip_prefix('(')?.split_once(')')?.0;
        Some(
            inner
                .split(',')
                .map(|p| p.trim().to_string())
                .filter(|p| !p.is_empty() && p != "~")
                .collect(),
        )
    })
}

/// Builtins whose use makes a value function effectful (report 12 §2.6).
const EFFECTFUL: &[(&str, &str)] = &[
    ("rand", "unseeded randomness"),
    ("randn", "unseeded randomness"),
    ("randi", "unseeded randomness"),
    ("randperm", "unseeded randomness"),
    ("tic", "the clock"),
    ("toc", "the clock"),
    ("clock", "the clock"),
    ("now", "the clock"),
    ("datetime", "the clock"),
    ("fopen", "file access"),
    ("fwrite", "file access"),
    ("fprintf", "file access"),
    ("save", "file access"),
    ("load", "file access"),
    ("readtable", "file access"),
    ("readmatrix", "file access"),
    ("webread", "the network"),
    ("websave", "the network"),
    ("input", "user input"),
    ("system", "a shell command"),
    ("getenv", "the environment"),
];

/// The first effect a function's text uses, ignoring comments and strings.
pub fn effect_of(text: &str) -> Option<String> {
    for line in text.lines() {
        let code = strip_comment_and_strings(line);
        let mut word = String::new();
        let flush = |w: &mut String, next: Option<char>| -> Option<String> {
            let found = (!w.is_empty() && next != Some('.'))
                .then(|| EFFECTFUL.iter().find(|(b, _)| *b == w.as_str()))
                .flatten()
                .map(|(b, why)| format!("`{b}` ({why})"));
            w.clear();
            found
        };
        let mut prev = ' ';
        let chars: Vec<char> = code.chars().collect();
        for (i, &c) in chars.iter().enumerate() {
            if c.is_alphanumeric() || c == '_' {
                if word.is_empty() && prev == '.' {
                    word.push('.'); // a field or method, never a builtin
                }
                word.push(c);
            } else if let Some(e) = flush(&mut word, chars.get(i).copied()) {
                return Some(e);
            }
            prev = c;
        }
        if let Some(e) = flush(&mut word, None) {
            return Some(e);
        }
    }
    None
}

fn strip_comment_and_strings(line: &str) -> String {
    let mut out = String::new();
    let mut quote: Option<char> = None;
    let mut prev = ' ';
    for c in line.chars() {
        match quote {
            Some(q) if c == q => quote = None,
            Some(_) => {}
            None if c == '%' => break,
            // A transpose follows a name, a bracket or another transpose; a quote opens a string.
            None if c == '"' || (c == '\'' && !(prev.is_alphanumeric() || ")]}'_.".contains(prev))) => {
                quote = Some(c)
            }
            None => out.push(c),
        }
        prev = c;
    }
    out
}

fn digest(path: &str, text: &str) -> String {
    let mut h = Sha256::new();
    h.update(path.as_bytes());
    h.update([0]);
    h.update(text.as_bytes());
    format!("sha256:{}", hex::encode(h.finalize()))
}

fn stem(path: &str) -> &str {
    let file = path.rsplit('/').next().unwrap_or(path);
    file.strip_suffix(".m").unwrap_or(file)
}

fn join(folder: &str, rel: &str) -> String {
    if folder.is_empty() {
        rel.to_string()
    } else {
        format!("{}/{rel}", folder.trim_end_matches('/'))
    }
}

fn binding_text(b: &Binding) -> String {
    match b {
        Binding::Local { name } => format!("the local `{name}`"),
        Binding::Vocabulary { word } => format!("the vocabulary word `{word}`"),
        Binding::Builtin { name } => format!("the runtime builtin `{name}`"),
        Binding::Declaration { kind, name, .. } => format!("the {kind} `{name}`"),
        _ => "a function".into(),
    }
}
