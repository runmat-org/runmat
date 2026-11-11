//! Circuit param values computed by `.m` functions (report 12 §2.3–2.6, PCB-38):
//! `component R2 = { part = "R0402", value = feedbackResistor(vout = 3.3 V, r1 = 10 kohm) as ohm }`.
//!
//! The rules are the part dialect's:
//! - **Resolution:** a bare name is a `.m` file in the circuit's folder (or its `private/`);
//!   `pkg.fn` is `+pkg/fn.m` in the circuit's folder or a folder above it.
//! - **Arguments** cross as SI doubles in the order the `function` line declares them; named
//!   arguments take their declared slot, positional ones fill the rest in order.
//! - **The result** is a number in the unit after `as` (`as ohm`); without `as` it is a plain
//!   number.
//! - **Determinism:** a function whose text (or a folder function it reaches) calls an
//!   effectful builtin (the clock, files, unseeded randomness) is an error.
//! - **Two phases:** the circuit compiler can't run code, so the host runs each call it lists
//!   ([`PendingCall`], keyed by [`call_key`]) in the RunMat runtime and compiles again with the
//!   results in [`super::CompileOptions::values`].

use super::{modules, Files};
use crate::ir::{Design, Param, ParamValue, ValueCall};
use runmat_design_syntax::m_calls::{call_key, effect_of, parameters_of, PendingCall, TableResult};
use runmat_rules::{parse_quantity, BuiltinUnits, Quantity, SourceKey};
use std::collections::BTreeMap;

/// Results the host evaluated, by [`call_key`].
pub type Values = BTreeMap<String, TableResult>;

/// The info a compile reports while computed values wait for the runtime; the compile output
/// then lists them as `pending_values`.
pub const PENDING_VALUES: &str =
    "waiting for the RunMat runtime to evaluate the values computed by .m functions";

/// What filling a design's computed values found.
#[derive(Debug, Default)]
pub struct Outcome {
    /// Calls still to run in the runtime.
    pub pending: Vec<PendingCall>,
    /// A call that can't be made or failed: where it is written and why.
    pub errors: Vec<(SourceKey, String)>,
}

fn folder_of(path: &str) -> &str {
    match path.rsplit_once('/') {
        Some(("", _)) if path.starts_with('/') => "/",
        Some((d, _)) => d,
        None => "",
    }
}

fn join(folder: &str, name: &str) -> String {
    match folder {
        "" => name.to_owned(),
        "/" => format!("/{name}"),
        f => format!("{f}/{name}"),
    }
}

/// The `.m` file `function` names, seen from the circuit `doc`.
fn resolve(doc: &str, function: &str, files: &Files) -> Result<String, String> {
    let (package, name) = match function.rsplit_once('.') {
        Some((p, n)) => (Some(p), n),
        None => (None, function),
    };
    let candidates: Vec<String> = match package {
        None => {
            let here = folder_of(doc);
            vec![
                join(here, &format!("{name}.m")),
                join(&join(here, "private"), &format!("{name}.m")),
            ]
        }
        Some(p) => modules::folders(doc, p)
            .into_iter()
            .map(|f| join(&f, &format!("{name}.m")))
            .collect(),
    };
    candidates
        .into_iter()
        .find(|c| files.contains_key(c))
        .ok_or_else(|| match package {
            None => format!("no `{name}.m` in this folder (or its private/)"),
            Some(p) => format!(
                "no `+{}/{name}.m` in this folder or one above it",
                p.replace('.', "/+")
            ),
        })
}

/// The function's file first, then the `.m` files of its folder and `private/`, which it may
/// call without a `use`.
fn sources(file: &str, files: &Files) -> Vec<(String, String)> {
    let folder = folder_of(file);
    let base = folder.strip_suffix("/private").unwrap_or(folder);
    let private = join(base, "private");
    let mut out = vec![(file.to_owned(), files[file].clone())];
    for (path, text) in files {
        let dir = folder_of(path);
        if path != file && path.ends_with(".m") && (dir == base || dir == private) {
            out.push((path.clone(), text.clone()));
        }
    }
    out
}

/// SI arguments in the order `parameters` declares them.
fn ordered(
    name: &str,
    parameters: &[String],
    arguments: &[(String, Quantity)],
) -> Result<Vec<f64>, String> {
    let mut slots: Vec<Option<f64>> = vec![None; parameters.len()];
    for (arg, q) in arguments.iter().filter(|(n, _)| !n.is_empty()) {
        let at = parameters
            .iter()
            .position(|p| p == arg)
            .ok_or_else(|| format!("`{name}` takes ({}), not `{arg}`", parameters.join(", ")))?;
        slots[at] = Some(q.value);
    }
    let mut free = slots.iter_mut().filter(|s| s.is_none());
    for (_, q) in arguments.iter().filter(|(n, _)| n.is_empty()) {
        *free
            .next()
            .ok_or_else(|| format!("`{name}` takes {} arguments", parameters.len()))? =
            Some(q.value);
    }
    let last = slots.iter().rposition(Option::is_some).map_or(0, |i| i + 1);
    slots[..last]
        .iter()
        .zip(parameters)
        .map(|(s, p)| s.ok_or_else(|| format!("`{name}` needs `{p}`")))
        .collect()
}

/// One call's SI result from `values`, or the call to run.
fn evaluate(
    doc: &str,
    call: &ValueCall,
    files: &Files,
    values: &Values,
) -> Result<Result<f64, PendingCall>, String> {
    let file = resolve(doc, &call.function, files)?;
    let name = file
        .rsplit('/')
        .next()
        .unwrap_or(&file)
        .trim_end_matches(".m")
        .to_owned();
    let parameters = parameters_of(&files[&file], &name)
        .ok_or_else(|| format!("`{file}` has no `function … = {name}(…)` line"))?;
    let sources = sources(&file, files);
    if let Some((path, effect)) = sources
        .iter()
        .find_map(|(p, t)| effect_of(t).map(|e| (p, e)))
    {
        return Err(format!("`{name}` uses {effect} in `{path}`; a circuit value must give the same result for the same arguments"));
    }
    let arguments = ordered(&name, &parameters, &call.arguments)?;
    let key = call_key(&sources, &name, &arguments);
    let scale = match &call.unit {
        Some(u) => parse_quantity(&format!("1 {u}"), &BuiltinUnits)?.value,
        None => 1.0,
    };
    match values.get(&key) {
        Some(TableResult::Values(v)) => match v.as_slice() {
            [x] => Ok(Ok(x * scale)),
            many => Err(format!(
                "`{name}` returned {} numbers; a circuit value takes one",
                many.len()
            )),
        },
        Some(TableResult::Error { error }) => Err(format!("`{name}` raised an error: {error}")),
        None => Ok(Err(PendingCall {
            key,
            function: name,
            sources,
            arguments,
        })),
    }
}

fn fill(doc: &str, param: &mut Param, files: &Files, values: &Values, out: &mut Outcome) {
    let Some(call) = &param.call else { return };
    match evaluate(doc, call, files, values) {
        Ok(Ok(si)) => {
            if let ParamValue::Quantity { nominal, .. } = &mut param.value {
                nominal.value = si;
            }
        }
        Ok(Err(pending)) => {
            if !out.pending.iter().any(|p| p.key == pending.key) {
                out.pending.push(pending);
            }
        }
        Err(message) => out.errors.push((param.source.clone(), message)),
    }
}

/// Fills every computed param of `design` (the circuit `doc`'s, before `use` lines bring other
/// circuits in) from `values`, listing the calls still to run and the ones that fail.
pub fn fill_values(design: &mut Design, doc: &str, files: &Files, values: &Values) -> Outcome {
    let mut out = Outcome::default();
    let each = |p: &mut BTreeMap<String, Param>, out: &mut Outcome| {
        for param in p.values_mut() {
            fill(doc, param, files, values, out);
        }
    };
    each(&mut design.params, &mut out);
    for class in design.net_classes.values_mut() {
        each(&mut class.params, &mut out);
    }
    for sheet in design.sheets.values_mut() {
        each(&mut sheet.params, &mut out);
        for c in &mut sheet.components {
            each(&mut c.params, &mut out);
        }
        for n in &mut sheet.nets {
            each(&mut n.params, &mut out);
        }
    }
    out
}

/// Whether `design` has a value computed by a `.m` function.
pub fn has_calls(design: &Design) -> bool {
    let any = |p: &BTreeMap<String, Param>| p.values().any(|p| p.call.is_some());
    any(&design.params)
        || design.net_classes.values().any(|c| any(&c.params))
        || design.sheets.values().any(|s| {
            any(&s.params)
                || s.components.iter().any(|c| any(&c.params))
                || s.nets.iter().any(|n| any(&n.params))
        })
}

/// The folders a circuit's computed values can reach (`<folder>/`, for the host to list): its
/// own, its `private/` and each `+pkg` its calls name, from its folder up.
pub fn value_folders(doc: &str, design: &Design) -> Vec<String> {
    let listing = |f: &str| {
        if f.ends_with('/') {
            f.to_owned()
        } else {
            format!("{f}/")
        }
    };
    let mut out = vec![
        listing(folder_of(doc)),
        listing(&join(folder_of(doc), "private")),
    ];
    let mut packages = std::collections::BTreeSet::new();
    let mut note = |p: &BTreeMap<String, Param>| {
        for call in p.values().filter_map(|p| p.call.as_ref()) {
            if let Some((pkg, _)) = call.function.rsplit_once('.') {
                packages.insert(pkg.to_owned());
            }
        }
    };
    note(&design.params);
    for s in design.sheets.values() {
        note(&s.params);
        s.components.iter().for_each(|c| note(&c.params));
    }
    for pkg in packages {
        out.extend(modules::folder_dependencies(doc, &pkg));
    }
    out
}

/// The `.m` file `function` resolves to from the circuit `doc`, if the workspace has it.
pub fn function_file(doc: &str, function: &str, files: &Files) -> Option<String> {
    resolve(doc, function, files).ok()
}

/// Fills each computed param's `file` in `index`: the `.m` file its function resolves to from
/// the circuit that makes the call.
pub fn note_files(index: &mut crate::index::CircuitIndex, files: &Files) {
    for component in &mut index.components {
        for row in component.computed.values_mut() {
            row.file = resolve(&row.source.path, &row.function, files).ok();
        }
    }
}

/// The calls the circuit `path` belongs to (a layout's circuit, or the circuit itself) still
/// waits for, given the results in `values`.
pub fn pending_values(path: &str, text: &str, files: &Files, values: &Values) -> Vec<PendingCall> {
    let (circuit, circuit_text) = if super::is_layout(path) {
        let uses = super::head_uses(runmat_design_syntax::Dialect::CircuitLayout, text);
        let Some(Ok(c)) = super::layout_circuit(path, &uses, files) else {
            return Vec::new();
        };
        let Some(t) = files.get(&c) else {
            return Vec::new();
        };
        (c, t.clone())
    } else {
        (path.to_owned(), text.to_owned())
    };
    match crate::syntax::parse_circuit(&circuit, &circuit_text).design {
        Some(mut design) => fill_values(&mut design, &circuit, files, values).pending,
        None => Vec::new(),
    }
}

/// `pending_values` as JSON, for a compile output whose `diagnostics` wait on them; `None` when
/// none waits.
pub fn pending_json(
    diagnostics: &[super::Diagnostic],
    path: &str,
    text: &str,
    files: &Files,
    values: &Values,
) -> Option<String> {
    diagnostics
        .iter()
        .any(|d| d.message == PENDING_VALUES)
        .then(|| {
            serde_json::to_string(&pending_values(path, text, files, values))
                .unwrap_or_else(|_| "[]".into())
        })
}
