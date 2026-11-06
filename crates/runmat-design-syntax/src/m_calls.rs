//! `.m` calls from design values, as the hosts exchange them (report 12 §2.3–2.6): the key a
//! call's result is kept under, a call a compile could not answer, the result a host returns,
//! and the text checks every design dialect makes on a function (its parameters, its effects).
//! Parts (runmat-cad) and circuits (runmat-circuit) share these, so a host runs either kind's
//! calls the same way.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The error a call fails with while its result is still to be evaluated.
pub const PENDING: &str = "waiting for the RunMat runtime to evaluate this value";

/// One call the table couldn't answer: what the host runs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PendingCall {
    /// The [`call_key`] the result is stored under.
    pub key: String,
    /// The function, defined in `sources[0]`.
    pub function: String,
    /// `(path, text)` of the function's file, then the `.m` files it can call without a `use`.
    pub sources: Vec<(String, String)>,
    /// SI doubles, in the order the `function` line declares them.
    pub arguments: Vec<f64>,
}

/// A result the host evaluated: the result's elements, or the error the function raised.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TableResult {
    Values(Vec<f64>),
    Error { error: String },
}

/// The key of one call: the sources' paths and texts, the function and the arguments bit for
/// bit. Editing any reached `.m` file, or changing an argument, gives a new key.
pub fn call_key(sources: &[(String, String)], function: &str, arguments: &[f64]) -> String {
    let mut h = Sha256::new();
    let mut part = |b: &[u8]| {
        h.update((b.len() as u64).to_be_bytes());
        h.update(b);
    };
    part(b"runmat-cad/m-call/v1");
    part(function.as_bytes());
    for (path, text) in sources {
        part(path.as_bytes());
        part(text.as_bytes());
    }
    for a in arguments {
        part(&a.to_bits().to_be_bytes());
    }
    hex::encode(h.finalize())
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

/// `line` without its `%` comment and string contents (a transpose is kept).
pub fn strip_comment_and_strings(line: &str) -> String {
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
