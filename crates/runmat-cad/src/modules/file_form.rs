//! A `from` that names one design file: `model = { part = l_bracket, from = "../l_bracket.part" }`
//! or `use part bracket from "../parts/bracket.part" as b`.
//!
//! The file form and the folder form resolve through the same scope: the file must be a design
//! file in the workspace, and the named design must be one the file declares. When the reference
//! leaves the name out (`model = { from = "../l_bracket.part" }`), the file's own design is used.
//! A path that names neither a design file nor a folder of designs suggests the nearest real one.

use super::{folder_of, join, kind_of_file, normalize, ModuleScope};
use crate::scope::edit_distance;

/// `"a"` or `"an"` before `word`.
fn article(word: &str) -> &'static str {
    if word.starts_with(['a', 'e', 'i', 'o', 'u']) {
        "an"
    } else {
        "a"
    }
}

/// `target` as written from the folder `here`: `./cabin`, `../wing/spar.part`, or `.` for `here`.
pub fn relative(here: &str, target: &str) -> String {
    let a: Vec<&str> = here.split('/').filter(|s| !s.is_empty()).collect();
    let b: Vec<&str> = target.split('/').filter(|s| !s.is_empty()).collect();
    let common = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
    let mut parts: Vec<String> = Vec::new();
    if common == a.len() {
        parts.push(".".into());
    }
    parts.extend(std::iter::repeat_n("..".to_string(), a.len() - common));
    parts.extend(b[common..].iter().map(|s| s.to_string()));
    parts.join("/")
}

/// `target` with its design-file extension removed (`/p/arm.part` → `/p/arm`).
fn without_extension(target: &str) -> &str {
    match kind_of_file(target) {
        Some(kind) => target.strip_suffix(&format!(".{kind}")).unwrap_or(target),
        None => target,
    }
}

impl ModuleScope {
    /// The workspace path a `from` path names, seen from `folder`: `//` is the project root, a
    /// leading `/` the workspace root, anything else is relative to `folder`.
    pub(super) fn path_of(&self, folder: &str, from: &str) -> Result<String, String> {
        if let Some(rest) = from.strip_prefix("//") {
            let root = self.root.as_deref().ok_or_else(|| {
                format!("`{from}` is root-relative, and no runmat.toml marks the project root")
            })?;
            return Ok(join(root, rest));
        }
        Ok(if from.starts_with('/') {
            normalize(from)
        } else {
            join(folder, from)
        })
    }

    /// The file a file-form `from` names, holding `kind name` (`name` empty: the file's own
    /// design of that kind), seen from `folder`.
    pub(super) fn locate_in_file(
        &self,
        folder: &str,
        kind: &str,
        name: &str,
        from: &str,
    ) -> Result<String, String> {
        let file = self.path_of(folder, from)?;
        if kind_of_file(&file).is_none() {
            return Err(format!(
                "`{from}` is a board layout, which declares no designs; name its `.circuit` file"
            ));
        }
        let Some(declared) = self.declared.get(&file) else {
            return Err(self.missing(folder, from, &file));
        };
        let holds = declared
            .iter()
            .map(|(k, n)| format!("{k} {n}"))
            .collect::<Vec<_>>()
            .join(", ");
        let of_kind: Vec<&str> = declared
            .iter()
            .filter(|(k, _)| k == kind)
            .map(|(_, n)| n.as_str())
            .collect();
        if of_kind.is_empty() {
            let what = if holds.is_empty() {
                "no designs".to_string()
            } else {
                holds
            };
            return Err(format!(
                "`{from}` holds {what}, not {} {kind}",
                article(kind)
            ));
        }
        if name.is_empty() || of_kind.contains(&name) {
            return Ok(file);
        }
        Err(format!("no `{kind} {name}` in `{from}`; it holds {holds}"))
    }

    /// The error for a `from` that names nothing in the workspace, with the nearest design
    /// file or folder of designs as written from `folder`.
    pub(super) fn missing(&self, folder: &str, from: &str, target: &str) -> String {
        let mut m = if super::is_file_path(from) {
            format!("`{from}` is neither a design file nor a folder in the workspace")
        } else {
            format!("the folder `{from}` holds no designs (or isn't in the workspace)")
        };
        if let Some(near) = self.nearest(target) {
            m.push_str(&format!("; did you mean `{}`?", relative(folder, &near)));
        }
        m
    }

    /// The design file or folder of designs whose path is nearest `target`, when one is close.
    fn nearest(&self, target: &str) -> Option<String> {
        let name = target.rsplit('/').next().unwrap_or(target);
        let limit = (name.chars().count() / 3).max(2);
        let files = self.declared.keys().map(|f| {
            (
                f,
                edit_distance(target, f).min(edit_distance(target, without_extension(f))),
            )
        });
        let folders = self
            .scope
            .folders
            .keys()
            .filter(|f| !f.is_empty() && f.as_str() != "/" && f.as_str() != folder_of(target))
            .map(|f| (f, edit_distance(target, f)));
        // A file of the same name in another folder is the likeliest meaning.
        let same_name = self
            .declared
            .keys()
            .filter(|f| f.rsplit('/').next() == Some(name))
            .min_by_key(|f| edit_distance(target, f));
        if let Some(f) = same_name {
            return Some(f.clone());
        }
        files
            .chain(folders)
            .filter(|(_, d)| *d <= limit)
            .min_by_key(|(_, d)| *d)
            .map(|(f, _)| f.clone())
    }
}

#[cfg(test)]
#[path = "file_form_tests.rs"]
mod tests;
