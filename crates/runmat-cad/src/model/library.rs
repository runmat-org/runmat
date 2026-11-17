//! A project's part library: the `library/` folder beside a part.
//!
//! - `library/*.part`: library features and parts. Each `feature_def` in a file is a library
//!   feature (`library(def = "library/fixtures.part#feature_def.slot", plane = …)`); a file with
//!   no `feature_def` is a library part, inserted whole as a body
//!   (`library(def = "library/pin.part")`): its `input`s are the instance's arguments and its
//!   features add the body.
//! - `library/profiles/*.part`: custom structural profiles, each a part with one sketch named
//!   `profile` whose closed region is the section and whose origin sits on the member's path.
//!
//! [`resolve`] reads the files a part's library instances name into its `feature_defs`, under
//! the full `def`, before regeneration; [`files`] lists them as the part's dependencies, so
//! editing a library file rebuilds the parts that use it. Nothing is fetched from a network.

use crate::model::detail::FeatureDef;
use crate::model::{FeatureKind, PartDocument};

/// The library folder, relative to the part's folder.
pub const FOLDER: &str = "library";
/// The custom-profile folder, relative to the part's folder.
pub const PROFILES: &str = "library/profiles";
/// The sketch a custom profile file holds.
pub const PROFILE_SKETCH: &str = "profile";

/// The file a library `def` names (`library/fixtures.part` of
/// `library/fixtures.part#feature_def.slot`), or `None` for a definition in the part itself.
pub fn def_file(def: &str) -> Option<&str> {
    let file = def.split('#').next().unwrap_or(def);
    file.ends_with(".part").then_some(file)
}

/// The library files a part's instances read.
pub fn files(document: &PartDocument) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for f in &document.features {
        if let FeatureKind::Library(inst) = &f.kind {
            if let Some(file) = def_file(&inst.def) {
                if !out.iter().any(|o| o == file) {
                    out.push(file.to_owned());
                }
            }
        }
    }
    out
}

/// The definition `def` names in the library file's text: a `feature_def` by name after `#`, or
/// the whole part.
pub fn definition(def: &str, file_text: &str) -> Result<FeatureDef, String> {
    let file = def_file(def).ok_or_else(|| format!("`{def}` doesn't name a library file"))?;
    let parsed = crate::syntax::parse_part(file, file_text);
    let document = parsed.document.ok_or_else(|| {
        let first = parsed
            .diagnostics
            .first()
            .map(|d| d.message.clone())
            .unwrap_or_default();
        format!("{file} doesn't parse: {first}")
    })?;
    match def.split_once('#') {
        Some((_, name)) => {
            let name = name.strip_prefix("feature_def.").unwrap_or(name);
            let mut found = document
                .feature_defs
                .iter()
                .find(|d| d.name == name)
                .cloned()
                .ok_or_else(|| format!("{file} has no feature_def `{name}`"))?;
            found.name = def.to_owned();
            Ok(found)
        }
        None => {
            if !document.lets.is_empty() {
                return Err(format!(
                    "{file} computes `let` values; a library part takes its sizes from inputs"
                ));
            }
            Ok(FeatureDef {
                name: def.to_owned(),
                params: document.inputs.clone(),
                slots: Vec::new(),
                features: document.features.clone(),
            })
        }
    }
}

/// The part with every library file its instances name read into `feature_defs`, and the
/// problems: a file missing or unreadable, or a definition it lacks.
pub fn resolve(
    document: &PartDocument,
    read: &dyn Fn(&str) -> Option<String>,
) -> (PartDocument, Vec<String>) {
    let mut out = document.clone();
    let mut problems = Vec::new();
    for f in &document.features {
        let FeatureKind::Library(inst) = &f.kind else {
            continue;
        };
        let Some(file) = def_file(&inst.def) else {
            continue;
        };
        if out.feature_defs.iter().any(|d| d.name == inst.def) {
            continue;
        }
        let Some(text) = read(file) else {
            problems.push(format!("feature `{}`: {file} isn't in the project", f.id));
            continue;
        };
        match definition(&inst.def, &text) {
            Ok(def) => out.feature_defs.push(def),
            Err(e) => problems.push(format!("feature `{}`: {e}", f.id)),
        }
    }
    (out, problems)
}

/// What a library file offers, for the panel's search.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct LibraryEntry {
    /// `feature`, `part` or `profile`.
    pub kind: &'static str,
    /// What the command's field takes: the `def`, or a profile's name.
    pub text: String,
    /// A line beside it: the parameters, or the standard and section.
    pub detail: String,
}

/// The entries of one library file at `relative` (`library/fixtures.part`).
pub fn file_entries(relative: &str, text: &str) -> Vec<LibraryEntry> {
    let Some(document) = crate::syntax::parse_part(relative, text).document else {
        return Vec::new();
    };
    let params = |n: usize| match n {
        0 => "no inputs".to_owned(),
        1 => "1 input".to_owned(),
        n => format!("{n} inputs"),
    };
    if let Some(name) = relative
        .strip_prefix(&format!("{PROFILES}/"))
        .and_then(|n| n.strip_suffix(".part"))
    {
        let has_sketch = document
            .features
            .iter()
            .any(|f| f.id.0 == PROFILE_SKETCH && matches!(f.kind, FeatureKind::Sketch(_)));
        return if has_sketch {
            vec![LibraryEntry {
                kind: "profile",
                text: name.to_owned(),
                detail: "Project profile".into(),
            }]
        } else {
            Vec::new()
        };
    }
    if document.feature_defs.is_empty() {
        return vec![LibraryEntry {
            kind: "part",
            text: relative.to_owned(),
            detail: format!("Part · {}", params(document.inputs.len())),
        }];
    }
    document
        .feature_defs
        .iter()
        .map(|d| LibraryEntry {
            kind: "feature",
            text: format!("{relative}#feature_def.{}", d.name),
            detail: format!("Feature · {}", params(d.params.len())),
        })
        .collect()
}

/// The standard sizes, as library entries.
pub fn standard_entries() -> Vec<LibraryEntry> {
    crate::model::weldment::standard::all()
        .into_iter()
        .map(|s| LibraryEntry {
            kind: "profile",
            detail: format!("{} · A {:.2} cm²", s.standard, s.shape.area() * 1e4),
            text: s.designation,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURES: &str = "runmat 0.7\nstd 1\n\npart fixtures {\n    feature_def slot = {\n        params = [{ name = width, kind = length, default = 6 mm }],\n        features = []\n    }\n}\n";
    const PIN: &str = "runmat 0.7\nstd 1\n\npart pin {\n    input d : length = 6 mm\n    sketch s on origin.xy {\n        c = circle(center = [0 mm, 0 mm], radius = input.d)\n    }\n    feature body = extrude(profile = sketch.s, distance = 20 mm)\n}\n";

    #[test]
    fn a_file_offers_its_feature_defs_or_itself_as_a_part() {
        let e = file_entries("library/fixtures.part", FIXTURES);
        assert_eq!(e.len(), 1);
        assert_eq!(e[0].text, "library/fixtures.part#feature_def.slot");
        assert_eq!(e[0].detail, "Feature · 1 input");
        let e = file_entries("library/pin.part", PIN);
        assert_eq!(e[0].kind, "part");
        let def = definition("library/pin.part", PIN).unwrap();
        assert_eq!(def.params.len(), 1);
        assert_eq!(def.features.len(), 2);
        let def = definition("library/fixtures.part#feature_def.slot", FIXTURES).unwrap();
        assert_eq!(def.name, "library/fixtures.part#feature_def.slot");
        assert!(
            definition("library/fixtures.part#feature_def.boss", FIXTURES)
                .unwrap_err()
                .contains("no feature_def `boss`")
        );
    }
}
