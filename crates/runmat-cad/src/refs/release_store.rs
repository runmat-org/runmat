//! The project's release store: the released revisions of a design file, kept in the project
//! beside it.
//!
//! Releasing `/p/bracket.part` saves its text as the next revision (`R1`, `R2`, …) in
//! `/p/.runmat/releases/bracket.part.json`. An assembly that pins a component (`use part bracket
//! from "./bracket.part" @ "R2"`) regenerates that revision's text, so later edits to the live file
//! don't move it. A pinned revision that isn't in the store is reported as missing
//! ([`crate::refs::XrefProblem::RevisionMissing`]); the store's last revision is the file's
//! latest release (Update pins → Latest released).
//!
//! The store is JSON: `{ "file": "bracket.part", "releases": [{ "revision": "R1", "text": "…" }] }`.
//! A released revision never changes: a release adds one at the end.

use crate::ids::RevisionId;
use serde::{Deserialize, Serialize};

/// The store's folder inside the released file's folder (internal: the file tree hides it).
pub const STORE_FOLDER: &str = ".runmat/releases";

/// Where the release store of the design file `file` is: `<folder>/.runmat/releases/<name>.json`.
pub fn store_path(file: &str) -> String {
    match file.rsplit_once('/') {
        Some((folder, name)) => format!("{folder}/{STORE_FOLDER}/{name}.json"),
        None => format!("{STORE_FOLDER}/{file}.json"),
    }
}

/// The release-store folder that holds the stores of the designs in `folder` (`/p/` →
/// `/p/.runmat/releases/`).
pub fn store_folder(folder: &str) -> String {
    let folder = folder.trim_end_matches('/');
    format!("{folder}/{STORE_FOLDER}/")
}

/// One released revision: its id and the file's text when it was released.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Release {
    pub revision: RevisionId,
    pub text: String,
}

/// The released revisions of one design file, oldest first.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReleaseStore {
    /// The released file's name in its folder.
    pub file: String,
    #[serde(default)]
    pub releases: Vec<Release>,
}

impl ReleaseStore {
    /// An empty store for the design file `file`.
    pub fn new(file: &str) -> Self {
        Self {
            file: file.rsplit('/').next().unwrap_or(file).to_owned(),
            releases: Vec::new(),
        }
    }

    /// The store in `text`, or why it can't be read.
    pub fn parse(text: &str) -> Result<Self, String> {
        serde_json::from_str(text).map_err(|e| format!("the release store doesn't read: {e}"))
    }

    pub fn to_text(&self) -> String {
        let mut text = serde_json::to_string_pretty(self).unwrap_or_default();
        text.push('\n');
        text
    }

    /// The revision ids, oldest first.
    pub fn revisions(&self) -> Vec<RevisionId> {
        self.releases.iter().map(|r| r.revision.clone()).collect()
    }

    /// The latest released revision.
    pub fn latest(&self) -> Option<&Release> {
        self.releases.last()
    }

    /// The text released as `revision`.
    pub fn text(&self, revision: &RevisionId) -> Option<&str> {
        self.releases
            .iter()
            .find(|r| &r.revision == revision)
            .map(|r| r.text.as_str())
    }

    /// The id the next release takes: `R` and one more than the highest `R<n>` so far.
    pub fn next_revision(&self) -> RevisionId {
        let n = self
            .releases
            .iter()
            .filter_map(|r| r.revision.0.strip_prefix('R')?.parse::<u32>().ok())
            .max()
            .unwrap_or(0);
        RevisionId::new(format!("R{}", n + 1))
    }

    /// Releases `text` as the next revision. A text equal to the latest release's is refused:
    /// that revision already holds it.
    pub fn release(&mut self, text: &str) -> Result<RevisionId, String> {
        if let Some(latest) = self.latest() {
            if latest.text == text {
                return Err(format!(
                    "{} is unchanged since {}, its latest release",
                    self.file, latest.revision
                ));
            }
        }
        let revision = self.next_revision();
        self.releases.push(Release {
            revision: revision.clone(),
            text: text.to_owned(),
        });
        Ok(revision)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_store_sits_in_the_folders_internal_releases_folder() {
        assert_eq!(
            store_path("/p/parts/bracket.part"),
            "/p/parts/.runmat/releases/bracket.part.json"
        );
        assert_eq!(store_path("top.assembly"), ".runmat/releases/top.assembly.json");
        assert_eq!(store_folder("/p/parts/"), "/p/parts/.runmat/releases/");
    }

    #[test]
    fn releases_count_up_and_an_unchanged_text_is_refused() {
        let mut store = ReleaseStore::new("/p/bracket.part");
        assert_eq!(store.release("a").unwrap().0, "R1");
        assert_eq!(store.release("b").unwrap().0, "R2");
        assert!(store.release("b").unwrap_err().contains("unchanged since R2"));
        let read = ReleaseStore::parse(&store.to_text()).unwrap();
        assert_eq!(read, store);
        assert_eq!(read.text(&RevisionId::new("R1")), Some("a"));
        assert_eq!(read.latest().map(|r| r.revision.0.as_str()), Some("R2"));
    }
}
