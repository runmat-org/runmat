//! The design closure read through the filesystem provider's async API.
//!
//! The closure walk ([`super::gather`]) is synchronous and runs over a [`DesignFileSource`]. Here
//! it runs over files read so far; every path or folder it asks for and does not find is
//! recorded, read asynchronously, and the walk runs again until it asks for nothing new. One
//! walk therefore decides the closure for disk, captured and async reads alike.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use runmat_cad_wasm::runtime::Files;

use super::{gather, DesignFileSource};

/// The text of the file at `path`, if it can be read and is UTF-8.
pub(super) async fn read_text(path: &str) -> Option<String> {
    let bytes = runmat_filesystem::read_async(Path::new(path)).await.ok()?;
    String::from_utf8(bytes).ok()
}

async fn list(folder: &str) -> Vec<String> {
    let Ok(entries) = runmat_filesystem::read_dir_async(Path::new(folder)).await else {
        return Vec::new();
    };
    entries
        .into_iter()
        .map(|entry| format!("{folder}/{}", entry.file_name().to_string_lossy()))
        .collect()
}

/// The design at `root` (whose text is `text`) and every file it reads, transitively.
pub(super) async fn capture(root: &str, text: String) -> Files {
    let mut read = BTreeMap::new();
    let mut listed = BTreeMap::new();
    loop {
        let (files, wanted) = {
            let source = Recorded {
                read: &read,
                listed: &listed,
                wanted: RefCell::default(),
            };
            let files = gather(root, text.clone(), &source);
            (files, source.wanted.into_inner())
        };
        if wanted.files.is_empty() && wanted.folders.is_empty() {
            return files;
        }
        for path in wanted.files {
            let text = read_text(&path).await;
            read.insert(path, text);
        }
        for folder in wanted.folders {
            let entries = list(&folder).await;
            listed.insert(folder, entries);
        }
    }
}

#[derive(Default)]
struct Wanted {
    files: BTreeSet<String>,
    folders: BTreeSet<String>,
}

/// Files and listings read so far; a miss is recorded for the next round. A file that could not
/// be read is remembered as absent, so each path is read at most once.
struct Recorded<'a> {
    read: &'a BTreeMap<String, Option<String>>,
    listed: &'a BTreeMap<String, Vec<String>>,
    wanted: RefCell<Wanted>,
}

impl DesignFileSource for Recorded<'_> {
    fn read(&self, path: &str) -> Option<String> {
        match self.read.get(path) {
            Some(text) => text.clone(),
            None => {
                self.wanted.borrow_mut().files.insert(path.to_owned());
                None
            }
        }
    }

    fn list(&self, folder: &str) -> Vec<String> {
        match self.listed.get(folder) {
            Some(entries) => entries.clone(),
            None => {
                self.wanted.borrow_mut().folders.insert(folder.to_owned());
                Vec::new()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_filesystem::{FsProvider, MemoryFsProvider};
    use std::sync::Arc;

    /// The async capture reads the same closure the synchronous walk reads from a full view.
    #[test]
    fn the_async_capture_reads_the_same_closure_as_a_full_view() {
        let _guard = runmat_filesystem::provider_override_lock();
        let provider = Arc::new(MemoryFsProvider::with_current_dir("/p"));
        let files = [
            (
                "/p/sim/study.assembly",
                "runmat 0.7\nstd 1\n\nassembly study {\n    use part arm from \"../mech/arm\" as arm\n}\n",
            ),
            (
                "/p/mech/arm/arm.part",
                "runmat 0.7\nstd 1\n\npart arm {\n    feature s = import(source = \"./arm.step\", digest = arm_step)\n}\n",
            ),
            ("/p/mech/arm/arm.step", "ISO-10303-21;"),
            ("/p/elsewhere/unrelated.part", "runmat 0.7\n"),
            ("/p/mech/arm/notes.md", "not a design file"),
        ];
        for (path, text) in files {
            let path = Path::new(path);
            futures::executor::block_on(provider.create_dir_all(path.parent().unwrap())).unwrap();
            futures::executor::block_on(provider.write(path, text.as_bytes())).unwrap();
        }
        let view = files
            .iter()
            .map(|(path, text)| (std::path::PathBuf::from(path), text.as_bytes().to_vec()))
            .collect::<BTreeMap<_, _>>();
        let root = "/p/sim/study.assembly";
        let expected = gather(
            root,
            files[0].1.to_owned(),
            &super::super::CapturedDesignFiles(&view),
        );
        let captured = runmat_filesystem::with_provider_override(provider, || {
            futures::executor::block_on(capture(root, files[0].1.to_owned()))
        });
        assert_eq!(captured, expected);
        assert!(captured.contains_key("/p/mech/arm/arm.step"));
        assert!(!captured.contains_key("/p/elsewhere/unrelated.part"));
    }
}
