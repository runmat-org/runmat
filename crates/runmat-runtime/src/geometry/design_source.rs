//! RunMat MCAD parts and assemblies as geometry sources.
//!
//! A `.part` or `.assembly` path is read through runmat-cad: the host gathers the design file and
//! every file it reads (the parts and sub-assemblies an assembly places, the folders it names
//! designs in, the STEP files a part imports), regenerates it once on the OCCT kernel and gets
//! the exact STEP AP242 bytes of that regeneration
//! ([`runmat_cad_wasm::exchange::study::study_geometry`]). Those bytes stand in for the file
//! wherever a geometry source is read (inspection, the checked snapshot, the run and the
//! preview), so the study's geometry and its region names come from one evaluation. No file is
//! written to the project.
//!
//! The bytes are deterministic, and their header carries the digest of the design's source
//! closure, so the geometry identity changes exactly when the design or one of its dependencies
//! changes.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use runmat_cad_wasm::exchange::study;
use runmat_cad_wasm::runtime::Files;

/// The design files a dependency folder is listed for.
const FOLDER_EXTENSIONS: &[&str] = &[".part", ".assembly"];

/// Whether `path` names a RunMat MCAD part or assembly.
pub fn is_design_geometry(path: &str) -> bool {
    study::is_study_design(path)
}

/// Where a design's files come from.
pub trait DesignFileSource {
    /// The text of the file at `path` (a normalized, slash-separated path), if present.
    fn read(&self, path: &str) -> Option<String>;
    /// The files directly in `folder` (no trailing slash).
    fn list(&self, folder: &str) -> Vec<String>;
}

/// The local filesystem.
pub struct DiskDesignFiles;

impl DesignFileSource for DiskDesignFiles {
    fn read(&self, path: &str) -> Option<String> {
        let bytes = runmat_filesystem::read(Path::new(path)).ok()?;
        String::from_utf8(bytes).ok()
    }

    fn list(&self, folder: &str) -> Vec<String> {
        let Ok(entries) = runmat_filesystem::read_dir(Path::new(folder)) else {
            return Vec::new();
        };
        entries
            .into_iter()
            .map(|entry| format!("{folder}/{}", entry.file_name().to_string_lossy()))
            .collect()
    }
}

/// An immutable captured view (a checked capture), with no fallback to live files.
pub struct CapturedDesignFiles<'a>(pub &'a BTreeMap<PathBuf, Vec<u8>>);

impl DesignFileSource for CapturedDesignFiles<'_> {
    fn read(&self, path: &str) -> Option<String> {
        let bytes = self.0.get(Path::new(path))?;
        String::from_utf8(bytes.clone()).ok()
    }

    fn list(&self, folder: &str) -> Vec<String> {
        self.0
            .keys()
            .filter(|path| path.parent() == Some(Path::new(folder)))
            .map(|path| design_path(path))
            .collect()
    }
}

/// The exact geometry bytes of the part or assembly at `path`, read from disk.
pub fn read_design_geometry(path: &Path) -> Result<Vec<u8>, String> {
    design_geometry(path, None, &DiskDesignFiles)
}

/// [`read_design_geometry`] with the design's own text supplied (an editor buffer or a file the
/// caller already read); its dependencies come from `source`.
pub fn design_geometry(
    path: &Path,
    text: Option<&[u8]>,
    source: &dyn DesignFileSource,
) -> Result<Vec<u8>, String> {
    let root = design_path(path);
    let text = match text {
        Some(bytes) => String::from_utf8(bytes.to_vec())
            .map_err(|_| format!("design file `{}` is not UTF-8 text", path.display()))?,
        None => source
            .read(&root)
            .ok_or_else(|| format!("cannot read design file `{}`", path.display()))?,
    };
    let files = gather(&root, text, source);
    evaluate(&root, &files).map_err(|message| {
        format!(
            "cannot evaluate design `{}` as study geometry: {message}",
            path.display()
        )
    })
}

/// A native path in the design compilers' slash-separated namespace.
fn design_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

/// The design at `root` and every file it reads, transitively.
fn gather(root: &str, text: String, source: &dyn DesignFileSource) -> Files {
    use runmat_cad_wasm::runtime::{dependencies, dependency_files};
    let mut files = Files::new();
    files.insert(root.to_owned(), text);
    let mut queue = vec![root.to_owned()];
    let mut visited = std::collections::BTreeSet::new();
    while let Some(path) = queue.pop() {
        if !visited.insert(path.clone()) || !is_design_geometry(&path) {
            continue;
        }
        let Some(text) = files.get(&path).cloned() else {
            continue;
        };
        for dependency in dependencies(&path, &text) {
            if let Some(folder) = dependency.strip_suffix('/') {
                for file in source.list(folder) {
                    if FOLDER_EXTENSIONS.iter().any(|x| file.ends_with(x))
                        && !files.contains_key(&file)
                    {
                        if let Some(text) = source.read(&file) {
                            files.insert(file, text);
                        }
                    }
                }
            } else if !files.contains_key(&dependency) {
                if let Some(text) = source.read(&dependency) {
                    files.insert(dependency.clone(), text);
                    queue.push(dependency);
                }
            }
        }
        for dependency in dependency_files(&path, &text, &files) {
            if !files.contains_key(&dependency) {
                let Some(text) = source.read(&dependency) else {
                    continue;
                };
                files.insert(dependency.clone(), text);
            }
            queue.push(dependency);
        }
    }
    files
}

#[cfg(all(not(target_arch = "wasm32"), feature = "occt-native"))]
fn evaluate(root: &str, files: &Files) -> Result<Vec<u8>, String> {
    kernel::evaluate(root, files)
}

#[cfg(not(all(not(target_arch = "wasm32"), feature = "occt-native")))]
fn evaluate(_root: &str, _files: &Files) -> Result<Vec<u8>, String> {
    Err("this build has no OCCT design kernel; open the study in RunMat Desktop or the CLI".into())
}

#[cfg(all(not(target_arch = "wasm32"), feature = "occt-native"))]
mod kernel {
    use std::collections::{BTreeMap, VecDeque};
    use std::sync::{Arc, Mutex};

    use runmat_cad_wasm::exchange::study;
    use runmat_cad_wasm::runtime::Files;
    use runmat_geometry_io::{ImportSource, OcctKernel};

    /// The STEP files the current evaluation's parts import, by path and by every path tail.
    static IMPORTS: Mutex<BTreeMap<String, ImportSource>> = Mutex::new(BTreeMap::new());
    /// One evaluation at a time: the imports above belong to it, and OCCT keeps bodies in one
    /// process-wide store.
    static EVALUATION: Mutex<()> = Mutex::new(());
    /// Recent results by the design's source closure. Regeneration dominates a check, and a
    /// check, a snapshot and a run read the same design in turn.
    static RESULTS: Mutex<VecDeque<(String, Vec<u8>)>> = Mutex::new(VecDeque::new());
    const RESULT_CAPACITY: usize = 8;

    /// Installs OCCT as the design kernel (once per process). Each kernel it makes reads
    /// imported STEP files from the current evaluation's files.
    fn install() {
        runmat_cad_wasm::kernel::install("occt", || {
            let mut kernel = OcctKernel::new();
            kernel.imports = IMPORTS.lock().unwrap_or_else(|e| e.into_inner()).clone();
            Box::new(kernel)
        });
    }

    pub(super) fn evaluate(root: &str, files: &Files) -> Result<Vec<u8>, String> {
        let _evaluation = EVALUATION.lock().unwrap_or_else(|e| e.into_inner());
        let key = format!("{root}\n{}", study::source_closure_digest(root, files));
        if let Some((_, bytes)) = RESULTS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .find(|(candidate, _)| candidate == &key)
        {
            return Ok(bytes.clone());
        }
        let bytes = evaluate_uncached(root, files)?;
        let mut results = RESULTS.lock().unwrap_or_else(|e| e.into_inner());
        if results.len() == RESULT_CAPACITY {
            results.pop_front();
        }
        results.push_back((key, bytes.clone()));
        Ok(bytes)
    }

    /// One regeneration; the caller holds `EVALUATION`.
    pub(super) fn evaluate_uncached(root: &str, files: &Files) -> Result<Vec<u8>, String> {
        install();
        if runmat_cad_wasm::kernel::name() != "occt" {
            return Err(format!(
                "the design kernel installed in this process is `{}`, not OCCT",
                runmat_cad_wasm::kernel::name()
            ));
        }
        let mut imports = BTreeMap::new();
        for (path, text) in files {
            if !(path.ends_with(".step") || path.ends_with(".stp")) {
                continue;
            }
            let source = ImportSource::StepText(Arc::from(text.as_str()));
            imports.insert(path.clone(), source.clone());
            for (i, _) in path.match_indices('/') {
                imports
                    .entry(path[i + 1..].to_string())
                    .or_insert_with(|| source.clone());
            }
        }
        *IMPORTS.lock().unwrap_or_else(|e| e.into_inner()) = imports;
        let bytes = study::study_geometry(root, files, None);
        IMPORTS.lock().unwrap_or_else(|e| e.into_inner()).clear();
        bytes
    }
}

/// Regenerates without the result cache, so a test can compare a fresh write with a cached one.
#[cfg(all(test, not(target_arch = "wasm32"), feature = "occt-native"))]
fn kernel_free_bytes_for_test(path: &Path, view: &BTreeMap<PathBuf, Vec<u8>>) -> Vec<u8> {
    let root = design_path(path);
    let source = CapturedDesignFiles(view);
    let files = gather(&root, source.read(&root).unwrap(), &source);
    kernel::evaluate_uncached(&root, &files).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Memory(BTreeMap<PathBuf, Vec<u8>>);

    #[test]
    fn gathering_follows_named_parts_and_their_imports_only() {
        let mut view = BTreeMap::new();
        let mut put = |path: &str, text: &str| {
            view.insert(PathBuf::from(path), text.as_bytes().to_vec());
        };
        put(
            "/p/sim/study.assembly",
            "runmat 0.7\nstd 1\n\nassembly study {\n    use part arm from \"../mech/arm\" as arm\n}\n",
        );
        put(
            "/p/mech/arm.part",
            "runmat 0.7\nstd 1\n\npart arm {\n    feature s = import(source = \"./arm.step\")\n}\n",
        );
        put("/p/mech/arm.step", "ISO-10303-21;");
        put("/p/elsewhere/unrelated.part", "runmat 0.7\n");
        let memory = Memory(view);
        let files = gather(
            "/p/sim/study.assembly",
            CapturedDesignFiles(&memory.0)
                .read("/p/sim/study.assembly")
                .unwrap(),
            &CapturedDesignFiles(&memory.0),
        );
        assert!(files.contains_key("/p/mech/arm.part"), "{:?}", files.keys());
        assert!(files.contains_key("/p/mech/arm.step"), "{:?}", files.keys());
        assert!(!files.contains_key("/p/elsewhere/unrelated.part"));
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "occt-native"))]
    #[test]
    fn a_part_evaluates_once_into_deterministic_named_exact_geometry() {
        let part = |height: &str| {
            format!(
                "runmat 0.7\nstd 1\n\npart puck {{\n    feature base = plane.offset(base = origin.xy, distance = 0 mm)\n    sketch s on plane.base {{\n        c = circle(center = [0 mm, 0 mm], radius = 7 mm)\n    }}\n    feature puck = extrude(profile = sketch.s, distance = {height})\n    region top : face = {{ select = [feature.puck.face.end], expect = exactly_one }}\n    output puck = {{ body = feature.puck.body, material = \"aluminum_6061\" }}\n}}\n"
            )
        };
        let view = |height: &str| {
            BTreeMap::from([(PathBuf::from("/p/sim/puck.part"), part(height).into_bytes())])
        };
        let path = Path::new("/p/sim/puck.part");
        let first = design_geometry(path, None, &CapturedDesignFiles(&view("13 mm"))).unwrap();
        let text = String::from_utf8(first.clone()).unwrap();
        assert!(text.contains("ADVANCED_FACE('top'"), "the region names its face");
        assert!(text.contains("PRODUCT('puck'"), "the output names its body");
        assert!(text.contains("sources sha256:"), "the header names the source closure");
        // The same sources give the same bytes; a dimension edit gives different ones.
        let again = design_geometry(path, None, &CapturedDesignFiles(&view("13 mm"))).unwrap();
        assert_eq!(first, again);
        let taller = design_geometry(path, None, &CapturedDesignFiles(&view("14 mm"))).unwrap();
        assert_ne!(first, taller);
        // A different process-level cache entry still writes the same bytes.
        let fresh = super::kernel_free_bytes_for_test(path, &view("13 mm"));
        assert_eq!(first, fresh);
        let inspected = runmat_geometry_io::detect_geometry_format("puck.part", &first);
        assert_eq!(inspected, runmat_geometry_io::GeometryFormat::Step);
    }

    #[test]
    fn only_parts_and_assemblies_are_design_geometry() {
        assert!(is_design_geometry("sim/arm.part"));
        assert!(is_design_geometry("sim/arm_study.assembly"));
        assert!(!is_design_geometry("sim/geometry/arm_study.step"));
        assert!(!is_design_geometry("sim/board.circuit.layout"));
    }
}
