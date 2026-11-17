//! Studies whose geometry is a RunMat MCAD part: the check reads the design through the async
//! provider API and regenerates on the installed study-geometry provider, and without one it
//! reports the reason at once instead of waiting.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use runmat_filesystem::{FsProvider, MemoryFsProvider};
use runmat_runtime::analysis::{FeaCheckStatus, FeaSourceCheckResult, LogicalSourcePath};
use runmat_runtime::geometry::design_source::{
    set_study_geometry_provider, StudyGeometryFuture, StudyGeometryProvider,
};

use crate::{FeaWorkingCheckRequest, RunMatSession};

/// A project in the browser's shape: one in-memory provider rooted at `/project`.
fn project(files: &[(&str, &[u8])]) -> Arc<MemoryFsProvider> {
    let provider = Arc::new(MemoryFsProvider::with_current_dir("/project"));
    for (path, bytes) in files {
        let path = Path::new(path);
        futures::executor::block_on(provider.create_dir_all(path.parent().unwrap())).unwrap();
        futures::executor::block_on(provider.write(path, bytes)).unwrap();
    }
    provider
}

fn full_check(provider: Arc<MemoryFsProvider>, study: &str) -> FeaSourceCheckResult {
    runmat_filesystem::with_provider_override(provider, || {
        let mut session = RunMatSession::with_options(false, false).unwrap();
        futures::executor::block_on(session.check_fea_source_path(Path::new(study))).unwrap()
    })
}

fn working_check(provider: Arc<MemoryFsProvider>, root: &str) -> FeaSourceCheckResult {
    runmat_filesystem::with_provider_override(provider, || {
        let mut session = RunMatSession::with_options(false, false).unwrap();
        futures::executor::block_on(session.check_fea_working_source(FeaWorkingCheckRequest {
            root: LogicalSourcePath::new(root).unwrap(),
            overlays: vec![],
        }))
        .unwrap()
    })
}

fn messages(result: &FeaSourceCheckResult) -> Vec<String> {
    result
        .diagnostics
        .iter()
        .map(|diagnostic| format!("{}: {}", diagnostic.code, diagnostic.message))
        .collect()
}

/// Installs a provider for one test and removes it afterwards.
struct Installed;

impl Installed {
    fn new(provider: Rc<dyn StudyGeometryProvider>) -> Self {
        set_study_geometry_provider(Some(provider));
        Self
    }
}

impl Drop for Installed {
    fn drop(&mut self) {
        set_study_geometry_provider(None);
    }
}

/// Answers every regeneration with fixed STEP bytes and records what it was sent.
struct Stub {
    step: Vec<u8>,
    calls: RefCell<Vec<(String, Vec<String>)>>,
}

impl StudyGeometryProvider for Stub {
    fn study_geometry(&self, root: &str, files: &BTreeMap<String, String>) -> StudyGeometryFuture {
        self.calls
            .borrow_mut()
            .push((root.to_owned(), files.keys().cloned().collect()));
        let step = self.step.clone();
        Box::pin(async move { Ok(step) })
    }
}

const MISSING_PART_STUDY: &str = r#"geometry skid {
    file = "./does_not_exist.part"
    region body : solid { select name "landing_skid"; expect exactly_one }
}
"#;

#[test]
fn a_study_on_a_missing_part_is_reported_at_once() {
    let _guard = runmat_filesystem::provider_override_lock();
    let study = "/project/simulation/landing_skid.fea";
    for installed in [false, true] {
        let _provider = installed.then(|| {
            Installed::new(Rc::new(Stub {
                step: Vec::new(),
                calls: RefCell::default(),
            }))
        });
        let started = Instant::now();
        let full = full_check(project(&[(study, MISSING_PART_STUDY.as_bytes())]), study);
        let working = working_check(
            project(&[(study, MISSING_PART_STUDY.as_bytes())]),
            "simulation/landing_skid.fea",
        );
        assert!(started.elapsed() < Duration::from_secs(30));
        for result in [&full, &working] {
            assert_ne!(result.status, FeaCheckStatus::Valid);
            let messages = messages(result);
            assert!(
                messages
                    .iter()
                    .any(|message| message.contains("does_not_exist.part")
                        && (message.contains("cannot read design file")
                            || message.contains("cannot evaluate design"))),
                "provider installed: {installed}; {messages:#?}"
            );
        }
    }
}

/// Without an in-process kernel and without a provider, the reason names the missing host.
#[cfg(not(feature = "occt-native"))]
#[test]
fn without_a_kernel_or_provider_the_reason_is_given_before_any_read() {
    let _guard = runmat_filesystem::provider_override_lock();
    let study = "/project/simulation/landing_skid.fea";
    let part_study = MISSING_PART_STUDY.replace("does_not_exist", "skid");
    let provider = project(&[
        (study, part_study.as_bytes()),
        (
            "/project/simulation/skid.part",
            b"runmat 0.7\nstd 1\n\npart skid {}\n",
        ),
    ]);
    let result = full_check(provider, study);
    let messages = messages(&result);
    assert!(
        messages
            .iter()
            .any(|message| message.contains("no OCCT design kernel")),
        "{messages:#?}"
    );
}

/// The STEP the provider returns stands for the part: the study checks valid in the full and
/// the working check, and the provider receives the part's whole closure.
#[cfg(feature = "occt-native")]
#[test]
fn a_part_study_checks_valid_on_an_installed_study_geometry_provider() {
    let _guard = runmat_filesystem::provider_override_lock();
    let step = include_bytes!("../../../../docs/simulation/examples/geometry/square-plate.step");
    let study = include_str!("../../../../docs/simulation/examples/square-plate.fea")
        .replace("./geometry/square-plate.step", "./plate.part");
    let files: [(&str, &[u8]); 3] = [
        ("/project/plate.fea", study.as_bytes()),
        (
            "/project/plate.part",
            b"runmat 0.7\nstd 1\n\npart plate {\n    feature s = import(source = \"./plate.step\", digest = plate_step)\n}\n",
        ),
        ("/project/plate.step", step),
    ];
    let stub = Rc::new(Stub {
        step: step.to_vec(),
        calls: RefCell::default(),
    });
    let _installed = Installed::new(stub.clone());
    let full = full_check(project(&files), "/project/plate.fea");
    assert_eq!(full.status, FeaCheckStatus::Valid, "{:#?}", messages(&full));
    let working = working_check(project(&files), "plate.fea");
    assert_eq!(
        working.status,
        FeaCheckStatus::Valid,
        "{:#?}",
        messages(&working)
    );
    let calls = stub.calls.borrow();
    assert!(!calls.is_empty());
    for (root, closure) in calls.iter() {
        assert_eq!(root, "/project/plate.part");
        assert!(
            closure.contains(&"/project/plate.step".to_string()),
            "the imported STEP travels with the part: {closure:?}"
        );
    }
}

/// G104 side finding: the editor's working check and the full check parse a study the same
/// way. This study (from the browser report) starts with a design file's version header, which
/// `.fea` has none of; both checks report those two lines and nothing else as syntax.
#[test]
fn the_working_and_full_checks_report_the_same_syntax() {
    let _guard = runmat_filesystem::provider_override_lock();
    // The part sits beside the study, so the full check (rooted at the study's folder) and the
    // working check (rooted at the project) resolve it alike.
    let text = include_str!("fea_source_tests/skr5_landing_skid.fea").replace(
        "../mechanical/body/landing_skid.part",
        "./landing_skid.part",
    );
    let study = "/project/simulation/landing_skid.fea";
    let syntax = |result: &FeaSourceCheckResult| {
        result
            .diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.code.starts_with("fea.syntax."))
            .map(|diagnostic| {
                let span = diagnostic.location.as_ref().map(|location| location.span);
                (diagnostic.code.clone(), span, diagnostic.message.clone())
            })
            .collect::<Vec<_>>()
    };
    let full = syntax(&full_check(project(&[(study, text.as_bytes())]), study));
    let working = syntax(&working_check(
        project(&[(study, text.as_bytes())]),
        "simulation/landing_skid.fea",
    ));
    assert_eq!(full, working);
    let starts = full
        .iter()
        .map(|(_, span, _)| span.map(|span| span.start))
        .collect::<Vec<_>>();
    assert_eq!(starts, [Some(0), Some(11)], "{full:#?}");
    assert!(full.iter().all(
        |(code, _, message)| code == "fea.syntax.expecteddeclaration"
            && message.contains("no version header")
    ));
    // Without the header lines the study has no syntax diagnostics in either check.
    let body = text.split_once("std 1\n").unwrap().1;
    let full = syntax(&full_check(project(&[(study, body.as_bytes())]), study));
    assert!(full.is_empty(), "{full:#?}");
}
