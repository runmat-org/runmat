use super::*;
use crate::context::host_effect::{
    HostEffectOutput, HostEffectRequest, HostEffectResourceEvidence, HostEffectSource,
    HostEffectWorkload,
};
use crate::context::{
    ParallelCapability, RuntimeContext, RuntimeHostEffectResourceService, RuntimeParallelResources,
    RuntimeParallelService, RuntimePlacementService, RuntimeServicePorts,
    RuntimeWholeWorkSelection,
};
use futures::executor::block_on;
use runmat_execution::{
    PlacementDecision, PlacementExecutionAuthorityRevision, PlacementFeedback,
    PlacementInvalidation, PlacementPlanRequest, PlacementRevision, ResourceByteBound,
    WholeWorkDecision, WholeWorkDecisionKind, WholeWorkObservation, WholeWorkObservationOutcome,
    WholeWorkPlacementRequest,
};
use std::cell::{Cell, RefCell};
use std::io::{Cursor, Write};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use zip::write::SimpleFileOptions;

const SELECTED_CWD: &str = "/selected-cwd";
const SELECTED_ARCHIVE: &str = "/selected-cwd/archive.zip";
const REPLACEMENT_CWD: &str = "/replacement-cwd";

fn archive_bytes(contents: &[u8]) -> Vec<u8> {
    let mut archive = zip::ZipWriter::new(Cursor::new(Vec::new()));
    archive
        .start_file("owner.txt", SimpleFileOptions::default())
        .expect("archive entry");
    archive.write_all(contents).expect("archive content");
    archive.finish().expect("completed archive").into_inner()
}

struct ExactArchiveResources {
    selected: Arc<dyn vfs::FsProvider>,
    observed: RefCell<Vec<(PathBuf, PathBuf)>>,
    valid: Arc<AtomicBool>,
}

struct ArchiveLiveness(Arc<AtomicBool>);
impl crate::context::host_effect::HostEffectResourceLivenessLease for ArchiveLiveness {
    fn is_current(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

impl RuntimeHostEffectResourceService for ExactArchiveResources {
    fn admission(
        &self,
        request: &HostEffectRequest<'_>,
        _: &RuntimeContext,
    ) -> Option<crate::context::host_effect::HostEffectResourceAdmission> {
        let HostEffectWorkload::ArchiveExtract {
            source: HostEffectSource::LocalArchive(source),
            destination: Some(destination),
            output: HostEffectOutput::ReturnedValue,
        } = request.workload
        else {
            return None;
        };
        self.observed
            .borrow_mut()
            .push((source.to_path_buf(), destination.to_path_buf()));
        (request.operation == BUILTIN_NAME
            && source == Path::new(SELECTED_ARCHIVE)
            && destination == Path::new(SELECTED_CWD)
            && Arc::ptr_eq(request.filesystem, &self.selected)
            && request.domain_facts == Digest::sha256(request.workload.identity_bytes()))
        .then_some(HostEffectResourceEvidence {
            authority_revision: Digest::sha256(b"exact-archive-resource-witness"),
            scratch_bytes: ResourceByteBound::Known(1024),
            returned_value_bytes: ResourceByteBound::Known(1024),
        })
        .map(
            |evidence| crate::context::host_effect::HostEffectResourceAdmission {
                evidence,
                liveness_lease: Arc::new(ArchiveLiveness(self.valid.clone())),
                output_allocation_permit: None,
            },
        )
    }
}

struct FiniteHostBudget;
impl RuntimeParallelService for FiniteHostBudget {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }
    fn placement_resources(&self) -> RuntimeParallelResources {
        RuntimeParallelResources {
            memory_available_bytes: Some(4096),
            ..Default::default()
        }
    }
}

struct SwapDuringSelection {
    admission: Rc<ExactArchiveResources>,
    replacement: Arc<vfs::MemoryFsProvider>,
    global_swap: RefCell<Option<vfs::ProviderGuard>>,
    selections: Cell<usize>,
    observations: RefCell<Vec<WholeWorkObservation>>,
}

impl RuntimePlacementService for SwapDuringSelection {
    fn whole_work_host_service_revision(
        &self,
        program: Option<runmat_execution::ProgramRevision>,
        host_service: Digest,
    ) -> Option<PlacementRevision> {
        let digest = Digest::sha256(b"archive-path-placement-witness");
        Some(PlacementRevision {
            program,
            catalog: digest,
            compiler: digest,
            authority: PlacementExecutionAuthorityRevision::HostService(host_service),
            policy: digest,
        })
    }

    fn select_whole_work(
        &self,
        request: WholeWorkPlacementRequest,
    ) -> Result<RuntimeWholeWorkSelection, RuntimeError> {
        request
            .validate()
            .expect("well-formed complete archive route");
        assert_eq!(request.resources.memory_available_bytes, Some(4096));
        let candidate = request.candidates.first().expect("host effect candidate");
        assert_eq!(
            candidate.resources.scratch_bytes,
            ResourceByteBound::Known(1024)
        );
        assert_eq!(
            candidate.resources.retained_bytes,
            ResourceByteBound::Known(1024)
        );
        assert_eq!(
            self.admission.observed.borrow().as_slice(),
            vec![
                (PathBuf::from(SELECTED_ARCHIVE), PathBuf::from(SELECTED_CWD));
                self.selections.get() + 1
            ]
            .as_slice()
        );
        self.selections.set(self.selections.get() + 1);

        let guard = vfs::replace_provider(self.replacement.clone());
        // The selection callback is entered from an async invocation. Change
        // virtual CWD on another thread to avoid nesting the FS sync adapter's
        // executor inside that invocation's executor.
        std::thread::spawn(|| vfs::set_current_dir(REPLACEMENT_CWD))
            .join()
            .expect("CWD worker completed")
            .expect("replacement CWD exists");
        *self.global_swap.borrow_mut() = Some(guard);

        Ok(RuntimeWholeWorkSelection {
            decision: WholeWorkDecision {
                signature: request.signature.clone(),
                selected: candidate.selected(&request.signature.operation),
                kind: WholeWorkDecisionKind::Unmeasured,
                predicted_total_ns: None,
            },
            probe: None,
            observation_permit: None,
        })
    }

    fn observe_whole_work(&self, observation: WholeWorkObservation) -> Result<(), RuntimeError> {
        self.observations.borrow_mut().push(observation);
        Ok(())
    }
    fn plan(&self, _: PlacementPlanRequest) -> Result<PlacementDecision, RuntimeError> {
        unreachable!("archive uses whole-work selection")
    }
    fn observe(&self, _: PlacementFeedback) -> Result<(), RuntimeError> {
        unreachable!("archive uses complete-call feedback")
    }
    fn invalidate(&self, _: PlacementInvalidation) {}
}

#[test]
fn selected_relative_archive_and_default_output_survive_global_owner_and_cwd_swap() {
    // Provider replacement and virtual CWD are process-wide. Keep this
    // otherwise unchanged witness isolated from parallel archive tests.
    const CHILD_MARKER: &str = "RUNMAT_UNZIP_OWNER_CAPTURE_TEST_CHILD";
    if std::env::var_os(CHILD_MARKER).is_none() {
        let output = std::process::Command::new(std::env::current_exe().expect("test executable"))
            .args([
                "--exact",
                "builtins::io::archive::unzip::path_authority_tests::selected_relative_archive_and_default_output_survive_global_owner_and_cwd_swap",
                "--test-threads=1",
                "--nocapture",
            ])
            .env(CHILD_MARKER, "1")
            .output()
            .expect("isolated archive filesystem owner witness");
        assert!(
            output.status.success(),
            "archive owner witness failed: {}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed; 0 failed"));
        return;
    }

    let _lock = vfs::provider_override_lock();
    let selected = Arc::new(vfs::MemoryFsProvider::with_current_dir(SELECTED_CWD));
    let replacement = Arc::new(vfs::MemoryFsProvider::with_current_dir(REPLACEMENT_CWD));
    selected
        .write_project_path(SELECTED_ARCHIVE, &archive_bytes(b"selected payload"))
        .expect("selected ZIP");
    replacement
        .write_project_path(SELECTED_ARCHIVE, &archive_bytes(b"foreign payload"))
        .expect("same absolute path on foreign owner");
    replacement
        .write_project_path(
            "/replacement-cwd/archive.zip",
            &archive_bytes(b"foreign CWD payload"),
        )
        .expect("same relative path under foreign CWD");
    let _selected_guard = vfs::replace_provider(selected.clone());
    vfs::set_current_dir(SELECTED_CWD).expect("selected CWD");

    let selected_owner: Arc<dyn vfs::FsProvider> = selected.clone();
    let resources = Rc::new(ExactArchiveResources {
        selected: selected_owner,
        observed: RefCell::new(Vec::new()),
        valid: Arc::new(AtomicBool::new(true)),
    });
    let placement = Rc::new(SwapDuringSelection {
        admission: resources.clone(),
        replacement: replacement.clone(),
        global_swap: RefCell::new(None),
        selections: Cell::new(0),
        observations: RefCell::new(Vec::new()),
    });
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_host_effect_resources(resources.clone())
                .with_placement(placement.clone())
                .with_parallel(Rc::new(FiniteHostBudget)),
        );
    let result = block_on(evaluate_with_output(
        crate::call::FunctionArgumentView::from_slice(&[Value::String("archive.zip".into())]),
        Some(&runtime),
        HostEffectOutput::ReturnedValue,
    ))
    .expect("admitted extraction");
    let expected = Path::new(SELECTED_CWD)
        .join("owner.txt")
        .to_str()
        .expect("UTF-8 fixture path")
        .to_owned();
    assert_eq!(result.filenames, vec![expected.clone()]);
    result
        .into_value_with(std::convert::identity)
        .expect("published filename cell");

    assert_eq!(placement.selections.get(), 1);
    assert_eq!(
        resources.observed.borrow().as_slice(),
        &[(PathBuf::from(SELECTED_ARCHIVE), PathBuf::from(SELECTED_CWD))]
    );
    assert_eq!(placement.observations.borrow().len(), 1);
    assert_eq!(
        placement.observations.borrow()[0].outcome,
        WholeWorkObservationOutcome::Success
    );
    assert_eq!(
        selected
            .read_project_path("/selected-cwd/owner.txt")
            .expect("selected output"),
        b"selected payload"
    );

    let selected_owner: Arc<dyn vfs::FsProvider> = selected.clone();
    let workload = HostEffectWorkload::ArchiveExtract {
        source: HostEffectSource::LocalArchive(Path::new(SELECTED_ARCHIVE)),
        destination: Some(Path::new(SELECTED_CWD)),
        output: HostEffectOutput::ReturnedValue,
    };
    let receipt = crate::context::host_effect::select_host_effect(
        Some(&runtime),
        HostEffectRequest {
            operation: BUILTIN_NAME,
            workload,
            domain_facts: Digest::sha256(workload.identity_bytes()),
            filesystem: &selected_owner,
        },
    )
    .expect("publication selection")
    .expect("selected receipt");
    assert_eq!(placement.selections.get(), 2);
    let second = UnzipResult {
        filenames: vec![expected],
        receipt: Some(receipt),
    };
    resources.valid.store(false, Ordering::SeqCst);
    let error = second
        .into_value_with(std::convert::identity)
        .expect_err("revoked authority prevents filename publication");
    assert_eq!(
        error.identifier(),
        Some("RunMat:Placement:StaleResourceAuthority")
    );
    assert_eq!(placement.observations.borrow().len(), 1);
    assert_eq!(
        selected
            .read_project_path("/selected-cwd/owner.txt")
            .expect("committed extraction is not rolled back"),
        b"selected payload"
    );
    assert!(replacement
        .read_project_path("/selected-cwd/owner.txt")
        .is_err());
    assert!(replacement
        .read_project_path("/replacement-cwd/owner.txt")
        .is_err());
    placement.global_swap.borrow_mut().take();
}
