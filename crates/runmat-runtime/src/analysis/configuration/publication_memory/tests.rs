use super::*;
use crate::analysis::storage::AnalysisPublicationError;
use crate::analysis::{replace_fea_runtime, FeaRuntimeConfig};
use std::sync::atomic::{AtomicU64, Ordering};

/// A host authority whose resident memory the test moves.
struct Host {
    admitted: u64,
    resident: AtomicU64,
}
impl PublicationMemoryAuthority for Host {
    fn publication_memory(&self) -> Result<PublicationMemory, String> {
        Ok(PublicationMemory {
            admitted_bytes: self.admitted,
            resident_bytes: self.resident.load(Ordering::SeqCst),
        })
    }
}

fn with_cap<R>(cap: Option<u64>, body: impl FnOnce() -> R) -> R {
    let previous = replace_fea_runtime(FeaRuntimeConfig {
        publication_staging_cap_bytes: cap,
        ..super::super::current_fea_runtime_config()
    })
    .unwrap();
    let result = body();
    replace_fea_runtime(previous).unwrap();
    result
}

#[test]
fn budget_is_the_admitted_run_memory_less_what_the_process_holds() {
    let _guard = crate::analysis::tests::analysis_test_guard();
    const GIB: u64 = 1 << 30;
    let host = Arc::new(Host {
        admitted: 16 * GIB,
        resident: AtomicU64::new(3 * GIB),
    });
    let previous = install_publication_memory_authority(host.clone()).unwrap();
    with_cap(None, || {
        assert_eq!(super::super::publication_staging_budget(), Ok(13 * GIB));
        // The budget is read when a publication starts, so it follows what the run holds then.
        host.resident.store(15 * GIB, Ordering::SeqCst);
        assert_eq!(super::super::publication_staging_budget(), Ok(GIB));
        // Far above the former fixed 128 MiB when the run's memory allows it.
        host.resident.store(0, Ordering::SeqCst);
        assert_eq!(super::super::publication_staging_budget(), Ok(16 * GIB));
    });
    // An explicit host cap lowers the budget and never raises it.
    with_cap(Some(2 * GIB), || {
        assert_eq!(super::super::publication_staging_budget(), Ok(2 * GIB));
    });
    with_cap(Some(64 * GIB), || {
        assert_eq!(super::super::publication_staging_budget(), Ok(16 * GIB));
    });
    // A run that already holds all its admitted memory has nothing left to stage.
    host.resident.store(16 * GIB, Ordering::SeqCst);
    let exhausted = with_cap(None, super::super::publication_staging_budget).unwrap_err();
    assert!(exhausted.contains("No memory is left to save this study's results"));
    assert!(exhausted.contains("Close other applications"));
    match previous {
        Some(previous) => install_publication_memory_authority(previous).map(|_| ()),
        None => clear_publication_memory_authority().map(|_| ()),
    }
    .unwrap();
}

#[test]
fn a_host_without_an_authority_publishes_within_this_machines_memory() {
    let _guard = crate::analysis::tests::analysis_test_guard();
    let previous = clear_publication_memory_authority().unwrap();
    let machine = with_cap(None, super::super::publication_staging_budget).unwrap();
    assert!(machine > 0);
    assert!(
        machine <= runmat_cpu_execution::machine_memory::available_memory_bytes().unwrap(),
        "{machine}"
    );
    // A host cap still lowers the machine's budget and never raises it.
    with_cap(Some(5_000), || {
        assert_eq!(super::super::publication_staging_budget(), Ok(5_000));
    });
    with_cap(Some(u64::MAX), || {
        assert!(super::super::publication_staging_budget().unwrap() < u64::MAX);
    });
    if let Some(previous) = previous {
        install_publication_memory_authority(previous).unwrap();
    }
}

#[test]
fn over_budget_refusal_is_a_plain_sentence_that_names_the_size_and_the_remedy() {
    let _guard = crate::analysis::tests::analysis_test_guard();
    let facts = "retained=0, preparation=119041313, source_workspace=0, additional=21857918, \
                 required=140899231, maximum=134217728 bytes";
    let message = with_cap(None, || {
        over_budget_message(140_899_231, 134_217_728, facts)
    });
    assert!(
        message.starts_with(
            "Saving this study's results needs 140.9 MB of memory, more than the 134.2 MB \
             available: the memory this run may use to save results"
        ),
        "{message}"
    );
    for remedy in [
        "free memory",
        "coarser mesh",
        "fewer result outputs",
        "run the study again",
    ] {
        assert!(message.contains(remedy), "missing `{remedy}`: {message}");
    }
    assert!(
        message.ends_with(&format!("Details: {facts}.")),
        "{message}"
    );
    // Under a host cap, the remedy names the cap.
    let capped = with_cap(Some(134_217_728), || {
        over_budget_message(140_899_231, 134_217_728, facts)
    });
    assert!(
        capped.contains("raise the host's publication staging cap"),
        "{capped}"
    );

    // The typed refusal displays as that sentence; phase context follows it, and no
    // serialized outcome reaches the message.
    let error = AnalysisPublicationError::ResourceLimit {
        message: message.clone(),
    }
    .context("original mesh encoding phase: mesh_wire_upper=7274630");
    let shown = error.to_string();
    assert!(shown.starts_with(&message), "{shown}");
    assert!(shown.ends_with("(original mesh encoding phase: mesh_wire_upper=7274630)"));
    assert!(
        !shown.contains('{') && !shown.contains("\"outcome\""),
        "{shown}"
    );
}
