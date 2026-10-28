use super::*;
use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use std::cell::Cell;
use std::num::NonZeroU64;
use std::rc::Rc;

fn cancelling_budget(turn_limit: usize) -> (Option<CpuCooperationBudget>, Rc<Cell<usize>>) {
    let cancellation = Rc::new(CancellationSource::new());
    let turns = Rc::new(Cell::new(0));
    let counted = turns.clone();
    let cancel = cancellation.clone();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        cancellation.token(),
        move || {
            counted.set(counted.get() + 1);
            if counted.get() == turn_limit {
                cancel.cancel(CancellationReason::User);
            }
            Box::pin(async { Ok(()) })
        },
    );
    (Some(budget), turns)
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn large_single_chunk_checks_each_collected_element() {
    let source = DataArrayValues::F64((0..256).map(f64::from).collect());
    let (mut budget, turns) = cancelling_budget(7);
    let error = futures::executor::block_on(collect_chunk_component_values(
        &source,
        "double",
        &[256],
        &[0],
        &[256],
        &mut budget,
    ))
    .expect_err("cancelled element collection cannot publish a chunk");
    assert_eq!(error.identifier.as_deref(), Some("RunMat:Cancelled"));
    assert_eq!(turns.get(), 7);
    assert_eq!(source.len(), 256);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn cancelled_chunk_write_never_publishes_index() {
    let provider = runmat_filesystem::MemoryFsProvider::with_current_dir("/data");
    let payload = DataArrayPayload {
        dtype: "double".into(),
        shape: vec![256],
        values: DataArrayValues::F64(vec![3.0; 256]),
        imaginary_values: None,
    };
    let (mut budget, turns) = cancelling_budget(7);
    let error = futures::executor::block_on(write_array_payload_on(
        &provider,
        Path::new("/data/cancelled.data"),
        "samples",
        &payload,
        &[256],
        &mut budget,
    ))
    .expect_err("cancelled chunk must not publish index");
    assert_eq!(error.identifier.as_deref(), Some("RunMat:Cancelled"));
    assert_eq!(turns.get(), 7);
    let revisions = futures::executor::block_on(runmat_filesystem::FsProvider::read_dir(
        &provider,
        Path::new("/data/cancelled.data/arrays/samples/revisions"),
    ))
    .unwrap();
    for revision in revisions {
        let entries = futures::executor::block_on(runmat_filesystem::FsProvider::read_dir(
            &provider,
            revision.path(),
        ))
        .unwrap();
        for entry in entries {
            assert_ne!(entry.file_name(), "index.json");
            if entry.file_name() == "chunks" {
                let chunks = futures::executor::block_on(runmat_filesystem::FsProvider::read_dir(
                    &provider,
                    entry.path(),
                ))
                .unwrap();
                assert!(chunks.iter().all(|chunk| chunk.file_name() != "index.json"));
            }
        }
    }
}
