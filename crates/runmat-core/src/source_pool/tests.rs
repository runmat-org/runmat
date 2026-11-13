use super::*;
use std::rc::Rc;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn equal_diagnostic_source_text_from_distinct_owners_issues_distinct_refs() {
    let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ));
    let entry = |owner: &str| crate::SourceMapEntry {
        source_id: runmat_types::ProgramSourceId(0),
        owner_identity: owner.into(),
        relative_path: "shared.m".into(),
        content_revision: runmat_execution::Digest::sha256(b"answer = 1;"),
        display_name: "shared.m".into(),
        full_path: None,
        text: "answer = 1;".into(),
    };
    let admitted = |entry: &crate::SourceMapEntry| {
        let map = runmat_execution::ExecutableSourceMapV2::new(vec![
            runmat_execution::ExecutableSourceRecordV2 {
                source_id: entry.source_id,
                owner_identity: entry.owner_identity.clone(),
                relative_path: entry.relative_path.clone(),
                content_revision: entry.content_revision,
                display_name: entry.display_name.clone(),
                text: entry.text.clone(),
            },
        ])
        .expect("valid exact portable source");
        runmat_runtime::source_context::ValidatedPortableSourceMap::new(std::sync::Arc::new(map))
            .expect("admitted map")
            .record(entry.source_id)
            .expect("admitted record")
    };
    let mut pool = SourcePool::default();
    let first_entry = entry("owner-a");
    let second_entry = entry("owner-b");
    let first = pool
        .intern_portable_entry(&first_entry, admitted(&first_entry))
        .expect("first source");
    let second = pool
        .intern_portable_entry(&second_entry, admitted(&second_entry))
        .expect("second source");
    assert_ne!(first, second);
    let first_ref = pool
        .issued_source_ref(first, &runtime)
        .expect("first owner");
    let second_ref = pool
        .issued_source_ref(second, &runtime)
        .expect("second owner");
    assert_ne!(first_ref, second_ref);
    assert_eq!(
        runmat_runtime::source_context::portable_source_identity_in_context(&runtime, first_ref)
            .unwrap()
            .owner_identity,
        "owner-a"
    );
    assert_eq!(
        runmat_runtime::source_context::portable_source_identity_in_context(&runtime, second_ref)
            .unwrap()
            .owner_identity,
        "owner-b"
    );
    assert_eq!(
        pool.intern_portable_entry(&first_entry, admitted(&first_entry))
            .expect("reuse first exact owner"),
        first
    );
}
