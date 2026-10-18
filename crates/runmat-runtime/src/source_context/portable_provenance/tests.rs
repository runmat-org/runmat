use super::*;
use crate::context::RuntimeContext;
use crate::execution::RuntimeExecutionService;
use crate::source_context::{lease_source_context_in_context, portable_source_identity_in_context};
use runmat_types::ProgramSourceId;
use std::{rc::Rc, sync::Arc};

fn runtime() -> RuntimeContext {
    RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
}

fn record(owner: &str, text: &str) -> ExecutableSourceRecordV2 {
    ExecutableSourceRecordV2 {
        source_id: ProgramSourceId(0),
        owner_identity: owner.into(),
        relative_path: "shared.m".into(),
        content_revision: Digest::sha256(text.as_bytes()),
        display_name: "shared.m".into(),
        text: text.into(),
    }
}

fn admitted(owner: &str, text: &str) -> ValidatedPortableSourceMap {
    ValidatedPortableSourceMap::new(Arc::new(
        runmat_execution::ExecutableSourceMapV2::new(vec![record(owner, text)])
            .expect("validated map"),
    ))
    .expect("admitted map")
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn portable_owner_is_exact_to_issuer_content_and_lease_lifetime() {
    let selected = runtime();
    let foreign = runtime();
    let first_map = admitted("owner-a", "answer = 1;");
    let second_map = admitted("owner-b", "answer = 1;");
    assert!(
        first_map.record(ProgramSourceId(1)).is_none(),
        "absent ordinal"
    );
    let first = lease_portable_source_context_in_context(
        &selected,
        &first_map.record(ProgramSourceId(0)).expect("first ordinal"),
        None,
    )
    .expect("first issued source");
    let second = lease_portable_source_context_in_context(
        &selected,
        &second_map
            .record(ProgramSourceId(0))
            .expect("second ordinal"),
        None,
    )
    .expect("second issued source");
    let first_ref = first.source_ref();
    let second_ref = second.source_ref();
    let mut invalid_revision = record("owner-a", "answer = 1;");
    invalid_revision.content_revision = Digest::sha256(b"answer = 2;");
    assert!(
        ValidatedPortableSourceMap::new(Arc::new(runmat_execution::ExecutableSourceMapV2 {
            entries: vec![invalid_revision],
        }))
        .is_err()
    );
    assert_eq!(
        portable_source_identity_in_context(&selected, first_ref)
            .expect("first owner")
            .owner_identity,
        "owner-a"
    );
    assert_eq!(
        portable_source_identity_in_context(&selected, second_ref)
            .expect("second owner")
            .owner_identity,
        "owner-b"
    );
    assert_eq!(
        portable_source_identity_in_context(&foreign, first_ref),
        None
    );
    drop(first);
    assert_eq!(
        portable_source_identity_in_context(&selected, first_ref),
        None
    );
    assert!(portable_source_identity_in_context(&selected, second_ref).is_some());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn admitted_program_binding_and_fork_retain_owner_but_callback_text_does_not() {
    let session = runtime();
    let callback = lease_source_context_in_context(&session, "shared.m", None, "answer = 1;")
        .expect("callback source");
    assert_eq!(
        portable_source_identity_in_context(&session, callback.source_ref()),
        None,
        "text-only callback admission cannot claim an executable owner"
    );
    let map =
        runmat_execution::ExecutableSourceMapV2::new(vec![record("imported-owner", "answer = 1;")])
            .expect("validated portable source map");
    let bound = crate::source_context::bind_portable_source_map_v2(session.clone(), &map)
        .expect("admitted imported source");
    let separately_issued = lease_portable_source_context_in_context(
        &session,
        &admitted("other-owner", "answer = 1;")
            .record(ProgramSourceId(0))
            .expect("other ordinal"),
        None,
    )
    .expect("separate portable source");
    assert_eq!(
        portable_source_identity_in_context(&session, callback.source_ref()),
        None,
        "issuing equal text with an admitted owner cannot relabel the callback"
    );
    assert!(
        portable_source_identity_in_context(&session, separately_issued.source_ref()).is_some()
    );
    let source = bound
        .program_source_ref(ProgramSourceId(0))
        .expect("bound ordinal");
    assert_eq!(
        portable_source_identity_in_context(&bound, source)
            .expect("bound owner")
            .owner_identity,
        "imported-owner"
    );
    let fork = bound
        .fork_parallel_lab(crate::context::RuntimeServicePorts::default())
        .expect("idle session fork");
    assert_eq!(
        portable_source_identity_in_context(&fork, source)
            .expect("inherited owner")
            .owner_identity,
        "imported-owner"
    );
    drop(bound);
    assert_eq!(portable_source_identity_in_context(&session, source), None);
    assert!(portable_source_identity_in_context(&fork, source).is_some());
}
