use super::*;
use runmat_execution::{Digest, ExecutableSourceRecordV2};

fn new_runtime() -> crate::context::RuntimeContext {
    crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ))
}

fn map() -> ExecutableSourceMapV2 {
    let text = "value = 1;";
    ExecutableSourceMapV2::new(vec![ExecutableSourceRecordV2 {
        source_id: ProgramSourceId(0),
        owner_identity: "fixture".into(),
        relative_path: "original.m".into(),
        content_revision: Digest::sha256(text.as_bytes()),
        display_name: "original.m".into(),
        text: text.into(),
    }])
    .expect("portable source map")
}

#[test]
fn exact_issued_owner_retains_a_validated_durable_alias_without_rebinding() {
    let map = map();
    let runtime = super::super::bind_portable_source_map_v2(new_runtime(), &map)
        .expect("issued selected source map");
    let issued = runtime
        .program_source_ref(ProgramSourceId(0))
        .expect("issued ordinal");
    let selected = SelectedProgramSourceOwner::capture(&runtime, &map).expect("exact owner");
    let cloned = selected.clone();
    assert!(std::rc::Rc::ptr_eq(&selected.facts, &cloned.facts));
    assert!(std::rc::Rc::ptr_eq(
        &selected.durable_ordinals,
        &cloned.durable_ordinals
    ));
    assert!(std::rc::Rc::ptr_eq(
        &selected.durable_aliases,
        &cloned.durable_aliases
    ));
    let durable =
        super::super::lease_source_context_in_context(&runtime, "original.m", None, "value = 1;")
            .expect("durable source pool lease");
    let alias = selected
        .with_durable_aliases(&runtime, [(issued, durable.source_ref())])
        .expect("published body pairs exact issued and durable sources");
    assert!(alias.same_issuance(&selected));
    assert_eq!(selected.source_ordinal_for(durable.source_ref()), None);
    assert_eq!(alias.source_ordinal_for(issued), Some(ProgramSourceId(0)));
    assert_eq!(
        alias.source_ordinal_for(durable.source_ref()),
        Some(ProgramSourceId(0))
    );
    assert_eq!(alias.durable_aliases().count(), 1);
    assert_eq!(alias.selected_sources().count(), 1);
    assert_eq!(
        alias
            .issued_source_info_for(ProgramSourceId(0))
            .expect("validated ordinal")
            .text
            .as_ref(),
        "value = 1;"
    );
    assert!(alias.issued_source_info_for(ProgramSourceId(1)).is_none());
}

#[test]
fn foreign_or_unpaired_source_cannot_become_a_durable_alias() {
    let map = map();
    let runtime = super::super::bind_portable_source_map_v2(new_runtime(), &map)
        .expect("issued selected source map");
    let issued = runtime
        .program_source_ref(ProgramSourceId(0))
        .expect("issued ordinal");
    let selected = SelectedProgramSourceOwner::capture(&runtime, &map).expect("exact owner");
    let foreign = new_runtime();
    assert!(selected.is_issued_in(&runtime));
    assert!(!selected.is_issued_in(&foreign));
    let foreign_source =
        super::super::lease_source_context_in_context(&foreign, "original.m", None, "value = 1;")
            .expect("foreign lease");
    assert!(matches!(
        selected.with_durable_aliases(&foreign, [(issued, foreign_source.source_ref())]),
        Err(SelectedProgramSourceError::ForeignSession)
    ));
    assert!(selected
        .with_durable_aliases(&runtime, [(issued, foreign_source.source_ref())])
        .is_err());
    let local =
        super::super::lease_source_context_in_context(&runtime, "original.m", None, "value = 1;")
            .expect("local lease");
    assert!(selected
        .with_durable_aliases(&runtime, [(local.source_ref(), issued)])
        .is_err());
}

#[test]
fn selected_source_owner_cannot_authorize_a_retired_session() {
    let map = map();
    let owner = {
        let runtime = super::super::bind_portable_source_map_v2(new_runtime(), &map)
            .expect("issued selected source map");
        SelectedProgramSourceOwner::capture(&runtime, &map).expect("exact owner")
    };
    assert!(!owner.is_live());
    assert!(!owner.is_issued_in(&new_runtime()));
}

#[test]
fn selected_owner_rebinds_exact_issuance_only_in_its_live_session() {
    let source_map = map();
    let issued = super::super::bind_portable_source_map_v2(new_runtime(), &source_map)
        .expect("issued selected source map");
    let owner = SelectedProgramSourceOwner::capture(&issued, &source_map).expect("exact owner");
    let expected = issued
        .program_source_ref(ProgramSourceId(0))
        .expect("issued source");
    let rebound = owner
        .bind_runtime(issued.clone())
        .expect("same live session reuses selected binding");
    assert_eq!(
        rebound.program_source_ref(ProgramSourceId(0)),
        Some(expected)
    );
    assert_eq!(rebound.portable_program_source_map(), Some(&source_map));
    assert_eq!(
        rebound.call_frame_source_owner_for_program_source(ProgramSourceId(0)),
        issued.call_frame_source_owner_for_program_source(ProgramSourceId(0))
    );
    assert!(matches!(
        owner.bind_runtime(new_runtime()),
        Err(SelectedProgramSourceError::ForeignSession)
    ));
}

#[test]
fn call_frame_basis_is_shared_by_exact_issuance_not_equal_source_content() {
    let source_map = map();
    let first = super::super::bind_portable_source_map_v2(new_runtime(), &source_map)
        .expect("first admitted source issuance");
    let second = super::super::bind_portable_source_map_v2(new_runtime(), &source_map)
        .expect("same bytes, distinct issuance");
    let first_owner =
        SelectedProgramSourceOwner::capture(&first, &source_map).expect("first exact owner");
    let second_owner =
        SelectedProgramSourceOwner::capture(&second, &source_map).expect("second exact owner");
    let first_basis = first_owner
        .call_frame_source_owner_for(ProgramSourceId(0))
        .expect("admitted ordinal");
    assert_eq!(
        first_basis,
        first
            .call_frame_source_owner_for_program_source(ProgramSourceId(0))
            .expect("same table basis")
    );
    assert_ne!(
        first_basis,
        second_owner
            .call_frame_source_owner_for(ProgramSourceId(0))
            .expect("second admitted ordinal"),
        "matching bytes and ordinal cannot collapse separately issued tables"
    );
    assert!(first_owner
        .call_frame_source_owner_for(ProgramSourceId(1))
        .is_none());
}
