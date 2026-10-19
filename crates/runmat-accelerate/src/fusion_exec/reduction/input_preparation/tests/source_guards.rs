use super::*;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn later_foreign_or_expired_resident_prevents_earlier_host_upload_queries() {
    for expired in [false, true] {
        let owner = Owner::new(8);
        let foreign = Owner::new(8);
        let _selected = ThreadProviderGuard::set(Some(owner));
        let host = Value::Num(2.0);
        let resident = allocation(
            if expired { owner } else { foreign },
            vec![1, 1],
            NumericElementType::F64,
        );
        if expired {
            resident.probe.mark_unavailable();
        }
        let selected = plan(
            owner,
            vec![
                host_entry(owner),
                resident_entry(&resident.handle, ProviderElementType::F64),
            ],
            vec![
                InputAction::Upload {
                    source: &host,
                    target: ProviderPrecision::F64,
                },
                InputAction::Borrow(&resident.handle),
            ],
            vec![],
            32,
        );
        let error = block_on(selected.prepare()).unwrap_err();
        assert!(!error.is_after_effect());
        assert_eq!(owner.evidence_count(), 0);
        assert_eq!(owner.upload_count(), 0);
        assert_eq!(owner.proposal_queries.load(Ordering::SeqCst), 0);
        assert_eq!(resident.probe.retirements(), 0);
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn over_cap_resident_and_upload_proposal_stop_before_effects() {
    let owner = Owner::new(16);
    let _selected = ThreadProviderGuard::set(Some(owner));
    let host = Value::Num(3.0);
    let large = allocation(owner, vec![2, 1], NumericElementType::F64);
    let resident_plan = plan(
        owner,
        vec![
            host_entry(owner),
            resident_entry(&large.handle, ProviderElementType::F64),
        ],
        vec![
            InputAction::Upload {
                source: &host,
                target: ProviderPrecision::F64,
            },
            InputAction::Borrow(&large.handle),
        ],
        vec![],
        8,
    );
    assert!(!block_on(resident_plan.prepare())
        .unwrap_err()
        .is_after_effect());
    assert_eq!(owner.evidence_count(), 0);
    assert_eq!(owner.upload_count(), 0);

    let upload_plan = plan(
        owner,
        vec![host_entry(owner)],
        vec![InputAction::Upload {
            source: &host,
            target: ProviderPrecision::F64,
        }],
        vec![],
        8,
    );
    assert!(!block_on(upload_plan.prepare())
        .unwrap_err()
        .is_after_effect());
    assert_eq!(owner.evidence_count(), 0);
    assert_eq!(owner.upload_count(), 0);
    assert_eq!(large.probe.retirements(), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn over_cap_cast_stops_before_requery_or_execution() {
    let owner = Owner::new(8);
    let _selected = ThreadProviderGuard::set(Some(owner));
    let source = allocation(owner, vec![1, 1], NumericElementType::F32);
    let source_rep = rep(
        ProviderElementType::F32,
        ProviderResidency::Device,
        vec![1, 1],
    );
    let cast = ProviderRealElementwiseRecipe::plan(
        owner,
        ProviderRealElementwiseRequest {
            operation: ProviderRealElementwiseOperation::FloatingCast {
                target: ProviderPrecision::F64,
            },
            source: source_rep.clone(),
            source_buffer_bytes: source.handle.physical().requested_bytes(),
        },
    )
    .unwrap()
    .unwrap();
    let proposal_count = owner.proposal_queries.load(Ordering::SeqCst);
    let entry = ProviderReductionInputPreparation {
        source: source_rep,
        prepared: cast.output().clone(),
        source_physical: Some(*source.handle.physical()),
        stage: ProviderReductionInputStage::ResidentCast {
            device_id: owner.device_id(),
            recipe_identity: cast.identity().unwrap(),
            physical: cast.physical().clone(),
        },
    };
    let selected = plan(
        owner,
        vec![entry],
        vec![InputAction::Cast {
            source: &source.handle,
            recipe_index: 0,
        }],
        vec![cast],
        8,
    );
    assert!(!block_on(selected.prepare()).unwrap_err().is_after_effect());
    assert_eq!(
        owner.proposal_queries.load(Ordering::SeqCst),
        proposal_count
    );
    assert_eq!(owner.upload_count(), 0);
    assert_eq!(source.probe.retirements(), 0);
}
