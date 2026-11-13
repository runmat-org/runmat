use crate::execution::{FluidTransientBlockWorkload, FluidTransientCheckpoint};
#[test]
fn original_full_history_cold_split_matches_unsplit_and_rejects_semantic_corruption() {
    let (workload, mesh) = super::transient_workload::accelerating_transient();
    let initial = workload.initial_checkpoint(&mesh).unwrap();
    assert_eq!(initial.next_interval(), 0);
    initial.recover(&workload, &mesh).unwrap();
    let full = FluidTransientBlockWorkload {
        workload: workload.clone(),
        predecessor: initial.clone(),
        end_interval: 3,
    };
    let prepared = full.prepare(&mesh).unwrap();
    let bound = prepared.result_encoding_profile().unwrap();
    let unsplit = prepared.execute().unwrap();
    assert!(unsplit.canonical_bytes().unwrap().len() as u64 <= bound.encoded_bytes_upper_bound);
    drop(prepared);
    let first = FluidTransientBlockWorkload {
        workload: workload.clone(),
        predecessor: initial,
        end_interval: 1,
    }
    .execute(&mesh)
    .unwrap();
    let bytes = first.canonical_bytes().unwrap();
    drop(first);
    let cold = FluidTransientCheckpoint::from_canonical_bytes(&bytes).unwrap();
    let request = FluidTransientBlockWorkload {
        workload: workload.clone(),
        predecessor: cold,
        end_interval: 3,
    };
    let bytes = request.canonical_bytes().unwrap();
    drop(request);
    let request = FluidTransientBlockWorkload::from_canonical_bytes(&bytes).unwrap();
    let split = request.execute(&mesh).unwrap();
    split.recover(&workload, &mesh).unwrap();
    request.recover_output(&split, &mesh).unwrap();
    // Immutable prefixes share original arrays in-process; cold decoding remains separate.
    assert!(std::ptr::eq(
        request.predecessor.retained_results().next().unwrap(),
        split.retained_results().next().unwrap()
    ));
    assert!(request.recover_output(&unsplit, &mesh).is_err());
    {
        let prepared = request.prepare(&mesh).unwrap();
        prepared.recover_output(&split).unwrap();
        assert!(prepared.recover_output(&unsplit).is_err());
    }
    assert_eq!(split.next_interval(), 3);
    assert_eq!(split.next_output_ordinal(), 3);
    assert_eq!(
        split.intervals().collect::<Vec<_>>(),
        unsplit.intervals().collect::<Vec<_>>()
    );
    assert_eq!(
        split.retained_results().last(),
        unsplit.retained_results().last()
    );
    assert!(split.charged_operator_work() > request.predecessor.charged_operator_work());
    for substitution in 0..5 {
        let mut data = serde_json::to_value(&split).unwrap();
        match substitution {
            0 => data["next_output_ordinal"] = serde_json::json!(2),
            1 => data["charged_operator_work"] = serde_json::json!(0),
            2 => data["blocks"][0]["charged_operator_work"] = serde_json::json!(0),
            3 => {
                data["blocks"][0]["original"]["retained"][0][1]["face_volume_flux_m3_per_s"][0] =
                    serde_json::json!(0.1)
            }
            _ => data["blocks"].as_array_mut().unwrap().swap(0, 1),
        }
        let bad: FluidTransientCheckpoint = serde_json::from_value(data).unwrap();
        assert!(
            bad.recover(&workload, &mesh).is_err(),
            "substitution {substitution}"
        );
    }
    let invalid = FluidTransientBlockWorkload {
        workload: workload.clone(),
        predecessor: split.clone(),
        end_interval: 3,
    };
    assert!(invalid.prepare(&mesh).is_err());
    let invalid = FluidTransientBlockWorkload {
        workload,
        predecessor: split,
        end_interval: 4,
    };
    assert!(invalid.prepare(&mesh).is_err());
}
#[test]
fn cumulative_work_rejects_a_later_locally_admissible_block_and_cancellation() {
    let (mut workload, mesh) = super::transient_workload::accelerating_transient();
    let initial = workload.initial_checkpoint(&mesh).unwrap();
    let probe = FluidTransientBlockWorkload {
        workload: workload.clone(),
        predecessor: initial,
        end_interval: 1,
    };
    let charge = probe
        .prepare(&mesh)
        .unwrap()
        .reservation()
        .unwrap()
        .operator_work_upper_bound;
    workload.policy.maximum_operator_work = Some(charge.checked_mul(3).unwrap());
    let initial = workload.initial_checkpoint(&mesh).unwrap();
    let first = FluidTransientBlockWorkload {
        workload: workload.clone(),
        predecessor: initial,
        end_interval: 1,
    }
    .execute(&mesh)
    .unwrap();
    let second = FluidTransientBlockWorkload {
        workload: workload.clone(),
        predecessor: first,
        end_interval: 2,
    }
    .execute(&mesh)
    .unwrap();
    let third = FluidTransientBlockWorkload {
        workload,
        predecessor: second,
        end_interval: 3,
    };
    assert!(third.prepare(&mesh).is_err());
    let (valid_workload, mesh) = super::transient_workload::accelerating_transient();
    let valid = FluidTransientBlockWorkload {
        predecessor: valid_workload.initial_checkpoint(&mesh).unwrap(),
        workload: valid_workload,
        end_interval: 1,
    };
    valid.prepare(&mesh).unwrap();
    let context = crate::FeaProgressContext::default();
    context.cancellation.cancel();
    let _guard = crate::replace_fea_progress_context(context);
    let error = valid.prepare(&mesh).err().unwrap().to_string();
    assert!(error.contains("cancelled"), "{error}");
}
