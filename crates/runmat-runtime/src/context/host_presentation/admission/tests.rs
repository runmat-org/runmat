use super::*;
use runmat_execution::{PlacementRevision, ProviderResourceSnapshot};

#[test]
fn selected_upload_metadata_quote_tracks_each_owned_shape_and_operation() {
    let output = ProviderRepresentation {
        element_type: runmat_accelerate_api::ProviderElementType::F32,
        storage: runmat_accelerate_api::ProviderStorage::DenseReal,
        layout: runmat_accelerate_api::ProviderLayout::ColumnMajorContiguous,
        shape: vec![2, 3],
        residency: ProviderResidency::Device,
    };
    let operation = ProviderOperationIdentity::new("transfer.upload");
    let one =
        selected_upload_metadata_bytes(std::slice::from_ref(&output), &operation, "randi").unwrap();
    let two =
        selected_upload_metadata_bytes(&[output.clone(), output], &operation, "randi").unwrap();
    assert_eq!(
        two - one,
        std::mem::size_of::<ProviderRepresentation>() as u64
            + 2 * std::mem::size_of::<u64>() as u64
    );
    assert_eq!(
        selected_upload_metadata_bytes(
            std::slice::from_ref(&ProviderRepresentation {
                element_type: runmat_accelerate_api::ProviderElementType::F32,
                storage: runmat_accelerate_api::ProviderStorage::DenseReal,
                layout: runmat_accelerate_api::ProviderLayout::ColumnMajorContiguous,
                shape: vec![2, 3],
                residency: ProviderResidency::Device,
            }),
            &operation,
            "randi2",
        )
        .unwrap()
            - one,
        1
    );
    let longer = ProviderOperationIdentity::new("transfer.upload.extra");
    assert_eq!(
        selected_upload_metadata_bytes(
            &[ProviderRepresentation {
                element_type: runmat_accelerate_api::ProviderElementType::F32,
                storage: runmat_accelerate_api::ProviderStorage::DenseReal,
                layout: runmat_accelerate_api::ProviderLayout::ColumnMajorContiguous,
                shape: vec![2, 3],
                residency: ProviderResidency::Device,
            }],
            &longer,
            "randi"
        )
        .unwrap()
            - one,
        6
    );
}

fn budget_request(logical: u64, physical: u64, capacity: u64) -> WholeWorkPlacementRequest {
    let digest = Digest::sha256(b"physical-host-upload-budget");
    let mut candidate = upload_candidate(
        17,
        logical,
        UploadStageResources {
            physical_retained_bytes: Some(physical),
            host_workspace: ResourceByteBound::Known(0),
            provider_scratch: ResourceByteBound::Known(0),
            queue_slots: 1,
        },
        ResourceByteBound::Known(0),
        digest,
        None,
    );
    // This synthetic route isolates retained-provider and host-workspace
    // budgets. Unknown provider scratch would independently prevent its
    // acceptance under the finite provider capacity in this fixture.
    if let CandidateOutputPresentation::ProviderUpload {
        provider_scratch_bytes,
        ..
    } = &mut candidate.presentation
    {
        *provider_scratch_bytes = ResourceByteBound::Known(0);
    }
    WholeWorkPlacementRequest {
        signature: PlacementSignature {
            region: None,
            operation: "imhist".into(),
            runtime_facts: digest,
            provider_provenance: None,
            graph_output_authority: None,
            revision: PlacementRevision {
                program: None,
                catalog: digest,
                compiler: digest,
                authority: runmat_execution::PlacementExecutionAuthorityRevision::Provider(digest),
                policy: digest,
            },
        },
        constraint: WholeWorkConstraint::AnyLegal,
        candidates: vec![candidate],
        resources: PlacementResourceSnapshot {
            cpu_millicores_available: 1_000,
            memory_available_bytes: Some(1_024),
            cancellation_requested: false,
            providers: vec![ProviderResourceSnapshot {
                device_id: 17,
                capacity_bytes: Some(capacity),
                scratch_available_bytes: None,
                ..Default::default()
            }],
            epoch: 0,
        },
        deterministic: true,
        require_transactional_results: false,
    }
}

#[test]
fn shared_service_budgets_physical_retention_not_logical_payload() {
    let session = runmat_accelerate::placement::PlacementSession::new(
        runmat_accelerate::placement::PlacementSessionConfig::default(),
    );
    assert!(session
        .select_whole_work(budget_request(32, 40, 39))
        .is_err());
    assert!(session
        .select_whole_work(budget_request(32, 40, 40))
        .is_ok());
    // Zero logical elements can still require a nonzero provider buffer.
    assert!(session.select_whole_work(budget_request(0, 8, 7)).is_err());
    assert!(session.select_whole_work(budget_request(0, 8, 8)).is_ok());
}

#[test]
fn provider_retention_is_not_logical_transfer_or_unknown_as_zero() {
    let logical_transfer = 32;
    assert_eq!(
        physical_retained_bound(Some(40)),
        ResourceByteBound::Known(40)
    );
    assert_ne!(
        physical_retained_bound(Some(40)),
        ResourceByteBound::Known(logical_transfer)
    );
    // A zero-element logical payload can still retain a nonzero provider
    // allocation floor; never replace the reported physical result by 0.
    assert_eq!(
        physical_retained_bound(Some(8)),
        ResourceByteBound::Known(8)
    );
    assert_eq!(
        physical_retained_bound(None),
        ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoRetainedValueBound)
    );
}

#[test]
fn readback_workspace_combination_preserves_unknown_and_checks_sum() {
    assert_eq!(
        combine_host_scratch(ResourceByteBound::Known(56), ResourceByteBound::Known(16)).unwrap(),
        ResourceByteBound::Known(72)
    );
    assert_eq!(
        combine_host_scratch(
            ResourceByteBound::Known(56),
            ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoScratchBound),
        )
        .unwrap(),
        ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoScratchBound)
    );
    assert_eq!(
        combine_host_scratch(
            ResourceByteBound::Known(u64::MAX),
            ResourceByteBound::Known(1)
        )
        .unwrap_err()
        .identifier(),
        Some("RunMat:Placement:ResourceOverflow")
    );
}

#[test]
fn unknown_readback_workspace_cannot_pass_finite_host_quota() {
    let session = runmat_accelerate::placement::PlacementSession::new(
        runmat_accelerate::placement::PlacementSessionConfig::default(),
    );
    let mut request = budget_request(32, 40, 1_024);
    request.candidates[0].resources.scratch_bytes =
        ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoScratchBound);
    assert!(session.select_whole_work(request.clone()).is_err());
    request.resources.memory_available_bytes = None;
    assert!(session.select_whole_work(request).is_ok());
}

#[test]
fn known_readback_workspace_is_charged_to_host_peak() {
    let session = runmat_accelerate::placement::PlacementSession::new(
        runmat_accelerate::placement::PlacementSessionConfig::default(),
    );
    let mut request = budget_request(32, 40, 1_024);
    // The exact stage reports 32 bytes of native upload conversion beyond
    // its 32-byte retained result; neither amount is inferred from the
    // logical transfer count by shared admission.
    request.candidates[0].resources.scratch_bytes = ResourceByteBound::Known(72);
    if let CandidateOutputPresentation::ProviderUpload {
        host_scratch_bytes, ..
    } = &mut request.candidates[0].presentation
    {
        *host_scratch_bytes = ResourceByteBound::Known(32);
    }
    request.resources.memory_available_bytes = Some(135);
    assert!(session.select_whole_work(request.clone()).is_err());
    request.resources.memory_available_bytes = Some(136);
    assert!(session.select_whole_work(request).is_ok());
}
