use super::*;

#[test]
fn source_workspace_is_additive_persistent_residency_not_hidden_in_phase_maximum() {
    let base = build_resource_plan(input(PartitioningPolicy::Monolithic)).unwrap();
    let mut source_input = input(PartitioningPolicy::Monolithic);
    operator_mut(&mut source_input).excitation_workspace = ExcitationWorkspaceResourceFacts {
        active_field_count: Count(1),
        scalar_component_count: Count(12),
        resident_bytes_upper_bound: ByteCount(8192),
    };
    let source = build_resource_plan(source_input).unwrap();
    assert_eq!(
        source.request.memory_bytes,
        base.request.memory_bytes + 8192
    );
    assert_eq!(source.request.scratch_bytes, base.request.scratch_bytes);
    assert_eq!(
        source.breakdown.phase_resident_bytes,
        base.breakdown.phase_resident_bytes
    );
    let cold = FeaResourcePlan::from_canonical_bytes(&source.canonical_bytes().unwrap()).unwrap();
    assert_eq!(cold, source);
}

#[test]
fn source_workspace_contract_requires_inventory_and_frozen_payload_and_rejects_unknown_fields() {
    for facts in [
        ExcitationWorkspaceResourceFacts {
            active_field_count: Count(0),
            scalar_component_count: Count(3),
            resident_bytes_upper_bound: ByteCount(48),
        },
        ExcitationWorkspaceResourceFacts {
            active_field_count: Count(1),
            scalar_component_count: Count(3),
            resident_bytes_upper_bound: ByteCount(47),
        },
        ExcitationWorkspaceResourceFacts {
            active_field_count: Count(2),
            scalar_component_count: Count(1),
            resident_bytes_upper_bound: ByteCount(48),
        },
        ExcitationWorkspaceResourceFacts {
            active_field_count: Count(1),
            scalar_component_count: Count(u64::MAX),
            resident_bytes_upper_bound: ByteCount(u64::MAX),
        },
    ] {
        let mut planned = input(PartitioningPolicy::Monolithic);
        operator_mut(&mut planned).excitation_workspace = facts;
        assert!(build_resource_plan(planned).is_err());
    }
    let mut encoded = serde_json::to_value(ExcitationWorkspaceResourceFacts::default()).unwrap();
    encoded["unclaimed_workspace"] = serde_json::json!(1);
    assert!(serde_json::from_value::<ExcitationWorkspaceResourceFacts>(encoded).is_err());
}
