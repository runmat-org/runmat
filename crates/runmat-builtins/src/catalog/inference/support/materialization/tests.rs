use super::gathered_value_fact;
use runmat_types::*;
use std::collections::BTreeMap;

fn resident(storage: StorageFact) -> ValueFact {
    let mut value = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(3)]),
        storage,
    );
    value.residency = ResidencyFact::Device {
        provider: Some("owner".into()),
    };
    value.alias = AliasFact::Identity;
    value
}

#[test]
fn downloads_preserve_class_shape_and_sparse_storage() {
    for storage in [
        StorageFact::Dense,
        StorageFact::Sparse,
        StorageFact::Unknown,
    ] {
        let input = resident(storage);
        let output = gathered_value_fact(&input);
        assert_eq!(output.kind, input.kind);
        assert_eq!(output.shape, input.shape);
        assert_eq!(output.storage, storage);
        assert_eq!(output.residency, ResidencyFact::Host);
        assert_eq!(output.alias, AliasFact::Unique);
        assert_eq!(output.view, ViewFact::Materialized);
    }
    let mut input = resident(StorageFact::Sparse);
    input.kind = ValueKindFact::Logical;
    assert_eq!(gathered_value_fact(&input).kind, ValueKindFact::Logical);
    assert_eq!(gathered_value_fact(&input).storage, StorageFact::Sparse);
}

#[test]
fn host_values_and_unknown_numeric_storage_are_not_reclassified() {
    let mut host = resident(StorageFact::Sparse);
    host.residency = ResidencyFact::Host;
    host.alias = AliasFact::Shared;
    host.view = ViewFact::ReadOnlyView;
    assert_eq!(gathered_value_fact(&host), host);
    let unknown = ValueFact::unknown(DynamicReason::ExternalData);
    assert_eq!(gathered_value_fact(&unknown), unknown);
    let mut uncertain = resident(StorageFact::Unknown);
    uncertain.residency = ResidencyFact::Unknown;
    assert_eq!(gathered_value_fact(&uncertain), uncertain);
    let mut unknown_device = unknown.clone();
    unknown_device.residency = ResidencyFact::Device { provider: None };
    let output = gathered_value_fact(&unknown_device);
    assert_eq!(output.kind, ValueKindFact::Unknown);
    assert_eq!(output.residency, ResidencyFact::Unknown);
    assert_eq!(output.storage, StorageFact::Unknown);
    assert_eq!(output.certainty, unknown.certainty);
}

#[test]
fn nested_cell_struct_summary_and_position_facts_are_gathered() {
    let device = resident(StorageFact::Sparse);
    let fields = BTreeMap::from([("payload".into(), device.clone())]);
    let structure = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, true)));
    let input = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(structure.clone()),
            elements: vec![structure],
            elements_complete: true,
        }),
        ShapeFact::from(vec![Some(1), Some(1)]),
        StorageFact::Dense,
    );
    let output = gathered_value_fact(&input);
    assert_eq!(output.shape, input.shape);
    let ValueKindFact::Cell(cell) = output.kind else {
        panic!("cell")
    };
    assert!(cell.elements_complete);
    for nested in [cell.element.as_ref(), &cell.elements[0]] {
        let ValueKindFact::Struct(structure) = &nested.kind else {
            panic!("struct")
        };
        for field in [
            &structure.fields["payload"],
            &structure.elements[0]["payload"],
        ] {
            assert_eq!(field.storage, StorageFact::Sparse);
            assert_eq!(field.kind, device.kind);
            assert_eq!(field.residency, ResidencyFact::Host);
        }
    }
}

#[test]
fn foreign_and_distributed_payloads_are_not_downloaded() {
    let mut foreign = ValueFact::scalar(ValueKindFact::Foreign(ForeignFact {
        family: "opaque-provider".into(),
        type_name: None,
        type_version: None,
        ownership: ForeignOwnershipFact::Borrowed,
        affinity: ForeignAffinityFact::OriginThread,
        lifetime: ForeignLifetimeFact::External,
    }));
    foreign.storage = StorageFact::Opaque;
    foreign.residency = ResidencyFact::Device {
        provider: Some("foreign-owner".into()),
    };
    assert_eq!(gathered_value_fact(&foreign), foreign);
    let function = ProgramFunctionId(1);
    let mut distributed = ValueFact::scalar(ValueKindFact::Distributed(DistributedFact {
        id: DistributedValueId {
            function,
            ordinal: 1,
        },
        owner: DistributedOwner::Client(function),
        scheme: None,
        value: Box::new(resident(StorageFact::Sparse)),
        materializable: true,
    }));
    distributed.storage = StorageFact::Opaque;
    distributed.residency = ResidencyFact::Remote {
        pool: Some("pool".into()),
    };
    assert_eq!(gathered_value_fact(&distributed), distributed);
}

#[test]
fn object_representation_ambiguity_does_not_invent_host_properties() {
    let mut object = ObjectFact {
        class: None,
        runtime_class: None,
        properties: BTreeMap::from([("data".into(), resident(StorageFact::Sparse))]),
        properties_complete: true,
        handle_semantics: Some(false),
    };
    let input = ValueFact::scalar(ValueKindFact::Object(object.clone()));
    let output = gathered_value_fact(&input);
    let ValueKindFact::Object(result) = output.kind else {
        panic!("object")
    };
    assert_eq!(result.properties["data"].residency, ResidencyFact::Unknown);
    assert_eq!(result.properties["data"].storage, StorageFact::Sparse);
    assert_eq!(
        result.properties["data"].kind,
        object.properties["data"].kind
    );
    object.handle_semantics = Some(true);
    let handle = ValueFact::scalar(ValueKindFact::Object(object));
    assert_eq!(gathered_value_fact(&handle), handle);
}

#[test]
fn callable_contracts_are_not_payloads_and_captures_remain_conservative() {
    let device = resident(StorageFact::Dense);
    let callable = CallableFact {
        identity: None,
        capabilities: CapabilitySet::default(),
        parameters: vec![device.clone()],
        parameters_complete: true,
        outputs: vec![device.clone()],
        outputs_complete: true,
        variadic_inputs: false,
        variadic_outputs: false,
        captures: vec![device.clone()],
        captures_complete: true,
    };
    let input = ValueFact::scalar(ValueKindFact::Callable(callable.clone()));
    let ValueKindFact::Callable(output) = gathered_value_fact(&input).kind else {
        panic!("callable")
    };
    assert_eq!(output.parameters, callable.parameters);
    assert_eq!(output.outputs, callable.outputs);
    assert_eq!(output.capabilities, callable.capabilities);
    assert_eq!(output.captures[0].kind, device.kind);
    assert_eq!(output.captures[0].residency, ResidencyFact::Unknown);
    assert!(output.captures_complete);
}

#[test]
fn transient_sequence_has_no_successful_gather_value() {
    let sequence = ValueFact::scalar(ValueKindFact::OutputList(ValueSequenceFact::single(
        resident(StorageFact::Dense),
    )));
    assert_eq!(gathered_value_fact(&sequence).kind, ValueKindFact::Never);
}
