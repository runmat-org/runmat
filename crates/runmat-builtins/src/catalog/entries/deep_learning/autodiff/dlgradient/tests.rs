use super::*;
use runmat_types::{
    standard, AliasFact, LiteralContext, NumericClass, NumericDomain, NumericFact, ObjectFact,
    OutputSelection, RequestedOutputCount, ResidencyFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact, ViewFact,
};
use std::collections::BTreeMap;

fn dlarray(data: ValueFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(standard::DL_ARRAY.owned()),
            properties: BTreeMap::from([(crate::DLARRAY_DATA_PROPERTY.into(), data.clone())]),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        data.shape,
        StorageFact::Opaque,
    )
}

#[test]
fn traced_dlarray_target_preserves_known_data_class_and_shape() {
    let data = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(1), Some(3)]),
        StorageFact::Dense,
    );
    let loss = dlarray(ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Single,
        domain: NumericDomain::Real,
    })));
    let request = CallRequest {
        arguments: vec![loss, dlarray(data.clone())],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let result = crate::infer_catalog_call(&DLGRADIENT_CATALOG_ENTRY, &request);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(result.outputs.len(), 1);
    let ValueKindFact::Object(object) = &result.outputs[0].kind else {
        panic!("dlarray gradient")
    };
    assert_eq!(object.runtime_class, Some(standard::DL_ARRAY.owned()));
    assert_eq!(object.properties[crate::DLARRAY_DATA_PROPERTY], data);
}

#[test]
fn gradient_rejects_excess_outputs_and_known_non_dlarray_loss() {
    let scalar = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let request = CallRequest {
        arguments: vec![scalar.clone(), dlarray(scalar)],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
    };
    let result = crate::infer_catalog_call(&DLGRADIENT_CATALOG_ENTRY, &request);
    assert!(result
        .diagnostics
        .iter()
        .any(|diag| diag.code == "RM-DL-DLGRADIENT-OUTPUTS"));
    assert!(result
        .diagnostics
        .iter()
        .any(|diag| diag.code == "RM-DL-DLGRADIENT-LOSS"));
}

#[test]
fn logical_target_gradient_is_double_and_complex_data_is_not_admitted() {
    let loss = dlarray(ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    })));
    let logical = dlarray(ValueFact::proven(
        ValueKindFact::Logical,
        ShapeFact::from(vec![Some(1), Some(2)]),
        StorageFact::Dense,
    ));
    let request = CallRequest {
        arguments: vec![loss.clone(), logical],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let result = crate::infer_catalog_call(&DLGRADIENT_CATALOG_ENTRY, &request);
    let ValueKindFact::Object(object) = &result.outputs[0].kind else {
        panic!("dlarray")
    };
    let gradient = &object.properties[crate::DLARRAY_DATA_PROPERTY];
    assert_eq!(gradient.numeric().unwrap().class, NumericClass::Double);
    let complex = dlarray(ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Single,
        domain: NumericDomain::Complex,
    })));
    let request = CallRequest {
        arguments: vec![loss, complex],
        ..request
    };
    let result = crate::infer_catalog_call(&DLGRADIENT_CATALOG_ENTRY, &request);
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-DLGRADIENT-CLASS"));
}

#[test]
fn known_resident_data_is_diagnosed_without_claiming_a_device_gradient() {
    let loss = dlarray(ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    })));
    let mut data = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(1), Some(3)]),
        StorageFact::Dense,
    );
    data.residency = ResidencyFact::Device { provider: None };
    let request = CallRequest {
        arguments: vec![loss.clone(), dlarray(data.clone())],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let result = crate::infer_catalog_call(&DLGRADIENT_CATALOG_ENTRY, &request);
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-DLGRADIENT-RESIDENCY"));
    let ValueKindFact::Object(object) = &result.outputs[0].kind else {
        panic!("gradient wrapper")
    };
    let gradient = &object.properties[crate::DLARRAY_DATA_PROPERTY];
    assert_eq!(gradient.kind, ValueKindFact::Unknown);
    assert_eq!(gradient.residency, ResidencyFact::Unknown);

    let result = crate::infer_catalog_call(
        &DLGRADIENT_CATALOG_ENTRY,
        &CallRequest {
            arguments: vec![
                dlarray(data),
                dlarray(ValueFact::scalar(ValueKindFact::Logical)),
            ],
            ..request
        },
    );
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-DLGRADIENT-RESIDENCY"));
}

#[test]
fn host_gradient_is_new_dense_storage_and_known_invalid_data_is_not_preserved() {
    let loss = dlarray(ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    })));
    let mut shared_view = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(2)]),
        StorageFact::Dense,
    );
    shared_view.alias = AliasFact::Shared;
    shared_view.view = ViewFact::ReadOnlyView;
    let request = CallRequest {
        arguments: vec![loss.clone(), dlarray(shared_view.clone())],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let result = crate::infer_catalog_call(&DLGRADIENT_CATALOG_ENTRY, &request);
    let ValueKindFact::Object(object) = &result.outputs[0].kind else {
        panic!("gradient wrapper")
    };
    let gradient = &object.properties[crate::DLARRAY_DATA_PROPERTY];
    assert_eq!(gradient.storage, StorageFact::Dense);
    assert_eq!(gradient.residency, ResidencyFact::Host);
    assert_eq!(gradient.shape, ShapeFact::from(vec![Some(2), Some(2)]));
    assert_eq!(gradient.alias, AliasFact::Unique);
    assert_eq!(gradient.view, ViewFact::Materialized);

    let mut sparse = shared_view;
    sparse.storage = StorageFact::Sparse;
    let result = crate::infer_catalog_call(
        &DLGRADIENT_CATALOG_ENTRY,
        &CallRequest {
            arguments: vec![loss.clone(), dlarray(sparse)],
            ..request.clone()
        },
    );
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-DLGRADIENT-STORAGE"));
    let ValueKindFact::Object(object) = &result.outputs[0].kind else {
        panic!("gradient wrapper")
    };
    assert_eq!(
        object.properties[crate::DLARRAY_DATA_PROPERTY].kind,
        ValueKindFact::Unknown
    );

    let result = crate::infer_catalog_call(
        &DLGRADIENT_CATALOG_ENTRY,
        &CallRequest {
            arguments: vec![loss, dlarray(ValueFact::scalar(ValueKindFact::String))],
            ..request
        },
    );
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-DLGRADIENT-CLASS"));
    let ValueKindFact::Object(object) = &result.outputs[0].kind else {
        panic!("gradient wrapper")
    };
    assert_eq!(
        object.properties[crate::DLARRAY_DATA_PROPERTY].kind,
        ValueKindFact::Unknown
    );

    let missing_data = ValueFact::scalar(ValueKindFact::Object(ObjectFact {
        class: None,
        runtime_class: Some(standard::DL_ARRAY.owned()),
        properties: BTreeMap::new(),
        properties_complete: true,
        handle_semantics: Some(false),
    }));
    let result = crate::infer_catalog_call(
        &DLGRADIENT_CATALOG_ENTRY,
        &CallRequest {
            arguments: vec![
                dlarray(ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                }))),
                missing_data,
            ],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
    );
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-DLGRADIENT-DATA"));
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
}
