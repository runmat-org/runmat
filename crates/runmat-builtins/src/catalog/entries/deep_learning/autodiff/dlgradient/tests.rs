use super::*;
use runmat_types::{
    standard, AliasFact, LiteralContext, NumericClass, NumericDomain, NumericFact, ObjectFact,
    OutputSelection, RequestedOutputCount, ResidencyFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact, ViewFact,
};
use std::collections::BTreeMap;

#[path = "tests/tree.rs"]
mod tree;

#[test]
fn nested_cell_gradient_preserves_positions_and_native_leaf_facts() {
    let leaf = dlarray(ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    ));
    let nested = ValueFact::proven(
        ValueKindFact::Cell(runmat_types::CellFact {
            element: Box::new(leaf.clone()),
            elements: vec![leaf],
            elements_complete: true,
        }),
        ShapeFact::from(vec![Some(1), Some(1)]),
        StorageFact::Opaque,
    );
    let target = ValueFact::proven(
        ValueKindFact::Cell(runmat_types::CellFact {
            element: Box::new(ValueFact::unknown(
                runmat_types::DynamicReason::RuntimeValue,
            )),
            elements: vec![nested],
            elements_complete: true,
        }),
        ShapeFact::from(vec![Some(1), Some(1)]),
        StorageFact::Opaque,
    );
    let result = crate::infer_catalog_call(
        &DLGRADIENT_CATALOG_ENTRY,
        &CallRequest {
            arguments: vec![dlarray(ValueFact::scalar(ValueKindFact::Logical)), target],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert!(result.diagnostics.is_empty());
    let ValueKindFact::Cell(outer) = &result.outputs[0].kind else {
        panic!("cell gradient")
    };
    let ValueKindFact::Cell(inner) = &outer.elements[0].kind else {
        panic!("nested cell gradient")
    };
    let ValueKindFact::Object(leaf) = &inner.elements[0].kind else {
        panic!("dlarray gradient")
    };
    let data = &leaf.properties[crate::DLARRAY_DATA_PROPERTY];
    assert_eq!(data.numeric().unwrap().class, NumericClass::Single);
    assert_eq!(data.shape, ShapeFact::from(vec![Some(2), Some(3)]));
    assert_eq!(data.alias, AliasFact::Unknown);
    assert!(outer.elements_complete && inner.elements_complete);
}

#[test]
fn cell_target_admission_checks_known_members_not_empty_common_facts() {
    let loss = dlarray(ValueFact::scalar(ValueKindFact::Logical));
    let invalid = ValueFact::scalar(ValueKindFact::Logical);
    let cell = |elements: Vec<ValueFact>, elements_complete| {
        ValueFact::proven(
            ValueKindFact::Cell(runmat_types::CellFact {
                element: Box::new(invalid.clone()),
                elements,
                elements_complete,
            }),
            ShapeFact::from(vec![Some(1), Some(0)]),
            StorageFact::Opaque,
        )
    };
    let request = |target| CallRequest {
        arguments: vec![loss.clone(), target],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let empty = crate::infer_catalog_call(&DLGRADIENT_CATALOG_ENTRY, &request(cell(vec![], true)));
    assert!(empty.diagnostics.is_empty());
    let incomplete =
        crate::infer_catalog_call(&DLGRADIENT_CATALOG_ENTRY, &request(cell(vec![], false)));
    assert!(incomplete.diagnostics.is_empty());
    let mut nonempty = cell(vec![], false);
    nonempty.shape = ShapeFact::Scalar;
    let result = crate::infer_catalog_call(&DLGRADIENT_CATALOG_ENTRY, &request(nonempty));
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-DLGRADIENT-TARGET"));
    let text_member = ValueFact::scalar(ValueKindFact::String);
    let result = crate::infer_catalog_call(
        &DLGRADIENT_CATALOG_ENTRY,
        &request(cell(vec![text_member], false)),
    );
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-DLGRADIENT-TARGET"));
    let invalid_cell = ValueFact::proven(
        ValueKindFact::Cell(runmat_types::CellFact {
            element: Box::new(invalid.clone()),
            elements: vec![invalid],
            elements_complete: true,
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    let result = crate::infer_catalog_call(&DLGRADIENT_CATALOG_ENTRY, &request(invalid_cell));
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-DLGRADIENT-TARGET"
            && diagnostic.argument == Some(1)));
}

#[test]
fn discarded_gradient_targets_do_not_emit_admission_diagnostics() {
    let loss = dlarray(ValueFact::scalar(ValueKindFact::Logical));
    let mut resident = ValueFact::scalar(ValueKindFact::Logical);
    resident.residency = ResidencyFact::Device { provider: None };
    let resident = dlarray(resident);
    let nested = ValueFact::proven(
        ValueKindFact::Cell(runmat_types::CellFact {
            element: Box::new(resident.clone()),
            elements: vec![resident],
            elements_complete: true,
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    let request = CallRequest {
        arguments: vec![loss.clone(), loss, nested],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let result = crate::infer_catalog_call(&DLGRADIENT_CATALOG_ENTRY, &request);
    assert!(result.diagnostics.is_empty());
    let dynamic = crate::infer_catalog_call(
        &DLGRADIENT_CATALOG_ENTRY,
        &CallRequest {
            outputs: OutputSelection::new(RequestedOutputCount::CurrentFunctionNargout),
            ..request.clone()
        },
    );
    assert!(!dynamic
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-DLGRADIENT-CLASS"));
    let result = crate::infer_catalog_call(
        &DLGRADIENT_CATALOG_ENTRY,
        &CallRequest {
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
            ..request
        },
    );
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-DLGRADIENT-CLASS"
            && diagnostic.argument == Some(2)));
}

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
    let mut data = ValueFact::proven(
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
    // Gradient class/shape survive, but loss-graph placement and repeated
    // target ownership cannot be inferred from the target's host Data.
    data.alias = AliasFact::Unknown;
    data.residency = ResidencyFact::Unknown;
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
fn resident_data_is_conservative_without_blanket_rejection_or_proven_device_gradient() {
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
    assert!(!result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-DLGRADIENT-RESIDENCY"));
    let ValueKindFact::Object(object) = &result.outputs[0].kind else {
        panic!("gradient wrapper")
    };
    let gradient = &object.properties[crate::DLARRAY_DATA_PROPERTY];
    assert_eq!(gradient.numeric().unwrap().class, NumericClass::Single);
    assert_eq!(gradient.shape, ShapeFact::from(vec![Some(1), Some(3)]));
    assert_eq!(gradient.residency, ResidencyFact::Unknown);
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);

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
    assert!(!result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-DLGRADIENT-RESIDENCY"));
}

#[test]
fn target_class_and_shape_are_known_but_loss_graph_controls_gradient_residency() {
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
    assert_eq!(gradient.residency, ResidencyFact::Unknown);
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    assert_eq!(gradient.shape, ShapeFact::from(vec![Some(2), Some(2)]));
    assert_eq!(gradient.alias, AliasFact::Unknown);
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

#[test]
fn resident_loss_with_host_parameter_target_does_not_prove_host_gradient() {
    let mut loss_data = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Single,
        domain: NumericDomain::Real,
    }));
    loss_data.residency = ResidencyFact::Device { provider: None };
    let parameter_data = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(1), Some(2)]),
        StorageFact::Dense,
    );
    let result = crate::infer_catalog_call(
        &DLGRADIENT_CATALOG_ENTRY,
        &CallRequest {
            arguments: vec![dlarray(loss_data), dlarray(parameter_data)],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert!(result.diagnostics.is_empty());
    let ValueKindFact::Object(wrapper) = &result.outputs[0].kind else {
        panic!("dlarray gradient expected")
    };
    let data = &wrapper.properties[crate::DLARRAY_DATA_PROPERTY];
    assert_eq!(data.numeric().unwrap().class, NumericClass::Single);
    assert_eq!(data.shape, ShapeFact::from(vec![Some(1), Some(2)]));
    assert_eq!(data.residency, ResidencyFact::Unknown);
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
}

#[test]
fn repeated_target_gradients_do_not_claim_exclusive_data_ownership() {
    let target_data = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(1)]),
        StorageFact::Dense,
    );
    let target = dlarray(target_data);
    let loss = dlarray(ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Single,
        domain: NumericDomain::Real,
    })));
    let request = CallRequest {
        arguments: vec![loss, target.clone(), target],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
    };
    let result = crate::infer_catalog_call(&DLGRADIENT_CATALOG_ENTRY, &request);
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs.len(), 2);
    for output in &result.outputs {
        let ValueKindFact::Object(object) = &output.kind else {
            panic!("gradient wrapper")
        };
        let gradient = &object.properties[crate::DLARRAY_DATA_PROPERTY];
        assert_eq!(gradient.alias, AliasFact::Unknown);
        assert_eq!(gradient.view, ViewFact::Materialized);
        assert_eq!(gradient.numeric().unwrap().class, NumericClass::Single);
    }
}
