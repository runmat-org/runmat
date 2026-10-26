use super::*;
use crate::{infer_catalog_call, BuiltinCatalogEntry};
use runmat_types::*;

fn numeric(class: NumericClass, shape: ShapeFact, storage: StorageFact) -> ValueFact {
    let mut value = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        shape,
        storage,
    );
    value.residency = ResidencyFact::Host;
    value
}

fn callback(output: ValueFact) -> ValueFact {
    ValueFact::scalar(ValueKindFact::Callable(CallableFact {
        identity: None,
        capabilities: CapabilitySet([CapabilityRequirement::Filesystem].into_iter().collect()),
        parameters: vec![],
        parameters_complete: false,
        outputs: vec![output],
        outputs_complete: true,
        variadic_inputs: true,
        variadic_outputs: false,
        captures: vec![],
        captures_complete: true,
    }))
}

fn request(entry: &BuiltinCatalogEntry, initial: ValueFact, count: usize) -> CallRequest {
    let scalar = numeric(NumericClass::Double, ShapeFact::Scalar, StorageFact::Scalar);
    let mut arguments = vec![callback(scalar.clone()), initial];
    if entry.identity == LSQCURVEFIT_CATALOG_ENTRY.identity {
        arguments.extend([ValueFact::scalar(ValueKindFact::String), scalar]);
    }
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
    }
}

fn entries() -> [(&'static BuiltinCatalogEntry, usize); 4] {
    [
        (&FMINUNC_CATALOG_ENTRY, 6),
        (&FSOLVE_CATALOG_ENTRY, 5),
        (&LSQCURVEFIT_CATALOG_ENTRY, 7),
        (&LSQNONLIN_CATALOG_ENTRY, 7),
    ]
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn nonlinear_family_preserves_double_solution_shape_outputs_and_callback_requirements() {
    for (entry, maximum) in entries() {
        for initial in [
            numeric(NumericClass::Double, ShapeFact::Scalar, StorageFact::Scalar),
            numeric(
                NumericClass::Single,
                ShapeFact::from(vec![Some(2), Some(1), Some(3)]),
                StorageFact::Dense,
            ),
        ] {
            for count in 0..=maximum {
                let result = infer_catalog_call(entry, &request(entry, initial.clone(), count));
                assert!(
                    result.diagnostics.is_empty(),
                    "{}: {:?}",
                    entry.identity.name,
                    result.diagnostics
                );
                assert_eq!(result.outputs.len(), count);
                if let Some(solution) = result.outputs.first() {
                    assert_eq!(solution.shape, initial.shape);
                    assert_eq!(solution.numeric().unwrap().class, NumericClass::Double);
                    assert_eq!(solution.residency, ResidencyFact::Host);
                    assert_eq!(solution.storage, initial.storage);
                }
                assert!(result
                    .capabilities
                    .0
                    .contains(&CapabilityRequirement::Filesystem));
                assert!(result.effects.0.contains(&EffectKind::HostCallback));
            }
        }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn nonlinear_family_keeps_unknown_carriers_and_rejects_proven_invalid_metadata() {
    for (entry, maximum) in entries() {
        let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
        let inferred = infer_catalog_call(entry, &request(entry, unknown, 1));
        assert!(inferred.diagnostics.is_empty());
        assert_eq!(inferred.outputs[0].storage, StorageFact::Unknown);
        let valid = numeric(NumericClass::Double, ShapeFact::Scalar, StorageFact::Scalar);
        let excessive = infer_catalog_call(entry, &request(entry, valid.clone(), maximum + 1));
        assert!(!excessive.diagnostics.is_empty());
        for invalid in [
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Complex,
                }),
                ShapeFact::Scalar,
                StorageFact::Scalar,
            ),
            numeric(
                NumericClass::Double,
                ShapeFact::from(vec![Some(0), Some(1)]),
                StorageFact::Dense,
            ),
            numeric(
                NumericClass::Double,
                ShapeFact::from(vec![Some(2), Some(2)]),
                StorageFact::Sparse,
            ),
        ] {
            let result = infer_catalog_call(entry, &request(entry, invalid, 1));
            assert!(!result.diagnostics.is_empty(), "{}", entry.identity.name);
            assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));
        }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn least_squares_forms_preserve_distinct_bound_and_problem_contracts() {
    let initial = numeric(
        NumericClass::Single,
        ShapeFact::from(vec![Some(3), Some(1)]),
        StorageFact::Dense,
    );
    let bound = numeric(
        NumericClass::Double,
        ShapeFact::from(vec![Some(2), Some(1)]),
        StorageFact::Dense,
    );
    let empty = numeric(
        NumericClass::Double,
        ShapeFact::from(vec![Some(0), Some(0)]),
        StorageFact::Dense,
    );
    let mut nonlin = request(&LSQNONLIN_CATALOG_ENTRY, initial.clone(), 7);
    nonlin.arguments.extend([bound.clone(), empty.clone()]);
    assert!(infer_catalog_call(&LSQNONLIN_CATALOG_ENTRY, &nonlin)
        .diagnostics
        .is_empty());
    let mut configurable = request(&LSQNONLIN_CATALOG_ENTRY, initial.clone(), 7);
    let controls = [
        (
            "SpecifyObjectiveGradient".into(),
            ValueFact::unknown(DynamicReason::RuntimeValue),
        ),
        (
            "MaxFunctionEvaluations".into(),
            numeric(NumericClass::Double, ShapeFact::Scalar, StorageFact::Scalar),
        ),
    ]
    .into_iter()
    .collect();
    configurable.arguments.extend([
        empty.clone(),
        empty.clone(),
        ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(controls, true))),
    ]);
    let products = infer_catalog_call(&LSQNONLIN_CATALOG_ENTRY, &configurable);
    assert!(products.diagnostics.is_empty());
    assert_eq!(
        products.outputs[2].shape,
        ShapeFact::Unknown,
        "an unevaluated residual has no proven callback shape"
    );
    assert!(
        matches!(products.outputs[6].kind, ValueKindFact::Unknown),
        "supplied J can retain native single or double and dense or CSC storage"
    );
    let ValueKindFact::Struct(info) = &products.outputs[4].kind else {
        panic!("information");
    };
    assert!(!info.fields_complete, "not-started adds requestedalgorithm");
    let mut curve = request(&LSQCURVEFIT_CATALOG_ENTRY, initial, 7);
    curve.arguments.extend([bound, empty.clone()]);
    assert!(!infer_catalog_call(&LSQCURVEFIT_CATALOG_ENTRY, &curve)
        .diagnostics
        .is_empty());
    let mut options_only = request(
        &LSQCURVEFIT_CATALOG_ENTRY,
        numeric(NumericClass::Double, ShapeFact::Scalar, StorageFact::Scalar),
        1,
    );
    options_only
        .arguments
        .push(ValueFact::scalar(ValueKindFact::Struct(
            StructFact::scalar(Default::default(), true),
        )));
    assert!(
        infer_catalog_call(&LSQCURVEFIT_CATALOG_ENTRY, &options_only)
            .diagnostics
            .is_empty()
    );
    let fields = [
        ("objective".into(), nonlin.arguments[0].clone()),
        ("x0".into(), nonlin.arguments[1].clone()),
    ]
    .into_iter()
    .collect();
    nonlin.arguments = vec![ValueFact::scalar(ValueKindFact::Struct(
        StructFact::scalar(fields, true),
    ))];
    assert!(infer_catalog_call(&LSQNONLIN_CATALOG_ENTRY, &nonlin)
        .diagnostics
        .is_empty());
    if let ValueKindFact::Struct(problem) = &mut nonlin.arguments[0].kind {
        problem.fields.insert(
            "Aeq".into(),
            numeric(NumericClass::Double, ShapeFact::Scalar, StorageFact::Scalar),
        );
    }
    assert!(!infer_catalog_call(&LSQNONLIN_CATALOG_ENTRY, &nonlin)
        .diagnostics
        .is_empty());
}
