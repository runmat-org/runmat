use super::*;
use runmat_types::{LiteralContext, NumericFact, OutputSelection, RequestedOutputCount};

fn request(classes: &[&str]) -> CallRequest {
    let mut arguments = vec![ValueFact::scalar(numeric_kind(
        NumericClass::Double,
        NumericDomain::Real,
    ))];
    let mut literals = vec![LiteralValue::Number(2.0)];
    for class in classes {
        arguments.push(ValueFact::scalar(ValueKindFact::String));
        literals.push(LiteralValue::String((*class).into()));
    }
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

#[test]
fn gpu_zeros_inference_preserves_underlying_class_and_residency() {
    let entry = crate::builtin_catalog_entry_by_name("zeros").expect("zeros catalog entry");
    for (controls, kind) in [
        (
            &["gpuArray"][..],
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
        ),
        (
            &["single", "gpuArray"][..],
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single,
                domain: NumericDomain::Real,
            }),
        ),
        (
            &["uint64", "gpuArray"][..],
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt64,
                domain: NumericDomain::Real,
            }),
        ),
        (&["logical", "gpuArray"][..], ValueKindFact::Logical),
    ] {
        let result = infer_zeros(&request(controls), entry);
        assert!(
            result.diagnostics.is_empty(),
            "{controls:?}: {:?}",
            result.diagnostics
        );
        assert_eq!(result.outputs[0].kind, kind);
        assert_eq!(
            result.outputs[0].residency,
            ResidencyFact::Device { provider: None }
        );
        assert_eq!(result.outputs[0].storage, StorageFact::Dense);
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(2)])
        );
    }
}

#[test]
fn no_dimension_static_gpu_zeros_is_still_a_resident_array() {
    let entry = crate::builtin_catalog_entry_by_name("zeros").unwrap();
    let result = infer_zeros(
        &CallRequest {
            arguments: vec![ValueFact::scalar(ValueKindFact::String)],
            literals: LiteralContext::new(vec![LiteralValue::String("gpuArray".into())]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
        entry,
    );
    assert_eq!(
        result.outputs[0].residency,
        ResidencyFact::Device { provider: None }
    );
    assert_eq!(result.outputs[0].storage, StorageFact::Dense);
}

#[test]
fn dynamic_gpu_element_class_never_infers_default_double() {
    let entry = crate::builtin_catalog_entry_by_name("zeros").unwrap();
    let mut dynamic = request(&["gpuArray"]);
    dynamic
        .arguments
        .insert(1, ValueFact::scalar(ValueKindFact::String));
    dynamic
        .literals
        .literal_args
        .insert(1, LiteralValue::Unknown);
    let result = infer_zeros(&dynamic, entry);
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(
        result.outputs[0].residency,
        ResidencyFact::Device { provider: None }
    );
}

#[test]
fn object_creation_result_is_not_inferred_from_its_prototype() {
    let entry = crate::builtin_catalog_entry_by_name("zeros").unwrap();
    let mut call = request(&["like"]);
    call.arguments.push(ValueFact::scalar(ValueKindFact::Object(
        runmat_types::ObjectFact {
            class: None,
            runtime_class: Some(runmat_types::standard::OPTIMIZATION_VARIABLE.owned()),
            properties: Default::default(),
            properties_complete: false,
            handle_semantics: Some(true),
        },
    )));
    call.literals.literal_args.push(LiteralValue::Unknown);
    let result = infer_zeros(&call, entry);
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(result.outputs[0].shape, ShapeFact::Unknown);
}

#[test]
fn definite_reordered_or_repeated_gpu_controls_are_diagnosed() {
    let entry = crate::builtin_catalog_entry_by_name("zeros").unwrap();
    for controls in [
        &["gpuArray", "single"][..],
        &["single", "double", "gpuArray"][..],
    ] {
        let result = infer_zeros(&request(controls), entry);
        assert!(result
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-ZEROS-CLASS-CONFLICT"));
    }
}
