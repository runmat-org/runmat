use super::*;
use crate::*;
use runmat_types::*;
#[test]
fn nnls_outputs_have_actual_rectangular_geometry_and_options_are_not_ignored() {
    let numeric = |shape| {
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(shape),
            StorageFact::Dense,
        )
    };
    let mut request = CallRequest {
        arguments: vec![
            numeric(vec![Some(4), Some(2)]),
            numeric(vec![Some(4), Some(1)]),
        ],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(6)),
    };
    let result = inference::infer(&request, &LSQNONNEG_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(1)])
    );
    assert_eq!(
        result.outputs[2].shape,
        ShapeFact::from(vec![Some(4), Some(1)])
    );
    assert_eq!(result.outputs[5].shape, result.outputs[0].shape);
    request.arguments[1] = numeric(vec![Some(3), Some(1)]);
    assert!(!inference::infer(&request, &LSQNONNEG_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}
#[test]
fn consumed_legacy_controls_and_class_extensions_are_explicit() {
    use super::super::{OptimParameterLookup, OptimParameterName};
    for (text, expected) in [
        ("TolX", LsqnonnegOption::Tolerance),
        ("MaxIter", LsqnonnegOption::MaxIterations),
        ("Display", LsqnonnegOption::Display),
    ] {
        let OptimParameterLookup::Matched(name) = OptimParameterName::resolve(text) else {
            panic!("known option");
        };
        assert_eq!(LsqnonnegOption::from_parameter(name), Some(expected));
    }
    let OptimParameterLookup::Matched(name) = OptimParameterName::resolve("OutputFcn") else {
        panic!("known option");
    };
    assert_eq!(LsqnonnegOption::from_parameter(name), None);
    assert_eq!(LSQNONNEG_EXTENSIONS.len(), 4);
    assert!(LSQNONNEG_INTEGER_CAPABILITIES
        .iter()
        .flat_map(|cap| cap.inputs)
        .all(|v| v.availability == BuiltinIntegerInputAvailability::RunMatOnly));
}
