//! Applicability: all five canonical structural contracts; shared metadata and
//! output-representation cases complement each operation's existing runtime tests.
use super::*;
use crate::*;
fn numeric(class: NumericClass, domain: NumericDomain, shape: &[usize]) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact { class, domain }),
        ShapeFact::from(shape.iter().copied().map(Some).collect::<Vec<_>>()),
        StorageFact::Dense,
    )
}
fn vector(values: &[f64]) -> LiteralValue {
    LiteralValue::Vector(values.iter().copied().map(LiteralValue::Number).collect())
}
fn call(
    entry: &BuiltinCatalogEntry,
    source: ValueFact,
    controls: Vec<ValueFact>,
    literals: Vec<LiteralValue>,
) -> CallInference {
    let mut arguments = vec![source];
    arguments.extend(controls);
    let mut literal_args = vec![LiteralValue::Unknown];
    literal_args.extend(literals);
    crate::infer_catalog_call(
        entry,
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literal_args),
            outputs: OutputSelection::default(),
        },
    )
}
#[test]
fn transforms_preserve_classes_domains_and_only_proven_physical_facts() {
    for class in [
        NumericClass::Single,
        NumericClass::Double,
        NumericClass::Int64,
        NumericClass::UInt64,
    ] {
        for domain in [NumericDomain::Real, NumericDomain::Complex] {
            for device in [false, true] {
                let mut source = numeric(class, domain, &[2, 3]);
                if device {
                    source.storage = StorageFact::Opaque;
                    source.layout = LayoutFact::Strided;
                    source.contiguity = ContiguityFact::NonContiguous;
                    source.residency = ResidencyFact::Device {
                        provider: Some("fixture".into()),
                    };
                }
                let cases = [
                    (&RESHAPE_CATALOG_ENTRY, vec![3.0, 2.0], vec![3, 2]),
                    (&REPMAT_CATALOG_ENTRY, vec![2.0], vec![4, 6]),
                    (&PERMUTE_CATALOG_ENTRY, vec![2.0, 1.0], vec![3, 2]),
                    (&IPERMUTE_CATALOG_ENTRY, vec![2.0, 1.0], vec![3, 2]),
                ];
                for (entry, values, expected) in cases {
                    let result = call(
                        entry,
                        source.clone(),
                        vec![numeric(
                            NumericClass::Double,
                            NumericDomain::Real,
                            &[1, values.len()],
                        )],
                        vec![vector(&values)],
                    );
                    assert!(
                        result.diagnostics.is_empty(),
                        "{}: {:?}",
                        entry.identity.name,
                        result.diagnostics
                    );
                    assert_eq!(result.outputs[0].numeric(), source.numeric());
                    assert_eq!(
                        result.outputs[0].shape,
                        ShapeFact::from(expected.into_iter().map(Some).collect::<Vec<_>>())
                    );
                    let retains_device = entry.identity.name == "reshape";
                    assert_eq!(
                        result.outputs[0].residency,
                        if device && !retains_device {
                            ResidencyFact::Unknown
                        } else {
                            source.residency.clone()
                        }
                    );
                    assert_eq!(
                        result.outputs[0].layout,
                        if device {
                            LayoutFact::Unknown
                        } else {
                            LayoutFact::ColumnMajor
                        }
                    );
                    if device {
                        assert_eq!(
                            result.outputs[0].storage,
                            if retains_device {
                                StorageFact::Opaque
                            } else {
                                StorageFact::Unknown
                            }
                        );
                    }
                }
                source.shape = ShapeFact::from(vec![Some(1), Some(2), Some(1), Some(3)]);
                let result = call(&SQUEEZE_CATALOG_ENTRY, source.clone(), vec![], vec![]);
                assert_eq!(
                    result.outputs[0].shape,
                    ShapeFact::from(vec![Some(2), Some(3)])
                );
                assert_eq!(result.outputs[0].kind, source.kind);
                assert_eq!(result.outputs[0].residency, source.residency);
                assert_eq!(result.outputs[0].storage, source.storage);
                if device {
                    assert_eq!(result.outputs[0].layout, LayoutFact::Unknown);
                    assert_eq!(result.outputs[0].contiguity, ContiguityFact::Unknown);
                }
            }
        }
    }
}
#[test]
fn structural_controls_use_contents_not_shape_and_unknowns_do_not_invent_extents() {
    let source = numeric(NumericClass::Single, NumericDomain::Real, &[2, 3]);
    let mut unknown_rank = source.clone();
    unknown_rank.shape = ShapeFact::Unknown;
    let unknown_tiling = call(
        &REPMAT_CATALOG_ENTRY,
        unknown_rank,
        vec![numeric(NumericClass::Double, NumericDomain::Real, &[1, 2])],
        vec![vector(&[2.0, 3.0])],
    );
    assert_eq!(unknown_tiling.outputs[0].shape, ShapeFact::Unknown);
    assert_eq!(unknown_tiling.outputs[0].numeric(), source.numeric());
    let control = numeric(NumericClass::UInt64, NumericDomain::Real, &[1, 2]);
    let result = call(
        &RESHAPE_CATALOG_ENTRY,
        source.clone(),
        vec![control.clone()],
        vec![LiteralValue::Unknown],
    );
    assert_eq!(result.outputs[0].shape, ShapeFact::from(vec![None, None]));
    let result = call(
        &REPMAT_CATALOG_ENTRY,
        source.clone(),
        vec![control],
        vec![LiteralValue::Unknown],
    );
    assert_eq!(result.outputs[0].shape, ShapeFact::from(vec![None, None]));
    let mut empty = source.clone();
    empty.shape = ShapeFact::from(vec![Some(0), Some(3)]);
    let result = call(
        &REPMAT_CATALOG_ENTRY,
        empty,
        vec![ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }))],
        vec![LiteralValue::Unknown],
    );
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(0), None])
    );
    let invalid = call(
        &PERMUTE_CATALOG_ENTRY,
        ValueFact::unknown(DynamicReason::RuntimeValue),
        vec![numeric(NumericClass::Double, NumericDomain::Real, &[1, 2])],
        vec![vector(&[1.0, 1.0])],
    );
    assert!(!invalid.diagnostics.is_empty());
    let mut request = CallRequest {
        arguments: vec![source],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
    };
    assert!(!crate::infer_catalog_call(&SQUEEZE_CATALOG_ENTRY, &request)
        .diagnostics
        .is_empty());
    request.outputs = OutputSelection::new(RequestedOutputCount::Exactly(0));
    assert!(crate::infer_catalog_call(&SQUEEZE_CATALOG_ENTRY, &request)
        .diagnostics
        .is_empty());
}
#[test]
fn metadata_admission_and_scalar_publication_match_existing_runtime_forms() {
    let structure = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
        Default::default(),
        true,
    )));
    let result = call(
        &REPMAT_CATALOG_ENTRY,
        structure.clone(),
        vec![numeric(NumericClass::Double, NumericDomain::Real, &[1, 1])],
        vec![LiteralValue::Number(2.0)],
    );
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(1)])
    );
    let mut array = structure;
    array.storage = StorageFact::Dense;
    array.shape = ShapeFact::from(vec![Some(2), Some(3), Some(1), Some(1)]);
    let result = call(
        &PERMUTE_CATALOG_ENTRY,
        array,
        vec![numeric(NumericClass::Double, NumericDomain::Real, &[1, 2])],
        vec![vector(&[2.0, 1.0])],
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(2)])
    );
    let mut logical = ValueFact::scalar(ValueKindFact::Logical);
    let result = call(
        &PERMUTE_CATALOG_ENTRY,
        logical.clone(),
        vec![numeric(NumericClass::Double, NumericDomain::Real, &[1, 2])],
        vec![vector(&[2.0, 1.0])],
    );
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    let result = call(
        &REPMAT_CATALOG_ENTRY,
        logical.clone(),
        vec![logical.clone()],
        vec![LiteralValue::Bool(true)],
    );
    assert_eq!(result.outputs[0].kind, ValueKindFact::Logical);
    assert_eq!(result.outputs[0].storage, StorageFact::Dense);
    logical.residency = ResidencyFact::Device { provider: None };
    assert!(!shape_transform_control_supported(
        &logical,
        ArrayShapeTransform::Permute,
        false
    ));
    assert!(shape_transform_control_supported(
        &logical,
        ArrayShapeTransform::Reshape,
        false
    ));
    let complex = numeric(NumericClass::Single, NumericDomain::Complex, &[1, 1]);
    assert!(!shape_transform_control_supported(
        &complex,
        ArrayShapeTransform::Tile,
        true
    ));
}
