use super::*;
use runmat_types::{
    CallRequest, DynamicReason, LiteralContext, LiteralValue, NumericClass, NumericDomain,
    NumericFact, ObjectFact, OutputSelection, RequestedOutputCount, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

fn request(arguments: Vec<ValueFact>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

fn with_literals(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

fn double() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))
}

fn numeric(class: NumericClass, dims: Vec<Option<usize>>) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(dims),
        StorageFact::Dense,
    )
}

fn vector3() -> ValueFact {
    numeric(NumericClass::Double, vec![Some(1), Some(3)])
}

fn matrix4() -> ValueFact {
    numeric(NumericClass::Double, vec![Some(4), Some(4)])
}

fn object(class: runmat_types::StaticClassIdentity) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(class.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

fn feature() -> ValueFact {
    object(GEOMETRY_FEATURE_CLASS)
}

fn unknown() -> ValueFact {
    ValueFact::unknown(DynamicReason::RuntimeValue)
}

fn infer(entry: &BuiltinCatalogEntry, request: &CallRequest) -> runmat_types::CallInference {
    let BuiltinInferenceRule::Identity(rule) = entry.contract.inference_rule else {
        panic!(
            "{} must use an identity inference rule",
            entry.identity.name
        );
    };
    rule.infer(request, entry)
}

fn codes(entry: &BuiltinCatalogEntry, request: &CallRequest) -> Vec<String> {
    infer(entry, request)
        .diagnostics
        .into_iter()
        .map(|diagnostic| diagnostic.code)
        .collect()
}

fn result_class(entry: &BuiltinCatalogEntry, request: &CallRequest) -> Option<String> {
    let inference = infer(entry, request);
    match &inference.outputs.first()?.kind {
        ValueKindFact::Object(object) => {
            object.runtime_class.as_ref().map(|class| class.to_string())
        }
        _ => None,
    }
}

#[test]
fn every_builder_identity_is_valid_catalog_content() {
    let errors = crate::validate_builtin_catalog(ENTRIES);
    assert!(errors.is_empty(), "{errors:?}");
    assert_eq!(ENTRIES.len(), 21);
    for entry in ENTRIES {
        assert!(entry.identity.name.starts_with("geometry."));
        assert_eq!(entry.contract.purity, BuiltinPurity::Pure);
        assert_eq!(
            entry.contract.async_behavior,
            BuiltinAsyncBehavior::NeverSuspends
        );
        assert_eq!(
            entry.placement.accelerator,
            BuiltinAcceleratorPolicy::Forbidden
        );
        assert_eq!(
            entry.placement.distributed,
            BuiltinDistributedPolicy::Unsupported
        );
        assert!(entry
            .documentation
            .sections
            .iter()
            .any(|section| section.heading == "Use in a study"));
        assert!(entry.documentation.examples.iter().any(|example| matches!(
            example.verification,
            BuiltinExampleVerification::Assertions { .. }
        )));
    }
}

#[test]
fn valid_calls_return_the_exact_typed_class() {
    let cases: Vec<(&BuiltinCatalogEntry, Vec<ValueFact>, &str)> = vec![
        (
            &r#box::ENTRY,
            vec![double(), double(), double()],
            "geometry.Feature",
        ),
        (
            &cylinder::ENTRY,
            vec![double(), double(), vector3(), vector3()],
            "geometry.Feature",
        ),
        (
            &sphere::ENTRY,
            vec![double(), vector3()],
            "geometry.Feature",
        ),
        (
            &transform::ENTRY,
            vec![feature(), matrix4()],
            "geometry.Feature",
        ),
        (
            &instance::ENTRY,
            vec![feature(), matrix4(), matrix4()],
            "geometry.Feature",
        ),
        (
            &fillet::ENTRY,
            vec![feature(), object(GEOMETRY_REGION_OUTPUT_CLASS), double()],
            "geometry.Feature",
        ),
        (
            &union::ENTRY,
            vec![feature(), feature(), feature()],
            "geometry.Feature",
        ),
        (
            &subtract::ENTRY,
            vec![feature(), feature()],
            "geometry.Feature",
        ),
        (&finish::ENTRY, vec![feature()], "geometry.Generated"),
        (&solid::ENTRY, vec![feature()], "geometry.RegionOutput"),
        (
            &faces_introduced_by::ENTRY,
            vec![feature()],
            "geometry.RegionOutput",
        ),
        (
            &with_region::ENTRY,
            vec![
                object(GEOMETRY_GENERATED_CLASS),
                ValueFact::scalar(ValueKindFact::String),
                object(GEOMETRY_REGION_OUTPUT_CLASS),
            ],
            "geometry.Generated",
        ),
    ];
    for (entry, arguments, class) in cases {
        let request = request(arguments);
        assert!(
            codes(entry, &request).is_empty(),
            "{}: {:?}",
            entry.identity.name,
            codes(entry, &request)
        );
        assert_eq!(
            result_class(entry, &request).as_deref(),
            Some(class),
            "{}",
            entry.identity.name
        );
    }
}

#[test]
fn wrong_arity_reports_a_stable_code() {
    assert!(codes(&r#box::ENTRY, &request(vec![double(), double()]))
        .contains(&"RM-CATALOG-GEOMETRY-BOX-ARITY".to_string()));
    assert!(codes(&union::ENTRY, &request(vec![feature()]))
        .contains(&"RM-CATALOG-GEOMETRY-UNION-ARITY".to_string()));
    assert!(codes(&subtract::ENTRY, &request(vec![feature()]))
        .contains(&"RM-CATALOG-GEOMETRY-SUBTRACT-ARITY".to_string()));
    assert!(codes(&instance::ENTRY, &request(vec![feature()]))
        .contains(&"RM-CATALOG-GEOMETRY-INSTANCE-ARITY".to_string()));
    assert!(codes(&finish::ENTRY, &request(vec![feature(), feature()]))
        .contains(&"RM-CATALOG-GEOMETRY-FINISH-ARITY".to_string()));
}

#[test]
fn wrong_kinds_are_rejected_where_statically_known() {
    let text = ValueFact::scalar(ValueKindFact::String);
    assert!(codes(
        &r#box::ENTRY,
        &request(vec![text.clone(), double(), double()])
    )
    .contains(&"RM-CATALOG-GEOMETRY-BOX-SCALAR".to_string()));
    let logical = ValueFact::scalar(ValueKindFact::Logical);
    assert!(
        codes(&r#box::ENTRY, &request(vec![double(), logical, double()]))
            .contains(&"RM-CATALOG-GEOMETRY-BOX-SCALAR".to_string())
    );
    let two = numeric(NumericClass::Double, vec![Some(1), Some(2)]);
    assert!(codes(&sphere::ENTRY, &request(vec![double(), two]))
        .contains(&"RM-CATALOG-GEOMETRY-SPHERE-VECTOR".to_string()));
    let integer_axis = numeric(NumericClass::Int32, vec![Some(1), Some(3)]);
    assert!(codes(
        &cylinder::ENTRY,
        &request(vec![double(), double(), integer_axis, vector3()])
    )
    .contains(&"RM-CATALOG-GEOMETRY-CYLINDER-VECTOR".to_string()));
    assert!(
        codes(&transform::ENTRY, &request(vec![feature(), vector3()]))
            .contains(&"RM-CATALOG-GEOMETRY-TRANSFORM-MATRIX".to_string())
    );
    assert!(codes(
        &fillet::ENTRY,
        &request(vec![feature(), feature(), double()])
    )
    .contains(&"RM-CATALOG-GEOMETRY-FILLET-OBJECT".to_string()));
    assert!(codes(&finish::ENTRY, &request(vec![double()]))
        .contains(&"RM-CATALOG-GEOMETRY-FINISH-OBJECT".to_string()));
    assert!(codes(
        &with_region::ENTRY,
        &request(vec![
            feature(),
            text.clone(),
            object(GEOMETRY_REGION_OUTPUT_CLASS)
        ])
    )
    .contains(&"RM-CATALOG-GEOMETRY-WITH_REGION-OBJECT".to_string()));
    assert!(codes(
        &with_region::ENTRY,
        &request(vec![
            object(GEOMETRY_GENERATED_CLASS),
            double(),
            object(GEOMETRY_REGION_OUTPUT_CLASS)
        ])
    )
    .contains(&"RM-CATALOG-GEOMETRY-WITH_REGION-TEXT".to_string()));
}

#[test]
fn integer_class_scalars_are_admitted() {
    let integer = numeric(NumericClass::Int32, vec![Some(1), Some(1)]);
    assert!(codes(
        &r#box::ENTRY,
        &request(vec![integer.clone(), integer.clone(), integer])
    )
    .is_empty());
}

#[test]
fn literal_nonpositive_dimensions_and_zero_directions_are_rejected() {
    let box_request = with_literals(
        vec![double(), double(), double()],
        vec![
            LiteralValue::Number(0.08),
            LiteralValue::Number(0.0),
            LiteralValue::Number(0.008),
        ],
    );
    assert!(codes(&r#box::ENTRY, &box_request)
        .contains(&"RM-CATALOG-GEOMETRY-BOX-NONPOSITIVE".to_string()));
    let zero = LiteralValue::Vector(vec![
        LiteralValue::Number(0.0),
        LiteralValue::Number(0.0),
        LiteralValue::Number(0.0),
    ]);
    let cylinder_request = with_literals(
        vec![double(), double(), vector3(), vector3()],
        vec![
            LiteralValue::Number(0.005),
            LiteralValue::Number(0.008),
            zero.clone(),
            zero,
        ],
    );
    let found = codes(&cylinder::ENTRY, &cylinder_request);
    // The axis is a direction; the origin is a point and may be zero.
    assert_eq!(
        found,
        vec!["RM-CATALOG-GEOMETRY-CYLINDER-DIRECTION".to_string()]
    );
}

#[test]
fn unknown_inputs_do_not_error() {
    for entry in ENTRIES {
        let minimum = match entry.identity.name {
            "geometry.box" => 3,
            "geometry.cylinder" | "geometry.revolve" | "geometry.pocket" => 4,
            "geometry.hole" => 5,
            "geometry.sphere" | "geometry.transform" | "geometry.instance" | "geometry.union"
            | "geometry.intersect" | "geometry.subtract" => 2,
            "geometry.extrude" | "geometry.fillet" | "geometry.chamfer" | "geometry.withRegion" => {
                3
            }
            _ => 1,
        };
        let request = request(vec![unknown(); minimum]);
        assert!(
            codes(entry, &request).is_empty(),
            "{}: {:?}",
            entry.identity.name,
            codes(entry, &request)
        );
    }
}
