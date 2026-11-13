use super::*;
use crate::execution::value_codec::{
    decode_inline_value, encode_inline_value, register_immutable_value_class,
    ImmutableValueClassSchema,
};
use serde_json::json;

const CLASS: &str = "runmat.test.NullableProjection";

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Probe {
    steps: Vec<Step>,
    required: u64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    items: Vec<u64>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
enum Step {
    Completed {
        #[serde(with = "runmat_types::wire_integer::option")]
        maximum_resident_bytes: Option<u64>,
        slots: Vec<serde_json::Value>,
        exact: u64,
    },
}

fn probe(limit: Option<u64>) -> Probe {
    Probe {
        steps: vec![Step::Completed {
            maximum_resident_bytes: limit,
            slots: vec![json!(null), json!([null, [], [null, 9007199254740993u64]])],
            exact: u64::MAX,
        }],
        required: 9,
        items: vec![],
    }
}

fn make_object(probe: &Probe) -> ObjectInstance {
    static REGISTER: std::sync::Once = std::sync::Once::new();
    REGISTER.call_once(|| {
        register_immutable_value_class(ImmutableValueClassSchema {
            class_name: CLASS,
            type_identity: "runmat.test.nullable-projection",
            schema_version: 1,
            allowed_fields: &["steps", "required", "items", PROPERTY],
            required_fields: &["steps", "required"],
            validate,
        })
        .unwrap();
    });
    serializable_to_object_value("fea.test", &ERROR_INTERNAL, CLASS, probe).unwrap()
}

#[test]
fn nested_nullable_enum_and_arrays_survive_value_and_immutable_codec_round_trip() {
    for limit in [None, Some(runmat_types::wire_integer::MAX_WIRE_INTEGER)] {
        let expected = probe(limit);
        let object = make_object(&expected);
        assert!(object.property(PROPERTY).is_some());
        let value = Value::Object(object);
        let wire = encode_inline_value(&value).unwrap();
        let cold = decode_inline_value(&wire).unwrap();
        let actual: Probe = super::super::object_payload("fea.test", &cold, CLASS).unwrap();
        assert_eq!(actual, expected);
    }
}

#[test]
fn missing_nonnull_or_originally_some_fields_are_not_repaired_and_wire_stays_strict() {
    let expected = probe(Some(7));
    let mut wire = serde_json::to_value(&expected).unwrap();
    wire.as_object_mut().unwrap().remove("required");
    assert!(serde_json::from_value::<Probe>(wire).is_err());
    let mut projected = serde_json::to_value(&expected).unwrap();
    let shape = Projection::capture(&mut projected).unwrap();
    projected["steps"][0]["Completed"]
        .as_object_mut()
        .unwrap()
        .remove("maximum_resident_bytes");
    shape.restore(&mut projected).unwrap();
    assert!(serde_json::from_value::<Probe>(projected).is_err());

    let mut builder = make_object(&probe(None)).into_unbounded_builder().unwrap();
    builder.remove(PROPERTY);
    let object = builder.finish();
    assert!(
        super::super::object_payload::<Probe>("fea.test", &Value::Object(object), CLASS).is_err()
    );
}

#[test]
fn metadata_is_validated_and_opaque_runtime_properties_stay_forbidden() {
    let mut object = make_object(&probe(None));
    object.replace_property_unbounded(PROPERTY.into(), Value::String("opaque".into()));
    assert!(validate(&object).is_err());
    let mut object = make_object(&probe(None));
    object.replace_property_unbounded(
        "__runmat_old_payload".into(),
        Value::String("opaque".into()),
    );
    assert!(to_json("fea.test", &object).is_err());
    assert!(encode_inline_value(&Value::Object(object)).is_err());
}

#[test]
fn omitted_host_bindings_are_not_evaluated_during_projection_validation_or_decode() {
    let expected = probe(None);
    let mut object = make_object(&expected);
    object.replace_property_unbounded("host_binding".into(), Value::Num(f64::NAN));
    validate(&object).unwrap();
    assert!(to_json("fea.test", &object).is_err());
    let actual: Probe =
        serde_json::from_value(to_json_omitting("fea.test", &object, &["host_binding"]).unwrap())
            .unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn implementation_private_projection_is_constant_with_private_access() {
    ensure_value_classes_registered();
    for name in [
        FEA_RUN_RESULT_CLASS,
        "fea.StructuralExcitation",
        "fea.AcousticExcitation",
        "fea.SignalDomain",
        "fea.StructuralInitialCondition",
        "fea.ReductionPolicy",
        "fea.AnalysisSequence",
        "fea.AccelerationSignal",
        "fea.ForceSignal",
        "fea.VolumeFlowSignal",
    ] {
        let class = crate::class_registry::get_class(ObjectInstance::new(name).class_name())
            .unwrap_or_else(|| panic!("class {name} must be registered"));
        let property = class
            .properties
            .iter()
            .find(|(name, _)| name.display_name() == PROPERTY)
            .unwrap_or_else(|| panic!("class {name} lacks projection declaration"))
            .1;
        assert!(property.is_constant, "{name}");
        assert_eq!(
            property.get_access,
            runmat_types::MemberAccess::Private,
            "{name}"
        );
        assert_eq!(
            property.set_access,
            runmat_types::MemberAccess::Private,
            "{name}"
        );
    }
}

#[test]
fn auxiliary_signal_schema_validates_shape_without_admitting_opaque_payloads() {
    let value = futures::executor::block_on(super::super::signals::force_constant(
        "projection.force".into(),
        Value::Tensor(runmat_value::Tensor::new_2d(vec![1.0, 2.0, 3.0], 1, 3).unwrap()),
    ))
    .unwrap();
    assert_eq!(
        decode_inline_value(&encode_inline_value(&value).unwrap()).unwrap(),
        value
    );
    let Value::Object(mut invalid) = value.clone() else {
        panic!("expected force signal")
    };
    super::super::object_api::replace(
        "fea.test",
        &mut invalid,
        PROPERTY,
        Value::String("opaque data is not shape".into()),
    )
    .unwrap();
    assert!(encode_inline_value(&Value::Object(invalid)).is_err());
    let Value::Object(mut opaque) = value else {
        panic!("expected force signal")
    };
    super::super::object_api::replace(
        "fea.test",
        &mut opaque,
        "__runmat_payload",
        Value::String("forbidden".into()),
    )
    .unwrap();
    assert!(encode_inline_value(&Value::Object(opaque)).is_err());
}
