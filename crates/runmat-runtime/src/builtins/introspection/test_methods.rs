//! Test-only class/method builtins used by semantic and VM pipeline tests.

use crate::execution::RootedValue;
use crate::{runtime_descriptor_error, runtime_descriptor_error_with_detail};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    OVERIDX_ERROR_RECEIVER_INVALID, OVERIDX_ERROR_SUBSASGN_PAYLOAD_UNSUPPORTED,
    OVERIDX_ERROR_SUBSREF_PAYLOAD_UNSUPPORTED,
};
use runmat_value::Value;

const POINT_MOVE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "obj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Updated point object.",
}];

const POINT_MOVE_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "obj",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Point object receiver.",
    },
    BuiltinParamDescriptor {
        name: "dx",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X displacement.",
    },
    BuiltinParamDescriptor {
        name: "dy",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y displacement.",
    },
];

const POINT_MOVE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "obj = Point.move(obj, dx, dy)",
    inputs: &POINT_MOVE_INPUTS,
    outputs: &POINT_MOVE_OUTPUT,
}];

const POINT_MOVE_ERROR_RECEIVER_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POINT_MOVE.RECEIVER_INVALID",
    identifier: Some("RunMat:PointMoveReceiverInvalid"),
    when: "Receiver is not an object value.",
    message: "Point.move: requires object receiver",
};

const POINT_MOVE_ERRORS: [BuiltinErrorDescriptor; 1] = [POINT_MOVE_ERROR_RECEIVER_INVALID];

pub const POINT_MOVE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &POINT_MOVE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &POINT_MOVE_ERRORS,
};

#[runmat_macros::runtime_builtin(
    name = "Point.move",
    descriptor(self::POINT_MOVE_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::test_methods"
)]
pub(crate) async fn point_move_method(obj: Value, dx: f64, dy: f64) -> crate::BuiltinResult<Value> {
    match obj {
        Value::Object(o) => {
            let mut x = 0.0;
            let mut y = 0.0;
            if let Some(Value::Num(v)) = o.property("x") {
                x = *v;
            }
            if let Some(Value::Num(v)) = o.property("y") {
                y = *v;
            }
            let mut o = o.into_unbounded_builder().map_err(|_| {
                runtime_descriptor_error("Point.move", &POINT_MOVE_ERROR_RECEIVER_INVALID)
            })?;
            o.insert("x".to_string(), Value::Num(x + dx));
            o.insert("y".to_string(), Value::Num(y + dy));
            Ok(Value::Object(o.finish()))
        }
        other => Err(runtime_descriptor_error_with_detail(
            "Point.move",
            &POINT_MOVE_ERROR_RECEIVER_INVALID,
            format!("got {other:?}"),
        )),
    }
}

const POINT_ORIGIN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "obj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Origin point object.",
}];

const POINT_ORIGIN_INPUTS: [BuiltinParamDescriptor; 0] = [];

const POINT_ORIGIN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "obj = Point.origin()",
    inputs: &POINT_ORIGIN_INPUTS,
    outputs: &POINT_ORIGIN_OUTPUT,
}];

pub const POINT_ORIGIN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &POINT_ORIGIN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[],
};

#[runmat_macros::runtime_builtin(
    name = "Point.origin",
    descriptor(self::POINT_ORIGIN_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::test_methods"
)]
pub(crate) async fn point_origin_method() -> crate::BuiltinResult<Value> {
    let mut o = runmat_value::UnboundedObjectBuilder::new("Point");
    o.insert("x".to_string(), Value::Num(0.0));
    o.insert("y".to_string(), Value::Num(0.0));
    Ok(Value::Object(o.finish()))
}

const SHAPE_AREA_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "area",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Area value.",
}];

const SHAPE_AREA_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "obj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Shape receiver.",
}];

const SHAPE_AREA_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "area = Shape.area(obj)",
    inputs: &SHAPE_AREA_INPUTS,
    outputs: &SHAPE_AREA_OUTPUT,
}];

pub const SHAPE_AREA_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SHAPE_AREA_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[],
};

#[runmat_macros::runtime_builtin(
    name = "Shape.area",
    descriptor(self::SHAPE_AREA_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::test_methods"
)]
pub(crate) async fn shape_area_method(_obj: Value) -> crate::BuiltinResult<Value> {
    Ok(Value::Num(0.0))
}

const CIRCLE_AREA_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "area",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Area value.",
}];

const CIRCLE_AREA_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "obj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Circle receiver.",
}];

const CIRCLE_AREA_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "area = Circle.area(obj)",
    inputs: &CIRCLE_AREA_INPUTS,
    outputs: &CIRCLE_AREA_OUTPUT,
}];

const CIRCLE_AREA_ERROR_RECEIVER_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CIRCLE_AREA.RECEIVER_INVALID",
    identifier: Some("RunMat:CircleAreaReceiverInvalid"),
    when: "Receiver is not an object value.",
    message: "Circle.area: requires object receiver",
};

const CIRCLE_AREA_ERRORS: [BuiltinErrorDescriptor; 1] = [CIRCLE_AREA_ERROR_RECEIVER_INVALID];

pub const CIRCLE_AREA_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CIRCLE_AREA_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &CIRCLE_AREA_ERRORS,
};

#[runmat_macros::runtime_builtin(
    name = "Circle.area",
    descriptor(self::CIRCLE_AREA_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::test_methods"
)]
pub(crate) async fn circle_area_method(obj: Value) -> crate::BuiltinResult<Value> {
    match obj {
        Value::Object(o) => {
            let r = if let Some(Value::Num(v)) = o.property("r") {
                *v
            } else {
                0.0
            };
            Ok(Value::Num(std::f64::consts::PI * r * r))
        }
        other => Err(runtime_descriptor_error_with_detail(
            "Circle.area",
            &CIRCLE_AREA_ERROR_RECEIVER_INVALID,
            format!("got {other:?}"),
        )),
    }
}

// --- Test-only helpers to validate constructors and subsref/subsasgn ---
const CTOR_CTOR_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "obj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Constructed Ctor object.",
}];

const CTOR_CTOR_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Initial x value.",
}];

const CTOR_CTOR_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "obj = Ctor.Ctor(x)",
    inputs: &CTOR_CTOR_INPUTS,
    outputs: &CTOR_CTOR_OUTPUT,
}];

pub const CTOR_CTOR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CTOR_CTOR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[],
};

#[runmat_macros::runtime_builtin(
    name = "Ctor.Ctor",
    descriptor(self::CTOR_CTOR_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::test_methods"
)]
pub(crate) async fn ctor_ctor_method(x: f64) -> crate::BuiltinResult<Value> {
    // Construct object with property 'x' initialized
    let mut o = runmat_value::UnboundedObjectBuilder::new("Ctor");
    o.insert("x".to_string(), Value::Num(x));
    Ok(Value::Object(o.finish()))
}

// --- Test-only package functions to exercise import precedence ---
const PKGF_FOO_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Constant test value.",
}];

const PKGF_FOO_INPUTS: [BuiltinParamDescriptor; 0] = [];

const PKGF_FOO_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "value = PkgF.foo()",
    inputs: &PKGF_FOO_INPUTS,
    outputs: &PKGF_FOO_OUTPUT,
}];

pub const PKGF_FOO_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PKGF_FOO_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[],
};

#[runmat_macros::runtime_builtin(
    name = "PkgF.foo",
    descriptor(self::PKGF_FOO_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::test_methods"
)]
pub(crate) async fn pkgf_foo() -> crate::BuiltinResult<Value> {
    Ok(Value::Num(10.0))
}

const PKGG_FOO_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Constant test value.",
}];

const PKGG_FOO_INPUTS: [BuiltinParamDescriptor; 0] = [];

const PKGG_FOO_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "value = PkgG.foo()",
    inputs: &PKGG_FOO_INPUTS,
    outputs: &PKGG_FOO_OUTPUT,
}];

pub const PKGG_FOO_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PKGG_FOO_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[],
};

#[runmat_macros::runtime_builtin(
    name = "PkgG.foo",
    descriptor(self::PKGG_FOO_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::test_methods"
)]
pub(crate) async fn pkgg_foo() -> crate::BuiltinResult<Value> {
    Ok(Value::Num(20.0))
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.subsref",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::test_methods"
)]
pub(crate) async fn overidx_subsref(
    obj: Value,
    subscript: Value,
) -> crate::BuiltinResult<RootedValue> {
    let path = crate::object::indexing::parse_standard_substruct(&subscript)?;
    crate::object::protocol::execute_owned_subsref(obj, path, overidx_read_step, None).await
}

async fn overidx_read_step(
    obj: Value,
    step: crate::object::indexing::ObjectSubscript,
) -> crate::BuiltinResult<RootedValue> {
    let payload = step.selector_value()?;
    // Simple sentinel implementation: return different values for '.' vs '()'
    let value = match (obj, step.kind(), payload) {
        (Value::Object(_), crate::object::indexing::ObjectIndexKind::Paren, Value::Cell(_)) => {
            Ok(Value::Num(99.0))
        }
        (Value::Object(o), crate::object::indexing::ObjectIndexKind::Brace, Value::Cell(_)) => {
            if let Some(v) = o.property("lastCell") {
                Ok(v.clone())
            } else {
                Ok(Value::Num(0.0))
            }
        }
        (
            Value::Object(o),
            crate::object::indexing::ObjectIndexKind::Member,
            Value::String(field),
        ) => {
            // If field exists, return it; otherwise sentinel 77
            if let Some(v) = o.property(field.as_str()) {
                Ok(v.clone())
            } else {
                Ok(Value::Num(77.0))
            }
        }
        (
            Value::Object(o),
            crate::object::indexing::ObjectIndexKind::Member,
            Value::CharArray(ca),
        ) => {
            let field: String = ca.data().iter().collect();
            if let Some(v) = o.property(&field) {
                Ok(v.clone())
            } else {
                Ok(Value::Num(77.0))
            }
        }
        _ => Err(runtime_descriptor_error(
            "OverIdx.subsref",
            &OVERIDX_ERROR_SUBSREF_PAYLOAD_UNSUPPORTED,
        )),
    }?;
    RootedValue::new(value).map_err(|error| error.into_runtime_error())
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.subsasgn",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::test_methods"
)]
pub(crate) async fn overidx_subsasgn(
    obj: Value,
    subscript: Value,
    rhs: Value,
) -> crate::BuiltinResult<RootedValue> {
    let path = crate::object::indexing::parse_standard_substruct(&subscript)?;
    crate::object::protocol::execute_owned_subsasgn(
        obj,
        path,
        vec![rhs],
        overidx_read_step,
        |obj, step, mut values| async move {
            let rhs = values.pop().ok_or_else(|| {
                runtime_descriptor_error(
                    "OverIdx.subsasgn",
                    &OVERIDX_ERROR_SUBSASGN_PAYLOAD_UNSUPPORTED,
                )
            })?;
            overidx_write_step(obj, step, rhs).await
        },
        None,
    )
    .await
}

async fn overidx_write_step(
    obj: Value,
    step: crate::object::indexing::ObjectSubscript,
    rhs: Value,
) -> crate::BuiltinResult<RootedValue> {
    let payload = step.selector_value()?;
    let Value::Object(object) = obj else {
        return Err(runtime_descriptor_error(
            "OverIdx.subsasgn",
            &OVERIDX_ERROR_SUBSASGN_PAYLOAD_UNSUPPORTED,
        ));
    };
    let mut object = object.into_unbounded_builder().map_err(|_| {
        runtime_descriptor_error(
            "OverIdx.subsasgn",
            &OVERIDX_ERROR_SUBSASGN_PAYLOAD_UNSUPPORTED,
        )
    })?;
    match (step.kind(), payload) {
        (crate::object::indexing::ObjectIndexKind::Paren, Value::Cell(_)) => {
            // Store into 'last' property
            object.insert("last".to_string(), rhs);
            Ok(())
        }
        (crate::object::indexing::ObjectIndexKind::Brace, Value::Cell(_)) => {
            object.insert("lastCell".to_string(), rhs);
            Ok(())
        }
        (crate::object::indexing::ObjectIndexKind::Member, Value::String(field)) => {
            object.insert(field.as_str().to_owned(), rhs);
            Ok(())
        }
        (crate::object::indexing::ObjectIndexKind::Member, Value::CharArray(ca)) => {
            let field: String = ca.data().iter().collect();
            object.insert(field, rhs);
            Ok(())
        }
        _ => Err(runtime_descriptor_error(
            "OverIdx.subsasgn",
            &OVERIDX_ERROR_SUBSASGN_PAYLOAD_UNSUPPORTED,
        )),
    }?;
    RootedValue::new(Value::Object(object.finish())).map_err(|error| error.into_runtime_error())
}

const OVERIDX_NUM_ARGUMENTS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Expected indexing argument count.",
}];

const OVERIDX_NUM_ARGUMENTS_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "obj",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "OverIdx object.",
    },
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Substruct-compatible indexing descriptor.",
    },
    BuiltinParamDescriptor {
        name: "indexingContext",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indexing context.",
    },
];

const OVERIDX_NUM_ARGUMENTS_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "n = OverIdx.numArgumentsFromSubscript(obj, S, indexingContext)",
        inputs: &OVERIDX_NUM_ARGUMENTS_INPUTS,
        outputs: &OVERIDX_NUM_ARGUMENTS_OUTPUT,
    }];

pub const OVERIDX_NUM_ARGUMENTS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_NUM_ARGUMENTS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[],
};

#[runmat_macros::runtime_builtin(
    name = "OverIdx.numArgumentsFromSubscript",
    descriptor(self::OVERIDX_NUM_ARGUMENTS_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::test_methods"
)]
pub(crate) async fn overidx_num_arguments_from_subscript(
    obj: Value,
    _subscript: Value,
    _indexing_context: Value,
) -> crate::BuiltinResult<Value> {
    let object = overidx_expect_object(obj, "OverIdx.numArgumentsFromSubscript")?;
    Ok(object.property("nargs").cloned().unwrap_or(Value::Num(3.0)))
}

const OVERIDX_SAVEOBJ_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "payload",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Serialized OverIdx payload.",
}];

const OVERIDX_SAVEOBJ_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "obj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "OverIdx object receiver.",
}];

const OVERIDX_SAVEOBJ_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "payload = OverIdx.saveobj(obj)",
    inputs: &OVERIDX_SAVEOBJ_INPUTS,
    outputs: &OVERIDX_SAVEOBJ_OUTPUT,
}];

pub const OVERIDX_SAVEOBJ_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_SAVEOBJ_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[],
};

#[runmat_macros::runtime_builtin(
    name = "OverIdx.saveobj",
    descriptor(self::OVERIDX_SAVEOBJ_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::test_methods"
)]
pub(crate) async fn overidx_saveobj(obj: Value) -> crate::BuiltinResult<Value> {
    let object = overidx_expect_object(obj, "OverIdx.saveobj")?;
    let mut payload = runmat_value::UnboundedStructValueBuilder::new();
    for key in ["k", "nargs"] {
        if let Some(value) = object.property(key) {
            payload.insert(key.to_string(), value.clone());
        }
    }
    payload.insert(
        "saved_by".to_string(),
        Value::string("OverIdx.saveobj".to_string()),
    );
    Ok(Value::Struct(payload.finish()))
}

const OVERIDX_LOADOBJ_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "obj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Restored OverIdx object.",
}];

const OVERIDX_LOADOBJ_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "payload",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Serialized OverIdx payload.",
}];

const OVERIDX_LOADOBJ_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "obj = OverIdx.loadobj(payload)",
    inputs: &OVERIDX_LOADOBJ_INPUTS,
    outputs: &OVERIDX_LOADOBJ_OUTPUT,
}];

pub const OVERIDX_LOADOBJ_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_LOADOBJ_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[],
};

#[runmat_macros::runtime_builtin(
    name = "OverIdx.loadobj",
    descriptor(self::OVERIDX_LOADOBJ_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::test_methods"
)]
pub(crate) async fn overidx_loadobj(payload: Value) -> crate::BuiltinResult<Value> {
    let mut object = runmat_value::UnboundedObjectBuilder::new("OverIdx");
    if let Value::Struct(st) = payload {
        for (name, value) in st.into_fields_unbounded() {
            if name != "saved_by" {
                object.insert(name, value);
            }
        }
    }
    object.insert(
        "loaded_by".to_string(),
        Value::string("OverIdx.loadobj".to_string()),
    );
    Ok(Value::Object(object.finish()))
}

fn overidx_expect_object(
    obj: Value,
    method: &str,
) -> crate::BuiltinResult<runmat_value::ObjectInstance> {
    match obj {
        Value::Object(o) => Ok(o),
        other => Err(runtime_descriptor_error_with_detail(
            "OverIdx",
            &OVERIDX_ERROR_RECEIVER_INVALID,
            format!("{method}: got {other:?}"),
        )),
    }
}

fn overidx_binary_operands(
    lhs: Value,
    rhs: Value,
    method: &str,
) -> crate::BuiltinResult<(f64, f64)> {
    fn object_value(object: runmat_value::ObjectInstance) -> f64 {
        match object.property("k") {
            Some(Value::Num(value)) => *value,
            _ => 0.0,
        }
    }

    match (lhs, rhs) {
        (Value::Object(object), rhs) => {
            let rhs: f64 = (&rhs).try_into()?;
            Ok((object_value(object), rhs))
        }
        (lhs, Value::Object(object)) => {
            let lhs: f64 = (&lhs).try_into()?;
            Ok((lhs, object_value(object)))
        }
        (lhs, _) => Err(runtime_descriptor_error_with_detail(
            "OverIdx",
            &OVERIDX_ERROR_RECEIVER_INVALID,
            format!("{method}: got {lhs:?}"),
        )),
    }
}

// --- Operator overloading methods for OverIdx (test scaffolding) ---
const OVERIDX_BINARY_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Operator result value.",
}];

const OVERIDX_BINARY_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "obj",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "OverIdx object receiver.",
    },
    BuiltinParamDescriptor {
        name: "rhs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Right-hand operand.",
    },
];

const OVERIDX_UNARY_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Operator result value.",
}];

const OVERIDX_UNARY_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "obj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "OverIdx object receiver.",
}];

const OVERIDX_PLUS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.plus(obj, rhs)",
    inputs: &OVERIDX_BINARY_INPUTS,
    outputs: &OVERIDX_BINARY_OUTPUT,
}];

const OVERIDX_TIMES_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.times(obj, rhs)",
    inputs: &OVERIDX_BINARY_INPUTS,
    outputs: &OVERIDX_BINARY_OUTPUT,
}];

const OVERIDX_MTIMES_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.mtimes(obj, rhs)",
    inputs: &OVERIDX_BINARY_INPUTS,
    outputs: &OVERIDX_BINARY_OUTPUT,
}];

const OVERIDX_LT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.lt(obj, rhs)",
    inputs: &OVERIDX_BINARY_INPUTS,
    outputs: &OVERIDX_BINARY_OUTPUT,
}];

const OVERIDX_GT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.gt(obj, rhs)",
    inputs: &OVERIDX_BINARY_INPUTS,
    outputs: &OVERIDX_BINARY_OUTPUT,
}];

const OVERIDX_EQ_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.eq(obj, rhs)",
    inputs: &OVERIDX_BINARY_INPUTS,
    outputs: &OVERIDX_BINARY_OUTPUT,
}];

const OVERIDX_UPLUS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.uplus(obj)",
    inputs: &OVERIDX_UNARY_INPUTS,
    outputs: &OVERIDX_UNARY_OUTPUT,
}];

const OVERIDX_RDIVIDE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.rdivide(obj, rhs)",
    inputs: &OVERIDX_BINARY_INPUTS,
    outputs: &OVERIDX_BINARY_OUTPUT,
}];

const OVERIDX_MRDIVIDE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.mrdivide(obj, rhs)",
    inputs: &OVERIDX_BINARY_INPUTS,
    outputs: &OVERIDX_BINARY_OUTPUT,
}];

const OVERIDX_LDIVIDE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.ldivide(obj, rhs)",
    inputs: &OVERIDX_BINARY_INPUTS,
    outputs: &OVERIDX_BINARY_OUTPUT,
}];

const OVERIDX_MLDIVIDE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.mldivide(obj, rhs)",
    inputs: &OVERIDX_BINARY_INPUTS,
    outputs: &OVERIDX_BINARY_OUTPUT,
}];

const OVERIDX_AND_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.and(obj, rhs)",
    inputs: &OVERIDX_BINARY_INPUTS,
    outputs: &OVERIDX_BINARY_OUTPUT,
}];

const OVERIDX_OR_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.or(obj, rhs)",
    inputs: &OVERIDX_BINARY_INPUTS,
    outputs: &OVERIDX_BINARY_OUTPUT,
}];

const OVERIDX_XOR_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.xor(obj, rhs)",
    inputs: &OVERIDX_BINARY_INPUTS,
    outputs: &OVERIDX_BINARY_OUTPUT,
}];

const OVERIDX_OPERATOR_ERRORS: [BuiltinErrorDescriptor; 1] = [OVERIDX_ERROR_RECEIVER_INVALID];

pub const OVERIDX_PLUS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_PLUS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &OVERIDX_OPERATOR_ERRORS,
};

pub const OVERIDX_TIMES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_TIMES_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &OVERIDX_OPERATOR_ERRORS,
};

pub const OVERIDX_MTIMES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_MTIMES_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &OVERIDX_OPERATOR_ERRORS,
};

pub const OVERIDX_LT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_LT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &OVERIDX_OPERATOR_ERRORS,
};

pub const OVERIDX_GT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_GT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &OVERIDX_OPERATOR_ERRORS,
};

pub const OVERIDX_EQ_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_EQ_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &OVERIDX_OPERATOR_ERRORS,
};

pub const OVERIDX_UPLUS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_UPLUS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &OVERIDX_OPERATOR_ERRORS,
};

pub const OVERIDX_RDIVIDE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_RDIVIDE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &OVERIDX_OPERATOR_ERRORS,
};

pub const OVERIDX_MRDIVIDE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_MRDIVIDE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &OVERIDX_OPERATOR_ERRORS,
};

pub const OVERIDX_LDIVIDE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_LDIVIDE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &OVERIDX_OPERATOR_ERRORS,
};

pub const OVERIDX_MLDIVIDE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_MLDIVIDE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &OVERIDX_OPERATOR_ERRORS,
};

pub const OVERIDX_AND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_AND_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &OVERIDX_OPERATOR_ERRORS,
};

pub const OVERIDX_OR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_OR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &OVERIDX_OPERATOR_ERRORS,
};

pub const OVERIDX_XOR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_XOR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &OVERIDX_OPERATOR_ERRORS,
};

#[runmat_macros::runtime_builtin(
    name = "OverIdx.plus",
    descriptor(self::OVERIDX_PLUS_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::test_methods"
)]
pub(crate) async fn overidx_plus(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let (lhs, rhs) = overidx_binary_operands(obj, rhs, "OverIdx.plus")?;
    Ok(Value::Num(lhs + rhs))
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.times",
    descriptor(self::OVERIDX_TIMES_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::test_methods"
)]
pub(crate) async fn overidx_times(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let (lhs, rhs) = overidx_binary_operands(obj, rhs, "OverIdx.times")?;
    Ok(Value::Num(lhs * rhs))
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.mtimes",
    descriptor(self::OVERIDX_MTIMES_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::test_methods"
)]
pub(crate) async fn overidx_mtimes(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let (lhs, rhs) = overidx_binary_operands(obj, rhs, "OverIdx.mtimes")?;
    Ok(Value::Num(lhs * rhs))
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.lt",
    descriptor(self::OVERIDX_LT_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::test_methods"
)]
pub(crate) async fn overidx_lt(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let (lhs, rhs) = overidx_binary_operands(obj, rhs, "OverIdx.lt")?;
    Ok(Value::Num(if lhs < rhs { 1.0 } else { 0.0 }))
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.gt",
    descriptor(self::OVERIDX_GT_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::test_methods"
)]
pub(crate) async fn overidx_gt(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let (lhs, rhs) = overidx_binary_operands(obj, rhs, "OverIdx.gt")?;
    Ok(Value::Num(if lhs > rhs { 1.0 } else { 0.0 }))
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.eq",
    descriptor(self::OVERIDX_EQ_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::test_methods"
)]
pub(crate) async fn overidx_eq(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let (lhs, rhs) = overidx_binary_operands(obj, rhs, "OverIdx.eq")?;
    Ok(Value::Num(if (lhs - rhs).abs() < 1e-12 {
        1.0
    } else {
        0.0
    }))
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.uplus",
    descriptor(self::OVERIDX_UPLUS_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::test_methods"
)]
pub(crate) async fn overidx_uplus(obj: Value) -> crate::BuiltinResult<Value> {
    // Identity
    Ok(obj)
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.rdivide",
    descriptor(self::OVERIDX_RDIVIDE_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::test_methods"
)]
pub(crate) async fn overidx_rdivide(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let (lhs, rhs) = overidx_binary_operands(obj, rhs, "OverIdx.rdivide")?;
    Ok(Value::Num(lhs / rhs))
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.mrdivide",
    descriptor(self::OVERIDX_MRDIVIDE_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::test_methods"
)]
pub(crate) async fn overidx_mrdivide(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    overidx_rdivide(obj, rhs).await
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.ldivide",
    descriptor(self::OVERIDX_LDIVIDE_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::test_methods"
)]
pub(crate) async fn overidx_ldivide(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let (lhs, rhs) = overidx_binary_operands(obj, rhs, "OverIdx.ldivide")?;
    Ok(Value::Num(rhs / lhs))
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.mldivide",
    descriptor(self::OVERIDX_MLDIVIDE_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::test_methods"
)]
pub(crate) async fn overidx_mldivide(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    overidx_ldivide(obj, rhs).await
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.and",
    descriptor(self::OVERIDX_AND_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::test_methods"
)]
pub(crate) async fn overidx_and(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let (lhs, rhs) = overidx_binary_operands(obj, rhs, "OverIdx.and")?;
    Ok(Value::Num(if (lhs != 0.0) && (rhs != 0.0) {
        1.0
    } else {
        0.0
    }))
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.or",
    descriptor(self::OVERIDX_OR_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::test_methods"
)]
pub(crate) async fn overidx_or(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let (lhs, rhs) = overidx_binary_operands(obj, rhs, "OverIdx.or")?;
    Ok(Value::Num(if (lhs != 0.0) || (rhs != 0.0) {
        1.0
    } else {
        0.0
    }))
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.xor",
    descriptor(self::OVERIDX_XOR_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::test_methods"
)]
pub(crate) async fn overidx_xor(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let (lhs, rhs) = overidx_binary_operands(obj, rhs, "OverIdx.xor")?;
    let a = lhs != 0.0;
    let b = rhs != 0.0;
    Ok(Value::Num(if a ^ b { 1.0 } else { 0.0 }))
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod rooted_fixture_tests {
    use super::*;
    use std::cell::RefCell;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    thread_local! {
        static OBSERVED_HANDLE: RefCell<Option<runmat_gc::GcHandle>> = const { RefCell::new(None) };
    }

    #[test]
    fn managed_property_survives_producer_root_retirement_until_fixture_result_drops() {
        runmat_gc::gc_test_context(|| {
            let initial = runmat_gc::gc_allocate_rooted(Value::String("fixture payload".into()))
                .expect("allocate managed test property");
            let handle = initial.handle();
            let mut object = runmat_value::UnboundedObjectBuilder::new("OverIdx");
            object.insert(
                "last".into(),
                Value::HandleObject(runmat_value::HandleRef {
                    class_name: "FixturePayload".into(),
                    target: handle,
                    valid: true,
                }),
            );
            let subscript = crate::object::indexing::standard_substruct_fixture_from_parts(
                ".",
                Value::String("last".into()),
            )
            .expect("member path");
            let output = futures::executor::block_on(overidx_subsref(
                Value::Object(object.finish()),
                subscript,
            ))
            .expect("fixture member read");
            OBSERVED_HANDLE.with(|slot| *slot.borrow_mut() = Some(handle));
            let seen = Arc::new(AtomicBool::new(false));
            let observed = Arc::clone(&seen);
            let _observer = runmat_gc::RootRetirementObserver::register(Arc::new(move |event| {
                OBSERVED_HANDLE.with(|slot| {
                    let Some(handle) = *slot.borrow() else {
                        return;
                    };
                    if event != runmat_gc::RootRetirement::Explicit(handle) {
                        return;
                    }
                    runmat_gc::gc_collect_major().expect("collect at producer retirement");
                    assert!(runmat_gc::gc_clone_value(&handle).is_ok());
                    observed.store(true, Ordering::SeqCst);
                });
            }))
            .expect("observe producer retirement");
            initial.unroot().expect("retire producer root");
            assert!(seen.load(Ordering::SeqCst));
            assert_eq!(
                runmat_gc::gc_clone_value(&handle).expect("fixture result retains property"),
                Value::String("fixture payload".into())
            );
            drop(_observer);
            OBSERVED_HANDLE.with(|slot| *slot.borrow_mut() = None);
            drop(output);
            runmat_gc::gc_collect_major().expect("collect after result drop");
            assert!(runmat_gc::gc_clone_value(&handle).is_err());
        });
    }
}
