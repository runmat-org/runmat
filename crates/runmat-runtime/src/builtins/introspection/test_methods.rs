//! Test-only class/method builtins used by semantic and VM pipeline tests.

use crate::{
    runtime_descriptor_error, runtime_descriptor_error_with_detail, OBJECT_INDEX_BRACE,
    OBJECT_INDEX_MEMBER, OBJECT_INDEX_PAREN,
};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};

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
        Value::Object(mut o) => {
            let mut x = 0.0;
            let mut y = 0.0;
            if let Some(Value::Num(v)) = o.properties.get("x") {
                x = *v;
            }
            if let Some(Value::Num(v)) = o.properties.get("y") {
                y = *v;
            }
            o.properties.insert("x".to_string(), Value::Num(x + dx));
            o.properties.insert("y".to_string(), Value::Num(y + dy));
            Ok(Value::Object(o))
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
    let mut o = runmat_builtins::ObjectInstance::new("Point".to_string());
    o.properties.insert("x".to_string(), Value::Num(0.0));
    o.properties.insert("y".to_string(), Value::Num(0.0));
    Ok(Value::Object(o))
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
            let r = if let Some(Value::Num(v)) = o.properties.get("r") {
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
    let mut o = runmat_builtins::ObjectInstance::new("Ctor".to_string());
    o.properties.insert("x".to_string(), Value::Num(x));
    Ok(Value::Object(o))
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

const OVERIDX_SUBSREF_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Indexed value.",
}];

const OVERIDX_SUBSREF_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "obj",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "OverIdx object.",
    },
    BuiltinParamDescriptor {
        name: "kind",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indexing kind token.",
    },
    BuiltinParamDescriptor {
        name: "payload",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indexing payload.",
    },
];

const OVERIDX_SUBSREF_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.subsref(obj, kind, payload)",
    inputs: &OVERIDX_SUBSREF_INPUTS,
    outputs: &OVERIDX_SUBSREF_OUTPUT,
}];

const OVERIDX_ERROR_SUBSREF_PAYLOAD_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OVERIDX.SUBSREF_PAYLOAD_UNSUPPORTED",
    identifier: Some("RunMat:OverIdxSubsrefPayloadUnsupported"),
    when: "Indexing kind/payload combination is unsupported.",
    message: "OverIdx.subsref: unsupported payload",
};

const OVERIDX_ERROR_SUBSASGN_PAYLOAD_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OVERIDX.SUBSASGN_PAYLOAD_UNSUPPORTED",
    identifier: Some("RunMat:OverIdxSubsasgnPayloadUnsupported"),
    when: "Indexing assignment kind/payload combination is unsupported.",
    message: "OverIdx.subsasgn: unsupported payload",
};

const OVERIDX_ERROR_RECEIVER_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OVERIDX.RECEIVER_INVALID",
    identifier: Some("RunMat:OverIdxReceiverInvalid"),
    when: "Receiver is not an object value.",
    message: "OverIdx: receiver must be object",
};

const OVERIDX_SUBSREF_ERRORS: [BuiltinErrorDescriptor; 2] = [
    OVERIDX_ERROR_SUBSREF_PAYLOAD_UNSUPPORTED,
    OVERIDX_ERROR_RECEIVER_INVALID,
];

pub const OVERIDX_SUBSREF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_SUBSREF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &OVERIDX_SUBSREF_ERRORS,
};

#[runmat_macros::runtime_builtin(
    name = "OverIdx.subsref",
    descriptor(self::OVERIDX_SUBSREF_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::test_methods"
)]
pub(crate) async fn overidx_subsref(
    obj: Value,
    kind: String,
    payload: Value,
) -> crate::BuiltinResult<Value> {
    // Simple sentinel implementation: return different values for '.' vs '()'
    match (obj, kind.as_str(), payload) {
        (Value::Object(_), OBJECT_INDEX_PAREN, Value::Cell(_)) => Ok(Value::Num(99.0)),
        (Value::Object(o), OBJECT_INDEX_BRACE, Value::Cell(_)) => {
            if let Some(v) = o.properties.get("lastCell") {
                Ok(v.clone())
            } else {
                Ok(Value::Num(0.0))
            }
        }
        (Value::Object(o), OBJECT_INDEX_MEMBER, Value::String(field)) => {
            // If field exists, return it; otherwise sentinel 77
            if let Some(v) = o.properties.get(&field) {
                Ok(v.clone())
            } else {
                Ok(Value::Num(77.0))
            }
        }
        (Value::Object(o), OBJECT_INDEX_MEMBER, Value::CharArray(ca)) => {
            let field: String = ca.data.iter().collect();
            if let Some(v) = o.properties.get(&field) {
                Ok(v.clone())
            } else {
                Ok(Value::Num(77.0))
            }
        }
        _ => Err(runtime_descriptor_error(
            "OverIdx.subsref",
            &OVERIDX_ERROR_SUBSREF_PAYLOAD_UNSUPPORTED,
        )),
    }
}

const OVERIDX_SUBSASGN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "obj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Updated object.",
}];

const OVERIDX_SUBSASGN_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "obj",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "OverIdx object.",
    },
    BuiltinParamDescriptor {
        name: "kind",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indexing kind token.",
    },
    BuiltinParamDescriptor {
        name: "payload",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indexing payload.",
    },
    BuiltinParamDescriptor {
        name: "rhs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Assigned value.",
    },
];

const OVERIDX_SUBSASGN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "obj = OverIdx.subsasgn(obj, kind, payload, rhs)",
    inputs: &OVERIDX_SUBSASGN_INPUTS,
    outputs: &OVERIDX_SUBSASGN_OUTPUT,
}];

const OVERIDX_SUBSASGN_ERRORS: [BuiltinErrorDescriptor; 2] = [
    OVERIDX_ERROR_SUBSASGN_PAYLOAD_UNSUPPORTED,
    OVERIDX_ERROR_RECEIVER_INVALID,
];

pub const OVERIDX_SUBSASGN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_SUBSASGN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &OVERIDX_SUBSASGN_ERRORS,
};

#[runmat_macros::runtime_builtin(
    name = "OverIdx.subsasgn",
    descriptor(self::OVERIDX_SUBSASGN_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::test_methods"
)]
pub(crate) async fn overidx_subsasgn(
    mut obj: Value,
    kind: String,
    payload: Value,
    rhs: Value,
) -> crate::BuiltinResult<Value> {
    match (&mut obj, kind.as_str(), payload) {
        (Value::Object(o), OBJECT_INDEX_PAREN, Value::Cell(_)) => {
            // Store into 'last' property
            o.properties.insert("last".to_string(), rhs);
            Ok(Value::Object(o.clone()))
        }
        (Value::Object(o), OBJECT_INDEX_BRACE, Value::Cell(_)) => {
            o.properties.insert("lastCell".to_string(), rhs);
            Ok(Value::Object(o.clone()))
        }
        (Value::Object(o), OBJECT_INDEX_MEMBER, Value::String(field)) => {
            o.properties.insert(field, rhs);
            Ok(Value::Object(o.clone()))
        }
        (Value::Object(o), OBJECT_INDEX_MEMBER, Value::CharArray(ca)) => {
            let field: String = ca.data.iter().collect();
            o.properties.insert(field, rhs);
            Ok(Value::Object(o.clone()))
        }
        _ => Err(runtime_descriptor_error(
            "OverIdx.subsasgn",
            &OVERIDX_ERROR_SUBSASGN_PAYLOAD_UNSUPPORTED,
        )),
    }
}

fn overidx_expect_object(
    obj: Value,
    method: &str,
) -> crate::BuiltinResult<runmat_builtins::ObjectInstance> {
    match obj {
        Value::Object(o) => Ok(o),
        other => Err(runtime_descriptor_error_with_detail(
            "OverIdx",
            &OVERIDX_ERROR_RECEIVER_INVALID,
            format!("{method}: got {other:?}"),
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
    let o = overidx_expect_object(obj, "OverIdx.plus")?;
    let k = if let Some(Value::Num(v)) = o.properties.get("k") {
        *v
    } else {
        0.0
    };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(k + r))
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.times",
    descriptor(self::OVERIDX_TIMES_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::test_methods"
)]
pub(crate) async fn overidx_times(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let o = overidx_expect_object(obj, "OverIdx.times")?;
    let k = if let Some(Value::Num(v)) = o.properties.get("k") {
        *v
    } else {
        0.0
    };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(k * r))
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.mtimes",
    descriptor(self::OVERIDX_MTIMES_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::test_methods"
)]
pub(crate) async fn overidx_mtimes(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let o = overidx_expect_object(obj, "OverIdx.mtimes")?;
    let k = if let Some(Value::Num(v)) = o.properties.get("k") {
        *v
    } else {
        0.0
    };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(k * r))
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.lt",
    descriptor(self::OVERIDX_LT_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::test_methods"
)]
pub(crate) async fn overidx_lt(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let o = overidx_expect_object(obj, "OverIdx.lt")?;
    let k = if let Some(Value::Num(v)) = o.properties.get("k") {
        *v
    } else {
        0.0
    };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(if k < r { 1.0 } else { 0.0 }))
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.gt",
    descriptor(self::OVERIDX_GT_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::test_methods"
)]
pub(crate) async fn overidx_gt(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let o = overidx_expect_object(obj, "OverIdx.gt")?;
    let k = if let Some(Value::Num(v)) = o.properties.get("k") {
        *v
    } else {
        0.0
    };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(if k > r { 1.0 } else { 0.0 }))
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.eq",
    descriptor(self::OVERIDX_EQ_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::test_methods"
)]
pub(crate) async fn overidx_eq(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let o = overidx_expect_object(obj, "OverIdx.eq")?;
    let k = if let Some(Value::Num(v)) = o.properties.get("k") {
        *v
    } else {
        0.0
    };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(if (k - r).abs() < 1e-12 { 1.0 } else { 0.0 }))
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
    let o = overidx_expect_object(obj, "OverIdx.rdivide")?;
    let k = if let Some(Value::Num(v)) = o.properties.get("k") {
        *v
    } else {
        0.0
    };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(k / r))
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
    let o = overidx_expect_object(obj, "OverIdx.ldivide")?;
    let k = if let Some(Value::Num(v)) = o.properties.get("k") {
        *v
    } else {
        0.0
    };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(r / k))
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
    let o = overidx_expect_object(obj, "OverIdx.and")?;
    let k = if let Some(Value::Num(v)) = o.properties.get("k") {
        *v
    } else {
        0.0
    };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(if (k != 0.0) && (r != 0.0) { 1.0 } else { 0.0 }))
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.or",
    descriptor(self::OVERIDX_OR_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::test_methods"
)]
pub(crate) async fn overidx_or(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let o = overidx_expect_object(obj, "OverIdx.or")?;
    let k = if let Some(Value::Num(v)) = o.properties.get("k") {
        *v
    } else {
        0.0
    };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(if (k != 0.0) || (r != 0.0) { 1.0 } else { 0.0 }))
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.xor",
    descriptor(self::OVERIDX_XOR_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::test_methods"
)]
pub(crate) async fn overidx_xor(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let o = overidx_expect_object(obj, "OverIdx.xor")?;
    let k = if let Some(Value::Num(v)) = o.properties.get("k") {
        *v
    } else {
        0.0
    };
    let r: f64 = (&rhs).try_into()?;
    let a = k != 0.0;
    let b = r != 0.0;
    Ok(Value::Num(if a ^ b { 1.0 } else { 0.0 }))
}
