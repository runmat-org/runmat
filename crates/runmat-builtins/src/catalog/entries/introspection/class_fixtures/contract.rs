use crate::*;
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
pub const POINT_MOVE_ERROR_RECEIVER_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POINT_MOVE.RECEIVER_INVALID",
    identifier: Some("RunMat:PointMoveReceiverInvalid"),
    when: "Receiver is not an object value.",
    message: "Point.move: requires object receiver",
};
pub const POINT_MOVE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &POINT_MOVE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[
        POINT_MOVE_ERROR_RECEIVER_INVALID,
        CLASS_FIXTURE_ERROR_ARITY,
        CLASS_FIXTURE_ERROR_OUTPUTS,
    ],
};
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
    errors: &[CLASS_FIXTURE_ERROR_ARITY, CLASS_FIXTURE_ERROR_OUTPUTS],
};
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
    errors: &[CLASS_FIXTURE_ERROR_ARITY, CLASS_FIXTURE_ERROR_OUTPUTS],
};
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
pub const CIRCLE_AREA_ERROR_RECEIVER_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CIRCLE_AREA.RECEIVER_INVALID",
    identifier: Some("RunMat:CircleAreaReceiverInvalid"),
    when: "Receiver is not an object value.",
    message: "Circle.area: requires object receiver",
};
pub const CIRCLE_AREA_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CIRCLE_AREA_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[
        CIRCLE_AREA_ERROR_RECEIVER_INVALID,
        CLASS_FIXTURE_ERROR_ARITY,
        CLASS_FIXTURE_ERROR_OUTPUTS,
    ],
};
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
    errors: &[CLASS_FIXTURE_ERROR_ARITY, CLASS_FIXTURE_ERROR_OUTPUTS],
};
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
    errors: &[CLASS_FIXTURE_ERROR_ARITY, CLASS_FIXTURE_ERROR_OUTPUTS],
};
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
    errors: &[CLASS_FIXTURE_ERROR_ARITY, CLASS_FIXTURE_ERROR_OUTPUTS],
};
pub const CLASS_FIXTURE_ERROR_ARITY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIXTURE_METHOD.ARITY",
    identifier: Some("RunMat:FixtureMethod:Arity"),
    when: "Argument count differs from the fixed method signature.",
    message: "fixture method: wrong number of arguments",
};
pub const CLASS_FIXTURE_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIXTURE_METHOD.OUTPUTS",
    identifier: Some("RunMat:TooManyOutputs"),
    when: "More than one result is requested.",
    message: "fixture method: at most one output is supported",
};
