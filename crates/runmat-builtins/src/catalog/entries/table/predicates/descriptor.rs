use crate::*;

const INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Value whose runtime class is inspected without reading array data.",
}];
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar logical result.",
}];

macro_rules! predicate_descriptor {
    ($signature:ident, $descriptor:ident, $label:literal) => {
        const $signature: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
            label: $label,
            inputs: &INPUT,
            outputs: &OUTPUT,
        }];
        pub const $descriptor: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &$signature,
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &[],
        };
    };
}

predicate_descriptor!(
    ISCATEGORICAL_SIGNATURE,
    ISCATEGORICAL_DESCRIPTOR,
    "tf = iscategorical(A)"
);
predicate_descriptor!(
    ISORDINAL_SIGNATURE,
    ISORDINAL_DESCRIPTOR,
    "tf = isordinal(A)"
);
predicate_descriptor!(ISTABLE_SIGNATURE, ISTABLE_DESCRIPTOR, "tf = istable(A)");
predicate_descriptor!(
    ISTIMETABLE_SIGNATURE,
    ISTIMETABLE_DESCRIPTOR,
    "tf = istimetable(A)"
);
