use runmat_builtins::{
    builtin_functions, BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor,
    BuiltinExtensionDescriptor, BuiltinExtensionMode, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;

const TEST_ERRORS: [BuiltinErrorDescriptor; 0] = [];
const FOO_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::IntegerScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input integer.",
}];
const FOO_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::IntegerScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Output integer.",
}];
const FOO_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = foo(x)",
    inputs: &FOO_INPUTS,
    outputs: &FOO_OUTPUTS,
}];
const FOO_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FOO_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &TEST_ERRORS,
};
const FOO_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [BuiltinExtensionDescriptor {
    id: "foo-extra",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "extra foo signature",
    error_identifier: Some("RunMat:test:FooExtension"),
}];

#[runtime_builtin(
    name = "foo",
    descriptor(crate::FOO_DESCRIPTOR),
    extensions(crate::FOO_EXTENSIONS),
    builtin_path = "tests::foo"
)]
fn foo(x: i32) -> Result<i32, String> {
    Ok(x + 1)
}

#[test]
fn works() {
    assert_eq!(foo(1).unwrap(), 2);
    let names: Vec<&str> = builtin_functions().into_iter().map(|b| b.name).collect();
    assert!(names.contains(&"foo"));
    let builtin = builtin_functions()
        .into_iter()
        .find(|builtin| builtin.name == "foo")
        .expect("foo metadata");
    assert_eq!(builtin.extensions, &FOO_EXTENSIONS);
}
