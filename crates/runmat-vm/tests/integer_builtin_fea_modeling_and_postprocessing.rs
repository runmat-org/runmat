#[path = "support/mod.rs"]
mod test_helpers;

use runmat_builtins::{
    BuiltinIntegerAuditKind, BuiltinIntegerBackendRule, BuiltinIntegerClass, Value,
};
use test_helpers::execute_source;

const INTEGER_CONSTRUCTORS: [&str; 8] = [
    "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
];

const INTEGER_CLASSES: [BuiltinIntegerClass; 8] = [
    BuiltinIntegerClass::Int8,
    BuiltinIntegerClass::Int16,
    BuiltinIntegerClass::Int32,
    BuiltinIntegerClass::Int64,
    BuiltinIntegerClass::Uint8,
    BuiltinIntegerClass::Uint16,
    BuiltinIntegerClass::Uint32,
    BuiltinIntegerClass::Uint64,
];

const FEA_INTEGER_CONSTRUCTORS: [&str; 4] = [
    "fea.domain",
    "fea.interface",
    "fea.loadCase",
    "fea.material",
];

const FEA_INTEGER_INAPPLICABLE: [&str; 6] = [
    "fea.compare",
    "fea.field",
    "fea.materialAssignment",
    "fea.model",
    "fea.plan",
    "fea.plot",
];

#[test]
fn fea_numeric_constructors_accept_every_host_integer_class() {
    for constructor in INTEGER_CONSTRUCTORS {
        let source = format!(
            r#"
            domain = fea.domain("electromagnetic", ...
                "AppliedCurrentA", {constructor}(2));
            interface = fea.interface("contact", "left", "right", ...
                "FrictionCoefficient", {constructor}(1));
            load = fea.loadCase("pressure", "face", "pressure", ...
                "MagnitudePa", {constructor}(3));
            material = fea.material("steel", ...
                "YoungsModulusPa", {constructor}(200), ...
                "PoissonRatio", {constructor}(0));
            "#
        );
        let values = execute_source(&source)
            .unwrap_or_else(|error| panic!("{constructor} compiled FEA constructors: {error}"));
        for class_name in [
            "fea.Domain",
            "fea.Interface",
            "fea.LoadCase",
            "fea.Material",
        ] {
            assert!(
                values.iter().any(|value| {
                    matches!(value, Value::Object(object) if object.class_name == class_name)
                }),
                "{constructor} must construct {class_name}"
            );
        }
    }
}

#[test]
fn fea_object_and_text_roles_reject_every_integer_class_without_coercion() {
    for constructor in INTEGER_CONSTRUCTORS {
        for source in [
            format!("value = fea.compare({constructor}(1), \"candidate\");"),
            format!("value = fea.field({constructor}(1), \"stress\");"),
            format!("value = fea.materialAssignment({constructor}(1), \"steel\");"),
            format!("value = fea.model({constructor}(1), {constructor}(2));"),
            format!("value = fea.plan({constructor}(1));"),
            format!("value = fea.plot({constructor}(1));"),
        ] {
            let error = execute_source(&source).expect_err("integer object/text role must reject");
            assert_eq!(
                error.identifier(),
                Some("RunMat:fea:InvalidInput"),
                "{source}"
            );
        }
    }
}

#[test]
fn fea_integer_metadata_declares_host_only_native_contracts() {
    for name in FEA_INTEGER_CONSTRUCTORS {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(
            !builtin.integer_capabilities.is_empty(),
            "{name} must declare its integer forms"
        );
        assert!(
            builtin.integer_audit.is_none(),
            "{name} capability contract"
        );
        assert!(builtin.extensions.is_empty(), "{name} is RunMat-native");
        for capability in builtin.integer_capabilities {
            assert_eq!(
                capability.backend,
                BuiltinIntegerBackendRule::HostOnly,
                "{name} {} backend",
                capability.form
            );
            assert!(
                !capability.inputs.is_empty(),
                "{name} {} numeric inputs",
                capability.form
            );
            for input in capability.inputs {
                assert_eq!(
                    input.classes, &INTEGER_CLASSES,
                    "{name} {} {} integer classes",
                    capability.form, input.name
                );
            }
        }
    }

    for name in FEA_INTEGER_INAPPLICABLE {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(
            builtin.integer_capabilities.is_empty(),
            "{name} has no numeric role"
        );
        let audit = builtin
            .integer_audit
            .unwrap_or_else(|| panic!("{name} must carry an integer audit"));
        assert_eq!(audit.kind, BuiltinIntegerAuditKind::NotApplicable, "{name}");
        assert_eq!(audit.canonical_builtin, None, "{name}");
        assert!(builtin.extensions.is_empty(), "{name} is RunMat-native");
    }
}
