//! RunMat `fieldnames` builtin for struct and gated object introspection.
#[cfg(test)]
use runmat_types::MemberAccess;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::structs::type_resolvers::fieldnames_type;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, CharArray, HandleRef, Listener, ObjectInstance, StructValue, Value};
use std::collections::{BTreeSet, HashSet};

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::structs::core::fieldnames")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fieldnames",
    op_kind: GpuOpKind::Custom("fieldnames"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host-only introspection; providers do not participate.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::structs::core::fieldnames")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fieldnames",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner treats fieldnames as a host inspector; it terminates any pending fusion group.",
};

const BUILTIN_NAME: &str = "fieldnames";

const FIELDNAMES_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "names",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cell array of field names.",
}];

const FIELDNAMES_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Struct, cell-backed struct array, or supported RunMat object input.",
}];

const FIELDNAMES_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "names = fieldnames(S)",
    inputs: &FIELDNAMES_INPUTS,
    outputs: &FIELDNAMES_OUTPUT,
}];

const FIELDNAMES_ERROR_INVALID_TARGET: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIELDNAMES.INVALID_TARGET",
    identifier: Some("RunMat:fieldnames:InvalidTarget"),
    when: "Input is not a struct, cell-backed struct array, or supported RunMat object value.",
    message: "fieldnames: expected struct, struct array, or supported object",
};

const FIELDNAMES_ERROR_STRUCT_ARRAY_CONTENTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIELDNAMES.STRUCT_ARRAY_CONTENTS",
    identifier: Some("RunMat:fieldnames:StructArrayContents"),
    when: "Cell-backed struct-array input contains a non-struct element.",
    message: "fieldnames: expected struct array contents to be structs",
};

const FIELDNAMES_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIELDNAMES.INTERNAL",
    identifier: Some("RunMat:fieldnames:InternalError"),
    when: "Building the output cell array or internal metadata processing fails.",
    message: "fieldnames: internal error",
};

const FIELDNAMES_ERRORS: [BuiltinErrorDescriptor; 3] = [
    FIELDNAMES_ERROR_INVALID_TARGET,
    FIELDNAMES_ERROR_STRUCT_ARRAY_CONTENTS,
    FIELDNAMES_ERROR_INTERNAL,
];

pub const FIELDNAMES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FIELDNAMES_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FIELDNAMES_ERRORS,
};

pub const FIELDNAMES_OBJECT_FAMILY_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fieldnames-object-family",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "fieldnames introspection for RunMat value, handle, and listener objects is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FieldnamesObjectFamilyExtension"),
    };

pub const FIELDNAMES_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [FIELDNAMES_OBJECT_FAMILY_EXTENSION];

pub const FIELDNAMES_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "fieldnames accepts structures and RunMat's cell-backed structure arrays in the compatibility surface. Numeric, logical, and resident arrays are not structure metadata and reject without conversion or provider access; RunMat object-family introspection is independently extension gated.",
    };

fn fieldnames_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "fieldnames",
    category = "structs/core",
    summary = "List struct or object field names.",
    keywords = "fieldnames,struct,introspection,fields",
    type_resolver(fieldnames_type),
    descriptor(crate::builtins::structs::core::fieldnames::FIELDNAMES_DESCRIPTOR),
    extensions(crate::builtins::structs::core::fieldnames::FIELDNAMES_EXTENSIONS),
    integer_audit(crate::builtins::structs::core::fieldnames::FIELDNAMES_INTEGER_AUDIT),
    builtin_path = "crate::builtins::structs::core::fieldnames"
)]
async fn fieldnames_builtin(value: Value) -> BuiltinResult<Value> {
    if matches!(
        &value,
        Value::Object(_) | Value::HandleObject(_) | Value::Listener(_)
    ) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FIELDNAMES_OBJECT_FAMILY_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let names = match &value {
        Value::Struct(st) => collect_struct_fieldnames(st),
        Value::Cell(cell) => collect_struct_array_fieldnames(cell)?,
        Value::Object(obj) => collect_object_fieldnames(obj),
        Value::HandleObject(handle) => collect_handle_fieldnames(handle)?,
        Value::Listener(listener) => collect_listener_fieldnames(listener),
        other => {
            return Err(fieldnames_error_with_message(
                format!(
                    "{} (got {other:?})",
                    FIELDNAMES_ERROR_INVALID_TARGET.message
                ),
                &FIELDNAMES_ERROR_INVALID_TARGET,
            ))
        }
    };

    let rows = names.len();
    let cells: Vec<Value> = names
        .into_iter()
        .map(|name| Value::CharArray(CharArray::new_row(&name)))
        .collect();
    crate::make_cell(cells, rows, 1).map_err(|e| {
        fieldnames_error_with_message(format!("fieldnames: {e}"), &FIELDNAMES_ERROR_INTERNAL)
    })
}

fn collect_struct_fieldnames(st: &StructValue) -> Vec<String> {
    st.field_names().cloned().collect()
}

fn collect_struct_array_fieldnames(array: &CellArray) -> BuiltinResult<Vec<String>> {
    let mut names = BTreeSet::new();
    for value in &array.data {
        let Value::Struct(st) = value else {
            return Err(fieldnames_error_with_message(
                FIELDNAMES_ERROR_STRUCT_ARRAY_CONTENTS.message,
                &FIELDNAMES_ERROR_STRUCT_ARRAY_CONTENTS,
            ));
        };
        names.extend(st.field_names().cloned());
    }
    Ok(names.into_iter().collect())
}

fn collect_object_fieldnames(obj: &ObjectInstance) -> Vec<String> {
    let mut names = class_instance_property_names(&obj.class_name);
    names.extend(obj.properties.keys().cloned());
    names.into_iter().collect()
}

fn collect_handle_fieldnames(handle: &HandleRef) -> BuiltinResult<Vec<String>> {
    let mut names = class_instance_property_names(&handle.class_name);

    if crate::is_handle_valid(handle) {
        let target = runmat_gc::gc_clone_value(&handle.target).map_err(|e| {
            fieldnames_error_with_message(
                format!("fieldnames: invalid handle target: {e}"),
                &FIELDNAMES_ERROR_INVALID_TARGET,
            )
        })?;
        match &target {
            Value::Struct(st) => {
                names.extend(collect_struct_fieldnames(st));
            }
            Value::Cell(array) => {
                names.extend(collect_struct_array_fieldnames(array)?);
            }
            Value::Object(obj) => {
                names.extend(collect_object_fieldnames(obj));
            }
            Value::Listener(listener) => {
                names.extend(collect_listener_fieldnames(listener));
            }
            Value::HandleObject(other) => {
                names.extend(class_instance_property_names(&other.class_name));
            }
            _ => {}
        }
    }

    Ok(names.into_iter().collect())
}

fn collect_listener_fieldnames(_listener: &Listener) -> Vec<String> {
    let mut names = vec![
        "callback".to_string(),
        "enabled".to_string(),
        "event_name".to_string(),
        "id".to_string(),
        "target".to_string(),
        "valid".to_string(),
    ];
    names.sort();
    names
}

fn class_instance_property_names(class_name: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut current = Some(class_name.to_string());
    let mut visited = HashSet::new();
    while let Some(name) = current {
        if !visited.insert(name.clone()) {
            break;
        }
        let Some(class_def) = crate::class_registry::get_class(&name) else {
            break;
        };
        for (prop_name, prop) in &class_def.properties {
            if !prop.is_static {
                names.insert(prop_name.clone());
            }
        }
        current = class_def.parent.clone();
    }
    names
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use runmat_value::{CellArray, HandleRef, IntValue, ObjectInstance, StructValue, Value};
    use std::collections::HashMap;

    fn error_message(err: crate::RuntimeError) -> String {
        err.message().to_string()
    }

    fn run_fieldnames(value: Value) -> BuiltinResult<Value> {
        futures::executor::block_on(fieldnames_builtin(value))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fieldnames_preserves_scalar_struct_insertion_order() {
        let mut fields = StructValue::new();
        fields.fields.insert("beta".to_string(), Value::Num(1.0));
        fields.fields.insert("alpha".to_string(), Value::Num(2.0));
        let result = run_fieldnames(Value::Struct(fields)).expect("fieldnames");
        let Value::Cell(cell) = result else {
            panic!("expected cell array result");
        };
        assert_eq!(cell.cols, 1);
        assert_eq!(cell.rows, 2);
        let collected = cell_strings(&cell);
        assert_eq!(collected, vec!["beta".to_string(), "alpha".to_string()]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fieldnames_collects_fields_from_cell_backed_struct_arrays() {
        let mut first = StructValue::new();
        first
            .fields
            .insert("name".to_string(), Value::from("Ada".to_string()));
        first.fields.insert("id".to_string(), Value::Num(101.0));

        let mut second = StructValue::new();
        second
            .fields
            .insert("name".to_string(), Value::from("Grace".to_string()));
        second
            .fields
            .insert("department".to_string(), Value::from("Research"));

        let cell = CellArray::new_with_shape(
            vec![Value::Struct(first), Value::Struct(second)],
            vec![1, 2],
        )
        .expect("struct array");

        let Value::Cell(names) = run_fieldnames(Value::Cell(cell)).expect("fieldnames") else {
            panic!("expected cell array result");
        };
        assert_eq!(names.cols, 1);
        assert_eq!(names.rows, 3);
        assert_eq!(
            cell_strings(&names),
            vec![
                "department".to_string(),
                "id".to_string(),
                "name".to_string()
            ]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fieldnames_errors_for_non_struct_inputs() {
        let err = error_message(run_fieldnames(Value::Num(1.0)).unwrap_err());
        assert!(
            err.contains("expected struct, struct array, or supported object"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fieldnames_returns_empty_column_for_empty_cell_backed_struct_array() {
        let empty_array = CellArray::new(Vec::new(), 0, 0).expect("empty struct array backing");
        let Value::Cell(names) = run_fieldnames(Value::Cell(empty_array)).expect("fieldnames")
        else {
            panic!("expected cell array result");
        };
        assert_eq!(names.rows, 0);
        assert_eq!(names.cols, 1);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fieldnames_cell_without_struct_errors() {
        let cell = CellArray::new(vec![Value::Num(1.0)], 1, 1).expect("cell");
        let err = error_message(run_fieldnames(Value::Cell(cell)).unwrap_err());
        assert!(err.contains("expected struct array contents to be structs"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fieldnames_preserves_case_distinctions() {
        let mut fields = StructValue::new();
        fields.fields.insert("name".to_string(), Value::Num(1.0));
        fields.fields.insert("Name".to_string(), Value::Num(2.0));
        let Value::Cell(cell) = run_fieldnames(Value::Struct(fields)).expect("fieldnames") else {
            panic!("expected cell array result");
        };
        let collected = cell_strings(&cell);
        assert_eq!(collected, vec!["name".to_string(), "Name".to_string()]);
    }

    #[test]
    fn fieldnames_integer_audit_is_explicitly_inapplicable() {
        assert_eq!(
            FIELDNAMES_INTEGER_AUDIT.kind,
            BuiltinIntegerAuditKind::NotApplicable
        );
        assert!(FIELDNAMES_INTEGER_AUDIT.canonical_builtin.is_none());
        assert!(FIELDNAMES_INTEGER_AUDIT.notes.contains("Numeric"));
    }

    #[test]
    fn fieldnames_rejects_integer_and_resident_inputs_without_introspection() {
        assert!(run_fieldnames(Value::Int(IntValue::U64(u64::MAX))).is_err());
        let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: u32::MAX,
            buffer_id: u64::MAX,
        });
        assert!(run_fieldnames(resident).is_err());
    }

    #[test]
    fn fieldnames_object_extension_gates_before_introspection() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = run_fieldnames(Value::Object(ObjectInstance::new(
            "runmat.unittest.FieldnamesGated".to_string(),
        )))
        .expect_err("object-family extension");
        assert_eq!(
            err.identifier(),
            FIELDNAMES_OBJECT_FAMILY_EXTENSION.error_identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fieldnames_object_includes_class_and_dynamic_properties() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let class_name = "runmat.unittest.FieldnamesObject";
        let mut def = crate::class_registry::RuntimeClass {
            name: class_name.to_string(),
            parent: None,
            properties: HashMap::new(),
            methods: HashMap::new(),
        };
        def.properties.insert(
            "Value".to_string(),
            crate::class_registry::RuntimeProperty {
                name: "Value".to_string(),
                is_static: false,
                is_constant: false,
                is_dependent: false,
                get_access: MemberAccess::Public,
                set_access: MemberAccess::Public,
                default_value: None,
            },
        );
        def.properties.insert(
            "Version".to_string(),
            crate::class_registry::RuntimeProperty {
                name: "Version".to_string(),
                is_static: true,
                is_constant: false,
                is_dependent: false,
                get_access: MemberAccess::Public,
                set_access: MemberAccess::Public,
                default_value: None,
            },
        );
        crate::class_registry::register_class(def);

        let mut obj = ObjectInstance::new(class_name.to_string());
        obj.properties.insert("Step".to_string(), Value::Num(2.0));

        let Value::Cell(cell) = run_fieldnames(Value::Object(obj)).expect("fieldnames object")
        else {
            panic!("expected cell array");
        };
        let collected = cell_strings(&cell);
        assert_eq!(collected, vec!["Step".to_string(), "Value".to_string()]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fieldnames_object_includes_inherited_class_properties() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let parent_name = "runmat.unittest.FieldnamesParent";
        let child_name = "runmat.unittest.FieldnamesChild";

        let mut parent = crate::class_registry::RuntimeClass {
            name: parent_name.to_string(),
            parent: None,
            properties: HashMap::new(),
            methods: HashMap::new(),
        };
        parent.properties.insert(
            "ParentValue".to_string(),
            crate::class_registry::RuntimeProperty {
                name: "ParentValue".to_string(),
                is_static: false,
                is_constant: false,
                is_dependent: false,
                get_access: MemberAccess::Public,
                set_access: MemberAccess::Public,
                default_value: None,
            },
        );
        crate::class_registry::register_class(parent);

        let mut child = crate::class_registry::RuntimeClass {
            name: child_name.to_string(),
            parent: Some(parent_name.to_string()),
            properties: HashMap::new(),
            methods: HashMap::new(),
        };
        child.properties.insert(
            "ChildValue".to_string(),
            crate::class_registry::RuntimeProperty {
                name: "ChildValue".to_string(),
                is_static: false,
                is_constant: false,
                is_dependent: false,
                get_access: MemberAccess::Public,
                set_access: MemberAccess::Public,
                default_value: None,
            },
        );
        crate::class_registry::register_class(child);

        let obj = ObjectInstance::new(child_name.to_string());
        let Value::Cell(cell) = run_fieldnames(Value::Object(obj)).expect("fieldnames object")
        else {
            panic!("expected cell array");
        };
        let collected = cell_strings(&cell);
        assert_eq!(
            collected,
            vec!["ChildValue".to_string(), "ParentValue".to_string()]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fieldnames_handle_object_merges_class_and_target() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let class_name = "runmat.unittest.FieldnamesHandle";
        let mut def = crate::class_registry::RuntimeClass {
            name: class_name.to_string(),
            parent: None,
            properties: HashMap::new(),
            methods: HashMap::new(),
        };
        def.properties.insert(
            "Enabled".to_string(),
            crate::class_registry::RuntimeProperty {
                name: "Enabled".to_string(),
                is_static: false,
                is_constant: false,
                is_dependent: false,
                get_access: MemberAccess::Public,
                set_access: MemberAccess::Public,
                default_value: None,
            },
        );
        crate::class_registry::register_class(def);

        let mut payload = ObjectInstance::new(class_name.to_string());
        payload
            .properties
            .insert("Status".to_string(), Value::from("ready"));
        let target = runmat_gc::gc_allocate(Value::Object(payload)).expect("gc allocate target");

        let handle = HandleRef {
            class_name: class_name.to_string(),
            target,
            valid: true,
        };

        let Value::Cell(cell) =
            run_fieldnames(Value::HandleObject(handle)).expect("fieldnames handle")
        else {
            panic!("expected cell array");
        };
        let collected = cell_strings(&cell);
        assert_eq!(collected, vec!["Enabled".to_string(), "Status".to_string()]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fieldnames_handle_object_includes_inherited_class_properties() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let parent_name = "runmat.unittest.FieldnamesHandleParent";
        let child_name = "runmat.unittest.FieldnamesHandleChild";

        let mut parent = crate::class_registry::RuntimeClass {
            name: parent_name.to_string(),
            parent: None,
            properties: HashMap::new(),
            methods: HashMap::new(),
        };
        parent.properties.insert(
            "ParentEnabled".to_string(),
            crate::class_registry::RuntimeProperty {
                name: "ParentEnabled".to_string(),
                is_static: false,
                is_constant: false,
                is_dependent: false,
                get_access: MemberAccess::Public,
                set_access: MemberAccess::Public,
                default_value: None,
            },
        );
        crate::class_registry::register_class(parent);

        let mut child = crate::class_registry::RuntimeClass {
            name: child_name.to_string(),
            parent: Some(parent_name.to_string()),
            properties: HashMap::new(),
            methods: HashMap::new(),
        };
        child.properties.insert(
            "ChildEnabled".to_string(),
            crate::class_registry::RuntimeProperty {
                name: "ChildEnabled".to_string(),
                is_static: false,
                is_constant: false,
                is_dependent: false,
                get_access: MemberAccess::Public,
                set_access: MemberAccess::Public,
                default_value: None,
            },
        );
        crate::class_registry::register_class(child);

        let mut payload = ObjectInstance::new(child_name.to_string());
        payload
            .properties
            .insert("Status".to_string(), Value::from("ready"));
        let target = runmat_gc::gc_allocate(Value::Object(payload)).expect("gc allocate target");

        let handle = HandleRef {
            class_name: child_name.to_string(),
            target,
            valid: true,
        };

        let Value::Cell(cell) =
            run_fieldnames(Value::HandleObject(handle)).expect("fieldnames handle")
        else {
            panic!("expected cell array");
        };
        let collected = cell_strings(&cell);
        assert_eq!(
            collected,
            vec![
                "ChildEnabled".to_string(),
                "ParentEnabled".to_string(),
                "Status".to_string()
            ]
        );
    }

    fn cell_strings(cell: &CellArray) -> Vec<String> {
        cell.data
            .iter()
            .map(|ptr| match ptr {
                Value::CharArray(ca) => ca.data.iter().collect(),
                other => panic!("expected character array cell element, got {other:?}"),
            })
            .collect()
    }
}
