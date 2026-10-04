//! MATLAB-compatible `matfile` compatibility object.
use runmat_types::MemberAccess;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use runmat_builtins::{BuiltinErrorDescriptor, MatFileOptionName};
use runmat_macros::runtime_builtin;
use runmat_value::{ObjectInstance, StringArray, StructValue, Value};

use super::authority::{issue_matfile, resolve_matfile, MatFsPath};
use super::load::read_mat_file_for_builtin_on;
use super::save::encode_workspace_to_mat_bytes;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{
    build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError,
    OBJECT_SUBSASGN_METHOD, OBJECT_SUBSREF_METHOD,
};

const BUILTIN_NAME: &str = "matfile";
pub(super) use runmat_builtins::MATFILE_CLASS;
const MATFILE_SUBSREF: &str = "matlab.io.MatFile.subsref";
const MATFILE_SUBSASGN: &str = "matlab.io.MatFile.subsasgn";
pub(super) const PROPERTIES_FIELD: &str = "Properties";
pub(super) const SOURCE_FIELD: &str = "Source";
pub(super) const WRITABLE_FIELD: &str = "Writable";

static MATFILE_CLASS_REGISTERED: crate::class_registry::ClassRegistration =
    crate::class_registry::ClassRegistration::new(MATFILE_CLASS);

use runmat_builtins::{
    MATFILE_ERROR_INVALID_ARGUMENT, MATFILE_ERROR_INVALID_OPTION, MATFILE_ERROR_IO,
    MATFILE_ERROR_SELECTION, MATFILE_ERROR_UNSUPPORTED,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::mat::matfile")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "matfile",
    op_kind: GpuOpKind::Custom("io-matfile"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "MAT-file object construction and whole-variable reads/writes are host I/O operations. GPU-resident assignment values are gathered before Level-5 MAT serialization.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::mat::matfile")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "matfile",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "MAT-file object creation and subsequent I/O are not fusion candidates.",
};

#[runtime_builtin(
    name = "matfile",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::mat::matfile"
)]
pub async fn matfile_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    filename: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    ensure_matfile_class_registered();
    let controls = prepare_writable_options(rest)?;
    if !matches!(&filename, Value::GpuTensor(_)) {
        normalise_path(&filename)?;
    }
    let host_filename = gather_if_needed_async(&filename).await?;
    let writable = controls.resolve().await?;
    let path = normalise_path(&host_filename)?;
    let public_path = path.to_string_lossy().into_owned();
    let source = std::sync::Arc::new(MatFsPath::capture(path).map_err(|err| {
        matfile_error_with_source(
            &MATFILE_ERROR_IO,
            format!("matfile: failed to resolve path: {err}"),
            err,
        )
    })?);
    let reference = issue_matfile(runtime.as_ref(), source, public_path.clone(), writable)?;
    Ok(Value::Object(matfile_object(
        &public_path,
        writable,
        reference,
    )))
}

#[runtime_builtin(
    name = "matlab.io.MatFile.subsref",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::mat::matfile"
)]
pub async fn matfile_subsref(
    runtime: Option<crate::context::RuntimeContext>,
    obj: Value,
    subscript: Value,
) -> BuiltinResult<Value> {
    ensure_matfile_class_registered();
    let path = crate::object::indexing::parse_standard_substruct(&subscript)?;
    let read_runtime = runtime.clone();
    crate::object::protocol::execute_owned_subsref(
        obj,
        path,
        move |obj, step| async move { matfile_read_step(read_runtime.as_ref(), obj, step).await },
        None,
    )
    .await
}

async fn matfile_read_step(
    runtime: Option<&crate::context::RuntimeContext>,
    obj: Value,
    step: crate::object::indexing::ObjectSubscript,
) -> BuiltinResult<Value> {
    let payload = step.selector_value()?;
    let object = into_matfile_object(obj, MATFILE_SUBSREF)?;
    let owner = resolve_matfile(runtime, &object)?;
    if step.kind() != crate::object::indexing::ObjectIndexKind::Member {
        return Err(matfile_error_with(
            &MATFILE_ERROR_UNSUPPORTED,
            format!("{MATFILE_SUBSREF}: only whole-variable member access is supported"),
        ));
    }
    let field = member_name(&payload, MATFILE_SUBSREF)?;
    if field == PROPERTIES_FIELD {
        return object
            .properties
            .get(PROPERTIES_FIELD)
            .cloned()
            .ok_or_else(|| matfile_error("matfile object is missing Properties"));
    }
    reject_internal_field(&field)?;
    let entries = read_mat_file_for_builtin_on(&owner.source, BUILTIN_NAME).await?;
    resolve_matfile(runtime, &object)?;
    entries
        .into_iter()
        .find(|(name, _)| name == &field)
        .map(|(_, value)| value)
        .ok_or_else(|| {
            matfile_error_with(
                &MATFILE_ERROR_SELECTION,
                format!(
                    "matfile: variable '{field}' was not found in '{}'",
                    owner.source.path().display()
                ),
            )
        })
}

#[runtime_builtin(
    name = "matlab.io.MatFile.subsasgn",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::mat::matfile"
)]
pub async fn matfile_subsasgn(
    runtime: Option<crate::context::RuntimeContext>,
    obj: Value,
    subscript: Value,
    rhs: Value,
) -> BuiltinResult<Value> {
    ensure_matfile_class_registered();
    let path = crate::object::indexing::parse_standard_substruct(&subscript)?;
    let read_runtime = runtime.clone();
    crate::object::protocol::execute_owned_subsasgn(
        obj,
        path,
        vec![rhs],
        move |obj, step| async move { matfile_read_step(read_runtime.as_ref(), obj, step).await },
        move |obj, step, mut values| async move {
            let rhs = values
                .pop()
                .ok_or_else(|| matfile_error("matfile assignment value is missing"))?;
            matfile_write_step(runtime.as_ref(), obj, step, rhs).await
        },
        None,
    )
    .await
}

async fn matfile_write_step(
    runtime: Option<&crate::context::RuntimeContext>,
    obj: Value,
    step: crate::object::indexing::ObjectSubscript,
    rhs: Value,
) -> BuiltinResult<Value> {
    let payload = step.selector_value()?;
    let object = into_matfile_object(obj, MATFILE_SUBSASGN)?;
    if step.kind() != crate::object::indexing::ObjectIndexKind::Member {
        return Err(matfile_error_with(
            &MATFILE_ERROR_UNSUPPORTED,
            format!("{MATFILE_SUBSASGN}: partial MAT-file assignment is not supported yet"),
        ));
    }
    let field = member_name(&payload, MATFILE_SUBSASGN)?;
    if field == PROPERTIES_FIELD || field == SOURCE_FIELD || field == WRITABLE_FIELD {
        return Err(matfile_error_with(
            &MATFILE_ERROR_UNSUPPORTED,
            format!("{MATFILE_SUBSASGN}: '{field}' is read-only"),
        ));
    }
    reject_internal_field(&field)?;
    validate_variable_name(&field, MATFILE_SUBSASGN)?;
    let owner = resolve_matfile(runtime, &object)?;
    if !owner.writable {
        return Err(matfile_error_with(
            &MATFILE_ERROR_UNSUPPORTED,
            "matfile: object is not writable; construct it with matfile(filename, 'Writable', true)",
        ));
    }

    let mut entries = read_existing_entries_for_assignment(runtime, &object, &owner.source).await?;
    resolve_matfile(runtime, &object)?;
    let value = gather_if_needed_async(&rhs).await?;
    resolve_matfile(runtime, &object)?;
    replace_or_append(&mut entries, field, value);
    let bytes = encode_workspace_to_mat_bytes(&entries).await?;
    resolve_matfile(runtime, &object)?;
    owner.source.write(&bytes).await.map_err(|err| {
        matfile_error_with_source(
            &MATFILE_ERROR_IO,
            format!(
                "matfile: failed to write '{}': {err}",
                owner.source.path().display()
            ),
            err,
        )
    })?;
    resolve_matfile(runtime, &object)?;
    Ok(Value::Object(object))
}

pub fn ensure_matfile_class_registered() {
    MATFILE_CLASS_REGISTERED.ensure(|| {
        let mut properties = HashMap::new();
        properties.insert(
            PROPERTIES_FIELD.into(),
            crate::class_registry::RuntimeProperty {
                name: PROPERTIES_FIELD.into(),
                is_static: false,
                is_constant: false,
                is_dependent: false,
                get_access: MemberAccess::Public,
                set_access: MemberAccess::Private,
                default_value: None,
            },
        );

        let mut methods = HashMap::new();
        methods.insert(
            OBJECT_SUBSREF_METHOD.into(),
            crate::class_registry::RuntimeMethod {
                name: OBJECT_SUBSREF_METHOD.into(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: MemberAccess::Public,
                function_name: MATFILE_SUBSREF.into(),
                implicit_class_argument: None,
            },
        );
        methods.insert(
            OBJECT_SUBSASGN_METHOD.into(),
            crate::class_registry::RuntimeMethod {
                name: OBJECT_SUBSASGN_METHOD.into(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: MemberAccess::Public,
                function_name: MATFILE_SUBSASGN.into(),
                implicit_class_argument: None,
            },
        );

        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: MATFILE_CLASS.into(),
            parent: None,
            properties,
            methods,
        });
    });
}

fn matfile_object(
    source: &str,
    writable: bool,
    authority: runmat_value::ForeignRef,
) -> ObjectInstance {
    let mut object = ObjectInstance::new(MATFILE_CLASS.owned()).with_matfile_authority(authority);
    object.properties.insert(
        PROPERTIES_FIELD.to_string(),
        properties_struct(source.to_string(), writable),
    );
    object
}

fn properties_struct(source: String, writable: bool) -> Value {
    let mut st = StructValue::new();
    st.insert(SOURCE_FIELD, Value::String(source));
    st.insert(WRITABLE_FIELD, Value::Bool(writable));
    Value::Struct(st)
}

fn into_matfile_object(value: Value, builtin: &str) -> BuiltinResult<ObjectInstance> {
    match value {
        Value::Object(object) if object.class_name == MATFILE_CLASS => Ok(object),
        other => Err(matfile_error_with(
            &MATFILE_ERROR_INVALID_ARGUMENT,
            format!("{builtin}: receiver must be a {MATFILE_CLASS} object, got {other:?}"),
        )),
    }
}

async fn read_existing_entries_for_assignment(
    runtime: Option<&crate::context::RuntimeContext>,
    object: &ObjectInstance,
    path: &MatFsPath,
) -> BuiltinResult<Vec<(String, Value)>> {
    match path.metadata().await {
        Ok(_) => {
            resolve_matfile(runtime, object)?;
            let entries = read_mat_file_for_builtin_on(path, BUILTIN_NAME).await?;
            resolve_matfile(runtime, object)?;
            Ok(entries)
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(err) => Err(matfile_error_with_source(
            &MATFILE_ERROR_IO,
            format!(
                "matfile: failed to inspect '{}': {err}",
                path.path().display()
            ),
            err,
        )),
    }
}

enum PreparedWritableName {
    Known,
    Resident(Value),
}

enum PreparedWritableValue {
    Known(bool),
    Resident(Value),
}

struct PreparedWritablePair {
    name: PreparedWritableName,
    value: PreparedWritableValue,
}

struct PreparedWritableOptions(Vec<PreparedWritablePair>);

fn prepare_writable_options(rest: Vec<Value>) -> BuiltinResult<PreparedWritableOptions> {
    if !rest.len().is_multiple_of(2) {
        return Err(matfile_error_with(
            &MATFILE_ERROR_INVALID_OPTION,
            "matfile: name-value arguments must come in pairs",
        ));
    }
    let mut pairs = Vec::with_capacity(rest.len() / 2);
    let mut values = rest.into_iter();
    while let Some(name) = values.next() {
        let value = values.next().expect("validated complete Writable pair");
        let name = if matches!(&name, Value::GpuTensor(_)) {
            PreparedWritableName::Resident(name)
        } else {
            let name = value_to_string_scalar(&name).ok_or_else(|| {
                matfile_error_with(
                    &MATFILE_ERROR_INVALID_OPTION,
                    "matfile: option names must be character vectors or string scalars",
                )
            })?;
            if MatFileOptionName::parse(&name) != Some(MatFileOptionName::Writable) {
                return Err(matfile_error_with(
                    &MATFILE_ERROR_INVALID_OPTION,
                    format!("matfile: unsupported option '{name}'"),
                ));
            }
            PreparedWritableName::Known
        };
        let value = if matches!(&value, Value::GpuTensor(_)) {
            PreparedWritableValue::Resident(value)
        } else {
            PreparedWritableValue::Known(value_to_bool_scalar(&value).ok_or_else(|| {
                matfile_error_with(
                    &MATFILE_ERROR_INVALID_OPTION,
                    "matfile: Writable must be a scalar logical or numeric value",
                )
            })?)
        };
        pairs.push(PreparedWritablePair { name, value });
    }
    Ok(PreparedWritableOptions(pairs))
}

impl PreparedWritableOptions {
    async fn resolve(self) -> BuiltinResult<bool> {
        let mut writable = false;
        for pair in self.0 {
            if let PreparedWritableName::Resident(name) = pair.name {
                let value = gather_if_needed_async(&name).await?;
                let name = value_to_string_scalar(&value).ok_or_else(|| {
                    matfile_error_with(
                        &MATFILE_ERROR_INVALID_OPTION,
                        "matfile: option names must be character vectors or string scalars",
                    )
                })?;
                if MatFileOptionName::parse(&name) != Some(MatFileOptionName::Writable) {
                    return Err(matfile_error_with(
                        &MATFILE_ERROR_INVALID_OPTION,
                        format!("matfile: unsupported option '{name}'"),
                    ));
                }
            }
            writable = match pair.value {
                PreparedWritableValue::Known(value) => value,
                PreparedWritableValue::Resident(value) => {
                    let gathered = gather_if_needed_async(&value).await?;
                    value_to_bool_scalar(&gathered).ok_or_else(|| {
                        matfile_error_with(
                            &MATFILE_ERROR_INVALID_OPTION,
                            "matfile: Writable must be a scalar logical or numeric value",
                        )
                    })?
                }
            };
        }
        Ok(writable)
    }
}

fn normalise_path(value: &Value) -> BuiltinResult<PathBuf> {
    let raw = value_to_string_scalar(value).ok_or_else(|| {
        matfile_error_with(
            &MATFILE_ERROR_INVALID_ARGUMENT,
            "matfile: filename must be a character vector or string scalar",
        )
    })?;
    if raw.trim().is_empty() {
        return Err(matfile_error_with(
            &MATFILE_ERROR_INVALID_ARGUMENT,
            "matfile: filename must not be empty",
        ));
    }
    let mut path = PathBuf::from(raw);
    if path.extension().is_none() {
        path.set_extension("mat");
    }
    Ok(path)
}

fn member_name(value: &Value, builtin: &str) -> BuiltinResult<String> {
    value_to_string_scalar(value).ok_or_else(|| {
        matfile_error_with(
            &MATFILE_ERROR_INVALID_ARGUMENT,
            format!("{builtin}: member name must be a character vector or string scalar"),
        )
    })
}

fn reject_internal_field(field: &str) -> BuiltinResult<()> {
    if field.starts_with("__runmat_") {
        return Err(matfile_error_with(
            &MATFILE_ERROR_SELECTION,
            format!("matfile: variable '{field}' was not found"),
        ));
    }
    Ok(())
}

fn validate_variable_name(field: &str, builtin: &str) -> BuiltinResult<()> {
    let mut chars = field.chars();
    let Some(first) = chars.next() else {
        return Err(matfile_error_with(
            &MATFILE_ERROR_SELECTION,
            format!("{builtin}: variable name must not be empty"),
        ));
    };
    if !(first == '_' || first.is_ascii_alphabetic())
        || chars.any(|ch| !(ch == '_' || ch.is_ascii_alphanumeric()))
    {
        return Err(matfile_error_with(
            &MATFILE_ERROR_SELECTION,
            format!("{builtin}: unsupported MAT-file variable name '{field}'"),
        ));
    }
    Ok(())
}

fn replace_or_append(entries: &mut Vec<(String, Value)>, field: String, value: Value) {
    if let Some((_, existing)) = entries.iter_mut().find(|(name, _)| name == &field) {
        *existing = value;
    } else {
        entries.push((field, value));
    }
}

fn value_to_string_scalar(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Some(ca.data.iter().collect()),
        Value::StringArray(StringArray { data, .. }) if data.len() == 1 => Some(data[0].clone()),
        _ => None,
    }
}

fn value_to_bool_scalar(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(b) => Some(*b),
        Value::Num(n) if n.is_finite() => Some(*n != 0.0),
        Value::Int(i) => Some(!i.is_zero()),
        _ => None,
    }
}

pub(super) fn matfile_error(message: impl Into<String>) -> RuntimeError {
    matfile_error_with(&MATFILE_ERROR_INVALID_ARGUMENT, message)
}

fn matfile_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn matfile_error_with_source(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
    source: impl crate::runtime_error::RuntimeErrorSourceBound,
) -> RuntimeError {
    let mut builder = build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::super::load::read_mat_file_for_builtin;
    use super::*;
    use runmat_filesystem::{self as fs, FsProvider};
    use std::sync::Arc;

    async fn matfile_builtin(filename: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        super::matfile_builtin(None, filename, rest).await
    }

    async fn matfile_subsref(obj: Value, kind: String, payload: Value) -> BuiltinResult<Value> {
        let subscript =
            crate::object::indexing::standard_substruct_fixture_from_parts(&kind, payload)?;
        super::matfile_subsref(None, obj, subscript).await
    }

    async fn matfile_subsasgn(
        obj: Value,
        kind: String,
        payload: Value,
        rhs: Value,
    ) -> BuiltinResult<Value> {
        let subscript =
            crate::object::indexing::standard_substruct_fixture_from_parts(&kind, payload)?;
        super::matfile_subsasgn(None, obj, subscript, rhs).await
    }
    use crate::builtins::io::mat::save::encode_workspace_to_mat_bytes;
    use futures::executor::block_on;
    use runmat_value::{CharArray, IntegerStorage, Tensor};
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP_FILE_ID: AtomicU64 = AtomicU64::new(0);

    fn unique_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "runmat_matfile_{name}_{}_{}.mat",
            std::process::id(),
            NEXT_TEMP_FILE_ID.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn tensor(data: &[f64], shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new(data.to_vec(), shape).expect("test tensor"))
    }

    fn write_sample(path: &Path) {
        let bytes = block_on(encode_workspace_to_mat_bytes(&[
            ("A".to_string(), tensor(&[1.0, 2.0, 3.0], vec![1, 3])),
            ("flag".to_string(), Value::Bool(true)),
        ]))
        .expect("encode MAT");
        std::fs::write(path, bytes).expect("write MAT");
    }

    fn string_scalar(value: &str) -> Value {
        Value::StringArray(StringArray {
            data: vec![value.to_string()],
            shape: vec![1, 1],
            rows: 1,
            cols: 1,
        })
    }

    fn char_row(value: &str) -> Value {
        Value::CharArray(CharArray::new_row(value))
    }

    #[test]
    fn matfile_constructs_properties_and_reads_whole_variables() {
        let path = unique_path("read");
        write_sample(&path);
        let object = block_on(matfile_builtin(
            Value::String(path.to_string_lossy().into_owned()),
            vec![],
        ))
        .expect("matfile");

        let props = block_on(matfile_subsref(
            object.clone(),
            ".".to_string(),
            Value::String("Properties".into()),
        ))
        .expect("properties");
        let Value::Struct(props) = props else {
            panic!("expected Properties struct");
        };
        assert_eq!(props.fields.get("Writable"), Some(&Value::Bool(false)));

        let value = block_on(matfile_subsref(
            object,
            ".".to_string(),
            Value::String("A".into()),
        ))
        .expect("A");
        assert_eq!(value, tensor(&[1.0, 2.0, 3.0], vec![1, 3]));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn matfile_accepts_text_forms_and_normalises_extension() {
        let path_without_ext =
            std::env::temp_dir().join(format!("runmat_matfile_text_{}", std::process::id()));
        let expected_path = path_without_ext.with_extension("mat");
        let _ = std::fs::remove_file(&expected_path);

        let object = block_on(matfile_builtin(
            string_scalar(&path_without_ext.to_string_lossy()),
            vec![char_row("Writable"), Value::Num(1.0)],
        ))
        .expect("string filename and char option");
        let Value::Object(object) = object else {
            panic!("expected matfile object");
        };
        assert_eq!(
            object
                .properties
                .get(PROPERTIES_FIELD)
                .and_then(|value| match value {
                    Value::Struct(props) => props.fields.get(SOURCE_FIELD),
                    _ => None,
                }),
            Some(&Value::String(expected_path.to_string_lossy().into_owned()))
        );
        assert_eq!(
            object
                .properties
                .get(PROPERTIES_FIELD)
                .and_then(|value| match value {
                    Value::Struct(props) => props.fields.get(WRITABLE_FIELD),
                    _ => None,
                }),
            Some(&Value::Bool(true))
        );
        assert!(object.matfile_authority().is_some());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn repeated_writable_pairs_are_prepared_once_and_last_value_wins() {
        let object = block_on(matfile_builtin(
            Value::String("repeated.mat".into()),
            vec![
                Value::String("Writable".into()),
                Value::Bool(true),
                Value::String("wRiTaBlE".into()),
                Value::Int(runmat_value::IntValue::I8(0)),
            ],
        ))
        .expect("repeated complete pairs");
        let Value::Object(object) = object else {
            panic!("MAT object");
        };
        let Some(Value::Struct(properties)) = object.properties.get(PROPERTIES_FIELD) else {
            panic!("Properties");
        };
        assert_eq!(
            properties.fields.get(WRITABLE_FIELD),
            Some(&Value::Bool(false))
        );
    }

    #[test]
    fn matfile_writable_assignment_replaces_and_appends_variables() {
        let path = unique_path("write");
        write_sample(&path);
        let mut object = block_on(matfile_builtin(
            Value::String(path.to_string_lossy().into_owned()),
            vec![Value::String("Writable".into()), Value::Bool(true)],
        ))
        .expect("matfile writable");

        object = block_on(matfile_subsasgn(
            object,
            ".".to_string(),
            Value::String("A".into()),
            tensor(&[9.0, 8.0], vec![1, 2]),
        ))
        .expect("replace A");
        block_on(matfile_subsasgn(
            object,
            ".".to_string(),
            Value::String("B".into()),
            Value::Num(42.0),
        ))
        .expect("append B");

        let entries = block_on(read_mat_file_for_builtin(&path, "matfile")).expect("read back");
        let by_name: HashMap<_, _> = entries.into_iter().collect();
        assert_eq!(by_name.get("A"), Some(&tensor(&[9.0, 8.0], vec![1, 2])));
        assert_eq!(by_name.get("B"), Some(&Value::Num(42.0)));
        assert_eq!(by_name.get("flag"), Some(&Value::Bool(true)));
        let _ = std::fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn matfile_receiver_keeps_original_provider_and_cwd_after_swap() {
        let _lock = fs::provider_override_lock();
        let selected = Arc::new(fs::MemoryFsProvider::with_current_dir("/selected"));
        let replacement = Arc::new(fs::MemoryFsProvider::with_current_dir("/replacement"));
        let selected_bytes = block_on(encode_workspace_to_mat_bytes(&[(
            "A".into(),
            Value::Num(7.0),
        )]))
        .expect("selected MAT");
        let replacement_bytes = block_on(encode_workspace_to_mat_bytes(&[(
            "A".into(),
            Value::Num(9.0),
        )]))
        .expect("replacement MAT");
        selected
            .write_project_path("/selected/fixture.mat", &selected_bytes)
            .expect("selected file");
        replacement
            .write_project_path("/replacement/fixture.mat", &replacement_bytes)
            .expect("replacement file");
        let _selected = fs::replace_provider(selected.clone());
        fs::set_current_dir("/selected").expect("selected cwd");
        let object = block_on(matfile_builtin(
            Value::String("fixture.mat".into()),
            vec![Value::String("Writable".into()), Value::Bool(true)],
        ))
        .expect("selected receiver");
        let Value::Object(ref presented) = object else {
            panic!("MAT object");
        };
        let Some(Value::Struct(properties)) = presented.properties.get(PROPERTIES_FIELD) else {
            panic!("MAT Properties");
        };
        assert_eq!(
            properties.fields.get(SOURCE_FIELD),
            Some(&Value::String("fixture.mat".into()))
        );
        let _replacement = fs::replace_provider(replacement.clone());
        fs::set_current_dir("/replacement").expect("replacement cwd");
        let value = block_on(matfile_subsref(
            object.clone(),
            ".".into(),
            Value::String("A".into()),
        ))
        .expect("read original owner");
        assert_eq!(value, Value::Num(7.0));
        block_on(matfile_subsasgn(
            object,
            ".".into(),
            Value::String("A".into()),
            Value::Num(11.0),
        ))
        .expect("write original owner");
        let original = block_on(selected.read(Path::new("/selected/fixture.mat")))
            .expect("read selected bytes");
        let competing = block_on(replacement.read(Path::new("/replacement/fixture.mat")))
            .expect("read replacement bytes");
        let original = crate::builtins::io::mat::load::decode_workspace_from_mat_bytes(&original)
            .expect("decode original");
        let competing = crate::builtins::io::mat::load::decode_workspace_from_mat_bytes(&competing)
            .expect("decode replacement");
        assert_eq!(original, vec![("A".into(), Value::Num(11.0))]);
        assert_eq!(competing, vec![("A".into(), Value::Num(9.0))]);
    }

    #[test]
    fn matfile_writable_assignment_preserves_exact_integer_storage() {
        let path = unique_path("integer_write");
        write_sample(&path);
        let object = block_on(matfile_builtin(
            Value::String(path.to_string_lossy().into_owned()),
            vec![Value::String("Writable".into()), Value::Bool(true)],
        ))
        .expect("matfile writable");
        let tensor = Tensor::new_integer(IntegerStorage::I64(vec![i64::MIN, i64::MAX]), vec![1, 2])
            .expect("integer tensor");
        let input = Value::Tensor(tensor);

        let object = block_on(matfile_subsasgn(
            object,
            ".".to_string(),
            Value::String("limits".into()),
            input,
        ))
        .expect("write integer variable");
        let value = block_on(matfile_subsref(
            object,
            ".".to_string(),
            Value::String("limits".into()),
        ))
        .expect("read integer variable");

        assert!(matches!(value, Value::Tensor(tensor)
            if tensor.integer_storage() == Some(&IntegerStorage::I64(vec![i64::MIN, i64::MAX]))));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn matfile_rejects_readonly_assignment_and_partial_indexing() {
        let path = unique_path("readonly");
        write_sample(&path);
        let object = block_on(matfile_builtin(
            Value::String(path.to_string_lossy().into_owned()),
            vec![],
        ))
        .expect("matfile readonly");

        let err = block_on(matfile_subsasgn(
            object.clone(),
            ".".to_string(),
            Value::String("A".into()),
            Value::Num(1.0),
        ))
        .expect_err("readonly assignment should fail");
        assert!(err.message().contains("not writable"));

        let err = block_on(matfile_subsref(
            object,
            "()".to_string(),
            Value::Cell(runmat_value::CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap()),
        ))
        .expect_err("partial indexing should fail");
        assert!(err.message().contains("whole-variable"));
        let _ = std::fs::remove_file(path);
    }
}
