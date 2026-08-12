//! MATLAB-compatible `matfile` compatibility object.

use std::cell::Cell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use runmat_builtins::{
    Access, BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ClassDef, MethodDef, ObjectInstance, PropertyDef, StringArray, StructValue, Value,
};
use runmat_filesystem::{metadata_async, write_async};
use runmat_macros::runtime_builtin;

use super::load::read_mat_file_for_builtin;
use super::save::encode_workspace_to_mat_bytes;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{
    build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError, OBJECT_INDEX_MEMBER,
    OBJECT_SUBSASGN_METHOD, OBJECT_SUBSREF_METHOD,
};

const BUILTIN_NAME: &str = "matfile";
const MATFILE_CLASS: &str = "matlab.io.MatFile";
const MATFILE_SUBSREF: &str = "matlab.io.MatFile.subsref";
const MATFILE_SUBSASGN: &str = "matlab.io.MatFile.subsasgn";
const PROPERTIES_FIELD: &str = "Properties";
const SOURCE_FIELD: &str = "Source";
const WRITABLE_FIELD: &str = "Writable";
const PATH_FIELD: &str = "__runmat_matfile_path";
const WRITABLE_STATE_FIELD: &str = "__runmat_matfile_writable";

thread_local! {
    static MATFILE_CLASS_REGISTERED: Cell<bool> = const { Cell::new(false) };
}

const MATFILE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "m",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "MAT-file compatibility object.",
}];
const MATFILE_INPUTS_FILENAME: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "MAT-file path.",
}];
const MATFILE_INPUTS_WRITABLE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "MAT-file path.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"Writable\""),
        description: "Writable option name.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: Some("false"),
        description: "Whether whole-variable dot assignment writes back to the MAT-file.",
    },
];
const MATFILE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "m = matfile(filename)",
        inputs: &MATFILE_INPUTS_FILENAME,
        outputs: &MATFILE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "m = matfile(filename, 'Writable', tf)",
        inputs: &MATFILE_INPUTS_WRITABLE,
        outputs: &MATFILE_OUTPUT,
    },
];
const MATFILE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MATFILE.INVALID_ARGUMENT",
    identifier: Some("RunMat:matfile:InvalidArgument"),
    when: "Arguments do not match supported matfile invocation forms.",
    message: "matfile: invalid argument",
};
const MATFILE_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MATFILE.INVALID_OPTION",
    identifier: Some("RunMat:matfile:InvalidOption"),
    when: "Name-value options are invalid or unsupported.",
    message: "matfile: invalid option",
};
const MATFILE_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MATFILE.IO",
    identifier: Some("RunMat:matfile:Io"),
    when: "MAT-file data cannot be read or written.",
    message: "matfile: MAT-file I/O failure",
};
const MATFILE_ERROR_SELECTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MATFILE.SELECTION",
    identifier: Some("RunMat:matfile:Selection"),
    when: "Requested MAT-file variable is missing or invalid.",
    message: "matfile: variable selection failed",
};
const MATFILE_ERROR_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MATFILE.UNSUPPORTED",
    identifier: Some("RunMat:matfile:Unsupported"),
    when: "Partial MAT-file indexing or unsupported object operations are requested.",
    message: "matfile: unsupported operation",
};
const MATFILE_ERRORS: [BuiltinErrorDescriptor; 5] = [
    MATFILE_ERROR_INVALID_ARGUMENT,
    MATFILE_ERROR_INVALID_OPTION,
    MATFILE_ERROR_IO,
    MATFILE_ERROR_SELECTION,
    MATFILE_ERROR_UNSUPPORTED,
];
pub const MATFILE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MATFILE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &MATFILE_ERRORS,
};

const SUBSREF_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "MAT-file property or whole variable value.",
}];
const SUBSREF_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "m",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "MAT-file object.",
    },
    BuiltinParamDescriptor {
        name: "kind",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\".\""),
        description: "Indexing kind.",
    },
    BuiltinParamDescriptor {
        name: "payload",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Member name.",
    },
];
const SUBSREF_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "value = matlab.io.MatFile.subsref(m, kind, payload)",
    inputs: &SUBSREF_INPUTS,
    outputs: &SUBSREF_OUTPUT,
}];
pub const MATFILE_SUBSREF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SUBSREF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &MATFILE_ERRORS,
};

const SUBSASGN_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "m",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "MAT-file object.",
    },
    BuiltinParamDescriptor {
        name: "kind",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\".\""),
        description: "Indexing kind.",
    },
    BuiltinParamDescriptor {
        name: "payload",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Member name.",
    },
    BuiltinParamDescriptor {
        name: "rhs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Value to write.",
    },
];
const SUBSASGN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "m = matlab.io.MatFile.subsasgn(m, kind, payload, rhs)",
    inputs: &SUBSASGN_INPUTS,
    outputs: &MATFILE_OUTPUT,
}];
pub const MATFILE_SUBSASGN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SUBSASGN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &MATFILE_ERRORS,
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
    category = "io/mat",
    summary = "Create a MAT-file object for whole-variable reads and writes.",
    keywords = "matfile,mat,MAT-file,Writable",
    type_resolver(crate::builtins::io::type_resolvers::matfile_type),
    descriptor(crate::builtins::io::mat::matfile::MATFILE_DESCRIPTOR),
    builtin_path = "crate::builtins::io::mat::matfile"
)]
pub async fn matfile_builtin(filename: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_matfile_class_registered();
    let host_filename = gather_if_needed_async(&filename).await?;
    let host_rest = gather_rest(rest).await?;
    let writable = parse_writable_option(&host_rest)?;
    let path = normalise_path(&host_filename)?;
    Ok(Value::Object(matfile_object(path, writable)))
}

#[runtime_builtin(
    name = "matlab.io.MatFile.subsref",
    category = "io/mat",
    summary = "Read MAT-file object properties or whole variables.",
    keywords = "matfile,subsref,MAT-file",
    type_resolver(crate::builtins::io::type_resolvers::matfile_subsref_type),
    descriptor(crate::builtins::io::mat::matfile::MATFILE_SUBSREF_DESCRIPTOR),
    builtin_path = "crate::builtins::io::mat::matfile"
)]
pub async fn matfile_subsref(obj: Value, kind: String, payload: Value) -> BuiltinResult<Value> {
    ensure_matfile_class_registered();
    let object = into_matfile_object(obj, MATFILE_SUBSREF)?;
    if kind != OBJECT_INDEX_MEMBER {
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
    let path = object_path(&object)?;
    let entries = read_mat_file_for_builtin(&path, BUILTIN_NAME).await?;
    entries
        .into_iter()
        .find(|(name, _)| name == &field)
        .map(|(_, value)| value)
        .ok_or_else(|| {
            matfile_error_with(
                &MATFILE_ERROR_SELECTION,
                format!(
                    "matfile: variable '{field}' was not found in '{}'",
                    path.display()
                ),
            )
        })
}

#[runtime_builtin(
    name = "matlab.io.MatFile.subsasgn",
    category = "io/mat",
    summary = "Write whole variables through a writable MAT-file object.",
    keywords = "matfile,subsasgn,MAT-file,Writable",
    type_resolver(crate::builtins::io::type_resolvers::matfile_type),
    descriptor(crate::builtins::io::mat::matfile::MATFILE_SUBSASGN_DESCRIPTOR),
    builtin_path = "crate::builtins::io::mat::matfile"
)]
pub async fn matfile_subsasgn(
    obj: Value,
    kind: String,
    payload: Value,
    rhs: Value,
) -> BuiltinResult<Value> {
    ensure_matfile_class_registered();
    let object = into_matfile_object(obj, MATFILE_SUBSASGN)?;
    if kind != OBJECT_INDEX_MEMBER {
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
    if !object_writable(&object)? {
        return Err(matfile_error_with(
            &MATFILE_ERROR_UNSUPPORTED,
            "matfile: object is not writable; construct it with matfile(filename, 'Writable', true)",
        ));
    }

    let path = object_path(&object)?;
    let mut entries = read_existing_entries_for_assignment(&path).await?;
    let value = gather_if_needed_async(&rhs).await?;
    replace_or_append(&mut entries, field, value);
    let bytes = encode_workspace_to_mat_bytes(&entries).await?;
    write_async(&path, &bytes).await.map_err(|err| {
        matfile_error_with_source(
            &MATFILE_ERROR_IO,
            format!("matfile: failed to write '{}': {err}", path.display()),
            err,
        )
    })?;
    Ok(Value::Object(object))
}

pub fn ensure_matfile_class_registered() {
    MATFILE_CLASS_REGISTERED.with(|registered| {
        if registered.get() {
            return;
        }

        let mut properties = HashMap::new();
        properties.insert(
            PROPERTIES_FIELD.to_string(),
            PropertyDef {
                name: PROPERTIES_FIELD.to_string(),
                is_static: false,
                is_constant: false,
                is_dependent: false,
                get_access: Access::Public,
                set_access: Access::Private,
                default_value: None,
            },
        );

        let mut methods = HashMap::new();
        methods.insert(
            OBJECT_SUBSREF_METHOD.to_string(),
            MethodDef {
                name: OBJECT_SUBSREF_METHOD.to_string(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: Access::Public,
                function_name: MATFILE_SUBSREF.to_string(),
                implicit_class_argument: None,
            },
        );
        methods.insert(
            OBJECT_SUBSASGN_METHOD.to_string(),
            MethodDef {
                name: OBJECT_SUBSASGN_METHOD.to_string(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: Access::Public,
                function_name: MATFILE_SUBSASGN.to_string(),
                implicit_class_argument: None,
            },
        );

        runmat_builtins::register_class(ClassDef {
            name: MATFILE_CLASS.to_string(),
            parent: None,
            properties,
            methods,
        });
        registered.set(true);
    });
}

fn matfile_object(path: PathBuf, writable: bool) -> ObjectInstance {
    let source = path.to_string_lossy().to_string();
    let mut object = ObjectInstance::new(MATFILE_CLASS.to_string());
    object
        .properties
        .insert(PATH_FIELD.to_string(), Value::String(source.clone()));
    object
        .properties
        .insert(WRITABLE_STATE_FIELD.to_string(), Value::Bool(writable));
    object.properties.insert(
        PROPERTIES_FIELD.to_string(),
        properties_struct(source, writable),
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

fn object_path(object: &ObjectInstance) -> BuiltinResult<PathBuf> {
    let Some(Value::String(path)) = object.properties.get(PATH_FIELD) else {
        return Err(matfile_error("matfile object is missing source path"));
    };
    Ok(PathBuf::from(path))
}

fn object_writable(object: &ObjectInstance) -> BuiltinResult<bool> {
    let Some(Value::Bool(writable)) = object.properties.get(WRITABLE_STATE_FIELD) else {
        return Err(matfile_error("matfile object is missing writable state"));
    };
    Ok(*writable)
}

async fn read_existing_entries_for_assignment(path: &Path) -> BuiltinResult<Vec<(String, Value)>> {
    match metadata_async(path).await {
        Ok(_) => read_mat_file_for_builtin(path, BUILTIN_NAME).await,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(err) => Err(matfile_error_with_source(
            &MATFILE_ERROR_IO,
            format!("matfile: failed to inspect '{}': {err}", path.display()),
            err,
        )),
    }
}

async fn gather_rest(rest: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(rest.len());
    for value in rest {
        out.push(gather_if_needed_async(&value).await?);
    }
    Ok(out)
}

fn parse_writable_option(rest: &[Value]) -> BuiltinResult<bool> {
    if rest.is_empty() {
        return Ok(false);
    }
    if !rest.len().is_multiple_of(2) {
        return Err(matfile_error_with(
            &MATFILE_ERROR_INVALID_OPTION,
            "matfile: name-value arguments must come in pairs",
        ));
    }
    let mut writable = false;
    for pair in rest.chunks_exact(2) {
        let name = value_to_string_scalar(&pair[0]).ok_or_else(|| {
            matfile_error_with(
                &MATFILE_ERROR_INVALID_OPTION,
                "matfile: option names must be character vectors or string scalars",
            )
        })?;
        if !name.eq_ignore_ascii_case(WRITABLE_FIELD) {
            return Err(matfile_error_with(
                &MATFILE_ERROR_INVALID_OPTION,
                format!("matfile: unsupported option '{name}'"),
            ));
        }
        writable = value_to_bool_scalar(&pair[1]).ok_or_else(|| {
            matfile_error_with(
                &MATFILE_ERROR_INVALID_OPTION,
                "matfile: Writable must be a scalar logical or numeric value",
            )
        })?;
    }
    Ok(writable)
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

fn matfile_error(message: impl Into<String>) -> RuntimeError {
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
    source: impl std::fmt::Display,
) -> RuntimeError {
    let source = std::io::Error::new(std::io::ErrorKind::Other, source.to_string());
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
    use super::*;
    use crate::builtins::io::mat::save::encode_workspace_to_mat_bytes;
    use futures::executor::block_on;
    use runmat_builtins::{CharArray, IntegerStorage, Tensor};
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
            object.properties.get(PATH_FIELD),
            Some(&Value::String(expected_path.to_string_lossy().into_owned()))
        );
        assert_eq!(
            object.properties.get(WRITABLE_STATE_FIELD),
            Some(&Value::Bool(true))
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
            Value::String("A".into()),
        ))
        .expect_err("partial indexing should fail");
        assert!(err.message().contains("whole-variable"));
        let _ = std::fs::remove_file(path);
    }
}
