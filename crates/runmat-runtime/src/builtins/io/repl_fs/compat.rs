//! Compatibility filesystem, path, process, preference, and environment helpers.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
#[cfg(not(target_arch = "wasm32"))]
use std::process::Command;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinExtensionDescriptor, BuiltinExtensionMode,
    BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, CellArray, CharArray, IntegerStorage, LogicalArray, NumericDType,
    NumericScalar, ObjectInstance, StructValue, Tensor, Value,
};
use runmat_filesystem as vfs;
use runmat_macros::runtime_builtin;

use crate::builtins::common::env as runtime_env;
use crate::builtins::common::fs::{expand_user_path, home_directory, path_to_string};
use crate::builtins::common::path_state::{set_path_string, PATH_LIST_SEPARATOR};
use crate::output_count;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

thread_local! {
    static PREFS: RefCell<BTreeMap<String, BTreeMap<String, Value>>> =
        const { RefCell::new(BTreeMap::new()) };
}

const INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
const INPUTS_ONE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "input",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input argument.",
}];
const INPUTS_TWO: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "input1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First input argument.",
    },
    BuiltinParamDescriptor {
        name: "input2",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second input argument.",
    },
];
const INPUTS_THREE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "input1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First input argument.",
    },
    BuiltinParamDescriptor {
        name: "input2",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second input argument.",
    },
    BuiltinParamDescriptor {
        name: "input3",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Optional third input argument.",
    },
];
const OUTPUT_VALUE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Result value.",
}];
const OUTPUT_THREE_TEXT: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "folder",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Folder component.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Base filename component.",
    },
    BuiltinParamDescriptor {
        name: "extension",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Extension component including the leading dot.",
    },
];
const OUTPUT_STATUS_MESSAGE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "status",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "0 on success, nonzero on failure.",
    },
    BuiltinParamDescriptor {
        name: "message",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Command output or diagnostic text.",
    },
];
const OUTPUT_STATUS_ATTRIB: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "status",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "1 when attributes were read, otherwise 0.",
    },
    BuiltinParamDescriptor {
        name: "attributes",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Attribute struct or diagnostic struct.",
    },
];

macro_rules! simple_descriptor {
    ($sig:ident, $desc:ident, $label:expr, $inputs:expr, $outputs:expr, $mode:expr) => {
        const $sig: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
            label: $label,
            inputs: $inputs,
            outputs: $outputs,
        }];
        pub const $desc: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &$sig,
            output_mode: $mode,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &[],
        };
    };
}

simple_descriptor!(
    FILEPARTS_SIGNATURES,
    FILEPARTS_DESCRIPTOR,
    "[folder, name, extension] = fileparts(filename)",
    &INPUTS_ONE,
    &OUTPUT_THREE_TEXT,
    BuiltinOutputMode::ByRequestedOutputCount
);
pub const ISFILE_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "isfile is a host-text filesystem predicate; integer and resident numeric paths reject before provider or filesystem access.",
};

pub const FILEPARTS_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "fileparts is a structural host-text parser with no numeric role; numeric and provider-resident values are rejected as invalid text before any gather or provider access.",
    };
simple_descriptor!(
    ISFILE_SIGNATURES,
    ISFILE_DESCRIPTOR,
    "tf = isfile(path)",
    &INPUTS_ONE,
    &OUTPUT_VALUE,
    BuiltinOutputMode::Fixed
);

const MEMMAPFILE_INTEGER_CONTROLS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "memmapfile-integer-property-controls",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "typed-integer Writable, Offset, Repeat, or Format-shape property values for memmapfile are a RunMat extension; the public MATLAB page specifies logical/double controls and does not establish typed-integer Format dimensions",
        error_identifier: Some("RunMat:compatibility:MemmapfileIntegerControlsExtension"),
    };

const MEMMAPFILE_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "memmapfile-explicit-gpu-argument",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "passing an explicit gpuArray argument to host-only memmapfile is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:MemmapfileExplicitGpuExtension"),
};

pub const MEMMAPFILE_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    MEMMAPFILE_INTEGER_CONTROLS_EXTENSION,
    MEMMAPFILE_EXPLICIT_GPU_EXTENSION,
];

const MEMMAPFILE_INTEGER_SHAPE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Format.shape",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "RunMat permits a typed-integer shape vector inside a Format cell and parses it from authoritative storage as positive platform dimensions without binary64 conversion; current public evidence does not establish that typed form.",
    }];

const MEMMAPFILE_INTEGER_CONTROL_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Writable, Offset, or Repeat",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "RunMat accepts exact typed-integer property controls, but compatibility mode retains the public logical/double property types.",
    }];

pub const MEMMAPFILE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "m.Data = memmapfile(..., 'Format', integer_typename)",
        inputs: &[],
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "The documented int8/int16/int32/int64/uint8/uint16/uint32/uint64 formats decode file bytes directly into matching integer classes without a numeric cast.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "m = memmapfile(..., 'Format', {type, integer_shape, field})",
        inputs: &MEMMAPFILE_INTEGER_SHAPE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "This compatibility-gated typed-shape extension checks dimensions exactly before byte-count arithmetic; the mapped Data class is selected independently by the textual type name.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "m = memmapfile(..., integer_property_control)",
        inputs: &MEMMAPFILE_INTEGER_CONTROL_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "This compatibility-gated extension parses typed controls exactly. Automatic residency gathers through its owner; explicit residency is separately gated before provider access.",
    },
];
pub const ISFOLDER_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "isfolder is a host-text filesystem predicate; integer and resident numeric paths reject before provider or filesystem access.",
};
simple_descriptor!(
    ISFOLDER_SIGNATURES,
    ISFOLDER_DESCRIPTOR,
    "tf = isfolder(path)",
    &INPUTS_ONE,
    &OUTPUT_VALUE,
    BuiltinOutputMode::Fixed
);
pub const ISENV_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "isenv accepts host-text environment names; integer and resident numeric paths reject before provider or environment access.",
};
simple_descriptor!(
    ISENV_SIGNATURES,
    ISENV_DESCRIPTOR,
    "tf = isenv(name)",
    &INPUTS_ONE,
    &OUTPUT_VALUE,
    BuiltinOutputMode::Fixed
);
simple_descriptor!(
    UNSETENV_SIGNATURES,
    UNSETENV_DESCRIPTOR,
    "status = unsetenv(name)",
    &INPUTS_ONE,
    &OUTPUT_VALUE,
    BuiltinOutputMode::Fixed
);
simple_descriptor!(
    MATLABROOT_SIGNATURES,
    MATLABROOT_DESCRIPTOR,
    "root = matlabroot()",
    &INPUTS_NONE,
    &OUTPUT_VALUE,
    BuiltinOutputMode::Fixed
);
simple_descriptor!(
    PATHSEP_SIGNATURES,
    PATHSEP_DESCRIPTOR,
    "sep = pathsep()",
    &INPUTS_NONE,
    &OUTPUT_VALUE,
    BuiltinOutputMode::Fixed
);
const SYSTEM_INPUTS_COMMAND: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "command",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Operating-system command to execute.",
}];
const SYSTEM_INPUTS_ECHO: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "command",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Operating-system command to execute.",
    },
    BuiltinParamDescriptor {
        name: "echo",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("'-echo'"),
        description: "The literal '-echo', which also writes command output to the console.",
    },
];
const SYSTEM_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "[status, output] = system(command)",
        inputs: &SYSTEM_INPUTS_COMMAND,
        outputs: &OUTPUT_STATUS_MESSAGE,
    },
    BuiltinSignatureDescriptor {
        label: "[status, output] = system(command, '-echo')",
        inputs: &SYSTEM_INPUTS_ECHO,
        outputs: &OUTPUT_STATUS_MESSAGE,
    },
];
pub const SYSTEM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SYSTEM_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};
pub const SYSTEM_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "system accepts a host character vector or string command, not numeric input; integer and resident numeric values reject before provider or process access.",
};
simple_descriptor!(
    WHAT_SIGNATURES,
    WHAT_DESCRIPTOR,
    "info = what(folder)",
    &INPUTS_ONE,
    &OUTPUT_VALUE,
    BuiltinOutputMode::Fixed
);
simple_descriptor!(
    FILEATTRIB_SIGNATURES,
    FILEATTRIB_DESCRIPTOR,
    "[status, attributes] = fileattrib(path)",
    &INPUTS_ONE,
    &OUTPUT_STATUS_ATTRIB,
    BuiltinOutputMode::ByRequestedOutputCount
);

pub const FILEATTRIB_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "fileattrib accepts host-text paths and textual controls; numeric and provider-resident values are rejected before any gather, provider access, or filesystem operation, while numeric status and attribute fields are outputs only.",
    };
const GETPREF_INPUTS_TWO: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "group",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Custom setting group name.",
    },
    BuiltinParamDescriptor {
        name: "pref",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Setting name or cell/string array of setting names.",
    },
];
const GETPREF_INPUTS_THREE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "group",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Custom setting group name.",
    },
    BuiltinParamDescriptor {
        name: "pref",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Setting name or cell/string array of setting names.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Default value, or cell array of defaults for multiple setting names.",
    },
];
const GETPREF_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "value = getpref()",
        inputs: &INPUTS_NONE,
        outputs: &OUTPUT_VALUE,
    },
    BuiltinSignatureDescriptor {
        label: "value = getpref(group, pref)",
        inputs: &GETPREF_INPUTS_TWO,
        outputs: &OUTPUT_VALUE,
    },
    BuiltinSignatureDescriptor {
        label: "value = getpref(group, pref, value)",
        inputs: &GETPREF_INPUTS_THREE,
        outputs: &OUTPUT_VALUE,
    },
];
pub const GETPREF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GETPREF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};

pub const GETPREF_GROUP_QUERY_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "getpref-group-query",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "getpref(group) group-structure query is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:GetprefGroupQueryExtension"),
};
pub const GETPREF_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [GETPREF_GROUP_QUERY_EXTENSION];

const GETPREF_INTEGER_DEFAULT_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "value",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight host integer classes are stored and returned unchanged when used as a missing preference default; group and preference controls remain text-only.",
    }];
pub const GETPREF_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "value = getpref(group, pref, integer_default)",
        inputs: &GETPREF_INTEGER_DEFAULT_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "A missing preference is stored for the current RunMat preference session as the exact integer scalar or array and returned without floating conversion; an existing preference wins unchanged. Durable cross-session storage remains a general preference-system gap.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "integer_value = getpref(group, pref)",
        inputs: &[],
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "A host integer scalar or array stored in the current RunMat preference session is returned with its exact class, shape, and value; multiple names wrap values in a shape-preserving cell array. Durable cross-session storage remains a general preference-system gap.",
    },
];
pub const ISPREF_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "ispref accepts text group and preference names; integer host or resident controls are invalid and reject before provider access.",
};
const SETPREF_INTEGER_VALUE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "value",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Preference values may use any MATLAB datatype; all eight integer classes retain their exact native class, shape, and payload.",
    }];
pub const SETPREF_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "setpref(group, preference, integer_value)",
        inputs: &SETPREF_INTEGER_VALUE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Host values are cloned into the RunMat preference session without conversion; automatic residency gathers exact storage before host persistence. Durable cross-session storage remains a general preference-system gap.",
    }];
simple_descriptor!(
    SETPREF_SIGNATURES,
    SETPREF_DESCRIPTOR,
    "setpref(group, preference, value)",
    &INPUTS_THREE,
    &OUTPUT_VALUE,
    BuiltinOutputMode::Fixed
);
simple_descriptor!(
    ISPREF_SIGNATURES,
    ISPREF_DESCRIPTOR,
    "tf = ispref(group, preference)",
    &INPUTS_TWO,
    &OUTPUT_VALUE,
    BuiltinOutputMode::Fixed
);
simple_descriptor!(
    REHASH_SIGNATURES,
    REHASH_DESCRIPTOR,
    "rehash()",
    &INPUTS_NONE,
    &OUTPUT_VALUE,
    BuiltinOutputMode::Fixed
);
simple_descriptor!(
    USERPATH_SIGNATURES,
    USERPATH_DESCRIPTOR,
    "path = userpath(option)",
    &INPUTS_ONE,
    &OUTPUT_VALUE,
    BuiltinOutputMode::Fixed
);
simple_descriptor!(
    RESTOREDEFAULTPATH_SIGNATURES,
    RESTOREDEFAULTPATH_DESCRIPTOR,
    "path = restoredefaultpath()",
    &INPUTS_NONE,
    &OUTPUT_VALUE,
    BuiltinOutputMode::Fixed
);
simple_descriptor!(
    MEMMAPFILE_SIGNATURES,
    MEMMAPFILE_DESCRIPTOR,
    "m = memmapfile(filename, Name, Value)",
    &INPUTS_THREE,
    &OUTPUT_VALUE,
    BuiltinOutputMode::Fixed
);
simple_descriptor!(
    WINQUERYREG_SIGNATURES,
    WINQUERYREG_DESCRIPTOR,
    "value = winqueryreg(root, key, valuename)",
    &INPUTS_THREE,
    &OUTPUT_VALUE,
    BuiltinOutputMode::Fixed
);

pub(super) fn compat_error(name: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(name).build()
}

fn map_control_flow(name: &str, err: RuntimeError) -> RuntimeError {
    let identifier = err.identifier().map(str::to_string);
    let mut builder = build_runtime_error(format!("{name}: {}", err.message()))
        .with_builtin(name)
        .with_source(err);
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(super) async fn gather_args(name: &str, args: &[Value]) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(args.len());
    for value in args {
        out.push(
            gather_if_needed_async(value)
                .await
                .map_err(|err| map_control_flow(name, err))?,
        );
    }
    Ok(out)
}

pub(super) fn scalar_text(value: &Value, name: &str, arg: &str) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(array) if array.rows == 1 => Ok(char_row_to_string(array)),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        _ => Err(compat_error(
            name,
            format!("{name}: {arg} must be a string scalar or character vector"),
        )),
    }
}

fn char_row_to_string(array: &CharArray) -> String {
    let mut text: String = array.data.iter().take(array.cols).collect();
    while text.ends_with(' ') {
        text.pop();
    }
    text
}

pub(super) fn char_value(text: &str) -> Value {
    Value::CharArray(CharArray::new_row(text))
}

fn logical_array(values: Vec<bool>, shape: Vec<usize>, name: &str) -> BuiltinResult<Value> {
    Ok(Value::LogicalArray(
        LogicalArray::new(values.into_iter().map(u8::from).collect(), shape)
            .map_err(|err| compat_error(name, err))?,
    ))
}

fn expand_path_for_builtin(text: &str, name: &str) -> BuiltinResult<PathBuf> {
    let expanded = expand_user_path(text.trim(), name).map_err(|err| compat_error(name, err))?;
    Ok(PathBuf::from(expanded))
}

pub(super) fn value_to_path(value: &Value, name: &str, arg: &str) -> BuiltinResult<PathBuf> {
    let text = scalar_text(value, name, arg)?;
    expand_path_for_builtin(&text, name)
}

fn output_list_for_count(default: Vec<Value>) -> Value {
    if let Some(count) = output_count::current_output_count() {
        return output_count::output_list_with_padding(count, default);
    }
    if default.len() == 1 {
        default.into_iter().next().unwrap()
    } else {
        Value::OutputList(default)
    }
}

#[runtime_builtin(
    name = "fileparts",
    category = "io/repl_fs",
    summary = "Split a file path into folder, base name, and extension.",
    keywords = "fileparts,path,filename,extension",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::fileparts_type),
    descriptor(crate::builtins::io::repl_fs::compat::FILEPARTS_DESCRIPTOR),
    integer_audit(crate::builtins::io::repl_fs::compat::FILEPARTS_INTEGER_AUDIT),
    builtin_path = "crate::builtins::io::repl_fs::compat"
)]
async fn fileparts_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.len() != 1 {
        return Err(compat_error(
            "fileparts",
            "fileparts: expected exactly one input argument",
        ));
    }
    let input = scalar_text(&args[0], "fileparts", "filename")?;
    let path = Path::new(&input);
    let folder = path.parent().map(path_to_string).unwrap_or_default();
    let filename = path.file_name().and_then(|v| v.to_str()).unwrap_or("");
    let (name, ext) = match filename.rfind('.') {
        Some(0) | None => (filename.to_string(), String::new()),
        Some(idx) => (filename[..idx].to_string(), filename[idx..].to_string()),
    };
    Ok(output_list_for_count(vec![
        char_value(&folder),
        char_value(&name),
        char_value(&ext),
    ]))
}

#[runtime_builtin(
    name = "isfile",
    category = "io/repl_fs",
    summary = "Return true for paths that name existing files.",
    keywords = "isfile,file,exists,predicate",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::bool_type),
    descriptor(crate::builtins::io::repl_fs::compat::ISFILE_DESCRIPTOR),
    integer_audit(crate::builtins::io::repl_fs::compat::ISFILE_INTEGER_AUDIT),
    builtin_path = "crate::builtins::io::repl_fs::compat"
)]
async fn isfile_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    path_predicate_builtin("isfile", args, |meta| meta.is_file()).await
}

#[runtime_builtin(
    name = "isfolder",
    category = "io/repl_fs",
    summary = "Return true for paths that name existing folders.",
    keywords = "isfolder,folder,directory,exists,predicate",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::bool_type),
    descriptor(crate::builtins::io::repl_fs::compat::ISFOLDER_DESCRIPTOR),
    integer_audit(crate::builtins::io::repl_fs::compat::ISFOLDER_INTEGER_AUDIT),
    builtin_path = "crate::builtins::io::repl_fs::compat"
)]
async fn isfolder_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    path_predicate_builtin("isfolder", args, |meta| meta.is_dir()).await
}

async fn path_predicate_builtin(
    name: &str,
    args: Vec<Value>,
    predicate: fn(&vfs::FsMetadata) -> bool,
) -> BuiltinResult<Value> {
    if args.iter().any(value_contains_resident) {
        return Err(compat_error(
            name,
            format!("{name}: path must be text; provider-resident numeric values are invalid"),
        ));
    }
    let args = gather_args(name, &args).await?;
    if args.len() != 1 {
        return Err(compat_error(
            name,
            format!("{name}: expected exactly one input argument"),
        ));
    }
    match &args[0] {
        Value::StringArray(array) => {
            let mut values = Vec::with_capacity(array.data.len());
            for text in &array.data {
                let path = expand_path_for_builtin(text, name)?;
                values.push(
                    vfs::metadata_async(&path)
                        .await
                        .is_ok_and(|m| predicate(&m)),
                );
            }
            logical_array(values, array.shape.clone(), name)
        }
        Value::Cell(array) => {
            let mut values = Vec::with_capacity(array.data.len());
            for value in &array.data {
                let path = value_to_path(value, name, "path")?;
                values.push(
                    vfs::metadata_async(&path)
                        .await
                        .is_ok_and(|m| predicate(&m)),
                );
            }
            logical_array(values, array.shape.clone(), name)
        }
        value => {
            let path = value_to_path(value, name, "path")?;
            Ok(Value::Bool(
                vfs::metadata_async(&path)
                    .await
                    .is_ok_and(|m| predicate(&m)),
            ))
        }
    }
}

#[runtime_builtin(
    name = "isenv",
    category = "io/repl_fs",
    summary = "Return true when environment variables are defined.",
    keywords = "isenv,environment,variable,predicate",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::bool_type),
    descriptor(crate::builtins::io::repl_fs::compat::ISENV_DESCRIPTOR),
    integer_audit(crate::builtins::io::repl_fs::compat::ISENV_INTEGER_AUDIT),
    builtin_path = "crate::builtins::io::repl_fs::compat"
)]
async fn isenv_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.iter().any(value_contains_resident) {
        return Err(compat_error(
            "isenv",
            "isenv: name must be text; provider-resident numeric values are invalid",
        ));
    }
    let args = gather_args("isenv", &args).await?;
    if args.len() != 1 {
        return Err(compat_error("isenv", "isenv: expected exactly one input"));
    }
    match &args[0] {
        Value::StringArray(array) => logical_array(
            array
                .data
                .iter()
                .map(|name| runtime_env::var(name).is_ok())
                .collect(),
            array.shape.clone(),
            "isenv",
        ),
        Value::Cell(array) => {
            let mut out = Vec::with_capacity(array.data.len());
            for value in &array.data {
                out.push(runtime_env::var(&scalar_text(value, "isenv", "name")?).is_ok());
            }
            logical_array(out, array.shape.clone(), "isenv")
        }
        value => Ok(Value::Bool(
            runtime_env::var(&scalar_text(value, "isenv", "name")?).is_ok(),
        )),
    }
}

fn value_contains_resident(value: &Value) -> bool {
    match value {
        Value::GpuTensor(_) => true,
        Value::Cell(value) => value.data.iter().any(value_contains_resident),
        Value::Struct(value) => value.fields.values().any(value_contains_resident),
        Value::Object(value) => value.properties.values().any(value_contains_resident),
        Value::Closure(value) => value.captures.iter().any(value_contains_resident),
        Value::OutputList(values) => values.iter().any(value_contains_resident),
        _ => false,
    }
}

#[runtime_builtin(
    name = "unsetenv",
    category = "io/repl_fs",
    summary = "Remove an environment variable.",
    keywords = "unsetenv,environment,variable,remove",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::num_type),
    descriptor(crate::builtins::io::repl_fs::compat::UNSETENV_DESCRIPTOR),
    builtin_path = "crate::builtins::io::repl_fs::compat"
)]
async fn unsetenv_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let args = gather_args("unsetenv", &args).await?;
    if args.len() != 1 {
        return Err(compat_error(
            "unsetenv",
            "unsetenv: expected exactly one input",
        ));
    }
    let name = scalar_text(&args[0], "unsetenv", "name")?;
    if name.is_empty() || name.contains('=') || name.contains('\0') {
        return Ok(Value::Num(1.0));
    }
    runtime_env::remove_var(&name);
    Ok(Value::Num(0.0))
}

#[runtime_builtin(
    name = "matlabroot",
    category = "io/repl_fs",
    summary = "Return the RunMat installation root as MATLAB-root compatibility text.",
    keywords = "matlabroot,root,installation,path",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::string_type),
    descriptor(crate::builtins::io::repl_fs::compat::MATLABROOT_DESCRIPTOR),
    builtin_path = "crate::builtins::io::repl_fs::compat"
)]
async fn matlabroot_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if !args.is_empty() {
        return Err(compat_error(
            "matlabroot",
            "matlabroot: too many input arguments",
        ));
    }
    let root = runtime_env::var("RUNMAT_ROOT")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(Path::to_path_buf))
        })
        .or_else(|| vfs::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("."));
    Ok(char_value(&path_to_string(&root)))
}

#[runtime_builtin(
    name = "pathsep",
    category = "io/repl_fs",
    summary = "Return the platform path-list separator.",
    keywords = "pathsep,path,separator",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::string_type),
    descriptor(crate::builtins::io::repl_fs::compat::PATHSEP_DESCRIPTOR),
    builtin_path = "crate::builtins::io::repl_fs::compat"
)]
async fn pathsep_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if !args.is_empty() {
        return Err(compat_error("pathsep", "pathsep: too many input arguments"));
    }
    Ok(char_value(&PATH_LIST_SEPARATOR.to_string()))
}

#[runtime_builtin(
    name = "system",
    category = "io/repl_fs",
    summary = "Execute an operating-system command.",
    keywords = "system,command,shell,process",
    accel = "cpu",
    suppress_auto_output = true,
    type_resolver(crate::builtins::io::type_resolvers::system_type),
    descriptor(crate::builtins::io::repl_fs::compat::SYSTEM_DESCRIPTOR),
    integer_audit(crate::builtins::io::repl_fs::compat::SYSTEM_INTEGER_AUDIT),
    builtin_path = "crate::builtins::io::repl_fs::compat"
)]
async fn system_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.is_empty() || args.len() > 2 {
        return Err(compat_error(
            "system",
            "system: expected command and optional echo flag",
        ));
    }
    let command = scalar_text(&args[0], "system", "command")?;
    let echo = if let Some(option) = args.get(1) {
        let option = scalar_text(option, "system", "option")?;
        if !option.eq_ignore_ascii_case("-echo") {
            return Err(compat_error(
                "system",
                "system: optional second argument must be '-echo'",
            ));
        }
        true
    } else {
        false
    };
    let result = run_system_command(&command)?;
    let requested = output_count::current_output_count();
    if (echo || requested == Some(0)) && !result.1.is_empty() {
        crate::console::record_console_line(crate::console::ConsoleStream::Stdout, &result.1);
    }
    let outputs = vec![Value::Num(result.0 as f64), char_value(&result.1)];
    if let Some(count) = requested {
        return Ok(output_count::output_list_with_padding(count, outputs));
    }
    Ok(Value::Num(result.0 as f64))
}

#[cfg(not(target_arch = "wasm32"))]
fn run_system_command(command: &str) -> BuiltinResult<(i32, String)> {
    #[cfg(windows)]
    let output = Command::new("cmd").args(["/C", command]).output();
    #[cfg(not(windows))]
    let output = Command::new("sh").args(["-c", command]).output();
    let output = output.map_err(|err| compat_error("system", format!("system: {err}")))?;
    let status = output.status.code().unwrap_or(1);
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    Ok((status, text))
}

#[cfg(target_arch = "wasm32")]
fn run_system_command(_command: &str) -> BuiltinResult<(i32, String)> {
    Ok((
        1,
        "system: process execution is unavailable in WebAssembly".to_string(),
    ))
}

fn truthy(value: &Value) -> bool {
    match value {
        Value::Bool(v) => *v,
        Value::Num(v) => *v != 0.0,
        Value::Int(v) => !v.is_zero(),
        Value::String(s) => !s.is_empty() && s != "0" && !s.eq_ignore_ascii_case("false"),
        Value::CharArray(ca) => {
            let s = char_row_to_string(ca);
            !s.is_empty() && s != "0" && !s.eq_ignore_ascii_case("false")
        }
        _ => true,
    }
}

#[runtime_builtin(
    name = "what",
    category = "io/repl_fs",
    summary = "Summarize MATLAB-related files in a folder.",
    keywords = "what,folder,files,classes,packages",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::struct_type),
    descriptor(crate::builtins::io::repl_fs::compat::WHAT_DESCRIPTOR),
    builtin_path = "crate::builtins::io::repl_fs::compat"
)]
async fn what_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let args = gather_args("what", &args).await?;
    if args.len() > 1 {
        return Err(compat_error("what", "what: too many input arguments"));
    }
    let folder = if let Some(value) = args.first() {
        value_to_path(value, "what", "folder")?
    } else {
        vfs::current_dir().map_err(|err| compat_error("what", format!("what: {err}")))?
    };
    let entries = vfs::read_dir_async(&folder)
        .await
        .map_err(|err| compat_error("what", format!("what: {err}")))?;
    let mut m = Vec::new();
    let mut mat = Vec::new();
    let mut mex = Vec::new();
    let mut classes = Vec::new();
    let mut packages = Vec::new();
    for entry in entries {
        let name = entry.file_name().to_string_lossy().into_owned();
        if entry.is_dir() {
            if let Some(stripped) = name.strip_prefix('@') {
                classes.push(stripped.to_string());
            } else if let Some(stripped) = name.strip_prefix('+') {
                packages.push(stripped.to_string());
            }
            continue;
        }
        if name.ends_with(".m") {
            m.push(name);
        } else if name.ends_with(".mat") {
            mat.push(name);
        } else if name.contains(".mex") {
            mex.push(name);
        }
    }
    let mut st = StructValue::new();
    st.insert("path", char_value(&path_to_string(&folder)));
    st.insert("m", cellstr(m)?);
    st.insert("mat", cellstr(mat)?);
    st.insert("mex", cellstr(mex)?);
    st.insert("classes", cellstr(classes)?);
    st.insert("packages", cellstr(packages)?);
    Ok(Value::Struct(st))
}

fn cellstr(values: Vec<String>) -> BuiltinResult<Value> {
    let len = values.len();
    Ok(Value::Cell(
        CellArray::new(values.into_iter().map(|s| char_value(&s)).collect(), len, 1)
            .map_err(|err| compat_error("cellstr", err))?,
    ))
}

#[runtime_builtin(
    name = "fileattrib",
    category = "io/repl_fs",
    summary = "Return file attribute metadata.",
    keywords = "fileattrib,file,folder,attributes,metadata",
    accel = "cpu",
    suppress_auto_output = true,
    type_resolver(crate::builtins::io::type_resolvers::fileattrib_type),
    descriptor(crate::builtins::io::repl_fs::compat::FILEATTRIB_DESCRIPTOR),
    integer_audit(crate::builtins::io::repl_fs::compat::FILEATTRIB_INTEGER_AUDIT),
    builtin_path = "crate::builtins::io::repl_fs::compat"
)]
async fn fileattrib_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.is_empty() || args.len() > 3 {
        return Err(compat_error(
            "fileattrib",
            "fileattrib: expected path and optional attribute flag",
        ));
    }
    let path = value_to_path(&args[0], "fileattrib", "path")?;
    if args.len() > 1 {
        apply_fileattrib_flags(&path, &args[1..]).await?;
    }
    match vfs::metadata_async(&path).await {
        Ok(meta) => Ok(output_list_for_count(vec![
            Value::Num(1.0),
            Value::Struct(fileattrib_struct(&path, &meta)),
        ])),
        Err(err) => Ok(output_list_for_count(vec![
            Value::Num(0.0),
            Value::Struct(error_struct(&err.to_string())),
        ])),
    }
}

async fn apply_fileattrib_flags(path: &Path, flags: &[Value]) -> BuiltinResult<()> {
    for flag_value in flags {
        let flag = scalar_text(flag_value, "fileattrib", "attribute")?;
        match flag.to_ascii_lowercase().as_str() {
            "+w" | "w" => vfs::set_readonly_async(path, false)
                .await
                .map_err(|err| compat_error("fileattrib", format!("fileattrib: {err}")))?,
            "-w" => vfs::set_readonly_async(path, true)
                .await
                .map_err(|err| compat_error("fileattrib", format!("fileattrib: {err}")))?,
            "+r" | "r" | "-r" => {
                // MATLAB exposes read flags, but the active filesystem provider only models
                // writable/read-only state. Treat read flags as accepted no-ops.
            }
            other => {
                return Err(compat_error(
                    "fileattrib",
                    format!("fileattrib: unsupported attribute flag '{other}'"),
                ));
            }
        }
    }
    Ok(())
}

fn fileattrib_struct(path: &Path, meta: &vfs::FsMetadata) -> StructValue {
    let mut st = StructValue::new();
    st.insert("Name", char_value(&path_to_string(path)));
    st.insert("archive", Value::Bool(false));
    st.insert("system", Value::Bool(false));
    st.insert("hidden", Value::Bool(is_hidden_path(path)));
    st.insert("directory", Value::Bool(meta.is_dir()));
    st.insert("UserRead", Value::Bool(true));
    st.insert("UserWrite", Value::Bool(!meta.is_readonly()));
    st.insert("UserExecute", Value::Bool(meta.is_dir()));
    st.insert("GroupRead", Value::Bool(true));
    st.insert("GroupWrite", Value::Bool(!meta.is_readonly()));
    st.insert("GroupExecute", Value::Bool(meta.is_dir()));
    st.insert("OtherRead", Value::Bool(true));
    st.insert("OtherWrite", Value::Bool(!meta.is_readonly()));
    st.insert("OtherExecute", Value::Bool(meta.is_dir()));
    st
}

fn error_struct(message: &str) -> StructValue {
    let mut st = StructValue::new();
    st.insert("message", char_value(message));
    st
}

fn is_hidden_path(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with('.'))
}

#[runtime_builtin(
    name = "getpref",
    category = "io/repl_fs",
    summary = "Read RunMat session preferences through getpref-shaped forms.",
    keywords = "getpref,preference,settings",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::getpref_type),
    descriptor(crate::builtins::io::repl_fs::compat::GETPREF_DESCRIPTOR),
    extensions(crate::builtins::io::repl_fs::compat::GETPREF_EXTENSIONS),
    integer_capabilities(crate::builtins::io::repl_fs::compat::GETPREF_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::io::repl_fs::compat"
)]
async fn getpref_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    match args.len() {
        0 => PREFS.with(|prefs| Ok(Value::Struct(all_prefs_struct(&prefs.borrow())))),
        1 => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &GETPREF_GROUP_QUERY_EXTENSION,
                "getpref",
            )?;
            let group = scalar_text(&args[0], "getpref", "group")?;
            PREFS.with(|prefs| {
                Ok(Value::Struct(group_prefs_struct(
                    prefs.borrow().get(&group),
                )))
            })
        }
        2 | 3 => getpref_named(&args),
        _ => Err(compat_error("getpref", "getpref: too many input arguments")),
    }
}

#[derive(Debug)]
enum PreferenceNames {
    Scalar(String),
    Array {
        names: Vec<String>,
        shape: Vec<usize>,
    },
}

fn getpref_named(args: &[Value]) -> BuiltinResult<Value> {
    let group = scalar_text(&args[0], "getpref", "group")?;
    let preferences = preference_names(&args[1])?;
    if args
        .get(2)
        .is_some_and(crate::dispatcher::value_contains_gpu)
    {
        return Err(compat_error(
            "getpref",
            "getpref: provider-resident default values are not supported",
        ));
    }

    match preferences {
        PreferenceNames::Scalar(preference) => {
            getpref_scalar(&group, &preference, args.get(2).cloned())
        }
        PreferenceNames::Array { names, shape } => {
            let defaults = match args.get(2) {
                None => None,
                Some(Value::Cell(cell)) if cell.shape == shape => Some(cell.data.clone()),
                Some(_) => {
                    return Err(compat_error(
                        "getpref",
                        "getpref: defaults for multiple preferences must be a cell array with matching shape",
                    ))
                }
            };
            let values = PREFS.with(|prefs| {
                let mut prefs = prefs.borrow_mut();
                let mut resolved = Vec::with_capacity(names.len());
                for (index, preference) in names.iter().enumerate() {
                    if let Some(value) = prefs
                        .get(&group)
                        .and_then(|group| group.get(preference))
                        .cloned()
                    {
                        resolved.push(value);
                    } else if let Some(defaults) = &defaults {
                        let value = defaults[index].clone();
                        prefs
                            .entry(group.clone())
                            .or_default()
                            .insert(preference.clone(), value.clone());
                        resolved.push(value);
                    } else {
                        return Err(compat_error(
                            "getpref",
                            format!("getpref: preference '{group}/{preference}' does not exist"),
                        ));
                    }
                }
                Ok(resolved)
            })?;
            CellArray::new_with_shape(values, shape)
                .map(Value::Cell)
                .map_err(|err| compat_error("getpref", format!("getpref: {err}")))
        }
    }
}

fn getpref_scalar(group: &str, preference: &str, default: Option<Value>) -> BuiltinResult<Value> {
    PREFS.with(|prefs| {
        let mut prefs = prefs.borrow_mut();
        if let Some(value) = prefs
            .get(group)
            .and_then(|group| group.get(preference))
            .cloned()
        {
            return Ok(value);
        }
        let Some(default) = default else {
            return Err(compat_error(
                "getpref",
                format!("getpref: preference '{group}/{preference}' does not exist"),
            ));
        };
        prefs
            .entry(group.to_string())
            .or_default()
            .insert(preference.to_string(), default.clone());
        Ok(default)
    })
}

fn preference_names(value: &Value) -> BuiltinResult<PreferenceNames> {
    match value {
        Value::String(text) => Ok(PreferenceNames::Scalar(text.clone())),
        Value::CharArray(array) if array.rows == 1 => Ok(PreferenceNames::Scalar(
            char_row_to_string(array),
        )),
        Value::StringArray(array) if array.data.len() == 1 => {
            Ok(PreferenceNames::Scalar(array.data[0].clone()))
        }
        Value::StringArray(array) => Ok(PreferenceNames::Array {
            names: array.data.clone(),
            shape: array.shape.clone(),
        }),
        Value::Cell(cell) => {
            let mut names = Vec::with_capacity(cell.data.len());
            for value in &cell.data {
                match value {
                    Value::CharArray(array) if array.rows == 1 => {
                        names.push(char_row_to_string(array));
                    }
                    _ => {
                        return Err(compat_error(
                            "getpref",
                            "getpref: preference cell entries must be character vectors",
                        ))
                    }
                }
            }
            Ok(PreferenceNames::Array {
                names,
                shape: cell.shape.clone(),
            })
        }
        _ => Err(compat_error(
            "getpref",
            "getpref: preference must be a character vector, string scalar, cell array of character vectors, or string array",
        )),
    }
}

fn all_prefs_struct(prefs: &BTreeMap<String, BTreeMap<String, Value>>) -> StructValue {
    let mut out = StructValue::new();
    for (group, values) in prefs {
        out.insert(
            group.clone(),
            Value::Struct(group_prefs_struct(Some(values))),
        );
    }
    out
}

fn group_prefs_struct(values: Option<&BTreeMap<String, Value>>) -> StructValue {
    let mut out = StructValue::new();
    if let Some(values) = values {
        for (name, value) in values {
            out.insert(name.clone(), value.clone());
        }
    }
    out
}

pub(crate) fn session_pref_text(group: &str, preference: &str) -> Option<String> {
    PREFS.with(|prefs| {
        prefs
            .borrow()
            .get(group)
            .and_then(|group| group.get(preference))
            .and_then(pref_value_to_text)
    })
}

fn pref_value_to_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::CharArray(array) if array.rows == 1 => Some(char_row_to_string(array)),
        Value::StringArray(array) if array.data.len() == 1 => Some(array.data[0].clone()),
        _ => None,
    }
}

#[runtime_builtin(
    name = "setpref",
    category = "io/repl_fs",
    summary = "Set RunMat session preferences.",
    keywords = "setpref,preference,settings",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::num_type),
    descriptor(crate::builtins::io::repl_fs::compat::SETPREF_DESCRIPTOR),
    integer_capabilities(crate::builtins::io::repl_fs::compat::SETPREF_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::io::repl_fs::compat"
)]
async fn setpref_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let args = gather_args("setpref", &args).await?;
    if args.len() != 3 {
        return Err(compat_error(
            "setpref",
            "setpref: expected group, preference, and value",
        ));
    }
    let group = scalar_text(&args[0], "setpref", "group")?;
    let pref = scalar_text(&args[1], "setpref", "preference")?;
    PREFS.with(|prefs| {
        prefs
            .borrow_mut()
            .entry(group)
            .or_default()
            .insert(pref, args[2].clone());
    });
    Ok(Value::Num(0.0))
}

#[runtime_builtin(
    name = "ispref",
    category = "io/repl_fs",
    summary = "Return true for existing RunMat session preferences.",
    keywords = "ispref,preference,settings,predicate",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::bool_type),
    descriptor(crate::builtins::io::repl_fs::compat::ISPREF_DESCRIPTOR),
    integer_audit(crate::builtins::io::repl_fs::compat::ISPREF_INTEGER_AUDIT),
    builtin_path = "crate::builtins::io::repl_fs::compat"
)]
async fn ispref_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.is_empty() || args.len() > 2 {
        return Err(compat_error(
            "ispref",
            "ispref: expected group and optional preference",
        ));
    }
    let group = scalar_text(&args[0], "ispref", "group")?;
    PREFS.with(|prefs| {
        let prefs = prefs.borrow();
        if args.len() == 1 {
            return Ok(Value::Bool(prefs.contains_key(&group)));
        }
        let pref = scalar_text(&args[1], "ispref", "preference")?;
        Ok(Value::Bool(
            prefs
                .get(&group)
                .is_some_and(|group| group.contains_key(&pref)),
        ))
    })
}

#[runtime_builtin(
    name = "rehash",
    category = "io/repl_fs",
    summary = "Refresh function/path caches.",
    keywords = "rehash,path,cache,refresh",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::num_type),
    descriptor(crate::builtins::io::repl_fs::compat::REHASH_DESCRIPTOR),
    builtin_path = "crate::builtins::io::repl_fs::compat"
)]
async fn rehash_builtin(_args: Vec<Value>) -> BuiltinResult<Value> {
    Ok(Value::Num(0.0))
}

#[runtime_builtin(
    name = "userpath",
    category = "io/repl_fs",
    summary = "Query or set the user path.",
    keywords = "userpath,path,user,home",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::string_type),
    descriptor(crate::builtins::io::repl_fs::compat::USERPATH_DESCRIPTOR),
    builtin_path = "crate::builtins::io::repl_fs::compat"
)]
async fn userpath_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let args = gather_args("userpath", &args).await?;
    match args.len() {
        0 => Ok(char_value(&default_userpath())),
        1 => {
            let command = scalar_text(&args[0], "userpath", "command")?;
            match command.to_ascii_lowercase().as_str() {
                "reset" => Ok(char_value(&default_userpath())),
                "clear" => Ok(char_value("")),
                path => Ok(char_value(path)),
            }
        }
        _ => Err(compat_error(
            "userpath",
            "userpath: too many input arguments",
        )),
    }
}

fn default_userpath() -> String {
    home_directory()
        .map(|home| home.join("Documents").join("MATLAB"))
        .map(|path| path_to_string(&path))
        .unwrap_or_default()
}

#[runtime_builtin(
    name = "restoredefaultpath",
    category = "io/repl_fs",
    summary = "Restore the RunMat search path to its default value.",
    keywords = "restoredefaultpath,path,default",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::string_type),
    descriptor(crate::builtins::io::repl_fs::compat::RESTOREDEFAULTPATH_DESCRIPTOR),
    builtin_path = "crate::builtins::io::repl_fs::compat"
)]
async fn restoredefaultpath_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if !args.is_empty() {
        return Err(compat_error(
            "restoredefaultpath",
            "restoredefaultpath: too many input arguments",
        ));
    }
    let root = match matlabroot_builtin(Vec::new()).await? {
        Value::CharArray(ca) => char_row_to_string(&ca),
        _ => String::new(),
    };
    set_path_string(&root);
    Ok(char_value(&root))
}

#[runtime_builtin(
    name = "memmapfile",
    category = "io/repl_fs",
    summary = "Map a file into a MATLAB-compatible memmapfile object.",
    keywords = "memmapfile,memory map,file,binary",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::struct_type),
    descriptor(crate::builtins::io::repl_fs::compat::MEMMAPFILE_DESCRIPTOR),
    extensions(crate::builtins::io::repl_fs::compat::MEMMAPFILE_EXTENSIONS),
    integer_capabilities(crate::builtins::io::repl_fs::compat::MEMMAPFILE_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::io::repl_fs::compat"
)]
async fn memmapfile_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if args
        .iter()
        .any(crate::builtins::common::validation::value_contains_explicit_gpu)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &MEMMAPFILE_EXPLICIT_GPU_EXTENSION,
            "memmapfile",
        )?;
    }
    for pair in args.get(1..).unwrap_or_default().chunks_exact(2) {
        let Some(name) = scalar_text(&pair[0], "memmapfile", "option").ok() else {
            continue;
        };
        if matches!(
            name.to_ascii_lowercase().as_str(),
            "writable" | "offset" | "repeat" | "format"
        ) && crate::builtins::common::validation::value_contains_native_integer_class(&pair[1])
        {
            crate::compatibility::ensure_builtin_extension_enabled(
                &MEMMAPFILE_INTEGER_CONTROLS_EXTENSION,
                "memmapfile",
            )?;
        }
    }
    let args = gather_args("memmapfile", &args).await?;
    if args.is_empty() {
        return Err(compat_error(
            "memmapfile",
            "memmapfile: filename is required",
        ));
    }
    let filename = value_to_path(&args[0], "memmapfile", "filename")?;
    let mut writable = false;
    let mut offset = 0usize;
    let mut format = MemmapFormat::default();
    let mut repeat: Option<usize> = None;
    let mut idx = 1usize;
    while idx < args.len() {
        if idx + 1 >= args.len() {
            return Err(compat_error(
                "memmapfile",
                "memmapfile: name-value options must be paired",
            ));
        }
        let name = scalar_text(&args[idx], "memmapfile", "option")?;
        match name.to_ascii_lowercase().as_str() {
            "writable" => writable = truthy(&args[idx + 1]),
            "offset" => offset = numeric_usize(&args[idx + 1], "memmapfile", "Offset")?,
            "format" => format = MemmapFormat::from_value(&args[idx + 1])?,
            "repeat" => repeat = Some(numeric_usize(&args[idx + 1], "memmapfile", "Repeat")?),
            _ => {}
        }
        idx += 2;
    }
    let bytes = vfs::read_async(&filename)
        .await
        .map_err(|err| compat_error("memmapfile", format!("memmapfile: {err}")))?;
    if offset > bytes.len() {
        return Err(compat_error(
            "memmapfile",
            "memmapfile: Offset exceeds file size",
        ));
    }
    let data_value = format.decode(&bytes[offset..], repeat)?;
    let mut object = ObjectInstance::new("memmapfile".to_string());
    object.properties.insert(
        "Filename".to_string(),
        char_value(&path_to_string(&filename)),
    );
    object
        .properties
        .insert("Writable".to_string(), Value::Bool(writable));
    object
        .properties
        .insert("Offset".to_string(), Value::Num(offset as f64));
    object
        .properties
        .insert("Format".to_string(), format.to_value()?);
    object.properties.insert(
        "Repeat".to_string(),
        repeat.map_or_else(|| char_value("Inf"), |value| Value::Num(value as f64)),
    );
    object.properties.insert("Data".to_string(), data_value);
    Ok(Value::Object(object))
}

#[derive(Clone, Debug)]
struct MemmapFormat {
    dtype: String,
    shape: Vec<usize>,
    field: Option<String>,
}

impl Default for MemmapFormat {
    fn default() -> Self {
        Self {
            dtype: "uint8".to_string(),
            shape: vec![1, 1],
            field: None,
        }
    }
}

impl MemmapFormat {
    fn from_value(value: &Value) -> BuiltinResult<Self> {
        match value {
            Value::String(_) | Value::CharArray(_) | Value::StringArray(_) => Ok(Self {
                dtype: scalar_text(value, "memmapfile", "Format")?.to_ascii_lowercase(),
                ..Self::default()
            }),
            Value::Cell(cell) => {
                if cell.data.is_empty() {
                    return Err(compat_error(
                        "memmapfile",
                        "memmapfile: Format cell is empty",
                    ));
                }
                let dtype =
                    scalar_text(&cell.data[0], "memmapfile", "Format type")?.to_ascii_lowercase();
                let shape = if cell.data.len() >= 2 {
                    shape_from_value(&cell.data[1])?
                } else {
                    vec![1, 1]
                };
                let field = if cell.data.len() >= 3 {
                    Some(scalar_text(&cell.data[2], "memmapfile", "Format field")?)
                } else {
                    None
                };
                Ok(Self {
                    dtype,
                    shape,
                    field,
                })
            }
            _ => Err(compat_error(
                "memmapfile",
                "memmapfile: Format must be a type name or format cell array",
            )),
        }
    }

    fn to_value(&self) -> BuiltinResult<Value> {
        if let Some(field) = &self.field {
            let shape_values = self.shape.iter().map(|value| *value as f64).collect();
            let shape = Value::Tensor(
                Tensor::new(shape_values, vec![1, self.shape.len()])
                    .map_err(|err| compat_error("memmapfile", err))?,
            );
            return Ok(Value::Cell(
                CellArray::new(
                    vec![char_value(&self.dtype), shape, char_value(field)],
                    1,
                    3,
                )
                .map_err(|err| compat_error("memmapfile", err))?,
            ));
        }
        Ok(char_value(&self.dtype))
    }

    fn decode(&self, bytes: &[u8], repeat: Option<usize>) -> BuiltinResult<Value> {
        let element_size = dtype_size(&self.dtype)?;
        let record_len = self
            .shape
            .iter()
            .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
            .ok_or_else(|| compat_error("memmapfile", "memmapfile: Format shape is too large"))?;
        let values_per_record = record_len.max(1);
        let available_values = bytes.len() / element_size;
        let total_values = repeat
            .map(|count| count.saturating_mul(values_per_record))
            .unwrap_or(available_values);
        let total_values = total_values.min(available_values);
        let mut shape = self.shape.clone();
        if let Some(repeat) = repeat.filter(|repeat| *repeat > 1) {
            shape.push(repeat);
        } else if shape.iter().product::<usize>() != total_values {
            shape = vec![total_values, 1];
        }
        let tensor = if let Some(storage) = decode_integer_storage(&self.dtype, bytes, total_values)
        {
            Value::Tensor(
                Tensor::new_integer(storage, shape)
                    .map_err(|err| compat_error("memmapfile", err))?,
            )
        } else {
            let mut data = Vec::with_capacity(total_values);
            for idx in 0..total_values {
                let start = idx * element_size;
                data.push(read_typed_value(
                    &self.dtype,
                    &bytes[start..start + element_size],
                )?);
            }
            Value::Tensor(
                Tensor::new_with_dtype(data, shape, tensor_dtype(&self.dtype))
                    .map_err(|err| compat_error("memmapfile", err))?,
            )
        };
        if let Some(field) = &self.field {
            let mut st = StructValue::new();
            st.insert(field.clone(), tensor);
            Ok(Value::Struct(st))
        } else {
            Ok(tensor)
        }
    }
}

fn shape_from_value(value: &Value) -> BuiltinResult<Vec<usize>> {
    match value {
        Value::Num(v) => Ok(vec![positive_platform_usize(*v, "Format shape")?, 1]),
        Value::Int(v) => Ok(vec![positive_integer_shape_dim(v)?, 1]),
        Value::Tensor(tensor) => {
            let len = tensor.shape.iter().product();
            let mut shape = Vec::with_capacity(len);
            for index in 0..len {
                let value = tensor.numeric_value_at(index).ok_or_else(|| {
                    compat_error("memmapfile", "memmapfile: invalid Format shape storage")
                })?;
                shape.push(shape_dim_from_numeric_scalar(value)?);
            }
            Ok(shape)
        }
        _ => Err(compat_error(
            "memmapfile",
            "memmapfile: Format shape must be a positive numeric vector",
        )),
    }
}

fn shape_dim_from_numeric_scalar(value: NumericScalar) -> BuiltinResult<usize> {
    match value {
        NumericScalar::F64(value) => positive_platform_usize(value, "Format shape"),
        NumericScalar::F32(value) => positive_platform_usize(f64::from(value), "Format shape"),
        integer => positive_integer_shape_dim(
            &integer
                .into_int_value()
                .expect("non-floating numeric scalar must be integer"),
        ),
    }
}

fn positive_integer_shape_dim(value: &runmat_builtins::IntValue) -> BuiltinResult<usize> {
    value
        .try_to_usize()
        .filter(|size| *size > 0)
        .ok_or_else(|| {
            compat_error(
                "memmapfile",
                "memmapfile: Format shape must contain positive integers",
            )
        })
}

fn positive_platform_usize(value: f64, arg: &str) -> BuiltinResult<usize> {
    let Some(size) = nonnegative_platform_usize(value) else {
        return Err(compat_error(
            "memmapfile",
            format!("memmapfile: {arg} must contain positive integers"),
        ));
    };
    if size == 0 {
        return Err(compat_error(
            "memmapfile",
            format!("memmapfile: {arg} must contain positive integers"),
        ));
    }
    Ok(size)
}

fn nonnegative_platform_usize(value: f64) -> Option<usize> {
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
        return None;
    }
    if value > usize::MAX as f64 || (usize::BITS == 64 && value == usize::MAX as f64) {
        return None;
    }
    Some(value as usize)
}

fn dtype_size(dtype: &str) -> BuiltinResult<usize> {
    match dtype.to_ascii_lowercase().as_str() {
        "uint8" | "int8" | "char" => Ok(1),
        "uint16" | "int16" => Ok(2),
        "uint32" | "int32" | "single" => Ok(4),
        "uint64" | "int64" | "double" => Ok(8),
        other => Err(compat_error(
            "memmapfile",
            format!("memmapfile: unsupported Format type '{other}'"),
        )),
    }
}

fn tensor_dtype(dtype: &str) -> NumericDType {
    match dtype.to_ascii_lowercase().as_str() {
        "single" => NumericDType::F32,
        "char" => NumericDType::U8,
        _ => NumericDType::F64,
    }
}

fn decode_integer_storage(dtype: &str, bytes: &[u8], value_count: usize) -> Option<IntegerStorage> {
    let bytes = &bytes[..value_count.checked_mul(dtype_size(dtype).ok()?)?];
    Some(match dtype.to_ascii_lowercase().as_str() {
        "int8" => IntegerStorage::I8(bytes.iter().map(|value| *value as i8).collect()),
        "uint8" => IntegerStorage::U8(bytes.to_vec()),
        "int16" => IntegerStorage::I16(
            bytes
                .chunks_exact(2)
                .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
                .collect(),
        ),
        "uint16" => IntegerStorage::U16(
            bytes
                .chunks_exact(2)
                .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                .collect(),
        ),
        "int32" => IntegerStorage::I32(
            bytes
                .chunks_exact(4)
                .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect(),
        ),
        "uint32" => IntegerStorage::U32(
            bytes
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect(),
        ),
        "int64" => IntegerStorage::I64(
            bytes
                .chunks_exact(8)
                .map(|chunk| {
                    i64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ])
                })
                .collect(),
        ),
        "uint64" => IntegerStorage::U64(
            bytes
                .chunks_exact(8)
                .map(|chunk| {
                    u64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ])
                })
                .collect(),
        ),
        _ => return None,
    })
}

fn read_typed_value(dtype: &str, bytes: &[u8]) -> BuiltinResult<f64> {
    Ok(match dtype.to_ascii_lowercase().as_str() {
        "uint8" | "char" => bytes[0] as f64,
        "int8" => i8::from_le_bytes([bytes[0]]) as f64,
        "uint16" => u16::from_le_bytes([bytes[0], bytes[1]]) as f64,
        "int16" => i16::from_le_bytes([bytes[0], bytes[1]]) as f64,
        "uint32" => u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64,
        "int32" => i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64,
        "uint64" => u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]) as f64,
        "int64" => i64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]) as f64,
        "single" => f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64,
        "double" => f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]),
        other => {
            return Err(compat_error(
                "memmapfile",
                format!("memmapfile: unsupported Format type '{other}'"),
            ));
        }
    })
}

fn numeric_usize(value: &Value, name: &str, arg: &str) -> BuiltinResult<usize> {
    match value {
        Value::Num(v) => nonnegative_platform_usize(*v).ok_or_else(|| {
            compat_error(name, format!("{name}: {arg} must be a nonnegative integer"))
        }),
        Value::Int(v) => v.try_to_usize().ok_or_else(|| {
            compat_error(name, format!("{name}: {arg} must be a nonnegative integer"))
        }),
        _ => Err(compat_error(
            name,
            format!("{name}: {arg} must be a nonnegative integer"),
        )),
    }
}

#[runtime_builtin(
    name = "winqueryreg",
    category = "io/repl_fs",
    summary = "Query values from the Windows registry.",
    keywords = "winqueryreg,windows,registry",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::string_type),
    descriptor(crate::builtins::io::repl_fs::compat::WINQUERYREG_DESCRIPTOR),
    builtin_path = "crate::builtins::io::repl_fs::compat"
)]
async fn winqueryreg_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let args = gather_args("winqueryreg", &args).await?;
    if args.len() < 2 || args.len() > 3 {
        return Err(compat_error(
            "winqueryreg",
            "winqueryreg: expected root, key, and optional value name",
        ));
    }
    let root = scalar_text(&args[0], "winqueryreg", "root")?;
    let key = scalar_text(&args[1], "winqueryreg", "key")?;
    let value = args
        .get(2)
        .map(|value| scalar_text(value, "winqueryreg", "value"))
        .transpose()?;
    query_windows_registry(&root, &key, value.as_deref())
}

#[cfg(windows)]
fn query_windows_registry(root: &str, key: &str, value: Option<&str>) -> BuiltinResult<Value> {
    let mut full = root.to_string();
    if !key.is_empty() {
        full.push('\\');
        full.push_str(key);
    }
    let mut cmd = Command::new("reg");
    cmd.args(["query", &full]);
    if let Some(value) = value {
        cmd.args(["/v", value]);
    }
    let output = cmd
        .output()
        .map_err(|err| compat_error("winqueryreg", format!("winqueryreg: {err}")))?;
    if !output.status.success() {
        return Err(compat_error(
            "winqueryreg",
            String::from_utf8_lossy(&output.stderr).trim().to_string(),
        ));
    }
    Ok(char_value(String::from_utf8_lossy(&output.stdout).trim()))
}

#[cfg(not(windows))]
fn query_windows_registry(_root: &str, _key: &str, _value: Option<&str>) -> BuiltinResult<Value> {
    Err(compat_error(
        "winqueryreg",
        "winqueryreg: Windows registry is unavailable on this platform",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK;

    fn run(value: impl std::future::Future<Output = BuiltinResult<Value>>) -> BuiltinResult<Value> {
        futures::executor::block_on(value)
    }

    fn unowned_resident_value() -> Value {
        Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: u32::MAX,
            buffer_id: u64::MAX,
        })
    }

    #[test]
    fn textual_filesystem_apis_are_integer_inapplicable() {
        assert_eq!(
            FILEPARTS_INTEGER_AUDIT.kind,
            BuiltinIntegerAuditKind::NotApplicable
        );
        assert_eq!(
            FILEATTRIB_INTEGER_AUDIT.kind,
            BuiltinIntegerAuditKind::NotApplicable
        );
    }

    #[test]
    fn ispref_rejects_integer_and_resident_controls_without_gather() {
        for integer in [
            runmat_builtins::IntValue::I8(-1),
            runmat_builtins::IntValue::I16(-2),
            runmat_builtins::IntValue::I32(-3),
            runmat_builtins::IntValue::I64(i64::MIN),
            runmat_builtins::IntValue::U8(1),
            runmat_builtins::IntValue::U16(2),
            runmat_builtins::IntValue::U32(3),
            runmat_builtins::IntValue::U64(u64::MAX),
        ] {
            assert!(run(ispref_builtin(vec![Value::Int(integer)])).is_err());
        }
        assert!(run(ispref_builtin(vec![unowned_resident_value()])).is_err());
    }

    #[test]
    fn fileparts_splits_folder_name_and_extension() {
        let value = run(fileparts_builtin(vec![Value::String(
            "/tmp/example.test.m".to_string(),
        )]))
        .unwrap();
        match value {
            Value::OutputList(values) => {
                assert_eq!(values[1], char_value("example.test"));
                assert_eq!(values[2], char_value(".m"));
            }
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn fileparts_rejects_numeric_and_resident_inputs_before_provider_access() {
        for invalid in [Value::Num(1.0), unowned_resident_value()] {
            let error = run(fileparts_builtin(vec![invalid])).expect_err("invalid text input");
            assert!(error.message().contains("filename must be"));
            assert!(!error.message().to_ascii_lowercase().contains("provider"));
        }
    }

    #[test]
    fn fileparts_preserves_scalar_string_array_text_input() {
        let input =
            runmat_builtins::StringArray::new(vec!["folder/example.m".to_string()], vec![1, 1])
                .expect("scalar string array");
        let value = run(fileparts_builtin(vec![Value::StringArray(input)])).expect("fileparts");
        let Value::OutputList(values) = value else {
            panic!("expected output list");
        };
        assert_eq!(values[1], char_value("example"));
        assert_eq!(values[2], char_value(".m"));
    }

    #[test]
    fn environment_predicates_and_unsetenv_share_runtime_env() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        runtime_env::set_var("RUNMAT_COMPAT_ENV_TEST", "1");
        assert_eq!(
            run(isenv_builtin(vec![Value::String(
                "RUNMAT_COMPAT_ENV_TEST".to_string()
            )]))
            .unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            run(unsetenv_builtin(vec![Value::String(
                "RUNMAT_COMPAT_ENV_TEST".to_string()
            )]))
            .unwrap(),
            Value::Num(0.0)
        );
        assert_eq!(
            run(isenv_builtin(vec![Value::String(
                "RUNMAT_COMPAT_ENV_TEST".to_string()
            )]))
            .unwrap(),
            Value::Bool(false)
        );
    }

    #[test]
    fn path_and_environment_predicates_reject_resident_numeric_inputs_before_access() {
        for result in [
            run(isfile_builtin(vec![unowned_resident_value()])),
            run(isfolder_builtin(vec![unowned_resident_value()])),
            run(isenv_builtin(vec![unowned_resident_value()])),
        ] {
            let error = result.expect_err("resident numeric input must be invalid text");
            assert!(error.message().contains("provider-resident numeric"));
            assert!(!error.message().contains("no acceleration provider"));
        }
    }

    #[test]
    fn preferences_round_trip() {
        run(setpref_builtin(vec![
            Value::String("runmatTest".to_string()),
            Value::String("answer".to_string()),
            Value::Num(42.0),
        ]))
        .unwrap();
        assert_eq!(
            run(ispref_builtin(vec![
                Value::String("runmatTest".to_string()),
                Value::String("answer".to_string())
            ]))
            .unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            run(getpref_builtin(vec![
                Value::String("runmatTest".to_string()),
                Value::String("answer".to_string())
            ]))
            .unwrap(),
            Value::Num(42.0)
        );
    }

    #[test]
    fn getpref_descriptor_and_integer_capabilities_cover_documented_forms() {
        assert_eq!(GETPREF_DESCRIPTOR.signatures.len(), 3);
        assert_eq!(GETPREF_INTEGER_CAPABILITIES.len(), 2);
        assert_eq!(GETPREF_INTEGER_CAPABILITIES[0].inputs[0].classes.len(), 8);
        assert_eq!(
            GETPREF_INTEGER_CAPABILITIES[0].output_class,
            BuiltinIntegerOutputClassRule::PreserveInput
        );
        assert_eq!(
            GETPREF_INTEGER_CAPABILITIES[0].backend,
            BuiltinIntegerBackendRule::HostOnly
        );
    }

    #[test]
    fn getpref_persists_and_preserves_every_integer_scalar_class() {
        for (index, value) in [
            Value::Int(runmat_builtins::IntValue::I8(i8::MIN)),
            Value::Int(runmat_builtins::IntValue::I16(i16::MIN)),
            Value::Int(runmat_builtins::IntValue::I32(i32::MIN)),
            Value::Int(runmat_builtins::IntValue::I64(i64::MIN)),
            Value::Int(runmat_builtins::IntValue::U8(u8::MAX)),
            Value::Int(runmat_builtins::IntValue::U16(u16::MAX)),
            Value::Int(runmat_builtins::IntValue::U32(u32::MAX)),
            Value::Int(runmat_builtins::IntValue::U64(u64::MAX)),
        ]
        .into_iter()
        .enumerate()
        {
            let preference = format!("scalar{index}");
            let created = run(getpref_builtin(vec![
                Value::String("getprefIntegerScalars".into()),
                Value::String(preference.clone()),
                value.clone(),
            ]))
            .expect("create integer preference");
            assert_eq!(created, value);
            let stored = run(getpref_builtin(vec![
                Value::String("getprefIntegerScalars".into()),
                Value::String(preference),
            ]))
            .expect("read integer preference");
            assert_eq!(stored, value);
        }
    }

    #[test]
    fn getpref_preserves_every_integer_array_class_and_shape() {
        for (index, storage) in [
            IntegerStorage::I8(vec![i8::MIN, i8::MAX]),
            IntegerStorage::I16(vec![i16::MIN, i16::MAX]),
            IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
            IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
            IntegerStorage::U8(vec![0, u8::MAX]),
            IntegerStorage::U16(vec![0, u16::MAX]),
            IntegerStorage::U32(vec![0, u32::MAX]),
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
        ]
        .into_iter()
        .enumerate()
        {
            let value = Value::Tensor(
                Tensor::new_integer(storage, vec![1, 2]).expect("integer preference array"),
            );
            let preference = format!("array{index}");
            let created = run(getpref_builtin(vec![
                Value::String("getprefIntegerArrays".into()),
                Value::String(preference.clone()),
                value.clone(),
            ]))
            .expect("create integer array preference");
            assert_eq!(created, value);
            let stored = run(getpref_builtin(vec![
                Value::String("getprefIntegerArrays".into()),
                Value::String(preference),
            ]))
            .expect("read integer array preference");
            assert_eq!(stored, value);
        }
    }

    #[test]
    fn getpref_existing_value_wins_over_default_without_conversion() {
        let existing = Value::Int(runmat_builtins::IntValue::U64(u64::MAX));
        run(getpref_builtin(vec![
            Value::String("getprefExistingWins".into()),
            Value::String("value".into()),
            existing.clone(),
        ]))
        .expect("create preference");
        let result = run(getpref_builtin(vec![
            Value::String("getprefExistingWins".into()),
            Value::String("value".into()),
            Value::Num(1.0),
        ]))
        .expect("read existing preference");
        assert_eq!(result, existing);
    }

    #[test]
    fn getpref_preserves_arbitrary_object_values_and_zero_argument_structure() {
        let mut object = ObjectInstance::new("PreferencePayload".to_string());
        object.properties.insert(
            "revision".to_string(),
            Value::Int(runmat_builtins::IntValue::U64(u64::MAX)),
        );
        let value = Value::Object(object);
        let created = run(getpref_builtin(vec![
            Value::String("getprefObjects".into()),
            Value::String("payload".into()),
            value.clone(),
        ]))
        .expect("create object preference");
        assert_eq!(created, value);
        let stored = run(getpref_builtin(vec![
            Value::String("getprefObjects".into()),
            Value::String("payload".into()),
        ]))
        .expect("read object preference");
        assert_eq!(stored, value);

        let all = run(getpref_builtin(Vec::new())).expect("all preferences");
        let Value::Struct(all) = all else {
            panic!("expected preference structure");
        };
        assert!(all.fields.contains_key("getprefObjects"));
    }

    #[test]
    fn getpref_multiple_names_preserve_query_shape_and_persist_defaults() {
        let names = Value::StringArray(
            runmat_builtins::StringArray::new(vec!["signed".into(), "unsigned".into()], vec![2, 1])
                .expect("preference names"),
        );
        let defaults = Value::Cell(
            CellArray::new_with_shape(
                vec![
                    Value::Int(runmat_builtins::IntValue::I64(i64::MIN)),
                    Value::Int(runmat_builtins::IntValue::U64(u64::MAX)),
                ],
                vec![2, 1],
            )
            .expect("defaults"),
        );
        let created = run(getpref_builtin(vec![
            Value::String("getprefMultiple".into()),
            names.clone(),
            defaults.clone(),
        ]))
        .expect("create multiple preferences");
        assert_eq!(created, defaults);
        let stored = run(getpref_builtin(vec![
            Value::String("getprefMultiple".into()),
            names,
        ]))
        .expect("read multiple preferences");
        assert_eq!(stored, defaults);

        let cellstr = Value::Cell(
            CellArray::new_with_shape(
                vec![char_value("signed"), char_value("unsigned")],
                vec![1, 2],
            )
            .expect("cellstr names"),
        );
        let stored = run(getpref_builtin(vec![
            Value::String("getprefMultiple".into()),
            cellstr,
        ]))
        .expect("read cellstr preferences");
        let Value::Cell(stored) = stored else {
            panic!("expected cell result");
        };
        assert_eq!(stored.shape, vec![1, 2]);
    }

    #[test]
    fn getpref_group_query_is_a_separately_gated_extension() {
        let strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = run(getpref_builtin(vec![Value::String("group".into())]))
            .expect_err("strict group query");
        assert_eq!(
            error.identifier(),
            GETPREF_GROUP_QUERY_EXTENSION.error_identifier
        );
        drop(strict);

        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = run(getpref_builtin(vec![Value::String("group".into())]))
            .expect("extension group query");
        assert!(matches!(result, Value::Struct(_)));
    }

    #[test]
    fn getpref_rejects_resident_controls_and_defaults_before_provider_access() {
        let resident = || unowned_resident_value();
        let error = run(getpref_builtin(vec![
            resident(),
            Value::String("name".into()),
        ]))
        .expect_err("resident group");
        assert!(error.message().contains("group must be"));

        let error = run(getpref_builtin(vec![
            Value::String("getprefResident".into()),
            resident(),
        ]))
        .expect_err("resident preference name");
        assert!(error.message().contains("preference must be"));

        let error = run(getpref_builtin(vec![
            Value::String("getprefResident".into()),
            Value::String("value".into()),
            resident(),
        ]))
        .expect_err("resident default");
        assert!(error.message().contains("provider-resident default"));
    }

    #[test]
    fn getpref_rejects_every_integer_class_in_text_control_positions() {
        for value in [
            Value::Int(runmat_builtins::IntValue::I8(1)),
            Value::Int(runmat_builtins::IntValue::I16(1)),
            Value::Int(runmat_builtins::IntValue::I32(1)),
            Value::Int(runmat_builtins::IntValue::I64(1)),
            Value::Int(runmat_builtins::IntValue::U8(1)),
            Value::Int(runmat_builtins::IntValue::U16(1)),
            Value::Int(runmat_builtins::IntValue::U32(1)),
            Value::Int(runmat_builtins::IntValue::U64(1)),
        ] {
            let error = run(getpref_builtin(vec![
                value.clone(),
                Value::String("name".into()),
            ]))
            .expect_err("integer group");
            assert!(error.message().contains("group must be"));
            let error = run(getpref_builtin(vec![
                Value::String("getprefIntegerControls".into()),
                value,
            ]))
            .expect_err("integer preference name");
            assert!(error.message().contains("preference must be"));
        }
    }

    #[test]
    fn memmapfile_decodes_named_typed_format() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        let path = std::env::temp_dir().join("runmat_memmapfile_test.bin");
        std::fs::write(&path, [1u8, 0, 2, 0]).unwrap();
        let shape = Value::Tensor(Tensor::new(vec![2.0, 1.0], vec![1, 2]).unwrap());
        let fmt = Value::Cell(
            CellArray::new(
                vec![char_value("uint16"), shape, char_value("samples")],
                1,
                3,
            )
            .unwrap(),
        );
        let value = run(memmapfile_builtin(vec![
            Value::String(path_to_string(&path)),
            char_value("Format"),
            fmt,
        ]))
        .unwrap();
        let Value::Object(object) = value else {
            panic!("expected object");
        };
        let Some(Value::Struct(data)) = object.properties.get("Data") else {
            panic!("expected Data struct");
        };
        let Some(Value::Tensor(samples)) = data.fields.get("samples") else {
            panic!("expected samples tensor");
        };
        assert_eq!(samples.shape, vec![2, 1]);
        assert_eq!(
            samples.integer_storage(),
            Some(&IntegerStorage::U16(vec![1, 2]))
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn memmapfile_integer_controls_are_gated_before_file_access() {
        assert_eq!(MEMMAPFILE_INTEGER_CAPABILITIES.len(), 3);
        let strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = run(memmapfile_builtin(vec![
            Value::String("definitely-missing.bin".into()),
            Value::String("Offset".into()),
            Value::Int(runmat_builtins::IntValue::U64(0)),
        ]))
        .expect_err("typed integer control must be gated");
        assert_eq!(
            error.identifier(),
            MEMMAPFILE_INTEGER_CONTROLS_EXTENSION.error_identifier
        );
        drop(strict);
    }

    #[test]
    fn memmapfile_typed_shape_offset_and_repeat_parsers_are_exact() {
        use runmat_builtins::{IntValue, IntegerStorage};

        assert_eq!(
            shape_from_value(&Value::Int(IntValue::U16(7))).unwrap(),
            vec![7, 1]
        );
        let shape = Tensor::new_integer(IntegerStorage::U64(vec![2, 3]), vec![1, 2])
            .expect("typed shape vector");
        assert_eq!(shape_from_value(&Value::Tensor(shape)).unwrap(), vec![2, 3]);
        let single_shape = Tensor::from_f32(vec![4.0, 5.0], vec![1, 2]).expect("single shape");
        assert_eq!(
            shape_from_value(&Value::Tensor(single_shape)).unwrap(),
            vec![4, 5]
        );
        assert!(shape_from_value(&Value::Int(IntValue::I8(-1))).is_err());
        assert!(shape_from_value(&Value::Int(IntValue::U8(0))).is_err());
        assert!(shape_from_value(&Value::Num(2.5)).is_err());
        assert!(shape_from_value(&Value::Num(usize::MAX as f64 + 1.0)).is_err());

        assert_eq!(
            numeric_usize(&Value::Int(IntValue::U16(64)), "memmapfile", "Offset").unwrap(),
            64
        );
        assert!(numeric_usize(&Value::Int(IntValue::I8(-1)), "memmapfile", "Offset").is_err());
        assert!(numeric_usize(&Value::Num(2.5), "memmapfile", "Offset").is_err());
        assert!(
            numeric_usize(&Value::Num(usize::MAX as f64 + 1.0), "memmapfile", "Offset").is_err()
        );

        let maximum = numeric_usize(&Value::Int(IntValue::U64(u64::MAX)), "memmapfile", "Repeat");
        if usize::BITS == 64 {
            assert_eq!(maximum.unwrap(), usize::MAX);
        } else {
            assert!(maximum.is_err());
        }
    }

    #[test]
    fn memmapfile_decodes_all_integer_formats_without_float_coercion() {
        let cases = [
            (
                "int8",
                vec![i8::MIN as u8, i8::MAX as u8],
                IntegerStorage::I8(vec![i8::MIN, i8::MAX]),
            ),
            (
                "uint8",
                vec![0, u8::MAX],
                IntegerStorage::U8(vec![0, u8::MAX]),
            ),
            (
                "int16",
                [i16::MIN.to_le_bytes(), i16::MAX.to_le_bytes()].concat(),
                IntegerStorage::I16(vec![i16::MIN, i16::MAX]),
            ),
            (
                "uint16",
                [0_u16.to_le_bytes(), u16::MAX.to_le_bytes()].concat(),
                IntegerStorage::U16(vec![0, u16::MAX]),
            ),
            (
                "int32",
                [i32::MIN.to_le_bytes(), i32::MAX.to_le_bytes()].concat(),
                IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
            ),
            (
                "uint32",
                [0_u32.to_le_bytes(), u32::MAX.to_le_bytes()].concat(),
                IntegerStorage::U32(vec![0, u32::MAX]),
            ),
            (
                "int64",
                [i64::MIN.to_le_bytes(), i64::MAX.to_le_bytes()].concat(),
                IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
            ),
            (
                "uint64",
                [0_u64.to_le_bytes(), u64::MAX.to_le_bytes()].concat(),
                IntegerStorage::U64(vec![0, u64::MAX]),
            ),
        ];

        for (dtype, bytes, expected) in cases {
            let mapped = MemmapFormat {
                dtype: dtype.to_string(),
                shape: vec![1, 2],
                field: None,
            }
            .decode(&bytes, None)
            .unwrap();
            let Value::Tensor(tensor) = mapped else {
                panic!("expected tensor");
            };
            assert_eq!(tensor.shape, vec![1, 2], "{dtype}");
            assert_eq!(tensor.integer_storage(), Some(&expected), "{dtype}");
        }
    }

    #[test]
    fn memmapfile_preserves_integer_storage_for_repeated_records() {
        let bytes = [
            1_u32.to_le_bytes(),
            u32::MAX.to_le_bytes(),
            3_u32.to_le_bytes(),
            4_u32.to_le_bytes(),
        ]
        .concat();
        let mapped = MemmapFormat {
            dtype: "uint32".to_string(),
            shape: vec![2, 1],
            field: None,
        }
        .decode(&bytes, Some(2))
        .unwrap();
        let Value::Tensor(tensor) = mapped else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![2, 1, 2]);
        assert_eq!(
            tensor.integer_storage(),
            Some(&IntegerStorage::U32(vec![1, u32::MAX, 3, 4]))
        );
    }

    #[test]
    fn system_two_outputs_returns_status_and_text() {
        #[cfg(windows)]
        let command = "echo hello";
        #[cfg(not(windows))]
        let command = "printf hello";
        let _guard = crate::output_count::push_output_count(Some(2));
        let value = run(system_builtin(vec![Value::String(command.to_string())])).unwrap();
        let Value::OutputList(values) = value else {
            panic!("expected output list");
        };
        assert_eq!(values[0], Value::Num(0.0));
        assert_eq!(values[1], char_value("hello"));
    }

    #[test]
    fn system_rejects_integer_commands_before_process_access() {
        let error = run(system_builtin(vec![Value::Int(
            runmat_builtins::IntValue::U64(u64::MAX),
        )]))
        .expect_err("integer command must reject");
        assert!(error.message().contains("command must be"));
    }

    #[test]
    fn system_rejects_integer_echo_options_before_process_access() {
        let error = run(system_builtin(vec![
            Value::String("printf should-not-run".to_string()),
            Value::Int(runmat_builtins::IntValue::U8(1)),
        ]))
        .expect_err("integer option must reject");
        assert!(error.message().contains("option must be"));
    }

    #[test]
    fn fileattrib_can_toggle_user_write_flag() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        let path = std::env::temp_dir().join("runmat_fileattrib_test.txt");
        std::fs::write(&path, "data").unwrap();
        run(fileattrib_builtin(vec![
            Value::String(path_to_string(&path)),
            char_value("-w"),
        ]))
        .unwrap();
        let readonly = std::fs::metadata(&path).unwrap().permissions().readonly();
        assert!(readonly);
        run(fileattrib_builtin(vec![
            Value::String(path_to_string(&path)),
            char_value("+w"),
        ]))
        .unwrap();
        let readonly = std::fs::metadata(&path).unwrap().permissions().readonly();
        assert!(!readonly);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn fileattrib_rejects_numeric_and_resident_inputs_before_external_access() {
        let numeric =
            run(fileattrib_builtin(vec![Value::Num(1.0)])).expect_err("numeric path must reject");
        assert!(numeric.message().contains("path must be"));

        let numeric_flag = run(fileattrib_builtin(vec![
            Value::String("missing-fileattrib-path".to_string()),
            Value::Num(1.0),
        ]))
        .expect_err("numeric attribute must reject");
        assert!(numeric_flag.message().contains("attribute must be"));

        let resident_path = run(fileattrib_builtin(vec![unowned_resident_value()]))
            .expect_err("resident path must reject");
        assert!(resident_path.message().contains("path must be"));
        assert!(!resident_path
            .message()
            .to_ascii_lowercase()
            .contains("provider"));

        let resident_flag = run(fileattrib_builtin(vec![
            Value::String("missing-fileattrib-path".to_string()),
            unowned_resident_value(),
        ]))
        .expect_err("resident attribute must reject");
        assert!(resident_flag.message().contains("attribute must be"));
        assert!(!resident_flag
            .message()
            .to_ascii_lowercase()
            .contains("provider"));
    }
}
