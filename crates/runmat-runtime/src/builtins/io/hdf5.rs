//! MATLAB-compatible HDF5 structured file I/O builtins.

use std::path::{Path, PathBuf};

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    IntValue, IntegerStorage, LogicalArray, NumericScalar, NumericStorage, StringArray, Tensor,
    Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::fs::expand_user_path;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const H5READ_NAME: &str = "h5read";
const H5WRITE_NAME: &str = "h5write";
const H5WRITEATT_NAME: &str = "h5writeatt";
const H5INFO_NAME: &str = "h5info";
const H5DISP_NAME: &str = "h5disp";
const HDF5READ_NAME: &str = "hdf5read";
const HDF5WRITE_NAME: &str = "hdf5write";
const HDF5INFO_NAME: &str = "hdf5info";

const OUT_DATA: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "data",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Dataset contents.",
}];
const OUT_INFO: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "info",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "HDF5 metadata structure.",
}];
const OUT_TEXT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "text",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Textual HDF5 metadata display.",
}];
const OUT_NONE: [BuiltinParamDescriptor; 0] = [];

const IN_FILE_DATASET: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "HDF5 file name.",
    },
    BuiltinParamDescriptor {
        name: "dataset",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Dataset path in the file.",
    },
];
const IN_FILE_DATASET_SELECTION: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "HDF5 file name.",
    },
    BuiltinParamDescriptor {
        name: "dataset",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Dataset path in the file.",
    },
    BuiltinParamDescriptor {
        name: "start",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "One-based starting subscript for each HDF5 dimension.",
    },
    BuiltinParamDescriptor {
        name: "count",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Number of elements to read in each dimension.",
    },
    BuiltinParamDescriptor {
        name: "stride",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Stride for each HDF5 dimension.",
    },
];
const IN_FILE_DATASET_VALUE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "HDF5 file name.",
    },
    BuiltinParamDescriptor {
        name: "dataset",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Dataset path in the file.",
    },
    BuiltinParamDescriptor {
        name: "data",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Data to write.",
    },
];
const IN_FILE_DATASET_VALUE_SELECTION: [BuiltinParamDescriptor; 6] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "HDF5 file name.",
    },
    BuiltinParamDescriptor {
        name: "dataset",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Dataset path in the file.",
    },
    BuiltinParamDescriptor {
        name: "data",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Data to write.",
    },
    BuiltinParamDescriptor {
        name: "start",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "One-based starting subscript for each HDF5 dimension.",
    },
    BuiltinParamDescriptor {
        name: "count",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Number of elements to write in each dimension.",
    },
    BuiltinParamDescriptor {
        name: "stride",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Stride for each HDF5 dimension.",
    },
];
const IN_FILE_LOCATION_ATTRIBUTE_VALUE: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "HDF5 file name.",
    },
    BuiltinParamDescriptor {
        name: "location",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Group or dataset path.",
    },
    BuiltinParamDescriptor {
        name: "attribute",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Attribute name.",
    },
    BuiltinParamDescriptor {
        name: "data",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Attribute value to write.",
    },
];
const IN_FILE_ONLY: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "HDF5 file name.",
}];
const IN_FILE_OPTIONAL_LOCATION: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "HDF5 file name.",
    },
    BuiltinParamDescriptor {
        name: "location",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("/"),
        description: "Group or dataset path.",
    },
];

const H5READ_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "data = h5read(filename, dataset)",
        inputs: &IN_FILE_DATASET,
        outputs: &OUT_DATA,
    },
    BuiltinSignatureDescriptor {
        label: "data = h5read(filename, dataset, start, count, stride)",
        inputs: &IN_FILE_DATASET_SELECTION,
        outputs: &OUT_DATA,
    },
];
const H5WRITE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "h5write(filename, dataset, data)",
        inputs: &IN_FILE_DATASET_VALUE,
        outputs: &OUT_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "h5write(filename, dataset, data, start, count, stride)",
        inputs: &IN_FILE_DATASET_VALUE_SELECTION,
        outputs: &OUT_NONE,
    },
];
const H5WRITEATT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "h5writeatt(filename, location, attribute, data)",
    inputs: &IN_FILE_LOCATION_ATTRIBUTE_VALUE,
    outputs: &OUT_NONE,
}];
const H5INFO_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "info = h5info(filename)",
        inputs: &IN_FILE_ONLY,
        outputs: &OUT_INFO,
    },
    BuiltinSignatureDescriptor {
        label: "info = h5info(filename, location)",
        inputs: &IN_FILE_OPTIONAL_LOCATION,
        outputs: &OUT_INFO,
    },
];
const H5DISP_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "text = h5disp(filename)",
        inputs: &IN_FILE_ONLY,
        outputs: &OUT_TEXT,
    },
    BuiltinSignatureDescriptor {
        label: "text = h5disp(filename, location)",
        inputs: &IN_FILE_OPTIONAL_LOCATION,
        outputs: &OUT_TEXT,
    },
];

const ERR_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HDF5.ARGUMENT",
    identifier: Some("RunMat:hdf5:InvalidArgument"),
    when: "Filename, object paths, selection vectors, or values are malformed.",
    message: "hdf5: invalid argument",
};
const ERR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HDF5.IO",
    identifier: Some("RunMat:hdf5:Io"),
    when: "The HDF5 file cannot be opened, read, written, or traversed.",
    message: "hdf5: file I/O failed",
};
const ERR_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HDF5.UNSUPPORTED",
    identifier: Some("RunMat:hdf5:UnsupportedType"),
    when: "The HDF5 type or MATLAB value is outside the supported dense numeric, logical, or string subset.",
    message: "hdf5: unsupported type",
};
const ERR_WASM: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HDF5.WASM_UNSUPPORTED",
    identifier: Some("RunMat:hdf5:UnsupportedTarget"),
    when: "Native HDF5 library support is unavailable on wasm targets.",
    message: "hdf5: native HDF5 is unavailable on this target",
};
const ERRORS: [BuiltinErrorDescriptor; 4] = [ERR_ARGUMENT, ERR_IO, ERR_UNSUPPORTED, ERR_WASM];

macro_rules! descriptor {
    ($name:ident, $sigs:expr, $outputs:expr) => {
        pub const $name: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &$sigs,
            output_mode: $outputs,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };
    };
}

descriptor!(
    H5READ_DESCRIPTOR,
    H5READ_SIGNATURES,
    BuiltinOutputMode::Fixed
);
descriptor!(
    H5WRITE_DESCRIPTOR,
    H5WRITE_SIGNATURES,
    BuiltinOutputMode::Fixed
);
descriptor!(
    H5WRITEATT_DESCRIPTOR,
    H5WRITEATT_SIGNATURES,
    BuiltinOutputMode::Fixed
);
descriptor!(
    H5INFO_DESCRIPTOR,
    H5INFO_SIGNATURES,
    BuiltinOutputMode::Fixed
);
descriptor!(
    H5DISP_DESCRIPTOR,
    H5DISP_SIGNATURES,
    BuiltinOutputMode::Fixed
);
descriptor!(
    HDF5READ_DESCRIPTOR,
    H5READ_SIGNATURES,
    BuiltinOutputMode::Fixed
);
descriptor!(
    HDF5WRITE_DESCRIPTOR,
    H5WRITE_SIGNATURES,
    BuiltinOutputMode::Fixed
);
descriptor!(
    HDF5INFO_DESCRIPTOR,
    H5INFO_SIGNATURES,
    BuiltinOutputMode::Fixed
);

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::hdf5")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "hdf5",
    op_kind: GpuOpKind::Custom("io-hdf5"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Runs on the host; HDF5 parsing and serialization are native file I/O operations.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::hdf5")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "hdf5",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not eligible for fusion; performs host-side HDF5 file I/O.",
};

fn hdf5_type(_args: &[Type], _ctx: &runmat_builtins::ResolveContext) -> Type {
    Type::Unknown
}

fn hdf5_error(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(builtin);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_control_flow(builtin: &'static str, err: RuntimeError) -> RuntimeError {
    let identifier = err.identifier().map(|value| value.to_string());
    let message = err.message().to_string();
    let mut builder = build_runtime_error(message)
        .with_builtin(builtin)
        .with_source(err);
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn string_scalar(value: &Value, builtin: &'static str, label: &str) -> BuiltinResult<String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        _ => Err(hdf5_error(
            builtin,
            &ERR_ARGUMENT,
            format!("{builtin}: expected {label} as a string scalar or character vector"),
        )),
    }
}

fn resolve_path(value: &Value, builtin: &'static str) -> BuiltinResult<PathBuf> {
    let raw = string_scalar(value, builtin, "filename")?;
    if raw.trim().is_empty() {
        return Err(hdf5_error(
            builtin,
            &ERR_ARGUMENT,
            format!("{builtin}: filename must not be empty"),
        ));
    }
    let expanded = expand_user_path(raw.trim(), builtin)
        .map_err(|msg| hdf5_error(builtin, &ERR_ARGUMENT, msg))?;
    Ok(Path::new(&expanded).to_path_buf())
}

fn hdf5_path(value: &Value, builtin: &'static str, label: &str) -> BuiltinResult<String> {
    let raw = string_scalar(value, builtin, label)?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(hdf5_error(
            builtin,
            &ERR_ARGUMENT,
            format!("{builtin}: {label} must not be empty"),
        ));
    }
    Ok(if trimmed.starts_with('/') {
        trimmed.to_string()
    } else {
        format!("/{trimmed}")
    })
}

async fn gather_args(builtin: &'static str, args: &[Value]) -> BuiltinResult<Vec<Value>> {
    let mut gathered = Vec::with_capacity(args.len());
    for arg in args {
        gathered.push(
            gather_if_needed_async(arg)
                .await
                .map_err(|err| map_control_flow(builtin, err))?,
        );
    }
    Ok(gathered)
}

#[runtime_builtin(
    name = "h5read",
    category = "io/hdf5",
    summary = "Read a dataset from an HDF5 file.",
    keywords = "h5read,hdf5,dataset,file,io",
    accel = "cpu",
    type_resolver(crate::builtins::io::hdf5::hdf5_type),
    descriptor(crate::builtins::io::hdf5::H5READ_DESCRIPTOR),
    builtin_path = "crate::builtins::io::hdf5"
)]
async fn h5read_builtin(filename: Value, dataset: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let filename = gather_if_needed_async(&filename)
        .await
        .map_err(|err| map_control_flow(H5READ_NAME, err))?;
    let dataset = gather_if_needed_async(&dataset)
        .await
        .map_err(|err| map_control_flow(H5READ_NAME, err))?;
    let rest = gather_args(H5READ_NAME, &rest).await?;
    let path = resolve_path(&filename, H5READ_NAME)?;
    let dataset_path = hdf5_path(&dataset, H5READ_NAME, "dataset")?;
    imp::read_dataset(H5READ_NAME, path, dataset_path, rest)
}

#[runtime_builtin(
    name = "hdf5read",
    category = "io/hdf5",
    summary = "Read a dataset from an HDF5 file using the legacy HDF5 API name.",
    keywords = "hdf5read,h5read,hdf5,dataset,file,io,legacy",
    accel = "cpu",
    type_resolver(crate::builtins::io::hdf5::hdf5_type),
    descriptor(crate::builtins::io::hdf5::HDF5READ_DESCRIPTOR),
    builtin_path = "crate::builtins::io::hdf5"
)]
async fn hdf5read_builtin(
    filename: Value,
    dataset: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let filename = gather_if_needed_async(&filename)
        .await
        .map_err(|err| map_control_flow(HDF5READ_NAME, err))?;
    let dataset = gather_if_needed_async(&dataset)
        .await
        .map_err(|err| map_control_flow(HDF5READ_NAME, err))?;
    let rest = gather_args(HDF5READ_NAME, &rest).await?;
    let path = resolve_path(&filename, HDF5READ_NAME)?;
    let dataset_path = hdf5_path(&dataset, HDF5READ_NAME, "dataset")?;
    imp::read_dataset(HDF5READ_NAME, path, dataset_path, rest)
}

#[runtime_builtin(
    name = "h5write",
    category = "io/hdf5",
    summary = "Write a dense dataset to an HDF5 file.",
    keywords = "h5write,hdf5,dataset,file,io",
    accel = "cpu",
    type_resolver(crate::builtins::io::hdf5::hdf5_type),
    descriptor(crate::builtins::io::hdf5::H5WRITE_DESCRIPTOR),
    builtin_path = "crate::builtins::io::hdf5"
)]
async fn h5write_builtin(
    filename: Value,
    dataset: Value,
    data: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let filename = gather_if_needed_async(&filename)
        .await
        .map_err(|err| map_control_flow(H5WRITE_NAME, err))?;
    let dataset = gather_if_needed_async(&dataset)
        .await
        .map_err(|err| map_control_flow(H5WRITE_NAME, err))?;
    let data = gather_if_needed_async(&data)
        .await
        .map_err(|err| map_control_flow(H5WRITE_NAME, err))?;
    let rest = gather_args(H5WRITE_NAME, &rest).await?;
    let path = resolve_path(&filename, H5WRITE_NAME)?;
    let dataset_path = hdf5_path(&dataset, H5WRITE_NAME, "dataset")?;
    imp::write_dataset(H5WRITE_NAME, path, dataset_path, data, rest)?;
    Ok(Value::OutputList(Vec::new()))
}

#[runtime_builtin(
    name = "hdf5write",
    category = "io/hdf5",
    summary = "Write a dense dataset to an HDF5 file using the legacy HDF5 API name.",
    keywords = "hdf5write,h5write,hdf5,dataset,file,io,legacy",
    accel = "cpu",
    type_resolver(crate::builtins::io::hdf5::hdf5_type),
    descriptor(crate::builtins::io::hdf5::HDF5WRITE_DESCRIPTOR),
    builtin_path = "crate::builtins::io::hdf5"
)]
async fn hdf5write_builtin(
    filename: Value,
    dataset: Value,
    data: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let filename = gather_if_needed_async(&filename)
        .await
        .map_err(|err| map_control_flow(HDF5WRITE_NAME, err))?;
    let dataset = gather_if_needed_async(&dataset)
        .await
        .map_err(|err| map_control_flow(HDF5WRITE_NAME, err))?;
    let data = gather_if_needed_async(&data)
        .await
        .map_err(|err| map_control_flow(HDF5WRITE_NAME, err))?;
    let rest = gather_args(HDF5WRITE_NAME, &rest).await?;
    let path = resolve_path(&filename, HDF5WRITE_NAME)?;
    let dataset_path = hdf5_path(&dataset, HDF5WRITE_NAME, "dataset")?;
    imp::write_dataset(HDF5WRITE_NAME, path, dataset_path, data, rest)?;
    Ok(Value::OutputList(Vec::new()))
}

#[runtime_builtin(
    name = "h5writeatt",
    category = "io/hdf5",
    summary = "Write an attribute on an HDF5 file, group, or dataset.",
    keywords = "h5writeatt,hdf5,attribute,dataset,group,file,io",
    accel = "cpu",
    type_resolver(crate::builtins::io::hdf5::hdf5_type),
    descriptor(crate::builtins::io::hdf5::H5WRITEATT_DESCRIPTOR),
    builtin_path = "crate::builtins::io::hdf5"
)]
async fn h5writeatt_builtin(
    filename: Value,
    location: Value,
    attribute: Value,
    data: Value,
) -> BuiltinResult<Value> {
    let filename = gather_if_needed_async(&filename)
        .await
        .map_err(|err| map_control_flow(H5WRITEATT_NAME, err))?;
    let location = gather_if_needed_async(&location)
        .await
        .map_err(|err| map_control_flow(H5WRITEATT_NAME, err))?;
    let attribute = gather_if_needed_async(&attribute)
        .await
        .map_err(|err| map_control_flow(H5WRITEATT_NAME, err))?;
    let data = gather_if_needed_async(&data)
        .await
        .map_err(|err| map_control_flow(H5WRITEATT_NAME, err))?;
    let path = resolve_path(&filename, H5WRITEATT_NAME)?;
    let location_path = hdf5_path(&location, H5WRITEATT_NAME, "location")?;
    let attr_name = string_scalar(&attribute, H5WRITEATT_NAME, "attribute")?;
    if attr_name.trim().is_empty() {
        return Err(hdf5_error(
            H5WRITEATT_NAME,
            &ERR_ARGUMENT,
            "h5writeatt: attribute must not be empty",
        ));
    }
    imp::write_attribute(H5WRITEATT_NAME, path, location_path, attr_name, data)?;
    Ok(Value::OutputList(Vec::new()))
}

#[runtime_builtin(
    name = "h5info",
    category = "io/hdf5",
    summary = "Return metadata for an HDF5 file object.",
    keywords = "h5info,hdf5,metadata,dataset,group,file,io",
    accel = "cpu",
    type_resolver(crate::builtins::io::hdf5::hdf5_type),
    descriptor(crate::builtins::io::hdf5::H5INFO_DESCRIPTOR),
    builtin_path = "crate::builtins::io::hdf5"
)]
async fn h5info_builtin(filename: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let filename = gather_if_needed_async(&filename)
        .await
        .map_err(|err| map_control_flow(H5INFO_NAME, err))?;
    let rest = gather_args(H5INFO_NAME, &rest).await?;
    let path = resolve_path(&filename, H5INFO_NAME)?;
    let location = optional_location(rest, H5INFO_NAME)?;
    imp::info(H5INFO_NAME, path, location)
}

#[runtime_builtin(
    name = "hdf5info",
    category = "io/hdf5",
    summary = "Return metadata for an HDF5 file object using the legacy HDF5 API name.",
    keywords = "hdf5info,h5info,hdf5,metadata,dataset,group,file,io,legacy",
    accel = "cpu",
    type_resolver(crate::builtins::io::hdf5::hdf5_type),
    descriptor(crate::builtins::io::hdf5::HDF5INFO_DESCRIPTOR),
    builtin_path = "crate::builtins::io::hdf5"
)]
async fn hdf5info_builtin(filename: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let filename = gather_if_needed_async(&filename)
        .await
        .map_err(|err| map_control_flow(HDF5INFO_NAME, err))?;
    let rest = gather_args(HDF5INFO_NAME, &rest).await?;
    let path = resolve_path(&filename, HDF5INFO_NAME)?;
    let location = optional_location(rest, HDF5INFO_NAME)?;
    imp::info(HDF5INFO_NAME, path, location)
}

#[runtime_builtin(
    name = "h5disp",
    category = "io/hdf5",
    summary = "Return a textual summary of an HDF5 file object.",
    keywords = "h5disp,hdf5,metadata,display,dataset,group,file,io",
    accel = "cpu",
    type_resolver(crate::builtins::io::hdf5::hdf5_type),
    descriptor(crate::builtins::io::hdf5::H5DISP_DESCRIPTOR),
    builtin_path = "crate::builtins::io::hdf5"
)]
async fn h5disp_builtin(filename: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let filename = gather_if_needed_async(&filename)
        .await
        .map_err(|err| map_control_flow(H5DISP_NAME, err))?;
    let rest = gather_args(H5DISP_NAME, &rest).await?;
    let path = resolve_path(&filename, H5DISP_NAME)?;
    let location = optional_location(rest, H5DISP_NAME)?;
    imp::disp(H5DISP_NAME, path, location).map(Value::String)
}

fn optional_location(rest: Vec<Value>, builtin: &'static str) -> BuiltinResult<String> {
    match rest.len() {
        0 => Ok("/".to_string()),
        1 => hdf5_path(&rest[0], builtin, "location"),
        _ => Err(hdf5_error(
            builtin,
            &ERR_ARGUMENT,
            format!("{builtin}: expected at most one location argument"),
        )),
    }
}

fn shape_or_scalar(shape: Vec<usize>) -> Vec<usize> {
    if shape.is_empty() {
        vec![1, 1]
    } else {
        shape
    }
}

fn matlab_value_from_f64(data: Vec<f64>, shape: Vec<usize>) -> BuiltinResult<Value> {
    let shape = shape_or_scalar(shape);
    if data.len() == 1 {
        Ok(Value::Num(data[0]))
    } else {
        Tensor::new(data, shape)
            .map(Value::Tensor)
            .map_err(|msg| hdf5_error(H5READ_NAME, &ERR_IO, msg))
    }
}

fn matlab_value_from_integer(storage: IntegerStorage, shape: Vec<usize>) -> BuiltinResult<Value> {
    let shape = shape_or_scalar(shape);
    if storage.len() == 1 {
        return Ok(Value::Int(integer_storage_scalar(&storage, 0)));
    }
    Tensor::new_integer(storage, shape)
        .map(Value::Tensor)
        .map_err(|msg| hdf5_error(H5READ_NAME, &ERR_IO, msg))
}

fn integer_storage_from_scalar(value: IntValue) -> IntegerStorage {
    match value {
        IntValue::I8(value) => IntegerStorage::I8(vec![value]),
        IntValue::I16(value) => IntegerStorage::I16(vec![value]),
        IntValue::I32(value) => IntegerStorage::I32(vec![value]),
        IntValue::I64(value) => IntegerStorage::I64(vec![value]),
        IntValue::U8(value) => IntegerStorage::U8(vec![value]),
        IntValue::U16(value) => IntegerStorage::U16(vec![value]),
        IntValue::U32(value) => IntegerStorage::U32(vec![value]),
        IntValue::U64(value) => IntegerStorage::U64(vec![value]),
    }
}

fn integer_storage_scalar(storage: &IntegerStorage, index: usize) -> IntValue {
    match storage {
        IntegerStorage::I8(values) => IntValue::I8(values[index]),
        IntegerStorage::I16(values) => IntValue::I16(values[index]),
        IntegerStorage::I32(values) => IntValue::I32(values[index]),
        IntegerStorage::I64(values) => IntValue::I64(values[index]),
        IntegerStorage::U8(values) => IntValue::U8(values[index]),
        IntegerStorage::U16(values) => IntValue::U16(values[index]),
        IntegerStorage::U32(values) => IntValue::U32(values[index]),
        IntegerStorage::U64(values) => IntValue::U64(values[index]),
    }
}

fn matlab_value_from_bool(data: Vec<bool>, shape: Vec<usize>) -> BuiltinResult<Value> {
    let bits: Vec<u8> = data.into_iter().map(u8::from).collect();
    let shape = shape_or_scalar(shape);
    if bits.len() == 1 {
        Ok(Value::Bool(bits[0] != 0))
    } else {
        LogicalArray::new(bits, shape)
            .map(Value::LogicalArray)
            .map_err(|msg| hdf5_error(H5READ_NAME, &ERR_IO, msg))
    }
}

fn matlab_value_from_strings(data: Vec<String>, shape: Vec<usize>) -> BuiltinResult<Value> {
    let shape = shape_or_scalar(shape);
    if data.len() == 1 {
        Ok(Value::String(data[0].clone()))
    } else {
        StringArray::new(data, shape)
            .map(Value::StringArray)
            .map_err(|msg| hdf5_error(H5READ_NAME, &ERR_IO, msg))
    }
}

fn numeric_vector_usize(
    value: &Value,
    builtin: &'static str,
    label: &str,
) -> BuiltinResult<Vec<usize>> {
    match value {
        Value::Num(v) => Ok(vec![numeric_usize_entry(*v, builtin, label)?]),
        Value::Int(v) => Ok(vec![int_usize_entry(v, builtin, label)?]),
        Value::Tensor(t) => {
            if let Some(storage) = t.integer_storage() {
                storage
                    .exact_values()
                    .iter()
                    .map(|value| int_usize_entry(value, builtin, label))
                    .collect()
            } else {
                (0..t.shape.iter().product())
                    .map(|index| {
                        let value = match t
                            .numeric_value_at(index)
                            .expect("numeric vector index is in bounds")
                        {
                            NumericScalar::F64(value) => value,
                            NumericScalar::F32(value) => f64::from(value),
                            NumericScalar::I8(_)
                            | NumericScalar::I16(_)
                            | NumericScalar::I32(_)
                            | NumericScalar::I64(_)
                            | NumericScalar::U8(_)
                            | NumericScalar::U16(_)
                            | NumericScalar::U32(_)
                            | NumericScalar::U64(_) => {
                                unreachable!("integer vectors use the exact parser")
                            }
                        };
                        numeric_usize_entry(value, builtin, label)
                    })
                    .collect()
            }
        }
        _ => Err(hdf5_error(
            builtin,
            &ERR_ARGUMENT,
            format!("{builtin}: expected {label} as a numeric vector"),
        )),
    }
}

fn int_usize_entry(value: &IntValue, builtin: &'static str, label: &str) -> BuiltinResult<usize> {
    let Some(parsed) = value.try_to_usize() else {
        return Err(hdf5_error(
            builtin,
            &ERR_ARGUMENT,
            format!("{builtin}: {label} entries must be positive finite integers"),
        ));
    };
    if parsed == 0 {
        return Err(hdf5_error(
            builtin,
            &ERR_ARGUMENT,
            format!("{builtin}: {label} entries must be positive finite integers"),
        ));
    }
    Ok(parsed)
}

fn numeric_usize_entry(value: f64, builtin: &'static str, label: &str) -> BuiltinResult<usize> {
    if !value.is_finite()
        || value < 1.0
        || value > usize::MAX as f64
        || (usize::BITS == 64 && value == usize::MAX as f64)
        || value.fract() != 0.0
    {
        Err(hdf5_error(
            builtin,
            &ERR_ARGUMENT,
            format!("{builtin}: {label} entries must be positive finite integers"),
        ))
    } else {
        Ok(value as usize)
    }
}

fn col_major_to_row_major<T: Clone>(data: &[T], shape: &[usize]) -> Vec<T> {
    let len = data.len();
    if shape.len() <= 1 || len <= 1 {
        return data.to_vec();
    }
    let mut out = Vec::with_capacity(len);
    for row_linear in 0..len {
        let idx = row_major_index_to_multi(row_linear, shape);
        let col_offset = multi_to_col_major(&idx, shape);
        out.push(data[col_offset].clone());
    }
    out
}

fn row_major_to_col_major<T: Clone>(data: &[T], shape: &[usize]) -> Vec<T> {
    let len = data.len();
    if shape.len() <= 1 || len <= 1 {
        return data.to_vec();
    }
    let mut out = Vec::with_capacity(len);
    for col_linear in 0..len {
        let idx = col_major_index_to_multi(col_linear, shape);
        let row_offset = multi_to_row_major(&idx, shape);
        out.push(data[row_offset].clone());
    }
    out
}

fn col_major_index_to_multi(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    let mut idx = Vec::with_capacity(shape.len());
    for &dim in shape {
        idx.push(linear % dim);
        linear /= dim;
    }
    idx
}

fn row_major_index_to_multi(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    let mut idx = vec![0; shape.len()];
    for dim in (0..shape.len()).rev() {
        idx[dim] = linear % shape[dim];
        linear /= shape[dim];
    }
    idx
}

fn multi_to_col_major(idx: &[usize], shape: &[usize]) -> usize {
    let mut stride = 1usize;
    let mut offset = 0usize;
    for (&i, &dim) in idx.iter().zip(shape) {
        offset += i * stride;
        stride *= dim;
    }
    offset
}

fn multi_to_row_major(idx: &[usize], shape: &[usize]) -> usize {
    let mut offset = 0usize;
    for (&i, &dim) in idx.iter().zip(shape) {
        offset = offset * dim + i;
    }
    offset
}

#[derive(Debug, Clone)]
struct MatlabSelection {
    start: Vec<usize>,
    count: Vec<usize>,
    stride: Vec<usize>,
}

fn parse_selection(
    rest: &[Value],
    builtin: &'static str,
    source_shape: &[usize],
) -> BuiltinResult<Option<MatlabSelection>> {
    match rest.len() {
        0 => Ok(None),
        2 | 3 => {
            let rank = source_shape.len();
            let mut start = numeric_vector_usize(&rest[0], builtin, "start")?;
            let count = numeric_vector_usize(&rest[1], builtin, "count")?;
            let stride = if rest.len() == 3 {
                numeric_vector_usize(&rest[2], builtin, "stride")?
            } else {
                vec![1; rank]
            };
            if start.len() != rank || count.len() != rank || stride.len() != rank {
                return Err(hdf5_error(
                    builtin,
                    &ERR_ARGUMENT,
                    format!("{builtin}: start, count, and stride must match dataset rank"),
                ));
            }
            for item in &mut start {
                *item -= 1;
            }
            for dim in 0..rank {
                if count[dim] == 0 {
                    return Err(hdf5_error(
                        builtin,
                        &ERR_ARGUMENT,
                        format!("{builtin}: count entries must be positive"),
                    ));
                }
                let last = count[dim]
                    .checked_sub(1)
                    .and_then(|count| count.checked_mul(stride[dim]))
                    .and_then(|offset| start[dim].checked_add(offset))
                    .ok_or_else(|| {
                        hdf5_error(
                            builtin,
                            &ERR_ARGUMENT,
                            format!("{builtin}: selection exceeds platform limits"),
                        )
                    })?;
                if last >= source_shape[dim] {
                    return Err(hdf5_error(
                        builtin,
                        &ERR_ARGUMENT,
                        format!("{builtin}: selection exceeds dataset bounds"),
                    ));
                }
            }
            checked_shape_len(&count, builtin)?;
            Ok(Some(MatlabSelection {
                start,
                count,
                stride,
            }))
        }
        _ => Err(hdf5_error(
            builtin,
            &ERR_ARGUMENT,
            format!("{builtin}: expected start/count or start/count/stride selection"),
        )),
    }
}

fn checked_shape_len(shape: &[usize], builtin: &'static str) -> BuiltinResult<usize> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or_else(|| {
            hdf5_error(
                builtin,
                &ERR_ARGUMENT,
                format!("{builtin}: shape exceeds platform limits"),
            )
        })
    })
}

#[cfg(target_arch = "wasm32")]
mod imp {
    use super::*;

    pub fn read_dataset(
        builtin: &'static str,
        _path: PathBuf,
        _dataset_path: String,
        _rest: Vec<Value>,
    ) -> BuiltinResult<Value> {
        Err(hdf5_error(builtin, &ERR_WASM, ERR_WASM.message))
    }

    pub fn write_dataset(
        builtin: &'static str,
        _path: PathBuf,
        _dataset_path: String,
        _data: Value,
        _rest: Vec<Value>,
    ) -> BuiltinResult<()> {
        Err(hdf5_error(builtin, &ERR_WASM, ERR_WASM.message))
    }

    pub fn write_attribute(
        builtin: &'static str,
        _path: PathBuf,
        _location: String,
        _attribute: String,
        _data: Value,
    ) -> BuiltinResult<()> {
        Err(hdf5_error(builtin, &ERR_WASM, ERR_WASM.message))
    }

    pub fn info(builtin: &'static str, _path: PathBuf, _location: String) -> BuiltinResult<Value> {
        Err(hdf5_error(builtin, &ERR_WASM, ERR_WASM.message))
    }

    pub fn disp(builtin: &'static str, _path: PathBuf, _location: String) -> BuiltinResult<String> {
        Err(hdf5_error(builtin, &ERR_WASM, ERR_WASM.message))
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod imp {
    use std::str::FromStr;

    use hdf5::types::{FloatSize, IntSize, TypeDescriptor, VarLenAscii, VarLenUnicode};
    use hdf5::{Attribute, Dataset, File, Group, Hyperslab, Location, SliceOrIndex};
    use runmat_builtins::{CharArray, StructValue};

    use super::*;

    impl MatlabSelection {
        fn to_hdf5_hyperslab(&self) -> Hyperslab {
            Hyperslab::new(
                self.start
                    .iter()
                    .zip(&self.count)
                    .zip(&self.stride)
                    .map(|((&start, &count), &step)| SliceOrIndex::SliceCount {
                        start,
                        step,
                        count,
                        block: 1,
                    })
                    .collect::<Vec<_>>(),
            )
        }
    }

    pub fn read_dataset(
        builtin: &'static str,
        path: PathBuf,
        dataset_path: String,
        rest: Vec<Value>,
    ) -> BuiltinResult<Value> {
        let file = open_file_read(builtin, &path)?;
        let ds = file.dataset(&dataset_path).map_err(|err| {
            io_error(
                builtin,
                format!("{builtin}: unable to open dataset {dataset_path}"),
                err,
            )
        })?;
        let shape = ds.shape();
        let selection = parse_selection(&rest, builtin, &shape)?;
        read_dataset_value(builtin, &ds, selection.as_ref())
    }

    pub fn write_dataset(
        builtin: &'static str,
        path: PathBuf,
        dataset_path: String,
        data: Value,
        rest: Vec<Value>,
    ) -> BuiltinResult<()> {
        let file = File::append(&path).map_err(|err| {
            io_error(
                builtin,
                format!("{builtin}: unable to open {}", path.display()),
                err,
            )
        })?;
        if !rest.is_empty() {
            return write_dataset_selection(builtin, &file, &dataset_path, data, rest);
        }
        let data = WritableData::from_value(builtin, data)?;
        if file.link_exists(&dataset_path) {
            let ds = file.dataset(&dataset_path).map_err(|err| {
                io_error(
                    builtin,
                    format!("{builtin}: unable to open existing dataset {dataset_path}"),
                    err,
                )
            })?;
            return write_existing_dataset(builtin, &ds, &dataset_path, data);
        }
        match data {
            WritableData::Numeric { data, shape } => {
                let row_major = col_major_to_row_major(&data, &shape);
                let ds = file
                    .new_dataset::<f64>()
                    .shape(shape.as_slice())
                    .create_intermediate_group(true)
                    .create(dataset_path.as_str())
                    .map_err(|err| {
                        io_error(
                            builtin,
                            format!("{builtin}: unable to create dataset {dataset_path}"),
                            err,
                        )
                    })?;
                ds.write_raw(&row_major).map_err(|err| {
                    io_error(
                        builtin,
                        format!("{builtin}: unable to write dataset {dataset_path}"),
                        err,
                    )
                })?;
            }
            WritableData::Single { data, shape } => {
                create_typed_dataset::<f32>(builtin, &file, &dataset_path, data, &shape)?;
            }
            WritableData::Integer { storage, shape } => {
                create_integer_dataset(builtin, &file, &dataset_path, storage, &shape)?;
            }
            WritableData::Logical { data, shape } => {
                let row_major = col_major_to_row_major(&data, &shape);
                let ds = file
                    .new_dataset::<bool>()
                    .shape(shape.as_slice())
                    .create_intermediate_group(true)
                    .create(dataset_path.as_str())
                    .map_err(|err| {
                        io_error(
                            builtin,
                            format!("{builtin}: unable to create dataset {dataset_path}"),
                            err,
                        )
                    })?;
                ds.write_raw(&row_major).map_err(|err| {
                    io_error(
                        builtin,
                        format!("{builtin}: unable to write dataset {dataset_path}"),
                        err,
                    )
                })?;
            }
            WritableData::Strings { data, shape } => {
                let row_major = col_major_to_row_major(&data, &shape);
                let encoded = encode_strings(builtin, &row_major)?;
                let ds = file
                    .new_dataset::<VarLenUnicode>()
                    .shape(shape.as_slice())
                    .create_intermediate_group(true)
                    .create(dataset_path.as_str())
                    .map_err(|err| {
                        io_error(
                            builtin,
                            format!("{builtin}: unable to create dataset {dataset_path}"),
                            err,
                        )
                    })?;
                ds.write_raw(&encoded).map_err(|err| {
                    io_error(
                        builtin,
                        format!("{builtin}: unable to write dataset {dataset_path}"),
                        err,
                    )
                })?;
            }
        }
        Ok(())
    }

    fn write_existing_dataset(
        builtin: &'static str,
        ds: &Dataset,
        dataset_path: &str,
        data: WritableData,
    ) -> BuiltinResult<()> {
        match data {
            WritableData::Numeric { data, shape } => {
                validate_full_write_shape(builtin, &shape, &ds.shape())?;
                let row_major = col_major_to_row_major(&data, &shape);
                ds.write_raw(&row_major).map_err(|err| {
                    io_error(
                        builtin,
                        format!("{builtin}: unable to write existing dataset {dataset_path}"),
                        err,
                    )
                })
            }
            WritableData::Single { data, shape } => {
                validate_full_write_shape(builtin, &shape, &ds.shape())?;
                let row_major = col_major_to_row_major(&data, &shape);
                ds.write_raw(&row_major).map_err(|err| {
                    io_error(
                        builtin,
                        format!("{builtin}: unable to write existing dataset {dataset_path}"),
                        err,
                    )
                })
            }
            WritableData::Integer { storage, shape } => {
                validate_full_write_shape(builtin, &shape, &ds.shape())?;
                write_existing_integer_dataset(builtin, ds, dataset_path, storage, &shape)
            }
            WritableData::Logical { data, shape } => {
                validate_full_write_shape(builtin, &shape, &ds.shape())?;
                let row_major = col_major_to_row_major(&data, &shape);
                ds.write_raw(&row_major).map_err(|err| {
                    io_error(
                        builtin,
                        format!("{builtin}: unable to write existing dataset {dataset_path}"),
                        err,
                    )
                })
            }
            WritableData::Strings { data, shape } => {
                validate_full_write_shape(builtin, &shape, &ds.shape())?;
                let row_major = col_major_to_row_major(&data, &shape);
                let encoded = encode_strings(builtin, &row_major)?;
                ds.write_raw(&encoded).map_err(|err| {
                    io_error(
                        builtin,
                        format!("{builtin}: unable to write existing dataset {dataset_path}"),
                        err,
                    )
                })
            }
        }
    }

    fn write_dataset_selection(
        builtin: &'static str,
        file: &File,
        dataset_path: &str,
        data: Value,
        rest: Vec<Value>,
    ) -> BuiltinResult<()> {
        let ds = file.dataset(dataset_path).map_err(|err| {
            io_error(
                builtin,
                format!("{builtin}: unable to open dataset {dataset_path}"),
                err,
            )
        })?;
        let shape = ds.shape();
        let selection = parse_selection(&rest, builtin, &shape)?.ok_or_else(|| {
            hdf5_error(
                builtin,
                &ERR_ARGUMENT,
                format!("{builtin}: expected start/count or start/count/stride selection"),
            )
        })?;
        match WritableData::from_value(builtin, data)? {
            WritableData::Numeric { data, shape } => {
                validate_write_selection_shape(builtin, &shape, &selection.count)?;
                let row_major = col_major_to_row_major(&data, &shape);
                write_hyperslab::<f64>(builtin, &ds, &selection, row_major)
            }
            WritableData::Single { data, shape } => {
                validate_write_selection_shape(builtin, &shape, &selection.count)?;
                let row_major = col_major_to_row_major(&data, &shape);
                write_hyperslab::<f32>(builtin, &ds, &selection, row_major)
            }
            WritableData::Integer { storage, shape } => {
                validate_write_selection_shape(builtin, &shape, &selection.count)?;
                write_integer_hyperslab(builtin, &ds, &selection, storage, &shape)
            }
            WritableData::Logical { data, shape } => {
                validate_write_selection_shape(builtin, &shape, &selection.count)?;
                let row_major = col_major_to_row_major(&data, &shape);
                write_hyperslab::<bool>(builtin, &ds, &selection, row_major)
            }
            WritableData::Strings { data, shape } => {
                validate_write_selection_shape(builtin, &shape, &selection.count)?;
                let row_major = col_major_to_row_major(&data, &shape);
                let encoded = encode_strings(builtin, &row_major)?;
                write_hyperslab::<VarLenUnicode>(builtin, &ds, &selection, encoded)
            }
        }
    }

    pub fn write_attribute(
        builtin: &'static str,
        path: PathBuf,
        location: String,
        attribute: String,
        data: Value,
    ) -> BuiltinResult<()> {
        let file = File::append(&path).map_err(|err| {
            io_error(
                builtin,
                format!("{builtin}: unable to open {}", path.display()),
                err,
            )
        })?;
        if location == "/" {
            write_attr_to_location(builtin, &file, &attribute, data)
        } else if let Ok(ds) = file.dataset(&location) {
            write_attr_to_location(builtin, &ds, &attribute, data)
        } else if let Ok(group) = file.group(&location) {
            write_attr_to_location(builtin, &group, &attribute, data)
        } else {
            Err(hdf5_error(
                builtin,
                &ERR_IO,
                format!("{builtin}: location {location} does not exist"),
            ))
        }
    }

    pub fn info(builtin: &'static str, path: PathBuf, location: String) -> BuiltinResult<Value> {
        let file = open_file_read(builtin, &path)?;
        let info = if location == "/" {
            let group = file.group("/").map_err(|err| {
                io_error(
                    builtin,
                    format!("{builtin}: unable to open root group"),
                    err,
                )
            })?;
            group_info(builtin, &group, "/", Some(path.as_path()))?
        } else if let Ok(ds) = file.dataset(&location) {
            dataset_info(builtin, &ds, &location)?
        } else {
            let group = file.group(&location).map_err(|err| {
                io_error(
                    builtin,
                    format!("{builtin}: location {location} does not exist"),
                    err,
                )
            })?;
            group_info(builtin, &group, &location, Some(path.as_path()))?
        };
        Ok(Value::Struct(info))
    }

    pub fn disp(builtin: &'static str, path: PathBuf, location: String) -> BuiltinResult<String> {
        let file = open_file_read(builtin, &path)?;
        let mut out = String::new();
        out.push_str(&format!("HDF5 {location}\n"));
        if location == "/" {
            let group = file.group("/").map_err(|err| {
                io_error(
                    builtin,
                    format!("{builtin}: unable to open root group"),
                    err,
                )
            })?;
            render_group(builtin, &group, "/", 0, &mut out)?;
        } else if let Ok(ds) = file.dataset(&location) {
            render_dataset(builtin, &ds, &location, 0, &mut out)?;
        } else {
            let group = file.group(&location).map_err(|err| {
                io_error(
                    builtin,
                    format!("{builtin}: location {location} does not exist"),
                    err,
                )
            })?;
            render_group(builtin, &group, &location, 0, &mut out)?;
        }
        Ok(out)
    }

    enum WritableData {
        Numeric {
            data: Vec<f64>,
            shape: Vec<usize>,
        },
        Single {
            data: Vec<f32>,
            shape: Vec<usize>,
        },
        Integer {
            storage: IntegerStorage,
            shape: Vec<usize>,
        },
        Logical {
            data: Vec<bool>,
            shape: Vec<usize>,
        },
        Strings {
            data: Vec<String>,
            shape: Vec<usize>,
        },
    }

    impl WritableData {
        fn from_value(builtin: &'static str, value: Value) -> BuiltinResult<Self> {
            match value {
                Value::Num(v) => Ok(Self::Numeric {
                    data: vec![v],
                    shape: vec![1, 1],
                }),
                Value::Int(v) => Ok(Self::Integer {
                    storage: integer_storage_from_scalar(v),
                    shape: vec![1, 1],
                }),
                Value::Tensor(t) => {
                    let shape = t.shape.clone();
                    let storage = t.into_numeric_storage().map_err(|error| {
                        hdf5_error(builtin, &ERR_UNSUPPORTED, format!("{builtin}: {error}"))
                    })?;
                    match storage {
                        NumericStorage::F64(data) => Ok(Self::Numeric { data, shape }),
                        NumericStorage::F32(data) => Ok(Self::Single { data, shape }),
                        storage => Ok(Self::Integer {
                            storage: storage
                                .into_integer_storage()
                                .expect("non-floating numeric storage is integer"),
                            shape,
                        }),
                    }
                }
                Value::Bool(v) => Ok(Self::Logical {
                    data: vec![v],
                    shape: vec![1, 1],
                }),
                Value::LogicalArray(array) => Ok(Self::Logical {
                    data: array.data.into_iter().map(|v| v != 0).collect(),
                    shape: array.shape,
                }),
                Value::String(text) => Ok(Self::Strings {
                    data: vec![text],
                    shape: vec![1, 1],
                }),
                Value::StringArray(array) => Ok(Self::Strings {
                    data: array.data,
                    shape: array.shape,
                }),
                Value::CharArray(chars) => Ok(Self::Strings {
                    data: char_rows(&chars),
                    shape: vec![chars.rows.max(1), 1],
                }),
                _ => Err(hdf5_error(
                    builtin,
                    &ERR_UNSUPPORTED,
                    format!("{builtin}: only dense numeric, logical, string, and char data can be written to HDF5"),
                )),
            }
        }
    }

    fn read_dataset_value(
        builtin: &'static str,
        ds: &Dataset,
        selection: Option<&MatlabSelection>,
    ) -> BuiltinResult<Value> {
        let shape = ds.shape();
        let dtype = ds
            .dtype()
            .and_then(|dt| dt.to_descriptor())
            .map_err(|err| {
                io_error(
                    builtin,
                    format!("{builtin}: unable to inspect dataset type"),
                    err,
                )
            })?;
        match dtype {
            TypeDescriptor::Float(FloatSize::U4) => {
                let (raw, out_shape) = read_raw_or_slice::<f32>(builtin, ds, &shape, selection)?;
                let data = row_major_to_col_major(&raw, &out_shape);
                Tensor::from_f32(data, out_shape)
                    .map(Value::Tensor)
                    .map_err(|error| hdf5_error(builtin, &ERR_IO, format!("{builtin}: {error}")))
            }
            TypeDescriptor::Float(FloatSize::U8) => {
                let (raw, out_shape) = read_raw_or_slice::<f64>(builtin, ds, &shape, selection)?;
                matlab_value_from_f64(row_major_to_col_major(&raw, &out_shape), out_shape)
            }
            TypeDescriptor::Integer(IntSize::U1) => {
                read_integer_dataset::<i8, _>(builtin, ds, &shape, selection, IntegerStorage::I8)
            }
            TypeDescriptor::Integer(IntSize::U2) => {
                read_integer_dataset::<i16, _>(builtin, ds, &shape, selection, IntegerStorage::I16)
            }
            TypeDescriptor::Integer(IntSize::U4) => {
                read_integer_dataset::<i32, _>(builtin, ds, &shape, selection, IntegerStorage::I32)
            }
            TypeDescriptor::Integer(IntSize::U8) => {
                read_integer_dataset::<i64, _>(builtin, ds, &shape, selection, IntegerStorage::I64)
            }
            TypeDescriptor::Unsigned(IntSize::U1) => {
                read_integer_dataset::<u8, _>(builtin, ds, &shape, selection, IntegerStorage::U8)
            }
            TypeDescriptor::Unsigned(IntSize::U2) => {
                read_integer_dataset::<u16, _>(builtin, ds, &shape, selection, IntegerStorage::U16)
            }
            TypeDescriptor::Unsigned(IntSize::U4) => {
                read_integer_dataset::<u32, _>(builtin, ds, &shape, selection, IntegerStorage::U32)
            }
            TypeDescriptor::Unsigned(IntSize::U8) => {
                read_integer_dataset::<u64, _>(builtin, ds, &shape, selection, IntegerStorage::U64)
            }
            TypeDescriptor::Boolean => {
                let (raw, out_shape) = read_raw_or_slice::<bool>(builtin, ds, &shape, selection)?;
                matlab_value_from_bool(row_major_to_col_major(&raw, &out_shape), out_shape)
            }
            TypeDescriptor::VarLenUnicode => {
                let (raw, out_shape) =
                    read_raw_or_slice::<VarLenUnicode>(builtin, ds, &shape, selection)?;
                let strings: Vec<String> =
                    raw.into_iter().map(|s| s.as_str().to_string()).collect();
                matlab_value_from_strings(row_major_to_col_major(&strings, &out_shape), out_shape)
            }
            TypeDescriptor::VarLenAscii => {
                let (raw, out_shape) =
                    read_raw_or_slice::<VarLenAscii>(builtin, ds, &shape, selection)?;
                let strings: Vec<String> =
                    raw.into_iter().map(|s| s.as_str().to_string()).collect();
                matlab_value_from_strings(row_major_to_col_major(&strings, &out_shape), out_shape)
            }
            other => Err(hdf5_error(
                builtin,
                &ERR_UNSUPPORTED,
                format!(
                    "{builtin}: unsupported HDF5 dataset type {}",
                    type_name(&other)
                ),
            )),
        }
    }

    fn read_integer_dataset<T, F>(
        builtin: &'static str,
        ds: &Dataset,
        shape: &[usize],
        selection: Option<&MatlabSelection>,
        convert: F,
    ) -> BuiltinResult<Value>
    where
        T: hdf5::H5Type + Copy,
        F: Fn(Vec<T>) -> IntegerStorage,
    {
        let (raw, out_shape) = read_raw_or_slice::<T>(builtin, ds, shape, selection)?;
        matlab_value_from_integer(convert(row_major_to_col_major(&raw, &out_shape)), out_shape)
    }

    fn read_raw_or_slice<T>(
        builtin: &'static str,
        ds: &Dataset,
        shape: &[usize],
        selection: Option<&MatlabSelection>,
    ) -> BuiltinResult<(Vec<T>, Vec<usize>)>
    where
        T: hdf5::H5Type,
    {
        if let Some(selection) = selection {
            let hyperslab = selection.to_hdf5_hyperslab();
            let array = ds
                .as_reader()
                .read_slice::<T, _, ndarray::IxDyn>(hdf5::Selection::Hyperslab(hyperslab))
                .map_err(|err| {
                    io_error(
                        builtin,
                        format!("{builtin}: unable to read HDF5 hyperslab"),
                        err,
                    )
                })?;
            Ok((array.into_raw_vec(), selection.count.clone()))
        } else {
            let raw = ds.read_raw::<T>().map_err(|err| {
                io_error(
                    builtin,
                    format!("{builtin}: unable to read HDF5 dataset"),
                    err,
                )
            })?;
            Ok((raw, shape.to_vec()))
        }
    }

    fn create_integer_dataset(
        builtin: &'static str,
        file: &File,
        dataset_path: &str,
        storage: IntegerStorage,
        shape: &[usize],
    ) -> BuiltinResult<()> {
        macro_rules! create {
            ($values:expr, $ty:ty) => {
                create_typed_dataset::<$ty>(builtin, file, dataset_path, $values, shape)
            };
        }
        match storage {
            IntegerStorage::I8(values) => create!(values, i8),
            IntegerStorage::I16(values) => create!(values, i16),
            IntegerStorage::I32(values) => create!(values, i32),
            IntegerStorage::I64(values) => create!(values, i64),
            IntegerStorage::U8(values) => create!(values, u8),
            IntegerStorage::U16(values) => create!(values, u16),
            IntegerStorage::U32(values) => create!(values, u32),
            IntegerStorage::U64(values) => create!(values, u64),
        }
    }

    fn create_typed_dataset<T>(
        builtin: &'static str,
        file: &File,
        dataset_path: &str,
        data: Vec<T>,
        shape: &[usize],
    ) -> BuiltinResult<()>
    where
        T: hdf5::H5Type + Clone,
    {
        let row_major = col_major_to_row_major(&data, shape);
        let ds = file
            .new_dataset::<T>()
            .shape(shape)
            .create_intermediate_group(true)
            .create(dataset_path)
            .map_err(|err| {
                io_error(
                    builtin,
                    format!("{builtin}: unable to create dataset {dataset_path}"),
                    err,
                )
            })?;
        ds.write_raw(&row_major).map_err(|err| {
            io_error(
                builtin,
                format!("{builtin}: unable to write dataset {dataset_path}"),
                err,
            )
        })
    }

    fn write_existing_integer_dataset(
        builtin: &'static str,
        ds: &Dataset,
        dataset_path: &str,
        storage: IntegerStorage,
        shape: &[usize],
    ) -> BuiltinResult<()> {
        macro_rules! write {
            ($values:expr) => {{
                let row_major = col_major_to_row_major(&$values, shape);
                ds.write_raw(&row_major).map_err(|err| {
                    io_error(
                        builtin,
                        format!("{builtin}: unable to write existing dataset {dataset_path}"),
                        err,
                    )
                })
            }};
        }
        match storage {
            IntegerStorage::I8(values) => write!(values),
            IntegerStorage::I16(values) => write!(values),
            IntegerStorage::I32(values) => write!(values),
            IntegerStorage::I64(values) => write!(values),
            IntegerStorage::U8(values) => write!(values),
            IntegerStorage::U16(values) => write!(values),
            IntegerStorage::U32(values) => write!(values),
            IntegerStorage::U64(values) => write!(values),
        }
    }

    fn write_integer_hyperslab(
        builtin: &'static str,
        ds: &Dataset,
        selection: &MatlabSelection,
        storage: IntegerStorage,
        shape: &[usize],
    ) -> BuiltinResult<()> {
        macro_rules! write {
            ($values:expr, $ty:ty) => {{
                let row_major = col_major_to_row_major(&$values, shape);
                write_hyperslab::<$ty>(builtin, ds, selection, row_major)
            }};
        }
        match storage {
            IntegerStorage::I8(values) => write!(values, i8),
            IntegerStorage::I16(values) => write!(values, i16),
            IntegerStorage::I32(values) => write!(values, i32),
            IntegerStorage::I64(values) => write!(values, i64),
            IntegerStorage::U8(values) => write!(values, u8),
            IntegerStorage::U16(values) => write!(values, u16),
            IntegerStorage::U32(values) => write!(values, u32),
            IntegerStorage::U64(values) => write!(values, u64),
        }
    }

    fn validate_write_selection_shape(
        builtin: &'static str,
        data_shape: &[usize],
        count: &[usize],
    ) -> BuiltinResult<()> {
        if data_shape == count {
            return Ok(());
        }
        if checked_shape_len(data_shape, builtin)? == 1 && checked_shape_len(count, builtin)? == 1 {
            return Ok(());
        }
        Err(hdf5_error(
            builtin,
            &ERR_ARGUMENT,
            format!("{builtin}: data shape must match the selected HDF5 count"),
        ))
    }

    fn validate_full_write_shape(
        builtin: &'static str,
        data_shape: &[usize],
        dataset_shape: &[usize],
    ) -> BuiltinResult<()> {
        if data_shape == dataset_shape {
            return Ok(());
        }
        Err(hdf5_error(
            builtin,
            &ERR_ARGUMENT,
            format!(
                "{builtin}: data shape {data_shape:?} must match existing HDF5 dataset shape {dataset_shape:?}"
            ),
        ))
    }

    fn write_hyperslab<T>(
        builtin: &'static str,
        ds: &Dataset,
        selection: &MatlabSelection,
        data: Vec<T>,
    ) -> BuiltinResult<()>
    where
        T: hdf5::H5Type,
    {
        let array = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&selection.count), data)
            .map_err(|err| {
                hdf5_error(
                    builtin,
                    &ERR_ARGUMENT,
                    format!("{builtin}: unable to shape selected write data: {err}"),
                )
            })?;
        ds.as_writer()
            .write_slice(
                array.view(),
                hdf5::Selection::Hyperslab(selection.to_hdf5_hyperslab()),
            )
            .map_err(|err| {
                io_error(
                    builtin,
                    format!("{builtin}: unable to write HDF5 hyperslab"),
                    err,
                )
            })
    }

    fn write_attr_to_location(
        builtin: &'static str,
        location: &Location,
        name: &str,
        data: Value,
    ) -> BuiltinResult<()> {
        if location
            .attr_names()
            .map(|names| names.iter().any(|item| item == name))
            .unwrap_or(false)
        {
            location.delete_attr(name).map_err(|err| {
                io_error(
                    builtin,
                    format!("{builtin}: unable to replace attribute {name}"),
                    err,
                )
            })?;
        }
        match WritableData::from_value(builtin, data)? {
            WritableData::Numeric { data, shape } => {
                let row_major = col_major_to_row_major(&data, &shape);
                let attr = location
                    .new_attr::<f64>()
                    .shape(shape.as_slice())
                    .create(name)
                    .map_err(|err| {
                        io_error(
                            builtin,
                            format!("{builtin}: unable to create attribute {name}"),
                            err,
                        )
                    })?;
                attr.write_raw(&row_major).map_err(|err| {
                    io_error(
                        builtin,
                        format!("{builtin}: unable to write attribute {name}"),
                        err,
                    )
                })?;
            }
            WritableData::Single { data, shape } => {
                let row_major = col_major_to_row_major(&data, &shape);
                let attr = location
                    .new_attr::<f32>()
                    .shape(shape.as_slice())
                    .create(name)
                    .map_err(|err| {
                        io_error(
                            builtin,
                            format!("{builtin}: unable to create attribute {name}"),
                            err,
                        )
                    })?;
                attr.write_raw(&row_major).map_err(|err| {
                    io_error(
                        builtin,
                        format!("{builtin}: unable to write attribute {name}"),
                        err,
                    )
                })?;
            }
            WritableData::Integer { storage, shape } => {
                write_integer_attribute(builtin, location, name, storage, &shape)?;
            }
            WritableData::Logical { data, shape } => {
                let row_major = col_major_to_row_major(&data, &shape);
                let attr = location
                    .new_attr::<bool>()
                    .shape(shape.as_slice())
                    .create(name)
                    .map_err(|err| {
                        io_error(
                            builtin,
                            format!("{builtin}: unable to create attribute {name}"),
                            err,
                        )
                    })?;
                attr.write_raw(&row_major).map_err(|err| {
                    io_error(
                        builtin,
                        format!("{builtin}: unable to write attribute {name}"),
                        err,
                    )
                })?;
            }
            WritableData::Strings { data, shape } => {
                let row_major = col_major_to_row_major(&data, &shape);
                let encoded = encode_strings(builtin, &row_major)?;
                let attr = location
                    .new_attr::<VarLenUnicode>()
                    .shape(shape.as_slice())
                    .create(name)
                    .map_err(|err| {
                        io_error(
                            builtin,
                            format!("{builtin}: unable to create attribute {name}"),
                            err,
                        )
                    })?;
                attr.write_raw(&encoded).map_err(|err| {
                    io_error(
                        builtin,
                        format!("{builtin}: unable to write attribute {name}"),
                        err,
                    )
                })?;
            }
        }
        Ok(())
    }

    fn write_integer_attribute(
        builtin: &'static str,
        location: &Location,
        name: &str,
        storage: IntegerStorage,
        shape: &[usize],
    ) -> BuiltinResult<()> {
        macro_rules! write {
            ($values:expr, $ty:ty) => {{
                let row_major = col_major_to_row_major(&$values, shape);
                let attr = location
                    .new_attr::<$ty>()
                    .shape(shape)
                    .create(name)
                    .map_err(|err| {
                        io_error(
                            builtin,
                            format!("{builtin}: unable to create attribute {name}"),
                            err,
                        )
                    })?;
                attr.write_raw(&row_major).map_err(|err| {
                    io_error(
                        builtin,
                        format!("{builtin}: unable to write attribute {name}"),
                        err,
                    )
                })
            }};
        }
        match storage {
            IntegerStorage::I8(values) => write!(values, i8),
            IntegerStorage::I16(values) => write!(values, i16),
            IntegerStorage::I32(values) => write!(values, i32),
            IntegerStorage::I64(values) => write!(values, i64),
            IntegerStorage::U8(values) => write!(values, u8),
            IntegerStorage::U16(values) => write!(values, u16),
            IntegerStorage::U32(values) => write!(values, u32),
            IntegerStorage::U64(values) => write!(values, u64),
        }
    }

    fn open_file_read(builtin: &'static str, path: &Path) -> BuiltinResult<File> {
        File::open(path).map_err(|err| {
            io_error(
                builtin,
                format!("{builtin}: unable to open {}", path.display()),
                err,
            )
        })
    }

    fn group_info(
        builtin: &'static str,
        group: &Group,
        name: &str,
        filename: Option<&Path>,
    ) -> BuiltinResult<StructValue> {
        let mut info = StructValue::new();
        if let Some(path) = filename {
            info.insert("Filename", Value::String(path.display().to_string()));
        }
        info.insert("Name", Value::String(name.to_string()));
        info.insert("Attributes", attributes_info(builtin, group)?);

        let mut groups = Vec::new();
        let mut datasets = Vec::new();
        for member in group.member_names().map_err(|err| {
            io_error(
                builtin,
                format!("{builtin}: unable to list group {name}"),
                err,
            )
        })? {
            let child_path = join_hdf5_path(name, &member);
            if let Ok(child) = group.group(&member) {
                groups.push(Value::Struct(group_info(
                    builtin,
                    &child,
                    &child_path,
                    None,
                )?));
            } else if let Ok(ds) = group.dataset(&member) {
                datasets.push(Value::Struct(dataset_info(builtin, &ds, &child_path)?));
            }
        }
        info.insert("Groups", cell_column(groups, builtin)?);
        info.insert("Datasets", cell_column(datasets, builtin)?);
        info.insert("Datatypes", cell_column(Vec::new(), builtin)?);
        info.insert("Links", cell_column(Vec::new(), builtin)?);
        Ok(info)
    }

    fn dataset_info(builtin: &'static str, ds: &Dataset, name: &str) -> BuiltinResult<StructValue> {
        let shape = ds.shape();
        let dtype = ds
            .dtype()
            .and_then(|dt| dt.to_descriptor())
            .map_err(|err| {
                io_error(
                    builtin,
                    format!("{builtin}: unable to inspect dataset type"),
                    err,
                )
            })?;
        let mut info = StructValue::new();
        info.insert("Name", Value::String(name.to_string()));
        info.insert("Datatype", Value::String(type_name(&dtype)));
        info.insert("Class", Value::String(matlab_class_for_type(&dtype)));
        info.insert(
            "Dataspace",
            Value::String(if shape.is_empty() { "scalar" } else { "simple" }.to_string()),
        );
        info.insert("Size", shape_value(shape)?);
        info.insert("Attributes", attributes_info(builtin, ds)?);
        Ok(info)
    }

    fn attributes_info(builtin: &'static str, location: &Location) -> BuiltinResult<Value> {
        let mut values = Vec::new();
        for name in location.attr_names().map_err(|err| {
            io_error(
                builtin,
                format!("{builtin}: unable to list attributes"),
                err,
            )
        })? {
            let attr = location.attr(&name).map_err(|err| {
                io_error(
                    builtin,
                    format!("{builtin}: unable to open attribute {name}"),
                    err,
                )
            })?;
            values.push(Value::Struct(attribute_info(builtin, &attr, &name)?));
        }
        cell_column(values, builtin)
    }

    fn attribute_info(
        builtin: &'static str,
        attr: &Attribute,
        name: &str,
    ) -> BuiltinResult<StructValue> {
        let dtype = attr
            .dtype()
            .and_then(|dt| dt.to_descriptor())
            .map_err(|err| {
                io_error(
                    builtin,
                    format!("{builtin}: unable to inspect attribute type"),
                    err,
                )
            })?;
        let mut info = StructValue::new();
        info.insert("Name", Value::String(name.to_string()));
        info.insert("Datatype", Value::String(type_name(&dtype)));
        info.insert("Class", Value::String(matlab_class_for_type(&dtype)));
        info.insert("Size", shape_value(attr.shape())?);
        Ok(info)
    }

    fn render_group(
        builtin: &'static str,
        group: &Group,
        name: &str,
        indent: usize,
        out: &mut String,
    ) -> BuiltinResult<()> {
        out.push_str(&format!("{}Group: {name}\n", " ".repeat(indent)));
        render_attributes(builtin, group, indent + 2, out)?;
        for member in group.member_names().map_err(|err| {
            io_error(
                builtin,
                format!("{builtin}: unable to list group {name}"),
                err,
            )
        })? {
            let child_path = join_hdf5_path(name, &member);
            if let Ok(child) = group.group(&member) {
                render_group(builtin, &child, &child_path, indent + 2, out)?;
            } else if let Ok(ds) = group.dataset(&member) {
                render_dataset(builtin, &ds, &child_path, indent + 2, out)?;
            }
        }
        Ok(())
    }

    fn render_dataset(
        builtin: &'static str,
        ds: &Dataset,
        name: &str,
        indent: usize,
        out: &mut String,
    ) -> BuiltinResult<()> {
        let dtype = ds
            .dtype()
            .and_then(|dt| dt.to_descriptor())
            .map_err(|err| {
                io_error(
                    builtin,
                    format!("{builtin}: unable to inspect dataset type"),
                    err,
                )
            })?;
        out.push_str(&format!(
            "{}Dataset: {name} [{}] {}\n",
            " ".repeat(indent),
            ds.shape()
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join("x"),
            type_name(&dtype)
        ));
        render_attributes(builtin, ds, indent + 2, out)
    }

    fn render_attributes(
        builtin: &'static str,
        location: &Location,
        indent: usize,
        out: &mut String,
    ) -> BuiltinResult<()> {
        for name in location.attr_names().map_err(|err| {
            io_error(
                builtin,
                format!("{builtin}: unable to list attributes"),
                err,
            )
        })? {
            let attr = location.attr(&name).map_err(|err| {
                io_error(
                    builtin,
                    format!("{builtin}: unable to open attribute {name}"),
                    err,
                )
            })?;
            let dtype = attr
                .dtype()
                .and_then(|dt| dt.to_descriptor())
                .map_err(|err| {
                    io_error(
                        builtin,
                        format!("{builtin}: unable to inspect attribute type"),
                        err,
                    )
                })?;
            out.push_str(&format!(
                "{}Attribute: {name} [{}] {}\n",
                " ".repeat(indent),
                attr.shape()
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join("x"),
                type_name(&dtype)
            ));
        }
        Ok(())
    }

    fn char_rows(chars: &CharArray) -> Vec<String> {
        if chars.rows == 0 || chars.cols == 0 {
            return vec![String::new()];
        }
        (0..chars.rows)
            .map(|row| {
                (0..chars.cols)
                    .map(|col| chars.data[row * chars.cols + col])
                    .collect()
            })
            .collect()
    }

    fn encode_strings(
        builtin: &'static str,
        values: &[String],
    ) -> BuiltinResult<Vec<VarLenUnicode>> {
        values
            .iter()
            .map(|value| {
                VarLenUnicode::from_str(value).map_err(|err| {
                    hdf5_error(
                        builtin,
                        &ERR_ARGUMENT,
                        format!("{builtin}: string value cannot be encoded as HDF5 UTF-8: {err}"),
                    )
                })
            })
            .collect()
    }

    fn shape_value(shape: Vec<usize>) -> BuiltinResult<Value> {
        if shape.is_empty() {
            Ok(Value::Num(1.0))
        } else {
            let rank = shape.len();
            Tensor::new(shape.into_iter().map(|v| v as f64).collect(), vec![1, rank])
                .map(Value::Tensor)
                .map_err(|msg| hdf5_error(H5INFO_NAME, &ERR_IO, msg))
        }
    }

    fn cell_column(values: Vec<Value>, builtin: &'static str) -> BuiltinResult<Value> {
        let rows = values.len();
        crate::make_cell(values, rows, 1).map_err(|msg| hdf5_error(builtin, &ERR_IO, msg))
    }

    fn join_hdf5_path(parent: &str, child: &str) -> String {
        if parent == "/" {
            format!("/{child}")
        } else {
            format!("{parent}/{child}")
        }
    }

    pub(super) fn type_name(dtype: &TypeDescriptor) -> String {
        match dtype {
            TypeDescriptor::Float(FloatSize::U4) => "single".to_string(),
            TypeDescriptor::Float(FloatSize::U8) => "double".to_string(),
            TypeDescriptor::Integer(IntSize::U1) => "int8".to_string(),
            TypeDescriptor::Integer(IntSize::U2) => "int16".to_string(),
            TypeDescriptor::Integer(IntSize::U4) => "int32".to_string(),
            TypeDescriptor::Integer(IntSize::U8) => "int64".to_string(),
            TypeDescriptor::Unsigned(IntSize::U1) => "uint8".to_string(),
            TypeDescriptor::Unsigned(IntSize::U2) => "uint16".to_string(),
            TypeDescriptor::Unsigned(IntSize::U4) => "uint32".to_string(),
            TypeDescriptor::Unsigned(IntSize::U8) => "uint64".to_string(),
            TypeDescriptor::Boolean => "logical".to_string(),
            TypeDescriptor::VarLenUnicode | TypeDescriptor::VarLenAscii => "string".to_string(),
            TypeDescriptor::FixedUnicode(_) | TypeDescriptor::FixedAscii(_) => "char".to_string(),
            TypeDescriptor::Compound(_) => "compound".to_string(),
            TypeDescriptor::Enum(_) => "enum".to_string(),
            TypeDescriptor::FixedArray(_, _) => "fixed-array".to_string(),
            TypeDescriptor::VarLenArray(_) => "variable-length-array".to_string(),
            TypeDescriptor::Reference(_) => "reference".to_string(),
        }
    }

    fn matlab_class_for_type(dtype: &TypeDescriptor) -> String {
        match dtype {
            TypeDescriptor::Integer(IntSize::U1) => "int8",
            TypeDescriptor::Integer(IntSize::U2) => "int16",
            TypeDescriptor::Integer(IntSize::U4) => "int32",
            TypeDescriptor::Integer(IntSize::U8) => "int64",
            TypeDescriptor::Unsigned(IntSize::U1) => "uint8",
            TypeDescriptor::Unsigned(IntSize::U2) => "uint16",
            TypeDescriptor::Unsigned(IntSize::U4) => "uint32",
            TypeDescriptor::Unsigned(IntSize::U8) => "uint64",
            TypeDescriptor::Boolean => "logical",
            TypeDescriptor::VarLenUnicode | TypeDescriptor::VarLenAscii => "string",
            TypeDescriptor::FixedUnicode(_) | TypeDescriptor::FixedAscii(_) => "char",
            _ => "double",
        }
        .to_string()
    }

    fn io_error<E>(builtin: &'static str, message: impl Into<String>, source: E) -> RuntimeError
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        let mut builder = build_runtime_error(message)
            .with_builtin(builtin)
            .with_source(source);
        if let Some(identifier) = ERR_IO.identifier {
            builder = builder.with_identifier(identifier);
        }
        builder.build()
    }
}

#[cfg(test)]
#[cfg(not(target_arch = "wasm32"))]
mod tests {
    use futures::executor::block_on;
    use runmat_builtins::{ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage};
    use tempfile::tempdir;

    use super::imp::type_name;
    use super::*;

    #[test]
    fn h5write_and_h5read_round_trip_column_major_numeric() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("roundtrip.h5");
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();

        block_on(h5write_builtin(
            Value::String(path.display().to_string()),
            Value::String("/group/data".to_string()),
            Value::Tensor(tensor),
            Vec::new(),
        ))
        .expect("write");

        let out = block_on(h5read_builtin(
            Value::String(path.display().to_string()),
            Value::String("/group/data".to_string()),
            Vec::new(),
        ))
        .expect("read");

        match out {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 3]);
                assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn hdf5_round_trips_native_single_storage() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("single_roundtrip.h5");
        let tensor = Tensor::from_f32(vec![1.25, 2.5, 3.75, 5.0], vec![2, 2]).unwrap();

        block_on(h5write_builtin(
            Value::String(path.display().to_string()),
            Value::String("/single".to_string()),
            Value::Tensor(tensor),
            Vec::new(),
        ))
        .expect("write single");

        let out = block_on(h5read_builtin(
            Value::String(path.display().to_string()),
            Value::String("/single".to_string()),
            Vec::new(),
        ))
        .expect("read single");
        let Value::Tensor(tensor) = out else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(
            tensor.into_numeric_storage().expect("single storage"),
            NumericStorage::F32(vec![1.25, 2.5, 3.75, 5.0])
        );
    }

    #[test]
    fn h5write_rejects_typed_complex_integer_without_float_coercion() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("typed_complex.h5");
        let storage = IntegerComplexStorage::new(
            IntegerStorage::U64(vec![u64::MAX]),
            IntegerStorage::U64(vec![1_u64 << 63]),
        )
        .expect("matching components");
        let tensor = ComplexTensor::new_integer(storage, vec![1, 1]).expect("typed complex");

        let err = block_on(h5write_builtin(
            Value::String(path.display().to_string()),
            Value::String("/data".to_string()),
            Value::ComplexTensor(tensor),
            Vec::new(),
        ))
        .expect_err("typed complex integer HDF5 writes are unsupported");

        assert_eq!(err.identifier(), Some("RunMat:hdf5:UnsupportedType"));
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn hdf5_selection_vectors_read_integer_storage_exactly() {
        let exact = (1_u64 << 53) + 1;

        assert_eq!(
            numeric_vector_usize(&Value::Int(IntValue::U64(exact)), H5READ_NAME, "start")
                .expect("scalar vector"),
            vec![exact as usize]
        );

        let mut vector =
            Tensor::new_integer(IntegerStorage::U64(vec![exact, 3]), vec![1, 2]).expect("vector");
        vector.data.fill(f64::NAN);
        assert_eq!(
            numeric_vector_usize(&Value::Tensor(vector), H5WRITE_NAME, "count").expect("vector"),
            vec![exact as usize, 3]
        );

        assert!(numeric_vector_usize(&Value::Int(IntValue::U8(0)), H5READ_NAME, "start").is_err());
        assert!(numeric_vector_usize(&Value::Int(IntValue::I8(-1)), H5READ_NAME, "start").is_err());
    }

    #[test]
    fn hdf5_selection_vectors_reject_unrepresentable_double_bounds() {
        let boundary = numeric_vector_usize(&Value::Num(usize::MAX as f64), H5READ_NAME, "start");
        if usize::BITS == 64 {
            assert!(boundary.is_err());
        } else {
            assert_eq!(boundary.unwrap(), vec![usize::MAX]);
        }
        assert!(
            numeric_vector_usize(&Value::Num((usize::MAX as f64) + 1.0), H5READ_NAME, "start")
                .is_err()
        );
    }

    #[test]
    fn hdf5_round_trips_all_integer_classes_without_f64_conversion() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("integer_classes.h5");
        let cases = [
            ("i8", IntegerStorage::I8(vec![i8::MIN, -1, 0, i8::MAX])),
            ("i16", IntegerStorage::I16(vec![i16::MIN, -1, 0, i16::MAX])),
            ("i32", IntegerStorage::I32(vec![i32::MIN, -1, 0, i32::MAX])),
            ("i64", IntegerStorage::I64(vec![i64::MIN, -1, 0, i64::MAX])),
            ("u8", IntegerStorage::U8(vec![0, 1, 127, u8::MAX])),
            ("u16", IntegerStorage::U16(vec![0, 1, 32_768, u16::MAX])),
            ("u32", IntegerStorage::U32(vec![0, 1, 1 << 31, u32::MAX])),
            (
                "u64",
                IntegerStorage::U64(vec![0, 1, 1_u64 << 63, u64::MAX]),
            ),
        ];

        for (name, storage) in cases {
            let tensor = Tensor::new_integer(storage.clone(), vec![2, 2]).expect("integer tensor");
            block_on(h5write_builtin(
                Value::String(path.display().to_string()),
                Value::String(format!("/{name}")),
                Value::Tensor(tensor),
                Vec::new(),
            ))
            .expect("write integer dataset");

            let out = block_on(h5read_builtin(
                Value::String(path.display().to_string()),
                Value::String(format!("/{name}")),
                Vec::new(),
            ))
            .expect("read integer dataset");
            assert!(
                matches!(out, Value::Tensor(tensor) if tensor.integer_storage() == Some(&storage))
            );

            let file = hdf5::File::open(&path).expect("open hdf5 file");
            let dtype = file
                .dataset(format!("/{name}").as_str())
                .expect("dataset")
                .dtype()
                .and_then(|dtype| dtype.to_descriptor())
                .expect("type descriptor");
            assert_eq!(type_name(&dtype), storage.class_name());
        }
    }

    #[test]
    fn hdf5_integer_scalar_selection_and_attribute_preserve_native_types() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("integer_paths.h5");

        block_on(h5write_builtin(
            Value::String(path.display().to_string()),
            Value::String("/scalar".to_string()),
            Value::Int(IntValue::U64(u64::MAX)),
            Vec::new(),
        ))
        .expect("write scalar");
        let scalar = block_on(h5read_builtin(
            Value::String(path.display().to_string()),
            Value::String("/scalar".to_string()),
            Vec::new(),
        ))
        .expect("read scalar");
        assert_eq!(scalar, Value::Int(IntValue::U64(u64::MAX)));

        let initial = Tensor::new_integer(IntegerStorage::I64(vec![1, 2, 3, 4]), vec![2, 2])
            .expect("initial tensor");
        block_on(h5write_builtin(
            Value::String(path.display().to_string()),
            Value::String("/selected".to_string()),
            Value::Tensor(initial),
            Vec::new(),
        ))
        .expect("write selected dataset");
        let patch = Tensor::new_integer(IntegerStorage::I64(vec![i64::MIN, i64::MAX]), vec![2, 1])
            .expect("patch tensor");
        block_on(h5write_builtin(
            Value::String(path.display().to_string()),
            Value::String("/selected".to_string()),
            Value::Tensor(patch),
            vec![
                Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap()),
                Value::Tensor(Tensor::new(vec![2.0, 1.0], vec![1, 2]).unwrap()),
            ],
        ))
        .expect("write integer selection");
        let selected = block_on(h5read_builtin(
            Value::String(path.display().to_string()),
            Value::String("/selected".to_string()),
            Vec::new(),
        ))
        .expect("read selected dataset");
        assert!(matches!(selected, Value::Tensor(tensor)
            if tensor.integer_storage() == Some(&IntegerStorage::I64(vec![i64::MIN, i64::MAX, 3, 4]))));

        block_on(h5writeatt_builtin(
            Value::String(path.display().to_string()),
            Value::String("/selected".to_string()),
            Value::String("limits".to_string()),
            Value::Int(IntValue::U32(u32::MAX)),
        ))
        .expect("write integer attribute");
        let file = hdf5::File::open(&path).expect("open hdf5 file");
        let dtype = file
            .dataset("/selected")
            .expect("dataset")
            .attr("limits")
            .expect("attribute")
            .dtype()
            .and_then(|dtype| dtype.to_descriptor())
            .expect("type descriptor");
        assert_eq!(type_name(&dtype), "uint32");
    }

    #[test]
    fn h5read_supports_start_count_stride_selection() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("slice.h5");
        let tensor = Tensor::new((1..=12).map(|v| v as f64).collect(), vec![3, 4]).unwrap();
        block_on(h5write_builtin(
            Value::String(path.display().to_string()),
            Value::String("/data".to_string()),
            Value::Tensor(tensor),
            Vec::new(),
        ))
        .expect("write");

        let out = block_on(h5read_builtin(
            Value::String(path.display().to_string()),
            Value::String("/data".to_string()),
            vec![
                Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
                Value::Tensor(Tensor::new(vec![2.0, 2.0], vec![1, 2]).unwrap()),
                Value::Tensor(Tensor::new(vec![2.0, 1.0], vec![1, 2]).unwrap()),
            ],
        ))
        .expect("slice");

        match out {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, vec![4.0, 6.0, 7.0, 9.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn h5write_supports_start_count_stride_selection() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("partial_write.h5");
        let base = Tensor::zeros(vec![3, 4]);
        block_on(h5write_builtin(
            Value::String(path.display().to_string()),
            Value::String("/data".to_string()),
            Value::Tensor(base),
            Vec::new(),
        ))
        .expect("write base");

        let patch = Tensor::new(vec![40.0, 60.0, 70.0, 90.0], vec![2, 2]).unwrap();
        block_on(h5write_builtin(
            Value::String(path.display().to_string()),
            Value::String("/data".to_string()),
            Value::Tensor(patch),
            vec![
                Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
                Value::Tensor(Tensor::new(vec![2.0, 2.0], vec![1, 2]).unwrap()),
                Value::Tensor(Tensor::new(vec![2.0, 1.0], vec![1, 2]).unwrap()),
            ],
        ))
        .expect("partial write");

        let out = block_on(h5read_builtin(
            Value::String(path.display().to_string()),
            Value::String("/data".to_string()),
            Vec::new(),
        ))
        .expect("read");

        match out {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 4]);
                assert_eq!(
                    t.data,
                    vec![0.0, 0.0, 0.0, 40.0, 0.0, 60.0, 70.0, 0.0, 90.0, 0.0, 0.0, 0.0]
                );
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn h5write_full_existing_dataset_preserves_attributes() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("rewrite_preserves_attrs.h5");
        block_on(h5write_builtin(
            Value::String(path.display().to_string()),
            Value::String("/data".to_string()),
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
            Vec::new(),
        ))
        .expect("initial write");
        block_on(h5writeatt_builtin(
            Value::String(path.display().to_string()),
            Value::String("/data".to_string()),
            Value::String("units".to_string()),
            Value::String("arb".to_string()),
        ))
        .expect("write attr");

        block_on(h5write_builtin(
            Value::String(path.display().to_string()),
            Value::String("/data".to_string()),
            Value::Tensor(Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap()),
            Vec::new(),
        ))
        .expect("rewrite existing");

        let out = block_on(h5read_builtin(
            Value::String(path.display().to_string()),
            Value::String("/data".to_string()),
            Vec::new(),
        ))
        .expect("read rewritten");
        assert!(matches!(out, Value::Tensor(t) if t.data == vec![3.0, 4.0]));

        let info = block_on(h5info_builtin(
            Value::String(path.display().to_string()),
            vec![Value::String("/data".to_string())],
        ))
        .expect("dataset info");
        let Value::Struct(info) = info else {
            panic!("expected dataset info struct");
        };
        let Value::Cell(attributes) = info.fields.get("Attributes").unwrap() else {
            panic!("expected attributes cell");
        };
        assert!(attributes.data.iter().any(|value| {
            matches!(value, Value::Struct(attr) if attr.fields.get("Name") == Some(&Value::String("units".to_string())))
        }));
    }

    #[test]
    fn h5writeatt_h5info_and_h5disp_report_metadata() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("info.h5");
        block_on(h5write_builtin(
            Value::String(path.display().to_string()),
            Value::String("/data".to_string()),
            Value::Bool(true),
            Vec::new(),
        ))
        .expect("write");
        block_on(h5writeatt_builtin(
            Value::String(path.display().to_string()),
            Value::String("/data".to_string()),
            Value::String("units".to_string()),
            Value::String("arb".to_string()),
        ))
        .expect("write attr");

        let info = block_on(h5info_builtin(
            Value::String(path.display().to_string()),
            Vec::new(),
        ))
        .expect("info");
        match info {
            Value::Struct(s) => {
                assert!(s.fields.contains_key("Groups"));
                assert!(s.fields.contains_key("Datasets"));
            }
            other => panic!("expected struct, got {other:?}"),
        }

        let text = block_on(h5disp_builtin(
            Value::String(path.display().to_string()),
            vec![Value::String("/".to_string())],
        ))
        .expect("disp");
        match text {
            Value::String(s) => {
                assert!(s.contains("Dataset: /data"));
                assert!(s.contains("Attribute: units"));
            }
            other => panic!("expected string, got {other:?}"),
        }
    }

    #[test]
    fn h5read_rejects_unsupported_compound_dataset() {
        #[derive(hdf5::H5Type, Clone, Copy)]
        #[repr(C)]
        #[allow(non_local_definitions)]
        struct Pair {
            a: f64,
            b: f64,
        }

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("compound.h5");
        let file = hdf5::File::create(&path).expect("file");
        file.new_dataset_builder()
            .with_data(&[Pair { a: 1.0, b: 2.0 }])
            .create("/compound")
            .expect("dataset");

        let err = block_on(h5read_builtin(
            Value::String(path.display().to_string()),
            Value::String("/compound".to_string()),
            Vec::new(),
        ))
        .expect_err("compound unsupported");
        assert_eq!(err.identifier(), Some("RunMat:hdf5:UnsupportedType"));
    }
}
