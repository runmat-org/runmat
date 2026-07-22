//! MATLAB-compatible `pcode` support for RunMat-generated P-code artefacts.
//!
//! MATLAB's proprietary P-code format is intentionally undocumented. RunMat
//! therefore writes a content-obscured RunMat P-code container that can be
//! executed by RunMat while rejecting non-RunMat `.p` files deterministically.

use std::collections::HashSet;
use std::fmt;
use std::io;
use std::path::{Component, Path, PathBuf};

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use glob::Pattern;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    Tensor, Value,
};
use runmat_filesystem as vfs;
use runmat_macros::runtime_builtin;
use sha2::{Digest, Sha256};

use crate::builtins::common::fs::expand_user_path;
use crate::builtins::common::path_search::{find_file_with_extensions, path_is_directory};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "pcode";
const RUNMAT_PCODE_MAGIC: &str = "RUNMAT-PCODE-V1";
const SOURCE_EXTENSIONS: &[&str] = &[".m"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcodeAlgorithm {
    R2007b,
    R2022a,
}

impl PcodeAlgorithm {
    fn label(self) -> &'static str {
        match self {
            PcodeAlgorithm::R2007b => "R2007b",
            PcodeAlgorithm::R2022a => "R2022a",
        }
    }

    fn from_label(label: &str) -> Option<Self> {
        match label {
            "R2007b" => Some(PcodeAlgorithm::R2007b),
            "R2022a" => Some(PcodeAlgorithm::R2022a),
            _ => None,
        }
    }

    fn key(self) -> &'static [u8] {
        match self {
            PcodeAlgorithm::R2007b => b"runmat-pcode-r2007b-v1",
            PcodeAlgorithm::R2022a => b"runmat-pcode-r2022a-v1",
        }
    }
}

const INPUT_ITEM: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "item",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "MATLAB source file, source name, folder, or wildcard pattern to encode.",
}];

const INPUT_ITEM_VERSION: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "item",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "MATLAB source file, source name, folder, or wildcard pattern to encode.",
    },
    BuiltinParamDescriptor {
        name: "version",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("-R2007b"),
        description: "Obfuscation compatibility selector, either -R2007b or -R2022a.",
    },
];

const INPUT_ITEMS_INPLACE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "item1",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First MATLAB source file, source name, folder, or wildcard pattern.",
    },
    BuiltinParamDescriptor {
        name: "itemN",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional source files, source names, folders, or wildcard patterns.",
    },
    BuiltinParamDescriptor {
        name: "inplace",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Use -inplace to write each P-code file beside its source file.",
    },
];

const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "pcode(item)",
        inputs: &INPUT_ITEM,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "pcode(item, version)",
        inputs: &INPUT_ITEM_VERSION,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "pcode(item1, item2, ..., itemN, \"-inplace\")",
        inputs: &INPUT_ITEMS_INPLACE,
        outputs: &[],
    },
];

const ERROR_NOT_ENOUGH_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PCODE.NOT_ENOUGH_INPUTS",
    identifier: Some("RunMat:pcode:NotEnoughInputs"),
    when: "No source item is provided.",
    message: "pcode: not enough input arguments",
};
const ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PCODE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:pcode:TooManyOutputs"),
    when: "`pcode` is called with requested output arguments.",
    message: "pcode: too many output arguments",
};
const ERROR_ARG_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PCODE.ARG_TYPE",
    identifier: Some("RunMat:pcode:InvalidArgument"),
    when: "An argument is not a character row, string scalar, or scalar string array.",
    message: "pcode: arguments must be character vectors or string scalars",
};
const ERROR_EMPTY_ITEM: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PCODE.EMPTY_ITEM",
    identifier: Some("RunMat:pcode:EmptyItem"),
    when: "A source item is empty.",
    message: "pcode: source item must not be empty",
};
const ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PCODE.INVALID_OPTION",
    identifier: Some("RunMat:pcode:InvalidOption"),
    when: "An option is not -inplace, -R2007b, or -R2022a.",
    message: "pcode: invalid option",
};
const ERROR_PATH_RESOLVE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PCODE.PATH_RESOLVE",
    identifier: Some("RunMat:pcode:PathResolveFailed"),
    when: "RunMat cannot resolve a source item or destination path.",
    message: "pcode: failed to resolve path",
};
const ERROR_FILE_NOT_FOUND: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PCODE.FILE_NOT_FOUND",
    identifier: Some("RunMat:pcode:FileNotFound"),
    when: "No MATLAB `.m` source file matches the item.",
    message: "pcode: source file not found",
};
const ERROR_UNSUPPORTED_EXTENSION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PCODE.UNSUPPORTED_EXTENSION",
    identifier: Some("RunMat:pcode:UnsupportedExtension"),
    when: "The source item names an unsupported file type, including `.mlx` live scripts.",
    message: "pcode: only MATLAB .m source files are supported",
};
const ERROR_FILE_READ: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PCODE.FILE_READ",
    identifier: Some("RunMat:pcode:FileReadFailed"),
    when: "A source file cannot be read as text.",
    message: "pcode: failed to read source file",
};
const ERROR_FILE_WRITE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PCODE.FILE_WRITE",
    identifier: Some("RunMat:pcode:FileWriteFailed"),
    when: "The destination `.p` file cannot be written.",
    message: "pcode: failed to write P-code file",
};
const ERROR_INVALID_PCODE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PCODE.INVALID_PCODE",
    identifier: Some("RunMat:pcode:InvalidPcode"),
    when: "A `.p` file is not a RunMat-generated P-code container or fails integrity checks.",
    message: "pcode: invalid or unsupported P-code file",
};

const ERRORS: [BuiltinErrorDescriptor; 11] = [
    ERROR_NOT_ENOUGH_INPUTS,
    ERROR_TOO_MANY_OUTPUTS,
    ERROR_ARG_TYPE,
    ERROR_EMPTY_ITEM,
    ERROR_INVALID_OPTION,
    ERROR_PATH_RESOLVE,
    ERROR_FILE_NOT_FOUND,
    ERROR_UNSUPPORTED_EXTENSION,
    ERROR_FILE_READ,
    ERROR_FILE_WRITE,
    ERROR_INVALID_PCODE,
];

pub const PCODE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::repl_fs::pcode")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "pcode",
    op_kind: GpuOpKind::Custom("io-pcode"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host-only source packaging builtin. GPU-resident string arguments are gathered before filesystem access; no numeric provider kernels are applicable.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::repl_fs::pcode")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "pcode",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "P-code generation performs host filesystem side effects and is a fusion barrier.",
};

#[derive(Debug, Clone)]
pub struct PcodeResult {
    pub generated: Vec<PathBuf>,
}

#[derive(Debug)]
pub enum PcodeSourceReadError {
    Io(io::Error),
    InvalidPcode(String),
}

impl fmt::Display for PcodeSourceReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PcodeSourceReadError::Io(err) => write!(f, "{err}"),
            PcodeSourceReadError::InvalidPcode(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for PcodeSourceReadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PcodeSourceReadError::Io(err) => Some(err),
            PcodeSourceReadError::InvalidPcode(_) => None,
        }
    }
}

impl From<io::Error> for PcodeSourceReadError {
    fn from(err: io::Error) -> Self {
        PcodeSourceReadError::Io(err)
    }
}

#[derive(Debug, Clone)]
struct PcodeOptions {
    algorithm: PcodeAlgorithm,
    inplace: bool,
    items: Vec<String>,
}

#[runtime_builtin(
    name = "pcode",
    category = "io/repl_fs",
    summary = "Generate RunMat-executable content-obscured P-code files from MATLAB source.",
    keywords = "pcode,p-code,source,obfuscate,file,folder,inplace",
    accel = "cpu",
    sink = true,
    suppress_auto_output = true,
    type_resolver(crate::builtins::io::type_resolvers::pcode_type),
    descriptor(crate::builtins::io::repl_fs::pcode::PCODE_DESCRIPTOR),
    builtin_path = "crate::builtins::io::repl_fs::pcode"
)]
async fn pcode_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    evaluate(&args).await?;
    Ok(empty_array())
}

pub async fn evaluate(args: &[Value]) -> BuiltinResult<PcodeResult> {
    if requested_output_count() > 0 {
        return Err(pcode_error(&ERROR_TOO_MANY_OUTPUTS));
    }
    let options = parse_options(args).await?;
    let sources = resolve_sources(&options.items).await?;
    if sources.is_empty() {
        return Err(pcode_error(&ERROR_FILE_NOT_FOUND));
    }

    let mut generated = Vec::with_capacity(sources.len());
    for source in sources {
        let text = vfs::read_to_string_async(&source).await.map_err(|err| {
            pcode_error_with_detail(&ERROR_FILE_READ, format!("{} ({err})", source.display()))
        })?;
        let destination = destination_for_source(&source, options.inplace)?;
        if let Some(parent) = destination.parent() {
            if !parent.as_os_str().is_empty() {
                vfs::create_dir_all_async(parent).await.map_err(|err| {
                    pcode_error_with_detail(
                        &ERROR_FILE_WRITE,
                        format!("{} ({err})", parent.display()),
                    )
                })?;
            }
        }
        let source_name = source.to_string_lossy();
        let encoded = encode_pcode_source(&text, &source_name, options.algorithm);
        vfs::write_async(&destination, encoded.as_bytes())
            .await
            .map_err(|err| {
                pcode_error_with_detail(
                    &ERROR_FILE_WRITE,
                    format!("{} ({err})", destination.display()),
                )
            })?;
        generated.push(destination);
    }
    Ok(PcodeResult { generated })
}

pub async fn prefer_pcode_source_path(path: &Path) -> PathBuf {
    if path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("m"))
    {
        let pcode_path = path.with_extension("p");
        if let Ok(metadata) = vfs::metadata_async(&pcode_path).await {
            if metadata.is_file() {
                return pcode_path;
            }
        }
    }
    path.to_path_buf()
}

pub async fn read_source_text_async(path: &Path) -> Result<String, PcodeSourceReadError> {
    if path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("p"))
    {
        let bytes = vfs::read_async(path).await?;
        decode_pcode_bytes(&bytes).map_err(PcodeSourceReadError::InvalidPcode)
    } else {
        vfs::read_to_string_async(path)
            .await
            .map_err(PcodeSourceReadError::Io)
    }
}

pub fn encode_pcode_source(source: &str, source_name: &str, algorithm: PcodeAlgorithm) -> String {
    let payload = transform_payload(source.as_bytes(), algorithm);
    let source_name = BASE64.encode(source_name.as_bytes());
    format!(
        "{RUNMAT_PCODE_MAGIC}\nalgorithm:{}\nsource:{}\nsha256:{}\npayload:{}\n",
        algorithm.label(),
        source_name,
        sha256_hex(source.as_bytes()),
        BASE64.encode(payload)
    )
}

pub fn decode_pcode_bytes(bytes: &[u8]) -> Result<String, String> {
    let text = std::str::from_utf8(bytes).map_err(|_| "P-code file is not UTF-8".to_string())?;
    let mut lines = text.lines();
    if lines.next() != Some(RUNMAT_PCODE_MAGIC) {
        return Err("not a RunMat-generated P-code file".to_string());
    }

    let mut algorithm = None;
    let mut expected_hash = None;
    let mut payload = None;
    for line in lines {
        if let Some(rest) = line.strip_prefix("algorithm:") {
            algorithm = PcodeAlgorithm::from_label(rest);
        } else if let Some(rest) = line.strip_prefix("sha256:") {
            expected_hash = Some(rest.to_string());
        } else if let Some(rest) = line.strip_prefix("payload:") {
            payload = Some(rest.to_string());
        }
    }

    let algorithm = algorithm.ok_or_else(|| "P-code algorithm is missing".to_string())?;
    let expected_hash = expected_hash.ok_or_else(|| "P-code hash is missing".to_string())?;
    let payload = payload.ok_or_else(|| "P-code payload is missing".to_string())?;
    let obfuscated = BASE64
        .decode(payload.as_bytes())
        .map_err(|err| format!("P-code payload is invalid base64: {err}"))?;
    let decoded = transform_payload(&obfuscated, algorithm);
    if sha256_hex(&decoded) != expected_hash {
        return Err("P-code payload hash mismatch".to_string());
    }
    String::from_utf8(decoded).map_err(|_| "P-code payload is not UTF-8 source".to_string())
}

async fn parse_options(args: &[Value]) -> BuiltinResult<PcodeOptions> {
    if args.is_empty() {
        return Err(pcode_error(&ERROR_NOT_ENOUGH_INPUTS));
    }
    let mut options = PcodeOptions {
        algorithm: PcodeAlgorithm::R2007b,
        inplace: false,
        items: Vec::new(),
    };

    for arg in args {
        let value = gather_if_needed_async(arg)
            .await
            .map_err(map_control_flow)?;
        let text = value_to_string_scalar(&value).ok_or_else(|| pcode_error(&ERROR_ARG_TYPE))?;
        if text.is_empty() {
            return Err(pcode_error(&ERROR_EMPTY_ITEM));
        }
        let lowered = text.to_ascii_lowercase();
        match lowered.as_str() {
            "-inplace" => options.inplace = true,
            "-r2007b" => options.algorithm = PcodeAlgorithm::R2007b,
            "-r2022a" => options.algorithm = PcodeAlgorithm::R2022a,
            option if option.starts_with('-') => {
                return Err(pcode_error_with_detail(&ERROR_INVALID_OPTION, text));
            }
            _ => options.items.push(text),
        }
    }

    if options.items.is_empty() {
        return Err(pcode_error(&ERROR_NOT_ENOUGH_INPUTS));
    }
    Ok(options)
}

async fn resolve_sources(items: &[String]) -> BuiltinResult<Vec<PathBuf>> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for item in items {
        for source in resolve_item(item).await? {
            let key = vfs::canonicalize_async(&source)
                .await
                .unwrap_or_else(|_| source.clone());
            if seen.insert(key) {
                out.push(source);
            }
        }
    }
    Ok(out)
}

async fn resolve_item(item: &str) -> BuiltinResult<Vec<PathBuf>> {
    if item_contains_wildcards(item) {
        return resolve_wildcard_item(item).await;
    }
    let expanded = expand_user_path(item, BUILTIN_NAME)
        .map(PathBuf::from)
        .map_err(|err| pcode_error_with_detail(&ERROR_PATH_RESOLVE, err))?;
    if path_is_directory(&expanded).await {
        return resolve_folder_item(&expanded).await;
    }
    if has_unsupported_extension(&expanded) {
        return Err(pcode_error_with_detail(&ERROR_UNSUPPORTED_EXTENSION, item));
    }
    let Some(path) = find_file_with_extensions(item, SOURCE_EXTENSIONS, BUILTIN_NAME)
        .await
        .map_err(|err| pcode_error_with_detail(&ERROR_PATH_RESOLVE, err))?
    else {
        return Err(pcode_error_with_detail(&ERROR_FILE_NOT_FOUND, item));
    };
    if !is_m_file(&path) {
        return Err(pcode_error_with_detail(
            &ERROR_UNSUPPORTED_EXTENSION,
            path.display().to_string(),
        ));
    }
    Ok(vec![path])
}

async fn resolve_folder_item(folder: &Path) -> BuiltinResult<Vec<PathBuf>> {
    let mut sources = Vec::new();
    let entries = vfs::read_dir_async(folder).await.map_err(|err| {
        pcode_error_with_detail(&ERROR_PATH_RESOLVE, format!("{} ({err})", folder.display()))
    })?;
    for entry in entries {
        if !entry.is_dir() && is_m_file(entry.path()) {
            sources.push(entry.path().to_path_buf());
        }
    }
    sources.sort();
    Ok(sources)
}

async fn resolve_wildcard_item(pattern_text: &str) -> BuiltinResult<Vec<PathBuf>> {
    let expanded = expand_user_path(pattern_text, BUILTIN_NAME)
        .map(PathBuf::from)
        .map_err(|err| pcode_error_with_detail(&ERROR_PATH_RESOLVE, err))?;
    let parent = expanded.parent().filter(|p| !p.as_os_str().is_empty());
    let dir = parent
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let file_pattern = expanded
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| pcode_error_with_detail(&ERROR_PATH_RESOLVE, pattern_text))?;
    let pattern = Pattern::new(file_pattern)
        .map_err(|err| pcode_error_with_detail(&ERROR_PATH_RESOLVE, err.to_string()))?;
    let entries = vfs::read_dir_async(&dir).await.map_err(|err| {
        pcode_error_with_detail(&ERROR_PATH_RESOLVE, format!("{} ({err})", dir.display()))
    })?;
    let mut sources = Vec::new();
    for entry in entries {
        let Some(name) = entry.file_name().to_str() else {
            continue;
        };
        if !entry.is_dir() && is_m_file(entry.path()) && pattern.matches(name) {
            sources.push(entry.path().to_path_buf());
        }
    }
    sources.sort();
    if sources.is_empty() {
        return Err(pcode_error_with_detail(&ERROR_FILE_NOT_FOUND, pattern_text));
    }
    Ok(sources)
}

fn destination_for_source(source: &Path, inplace: bool) -> BuiltinResult<PathBuf> {
    let mut filename = source
        .file_name()
        .ok_or_else(|| pcode_error_with_detail(&ERROR_PATH_RESOLVE, source.display().to_string()))?
        .to_os_string();
    filename = PathBuf::from(filename).with_extension("p").into_os_string();

    if inplace {
        return Ok(source.with_file_name(filename));
    }

    let mut destination = vfs::current_dir()
        .map_err(|err| pcode_error_with_detail(&ERROR_PATH_RESOLVE, err.to_string()))?;
    for component in package_or_class_suffix(source.parent()) {
        destination.push(component);
    }
    destination.push(filename);
    Ok(destination)
}

fn package_or_class_suffix(parent: Option<&Path>) -> Vec<String> {
    let Some(parent) = parent else {
        return Vec::new();
    };
    let mut current_run = Vec::new();
    for component in parent.components() {
        let Component::Normal(part) = component else {
            current_run.clear();
            continue;
        };
        let Some(part) = part.to_str() else {
            current_run.clear();
            continue;
        };
        if part.starts_with('+') || part.starts_with('@') {
            current_run.push(part.to_string());
        } else {
            current_run.clear();
        }
    }
    current_run
}

fn item_contains_wildcards(item: &str) -> bool {
    item.contains('*') || item.contains('?') || item.contains('[')
}

fn has_unsupported_extension(path: &Path) -> bool {
    let Some(extension) = path.extension().and_then(|ext| ext.to_str()) else {
        return false;
    };
    !extension.eq_ignore_ascii_case("m")
}

fn is_m_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("m"))
}

fn value_to_string_scalar(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::CharArray(array) if array.rows == 1 => Some(array.data.iter().collect()),
        Value::StringArray(array) if array.data.len() == 1 => Some(array.data[0].clone()),
        _ => None,
    }
}

fn transform_payload(bytes: &[u8], algorithm: PcodeAlgorithm) -> Vec<u8> {
    let key = algorithm.key();
    bytes
        .iter()
        .enumerate()
        .map(|(index, byte)| {
            let salt = match algorithm {
                PcodeAlgorithm::R2007b => index.wrapping_mul(31) as u8,
                PcodeAlgorithm::R2022a => index.wrapping_mul(73).rotate_left(1) as u8,
            };
            byte ^ key[index % key.len()] ^ salt
        })
        .collect()
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

fn empty_array() -> Value {
    Value::Tensor(Tensor::zeros(vec![0, 0]))
}

fn requested_output_count() -> usize {
    crate::output_count::current_output_count()
        .or_else(crate::output_context::requested_output_count)
        .unwrap_or(0)
}

fn pcode_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    pcode_error_with_detail(error, "")
}

pub fn invalid_pcode_runtime_error(detail: impl AsRef<str>) -> RuntimeError {
    pcode_error_with_detail(&ERROR_INVALID_PCODE, detail)
}

fn pcode_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.is_empty() {
        error.message.to_string()
    } else {
        format!("{}: {detail}", error.message)
    };
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    let identifier = err.identifier().map(str::to_string);
    let mut builder = build_runtime_error(format!("{BUILTIN_NAME}: {}", err.message()))
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK;
    use futures::executor::block_on;
    use std::env;

    struct CwdGuard {
        original: PathBuf,
    }

    impl Drop for CwdGuard {
        fn drop(&mut self) {
            let _ = env::set_current_dir(&self.original);
        }
    }

    fn push_cwd(path: &Path) -> CwdGuard {
        let original = env::current_dir().expect("current dir");
        env::set_current_dir(path).expect("set current dir");
        CwdGuard { original }
    }

    #[test]
    fn pcode_creates_obscured_executable_p_file_from_source_name() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let temp = tempfile::TempDir::new().expect("tempdir");
        std::fs::write(temp.path().join("worker.m"), "answer = 42;\n").expect("write source");
        let _cwd = push_cwd(temp.path());

        let result = block_on(evaluate(&[Value::from("worker")])).expect("pcode");
        assert_eq!(result.generated.len(), 1);
        assert!(result.generated[0].ends_with("worker.p"));
        let encoded = std::fs::read_to_string(temp.path().join("worker.p")).expect("read pcode");
        assert!(encoded.starts_with(RUNMAT_PCODE_MAGIC));
        assert!(!encoded.contains("answer = 42"));
        let decoded = decode_pcode_bytes(encoded.as_bytes()).expect("decode pcode");
        assert_eq!(decoded, "answer = 42;\n");
    }

    #[test]
    fn pcode_inplace_writes_next_to_source() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let temp = tempfile::TempDir::new().expect("tempdir");
        let src_dir = temp.path().join("src");
        std::fs::create_dir_all(&src_dir).expect("create src");
        let source = src_dir.join("worker.m");
        std::fs::write(&source, "value = 7;\n").expect("write source");
        let _cwd = push_cwd(temp.path());

        let result = block_on(evaluate(&[
            Value::from(source.to_string_lossy().to_string()),
            Value::from("-inplace"),
        ]))
        .expect("pcode");
        assert_eq!(result.generated.len(), 1);
        assert!(result.generated[0].ends_with("src/worker.p"));
        assert!(src_dir.join("worker.p").is_file());
        assert!(!temp.path().join("worker.p").exists());
    }

    #[test]
    fn pcode_folder_preserves_package_and_class_suffix_in_current_dir() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let temp = tempfile::TempDir::new().expect("tempdir");
        let source_root = temp.path().join("source").join("+pkg").join("@Widget");
        std::fs::create_dir_all(&source_root).expect("create source root");
        std::fs::write(source_root.join("Widget.m"), "classdef Widget\nend\n")
            .expect("write class source");
        let out = temp.path().join("out");
        std::fs::create_dir_all(&out).expect("create out");
        let _cwd = push_cwd(&out);

        let result = block_on(evaluate(&[Value::from(
            source_root.to_string_lossy().to_string(),
        )]))
        .expect("pcode");
        assert_eq!(result.generated.len(), 1);
        assert!(result.generated[0].ends_with("+pkg/@Widget/Widget.p"));
        assert!(out.join("+pkg/@Widget/Widget.p").is_file());
    }

    #[test]
    fn pcode_wildcard_ignores_non_m_files() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let temp = tempfile::TempDir::new().expect("tempdir");
        std::fs::write(temp.path().join("one.m"), "one = 1;\n").expect("write one");
        std::fs::write(temp.path().join("one.txt"), "not matlab\n").expect("write txt");
        let _cwd = push_cwd(temp.path());

        let result = block_on(evaluate(&[Value::from("one.*")])).expect("pcode");
        assert_eq!(result.generated.len(), 1);
        assert!(result.generated[0].ends_with("one.p"));
        assert!(temp.path().join("one.p").is_file());
        assert!(!temp.path().join("one.txt.p").exists());
    }

    #[test]
    fn pcode_rejects_live_scripts_and_unknown_options() {
        let err = block_on(evaluate(&[Value::from("demo.mlx")]))
            .expect_err("live script source is unsupported");
        assert_eq!(err.identifier(), Some("RunMat:pcode:UnsupportedExtension"));

        let err = block_on(evaluate(&[Value::from("demo"), Value::from("-bad")]))
            .expect_err("unknown option is invalid");
        assert_eq!(err.identifier(), Some("RunMat:pcode:InvalidOption"));
    }

    #[test]
    fn invalid_pcode_decode_reports_error() {
        let err = decode_pcode_bytes(b"not matlab pcode").expect_err("invalid pcode");
        assert!(err.contains("not a RunMat-generated"));
    }
}
