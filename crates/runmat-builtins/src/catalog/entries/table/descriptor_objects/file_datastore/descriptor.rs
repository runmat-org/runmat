use crate::*;

/// Supported constructor options, shared by Runtime validation and inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileDatastoreOption {
    ReadFcn,
    FileExtensions,
    IncludeSubfolders,
    ReadMode,
}

impl FileDatastoreOption {
    pub fn parse(text: &str) -> Option<Self> {
        [
            ("ReadFcn", Self::ReadFcn),
            ("FileExtensions", Self::FileExtensions),
            ("IncludeSubfolders", Self::IncludeSubfolders),
            ("ReadMode", Self::ReadMode),
        ]
        .into_iter()
        .find_map(|(name, option)| text.eq_ignore_ascii_case(name).then_some(option))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileDatastoreReadMode {
    File,
    PartialFile,
}

impl FileDatastoreReadMode {
    pub fn parse(text: &str) -> Option<Self> {
        if text.eq_ignore_ascii_case("file") {
            Some(Self::File)
        } else if text.eq_ignore_ascii_case("partialfile") {
            Some(Self::PartialFile)
        } else {
            None
        }
    }
}

pub const FILE_DATASTORE_METADATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "filedatastore-metadata-payloads",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fileDatastore with an omitted or non-callable ReadFcn, an opaque Files payload, or nested text cells is a RunMat metadata extension",
    error_identifier: Some("RunMat:compatibility:FileDatastoreMetadataExtension"),
};
pub const FILE_DATASTORE_BOOLEAN_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "filedatastore-boolean-coercion",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fileDatastore IncludeSubfolders with a native integer scalar, a finite double other than zero or one, or a text boolean alias is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FileDatastoreBooleanExtension"),
};
pub const FILE_DATASTORE_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "filedatastore-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fileDatastore with an interactive resident input is a RunMat extension that gathers to host metadata",
    error_identifier: Some("RunMat:compatibility:FileDatastoreGpuInputExtension"),
};
pub const FILE_DATASTORE_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    FILE_DATASTORE_METADATA_EXTENSION,
    FILE_DATASTORE_BOOLEAN_EXTENSION,
    FILE_DATASTORE_GPU_INPUT_EXTENSION,
];

const PAYLOADS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "Files",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Native integer Files payloads are opaque RunMat metadata, not file locations; storage is retained exactly.",
    },
    BuiltinIntegerInputCapability {
        name: "ReadFcn",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "A native integer ReadFcn payload is opaque RunMat metadata, never an executable reader.",
    },
];
const BOOLEAN: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "IncludeSubfolders",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Only native integer scalars are accepted, with exact zero/nonzero conversion; integer arrays are rejected. MATLAB-compatible controls are logical or double zero/one.",
}];
const TEXT_CONTROLS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "option names, FileExtensions, ReadMode",
    classes: &[],
    availability: BuiltinIntegerInputAvailability::Rejected,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Option names and ReadMode require scalar text; FileExtensions requires a text list. Native integer payloads in these roles reject.",
}];
pub const FILE_DATASTORE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "ds = fileDatastore(integer_files, 'ReadFcn', integer_payload)",
        inputs: &PAYLOADS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Gated opaque payloads retain exact class and values; no file access or callback execution occurs.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "ds = fileDatastore(files, 'IncludeSubfolders', integer_scalar, Name,Value...)",
        inputs: &BOOLEAN,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "IncludeSubfolders is stored as a logical scalar, without floating-point conversion of integer values.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "ds = fileDatastore(files, typed_integer_text_control)",
        inputs: &TEXT_CONTROLS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Integer values do not become text names, extension lists, or read modes in either compatibility mode.",
    },
];

pub const FILE_DATASTORE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "ds = fileDatastore(files, 'ReadFcn', reader, Name, Value...)",
        inputs: &[
            BuiltinParamDescriptor { name: "files", ty: BuiltinParamType::Any, arity: BuiltinParamArity::Required, default: None, description: "Text file specifications retained without enumeration or opening files. Opaque payloads require RunMat extension mode." },
            BuiltinParamDescriptor { name: "Name,Value", ty: BuiltinParamType::Any, arity: BuiltinParamArity::Variadic, default: None, description: "ReadFcn (required function handle in MATLAB mode), FileExtensions (default empty string array), IncludeSubfolders (default false), ReadMode ('file' or 'partialfile', default 'file'). Duplicate options use the last value." },
        ],
        outputs: &[BuiltinParamDescriptor { name: "ds", ty: BuiltinParamType::Any, arity: BuiltinParamArity::Required, default: None, description: "Scalar host fileDatastore value descriptor with Files, ReadFcn, FileExtensions, IncludeSubfolders, and ReadMode properties. Not a live file iterator." }],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[super::super::super::shared::TABLE_ERROR_INVALID_ARGUMENT],
};
