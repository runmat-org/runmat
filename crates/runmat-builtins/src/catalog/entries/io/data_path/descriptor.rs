//! Public positional forms and diagnostics for dataset path operations.

use crate::*;

const fn param(
    name: &'static str,
    ty: BuiltinParamType,
    description: &'static str,
) -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name,
        ty,
        arity: BuiltinParamArity::Required,
        default: None,
        description,
    }
}

const PATH: BuiltinParamDescriptor = param(
    "path",
    BuiltinParamType::StringScalar,
    "Dataset path (.data).",
);
const PREFIX: BuiltinParamDescriptor = param(
    "prefix",
    BuiltinParamType::StringScalar,
    "Directory prefix to scan for datasets.",
);
const SCHEMA: BuiltinParamDescriptor =
    param("schema", BuiltinParamType::Any, "Dataset schema structure.");
const FROM: BuiltinParamDescriptor = param(
    "fromPath",
    BuiltinParamType::StringScalar,
    "Source dataset path.",
);
const TO: BuiltinParamDescriptor = param(
    "toPath",
    BuiltinParamType::StringScalar,
    "Destination dataset path.",
);
const FORMAT: BuiltinParamDescriptor = param(
    "format",
    BuiltinParamType::StringScalar,
    "Format token; currently data.",
);
const SOURCE: BuiltinParamDescriptor = param(
    "sourcePath",
    BuiltinParamType::StringScalar,
    "Source dataset path.",
);
const TARGET: BuiltinParamDescriptor = param(
    "targetPath",
    BuiltinParamType::StringScalar,
    "Target dataset path.",
);

const DATASET: BuiltinParamDescriptor =
    param("ds", BuiltinParamType::Any, "Dataset handle object.");
const FLAG: BuiltinParamDescriptor =
    param("ok", BuiltinParamType::LogicalArray, "Logical result flag.");
const LIST: BuiltinParamDescriptor =
    param("C", BuiltinParamType::Any, "Cell array of dataset paths.");
const INSPECTION: BuiltinParamDescriptor = param("S", BuiltinParamType::Any, "Metadata structure.");

pub const DATA_ERROR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DATA.ERROR",
    identifier: Some("RUNMAT:Data:Error"),
    when: "A path, schema, format, manifest, or filesystem operation fails during the requested data call.",
    message: "data: operation failed",
};
const RESOURCE_EVIDENCE_ERROR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DATA.RESOURCE_EVIDENCE",
    identifier: Some("RunMat:Placement:InsufficientResourceEvidence"),
    when: "A finite resource budget is active but no complete dataset-path host-effect bound is certified before filesystem work.",
    message: "data: complete-call resource evidence is unavailable",
};
const RESOURCE_AUTHORITY_ERROR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DATA.RESOURCE_AUTHORITY",
    identifier: Some("RunMat:Placement:StaleResourceAuthority"),
    when: "The certifying host-effect resource authority is revoked before result publication.",
    message: "data: resource authority changed",
};
const RESOURCE_LIMIT_ERROR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DATA.RESOURCE_LIMIT",
    identifier: Some("RunMat:Placement:ResourceLimit"),
    when: "A certified complete-call demand exceeds the available placement resources.",
    message: "data: placement resource limit was exceeded",
};
const PLACEMENT_UNAVAILABLE_ERROR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DATA.PLACEMENT_UNAVAILABLE",
    identifier: Some("RunMat:Placement:Unavailable"),
    when: "The installed placement service cannot select an admitted host-effect route.",
    message: "data: placement route is unavailable",
};
const PLACEMENT_REVISION_ERROR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DATA.PLACEMENT_REVISION",
    identifier: Some("RunMat:Placement:InvalidRevision"),
    when: "The selected host-effect revision does not belong to the certifying authority.",
    message: "data: placement revision is invalid",
};
const PLACEMENT_DECISION_ERROR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DATA.PLACEMENT_DECISION",
    identifier: Some("RunMat:Placement:InvalidDecision"),
    when: "Placement returns a route other than the admitted dataset-path host effect.",
    message: "data: placement decision is invalid",
};
const CANCELLED_ERROR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DATA.CANCELLED",
    identifier: Some("RunMat:Cancelled"),
    when: "The invocation is cancelled before the dataset-path result is published.",
    message: "data invocation was cancelled",
};
const ERRORS: &[BuiltinErrorDescriptor] = &[
    DATA_ERROR,
    RESOURCE_EVIDENCE_ERROR,
    RESOURCE_AUTHORITY_ERROR,
    RESOURCE_LIMIT_ERROR,
    PLACEMENT_UNAVAILABLE_ERROR,
    PLACEMENT_REVISION_ERROR,
    PLACEMENT_DECISION_ERROR,
    CANCELLED_ERROR,
];

const fn signature(
    label: &'static str,
    inputs: &'static [BuiltinParamDescriptor],
    outputs: &'static [BuiltinParamDescriptor],
) -> BuiltinSignatureDescriptor {
    BuiltinSignatureDescriptor {
        label,
        inputs,
        outputs,
    }
}
const fn descriptor(signatures: &'static [BuiltinSignatureDescriptor]) -> BuiltinDescriptor {
    BuiltinDescriptor {
        signatures,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: ERRORS,
    }
}

pub const DATA_CREATE_DESCRIPTOR: BuiltinDescriptor = descriptor(&[signature(
    "ds = data.create(path, schema)",
    &[PATH, SCHEMA],
    &[DATASET],
)]);
pub const DATA_OPEN_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&[signature("ds = data.open(path)", &[PATH], &[DATASET])]);
pub const DATA_EXISTS_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&[signature("tf = data.exists(path)", &[PATH], &[FLAG])]);
pub const DATA_DELETE_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&[signature("tf = data.delete(path)", &[PATH], &[FLAG])]);
pub const DATA_COPY_DESCRIPTOR: BuiltinDescriptor = descriptor(&[signature(
    "tf = data.copy(fromPath, toPath)",
    &[FROM, TO],
    &[FLAG],
)]);
pub const DATA_MOVE_DESCRIPTOR: BuiltinDescriptor = descriptor(&[signature(
    "tf = data.move(fromPath, toPath)",
    &[FROM, TO],
    &[FLAG],
)]);
pub const DATA_IMPORT_DESCRIPTOR: BuiltinDescriptor = descriptor(&[signature(
    "ds = data.import(path, format, sourcePath)",
    &[PATH, FORMAT, SOURCE],
    &[DATASET],
)]);
pub const DATA_EXPORT_DESCRIPTOR: BuiltinDescriptor = descriptor(&[signature(
    "tf = data.export(path, format, targetPath)",
    &[PATH, FORMAT, TARGET],
    &[FLAG],
)]);
pub const DATA_LIST_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&[signature("C = data.list(prefix)", &[PREFIX], &[LIST])]);
pub const DATA_INSPECT_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&[signature("S = data.inspect(path)", &[PATH], &[INSPECTION])]);

const CREATE_INTEGER_INPUTS: &[BuiltinIntegerInputCapability] = &[
    BuiltinIntegerInputCapability {
        name: "schema.arrays.*.shape",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Array shapes accept scalar or vector dimensions in every integer class, or exactly integral floating values, within nonnegative platform and checked total-element bounds.",
    },
    BuiltinIntegerInputCapability {
        name: "schema.arrays.*.chunk",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Optional chunk shapes accept scalar or vector dimensions in every integer class, or exactly integral floating values; dimensions must be positive, platform-bounded, and rank-matched to the array shape.",
    },
];
pub const DATA_CREATE_INTEGER_CAPABILITIES: &[BuiltinIntegerCapabilityDescriptor] = &[
    BuiltinIntegerCapabilityDescriptor {
        form: "ds = data.create(path, struct_with_integer_shape_and_chunk)",
        inputs: CREATE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Schema dimensions are parsed directly from authoritative typed storage; the host/filesystem constructor allocates zero-filled storage in each array's declared dtype and returns a Dataset object.",
    },
];
