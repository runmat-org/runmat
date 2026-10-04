//! Shared public receiver slots and stable data/placement diagnostics.

use crate::*;

macro_rules! receiver_descriptor {
    ($name:ident, $label:literal, $inputs:expr, $outputs:expr) => {
        pub const $name: crate::BuiltinDescriptor = crate::BuiltinDescriptor {
            signatures: &[crate::BuiltinSignatureDescriptor {
                label: $label,
                inputs: $inputs,
                outputs: $outputs,
            }],
            output_mode: crate::BuiltinOutputMode::Fixed,
            completion_policy: crate::BuiltinCompletionPolicy::Public,
            errors: super::super::descriptor::ERRORS,
        };
    };
}

pub(super) use receiver_descriptor;

pub(super) const fn param(
    name: &'static str,
    ty: BuiltinParamType,
    arity: BuiltinParamArity,
    description: &'static str,
) -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name,
        ty,
        arity,
        default: None,
        description,
    }
}

pub(super) const DATASET: BuiltinParamDescriptor = param(
    "ds",
    BuiltinParamType::Any,
    BuiltinParamArity::Required,
    "Managed Dataset receiver.",
);
pub(super) const ARRAY: BuiltinParamDescriptor = param(
    "arr",
    BuiltinParamType::Any,
    BuiltinParamArity::Required,
    "Managed DataArray receiver.",
);
pub(super) const TRANSACTION: BuiltinParamDescriptor = param(
    "tx",
    BuiltinParamType::Any,
    BuiltinParamArity::Required,
    "Managed DataTransaction receiver.",
);
pub(super) const NAME: BuiltinParamDescriptor = param(
    "name",
    BuiltinParamType::StringScalar,
    BuiltinParamArity::Required,
    "Array name.",
);
pub(super) const KEY: BuiltinParamDescriptor = param(
    "key",
    BuiltinParamType::StringScalar,
    BuiltinParamArity::Required,
    "Attribute key.",
);
pub(super) const VALUE: BuiltinParamDescriptor = param(
    "value",
    BuiltinParamType::Any,
    BuiltinParamArity::Required,
    "Value to store.",
);
pub(super) const ATTRS: BuiltinParamDescriptor = param(
    "attrs",
    BuiltinParamType::Any,
    BuiltinParamArity::Required,
    "Attribute structure.",
);
pub(super) const LABEL: BuiltinParamDescriptor = param(
    "label",
    BuiltinParamType::StringScalar,
    BuiltinParamArity::Required,
    "Snapshot label.",
);
pub(super) const SLICE_OPTIONAL: BuiltinParamDescriptor = param(
    "sliceSpec",
    BuiltinParamType::Any,
    BuiltinParamArity::Optional,
    "Optional cell slice specification or ':' text.",
);
pub(super) const SLICE_REQUIRED: BuiltinParamDescriptor = param(
    "sliceSpec",
    BuiltinParamType::Any,
    BuiltinParamArity::Required,
    "Cell slice specification or ':' text.",
);
pub(super) const VALUES: BuiltinParamDescriptor = param(
    "values",
    BuiltinParamType::Any,
    BuiltinParamArity::Required,
    "Numeric values to persist.",
);
pub(super) const NEW_SHAPE: BuiltinParamDescriptor = param(
    "newShape",
    BuiltinParamType::Any,
    BuiltinParamArity::Required,
    "New checked numeric dimension vector.",
);
pub(super) const META: BuiltinParamDescriptor = param(
    "meta",
    BuiltinParamType::Any,
    BuiltinParamArity::Required,
    "Array metadata structure.",
);
pub(super) const COMMIT_OPTIONS: BuiltinParamDescriptor = param(
    "options",
    BuiltinParamType::Any,
    BuiltinParamArity::Optional,
    "Optional scalar structure with an if_manifest precondition.",
);
pub(super) const DEFAULT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "defaultValue",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: Some("0"),
    description: "Fallback when the attribute is absent.",
};
pub(super) const OUT_STRING: BuiltinParamDescriptor = param(
    "s",
    BuiltinParamType::StringScalar,
    BuiltinParamArity::Required,
    "Text result.",
);
pub(super) const OUT_BOOL: BuiltinParamDescriptor = param(
    "tf",
    BuiltinParamType::LogicalArray,
    BuiltinParamArity::Required,
    "Logical result.",
);
pub(super) const OUT_VALUE: BuiltinParamDescriptor = param(
    "value",
    BuiltinParamType::Any,
    BuiltinParamArity::Required,
    "Value result.",
);
pub(super) const OUT_DATASET: BuiltinParamDescriptor = param(
    "ds",
    BuiltinParamType::Any,
    BuiltinParamArity::Required,
    "Managed Dataset handle.",
);
pub(super) const OUT_ARRAY: BuiltinParamDescriptor = param(
    "arr",
    BuiltinParamType::Any,
    BuiltinParamArity::Required,
    "Managed DataArray handle.",
);
pub(super) const OUT_TRANSACTION: BuiltinParamDescriptor = param(
    "tx",
    BuiltinParamType::Any,
    BuiltinParamArity::Required,
    "Managed DataTransaction handle.",
);
pub(super) const OUT_CELL: BuiltinParamDescriptor = param(
    "C",
    BuiltinParamType::Any,
    BuiltinParamArity::Required,
    "Cell array of names.",
);
pub(super) const OUT_STRUCT: BuiltinParamDescriptor = param(
    "S",
    BuiltinParamType::Any,
    BuiltinParamArity::Required,
    "Attribute structure.",
);
pub(super) const OUT_TENSOR: BuiltinParamDescriptor = param(
    "X",
    BuiltinParamType::NumericArray,
    BuiltinParamArity::Required,
    "Numeric tensor result.",
);

pub(super) const DATA_ERROR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DATA.ERROR",
    identifier: Some("RUNMAT:Data:Error"),
    when: "A Dataset family receiver, argument, manifest, payload, or filesystem operation fails.",
    message: "data: operation failed",
};
pub(super) const MANIFEST_CONFLICT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DATA.MANIFEST_CONFLICT",
    identifier: Some("RunMat:data:ManifestConflict"),
    when: "A transaction's expected manifest sequence no longer matches the stored manifest.",
    message: "data: manifest conflict",
};
pub(super) const TRANSACTION_NOT_FOUND: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DATA.TRANSACTION_NOT_FOUND",
    identifier: Some("RunMat:data:TransactionNotFound"),
    when: "The transaction is absent from the process-local registry.",
    message: "data: transaction not found",
};
pub(super) const ERRORS: &[BuiltinErrorDescriptor] =
    &[DATA_ERROR, MANIFEST_CONFLICT, TRANSACTION_NOT_FOUND];
