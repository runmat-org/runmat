use crate::*;
use runmat_types::EffectKind;

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar struct or struct array.",
}];
const EMPTY: [BuiltinParamDescriptor; 0] = [];
const TEMPLATE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "template",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Existing struct/array template or empty array.",
}];
const PAIRS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "field",
        ty: BuiltinParamType::PropertyName,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Field name.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Field value or cell array.",
    },
    BuiltinParamDescriptor {
        name: "name_value_pairs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional field/value pairs.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "S = struct()",
        inputs: &EMPTY,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = struct(template)",
        inputs: &TEMPLATE,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = struct(field, value, ...)",
        inputs: &PAIRS,
        outputs: &OUTPUT,
    },
];
macro_rules! struct_error {
    ($name:ident,$code:literal,$id:literal,$when:literal,$message:literal) => {
        pub const $name: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
            code: $code,
            identifier: Some($id),
            when: $when,
            message: $message,
        };
    };
}
struct_error!(STRUCT_ERROR_INVALID_SINGLE_INPUT,"RM.STRUCT.INVALID_SINGLE_INPUT","RunMat:struct:InvalidSingleInput","Single input is not a valid struct template.","struct: expected name/value pairs, an existing struct or struct array, or [] to create an empty struct array");
struct_error!(
    STRUCT_ERROR_NAME_VALUE_PAIRS,
    "RM.STRUCT.NAME_VALUE_PAIRS",
    "RunMat:struct:NameValuePairs",
    "Arguments are not complete pairs.",
    "struct: expected name/value pairs"
);
struct_error!(
    STRUCT_ERROR_CELL_SIZE_MISMATCH,
    "RM.STRUCT.CELL_SIZE_MISMATCH",
    "RunMat:struct:CellSizeMismatch",
    "Cell values have mismatched shapes.",
    "struct: cell inputs must have matching sizes"
);
struct_error!(
    STRUCT_ERROR_SIZE_OVERFLOW,
    "RM.STRUCT.SIZE_OVERFLOW",
    "RunMat:struct:SizeOverflow",
    "Requested size exceeds limits.",
    "struct: struct array size exceeds platform limits"
);
struct_error!(
    STRUCT_ERROR_ASSEMBLE_FAILED,
    "RM.STRUCT.ASSEMBLE_FAILED",
    "RunMat:struct:AssembleFailed",
    "Internal assembly failed.",
    "struct: failed to assemble struct array"
);
struct_error!(
    STRUCT_ERROR_EMPTY_ARRAY_FAILED,
    "RM.STRUCT.EMPTY_ARRAY_FAILED",
    "RunMat:struct:EmptyArrayFailed",
    "Empty array creation failed.",
    "struct: failed to create empty struct array"
);
struct_error!(
    STRUCT_ERROR_STRUCT_ARRAY_CONTENTS,
    "RM.STRUCT.STRUCT_ARRAY_CONTENTS",
    "RunMat:struct:StructArrayContents",
    "Cell contains non-struct values.",
    "struct: single argument cell input must contain structs"
);
struct_error!(
    STRUCT_ERROR_STRUCT_ARRAY_COPY_FAILED,
    "RM.STRUCT.STRUCT_ARRAY_COPY_FAILED",
    "RunMat:struct:StructArrayCopyFailed",
    "Struct array copy failed.",
    "struct: failed to copy struct array"
);
struct_error!(
    STRUCT_ERROR_FIELD_NAME_TYPE,
    "RM.STRUCT.FIELD_NAME_TYPE",
    "RunMat:struct:FieldNameType",
    "Field name has invalid type.",
    "struct: field names must be strings or character vectors"
);
struct_error!(
    STRUCT_ERROR_FIELD_NAME_SCALAR,
    "RM.STRUCT.FIELD_NAME_SCALAR",
    "RunMat:struct:FieldNameScalar",
    "Field name is not scalar.",
    "struct: field names must be scalar string arrays or character vectors"
);
struct_error!(
    STRUCT_ERROR_FIELD_NAME_CHAR_VECTOR,
    "RM.STRUCT.FIELD_NAME_CHAR_VECTOR",
    "RunMat:struct:FieldNameCharVector",
    "Character field name is not a row vector.",
    "struct: field names must be 1-by-N character vectors"
);
struct_error!(
    STRUCT_ERROR_FIELD_NAME_EMPTY,
    "RM.STRUCT.FIELD_NAME_EMPTY",
    "RunMat:struct:FieldNameEmpty",
    "Field name is empty.",
    "struct: field names must be nonempty"
);
struct_error!(
    STRUCT_ERROR_FIELD_NAME_START_CHAR,
    "RM.STRUCT.FIELD_NAME_START_CHAR",
    "RunMat:struct:FieldNameStartChar",
    "Field name is not a valid identifier.",
    "struct: field names must be valid MATLAB identifiers"
);
const ERRORS: [BuiltinErrorDescriptor; 13] = [
    STRUCT_ERROR_INVALID_SINGLE_INPUT,
    STRUCT_ERROR_NAME_VALUE_PAIRS,
    STRUCT_ERROR_CELL_SIZE_MISMATCH,
    STRUCT_ERROR_SIZE_OVERFLOW,
    STRUCT_ERROR_ASSEMBLE_FAILED,
    STRUCT_ERROR_EMPTY_ARRAY_FAILED,
    STRUCT_ERROR_STRUCT_ARRAY_CONTENTS,
    STRUCT_ERROR_STRUCT_ARRAY_COPY_FAILED,
    STRUCT_ERROR_FIELD_NAME_TYPE,
    STRUCT_ERROR_FIELD_NAME_SCALAR,
    STRUCT_ERROR_FIELD_NAME_CHAR_VECTOR,
    STRUCT_ERROR_FIELD_NAME_EMPTY,
    STRUCT_ERROR_FIELD_NAME_START_CHAR,
];
pub const STRUCT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
const BINDINGS: [BuiltinBindingDeclaration; 1] = [BuiltinBindingDeclaration {
    identity: BuiltinBindingIdentity {
        builtin: BuiltinCatalogIdentity { name: "struct" },
        variant: "default",
    },
    availability: BuiltinBindingAvailability::Required,
}];
const EFFECTS: [EffectKind; 1] = [EffectKind::MayThrow];
pub const STRUCT_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: BuiltinCatalogIdentity { name: "struct" },
    category: "structs/core",
    documentation: BuiltinDocumentation {
        summary: "Create scalar structs or struct arrays from field/value inputs.",
        keywords: &["name-value", "record", "struct", "structure"],
        related: &[],
        introduced: None,
        status: None,
        examples: &[],
    },
    descriptor: &STRUCT_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRuleId("aggregate.struct"),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::NeverSuspends,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &EFFECTS,
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
        residency: BuiltinResidencyPolicy::PreserveInputs,
        fusion: BuiltinFusionPolicy::Boundary,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        artifact_dependencies: &[],
    },
    bindings: &BINDINGS,
    extensions: &[],
    integer_capabilities: &[],
    integer_audit: None,
    suppress_auto_output: false,
};
