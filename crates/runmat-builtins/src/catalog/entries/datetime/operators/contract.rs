use super::*;
const INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor { name: "lhs", ty: BuiltinParamType::Any, arity: BuiltinParamArity::Required, default: None, description: "Datetime value receiver, with host serial-date storage." },
    BuiltinParamDescriptor { name: "rhs", ty: BuiltinParamType::Any, arity: BuiltinParamArity::Required, default: None, description: "Operation-specific datetime, duration, calendarDuration or host real numeric/logical operand." },
];
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out", ty: BuiltinParamType::Any, arity: BuiltinParamArity::Required, default: None,
    description: "Host double 0/1 comparison, datetime arithmetic result, or double day difference for datetime subtraction.",
}];
pub const DATETIME_OPERATOR_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DATETIME.OPERATOR.INVALID_INPUT", identifier: Some("RunMat:datetime:InvalidInput"),
    when: "Receiver storage or the operation-specific RHS class is invalid, shapes differ without scalar expansion, or calendar/integer range validation fails.",
    message: "datetime: invalid operator input",
};
const EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [crate::DATETIME_LOGICAL_INPUT_EXTENSION];
const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "rhs", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "RunMat accepts host numeric RHS day counts/serials. Every native integer is checked exactly against [-1000000000,1000000000] before conversion to binary64. This existing numeric method form is not gated by the constructor-only implicit-datenum extension.",
}];
pub const DATETIME_OPERATOR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "out = datetime_operator(datetime_lhs, integer_rhs)",
    inputs: &INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
    output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error,
    backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::SameSizeOrScalar,
    notes: "The left receiver is datetime, not integer. Numeric comparisons publish double 0/1 and numeric plus/minus publish datetime with double serial storage. Arrays require exact shape equality unless one operand has one element. No resident download is admitted.",
}];
const fn entry(
    name: &'static str,
    descriptor: &'static BuiltinDescriptor,
    documentation: BuiltinDocumentation,
    infer: fn(&runmat_types::CallRequest, &BuiltinCatalogEntry) -> runmat_types::CallInference,
) -> BuiltinCatalogEntry {
    BuiltinCatalogEntry {
        audience: BuiltinCatalogAudience::Public,
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        identity: BuiltinCatalogIdentity { name },
        category: "datetime/operators",
        documentation,
        descriptor,
        contract: BuiltinContractDeclaration {
            maturity: BuiltinContractMaturity::Incomplete,
            inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(infer)),
            compatibility: BuiltinCompatibility::Matlab,
            async_behavior: BuiltinAsyncBehavior::NeverSuspends,
            purity: BuiltinPurity::Pure,
            semantic_kind: BuiltinSemanticKind::General,
            workspace_effect: None,
            environment_effect: None,
            effects: &[runmat_types::EffectKind::MayThrow],
            capabilities: &[],
        },
        placement: BuiltinPlacementContract {
            preparation: BuiltinOperandPreparation::RuntimeOwned,
            portability: BuiltinPortability::NativeAndWasm,
            accelerator: BuiltinAcceleratorPolicy::Forbidden,
            residency: BuiltinResidencyPolicy::Host,
            fusion: BuiltinFusionPolicy::Boundary,
            distributed: BuiltinDistributedPolicy::Unsupported,
        },
        link: BuiltinLinkContract {
            reachability: BuiltinReachability::Always,
            policy: BuiltinLinkPolicy::PortableRuntime,
            execution_stack: runmat_types::ExecutionStackRequirement::Any,
            artifact_dependencies: &[],
        },
        bindings: &REQUIRED_DEFAULT_BINDING,
        extensions: &EXTENSIONS,
        integer_capabilities: &DATETIME_OPERATOR_INTEGER_CAPABILITIES,
        integer_audit: None,
        suppress_auto_output: false,
    }
}
macro_rules! method {
    ($catalog:ident, $descriptor:ident, $argument:ident, $outputs:ident, $name:literal, $label:literal, $argument_id:literal, $output_id:literal, $argument_code:literal, $output_code:literal, $doc:ident, $infer:ident) => {
        pub const $argument: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
            code: $argument_code,
            identifier: Some($argument_id),
            when: "The method input count is not exactly two.",
            message: concat!($name, ": expected datetime receiver and RHS"),
        };
        pub const $outputs: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
            code: $output_code,
            identifier: Some($output_id),
            when: "More than one output is requested.",
            message: concat!($name, ": too many output arguments"),
        };
        pub const $descriptor: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &[BuiltinSignatureDescriptor {
                label: $label,
                inputs: &INPUTS,
                outputs: &OUTPUT,
            }],
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::MethodOnly,
            errors: &[$argument, DATETIME_OPERATOR_ERROR_INVALID_INPUT, $outputs],
        };
        pub const $catalog: BuiltinCatalogEntry =
            entry($name, &$descriptor, documentation::$doc, inference::$infer);
    };
}
method!(
    DATETIME_EQ_CATALOG_ENTRY,
    DATETIME_EQ_DESCRIPTOR,
    DATETIME_EQ_ERROR_INVALID_ARGUMENT,
    DATETIME_EQ_ERROR_TOO_MANY_OUTPUTS,
    "datetime.eq",
    "out = datetime.eq(lhs, rhs)",
    "RunMat:datetime.eq:InvalidArgument",
    "RunMat:datetime.eq:TooManyOutputs",
    "RM.DATETIME.EQ.INVALID_ARGUMENT",
    "RM.DATETIME.EQ.TOO_MANY_OUTPUTS",
    EQ_DOC,
    infer_eq
);
method!(
    DATETIME_NE_CATALOG_ENTRY,
    DATETIME_NE_DESCRIPTOR,
    DATETIME_NE_ERROR_INVALID_ARGUMENT,
    DATETIME_NE_ERROR_TOO_MANY_OUTPUTS,
    "datetime.ne",
    "out = datetime.ne(lhs, rhs)",
    "RunMat:datetime.ne:InvalidArgument",
    "RunMat:datetime.ne:TooManyOutputs",
    "RM.DATETIME.NE.INVALID_ARGUMENT",
    "RM.DATETIME.NE.TOO_MANY_OUTPUTS",
    NE_DOC,
    infer_ne
);
method!(
    DATETIME_LT_CATALOG_ENTRY,
    DATETIME_LT_DESCRIPTOR,
    DATETIME_LT_ERROR_INVALID_ARGUMENT,
    DATETIME_LT_ERROR_TOO_MANY_OUTPUTS,
    "datetime.lt",
    "out = datetime.lt(lhs, rhs)",
    "RunMat:datetime.lt:InvalidArgument",
    "RunMat:datetime.lt:TooManyOutputs",
    "RM.DATETIME.LT.INVALID_ARGUMENT",
    "RM.DATETIME.LT.TOO_MANY_OUTPUTS",
    LT_DOC,
    infer_lt
);
method!(
    DATETIME_LE_CATALOG_ENTRY,
    DATETIME_LE_DESCRIPTOR,
    DATETIME_LE_ERROR_INVALID_ARGUMENT,
    DATETIME_LE_ERROR_TOO_MANY_OUTPUTS,
    "datetime.le",
    "out = datetime.le(lhs, rhs)",
    "RunMat:datetime.le:InvalidArgument",
    "RunMat:datetime.le:TooManyOutputs",
    "RM.DATETIME.LE.INVALID_ARGUMENT",
    "RM.DATETIME.LE.TOO_MANY_OUTPUTS",
    LE_DOC,
    infer_le
);
method!(
    DATETIME_GT_CATALOG_ENTRY,
    DATETIME_GT_DESCRIPTOR,
    DATETIME_GT_ERROR_INVALID_ARGUMENT,
    DATETIME_GT_ERROR_TOO_MANY_OUTPUTS,
    "datetime.gt",
    "out = datetime.gt(lhs, rhs)",
    "RunMat:datetime.gt:InvalidArgument",
    "RunMat:datetime.gt:TooManyOutputs",
    "RM.DATETIME.GT.INVALID_ARGUMENT",
    "RM.DATETIME.GT.TOO_MANY_OUTPUTS",
    GT_DOC,
    infer_gt
);
method!(
    DATETIME_GE_CATALOG_ENTRY,
    DATETIME_GE_DESCRIPTOR,
    DATETIME_GE_ERROR_INVALID_ARGUMENT,
    DATETIME_GE_ERROR_TOO_MANY_OUTPUTS,
    "datetime.ge",
    "out = datetime.ge(lhs, rhs)",
    "RunMat:datetime.ge:InvalidArgument",
    "RunMat:datetime.ge:TooManyOutputs",
    "RM.DATETIME.GE.INVALID_ARGUMENT",
    "RM.DATETIME.GE.TOO_MANY_OUTPUTS",
    GE_DOC,
    infer_ge
);
method!(
    DATETIME_PLUS_CATALOG_ENTRY,
    DATETIME_PLUS_DESCRIPTOR,
    DATETIME_PLUS_ERROR_INVALID_ARGUMENT,
    DATETIME_PLUS_ERROR_TOO_MANY_OUTPUTS,
    "datetime.plus",
    "out = datetime.plus(lhs, rhs)",
    "RunMat:datetime.plus:InvalidArgument",
    "RunMat:datetime.plus:TooManyOutputs",
    "RM.DATETIME.PLUS.INVALID_ARGUMENT",
    "RM.DATETIME.PLUS.TOO_MANY_OUTPUTS",
    PLUS_DOC,
    infer_plus
);
method!(
    DATETIME_MINUS_CATALOG_ENTRY,
    DATETIME_MINUS_DESCRIPTOR,
    DATETIME_MINUS_ERROR_INVALID_ARGUMENT,
    DATETIME_MINUS_ERROR_TOO_MANY_OUTPUTS,
    "datetime.minus",
    "out = datetime.minus(lhs, rhs)",
    "RunMat:datetime.minus:InvalidArgument",
    "RunMat:datetime.minus:TooManyOutputs",
    "RM.DATETIME.MINUS.INVALID_ARGUMENT",
    "RM.DATETIME.MINUS.TOO_MANY_OUTPUTS",
    MINUS_DOC,
    infer_minus
);
