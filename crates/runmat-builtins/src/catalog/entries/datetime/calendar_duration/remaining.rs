//! Remaining datetime predicate, compatibility call, and calendar methods.
use crate::*;
use runmat_types::{CallInference, CallRequest, EffectKind, ExecutionStackRequirement};

mod documentation;
mod inference;
#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CalendarResidualOperation {
    IsDatetime,
    DateTick,
    Plus,
    Minus,
    Eq,
    Ne,
}
impl CalendarResidualOperation {
    pub const fn name(self) -> &'static str {
        match self {
            Self::IsDatetime => "isdatetime",
            Self::DateTick => "datetick",
            Self::Plus => "calendarDuration.plus",
            Self::Minus => "calendarDuration.minus",
            Self::Eq => "calendarDuration.eq",
            Self::Ne => "calendarDuration.ne",
        }
    }
    pub const fn entry(self) -> &'static BuiltinCatalogEntry {
        match self {
            Self::IsDatetime => &ISDATETIME_CATALOG_ENTRY,
            Self::DateTick => &DATETICK_CATALOG_ENTRY,
            Self::Plus => &CALENDAR_DURATION_PLUS_CATALOG_ENTRY,
            Self::Minus => &CALENDAR_DURATION_MINUS_CATALOG_ENTRY,
            Self::Eq => &CALENDAR_DURATION_EQ_CATALOG_ENTRY,
            Self::Ne => &CALENDAR_DURATION_NE_CATALOG_ENTRY,
        }
    }
    pub const fn input_count(self) -> Option<usize> {
        match self {
            Self::IsDatetime => Some(1),
            Self::DateTick => None,
            _ => Some(2),
        }
    }
}
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &ISDATETIME_CATALOG_ENTRY,
    &DATETICK_CATALOG_ENTRY,
    &CALENDAR_DURATION_PLUS_CATALOG_ENTRY,
    &CALENDAR_DURATION_MINUS_CATALOG_ENTRY,
    &CALENDAR_DURATION_EQ_CATALOG_ENTRY,
    &CALENDAR_DURATION_NE_CATALOG_ENTRY,
];

pub const CALENDAR_RESIDUAL_ERROR_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DATETIME.RESIDUAL.ARGUMENT",
    identifier: Some("RunMat:datetime:InvalidArgument"),
    when: "Input count is not one for isdatetime or two for a calendarDuration method.",
    message: "datetime: invalid argument count",
};
pub const CALENDAR_RESIDUAL_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DATETIME.RESIDUAL.OUTPUTS",
    identifier: Some("RunMat:datetime:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "datetime: too many output arguments",
};
pub const CALENDAR_RESIDUAL_ERROR_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DATETIME.RESIDUAL.INPUT", identifier: Some("RunMat:datetime:InvalidInput"),
    when: "Calendar receiver/component storage, operand class, scalar expansion, or calendar range is invalid.",
    message: "datetime: invalid calendar operator input",
};
pub const CALENDAR_RESIDUAL_ERROR_WORKSPACE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DATETIME.RESIDUAL.WORKSPACE", identifier: Some("RunMat:parallel:InfrastructureFailure"),
    when: "Calendar scratch has no complete managed owner under the selected finite allocation authority; the typed MissingManagedOwner control failure is preserved as infrastructure failure.",
    message: "datetime: calendar workspace requires unrestricted allocation",
};
const ERRORS: &[BuiltinErrorDescriptor] = &[
    CALENDAR_RESIDUAL_ERROR_ARGUMENT,
    CALENDAR_RESIDUAL_ERROR_OUTPUTS,
    CALENDAR_RESIDUAL_ERROR_INPUT,
    CALENDAR_RESIDUAL_ERROR_WORKSPACE,
];
const INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None,
    notes: "isdatetime inspects class without numeric conversion; datetick ignores gathered inputs; calendar methods accept typed calendar/datetime objects, not integer values or IDs.",
};
const VALUE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Value whose exact runtime class is inspected.",
};
const BINARY: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "lhs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "calendarDuration receiver.",
    },
    BuiltinParamDescriptor {
        name: "rhs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "calendarDuration; plus also accepts datetime.",
    },
];
const IGNORED: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "args",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Accepted compatibility arguments; no axis formatting is performed.",
}];
const OUT: [BuiltinParamDescriptor;1] = [BuiltinParamDescriptor {
    name: "out", ty: BuiltinParamType::Any, arity: BuiltinParamArity::Required, default: None,
    description: "Logical scalar predicate, double-zero no-op status, calendar/datetime result, or host double 0/1 comparison.",
}];
const LOGICAL_OUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "One logical scalar class predicate.",
}];
const ZERO_OUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Double zero; no axis change is made.",
}];
const DOUBLE_OUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host double zero/one, not logical; scalar or matched/expanded shape.",
}];
const fn entry(
    op: CalendarResidualOperation,
    descriptor: &'static BuiltinDescriptor,
    documentation: BuiltinDocumentation,
    infer: fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
) -> BuiltinCatalogEntry {
    let tick = matches!(op, CalendarResidualOperation::DateTick);
    BuiltinCatalogEntry {
        audience: BuiltinCatalogAudience::Public,
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        identity: BuiltinCatalogIdentity { name: op.name() },
        category: "datetime/calendar-duration",
        documentation,
        descriptor,
        contract: BuiltinContractDeclaration {
            maturity: BuiltinContractMaturity::Incomplete,
            inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(infer)),
            compatibility: if tick {
                BuiltinCompatibility::RunMat
            } else {
                BuiltinCompatibility::Matlab
            },
            async_behavior: if tick {
                BuiltinAsyncBehavior::MaySuspend
            } else {
                BuiltinAsyncBehavior::NeverSuspends
            },
            purity: BuiltinPurity::Pure,
            semantic_kind: BuiltinSemanticKind::General,
            workspace_effect: None,
            environment_effect: None,
            effects: if tick {
                &[
                    EffectKind::HostCallback,
                    EffectKind::MaySuspend,
                    EffectKind::MayThrow,
                ]
            } else {
                &[EffectKind::MayThrow]
            },
            capabilities: &[],
        },
        placement: BuiltinPlacementContract {
            preparation: BuiltinOperandPreparation::RuntimeOwned,
            portability: BuiltinPortability::NativeAndWasm,
            accelerator: BuiltinAcceleratorPolicy::Forbidden,
            residency: if tick {
                BuiltinResidencyPolicy::GatherToHost
            } else {
                BuiltinResidencyPolicy::Host
            },
            fusion: BuiltinFusionPolicy::Boundary,
            distributed: BuiltinDistributedPolicy::Unsupported,
        },
        link: BuiltinLinkContract {
            reachability: BuiltinReachability::Always,
            policy: BuiltinLinkPolicy::PortableRuntime,
            execution_stack: ExecutionStackRequirement::Any,
            artifact_dependencies: &[],
        },
        bindings: &REQUIRED_DEFAULT_BINDING,
        extensions: &[],
        integer_capabilities: &[],
        integer_audit: Some(&INTEGER_AUDIT),
        suppress_auto_output: tick,
    }
}
macro_rules! operation {
    ($catalog:ident,$descriptor:ident,$op:ident,$label:literal,$inputs:expr,$method:literal,$doc:ident,$infer:ident) => {
        pub const $descriptor: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &[BuiltinSignatureDescriptor {
                label: $label,
                inputs: $inputs,
                outputs: match CalendarResidualOperation::$op {
                    CalendarResidualOperation::IsDatetime => &LOGICAL_OUT,
                    CalendarResidualOperation::DateTick => &ZERO_OUT,
                    CalendarResidualOperation::Eq | CalendarResidualOperation::Ne => &DOUBLE_OUT,
                    _ => &OUT,
                },
            }],
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: if $method {
                BuiltinCompletionPolicy::MethodOnly
            } else {
                BuiltinCompletionPolicy::Public
            },
            errors: ERRORS,
        };
        pub const $catalog: BuiltinCatalogEntry = entry(
            CalendarResidualOperation::$op,
            &$descriptor,
            documentation::$doc,
            inference::$infer,
        );
    };
}
operation!(
    ISDATETIME_CATALOG_ENTRY,
    ISDATETIME_DESCRIPTOR,
    IsDatetime,
    "tf = isdatetime(X)",
    &[VALUE],
    false,
    ISDATETIME,
    predicate
);
operation!(
    DATETICK_CATALOG_ENTRY,
    DATETICK_DESCRIPTOR,
    DateTick,
    "status = datetick(args...)",
    &IGNORED,
    false,
    DATETICK,
    tick
);
operation!(
    CALENDAR_DURATION_PLUS_CATALOG_ENTRY,
    CALENDAR_DURATION_PLUS_DESCRIPTOR,
    Plus,
    "out = calendarDuration.plus(lhs,rhs)",
    &BINARY,
    true,
    PLUS,
    plus
);
operation!(
    CALENDAR_DURATION_MINUS_CATALOG_ENTRY,
    CALENDAR_DURATION_MINUS_DESCRIPTOR,
    Minus,
    "out = calendarDuration.minus(lhs,rhs)",
    &BINARY,
    true,
    MINUS,
    minus
);
operation!(
    CALENDAR_DURATION_EQ_CATALOG_ENTRY,
    CALENDAR_DURATION_EQ_DESCRIPTOR,
    Eq,
    "out = calendarDuration.eq(lhs,rhs)",
    &BINARY,
    true,
    EQ,
    comparison
);
operation!(
    CALENDAR_DURATION_NE_CATALOG_ENTRY,
    CALENDAR_DURATION_NE_DESCRIPTOR,
    Ne,
    "out = calendarDuration.ne(lhs,rhs)",
    &BINARY,
    true,
    NE,
    comparison
);
