//! Method-only datetime indexing. The result of subsref depends on its path:
//! parentheses preserve datetime, while Format member access returns text.

use crate::catalog::inference::{argument_error, finish_fixed};
use crate::*;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, EffectKind, ExecutionStackRequirement, ShapeFact,
    ValueFact, ValueKindFact,
};

const OUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Indexed datetime or Format property; assignment returns the updated datetime.",
}];
const OBJ: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "obj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Datetime receiver object.",
};
const PATH: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Standard substruct-compatible indexing path.",
};
const SUBREF_INPUTS: [BuiltinParamDescriptor; 2] = [OBJ, PATH];
const SUBASGN_INPUTS: [BuiltinParamDescriptor; 3] = [
    OBJ,
    PATH,
    BuiltinParamDescriptor {
        name: "rhs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Assigned value.",
    },
];

pub const DATETIME_SUBSREF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "out = datetime.subsref(obj, S)",
        inputs: &SUBREF_INPUTS,
        outputs: &OUT,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &super::construction::DATETIME_ERRORS,
};
pub const DATETIME_SUBSASGN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "out = datetime.subsasgn(obj, S, rhs)",
        inputs: &SUBASGN_INPUTS,
        outputs: &OUT,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &super::construction::DATETIME_ERRORS,
};

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Datetime indexing implementation",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/datetime/operators.rs",
    ),
};
const TESTS: &[BuiltinEvidenceReference] = &[BuiltinEvidenceReference {
    kind: BuiltinEvidenceKind::UnitTest,
    label: "Datetime indexing and Format assignment",
    location: "crates/runmat-runtime/src/builtins/datetime/tests.rs",
}];
const SUBREF_DOC: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("datetime.subsref"),
    slug: Some("datetime.subsref"),
    summary: "Index datetime elements or read their Format property.",
    description: "Parentheses select datetime elements by a standard substruct-compatible path and preserve the datetime class and Format. Format member access returns text. Brace indexing and other properties are unsupported. The current implementation accepts one linear selector; native integer selectors are validated before conversion.",
    keywords: &["datetime", "subsref", "indexing", "Format"],
    related: &["datetime"],
    sections: &[BuiltinDocumentationSection {
        heading: "Results and restrictions",
        paragraphs: &["Parentheses indexing produces a host datetime object with the selected shape. Format member access produces a host string. Numeric selectors may suspend for a resident payload; the datetime serial-date storage itself is host-side. Multiple indexing dimensions, braces, and unrelated members are not implemented."],
    }],
    examples: &[BuiltinExample {
        id: "linear-index",
        title: "Select a datetime element",
        program: "t = datetime([2024 1 1; 2025 6 20]); one = t(2);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isdatetime(one)); assert(year(one)==2025);",
        },
    }],
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink {
        label: "datetime",
        target: BuiltinDocumentationLinkTarget::Builtin("datetime"),
    }, SOURCE],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE],
        verification: TESTS,
        notes: &["The result class cannot be fixed statically without the runtime index path."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
const SUBASGN_DOC: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("datetime.subsasgn"),
    slug: Some("datetime.subsasgn"),
    summary: "Assign the Format property of a datetime array.",
    description: "Format member assignment changes display formatting without changing stored serial dates. The value must be text. Parentheses element assignment, brace assignment, and other members are not implemented in the current datetime object path.",
    keywords: &["datetime", "subsasgn", "Format"],
    related: &["datetime"],
    sections: &[BuiltinDocumentationSection {
        heading: "Results and restrictions",
        paragraphs: &["The returned datetime object remains host-side and retains its serial-date values. Invalid properties or a non-text Format value fail without modifying the receiver."],
    }],
    examples: &[BuiltinExample {
        id: "format",
        title: "Change datetime rendering",
        program: "t = datetime(2024, 3, 14); t.Format = 'yyyy-MM-dd'; y = year(t);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isdatetime(t)); assert(strcmp(t.Format,'yyyy-MM-dd')); assert(y==2024);",
        },
    }],
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink {
        label: "datetime",
        target: BuiltinDocumentationLinkTarget::Builtin("datetime"),
    }, SOURCE],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE],
        verification: TESTS,
        notes: &["Only Format member assignment is implemented."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

const EFFECTS: &[EffectKind] = &[EffectKind::MaySuspend, EffectKind::MayThrow];
fn infer_index(request: &CallRequest, entry: &BuiltinCatalogEntry, arity: usize) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != arity {
        diagnostics.push(argument_error(
            "RM-DATETIME-INDEX-ARITY",
            "datetime indexing expects a receiver, path, and (for assignment) value",
            0,
        ));
    }
    if let Some(receiver) = request.arguments.first() {
        if !matches!(receiver.kind, ValueKindFact::Unknown)
            && !matches!(&receiver.kind, ValueKindFact::Object(object) if object.runtime_class.as_ref().is_none_or(|class| class == &runmat_types::standard::DATETIME.owned()))
        {
            diagnostics.push(argument_error(
                "RM-DATETIME-INDEX-RECEIVER",
                "expected datetime receiver",
                0,
            ));
        }
    }
    let mut result = ValueFact::unknown(DynamicReason::RuntimeValue);
    result.shape = if arity == 3 {
        request
            .arguments
            .first()
            .map_or(ShapeFact::Unknown, |value| value.shape.clone())
    } else {
        ShapeFact::Unknown
    };
    result.residency = runmat_types::ResidencyFact::Host;
    finish_fixed(entry, request, result, diagnostics)
}
fn infer_subref(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_index(request, entry, 2)
}
fn infer_subsasgn(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_index(request, entry, 3)
}

const fn entry(
    name: &'static str,
    descriptor: &'static BuiltinDescriptor,
    documentation: BuiltinDocumentation,
    inference: fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
) -> BuiltinCatalogEntry {
    BuiltinCatalogEntry {
        audience: crate::BuiltinCatalogAudience::Public,
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        identity: BuiltinCatalogIdentity { name },
        category: "datetime/indexing",
        documentation,
        descriptor,
        contract: BuiltinContractDeclaration {
            maturity: BuiltinContractMaturity::Incomplete,
            inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(inference)),
            compatibility: BuiltinCompatibility::Matlab,
            async_behavior: BuiltinAsyncBehavior::MaySuspend,
            purity: BuiltinPurity::Pure,
            semantic_kind: BuiltinSemanticKind::General,
            workspace_effect: None,
            environment_effect: None,
            effects: EFFECTS,
            capabilities: &[],
        },
        placement: BuiltinPlacementContract {
            preparation: BuiltinOperandPreparation::RuntimeOwned,
            portability: BuiltinPortability::NativeAndWasm,
            accelerator: BuiltinAcceleratorPolicy::Forbidden,
            residency: BuiltinResidencyPolicy::GatherToHost,
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
        integer_audit: None,
        suppress_auto_output: false,
    }
}

pub const DATETIME_SUBSREF_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "datetime.subsref",
    &DATETIME_SUBSREF_DESCRIPTOR,
    SUBREF_DOC,
    infer_subref,
);
pub const DATETIME_SUBSASGN_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "datetime.subsasgn",
    &DATETIME_SUBSASGN_DESCRIPTOR,
    SUBASGN_DOC,
    infer_subsasgn,
);

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &DATETIME_SUBSREF_CATALOG_ENTRY,
    &DATETIME_SUBSASGN_CATALOG_ENTRY,
];
