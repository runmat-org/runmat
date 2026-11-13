//! Canonical contracts for the existing inputParser handle implementation.
use crate::*;
mod contract;
mod documentation;
mod inference;
pub use contract::*;

struct Definition {
    name: &'static str,
    summary: &'static str,
    keywords: &'static [&'static str],
    documentation: Option<BuiltinDocumentation>,
    descriptor: &'static BuiltinDescriptor,
    infer: fn(&runmat_types::CallRequest, &BuiltinCatalogEntry) -> runmat_types::CallInference,
}

const fn entry(definition: Definition) -> BuiltinCatalogEntry {
    BuiltinCatalogEntry {
        audience: BuiltinCatalogAudience::Public,
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        identity: BuiltinCatalogIdentity { name: definition.name },
        category: "introspection",
        documentation: match definition.documentation {
            Some(documentation) => documentation,
            None => BuiltinDocumentation {
                authority: BuiltinDocumentationAuthority::Catalog,
                title: Some(definition.name),
                slug: Some(definition.name),
                summary: definition.summary,
                description: "inputParser supports case-sensitive named parameters with defaults. addParameter registers a parameter and rejects duplicates. parse accepts complete name-value pairs, rejects unknown names, uses the last supplied value for duplicates, and publishes Results only after validation. The current implementation does not support validators, positional parameters, or parser configuration options.",
                keywords: definition.keywords,
                related: &["inputParser", "addParameter", "parse"],
                examples: &[BuiltinExample {
                    id: "named-default", title: "Parse a named parameter",
                    program: "p = inputParser(); addParameter(p, 'Count', 2); parse(p, 'Count', 5); result = p.Results.Count;",
                    display_output: None,
                    compatibility: BuiltinExampleCompatibility::Matlab,
                    harness: BuiltinExampleHarness::Portable,
                    fixture: BuiltinExampleFixture::None,
                    requirements: BuiltinExampleRequirements::NONE,
                    verification: BuiltinExampleVerification::Assertions { source: "assert(result == 5);" },
                }],
                ..BuiltinDocumentation::EMPTY
            },
        },
        descriptor: definition.descriptor,
        contract: BuiltinContractDeclaration {
            maturity: BuiltinContractMaturity::Incomplete,
            inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(definition.infer)),
            compatibility: BuiltinCompatibility::Matlab,
            async_behavior: BuiltinAsyncBehavior::MaySuspend,
            purity: BuiltinPurity::Impure,
            semantic_kind: BuiltinSemanticKind::General,
            workspace_effect: None,
            environment_effect: None,
            // Mutating an aliased handle is not a side-effect-free expression.
            // Unknown is retained until the effect vocabulary models local
            // handle mutation; it does not invent an external service.
            effects: &[runmat_types::EffectKind::MaySuspend, runmat_types::EffectKind::MayThrow, runmat_types::EffectKind::Unknown],
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
            execution_stack: runmat_types::ExecutionStackRequirement::Process,
            artifact_dependencies: &[],
        },
        bindings: &REQUIRED_DEFAULT_BINDING,
        extensions: &[],
        integer_capabilities: &[],
        integer_audit: None,
        suppress_auto_output: false,
    }
}

pub const INPUT_PARSER_CATALOG_ENTRY: BuiltinCatalogEntry = entry(Definition {
    name: "inputParser",
    documentation: Some(documentation::DOCUMENTATION),
    summary: "Create a parser for MATLAB-style name-value arguments.",
    keywords: &["inputParser", "name-value", "arguments", "varargin"],
    descriptor: &INPUT_PARSER_DESCRIPTOR,
    infer: inference::construct,
});
pub const ADD_PARAMETER_CATALOG_ENTRY: BuiltinCatalogEntry = entry(Definition {
    name: "addParameter",
    documentation: None,
    summary: "Register an inputParser name-value parameter and default.",
    keywords: &["inputParser", "addParameter", "name-value", "varargin"],
    descriptor: &ADD_PARAMETER_DESCRIPTOR,
    infer: inference::add_parameter,
});
pub const PARSE_CATALOG_ENTRY: BuiltinCatalogEntry = entry(Definition {
    name: "parse",
    documentation: None,
    summary: "Parse name-value arguments with an inputParser.",
    keywords: &["inputParser", "parse", "name-value", "varargin"],
    descriptor: &PARSE_DESCRIPTOR,
    infer: inference::parse,
});
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &INPUT_PARSER_CATALOG_ENTRY,
    &ADD_PARAMETER_CATALOG_ENTRY,
    &PARSE_CATALOG_ENTRY,
];
