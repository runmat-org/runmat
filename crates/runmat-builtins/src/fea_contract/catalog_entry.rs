//! Shared FEA constructor catalog fields; identity facts stay in each entry.

use crate::*;
use runmat_types::{CallInference, CallRequest, EffectKind, ExecutionStackRequirement};

pub struct ConstructorEntryDefinition {
    pub provenance: BuiltinCatalogProvenance,
    pub name: &'static str,
    pub documentation: BuiltinDocumentation,
    pub descriptor: &'static BuiltinDescriptor,
    pub inference: fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
    pub integer_capabilities: &'static [BuiltinIntegerCapabilityDescriptor],
    pub integer_audit: Option<&'static BuiltinIntegerAuditDescriptor>,
    pub filesystem_read: bool,
}

/// Only layout shared by the family; prose and example values are supplied by
/// the owning identity module, not inferred from runtime macro metadata.
pub struct ConstructorDocumentationDefinition {
    pub title: &'static str,
    pub summary: &'static str,
    pub description: &'static str,
    pub keywords: &'static [&'static str],
    pub related: &'static [&'static str],
    pub sections: &'static [BuiltinDocumentationSection],
    pub examples: &'static [BuiltinExample],
    pub faqs: &'static [BuiltinDocumentationFaq],
    pub links: &'static [BuiltinDocumentationLink],
    pub implementation: &'static [BuiltinDocumentationLink],
    pub verification: &'static [BuiltinEvidenceReference],
    pub notes: &'static [&'static str],
}

pub const fn documentation(definition: ConstructorDocumentationDefinition) -> BuiltinDocumentation {
    BuiltinDocumentation {
        authority: BuiltinDocumentationAuthority::Catalog,
        title: Some(definition.title),
        slug: Some(definition.title),
        summary: definition.summary,
        description: definition.description,
        keywords: definition.keywords,
        related: definition.related,
        sections: definition.sections,
        examples: definition.examples,
        example_exemption: None,
        faqs: definition.faqs,
        links: definition.links,
        media: &[],
        evidence: BuiltinDocumentationEvidence {
            implementation: definition.implementation,
            verification: definition.verification,
            notes: definition.notes,
        },
        introduced: None,
        status: Some(BuiltinDocumentationStatus::Stable),
    }
}

pub const fn entry(definition: ConstructorEntryDefinition) -> BuiltinCatalogEntry {
    BuiltinCatalogEntry {
        provenance: definition.provenance,
        identity: BuiltinCatalogIdentity {
            name: definition.name,
        },
        category: "fea",
        documentation: definition.documentation,
        descriptor: definition.descriptor,
        contract: BuiltinContractDeclaration {
            maturity: BuiltinContractMaturity::Incomplete,
            inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
                definition.inference,
            )),
            compatibility: BuiltinCompatibility::Matlab,
            async_behavior: if definition.filesystem_read {
                BuiltinAsyncBehavior::MaySuspend
            } else {
                BuiltinAsyncBehavior::NeverSuspends
            },
            purity: if definition.filesystem_read {
                BuiltinPurity::DeterministicReadOnly
            } else {
                BuiltinPurity::Pure
            },
            semantic_kind: if definition.filesystem_read {
                BuiltinSemanticKind::Filesystem
            } else {
                BuiltinSemanticKind::General
            },
            workspace_effect: None,
            environment_effect: None,
            effects: if definition.filesystem_read {
                &[
                    EffectKind::EnvironmentRead,
                    EffectKind::FilesystemRead,
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
            residency: BuiltinResidencyPolicy::Host,
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
        integer_capabilities: definition.integer_capabilities,
        integer_audit: definition.integer_audit,
        suppress_auto_output: false,
    }
}
