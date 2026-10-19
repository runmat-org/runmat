//! Shared FEA constructor catalog fields; identity facts stay in each entry.

use crate::*;
use runmat_types::{
    CallInference, CallRequest, CapabilityRequirement, EffectKind, ExecutionStackRequirement,
};

pub struct ConstructorEntryDefinition {
    pub provenance: BuiltinCatalogProvenance,
    pub name: &'static str,
    pub documentation: BuiltinDocumentation,
    pub descriptor: &'static BuiltinDescriptor,
    pub inference: fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
    pub integer_capabilities: &'static [BuiltinIntegerCapabilityDescriptor],
    pub integer_audit: Option<&'static BuiltinIntegerAuditDescriptor>,
    pub filesystem_read: bool,
    /// Identity-specific distributed admission; never infer from FEA category.
    pub distributed: BuiltinDistributedPolicy,
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
        audience: crate::BuiltinCatalogAudience::Public,
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
            compatibility: BuiltinCompatibility::RunMat,
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
            capabilities: if definition.filesystem_read {
                &[CapabilityRequirement::Filesystem]
            } else {
                &[]
            },
        },
        placement: BuiltinPlacementContract {
            preparation: BuiltinOperandPreparation::RuntimeOwned,
            portability: BuiltinPortability::NativeAndWasm,
            accelerator: BuiltinAcceleratorPolicy::Forbidden,
            residency: BuiltinResidencyPolicy::Host,
            fusion: BuiltinFusionPolicy::Boundary,
            distributed: definition.distributed,
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

/// A constructor that may persist a supplied artifact (for example, a solver
/// mesh passed to `fea.study`) into the configured analysis artifact store.
/// The store can be in memory, so filesystem access is an effect, not a
/// universal capability prerequisite.
pub const fn artifact_writing_entry(definition: ConstructorEntryDefinition) -> BuiltinCatalogEntry {
    let mut entry = entry(definition);
    entry.contract.purity = BuiltinPurity::Impure;
    entry.contract.semantic_kind = BuiltinSemanticKind::Filesystem;
    entry.contract.async_behavior = BuiltinAsyncBehavior::NeverSuspends;
    entry.contract.effects = &[
        EffectKind::EnvironmentRead,
        EffectKind::FilesystemWrite,
        EffectKind::MayThrow,
    ];
    entry.contract.capabilities = &[];
    entry
}
