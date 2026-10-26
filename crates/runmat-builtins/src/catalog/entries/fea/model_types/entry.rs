//! Documentation migration retains conservative legacy result/effect knowledge.
use crate::*;
use runmat_types::{CallInference, CallRequest, DynamicReason, EffectKind, ValueFact};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    // The old Value-returning registrations had no identity-specific resolver.
    // Do not invent static object fields as part of a documentation cutover.
    crate::catalog::finish_fixed(
        entry,
        request,
        ValueFact::unknown(DynamicReason::RuntimeValue),
        Vec::new(),
    )
}

pub(super) const fn entry(
    provenance: BuiltinCatalogProvenance,
    name: &'static str,
    documentation: BuiltinDocumentation,
    descriptor: &'static BuiltinDescriptor,
    integer_capabilities: &'static [BuiltinIntegerCapabilityDescriptor],
    integer_audit: Option<&'static BuiltinIntegerAuditDescriptor>,
) -> BuiltinCatalogEntry {
    let mut entry = crate::fea_contract::catalog_entry::entry(
        crate::fea_contract::catalog_entry::ConstructorEntryDefinition {
            provenance,
            name,
            documentation,
            descriptor,
            inference: infer,
            integer_capabilities,
            integer_audit,
            filesystem_read: false,
            distributed: BuiltinDistributedPolicy::Unsupported,
        },
    );
    // Keep the legacy async/unknown-effect conservatism. This metadata-only
    // migration does not certify purity or a stronger static result contract.
    entry.contract.async_behavior = BuiltinAsyncBehavior::MaySuspend;
    // Preserve the legacy fallback flag; this is not a newly asserted MATLAB
    // API-parity guarantee for these RunMat-namespaced constructors.
    entry.contract.compatibility = BuiltinCompatibility::Matlab;
    entry.contract.purity = BuiltinPurity::Impure;
    entry.contract.effects = &[
        EffectKind::MaySuspend,
        EffectKind::MayThrow,
        EffectKind::Unknown,
    ];
    entry
}
