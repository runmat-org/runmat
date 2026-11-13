//! Conservative semantics for previously unjoined default support bindings.
use crate::*;
use runmat_types::{CallInference, CallRequest, DynamicReason, EffectKind, ValueFact};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
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
) -> BuiltinCatalogEntry {
    let mut entry = crate::fea_contract::catalog_entry::entry(
        crate::fea_contract::catalog_entry::ConstructorEntryDefinition {
            provenance,
            name,
            documentation,
            descriptor,
            inference: infer,
            integer_capabilities: &[],
            integer_audit: Some(&super::constraint_support::FEA_TYPED_CONSTRAINT_INTEGER_AUDIT),
            filesystem_read: false,
            distributed: BuiltinDistributedPolicy::Unsupported,
        },
    );
    // These bindings had no canonical inference declaration. Retain the
    // pre-migration Runtime fallback rather than certify stronger static facts.
    entry.contract.compatibility = BuiltinCompatibility::Matlab;
    entry.contract.async_behavior = BuiltinAsyncBehavior::MaySuspend;
    entry.contract.purity = BuiltinPurity::Impure;
    entry.contract.effects = &[
        EffectKind::MaySuspend,
        EffectKind::MayThrow,
        EffectKind::Unknown,
    ];
    entry
}
