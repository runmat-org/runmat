use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, CapabilityRequirement, DynamicReason,
    EffectKind, NumericClass, NumericDomain, NumericFact, ResidencyFact, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(1..=2).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-IMREAD-ARITY",
            "imread requires a source and optional format",
            0,
        ));
    }
    for (index, value) in request.arguments.iter().take(2).enumerate() {
        if !super::super::text::could_be_scalar_text(value) {
            diagnostics.push(argument_error(
                "RM-CATALOG-IMREAD-TEXT",
                "source and format must be scalar text",
                index,
            ));
        }
    }
    if request.arguments.len() >= 2
        && request
            .literals
            .literal_string_at(1)
            .is_some_and(|format| format.trim().trim_start_matches('.').is_empty())
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-IMREAD-FORMAT",
            "format hint must not be empty",
            1,
        ));
    }
    let mut image = ValueFact::unknown(DynamicReason::ExternalData);
    image.storage = StorageFact::Dense;
    image.residency = ResidencyFact::Host;
    let mut map = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(0), Some(0)]),
        StorageFact::Dense,
    );
    map.residency = ResidencyFact::Host;
    let mut alpha = ValueFact::unknown(DynamicReason::ExternalData);
    alpha.storage = StorageFact::Dense;
    alpha.residency = ResidencyFact::Host;
    let mut contract = CallContract::fixed(vec![image, map, alpha]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    match request.literals.literal_string_at(0) {
        Some(source)
            if source
                .get(..7)
                .is_some_and(|prefix| prefix.eq_ignore_ascii_case("http://"))
                || source
                    .get(..8)
                    .is_some_and(|prefix| prefix.eq_ignore_ascii_case("https://")) =>
        {
            contract.effects.0.remove(&EffectKind::FilesystemRead);
            contract
                .capabilities
                .0
                .remove(&CapabilityRequirement::Filesystem);
            contract
                .capabilities
                .0
                .insert(CapabilityRequirement::Network);
        }
        Some(_) => {
            contract.effects.0.remove(&EffectKind::Network);
            contract
                .capabilities
                .0
                .remove(&CapabilityRequirement::Network);
            contract
                .capabilities
                .0
                .insert(CapabilityRequirement::Filesystem);
        }
        None => {
            // The source value chooses one of two ambient services at runtime.
            // Until resolved, neither service may be presumed available.
            contract
                .capabilities
                .0
                .insert(CapabilityRequirement::Filesystem);
            contract
                .capabilities
                .0
                .insert(CapabilityRequirement::Network);
        }
    }
    let mut result = infer_call(&contract, request);
    if request
        .outputs
        .requested
        .known_count()
        .is_some_and(|count| count > 3)
    {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-IMREAD-OUTPUTS",
            "imread returns at most image, map, and alpha",
            3,
        ));
    }
    result.diagnostics.extend(diagnostics);
    result
}
