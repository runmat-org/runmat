//! Shared spectral output-class rule; controls never determine signal class.
use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;

mod geometry;
#[cfg(test)]
mod tests;

#[derive(Clone, Copy)]
pub(super) enum Spectrum {
    Periodogram,
    Spectrogram,
}

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    spectrum: Spectrum,
) -> CallInference {
    let (named, max_args) = match spectrum {
        Spectrum::Periodogram => (2, 7),
        Spectrum::Spectrogram => (4, 10),
    };
    let mut diagnostics = Vec::new();
    let invalid_grid = geometry::invalid_grid(request, spectrum);
    if let Some(diagnostic) = invalid_grid.as_ref() {
        diagnostics.push(diagnostic.clone());
    }
    let invalid_options = geometry::invalid_options(request, spectrum);
    if let Some(diagnostic) = invalid_options.as_ref() {
        diagnostics.push(diagnostic.clone());
    }
    let invalid_controls = geometry::invalid_controls(request, spectrum);
    if let Some(diagnostic) = invalid_controls.as_ref() {
        diagnostics.push(diagnostic.clone());
    }
    let invalid_arity = request.arguments.is_empty() || request.arguments.len() > max_args;
    if invalid_arity {
        diagnostics.push(argument_error(
            "RM-CATALOG-SPECTRAL-ARITY",
            format!(
                "{} expects a signal and at most {} controls",
                entry.identity.name,
                max_args - 1
            ),
            request.arguments.len().saturating_sub(1),
        ));
    }
    let input = request.arguments.first();
    let class = match input.map(|value| &value.kind) {
        Some(ValueKindFact::Numeric(value)) if value.class == NumericClass::Single => {
            Some(NumericClass::Single)
        }
        Some(ValueKindFact::Numeric(value))
            if value.class == NumericClass::Double || value.domain == NumericDomain::Real =>
        {
            Some(NumericClass::Double)
        }
        Some(ValueKindFact::Logical) => Some(NumericClass::Double),
        Some(ValueKindFact::Unknown) | None => None,
        _ => {
            diagnostics.push(argument_error("RM-CATALOG-SPECTRAL-SIGNAL", "spectral estimation expects a real or complex floating signal, or an admitted real integer/logical signal", 0));
            None
        }
    };
    let count = request.outputs.requested.known_count().unwrap_or(1);
    let outputs = (0..count)
        .map(|slot| {
            if slot >= named {
                // Shared ABI zero padding remains host Double, not signal data.
                return ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                }));
            }
            if invalid_arity
                || invalid_grid.is_some()
                || invalid_options.is_some()
                || invalid_controls.is_some()
            {
                return ValueFact::unknown(DynamicReason::RuntimeValue);
            }
            let Some(class) = class else {
                return ValueFact::unknown(DynamicReason::RuntimeValue);
            };
            let shape = geometry::output_shape(request, spectrum, slot);
            let domain = if matches!(spectrum, Spectrum::Spectrogram) && slot == 0 {
                NumericDomain::Complex
            } else {
                NumericDomain::Real
            };
            let mut value = ValueFact::proven(
                ValueKindFact::Numeric(NumericFact { class, domain }),
                shape,
                StorageFact::Dense,
            );
            if !input.is_some_and(|input| input.residency == ResidencyFact::Host) {
                // Capability/intent decide residency. No provider layout proof is
                // implied by knowing the native numerical class.
                value.residency = ResidencyFact::Unknown;
                value.layout = LayoutFact::Unknown;
                value.contiguity = ContiguityFact::Unknown;
                value.view = ViewFact::Unknown;
                value.alias = AliasFact::Unknown;
            }
            value
        })
        .collect();
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.dynamic_outputs = request.outputs.requested.known_count().is_none();
    result.diagnostics.extend(diagnostics);
    result
}
