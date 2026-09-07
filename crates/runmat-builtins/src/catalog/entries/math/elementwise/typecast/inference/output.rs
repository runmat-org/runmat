use super::{representation::OutputRepresentation, shape, source};
use crate::catalog::inference::materialize;
use runmat_types::{DynamicReason, InferenceDiagnostic, ResidencyFact, StorageFact, ValueFact};

pub(super) fn fact(
    source: &ValueFact,
    target: OutputRepresentation,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> ValueFact {
    let Some(source_width) = source::byte_width(source) else {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    };
    let output_shape = shape::output(
        &source.shape,
        source_width,
        target.byte_width(),
        diagnostics,
    );
    let mut output = ValueFact::proven(target.kind(), output_shape, StorageFact::Dense);
    materialize(&mut output);
    output.residency = match &source.residency {
        ResidencyFact::Host => ResidencyFact::Host,
        ResidencyFact::Device { provider } => ResidencyFact::Device {
            provider: provider.clone(),
        },
        ResidencyFact::Unknown | ResidencyFact::Remote { .. } => ResidencyFact::Unknown,
    };
    output
}
