//! Shared, value-independent facts for the named RGB palette generators.

use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    *,
};
use runmat_types::*;

pub const MAX_PALETTE_LENGTH: usize = 1_000_000;

pub(super) const INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
pub(super) const INPUTS_LENGTH: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "m",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Nonnegative integral number of RGB rows; defaults to the current colormap length.",
}];
pub(super) const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "c",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host double m-by-3 RGB matrix with entries in [0,1].",
}];
pub const PALETTE_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COLORMAP_ARRAY.INVALID_ARGUMENT",
    identifier: Some("RunMat:colormapArray:InvalidArgument"),
    when: "The length is not a host real numeric scalar containing a nonnegative integer, or exceeds the existing allocation guard.",
    message: "colormap array: invalid argument",
};
pub(super) const ERRORS: [BuiltinErrorDescriptor; 1] = [PALETTE_INVALID_ARGUMENT];

pub(super) const LENGTH_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "m",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "All eight host integer classes are read exactly as a structural row count. Resident controls are rejected without provider access.",
    }];

pub(super) const EXTENSION_LENGTH_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "m",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The RunMat-only palette accepts all eight exact host integer row-count classes; no MATLAB builtin compatibility is claimed.",
    }];
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-PALETTE-ARITY",
            "a palette generator accepts at most one length",
            1,
        ));
    }
    let mut rows = None;
    if let Some(argument) = request.arguments.first() {
        let invalid_kind = match argument.kind {
            ValueKindFact::Numeric(numeric) => numeric.domain == NumericDomain::Complex,
            ValueKindFact::Unknown => false,
            _ => true,
        };
        if invalid_kind
            || argument
                .shape
                .element_count()
                .is_some_and(|count| count != 1)
            || matches!(argument.storage, StorageFact::Sparse)
            || matches!(
                argument.residency,
                ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
            )
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-PALETTE-LENGTH",
                "palette length must be one host real numeric scalar",
                0,
            ));
        }
        if let Some(literal) = request.literals.literal_args.first() {
            let known = match literal {
                LiteralValue::Number(value) => Some(*value),
                LiteralValue::Real { text, .. } => text.parse::<f64>().ok(),
                LiteralValue::Integer { .. } => literal.integer_value_i128().map(|n| n as f64),
                _ => None,
            };
            if let Some(value) = known {
                if !value.is_finite()
                    || value < 0.0
                    || value.fract() != 0.0
                    || value > MAX_PALETTE_LENGTH as f64
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-PALETTE-LENGTH-RANGE",
                        "palette length must be a nonnegative integer within the allocation guard",
                        0,
                    ));
                } else {
                    rows = Some(value as usize);
                }
            }
        }
    }
    let output = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![rows, Some(3)]),
        StorageFact::Dense,
    );
    let mut result = finish_fixed_outputs(entry, request, vec![output], diagnostics);
    if !request.arguments.is_empty() {
        // Only the omitted-length form observes selected graphics state.
        result.effects.0.remove(&EffectKind::EnvironmentRead);
    }
    result
}
