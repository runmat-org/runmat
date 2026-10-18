//! Conservative facts for `besselj`: a real Z may cross the complex branch cut.

use crate::catalog::inference::{argument_error, finish_fixed, numeric_kind};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericClass, NumericDomain, ResidencyFact,
    ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

pub(in crate::catalog::inference) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(2..=3).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-BESSELJ-ARITY",
            "besselj requires nu, Z, and optional scale",
            request.arguments.len().min(2),
        ));
    }
    let [order, argument, rest @ ..] = request.arguments.as_slice() else {
        return finish_fixed(entry, request, unknown(), diagnostics);
    };

    let order_class = input_class(order, false, 0, &mut diagnostics);
    let argument_class = input_class(argument, true, 1, &mut diagnostics);
    if let Some(scale) = rest.first() {
        if !matches!(&scale.kind, ValueKindFact::Unknown)
            && (!scale.is_scalar()
                || !matches!(&scale.kind, ValueKindFact::Numeric(n) if n.domain == NumericDomain::Real && matches!(n.class, NumericClass::Single | NumericClass::Double)))
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-BESSELJ-SCALE",
                "scale must be a real floating scalar equal to 0 or 1",
                2,
            ));
        }
    }

    let shape = if order.is_scalar() {
        argument.shape.clone()
    } else if argument.is_scalar() || order.shape == argument.shape {
        order.shape.clone()
    } else if matches!(order.shape, ShapeFact::Unknown)
        || matches!(argument.shape, ShapeFact::Unknown)
    {
        ShapeFact::Unknown
    } else {
        diagnostics.push(argument_error(
            "RM-CATALOG-BESSELJ-SIZE",
            "nu and Z must be equal-size or one must be scalar",
            1,
        ));
        ShapeFact::Unknown
    };

    let Some((order_class, argument_class)) = order_class.zip(argument_class) else {
        return finish_fixed(entry, request, unknown_with_shape(shape), diagnostics);
    };
    let class = if order_class == NumericClass::Single || argument_class == NumericClass::Single {
        NumericClass::Single
    } else {
        NumericClass::Double
    };
    let mut output = unknown_with_shape(shape);
    if matches!(&argument.kind, ValueKindFact::Numeric(n) if n.domain == NumericDomain::Complex) {
        output.kind = numeric_kind(class, NumericDomain::Complex);
    }
    // A real argument may be negative with fractional nu.  Without value-range
    // evidence, claiming a real result would make a static lowering unsound.
    output.storage = if output.is_scalar() {
        StorageFact::Scalar
    } else {
        StorageFact::Dense
    };
    output.residency = ResidencyFact::Host;
    crate::catalog::inference::support::facts::materialize(&mut output);
    finish_fixed(entry, request, output, diagnostics)
}

fn input_class(
    input: &ValueFact,
    complex_allowed: bool,
    index: usize,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) -> Option<NumericClass> {
    if matches!(&input.kind, ValueKindFact::Unknown) {
        return None;
    }
    if let ValueKindFact::Numeric(numeric) = &input.kind {
        if matches!(numeric.class, NumericClass::Single | NumericClass::Double)
            && (complex_allowed || numeric.domain == NumericDomain::Real)
            && !matches!(input.storage, StorageFact::Sparse)
            && !matches!(input.residency, ResidencyFact::Device { .. })
        {
            return Some(numeric.class);
        }
    }
    diagnostics.push(argument_error(
        "RM-CATALOG-BESSELJ-INPUT",
        if complex_allowed {
            "Z must be host dense real or complex single/double"
        } else {
            "nu must be host dense real single/double"
        },
        index,
    ));
    None
}

fn unknown() -> ValueFact {
    ValueFact::unknown(DynamicReason::RuntimeValue)
}

fn unknown_with_shape(shape: ShapeFact) -> ValueFact {
    let mut result = unknown();
    result.shape = shape;
    result
}
