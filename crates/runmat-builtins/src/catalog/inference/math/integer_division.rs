use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    broadcast_shape, AliasFact, CallInference, CallRequest, DynamicReason, MutationFact,
    NumericClass, NumericDomain, NumericFact, StorageFact, ValueFact, ValueKindFact,
};

pub(in crate::catalog::inference) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(2..=3).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-IDIVIDE-ARITY",
            "idivide requires two operands and accepts one optional rounding mode",
            request.arguments.len().min(2),
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    }
    let left = &request.arguments[0];
    let right = &request.arguments[1];
    let shape = match broadcast_shape(&left.shape, &right.shape) {
        Ok(shape) => shape,
        Err(diagnostic) => {
            diagnostics.push(diagnostic);
            return finish_fixed(
                entry,
                request,
                ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
                diagnostics,
            );
        }
    };
    let Some(class) = output_class(left, right) else {
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    let mut output = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        shape.clone(),
        if shape.element_count() == Some(1) {
            StorageFact::Scalar
        } else {
            StorageFact::Dense
        },
    );
    output.alias = AliasFact::Unique;
    output.mutation = MutationFact::ValueSemantics;
    output.residency = if left.residency == right.residency {
        left.residency.clone()
    } else {
        runmat_types::ResidencyFact::Unknown
    };
    finish_fixed(entry, request, output, diagnostics)
}

fn output_class(left: &ValueFact, right: &ValueFact) -> Option<NumericClass> {
    let left_numeric = left.numeric()?;
    let right_numeric = right.numeric()?;
    if left_numeric.domain != NumericDomain::Real || right_numeric.domain != NumericDomain::Real {
        return None;
    }
    if left_numeric.class == right_numeric.class && left_numeric.class.integer_class().is_some() {
        return Some(left_numeric.class);
    }
    if left_numeric.class == NumericClass::Double
        && left.is_scalar()
        && right_numeric
            .class
            .integer_class()
            .is_some_and(|class| class.bit_width() < 64)
    {
        return Some(right_numeric.class);
    }
    if right_numeric.class == NumericClass::Double
        && right.is_scalar()
        && left_numeric
            .class
            .integer_class()
            .is_some_and(|class| class.bit_width() < 64)
    {
        return Some(left_numeric.class);
    }
    None
}
