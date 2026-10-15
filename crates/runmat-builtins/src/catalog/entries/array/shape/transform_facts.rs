//! Shared facts for structural transforms, not executable dispatch or numerical admission.
use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::*;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ArrayShapeTransform {
    Reshape,
    Tile,
    Permute,
    InversePermute,
    Squeeze,
}

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    operation: ArrayShapeTransform,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let valid_arity = match operation {
        ArrayShapeTransform::Squeeze => request.arguments.len() == 1,
        ArrayShapeTransform::Permute | ArrayShapeTransform::InversePermute => {
            request.arguments.len() == 2
        }
        ArrayShapeTransform::Reshape | ArrayShapeTransform::Tile => request.arguments.len() >= 2,
    };
    if !valid_arity {
        diagnostics.push(argument_error(
            "RM-CATALOG-SHAPE-ARITY",
            "invalid structural transform argument count",
            0,
        ));
    }
    let Some(source) = request.arguments.first() else {
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    if !shape_transform_input_supported(source, operation) {
        diagnostics.push(argument_error(
            "RM-CATALOG-SHAPE-INPUT",
            "unsupported structural transform input",
            0,
        ));
    }
    let mut output = source.clone();
    let shape = if operation == ArrayShapeTransform::Squeeze {
        squeeze_shape(&source.shape)
    } else {
        controls_shape(request, source, operation, &mut diagnostics)
    };
    output.shape = shape;
    // Shape transforms can materialize a lazy device layout. Preserve only
    // the canonical host layout; do not copy physical promises across routes.
    output.layout =
        if source.residency == ResidencyFact::Host && source.layout == LayoutFact::ColumnMajor {
            LayoutFact::ColumnMajor
        } else {
            LayoutFact::Unknown
        };
    output.contiguity = if source.contiguity == ContiguityFact::Contiguous
        && (source.residency == ResidencyFact::Host
            || matches!(
                operation,
                ArrayShapeTransform::Reshape | ArrayShapeTransform::Squeeze
            )) {
        ContiguityFact::Contiguous
    } else {
        ContiguityFact::Unknown
    };
    if matches!(
        operation,
        ArrayShapeTransform::Tile
            | ArrayShapeTransform::Permute
            | ArrayShapeTransform::InversePermute
    ) && matches!(
        source.residency,
        ResidencyFact::Device { .. } | ResidencyFact::Unknown
    ) {
        // Automatic resident inputs may legitimately restore to host when
        // upload is unsupported. Facts do not encode explicit-vs-auto intent.
        output.residency = ResidencyFact::Unknown;
        output.storage = StorageFact::Unknown;
    }
    // Abstract aggregate contents survive structural transformation. Positional
    // facts only survive operations that preserve the linear element sequence.
    if matches!(
        operation,
        ArrayShapeTransform::Tile
            | ArrayShapeTransform::Permute
            | ArrayShapeTransform::InversePermute
    ) {
        match &mut output.kind {
            ValueKindFact::Cell(cell) => {
                cell.elements.clear();
                cell.elements_complete = false;
            }
            ValueKindFact::Struct(structure) => {
                structure.elements.clear();
                structure.elements_complete = false;
            }
            _ => {}
        }
    }
    if operation == ArrayShapeTransform::Squeeze {
        // A contiguous device view can alias its input; a lazy layout can materialize.
        output.alias = AliasFact::Unknown;
        output.view = ViewFact::Unknown;
    } else {
        output.alias = AliasFact::Unknown;
        output.view = ViewFact::Unknown;
        if operation == ArrayShapeTransform::Tile && source.residency == ResidencyFact::Host {
            output.storage = StorageFact::Dense;
        }
        if matches!(
            operation,
            ArrayShapeTransform::Permute | ArrayShapeTransform::InversePermute
        ) && source.kind == ValueKindFact::Logical
            && source.storage == StorageFact::Scalar
        {
            // The existing scalar Bool branch deliberately passes through numeric Tensor.
            output.kind = ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            });
        }
        if matches!(
            operation,
            ArrayShapeTransform::Tile
                | ArrayShapeTransform::Permute
                | ArrayShapeTransform::InversePermute
        ) && output.residency == ResidencyFact::Host
        {
            if let Some(numeric) = output.numeric() {
                let scalarizes = numeric.domain == NumericDomain::Real
                    && (numeric.class == NumericClass::Double
                        || numeric.class.integer_class().is_some());
                output.storage = match (scalarizes, output.shape.element_count()) {
                    (true, Some(1)) => StorageFact::Scalar,
                    (true, None) => StorageFact::Unknown,
                    _ => StorageFact::Dense,
                };
                if output.storage == StorageFact::Scalar {
                    output.shape = ShapeFact::Scalar;
                }
            }
        }
        if operation == ArrayShapeTransform::Reshape && source.storage == StorageFact::Scalar {
            output.storage = match output.shape.known_dims() {
                Some(dims) if dims.len() <= 2 && dims.iter().all(|dim| *dim == Some(1)) => {
                    StorageFact::Scalar
                }
                Some(dims) if dims.len() > 2 => StorageFact::Dense,
                _ => StorageFact::Unknown,
            };
            if matches!(source.kind, ValueKindFact::Struct(_)) {
                output.shape = source.shape.clone();
                output.storage = source.storage;
            }
        }
    }
    if matches!(source.kind, ValueKindFact::Struct(_)) && source.storage == StorageFact::Scalar {
        if operation == ArrayShapeTransform::Tile {
            output.storage = match output.shape.element_count() {
                Some(1) => StorageFact::Scalar,
                None => StorageFact::Unknown,
                _ => StorageFact::Dense,
            };
        } else {
            output.storage = StorageFact::Scalar;
        }
        if output.storage == StorageFact::Scalar {
            output.shape = ShapeFact::Scalar;
        }
    }
    if operation == ArrayShapeTransform::Tile
        && source.residency == ResidencyFact::Host
        && source.numeric().is_some_and(|numeric| {
            numeric.domain == NumericDomain::Complex && numeric.class == NumericClass::Double
        })
        && source.storage != StorageFact::Dense
    {
        output.storage = match (source.storage, output.shape.element_count()) {
            (StorageFact::Scalar, Some(1)) => StorageFact::Scalar,
            (_, None) | (StorageFact::Unknown, Some(1)) => StorageFact::Unknown,
            _ => StorageFact::Dense,
        };
        if output.storage == StorageFact::Scalar {
            output.shape = ShapeFact::Scalar;
        }
    }
    if !diagnostics.is_empty() {
        // Preserve operand class/aggregate information; a rejected call has no proven
        // result geometry. Runtime validation still owns payload/error precedence.
        output.shape = ShapeFact::Unknown;
        output.certainty = CertaintyFact::Dynamic(DynamicReason::UnsupportedRepresentation);
    }
    finish_fixed(entry, request, output, diagnostics)
}

pub fn shape_transform_input_supported(source: &ValueFact, operation: ArrayShapeTransform) -> bool {
    if source.storage == StorageFact::Sparse {
        return false;
    }
    match &source.kind {
        ValueKindFact::Unknown
        | ValueKindFact::Numeric(_)
        | ValueKindFact::Logical
        | ValueKindFact::Character
        | ValueKindFact::String
        | ValueKindFact::Struct(_) => {
            !(matches!(
                operation,
                ArrayShapeTransform::Permute | ArrayShapeTransform::InversePermute
            ) && source.storage == StorageFact::Scalar
                && (source.kind == ValueKindFact::String
                    || source
                        .numeric()
                        .is_some_and(|n| n.domain == NumericDomain::Complex)))
        }
        ValueKindFact::Cell(_) => !matches!(
            operation,
            ArrayShapeTransform::Permute | ArrayShapeTransform::InversePermute
        ),
        // Object facts do not distinguish a scalar ObjectArray carrier from an
        // individual object. Successful reshape calls retain the array's class;
        // the runtime adapter rejects unsupported individual-object carriers.
        ValueKindFact::Object(_) => operation == ArrayShapeTransform::Reshape,
        _ => false,
    }
}

mod controls;
use controls::controls_shape;
pub use controls::shape_transform_control_supported;
#[cfg(test)]
mod tests;
