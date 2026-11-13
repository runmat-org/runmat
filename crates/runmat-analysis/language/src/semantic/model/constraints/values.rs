use super::super::{diagnostic, properties::*};
use crate::{
    CheckedExpression, CheckedExpressionKind, CheckedFeaDeclaration, CheckedFeaStatement,
    CheckedFeaStatementKind, FeaDiagnostic, FeaType, PhysicalDimension,
};
use runmat_analysis_core::{
    CoordinateFrame, PhysicalQuantityKind, PhysicalSignal, PrescribedQuantity, PrescribedValue,
    SignalId, StructuralTranslationDof,
};

pub(super) fn optional_assignment<'a>(
    body: &'a [CheckedFeaStatement],
    name: &str,
) -> Option<&'a CheckedExpression> {
    body.iter().find_map(|statement| {
        let CheckedFeaStatementKind::Assignment { target, value, .. } = &statement.kind else {
            return None;
        };
        (target.as_slice() == [name]).then_some(value)
    })
}

pub(super) fn optional_quantity(
    declaration: &CheckedFeaDeclaration,
    body: &[CheckedFeaStatement],
    name: &str,
    dimension: impl Into<runmat_types::physical::DynamicDimension>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<Option<f64>> {
    match optional_assignment(body, name) {
        Some(value) => quantity(declaration, value, name, dimension, diagnostics).map(Some),
        None => Some(None),
    }
}

pub(super) fn quantity(
    declaration: &CheckedFeaDeclaration,
    value: &CheckedExpression,
    name: &str,
    dimension: impl Into<runmat_types::physical::DynamicDimension>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<f64> {
    let dimension = dimension.into();
    let type_matches = if dimension == PhysicalDimension::DIMENSIONLESS.into() {
        matches!(value.value_type, FeaType::Integer | FeaType::Real)
            || (value.value_type).is_quantity(dimension)
    } else {
        (value.value_type).is_quantity(dimension)
    };
    if !type_matches {
        invalid(
            declaration,
            value,
            name,
            "quantity of the required dimension",
            diagnostics,
        );
        return None;
    }
    scalar(value).or_else(|| {
        invalid(declaration, value, name, "constant quantity", diagnostics);
        None
    })
}

pub(super) fn prescribed_value<Q: PrescribedQuantity>(
    declaration: &CheckedFeaDeclaration,
    body: &[CheckedFeaStatement],
    name: &str,
    expected: (PhysicalDimension, PhysicalQuantityKind),
    signals: &[PhysicalSignal],
    constant: impl FnOnce(f64) -> Q,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<PrescribedValue<Q>> {
    prescribed_source(
        declaration,
        body,
        name,
        expected.1,
        signals,
        |value, diagnostics| {
            if value.validate_quantity_kind(expected.1).is_err() {
                invalid(
                    declaration,
                    value,
                    name,
                    "the required absolute or difference quantity",
                    diagnostics,
                );
                return None;
            }
            // The physical kind retains nominal angle powers that SI dimensions
            // alone cannot distinguish from ordinary dimensionless values.
            quantity(
                declaration,
                value,
                name,
                runmat_types::physical::DynamicDimension::quantity_type(expected.1),
                diagnostics,
            )
            .map(constant)
        },
        diagnostics,
    )
}

/// Inline components use the global frame. A referenced vector signal retains its
/// original frame and representation; no scalar speed or guessed direction is accepted.
pub(super) fn prescribed_vector<Q>(
    declaration: &CheckedFeaDeclaration,
    body: &[CheckedFeaStatement],
    name: &str,
    expected: (PhysicalDimension, PhysicalQuantityKind),
    signals: &[PhysicalSignal],
    constant: impl Fn(f64) -> Q,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<PrescribedValue<[Q; 3]>>
where
    [Q; 3]: PrescribedQuantity,
{
    prescribed_source(
        declaration,
        body,
        name,
        expected.1,
        signals,
        |_, diagnostics| {
            required_vector(declaration, body, name, expected.0, diagnostics)
                .map(|values| values.map(constant))
        },
        diagnostics,
    )
}

fn prescribed_source<Q: PrescribedQuantity>(
    declaration: &CheckedFeaDeclaration,
    body: &[CheckedFeaStatement],
    name: &str,
    quantity_kind: PhysicalQuantityKind,
    signals: &[PhysicalSignal],
    constant: impl FnOnce(&CheckedExpression, &mut Vec<FeaDiagnostic>) -> Option<Q>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<PrescribedValue<Q>> {
    let value = required_assignment(declaration, body, name, diagnostics)?;
    if let CheckedExpressionKind::Reference(path) = &value.kind {
        let [namespace, signal_name] = path.as_slice() else {
            invalid(
                declaration,
                value,
                name,
                "a constant quantity or `signal.<name>`",
                diagnostics,
            );
            return None;
        };
        if namespace != "signal" {
            invalid(
                declaration,
                value,
                name,
                "a constant quantity or `signal.<name>`",
                diagnostics,
            );
            return None;
        }
        let id = SignalId(signal_name.clone());
        if !signals.iter().any(|signal| {
            signal.signal_id() == &id
                && signal.quantity() == quantity_kind
                && signal.shape() == &Q::SHAPE
        }) {
            invalid(
                declaration,
                value,
                name,
                if Q::SHAPE == runmat_analysis_core::SignalShape::Scalar {
                    "a declared scalar signal with the required physical quantity"
                } else {
                    "a declared vector signal with the required physical quantity and shape"
                },
                diagnostics,
            );
            return None;
        }
        return Some(PrescribedValue::Signal(id));
    }
    constant(value, diagnostics).map(PrescribedValue::Constant)
}

pub(super) fn optional_vector(
    declaration: &CheckedFeaDeclaration,
    body: &[CheckedFeaStatement],
    name: &str,
    dimension: PhysicalDimension,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<Option<[f64; 3]>> {
    if optional_assignment(body, name).is_none() {
        return Some(None);
    }
    required_vector(declaration, body, name, dimension, diagnostics).map(Some)
}

pub(super) fn required_string(
    declaration: &CheckedFeaDeclaration,
    body: &[CheckedFeaStatement],
    name: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<String> {
    let value = required_assignment(declaration, body, name, diagnostics)?;
    let CheckedExpressionKind::String(value) = &value.kind else {
        invalid(declaration, value, name, "string", diagnostics);
        return None;
    };
    Some(value.clone())
}

pub(super) fn frame(
    declaration: &CheckedFeaDeclaration,
    value: &CheckedExpression,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<CoordinateFrame> {
    let CheckedExpressionKind::Enum(frame_name) = &value.kind else {
        invalid(
            declaration,
            value,
            "frame",
            "coordinate-frame enum",
            diagnostics,
        );
        return None;
    };
    match frame_name.as_str() {
        "global" => Some(CoordinateFrame::Global),
        "material" => Some(CoordinateFrame::Material),
        "element" => Some(CoordinateFrame::Element),
        "boundary_normal_tangent" => Some(CoordinateFrame::BoundaryNormalTangent),
        _ => {
            invalid(
                declaration,
                value,
                "frame",
                "supported coordinate frame",
                diagnostics,
            );
            None
        }
    }
}

pub(super) fn translation_dof(
    declaration: &CheckedFeaDeclaration,
    value: &CheckedExpression,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<StructuralTranslationDof> {
    let CheckedExpressionKind::Enum(name) = &value.kind else {
        invalid(
            declaration,
            value,
            "dof",
            "translation degree of freedom",
            diagnostics,
        );
        return None;
    };
    match name.as_str() {
        "ux" => Some(StructuralTranslationDof::Ux),
        "uy" => Some(StructuralTranslationDof::Uy),
        "uz" => Some(StructuralTranslationDof::Uz),
        _ => {
            invalid(declaration, value, "dof", "ux, uy, or uz", diagnostics);
            None
        }
    }
}

pub(super) fn dimensionless_scalar(
    declaration: &CheckedFeaDeclaration,
    value: &CheckedExpression,
    name: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<f64> {
    quantity(
        declaration,
        value,
        name,
        PhysicalDimension::DIMENSIONLESS,
        diagnostics,
    )
}

pub(super) fn invalid(
    declaration: &CheckedFeaDeclaration,
    value: &CheckedExpression,
    name: &str,
    expected: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    diagnostic(
        declaration,
        value.span,
        format!("constraint member `{name}` requires {expected}"),
        diagnostics,
    );
}
