use super::{diagnostic, properties::*};
use crate::{
    CheckedExpression, CheckedExpressionKind, CheckedFeaDeclaration, CheckedFeaStatement,
    CheckedFeaStatementKind, FeaDiagnostic, FeaType, PhysicalDimension,
};
use runmat_analysis_core::{
    CanonicalQuantity, CoordinateFrame, PhysicalSignal, Signal, SignalApproximationEvidence,
    SignalId, SignalPrecision, SignalProvenance, SignalRepresentation, SignalShape, SignalValue,
};

pub(super) fn lower(
    declaration: &CheckedFeaDeclaration,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Vec<PhysicalSignal> {
    declaration
        .statements
        .iter()
        .filter_map(|statement| {
            let CheckedFeaStatementKind::SignalDefinition {
                name,
                quantity,
                domain,
                statements,
            } = &statement.kind
            else {
                return None;
            };
            constant_signal(declaration, name, quantity, domain, statements, diagnostics)
        })
        .collect()
}

fn constant_signal(
    declaration: &CheckedFeaDeclaration,
    id: &str,
    quantity: &str,
    domain: &str,
    body: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<PhysicalSignal> {
    if domain != "constant" {
        diagnostic(
            declaration,
            declaration.span,
            format!(
                "signal domain `{domain}` requires an explicit sampled or generated representation"
            ),
            diagnostics,
        );
        return None;
    }
    validate_assignments(declaration, body, &["value", "frame"], diagnostics);
    let expression = required_assignment(declaration, body, "value", diagnostics)?;
    let (dimension, wrap): (_, fn(Signal<f64>) -> PhysicalSignal) = match quantity {
        "displacement" => (PhysicalDimension::LENGTH, |value| {
            convert(value, PhysicalSignal::Displacement)
        }),
        "rotation" => (PhysicalDimension::DIMENSIONLESS, |value| {
            convert(value, PhysicalSignal::Rotation)
        }),
        "velocity" => (PhysicalDimension::VELOCITY, |value| {
            convert(value, PhysicalSignal::Velocity)
        }),
        "acceleration" => (PhysicalDimension::ACCELERATION, |value| {
            convert(value, PhysicalSignal::Acceleration)
        }),
        "angular_velocity" => (PhysicalDimension::FREQUENCY, |value| {
            convert(value, PhysicalSignal::AngularVelocity)
        }),
        "angular_acceleration" => (PhysicalDimension::TIME.power(-2)?, |value| {
            convert(value, PhysicalSignal::AngularAcceleration)
        }),
        "force" => (PhysicalDimension::FORCE, |value| {
            convert(value, PhysicalSignal::Force)
        }),
        "moment" => (PhysicalDimension::MOMENT, |value| {
            convert(value, PhysicalSignal::Moment)
        }),
        "traction" => (PhysicalDimension::PRESSURE, |value| {
            convert(value, PhysicalSignal::Traction)
        }),
        "pressure" => (PhysicalDimension::PRESSURE, |value| {
            convert(value, PhysicalSignal::Pressure)
        }),
        "body_force_density" => (
            PhysicalDimension::FORCE.divide(PhysicalDimension::VOLUME)?,
            |value| convert(value, PhysicalSignal::BodyForceDensity),
        ),
        "temperature" => (PhysicalDimension::TEMPERATURE, |value| {
            convert(value, PhysicalSignal::Temperature)
        }),
        "temperature_difference" => (PhysicalDimension::TEMPERATURE, |value| {
            convert(value, PhysicalSignal::TemperatureDifference)
        }),
        "heat_flux" => (PhysicalDimension::HEAT_FLUX, |value| {
            convert(value, PhysicalSignal::HeatFlux)
        }),
        "heat_generation_density" => (
            PhysicalDimension::POWER.divide(PhysicalDimension::VOLUME)?,
            |value| convert(value, PhysicalSignal::HeatGenerationDensity),
        ),
        "electric_potential" => (PhysicalDimension::ELECTRIC_POTENTIAL, |value| {
            convert(value, PhysicalSignal::ElectricPotential)
        }),
        "electric_current" => (PhysicalDimension::ELECTRIC_CURRENT, |value| {
            convert(value, PhysicalSignal::ElectricCurrent)
        }),
        "electric_charge" => (
            PhysicalDimension::ELECTRIC_CURRENT.multiply(PhysicalDimension::TIME)?,
            |value| convert(value, PhysicalSignal::ElectricCharge),
        ),
        "electric_current_density" => (
            PhysicalDimension::ELECTRIC_CURRENT.divide(PhysicalDimension::AREA)?,
            |value| convert(value, PhysicalSignal::ElectricCurrentDensity),
        ),
        "electric_charge_density" => (
            PhysicalDimension::ELECTRIC_CURRENT
                .multiply(PhysicalDimension::TIME)?
                .divide(PhysicalDimension::VOLUME)?,
            |value| convert(value, PhysicalSignal::ElectricChargeDensity),
        ),
        "magnetic_vector_potential" => (
            PhysicalDimension::MAGNETIC_FLUX_DENSITY.multiply(PhysicalDimension::LENGTH)?,
            |value| convert(value, PhysicalSignal::MagneticVectorPotential),
        ),
        "mass_flow_rate" => (
            PhysicalDimension::MASS.divide(PhysicalDimension::TIME)?,
            |value| convert(value, PhysicalSignal::MassFlowRate),
        ),
        _ => {
            diagnostic(
                declaration,
                declaration.span,
                format!("unsupported physical signal quantity `{quantity}`"),
                diagnostics,
            );
            return None;
        }
    };
    let raw = signal_value(declaration, expression, dimension, diagnostics)?;
    let frame = optional_frame(declaration, body, raw.is_vector(), diagnostics)?;
    Some(wrap(Signal {
        signal_id: SignalId(id.into()),
        shape: if raw.is_vector() {
            SignalShape::Vector3
        } else {
            SignalShape::Scalar
        },
        frame,
        representation: SignalRepresentation::Constant(raw),
        precision: SignalPrecision::Real64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: Some(0.0),
            max_relative_error: Some(0.0),
            derivative_max_absolute_error: None,
            sample_count: 1,
        },
        provenance: SignalProvenance::Authored,
    }))
}

trait RawSignalValue {
    fn is_vector(&self) -> bool;
}

impl RawSignalValue for SignalValue<f64> {
    fn is_vector(&self) -> bool {
        matches!(self, Self::Vector3(_))
    }
}

fn signal_value(
    declaration: &CheckedFeaDeclaration,
    value: &CheckedExpression,
    dimension: PhysicalDimension,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<SignalValue<f64>> {
    if value.value_type == FeaType::Quantity(dimension) {
        if let Some(value) = scalar(value) {
            return Some(SignalValue::Scalar(value));
        }
    }
    let values = vector_components(value)?;
    if values.len() != 3 {
        invalid_signal_value(declaration, value, diagnostics);
        return None;
    }
    let mut result = [0.0; 3];
    for (index, value) in values.iter().enumerate() {
        if value.value_type != FeaType::Quantity(dimension) {
            invalid_signal_value(declaration, value, diagnostics);
            return None;
        }
        result[index] = scalar(value)?;
    }
    Some(SignalValue::Vector3(result))
}

fn vector_components(value: &CheckedExpression) -> Option<&[CheckedExpression]> {
    match &value.kind {
        CheckedExpressionKind::List(values) => Some(values),
        _ => None,
    }
}

fn optional_frame(
    declaration: &CheckedFeaDeclaration,
    body: &[CheckedFeaStatement],
    required: bool,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<Option<CoordinateFrame>> {
    let expression = body.iter().find_map(|statement| {
        let CheckedFeaStatementKind::Assignment { target, value, .. } = &statement.kind else {
            return None;
        };
        (target.as_slice() == ["frame"]).then_some(value)
    });
    match expression {
        None if required => {
            diagnostic(
                declaration,
                declaration.span,
                "vector signal requires an explicit coordinate frame",
                diagnostics,
            );
            None
        }
        None => Some(None),
        Some(value) if !required => {
            diagnostic(
                declaration,
                value.span,
                "scalar signal cannot declare a coordinate frame",
                diagnostics,
            );
            None
        }
        Some(value) => frame(declaration, value, diagnostics).map(Some),
    }
}

fn frame(
    declaration: &CheckedFeaDeclaration,
    value: &CheckedExpression,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<CoordinateFrame> {
    let CheckedExpressionKind::Enum(name) = &value.kind else {
        invalid_signal_value(declaration, value, diagnostics);
        return None;
    };
    match name.as_str() {
        "global" => Some(CoordinateFrame::Global),
        "material" => Some(CoordinateFrame::Material),
        "element" => Some(CoordinateFrame::Element),
        "boundary_normal_tangent" => Some(CoordinateFrame::BoundaryNormalTangent),
        _ => {
            diagnostic(
                declaration,
                value.span,
                format!("unsupported signal coordinate frame `{name}`"),
                diagnostics,
            );
            None
        }
    }
}

fn invalid_signal_value(
    declaration: &CheckedFeaDeclaration,
    value: &CheckedExpression,
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    diagnostic(
        declaration,
        value.span,
        "signal value must be a constant scalar or three-component vector of its declared quantity",
        diagnostics,
    );
}

fn convert<Q: CanonicalQuantity>(
    signal: Signal<f64>,
    wrap: impl FnOnce(Signal<Q>) -> PhysicalSignal,
) -> PhysicalSignal {
    wrap(Signal {
        signal_id: signal.signal_id,
        shape: signal.shape,
        frame: signal.frame,
        representation: match signal.representation {
            SignalRepresentation::Constant(SignalValue::Scalar(value)) => {
                SignalRepresentation::Constant(SignalValue::Scalar(Q::from_canonical_value(value)))
            }
            SignalRepresentation::Constant(SignalValue::Vector3(values)) => {
                SignalRepresentation::Constant(SignalValue::Vector3(
                    values.map(Q::from_canonical_value),
                ))
            }
            _ => unreachable!("constant signal lowering creates only scalar or vector values"),
        },
        precision: signal.precision,
        evidence: signal.evidence,
        provenance: signal.provenance,
    })
}
