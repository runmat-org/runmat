use super::{diagnostic, properties::*};
use crate::{
    CheckedExpression, CheckedExpressionKind, CheckedFeaDeclaration, CheckedFeaStatement,
    CheckedFeaStatementKind, FeaDiagnostic, FeaType,
};
use runmat_analysis_core::{
    CanonicalQuantity, CoordinateFrame, PhysicalSignal, Signal, SignalApproximationEvidence,
    SignalId, SignalPrecision, SignalProvenance, SignalRepresentation, SignalShape, SignalValue,
};

pub(in crate::semantic::model) mod generated;
mod parameterized;
mod policy;
mod tabulated;
mod waveform;

pub(super) type SignalWrapper = fn(Signal<f64>) -> PhysicalSignal;

pub(super) fn lower(
    declaration: &CheckedFeaDeclaration,
    bound: &crate::BoundFeaProgram,
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
            signal(
                declaration,
                bound,
                name,
                quantity,
                domain,
                statements,
                diagnostics,
            )
        })
        .collect()
}

fn signal(
    declaration: &CheckedFeaDeclaration,
    bound: &crate::BoundFeaProgram,
    id: &str,
    quantity: &str,
    domain: &str,
    body: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<PhysicalSignal> {
    let (kind, wrap) = signal_kind(declaration, quantity, diagnostics)?;
    let generated = body.iter().any(|statement| {
        matches!(
            statement.kind,
            CheckedFeaStatementKind::GeneratedByFunction { .. }
        )
    });
    if let Some(samples) = assignment(body, "samples").filter(|samples| {
        matches!(&samples.kind, CheckedExpressionKind::FunctionCall { function, .. } if function.last().is_some_and(|name| name == "file"))
            || matches!(&samples.kind, CheckedExpressionKind::OperationCall { operation, .. } if operation.as_str() == "file")
            || matches!(samples.kind, CheckedExpressionKind::Missing)
    }) {
        diagnostic(
            declaration,
            samples.span,
            crate::semantic::pending::with_pending(
                "signal samples are an inline table of (abscissa, value) pairs; samples read from a file are not accepted yet",
                Some(crate::semantic::pending::SIGNAL_TABULATED_FILE),
            ),
            diagnostics,
        );
        return None;
    }
    let tabulated = assignment(body, "samples").is_some() || assignment(body, "jumps").is_some();
    let waveform = assignment(body, "waveform").is_some();
    let constant = assignment(body, "value").is_some();
    if usize::from(generated)
        + usize::from(tabulated)
        + usize::from(waveform)
        + usize::from(constant)
        != 1
    {
        diagnostic(
            declaration,
            declaration.span,
            "signal requires exactly one constant, tabulated, waveform, or generated representation",
            diagnostics,
        );
        return None;
    }
    if generated {
        return generated::lower(
            generated::LowerSignal {
                declaration,
                bound,
                quantity_kind: kind,
                wrap,
                diagnostics,
            },
            id,
            domain,
            body,
        );
    }
    if tabulated {
        return tabulated::lower(declaration, id, domain, body, kind, wrap, diagnostics);
    }
    if waveform {
        return waveform::lower(declaration, id, domain, body, kind, wrap, diagnostics);
    }
    constant_signal(
        declaration,
        id,
        quantity,
        domain,
        body,
        kind,
        wrap,
        diagnostics,
    )
}

#[allow(clippy::too_many_arguments)]
fn constant_signal(
    declaration: &CheckedFeaDeclaration,
    id: &str,
    quantity: &str,
    domain: &str,
    body: &[CheckedFeaStatement],
    kind: runmat_types::physical::QuantityKind,
    wrap: SignalWrapper,
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
    authored_unit_matches(
        declaration,
        quantity,
        expression,
        kind.dimension(),
        diagnostics,
    )?;
    let raw = signal_value(declaration, expression, kind, diagnostics)?;
    let frame = optional_frame(declaration, body, raw.is_vector(), diagnostics)?;
    Some(wrap(signal_contract(
        id,
        raw,
        frame,
        SignalRepresentation::Constant,
    )))
}

fn assignment<'a>(body: &'a [CheckedFeaStatement], name: &str) -> Option<&'a CheckedExpression> {
    body.iter().find_map(|statement| match &statement.kind {
        CheckedFeaStatementKind::Assignment { target, value, .. }
            if target.as_slice() == [name] =>
        {
            Some(value)
        }
        _ => None,
    })
}

fn signal_kind(
    declaration: &CheckedFeaDeclaration,
    quantity: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<(runmat_types::physical::QuantityKind, SignalWrapper)> {
    use runmat_types::physical::QuantityKind as K;
    Some(match quantity {
        "displacement" => (K::Displacement, |value| {
            convert(value, PhysicalSignal::Displacement)
        }),
        "rotation" => (K::Rotation, |value| {
            convert(value, PhysicalSignal::Rotation)
        }),
        "velocity" => (K::Velocity, |value| {
            convert(value, PhysicalSignal::Velocity)
        }),
        "acceleration" => (K::Acceleration, |value| {
            convert(value, PhysicalSignal::Acceleration)
        }),
        "angular_velocity" => (K::AngularVelocity, |value| {
            convert(value, PhysicalSignal::AngularVelocity)
        }),
        "angular_acceleration" => (K::AngularAcceleration, |value| {
            convert(value, PhysicalSignal::AngularAcceleration)
        }),
        "force" => (K::Force, |value| convert(value, PhysicalSignal::Force)),
        "moment" => (K::Moment, |value| convert(value, PhysicalSignal::Moment)),
        "traction" => (K::Traction, |value| {
            convert(value, PhysicalSignal::Traction)
        }),
        "pressure" => (K::Pressure, |value| {
            convert(value, PhysicalSignal::Pressure)
        }),
        "body_force_density" => (K::BodyForceDensity, |value| {
            convert(value, PhysicalSignal::BodyForceDensity)
        }),
        "temperature" => (K::Temperature, |value| {
            convert(value, PhysicalSignal::Temperature)
        }),
        "temperature_difference" => (K::TemperatureDifference, |value| {
            convert(value, PhysicalSignal::TemperatureDifference)
        }),
        "heat_flux" => (K::HeatFlux, |value| {
            convert(value, PhysicalSignal::HeatFlux)
        }),
        "heat_generation_density" => (K::HeatGenerationDensity, |value| {
            convert(value, PhysicalSignal::HeatGenerationDensity)
        }),
        // A total heat rate (W) of a heat-generation region.
        "heat_rate" => (K::Power, |value| convert(value, PhysicalSignal::HeatRate)),
        "electric_potential" => (K::ElectricPotential, |value| {
            convert(value, PhysicalSignal::ElectricPotential)
        }),
        "electric_current" => (K::ElectricCurrent, |value| {
            convert(value, PhysicalSignal::ElectricCurrent)
        }),
        "electric_charge" => (K::ElectricCharge, |value| {
            convert(value, PhysicalSignal::ElectricCharge)
        }),
        "electric_current_density" => (K::ElectricCurrentDensity, |value| {
            convert(value, PhysicalSignal::ElectricCurrentDensity)
        }),
        "electric_charge_density" => (K::ElectricChargeDensity, |value| {
            convert(value, PhysicalSignal::ElectricChargeDensity)
        }),
        "magnetic_vector_potential" => (K::MagneticVectorPotential, |value| {
            convert(value, PhysicalSignal::MagneticVectorPotential)
        }),
        "volume_flow_rate" => (K::VolumeFlowRate, |value| {
            convert(value, PhysicalSignal::VolumeFlowRate)
        }),
        "mass_flow_rate" => (K::MassFlowRate, |value| {
            convert(value, PhysicalSignal::MassFlowRate)
        }),
        _ => {
            diagnostic(
                declaration,
                declaration.span,
                format!("unsupported physical signal quantity `{quantity}`"),
                diagnostics,
            );
            return None;
        }
    })
}

pub(super) fn constant_from_expression(
    declaration: &CheckedFeaDeclaration,
    id: &str,
    quantity: &str,
    value: &CheckedExpression,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<PhysicalSignal> {
    let (kind, wrap) = signal_kind(declaration, quantity, diagnostics)?;
    authored_unit_matches(declaration, quantity, value, kind.dimension(), diagnostics)?;
    let raw = signal_value(declaration, value, kind, diagnostics)?;
    let frame = raw.is_vector().then_some(CoordinateFrame::Global);
    Some(wrap(signal_contract(
        id,
        raw,
        frame,
        SignalRepresentation::Constant,
    )))
}

pub(super) fn parameterized_from_expression(
    declaration: &CheckedFeaDeclaration,
    id: &str,
    quantity: runmat_analysis_core::PhysicalQuantityKind,
    quantity_name: &str,
    value: &CheckedExpression,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<runmat_analysis_core::AuthoredParameterizedSignal> {
    let (kind, _) = signal_kind(declaration, quantity_name, diagnostics)?;
    if kind != quantity || value.validate_quantity_kind(quantity).is_err() {
        invalid_signal_value(declaration, value, diagnostics);
        return None;
    }
    parameterized::lower(declaration, id, quantity, value, diagnostics)
}

pub(super) fn depends_on_model_input(value: &CheckedExpression) -> bool {
    parameterized::depends_on_model_input(value)
}

/// Lower pure arithmetic over numeric model inputs whose result has the checked value type.
pub(super) fn model_input_expression(
    declaration: &CheckedFeaDeclaration,
    value: &CheckedExpression,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<runmat_analysis_core::AuthoredNumericExpression> {
    let expected = super::super::input::authored_type(&value.value_type)?;
    parameterized::expression(declaration, value, expected, diagnostics)
}

fn signal_contract(
    id: &str,
    shape_source: SignalValue<f64>,
    frame: Option<CoordinateFrame>,
    representation: impl FnOnce(SignalValue<f64>) -> SignalRepresentation<f64>,
) -> Signal<f64> {
    let is_vector = shape_source.is_vector();
    Signal {
        signal_id: SignalId(id.into()),
        shape: if is_vector {
            SignalShape::Vector3
        } else {
            SignalShape::Scalar
        },
        frame,
        representation: representation(shape_source),
        precision: SignalPrecision::Real64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: Some(0.0),
            max_relative_error: Some(0.0),
            derivative_max_absolute_error: None,
            sample_count: 1,
        },
        provenance: SignalProvenance::Authored,
    }
}

trait RawSignalValue {
    fn is_vector(&self) -> bool;
}

impl RawSignalValue for SignalValue<f64> {
    fn is_vector(&self) -> bool {
        matches!(self, Self::Vector3(_))
    }
}

pub(super) fn signal_value(
    declaration: &CheckedFeaDeclaration,
    value: &CheckedExpression,
    kind: runmat_types::physical::QuantityKind,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<SignalValue<f64>> {
    if value.value_type == FeaType::Error {
        // The expression checker already reported why this value has no type.
        return None;
    }
    if value.validate_quantity_kind(kind).is_err() {
        diagnostic(declaration, value.span, "absolute temperatures and temperature differences must use their respective units and arithmetic", diagnostics);
        return None;
    }
    let dimension = runmat_types::physical::DynamicDimension::quantity_type(kind);
    if (value.value_type).is_quantity(dimension) {
        if let Some(value) = scalar(value) {
            return Some(SignalValue::Scalar(value));
        }
    }
    if matches!(
        &value.value_type,
        FeaType::Vector { element, length: Some(3) }
            if element.is_quantity(dimension)
    ) {
        if let CheckedExpressionKind::Quantity {
            value: components,
            unit,
            ..
        } = &value.kind
        {
            if let CheckedExpressionKind::List(values) = &components.kind {
                if values.len() != 3 {
                    invalid_signal_value(declaration, value, diagnostics);
                    return None;
                }
                let scale = unit.scale();
                let offset = unit.offset();
                let mut result = [0.0; 3];
                for (index, component) in values.iter().enumerate() {
                    result[index] = scalar(component)? * scale + offset;
                }
                return Some(SignalValue::Vector3(result));
            }
        }
    }
    let Some(values) = vector_components(value) else {
        invalid_signal_value(declaration, value, diagnostics);
        return None;
    };
    if values.len() != 3 {
        invalid_signal_value(declaration, value, diagnostics);
        return None;
    }
    let mut result = [0.0; 3];
    for (index, value) in values.iter().enumerate() {
        if !(value.value_type).is_quantity(dimension) {
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
        CheckedExpressionKind::Quantity { value, .. } => vector_components(value),
        _ => None,
    }
}

pub(super) fn optional_frame(
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

/// Rejects a quantity whose authored unit measures something other than the signal's quantity,
/// naming what was expected and what was found at the authored value.
fn authored_unit_matches(
    declaration: &CheckedFeaDeclaration,
    quantity: &str,
    value: &CheckedExpression,
    dimension: runmat_types::PhysicalDimension,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<()> {
    let Some((message, label)) = crate::semantic::units::unit_mismatch(quantity, value, dimension)
    else {
        return Some(());
    };
    diagnostics.push(
        crate::FeaDiagnostic::error(
            "fea.semantic.model",
            message,
            crate::FeaDiagnosticPhase::Lowering,
            Some(crate::FeaDiagnosticLocation {
                path: declaration.module.clone(),
                span: value.span,
            }),
        )
        .with_label(label),
    );
    None
}

fn invalid_signal_value(
    declaration: &CheckedFeaDeclaration,
    value: &CheckedExpression,
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    // A value computed from another analysis (a state reference or an operation) is planned.
    let expression = matches!(
        value.kind,
        CheckedExpressionKind::Reference(_)
            | CheckedExpressionKind::OperationCall { .. }
            | CheckedExpressionKind::FunctionCall { .. }
    );
    diagnostic(
        declaration,
        value.span,
        crate::semantic::pending::with_pending(
            "signal value must be a constant scalar or three-component vector of its declared quantity",
            expression.then_some(crate::semantic::pending::SIGNAL_HARMONIC_NONCONSTANT_VALUE),
        ),
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
            SignalRepresentation::Constant(SignalValue::Tensor { shape, values }) => {
                SignalRepresentation::Constant(SignalValue::Tensor {
                    shape,
                    values: values.into_iter().map(Q::from_canonical_value).collect(),
                })
            }
            SignalRepresentation::Generator(generator) => {
                SignalRepresentation::Generator(generator)
            }
            SignalRepresentation::Artifact(artifact) => SignalRepresentation::Artifact(artifact),
            SignalRepresentation::Tabulated(table) => {
                SignalRepresentation::Tabulated(map_table(table))
            }
            SignalRepresentation::Waveform(waveform) => {
                SignalRepresentation::Waveform(runmat_analysis_core::StandardWaveform {
                    kind: waveform.kind,
                    amplitude: map_value(waveform.amplitude),
                    offset: waveform.offset.map(map_value),
                    start_seconds: runmat_types::physical::Time::new(
                        waveform.start_seconds.canonical_value(),
                    ),
                    duration_seconds: (waveform
                        .duration_seconds
                        .map(|quantity| quantity.canonical_value()))
                    .map(runmat_types::physical::Time::new),
                    frequency_hz: (waveform
                        .frequency_hz
                        .map(|quantity| quantity.canonical_value()))
                    .map(runmat_types::physical::Frequency::new),
                    phase_radians: runmat_types::physical::Angle::new(
                        waveform.phase_radians.canonical_value(),
                    ),
                    duty_cycle: waveform.duty_cycle,
                })
            }
        },
        precision: signal.precision,
        evidence: signal.evidence,
        provenance: signal.provenance,
    })
}

fn map_table<Q: CanonicalQuantity>(
    table: runmat_analysis_core::TabulatedSignal<f64>,
) -> runmat_analysis_core::TabulatedSignal<Q> {
    runmat_analysis_core::TabulatedSignal {
        domain: table.domain,
        knots: table
            .knots
            .into_iter()
            .map(|knot| runmat_analysis_core::SignalKnot {
                abscissa: knot.abscissa,
                value: match knot.value {
                    runmat_analysis_core::SignalKnotValue::Continuous(value) => {
                        runmat_analysis_core::SignalKnotValue::Continuous(map_ordinate(value))
                    }
                    runmat_analysis_core::SignalKnotValue::Jump {
                        left_limit,
                        right_limit,
                        continuity,
                    } => runmat_analysis_core::SignalKnotValue::Jump {
                        left_limit: map_ordinate(left_limit),
                        right_limit: map_ordinate(right_limit),
                        continuity,
                    },
                },
            })
            .collect(),
        interpolation: table.interpolation,
        extrapolation: table.extrapolation,
    }
}

fn map_ordinate<Q: CanonicalQuantity>(
    ordinate: runmat_analysis_core::SignalOrdinate<f64>,
) -> runmat_analysis_core::SignalOrdinate<Q> {
    match ordinate {
        runmat_analysis_core::SignalOrdinate::Real(value) => {
            runmat_analysis_core::SignalOrdinate::Real(map_value(value))
        }
        runmat_analysis_core::SignalOrdinate::Complex { real, imaginary } => {
            runmat_analysis_core::SignalOrdinate::Complex {
                real: map_value(real),
                imaginary: map_value(imaginary),
            }
        }
    }
}

fn map_value<Q: CanonicalQuantity>(value: SignalValue<f64>) -> SignalValue<Q> {
    match value {
        SignalValue::Scalar(value) => SignalValue::Scalar(Q::from_canonical_value(value)),
        SignalValue::Vector3(values) => SignalValue::Vector3(values.map(Q::from_canonical_value)),
        SignalValue::Tensor { shape, values } => SignalValue::Tensor {
            shape,
            values: values.into_iter().map(Q::from_canonical_value).collect(),
        },
    }
}
