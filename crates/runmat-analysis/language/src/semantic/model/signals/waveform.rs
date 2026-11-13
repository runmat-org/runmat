use super::{optional_frame, policy, signal_value, SignalWrapper};
use crate::{
    CheckedExpression, CheckedFeaDeclaration, CheckedFeaStatement, FeaDiagnostic, FeaType,
    PhysicalDimension,
};
use runmat_analysis_core::{
    PhysicalSignal, Signal, SignalApproximationEvidence, SignalId, SignalPrecision,
    SignalProvenance, SignalRepresentation, SignalShape, SignalValue, StandardWaveform,
    StandardWaveformKind,
};

use super::super::{diagnostic, properties::*};

pub(super) fn lower(
    declaration: &CheckedFeaDeclaration,
    id: &str,
    domain: &str,
    body: &[CheckedFeaStatement],
    quantity_kind: runmat_types::physical::QuantityKind,
    wrap: SignalWrapper,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<PhysicalSignal> {
    if domain != "time" {
        diagnostic(
            declaration,
            declaration.span,
            "standard waveforms require the time domain",
            diagnostics,
        );
        return None;
    }
    validate_assignments(
        declaration,
        body,
        &[
            "waveform",
            "amplitude",
            "offset",
            "start",
            "duration",
            "frequency",
            "phase",
            "duty_cycle",
            "frame",
        ],
        diagnostics,
    );
    let kind_expression = required_assignment(declaration, body, "waveform", diagnostics)?;
    let kind = kind(declaration, kind_expression, diagnostics)?;
    let amplitude_expression = required_assignment(declaration, body, "amplitude", diagnostics)?;
    let amplitude = signal_value(
        declaration,
        amplitude_expression,
        if quantity_kind == runmat_types::physical::QuantityKind::Temperature {
            runmat_types::physical::QuantityKind::TemperatureDifference
        } else {
            quantity_kind
        },
        diagnostics,
    )?;
    let offset = match assignment(body, "offset") {
        Some(value) => Some(signal_value(
            declaration,
            value,
            quantity_kind,
            diagnostics,
        )?),
        None => None,
    };
    let vector = matches!(amplitude, SignalValue::Vector3(_));
    let frame = optional_frame(declaration, body, vector, diagnostics)?;
    let start_seconds = required_quantity(
        declaration,
        body,
        "start",
        PhysicalDimension::TIME,
        diagnostics,
    )?;
    let duration_seconds = optional_quantity(
        declaration,
        body,
        "duration",
        PhysicalDimension::TIME,
        diagnostics,
    )?;
    let frequency_hz = optional_quantity(
        declaration,
        body,
        "frequency",
        PhysicalDimension::FREQUENCY,
        diagnostics,
    )?;
    let phase_radians = optional_phase(declaration, body, diagnostics)?;
    let duty_cycle = optional_dimensionless(declaration, body, "duty_cycle", diagnostics)?;
    validate_kind_parameters(
        declaration,
        kind,
        duration_seconds,
        frequency_hz,
        phase_radians,
        duty_cycle,
        diagnostics,
    )?;
    Some(wrap(Signal {
        signal_id: SignalId(id.to_string()),
        shape: if vector {
            SignalShape::Vector3
        } else {
            SignalShape::Scalar
        },
        frame,
        representation: SignalRepresentation::Waveform(StandardWaveform {
            kind,
            amplitude,
            offset,
            start_seconds: runmat_types::physical::Time::new(start_seconds),
            duration_seconds: (duration_seconds).map(runmat_types::physical::Time::new),
            frequency_hz: (frequency_hz).map(runmat_types::physical::Frequency::new),
            phase_radians: runmat_types::physical::Angle::new(phase_radians.unwrap_or(0.0)),
            duty_cycle,
        }),
        precision: SignalPrecision::Real64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: Some(0.0),
            max_relative_error: Some(0.0),
            derivative_max_absolute_error: Some(0.0),
            sample_count: 0,
        },
        provenance: SignalProvenance::Authored,
    }))
}

fn kind(
    declaration: &CheckedFeaDeclaration,
    expression: &CheckedExpression,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<StandardWaveformKind> {
    match policy::enum_name(expression) {
        Some("step") => Some(StandardWaveformKind::Step),
        Some("ramp") => Some(StandardWaveformKind::Ramp),
        Some("sine") => Some(StandardWaveformKind::Sine),
        Some("half_sine_pulse") => Some(StandardWaveformKind::HalfSinePulse),
        Some("triangular_pulse") => Some(StandardWaveformKind::TriangularPulse),
        Some("rectangular_pulse") => Some(StandardWaveformKind::RectangularPulse),
        _ => {
            diagnostic(
                declaration,
                expression.span,
                "unsupported standard waveform",
                diagnostics,
            );
            None
        }
    }
}

fn validate_kind_parameters(
    declaration: &CheckedFeaDeclaration,
    kind: StandardWaveformKind,
    duration: Option<f64>,
    frequency: Option<f64>,
    phase: Option<f64>,
    duty_cycle: Option<f64>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<()> {
    let valid = match kind {
        StandardWaveformKind::Step => {
            duration.is_none() && frequency.is_none() && phase.is_none() && duty_cycle.is_none()
        }
        StandardWaveformKind::Ramp
        | StandardWaveformKind::HalfSinePulse
        | StandardWaveformKind::TriangularPulse => {
            duration.is_some() && frequency.is_none() && phase.is_none() && duty_cycle.is_none()
        }
        StandardWaveformKind::Sine => {
            frequency.is_some() && phase.is_some() && duty_cycle.is_none()
        }
        StandardWaveformKind::RectangularPulse => {
            duration.is_some() && frequency.is_none() && phase.is_none() && duty_cycle.is_some()
        }
    };
    if valid {
        return Some(());
    }
    diagnostic(
        declaration,
        declaration.span,
        format!(
            "waveform `{}` has missing or inapplicable parameters",
            waveform_name(kind)
        ),
        diagnostics,
    );
    None
}

fn waveform_name(kind: StandardWaveformKind) -> &'static str {
    match kind {
        StandardWaveformKind::Step => "step",
        StandardWaveformKind::Ramp => "ramp",
        StandardWaveformKind::Sine => "sine",
        StandardWaveformKind::HalfSinePulse => "half_sine_pulse",
        StandardWaveformKind::TriangularPulse => "triangular_pulse",
        StandardWaveformKind::RectangularPulse => "rectangular_pulse",
    }
}

fn optional_quantity(
    declaration: &CheckedFeaDeclaration,
    body: &[CheckedFeaStatement],
    name: &str,
    dimension: PhysicalDimension,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<Option<f64>> {
    let Some(value) = assignment(body, name) else {
        return Some(None);
    };
    if !(value.value_type).is_quantity((dimension).into()) {
        diagnostic(
            declaration,
            value.span,
            format!("waveform `{name}` has the wrong physical dimension"),
            diagnostics,
        );
        return None;
    }
    scalar(value).map(Some).or_else(|| {
        diagnostic(
            declaration,
            value.span,
            format!("waveform `{name}` must be a constant scalar"),
            diagnostics,
        );
        None
    })
}

fn optional_dimensionless(
    declaration: &CheckedFeaDeclaration,
    body: &[CheckedFeaStatement],
    name: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<Option<f64>> {
    let Some(value) = assignment(body, name) else {
        return Some(None);
    };
    if !matches!(value.value_type, FeaType::Integer | FeaType::Real)
        && !(value.value_type).is_quantity((PhysicalDimension::DIMENSIONLESS).into())
    {
        diagnostic(
            declaration,
            value.span,
            format!("waveform `{name}` must be dimensionless"),
            diagnostics,
        );
        return None;
    }
    scalar(value).map(Some)
}

fn assignment<'a>(body: &'a [CheckedFeaStatement], name: &str) -> Option<&'a CheckedExpression> {
    body.iter().find_map(|statement| match &statement.kind {
        crate::CheckedFeaStatementKind::Assignment { target, value, .. }
            if target.as_slice() == [name] =>
        {
            Some(value)
        }
        _ => None,
    })
}

/// Phase retains its nominal angle unit; a bare zero remains a harmless legacy
/// zero-angle spelling, while nonzero numbers require rad/deg explicitly.
fn optional_phase(
    declaration: &CheckedFeaDeclaration,
    body: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<Option<f64>> {
    let Some(value) = assignment(body, "phase") else {
        return Some(None);
    };
    let angle = runmat_types::physical::DynamicDimension::quantity_type(
        runmat_types::physical::QuantityKind::Angle,
    );
    let number = scalar(value)?;
    if value.value_type.is_quantity(angle)
        || (matches!(value.value_type, FeaType::Integer | FeaType::Real) && number == 0.0)
    {
        Some(Some(number))
    } else {
        diagnostic(
            declaration,
            value.span,
            "waveform phase requires an angle in rad or deg",
            diagnostics,
        );
        None
    }
}
