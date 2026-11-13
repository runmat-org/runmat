use super::super::properties::{required_assignment, scalar};
use super::values::*;
use crate::{
    CheckedExpressionKind, CheckedFeaDeclaration, CheckedFeaStatement, FeaDiagnostic,
    PhysicalDimension,
};
use runmat_analysis_core::{
    ComplexSpectralDensity, CoordinateFrame, CrossPowerSpectralDensity, NumericalDataSource,
    RandomExcitationChannel, RandomExcitationChannelId, RandomExcitationId,
    RandomExcitationQuantity, RandomExcitationSpectrum, RandomSpectrumEvidence,
    RandomSpectrumExtrapolation, RandomSpectrumInterpolation, SpectralDensityConvention,
    SpectrumDirection,
};

const MEMBERS: &[&str] = &[
    "frequencies",
    "channels",
    "spectral_density_real",
    "spectral_density_imaginary",
    "convention",
    "interpolation",
    "extrapolation",
    "max_absolute_density_error",
    "equivalent_degrees_of_freedom",
];

pub(super) fn lower(
    bound: &crate::BoundFeaProgram,
    declaration: &CheckedFeaDeclaration,
    name: &str,
    statements: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<RandomExcitationSpectrum> {
    validate_members(declaration, statements, MEMBERS, diagnostics);
    let frequencies_hz = quantity_vector(
        declaration,
        required_assignment(declaration, statements, "frequencies", diagnostics)?,
        "frequencies",
        PhysicalDimension::FREQUENCY,
        diagnostics,
    )?;
    let channels = channels(
        declaration,
        required_assignment(declaration, statements, "channels", diagnostics)?,
        diagnostics,
    )?;
    let density_dimension = PhysicalDimension::ACCELERATION
        .power(2)?
        .divide(PhysicalDimension::FREQUENCY)?;
    let has_generator = statements.iter().any(|statement| {
        matches!(
            statement.kind,
            crate::CheckedFeaStatementKind::GeneratedByFunction { .. }
        )
    });
    let spectral_density = match (
        assignment(statements, "spectral_density_real"),
        has_generator,
    ) {
        (Some(real), false) => {
            let real = quantity_tensor3(
                declaration,
                real,
                "spectral_density_real",
                density_dimension,
                diagnostics,
            )?;
            let imaginary = match assignment(statements, "spectral_density_imaginary") {
                Some(value) => quantity_tensor3(
                    declaration,
                    value,
                    "spectral_density_imaginary",
                    density_dimension,
                    diagnostics,
                )?,
                None => real
                    .iter()
                    .map(|sample| sample.iter().map(|row| vec![0.0; row.len()]).collect())
                    .collect(),
            };
            NumericalDataSource::authored(
                matrices(declaration, &real, &imaginary, diagnostics)?,
                super::authored_digest(bound, declaration, "random-excitation", name)?,
            )
        }
        (None, true) if assignment(statements, "spectral_density_imaginary").is_none() => {
            NumericalDataSource::Generator(
                super::super::signals::generated::lower_numerical_invocation(
                    declaration,
                    bound,
                    statements,
                    "random spectrum",
                    diagnostics,
                )?,
            )
        }
        _ => {
            super::super::diagnostic(
                declaration,
                declaration.span,
                "random excitation requires exactly one authored spectral-density tensor pair or generator",
                diagnostics,
            );
            return None;
        }
    };
    let convention = match optional_enum(declaration, statements, "convention", diagnostics)? {
        None | Some("one_sided_per_hertz") => SpectralDensityConvention::OneSidedPerHertz,
        Some(_) => return unsupported(declaration, "convention", diagnostics),
    };
    let interpolation = match optional_enum(declaration, statements, "interpolation", diagnostics)?
    {
        None | Some("linear_frequency") => RandomSpectrumInterpolation::LinearFrequency,
        Some(_) => return unsupported(declaration, "interpolation", diagnostics),
    };
    let extrapolation = match optional_enum(declaration, statements, "extrapolation", diagnostics)?
    {
        None | Some("reject") => RandomSpectrumExtrapolation::Reject,
        Some("hold_endpoint") => RandomSpectrumExtrapolation::HoldEndpoint,
        Some(_) => return unsupported(declaration, "extrapolation", diagnostics),
    };
    let excitation = RandomExcitationSpectrum {
        excitation_id: RandomExcitationId(name.to_string()),
        frequencies_hz: runmat_types::physical::Quantity::new(frequencies_hz),
        channels,
        spectral_density,
        convention,
        interpolation,
        extrapolation,
        evidence: RandomSpectrumEvidence {
            max_absolute_density_error: (optional_density(
                declaration,
                statements,
                "max_absolute_density_error",
                density_dimension,
                diagnostics,
            )?)
            .map(runmat_types::physical::AccelerationPowerSpectralDensity::new),
            equivalent_degrees_of_freedom: optional_scalar(
                declaration,
                statements,
                "equivalent_degrees_of_freedom",
                diagnostics,
            )?,
        },
    };
    excitation.validate_intrinsic().map_or_else(
        |error| {
            super::super::diagnostic(declaration, declaration.span, error, diagnostics);
            None
        },
        |()| Some(excitation),
    )
}

fn channels(
    declaration: &CheckedFeaDeclaration,
    expression: &crate::CheckedExpression,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<Vec<RandomExcitationChannel>> {
    let CheckedExpressionKind::List(values) = &expression.kind else {
        return unsupported(declaration, "channels", diagnostics);
    };
    values
        .iter()
        .map(|value| {
            let record = strict_record(
                declaration,
                value,
                "channel",
                &["id", "quantity", "frame", "direction"],
                diagnostics,
            )?;
            let frame = match record.get("frame") {
                None => CoordinateFrame::Global,
                Some(value)
                    if enum_expression(declaration, value, "frame", diagnostics)? == "global" =>
                {
                    CoordinateFrame::Global
                }
                Some(_) => return unsupported(declaration, "channel frame", diagnostics),
            };
            let quantity = enum_expression(
                declaration,
                required_record_member(declaration, record, "channel", "quantity", diagnostics)?,
                "quantity",
                diagnostics,
            )?;
            if quantity != "base_acceleration" {
                return unsupported(declaration, "channel quantity", diagnostics);
            }
            Some(RandomExcitationChannel {
                channel_id: RandomExcitationChannelId(string(
                    declaration,
                    required_record_member(declaration, record, "channel", "id", diagnostics)?,
                    "id",
                    diagnostics,
                )?),
                quantity: RandomExcitationQuantity::BaseAcceleration,
                direction: SpectrumDirection {
                    frame,
                    vector: vector(
                        declaration,
                        required_record_member(
                            declaration,
                            record,
                            "channel",
                            "direction",
                            diagnostics,
                        )?,
                        "direction",
                        diagnostics,
                    )?,
                },
            })
        })
        .collect()
}

fn matrices(
    declaration: &CheckedFeaDeclaration,
    real: &[Vec<Vec<f64>>],
    imaginary: &[Vec<Vec<f64>>],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<Vec<CrossPowerSpectralDensity>> {
    if real.len() != imaginary.len() {
        return unsupported(declaration, "spectral density matrix shape", diagnostics);
    }
    real.iter()
        .zip(imaginary)
        .map(|(real, imaginary)| {
            if real.len() != imaginary.len()
                || real.iter().map(Vec::len).collect::<Vec<_>>()
                    != imaginary.iter().map(Vec::len).collect::<Vec<_>>()
            {
                return unsupported(declaration, "spectral density matrix shape", diagnostics);
            }
            Some(CrossPowerSpectralDensity {
                values: runmat_types::physical::Quantity::new(
                    real.iter()
                        .zip(imaginary)
                        .flat_map(|(real, imaginary)| real.iter().zip(imaginary))
                        .map(|(real, imaginary)| ComplexSpectralDensity::new(*real, *imaginary))
                        .collect(),
                ),
            })
        })
        .collect()
}

fn optional_density(
    declaration: &CheckedFeaDeclaration,
    statements: &[CheckedFeaStatement],
    name: &str,
    dimension: PhysicalDimension,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<Option<f64>> {
    match assignment(statements, name) {
        None => Some(None),
        Some(value) if (value.value_type).is_quantity((dimension).into()) => {
            Some(Some(scalar(value)?))
        }
        Some(_) => unsupported(declaration, name, diagnostics),
    }
}

fn unsupported<T>(
    declaration: &CheckedFeaDeclaration,
    member: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<T> {
    super::super::diagnostic(
        declaration,
        declaration.span,
        format!("unsupported random-excitation `{member}` value"),
        diagnostics,
    );
    None
}
