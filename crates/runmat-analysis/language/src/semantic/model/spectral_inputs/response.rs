use super::super::properties::{required_assignment, required_quantity, required_scalar};
use super::values::*;
use crate::{CheckedFeaDeclaration, CheckedFeaStatement, FeaDiagnostic, PhysicalDimension};
use runmat_analysis_core::{
    CoordinateFrame, DirectionalCombination, ModalCombination, ModalTruncationCorrection,
    ModalTruncationPolicy, NumericalDataSource, ResponseSpectrum, ResponseSpectrumId,
    RigidResponsePolicy, ShockResponseSpectrumMethod, SignalId, SpectrumAbscissa,
    SpectrumConstruction, SpectrumDamping, SpectrumDirection, SpectrumEnvelope, SpectrumEvidence,
    SpectrumExtrapolation, SpectrumInterpolation, SpectrumResponse,
};

const MEMBERS: &[&str] = &[
    "frequencies",
    "periods",
    "ordinates",
    "damping_ratio",
    "quality_factor",
    "response",
    "envelope",
    "interpolation",
    "extrapolation",
    "frame",
    "direction",
    "modal_combination",
    "directional_combination",
    "primary_percentage",
    "secondary_percentage",
    "minimum_cumulative_mass_ratio",
    "truncation_correction",
    "zero_period_acceleration",
    "rigid_response",
    "periodic_frequency",
    "rigid_frequency",
    "construction",
    "source_signal",
    "shock_method",
    "sampling_adequacy_ratio",
    "max_interpolation_error",
];

pub(super) fn lower(
    bound: &crate::BoundFeaProgram,
    declaration: &CheckedFeaDeclaration,
    name: &str,
    statements: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<ResponseSpectrum> {
    validate_members(declaration, statements, MEMBERS, diagnostics);
    let frequency = assignment(statements, "frequencies");
    let period = assignment(statements, "periods");
    let (abscissa, default_interpolation) = match (frequency, period) {
        (Some(value), None) => (
            SpectrumAbscissa::FrequencyHertz(runmat_types::physical::Quantity::new(
                quantity_vector(
                    declaration,
                    value,
                    "frequencies",
                    PhysicalDimension::FREQUENCY,
                    diagnostics,
                )?,
            )),
            SpectrumInterpolation::LinearFrequency,
        ),
        (None, Some(value)) => (
            SpectrumAbscissa::PeriodSeconds(runmat_types::physical::Quantity::new(
                quantity_vector(
                    declaration,
                    value,
                    "periods",
                    PhysicalDimension::TIME,
                    diagnostics,
                )?,
            )),
            SpectrumInterpolation::LinearPeriod,
        ),
        _ => {
            super::super::diagnostic(
                declaration,
                declaration.span,
                "response spectrum requires exactly one of `frequencies` or `periods`",
                diagnostics,
            );
            return None;
        }
    };
    let response = match enum_assignment(declaration, statements, "response", diagnostics)? {
        "relative_displacement" => SpectrumResponse::RelativeDisplacement,
        "relative_velocity" => SpectrumResponse::RelativeVelocity,
        "relative_acceleration" => SpectrumResponse::RelativeAcceleration,
        "absolute_acceleration" => SpectrumResponse::AbsoluteAcceleration,
        "pseudo_velocity" => SpectrumResponse::PseudoVelocity,
        "pseudo_acceleration" => SpectrumResponse::PseudoAcceleration,
        _ => return unsupported(declaration, "response", diagnostics),
    };
    let has_generator = statements.iter().any(|statement| {
        matches!(
            statement.kind,
            crate::CheckedFeaStatementKind::GeneratedByFunction { .. }
        )
    });
    let ordinates = match (assignment(statements, "ordinates"), has_generator) {
        (Some(value), false) => NumericalDataSource::authored(
            quantity_vector(
                declaration,
                value,
                "ordinates",
                response_dimension(response),
                diagnostics,
            )?,
            super::authored_digest(bound, declaration, "response-spectrum", name)?,
        ),
        (None, true) => NumericalDataSource::Generator(
            super::super::signals::generated::lower_numerical_invocation(
                declaration,
                bound,
                statements,
                "response spectrum",
                diagnostics,
            )?,
        ),
        _ => {
            super::super::diagnostic(
                declaration,
                declaration.span,
                "response spectrum requires exactly one authored `ordinates` array or generator",
                diagnostics,
            );
            return None;
        }
    };
    let damping = match (
        optional_scalar(declaration, statements, "damping_ratio", diagnostics)?,
        optional_scalar(declaration, statements, "quality_factor", diagnostics)?,
    ) {
        (Some(value), None) => SpectrumDamping::Ratio(value),
        (None, Some(value)) => SpectrumDamping::QualityFactor(value),
        _ => {
            super::super::diagnostic(
                declaration,
                declaration.span,
                "response spectrum requires exactly one of `damping_ratio` or `quality_factor`",
                diagnostics,
            );
            return None;
        }
    };
    let frame = match optional_enum(declaration, statements, "frame", diagnostics)? {
        None | Some("global") => CoordinateFrame::Global,
        Some(_) => return unsupported(declaration, "frame", diagnostics),
    };
    let direction = vector(
        declaration,
        required_assignment(declaration, statements, "direction", diagnostics)?,
        "direction",
        diagnostics,
    )?;
    let modal_combination =
        match optional_enum(declaration, statements, "modal_combination", diagnostics)? {
            None | Some("cqc") => ModalCombination::CompleteQuadraticCombination,
            Some("srss") => ModalCombination::Srss,
            Some("ten_percent") => ModalCombination::TenPercent,
            Some("absolute_sum") => ModalCombination::AbsoluteSum,
            Some(_) => return unsupported(declaration, "modal_combination", diagnostics),
        };
    let directional_combination = directional_combination(declaration, statements, diagnostics)?;
    let truncation = ModalTruncationPolicy {
        minimum_cumulative_mass_ratio: optional_scalar(
            declaration,
            statements,
            "minimum_cumulative_mass_ratio",
            diagnostics,
        )?
        .unwrap_or(0.9),
        correction: truncation_correction(declaration, statements, response, diagnostics)?,
    };
    let rigid_response = rigid_response(declaration, statements, response, diagnostics)?;
    let construction = construction(declaration, statements, diagnostics)?;
    let interpolation = match optional_enum(declaration, statements, "interpolation", diagnostics)?
    {
        None => default_interpolation,
        Some("linear_frequency") => SpectrumInterpolation::LinearFrequency,
        Some("linear_period") => SpectrumInterpolation::LinearPeriod,
        Some("log_log") => SpectrumInterpolation::LogLog,
        Some(_) => return unsupported(declaration, "interpolation", diagnostics),
    };
    let spectrum = ResponseSpectrum {
        spectrum_id: ResponseSpectrumId(name.to_string()),
        abscissa,
        ordinates,
        damping,
        response,
        envelope: match optional_enum(declaration, statements, "envelope", diagnostics)? {
            None | Some("maximax") => SpectrumEnvelope::Maximax,
            Some("positive") => SpectrumEnvelope::Positive,
            Some("negative") => SpectrumEnvelope::Negative,
            Some(_) => return unsupported(declaration, "envelope", diagnostics),
        },
        interpolation,
        extrapolation: match optional_enum(declaration, statements, "extrapolation", diagnostics)? {
            None | Some("reject") => SpectrumExtrapolation::Reject,
            Some("hold_endpoint") => SpectrumExtrapolation::HoldEndpoint,
            Some(_) => return unsupported(declaration, "extrapolation", diagnostics),
        },
        direction: SpectrumDirection {
            frame,
            vector: direction,
        },
        modal_combination,
        directional_combination,
        truncation,
        rigid_response,
        construction,
        evidence: SpectrumEvidence {
            sampling_adequacy_ratio: optional_scalar(
                declaration,
                statements,
                "sampling_adequacy_ratio",
                diagnostics,
            )?
            .unwrap_or(1.0),
            max_interpolation_error: optional_scalar(
                declaration,
                statements,
                "max_interpolation_error",
                diagnostics,
            )?,
        },
    };
    spectrum.validate_intrinsic().map_or_else(
        |error| {
            super::super::diagnostic(declaration, declaration.span, error, diagnostics);
            None
        },
        |()| Some(spectrum),
    )
}

fn response_dimension(value: SpectrumResponse) -> PhysicalDimension {
    match value {
        SpectrumResponse::RelativeDisplacement => PhysicalDimension::LENGTH,
        SpectrumResponse::RelativeVelocity | SpectrumResponse::PseudoVelocity => {
            PhysicalDimension::VELOCITY
        }
        SpectrumResponse::RelativeAcceleration
        | SpectrumResponse::AbsoluteAcceleration
        | SpectrumResponse::PseudoAcceleration => PhysicalDimension::ACCELERATION,
    }
}

fn directional_combination(
    declaration: &CheckedFeaDeclaration,
    statements: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<DirectionalCombination> {
    match optional_enum(
        declaration,
        statements,
        "directional_combination",
        diagnostics,
    )? {
        None | Some("srss") => Some(DirectionalCombination::Srss),
        Some("absolute_sum") => Some(DirectionalCombination::AbsoluteSum),
        Some("percentage") => Some(DirectionalCombination::PercentageRule {
            primary: required_scalar(declaration, statements, "primary_percentage", diagnostics)?,
            secondary: required_scalar(
                declaration,
                statements,
                "secondary_percentage",
                diagnostics,
            )?,
        }),
        Some(_) => unsupported(declaration, "directional_combination", diagnostics),
    }
}

fn truncation_correction(
    declaration: &CheckedFeaDeclaration,
    statements: &[CheckedFeaStatement],
    response: SpectrumResponse,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<ModalTruncationCorrection> {
    match optional_enum(
        declaration,
        statements,
        "truncation_correction",
        diagnostics,
    )? {
        None | Some("none") => Some(ModalTruncationCorrection::None),
        Some("residual_mode") => Some(ModalTruncationCorrection::ResidualMode),
        Some("missing_mass") => Some(ModalTruncationCorrection::MissingMass {
            zero_period_acceleration: required_quantity(
                declaration,
                statements,
                "zero_period_acceleration",
                response_dimension(response),
                diagnostics,
            )?,
        }),
        Some(_) => unsupported(declaration, "truncation_correction", diagnostics),
    }
}

fn rigid_response(
    declaration: &CheckedFeaDeclaration,
    statements: &[CheckedFeaStatement],
    response: SpectrumResponse,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<RigidResponsePolicy> {
    match optional_enum(declaration, statements, "rigid_response", diagnostics)? {
        None | Some("disabled") => Some(RigidResponsePolicy::Disabled),
        Some("gupta") => Some(RigidResponsePolicy::Gupta {
            periodic_frequency_hz: required_quantity(
                declaration,
                statements,
                "periodic_frequency",
                PhysicalDimension::FREQUENCY,
                diagnostics,
            )?,
            rigid_frequency_hz: required_quantity(
                declaration,
                statements,
                "rigid_frequency",
                PhysicalDimension::FREQUENCY,
                diagnostics,
            )?,
        }),
        Some("lindley_yow") => Some(RigidResponsePolicy::LindleyYow {
            zero_period_acceleration: required_quantity(
                declaration,
                statements,
                "zero_period_acceleration",
                response_dimension(response),
                diagnostics,
            )?,
        }),
        Some(_) => unsupported(declaration, "rigid_response", diagnostics),
    }
}

fn construction(
    declaration: &CheckedFeaDeclaration,
    statements: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<SpectrumConstruction> {
    match optional_enum(declaration, statements, "construction", diagnostics)? {
        None | Some("authored_design") => Some(SpectrumConstruction::AuthoredDesign),
        Some("derived_from_acceleration_history") => {
            let source =
                required_assignment(declaration, statements, "source_signal", diagnostics)?;
            let crate::CheckedExpressionKind::Reference(path) = &source.kind else {
                return unsupported(declaration, "source_signal", diagnostics);
            };
            let [namespace, name] = path.as_slice() else {
                return unsupported(declaration, "source_signal", diagnostics);
            };
            if namespace != "signal" {
                return unsupported(declaration, "source_signal", diagnostics);
            }
            let method =
                match enum_assignment(declaration, statements, "shock_method", diagnostics)? {
                    "exact_piecewise_linear" => ShockResponseSpectrumMethod::ExactPiecewiseLinear,
                    "recursive_filter" => ShockResponseSpectrumMethod::RecursiveFilter,
                    _ => return unsupported(declaration, "shock_method", diagnostics),
                };
            Some(SpectrumConstruction::DerivedFromAccelerationHistory {
                source_signal_id: SignalId(name.clone()),
                method,
            })
        }
        Some(_) => unsupported(declaration, "construction", diagnostics),
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
        format!("unsupported response-spectrum `{member}` value"),
        diagnostics,
    );
    None
}
