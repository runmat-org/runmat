//! Presentation-only thermal diagnostics and their pre-solve serialization projection.

use std::fmt::{self, Display};

use runmat_canonical_codec::{
    profile_serialized, CanonicalCodecError, CanonicalLimits, SerializedCanonicalProfile,
};
use runmat_execution::Digest;
use serde::Serialize;

use super::{DiagnosticSerialization, FeaDiagnostic, FeaDiagnosticSeverity};

/// Numeric positions are shared by actual reporting and the bounded projection below.
pub(crate) enum ThermalDiagnosticMessage<'a> {
    Initial {
        step: &'a str,
        temperatures: [f64; 3],
        counts: [usize; 5],
    },
    Stability([f64; 4]),
    Constitutive {
        digest: Digest,
        values: [f64; 5],
    },
    Energy([f64; 9]),
    Recovery([usize; 3]),
}

impl ThermalDiagnosticMessage<'_> {
    fn code(&self) -> &'static str {
        match self {
            Self::Initial { .. } => "FEA_THERMAL_INITIAL_STATE",
            Self::Stability(_) => "FEA_THERMAL_STABILITY",
            Self::Constitutive { .. } => "FEA_THERMAL_CONSTITUTIVE",
            Self::Energy(_) => "FEA_THERMAL_ENERGY_BALANCE",
            Self::Recovery(_) => "FEA_THERMAL_FIELD_RECOVERY",
        }
    }

    pub(crate) fn diagnostic(&self, severity: FeaDiagnosticSeverity) -> FeaDiagnostic {
        FeaDiagnostic {
            code: self.code().into(),
            severity,
            message: self.to_string(),
        }
    }
}

impl Display for ThermalDiagnosticMessage<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Initial { step, temperatures: [reference, minimum, maximum], counts: [applied, authored, prescribed, both, reference_count] } => write!(f,
                "procedure_step_id={step} reference_temperature_k={reference:.16e} applied_initial_condition_count={applied} authored_node_count={authored} prescribed_node_count={prescribed} authored_and_prescribed_node_count={both} reference_node_count={reference_count} minimum_temperature_k={minimum:.16e} maximum_temperature_k={maximum:.16e}"),
            Self::Stability([residual, target, minimum, maximum]) => write!(f,
                "all_linear_solves_converged=true max_relative_residual_norm={residual:.16e} residual_target={target:.16e} minimum_temperature_k={minimum:.16e} maximum_temperature_k={maximum:.16e}"),
            Self::Constitutive { digest, values: [conductivity, capacity, density, conductivity_spread, capacity_spread] } => write!(f,
                "property_semantic_digest={digest} conductivity_mean={conductivity:.16e} heat_capacity_mean={capacity:.16e} density_mean={density:.16e} conductivity_spread_ratio={conductivity_spread:.16e} heat_capacity_spread_ratio={capacity_spread:.16e}"),
            Self::Energy([storage, boundary, applied, prescribed, entering, leaving, residual, relative, maximum]) => write!(f,
                "storage_rate_w={storage:.16e} boundary_loss_w={boundary:.16e} applied_power_w={applied:.16e} prescribed_boundary_power_w={prescribed:.16e} prescribed_heat_entering_w={entering:.16e} prescribed_heat_leaving_w={leaving:.16e} residual_w={residual:.16e} final_relative_residual={relative:.16e} max_relative_residual={maximum:.16e}"),
            Self::Recovery([nodes, elements, faces]) => write!(f,
                "basis=physical_quadrature node_count={nodes} element_count={elements} boundary_face_count={faces}"),
        }
    }
}

impl Serialize for ThermalDiagnosticMessage<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

const PROFILE_LIMITS: CanonicalLimits =
    CanonicalLimits::new(usize::MAX, usize::MAX, usize::MAX, 32);

/// Profiles exactly the five producer-owned messages, not material-assignment or solver messages.
/// No messages, value trees, or dummy numerical arrays are allocated by this traversal.
pub(crate) fn thermal_diagnostic_profile(
    procedure_step_id: &str,
) -> Result<SerializedCanonicalProfile, CanonicalCodecError> {
    profile_serialized(&projection(procedure_step_id), PROFILE_LIMITS)
}

pub(crate) fn thermal_diagnostic_record_count() -> usize {
    projection("").len()
}

fn projection(step: &str) -> [DiagnosticSerialization<'_, ThermalDiagnosticMessage<'_>>; 5] {
    // IEEE binary64 finite decimal exponents range from -324 through 308. Scientific .16e
    // has 17 significant digits (enough to round-trip), one decimal point, an optional value
    // sign, 'e', an optional exponent sign, and at most three exponent digits: 24 bytes.
    // This finite negative value attains all those positions, including the exponent sign.
    const WIDEST_FLOAT: f64 = -f64::MIN_POSITIVE;
    // Digest Display is exactly "sha256:" plus 64 lowercase hexadecimal bytes (71 total).
    let digest = Digest::sha256([]);
    let messages = [
        ThermalDiagnosticMessage::Initial {
            step,
            temperatures: [WIDEST_FLOAT; 3],
            counts: [usize::MAX; 5],
        },
        ThermalDiagnosticMessage::Stability([WIDEST_FLOAT; 4]),
        ThermalDiagnosticMessage::Constitutive {
            digest,
            values: [WIDEST_FLOAT; 5],
        },
        ThermalDiagnosticMessage::Energy([WIDEST_FLOAT; 9]),
        ThermalDiagnosticMessage::Recovery([usize::MAX; 3]),
    ];
    messages.map(|message| DiagnosticSerialization {
        code: message.code(),
        severity: match message {
            ThermalDiagnosticMessage::Stability(_) => FeaDiagnosticSeverity::Warning,
            ThermalDiagnosticMessage::Energy(_) => FeaDiagnosticSeverity::Error,
            _ => FeaDiagnosticSeverity::Info,
        },
        message,
    })
}

#[cfg(test)]
mod tests;
