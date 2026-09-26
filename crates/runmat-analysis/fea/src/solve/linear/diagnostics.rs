//! Shared successful PCG diagnostic templates and their bounded serialization profile.

use std::fmt::{self, Display};

use runmat_canonical_codec::{
    profile_serialized, CanonicalCodecError, CanonicalLimits, SerializedCanonicalProfile,
};
use serde::Serialize;

use crate::{
    diagnostics::{DiagnosticSerialization, FeaDiagnostic, FeaDiagnosticSeverity},
    solve::preconditioner::SpdPreconditionerKind,
};

#[derive(Clone, Copy)]
pub(super) enum SuccessfulDiagnostic {
    CpuMethod(SpdPreconditionerKind),
    AcceleratorMethod(SpdPreconditionerKind),
    AdmissionFallback,
}

impl SuccessfulDiagnostic {
    fn wire(self) -> DiagnosticSerialization<'static, Self> {
        DiagnosticSerialization {
            code: match self {
                Self::CpuMethod(_) | Self::AcceleratorMethod(_) => "FEA_SOLVER_METHOD",
                Self::AdmissionFallback => "FEA_ACCELERATOR_ADMISSION_FALLBACK",
            },
            severity: match self {
                Self::AdmissionFallback => FeaDiagnosticSeverity::Warning,
                _ => FeaDiagnosticSeverity::Info,
            },
            message: self,
        }
    }

    pub(super) fn diagnostic(self) -> FeaDiagnostic {
        let wire = self.wire();
        FeaDiagnostic {
            code: wire.code.into(),
            severity: wire.severity,
            message: self.to_string(),
        }
    }
}

impl Display for SuccessfulDiagnostic {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CpuMethod(preconditioner) => write!(formatter,
                "solver=pcg preconditioner={} matrix_free=true", preconditioner.as_str()),
            Self::AcceleratorMethod(preconditioner) => write!(formatter,
                "solver=pcg preconditioner={} matrix_free=false backend=accelerator_native_sparse", preconditioner.as_str()),
            Self::AdmissionFallback => formatter.write_str(
                "native sparse execution was rejected before provider allocation; using the cpu_reference solver"),
        }
    }
}

impl Serialize for SuccessfulDiagnostic {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

const PROFILE_LIMITS: CanonicalLimits =
    CanonicalLimits::new(usize::MAX, usize::MAX, usize::MAX, 32);

/// Bounds successful shared PCG diagnostics for Jacobi and ILU0, irrespective of placement.
/// Nonconvergence and provider errors are deliberately excluded: callers must first require
/// convergence. Profiles the fallback-plus-CPU and accelerator alternatives through the actual
/// message and diagnostic serializers, without allocating owned messages or diagnostic vectors.
pub(crate) fn successful_diagnostics_profile(
) -> Result<SerializedCanonicalProfile, CanonicalCodecError> {
    let mut bound = SerializedCanonicalProfile {
        encoded_bytes_upper_bound: 0,
        materialization_resident_bytes_estimate: 0,
    };
    for preconditioner in [SpdPreconditionerKind::Jacobi, SpdPreconditionerKind::Ilu0] {
        let cpu = profile_serialized(
            &[
                SuccessfulDiagnostic::AdmissionFallback.wire(),
                SuccessfulDiagnostic::CpuMethod(preconditioner).wire(),
            ],
            PROFILE_LIMITS,
        )?;
        let accelerator = profile_serialized(
            &[SuccessfulDiagnostic::AcceleratorMethod(preconditioner).wire()],
            PROFILE_LIMITS,
        )?;
        bound = SerializedCanonicalProfile {
            encoded_bytes_upper_bound: bound.encoded_bytes_upper_bound.max(
                cpu.encoded_bytes_upper_bound
                    .max(accelerator.encoded_bytes_upper_bound),
            ),
            materialization_resident_bytes_estimate: bound
                .materialization_resident_bytes_estimate
                .max(
                    cpu.materialization_resident_bytes_estimate
                        .max(accelerator.materialization_resident_bytes_estimate),
                ),
        };
    }
    Ok(bound)
}

#[cfg(test)]
mod tests;
