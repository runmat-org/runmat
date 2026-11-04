//! The one accelerator-admission decision shared by every physics family.
//!
//! Owner decision (2026-10-28): when a study requests accelerator placement and the accelerator
//! path cannot run it, every family runs on the CPU reference path, publishes `cpu_reference` as
//! its execution backend and emits [`ACCELERATOR_ADMISSION_FALLBACK_CODE`]. The authored strict
//! option (`accelerators = required`, [`AcceleratorAdmissionPolicy::Required`]) fails closed with
//! [`AcceleratorRequiredError`] instead.
//!
//! Families decide here twice at most: [`admit_accelerator`] before execution (a family with no
//! device solve path never reaches the device), and [`confirm_accelerator_execution`] after a
//! family whose device path can still decline a system at solve time (for example a provider that
//! cannot hold the operator) reports the backend that actually ran.

use std::cell::Cell;

pub use runmat_analysis_core::AcceleratorAdmissionPolicy;

use super::{ComputeBackend, ExecutionBackend};
use crate::diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity};

/// Warning published when a requested accelerator solve ran on `cpu_reference`.
pub const ACCELERATOR_ADMISSION_FALLBACK_CODE: &str = "FEA_ACCELERATOR_ADMISSION_FALLBACK";
/// Error raised when strict placement cannot run on the accelerator.
pub const ACCELERATOR_REQUIRED_CODE: &str = "FEA_ACCELERATOR_REQUIRED";

/// Physics families that accept a placement request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AcceleratorFamily {
    StructuralStatic,
    StructuralModal,
    StructuralBuckling,
    StructuralFrequency,
    StructuralTransient,
    /// Checkpointed transient block segments: the block solve has no device path.
    StructuralTransientBlock,
    StructuralNonlinear,
    Thermal,
    /// Thermal with radiation boundaries: the damped Newton solve has no device path.
    ThermalRadiation,
    Acoustic,
    Electromagnetic,
    Fluid,
    ConjugateHeatTransfer,
    FluidStructureInteraction,
}

impl AcceleratorFamily {
    pub const fn label(self) -> &'static str {
        match self {
            Self::StructuralStatic => "structural static",
            Self::StructuralModal => "structural modal",
            Self::StructuralBuckling => "structural buckling",
            Self::StructuralFrequency => "structural frequency response",
            Self::StructuralTransient | Self::StructuralTransientBlock => "structural transient",
            Self::StructuralNonlinear => "structural nonlinear",
            Self::Thermal => "thermal",
            Self::ThermalRadiation => "thermal radiation",
            Self::Acoustic => "acoustic",
            Self::Electromagnetic => "electromagnetic",
            Self::Fluid => "fluid",
            Self::ConjugateHeatTransfer => "conjugate heat transfer",
            Self::FluidStructureInteraction => "fluid-structure interaction",
        }
    }

    /// Why the family has no device solve path, or `None` when its linear solver admits the
    /// device at solve time (and reports the backend that ran).
    pub const fn unavailable_reason(self) -> Option<&'static str> {
        match self {
            Self::StructuralTransient | Self::Thermal => None,
            Self::StructuralStatic => {
                Some("static execution has no accelerator-bound native sparse admission")
            }
            Self::StructuralModal | Self::StructuralBuckling | Self::StructuralFrequency => Some(
                "the eigensolver runs on the shared CPU sparse layer (shift-invert block Lanczos)",
            ),
            Self::StructuralNonlinear => {
                Some("the Newton tangent solves run on the shared CPU sparse layer")
            }
            Self::StructuralTransientBlock => {
                Some("checkpointed transient block segments run on the shared CPU sparse layer")
            }
            Self::ThermalRadiation => {
                Some("the damped Newton radiation solve runs on the shared CPU sparse layer")
            }
            Self::Acoustic
            | Self::Electromagnetic
            | Self::Fluid
            | Self::ConjugateHeatTransfer
            | Self::FluidStructureInteraction => {
                Some("the family's solver runs on the shared CPU sparse layer")
            }
        }
    }
}

impl AcceleratorFamily {
    /// The family that executes one authored procedure step.
    pub fn of_step(step: &runmat_analysis_core::ProcedureStep) -> Self {
        use runmat_analysis_core::{
            AnalysisProcedure as Procedure, CoupledPhysicsFamily as Coupled,
            PhysicsFamily as Family, SolutionSpace,
        };
        match step.family {
            Family::Thermal => Self::Thermal,
            Family::Electromagnetic | Family::Coupled(Coupled::Electrothermal) => {
                Self::Electromagnetic
            }
            Family::Acoustic => Self::Acoustic,
            Family::Fluid => Self::Fluid,
            Family::Coupled(Coupled::ConjugateHeatTransfer) => Self::ConjugateHeatTransfer,
            Family::Coupled(Coupled::FluidStructureInteraction) => Self::FluidStructureInteraction,
            Family::Structural | Family::Coupled(Coupled::Thermomechanical) => {
                match step.procedure {
                    Procedure::Static | Procedure::Adaptive => Self::StructuralStatic,
                    Procedure::ModalExtraction | Procedure::PrestressedModalExtraction => {
                        Self::StructuralModal
                    }
                    Procedure::EigenvalueBuckling => Self::StructuralBuckling,
                    Procedure::TransientResponse
                        if step.solution_space == SolutionSpace::ModalReduced =>
                    {
                        Self::StructuralModal
                    }
                    Procedure::TransientResponse => Self::StructuralTransient,
                    Procedure::HarmonicResponse
                    | Procedure::PrestressedHarmonicResponse
                    | Procedure::ModalResponseSpectrum
                    | Procedure::RandomResponse => Self::StructuralFrequency,
                    Procedure::Nonlinear => Self::StructuralNonlinear,
                }
            }
        }
    }
}

/// The first family of a study with no device solve path. Under `accelerators = required` such a
/// study fails before meshing; this is the static half of accelerator admission that
/// `runmat check` reports, while device availability is decided on the executing host.
pub fn first_family_without_device_path(
    families: impl IntoIterator<Item = AcceleratorFamily>,
) -> Option<AcceleratorFamily> {
    families
        .into_iter()
        .find(|family| family.unavailable_reason().is_some())
}

/// Admits one placement for a whole authored sequence, where one backend serves every step.
/// The first step whose family has no device path decides the fallback (or the strict failure);
/// otherwise `device` reports whether an accelerator can be allocated for the study.
pub fn admit_study_accelerator<'a>(
    steps: impl IntoIterator<Item = &'a runmat_analysis_core::ProcedureStep>,
    requested: ComputeBackend,
    policy: AcceleratorAdmissionPolicy,
    device: impl FnOnce() -> Result<(), String>,
) -> Result<AcceleratorAdmission, AcceleratorRequiredError> {
    let families = steps
        .into_iter()
        .map(AcceleratorFamily::of_step)
        .collect::<Vec<_>>();
    let without_device_path = first_family_without_device_path(families.iter().copied());
    if requested == ComputeBackend::Cpu {
        return Ok(AcceleratorAdmission {
            requested,
            effective: requested,
            fallback: None,
        });
    }
    if let Some(family) = without_device_path {
        return admit_accelerator_with_policy(family, requested, policy);
    }
    let fallback = match (device(), families.first()) {
        (Ok(()), _) | (Err(_), None) => None,
        (Err(reason), Some(family)) => Some(fallback_decision(*family, &reason, policy)?),
    };
    Ok(AcceleratorAdmission {
        requested,
        effective: if fallback.is_some() {
            ComputeBackend::Cpu
        } else {
            requested
        },
        fallback,
    })
}

/// The placement a host admitted for a whole study before submission: the authored policy and,
/// when the study was planned on `cpu_reference` although the accelerator was requested, the one
/// fallback warning its published result carries.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcceleratorPlacement {
    #[serde(default)]
    pub policy: AcceleratorAdmissionPolicy,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback: Option<FeaDiagnostic>,
}

impl AcceleratorPlacement {
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    /// Settles a finished study under the admitted placement. `steps` yields, per published
    /// step, its step identifier, requested backend and executed backend. Strict placement fails
    /// closed when any requested accelerator step ran on the CPU (a decline a worker could only
    /// discover at solve time); otherwise the planned fallback warning is returned for the
    /// study publication.
    pub fn settle_study<'a>(
        &self,
        steps: impl IntoIterator<Item = (&'a str, ComputeBackend, ExecutionBackend)>,
    ) -> Result<Option<FeaDiagnostic>, AcceleratorRequiredError> {
        if self.policy == AcceleratorAdmissionPolicy::Required {
            if let Some((step_id, _, _)) = steps.into_iter().find(|(_, requested, executed)| {
                *requested == ComputeBackend::Gpu && *executed != ExecutionBackend::Accelerator
            }) {
                return Err(AcceleratorRequiredError {
                    family: "study",
                    reason: format!("step `{step_id}` ran on cpu_reference"),
                });
            }
        }
        Ok(self.fallback.clone())
    }
}

/// Strict placement could not run on the accelerator.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
    "FEA_ACCELERATOR_REQUIRED: {family} execution requested the accelerator with \
     `accelerators = required`, but {reason}; strict placement does not fall back to cpu_reference"
)]
pub struct AcceleratorRequiredError {
    pub family: &'static str,
    pub reason: String,
}

/// The admitted placement of one family solve.
#[derive(Debug, Clone, PartialEq)]
pub struct AcceleratorAdmission {
    pub requested: ComputeBackend,
    /// The backend the family executes with.
    pub effective: ComputeBackend,
    /// The fallback warning to publish with the result, when the request was not honored.
    pub fallback: Option<FeaDiagnostic>,
}

thread_local! {
    static POLICY: Cell<AcceleratorAdmissionPolicy> =
        const { Cell::new(AcceleratorAdmissionPolicy::FallbackToCpu) };
}

/// Runs `body` with the authored admission policy in scope for every family solve it starts on
/// this thread. Hosts set it where the authored placement is resolved; without a scope the
/// default policy (fallback to `cpu_reference`) applies.
pub fn with_accelerator_admission_policy<R>(
    policy: AcceleratorAdmissionPolicy,
    body: impl FnOnce() -> R,
) -> R {
    struct Restore(AcceleratorAdmissionPolicy);
    impl Drop for Restore {
        fn drop(&mut self) {
            POLICY.with(|cell| cell.set(self.0));
        }
    }
    // A nested authored solve may strengthen its enclosing worker admission,
    // but must not weaken a strict invocation into CPU fallback.
    let _restore = Restore(POLICY.with(|cell| {
        let previous = cell.get();
        cell.set(if previous == AcceleratorAdmissionPolicy::Required {
            previous
        } else {
            policy
        });
        previous
    }));
    body()
}

/// The admission policy in scope on this thread.
pub fn accelerator_admission_policy() -> AcceleratorAdmissionPolicy {
    POLICY.with(Cell::get)
}

/// Admits `requested` for `family` under the policy in scope.
pub fn admit_accelerator(
    family: AcceleratorFamily,
    requested: ComputeBackend,
) -> Result<AcceleratorAdmission, AcceleratorRequiredError> {
    admit_accelerator_with_policy(family, requested, accelerator_admission_policy())
}

/// Admits `requested` for `family` under an explicit policy.
pub fn admit_accelerator_with_policy(
    family: AcceleratorFamily,
    requested: ComputeBackend,
    policy: AcceleratorAdmissionPolicy,
) -> Result<AcceleratorAdmission, AcceleratorRequiredError> {
    let unavailable = match (requested, family.unavailable_reason()) {
        (ComputeBackend::Cpu, _) => {
            return Ok(AcceleratorAdmission {
                requested,
                effective: requested,
                fallback: None,
            })
        }
        (ComputeBackend::Gpu, Some(reason)) => reason,
        // A device-capable family: what admission can already tell (no provider is assigned to
        // this worker) decides before the solve; the solve-time decline is settled afterwards.
        (ComputeBackend::Gpu, None) => match device_unavailable_reason() {
            Some(reason) => reason,
            None => {
                return Ok(AcceleratorAdmission {
                    requested,
                    effective: requested,
                    fallback: None,
                })
            }
        },
    };
    Ok(AcceleratorAdmission {
        requested,
        effective: ComputeBackend::Cpu,
        fallback: Some(fallback_decision(family, unavailable, policy)?),
    })
}

/// Why no device can run a solve on this worker, when that is known before the solve.
pub fn device_unavailable_reason() -> Option<&'static str> {
    runmat_accelerate_api::provider()
        .is_none()
        .then_some("no accelerator provider is assigned to this worker")
}

/// Completes a run admitted by [`admit_accelerator`]: a run executed on `cpu_reference` after a
/// pre-solve fallback reports the requested placement and carries the one fallback warning; a
/// run admitted onto the device settles a solve-time decline.
pub fn complete_accelerator_admission(
    family: AcceleratorFamily,
    admission: AcceleratorAdmission,
    run: &mut super::FeaRunResult,
) -> Result<(), AcceleratorRequiredError> {
    match admission.fallback {
        Some(fallback) => {
            run.requested_backend = admission.requested;
            run.diagnostics
                .retain(|diagnostic| diagnostic.code != ACCELERATOR_ADMISSION_FALLBACK_CODE);
            run.diagnostics.push(fallback);
            Ok(())
        }
        None => settle_accelerator_execution(family, run),
    }
}

/// Confirms the backend a device-capable family actually ran with under the policy in scope.
/// Returns the fallback warning to publish when a requested accelerator solve ran on the CPU.
pub fn confirm_accelerator_execution(
    family: AcceleratorFamily,
    requested: ComputeBackend,
    executed: ExecutionBackend,
) -> Result<Option<FeaDiagnostic>, AcceleratorRequiredError> {
    if requested == ComputeBackend::Cpu || executed == ExecutionBackend::Accelerator {
        return Ok(None);
    }
    fallback_decision(
        family,
        family
            .unavailable_reason()
            .unwrap_or("the accelerator declined the system at solve time"),
        accelerator_admission_policy(),
    )
    .map(Some)
}

/// Settles a finished run of a device-capable family: strict placement fails closed when the
/// device did not run the solve; otherwise the run carries exactly one fallback warning.
pub fn settle_accelerator_execution(
    family: AcceleratorFamily,
    run: &mut super::FeaRunResult,
) -> Result<(), AcceleratorRequiredError> {
    if let Some(fallback) =
        confirm_accelerator_execution(family, run.requested_backend, run.execution_backend)?
    {
        // Linear solves report their own decline once per solve; the run carries one warning.
        run.diagnostics
            .retain(|diagnostic| diagnostic.code != ACCELERATOR_ADMISSION_FALLBACK_CODE);
        run.diagnostics.push(fallback);
    }
    Ok(())
}

fn fallback_decision(
    family: AcceleratorFamily,
    reason: &str,
    policy: AcceleratorAdmissionPolicy,
) -> Result<FeaDiagnostic, AcceleratorRequiredError> {
    match policy {
        AcceleratorAdmissionPolicy::Required => Err(AcceleratorRequiredError {
            family: family.label(),
            reason: reason.to_string(),
        }),
        AcceleratorAdmissionPolicy::FallbackToCpu => Ok(fallback_diagnostic(family, reason)),
    }
}

/// The shared fallback warning.
pub fn fallback_diagnostic(family: AcceleratorFamily, reason: &str) -> FeaDiagnostic {
    FeaDiagnostic {
        code: ACCELERATOR_ADMISSION_FALLBACK_CODE.to_string(),
        severity: FeaDiagnosticSeverity::Warning,
        message: format!(
            "{} accelerator placement was requested but {reason}; executed on cpu_reference",
            family.label()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_requests_are_admitted_unchanged_for_every_policy() {
        for policy in [
            AcceleratorAdmissionPolicy::FallbackToCpu,
            AcceleratorAdmissionPolicy::Required,
        ] {
            let admission = admit_accelerator_with_policy(
                AcceleratorFamily::StructuralStatic,
                ComputeBackend::Cpu,
                policy,
            )
            .unwrap();
            assert_eq!(admission.effective, ComputeBackend::Cpu);
            assert!(admission.fallback.is_none());
        }
    }

    #[test]
    fn families_without_a_device_path_fall_back_by_default_and_fail_closed_when_required() {
        let admission =
            admit_accelerator(AcceleratorFamily::StructuralModal, ComputeBackend::Gpu).unwrap();
        assert_eq!(admission.effective, ComputeBackend::Cpu);
        let fallback = admission.fallback.unwrap();
        assert_eq!(fallback.code, ACCELERATOR_ADMISSION_FALLBACK_CODE);
        assert_eq!(fallback.severity, FeaDiagnosticSeverity::Warning);
        assert!(fallback.message.contains("cpu_reference"));

        let error = with_accelerator_admission_policy(AcceleratorAdmissionPolicy::Required, || {
            admit_accelerator(AcceleratorFamily::StructuralStatic, ComputeBackend::Gpu)
        })
        .unwrap_err();
        assert!(error.to_string().starts_with(ACCELERATOR_REQUIRED_CODE));
        assert!(error.to_string().contains("structural static"));
        with_accelerator_admission_policy(AcceleratorAdmissionPolicy::Required, || {
            with_accelerator_admission_policy(AcceleratorAdmissionPolicy::FallbackToCpu, || {
                assert_eq!(
                    accelerator_admission_policy(),
                    AcceleratorAdmissionPolicy::Required
                );
                assert!(
                    admit_accelerator(AcceleratorFamily::StructuralModal, ComputeBackend::Gpu)
                        .is_err()
                );
                assert!(
                    admit_accelerator(AcceleratorFamily::StructuralModal, ComputeBackend::Cpu)
                        .is_ok()
                );
            });
        });
        // The scope restores the default.
        assert_eq!(
            accelerator_admission_policy(),
            AcceleratorAdmissionPolicy::FallbackToCpu
        );
    }

    #[test]
    fn device_capable_families_keep_the_request_and_confirm_the_backend_that_ran() {
        let admission = admit_accelerator(AcceleratorFamily::Thermal, ComputeBackend::Gpu).unwrap();
        let strict =
            with_accelerator_admission_policy(AcceleratorAdmissionPolicy::Required, || {
                admit_accelerator(AcceleratorFamily::Thermal, ComputeBackend::Gpu)
            });
        if device_unavailable_reason().is_some() {
            // Known before the solve: fall back now, and strict fails before any solve.
            assert_eq!(admission.effective, ComputeBackend::Cpu);
            assert_eq!(
                admission.fallback.unwrap().code,
                ACCELERATOR_ADMISSION_FALLBACK_CODE
            );
            assert!(strict.is_err());
        } else {
            assert_eq!(admission.effective, ComputeBackend::Gpu);
            assert!(admission.fallback.is_none());
            assert!(strict.is_ok());
        }
        assert!(confirm_accelerator_execution(
            AcceleratorFamily::Thermal,
            ComputeBackend::Gpu,
            ExecutionBackend::Accelerator
        )
        .unwrap()
        .is_none());
        let fallback = confirm_accelerator_execution(
            AcceleratorFamily::Thermal,
            ComputeBackend::Gpu,
            ExecutionBackend::Cpu,
        )
        .unwrap()
        .unwrap();
        assert_eq!(fallback.code, ACCELERATOR_ADMISSION_FALLBACK_CODE);
        assert!(
            with_accelerator_admission_policy(AcceleratorAdmissionPolicy::Required, || {
                confirm_accelerator_execution(
                    AcceleratorFamily::Thermal,
                    ComputeBackend::Gpu,
                    ExecutionBackend::Cpu,
                )
            })
            .is_err()
        );
    }

    #[test]
    fn study_placement_publishes_the_planned_fallback_and_strict_fails_closed_on_cpu_steps() {
        let planned = AcceleratorPlacement {
            policy: AcceleratorAdmissionPolicy::FallbackToCpu,
            fallback: Some(fallback_diagnostic(
                AcceleratorFamily::StructuralModal,
                "test",
            )),
        };
        let cpu_step = [("modal", ComputeBackend::Cpu, ExecutionBackend::Cpu)];
        assert_eq!(
            planned.settle_study(cpu_step).unwrap().unwrap().code,
            ACCELERATOR_ADMISSION_FALLBACK_CODE
        );
        let strict = AcceleratorPlacement {
            policy: AcceleratorAdmissionPolicy::Required,
            fallback: None,
        };
        assert!(strict
            .settle_study([("heat", ComputeBackend::Gpu, ExecutionBackend::Accelerator)])
            .unwrap()
            .is_none());
        let error = strict
            .settle_study([("heat", ComputeBackend::Gpu, ExecutionBackend::Cpu)])
            .unwrap_err();
        assert!(error.to_string().starts_with(ACCELERATOR_REQUIRED_CODE));
        assert!(AcceleratorPlacement::default().is_default());
        let json = serde_json::to_string(&planned).unwrap();
        assert_eq!(
            serde_json::from_str::<AcceleratorPlacement>(&json).unwrap(),
            planned
        );
    }
}
