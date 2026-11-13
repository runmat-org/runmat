//! Selected clock identity is independent of generic timing injection.
use super::*;
use runmat_execution::host::{
    native_clock_service_abi, NativeClockInvocationIdentity, NativeDeploymentServiceBehavior,
    NativeDeploymentServiceKind, VerifiedNativeServiceIssuer,
};

/// One live port state; absence cannot retain or expose an ambient clock.
#[derive(Clone)]
pub(super) enum RuntimeTimingPort {
    Hosted(Rc<dyn RuntimeTimingService>),
    SelectedUnavailable,
    Selected {
        service: Rc<dyn RuntimeTimingService>,
        issuer: VerifiedNativeServiceIssuer,
        invocation: NativeClockInvocationIdentity,
    },
}

impl RuntimeServicePorts {
    pub fn with_selected_timing(
        mut self,
        service: Rc<dyn RuntimeTimingService>,
        issuer: VerifiedNativeServiceIssuer,
        invocation: NativeClockInvocationIdentity,
    ) -> Result<Self, crate::RuntimeError> {
        if issuer.binding().kind
            != NativeDeploymentServiceKind::Effect(runmat_types::EffectKind::Clock)
            || issuer.binding().abi != native_clock_service_abi()
            || (issuer.binding().behavior == NativeDeploymentServiceBehavior::DeterministicDeclared)
                != invocation.inputs().is_some()
        {
            return Err(mismatch());
        }
        Rc::make_mut(&mut self.shared).timing = RuntimeTimingPort::Selected {
            service,
            issuer,
            invocation,
        };
        Ok(self)
    }

    pub fn require_selected_timing(
        &self,
        issuer: &VerifiedNativeServiceIssuer,
        invocation: &NativeClockInvocationIdentity,
    ) -> Result<&Rc<dyn RuntimeTimingService>, crate::RuntimeError> {
        match &self.shared.timing {
            RuntimeTimingPort::Selected {
                service,
                issuer: actual,
                invocation: identity,
            } if actual == issuer && identity == invocation => Ok(service),
            _ => Err(mismatch()),
        }
    }
}

fn mismatch() -> crate::RuntimeError {
    crate::build_runtime_error(
        "live clock port differs from the admitted native service/invocation inputs",
    )
    .with_identifier("RunMat:Clock:SelectedServiceMismatch")
    .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unadmitted_selected_clock_cannot_be_restored_by_generic_injection() {
        let ports = RuntimeServicePorts::default();
        let clock = Rc::clone(ports.timing().unwrap());
        let weak = Rc::downgrade(&clock);
        let ports = ports.without_timing();
        assert_eq!(
            ports.timing().err().unwrap().identifier(),
            Some("RunMat:Clock:ServiceUnavailable")
        );
        assert!(ports.with_timing(clock).timing().is_err());
        assert!(
            weak.upgrade().is_none(),
            "an unavailable port retains no ambient implementation"
        );
    }
}
