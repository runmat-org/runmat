//! Exact selected Network port identity. Ordinary session HTTP/TCP ports retain
//! their existing behavior; selected native invocations require this owner.

use super::{RuntimeHttpService, RuntimeServicePorts, RuntimeTcpService};
use runmat_execution::host::{NativeDeploymentServiceKind, VerifiedNativeServiceIssuer};
use runmat_execution::Digest;
use std::rc::Rc;

#[derive(Clone)]
pub(super) struct RuntimeSelectedNetworkService {
    issuer: VerifiedNativeServiceIssuer,
    invocation: Digest,
    http: Rc<dyn RuntimeHttpService>,
    tcp: Rc<dyn RuntimeTcpService>,
}

impl RuntimeServicePorts {
    pub fn with_selected_network(
        mut self,
        issuer: VerifiedNativeServiceIssuer,
        invocation: Digest,
        http: Rc<dyn RuntimeHttpService>,
        tcp: Rc<dyn RuntimeTcpService>,
    ) -> Result<Self, crate::RuntimeError> {
        if issuer.binding().kind
            != NativeDeploymentServiceKind::Effect(runmat_types::EffectKind::Network)
            || invocation.bytes().iter().all(|byte| *byte == 0)
            || !self.selected_local_matches(issuer.product(), invocation)
        {
            return Err(mismatch("selected Network issuer/invocation is invalid"));
        }
        Rc::make_mut(&mut self.shared).http = Some(http.clone());
        Rc::make_mut(&mut self.shared).tcp = Some(tcp.clone());
        Rc::make_mut(&mut self.shared).selected_network = Some(RuntimeSelectedNetworkService {
            issuer,
            invocation,
            http,
            tcp,
        });
        Ok(self)
    }

    pub fn require_selected_network(
        &self,
        issuer: &VerifiedNativeServiceIssuer,
        invocation: Digest,
    ) -> Result<(), crate::RuntimeError> {
        let selected = self
            .shared
            .selected_network
            .as_ref()
            .ok_or_else(|| mismatch("selected Network ports are absent"))?;
        if selected.issuer != *issuer
            || selected.invocation != invocation
            || self
                .shared
                .http
                .as_ref()
                .is_none_or(|port| !Rc::ptr_eq(port, &selected.http))
            || self
                .shared
                .tcp
                .as_ref()
                .is_none_or(|port| !Rc::ptr_eq(port, &selected.tcp))
        {
            return Err(mismatch(
                "selected Network ports differ from issuer/invocation",
            ));
        }
        Ok(())
    }

    pub(super) fn selected_http_matches(&self) -> bool {
        self.shared.selected_local.is_none()
            || self
                .shared
                .selected_network
                .as_ref()
                .is_some_and(|selected| {
                    self.shared
                        .http
                        .as_ref()
                        .is_some_and(|port| Rc::ptr_eq(port, &selected.http))
                })
    }

    pub(super) fn selected_tcp_matches(&self) -> bool {
        self.shared.selected_local.is_none()
            || self
                .shared
                .selected_network
                .as_ref()
                .is_some_and(|selected| {
                    self.shared
                        .tcp
                        .as_ref()
                        .is_some_and(|port| Rc::ptr_eq(port, &selected.tcp))
                })
    }
}

fn mismatch(message: &str) -> crate::RuntimeError {
    crate::build_runtime_error(message)
        .with_identifier("RunMat:Native:SelectedNetworkMismatch")
        .build()
}
