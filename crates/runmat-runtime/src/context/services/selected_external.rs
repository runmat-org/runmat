//! Live, invocation-bound ports supplied by a qualified linked host component.
//! Service declarations alone cannot construct these owners.

use super::{RuntimeBuiltinService, RuntimeCallService, RuntimeHostService, RuntimeServicePorts};
use crate::context::{
    RuntimeContext, RuntimeGraphicsCallbackQueue, RuntimeGraphicsPresentationService,
    WeakRuntimeGraphicsCallbackQueue,
};
use runmat_execution::{
    host::{NativeDeploymentServiceKind, VerifiedNativeServiceIssuer},
    Digest,
};
use runmat_types::EffectKind;
use std::rc::Rc;

/// A console host may be noninteractive. Interactive hosts additionally own
/// the exact presentation and callback queue used by graphical operations.
pub struct NativeLinkedHostPorts {
    host: Rc<dyn RuntimeHostService>,
    graphics: Option<Rc<dyn RuntimeGraphicsPresentationService>>,
    callbacks: Option<RuntimeGraphicsCallbackQueue>,
}

impl NativeLinkedHostPorts {
    pub fn console(host: Rc<dyn RuntimeHostService>) -> Self {
        Self {
            host,
            graphics: None,
            callbacks: None,
        }
    }

    pub fn interactive(
        host: Rc<dyn RuntimeHostService>,
        graphics: Rc<dyn RuntimeGraphicsPresentationService>,
        callbacks: RuntimeGraphicsCallbackQueue,
    ) -> Result<Self, crate::RuntimeError> {
        if !host.accepts_queued_interaction() {
            return Err(
                crate::build_runtime_error("linked UI host does not provide interaction")
                    .with_identifier("RunMat:NativeSelected:HostUnavailable")
                    .build(),
            );
        }
        Ok(Self {
            host,
            graphics: Some(graphics),
            callbacks: Some(callbacks),
        })
    }
}

/// Selected callable routing is independent of console and graphics UI.
pub struct NativeLinkedCallbackPort {
    call: Rc<dyn RuntimeCallService>,
}

impl NativeLinkedCallbackPort {
    pub fn new(call: Rc<dyn RuntimeCallService>) -> Self {
        Self { call }
    }
}

/// A linked dynamic binding owner, not a process-global builtin registry.
/// The service must issue a registration revision for every returned binding.
pub struct NativeLinkedDynamicPort {
    service: Rc<dyn RuntimeBuiltinService>,
}

impl NativeLinkedDynamicPort {
    pub fn new(service: Rc<dyn RuntimeBuiltinService>) -> Self {
        Self { service }
    }

    pub fn into_service(self) -> Rc<dyn RuntimeBuiltinService> {
        self.service
    }
}

#[derive(Clone)]
pub(super) struct SelectedHostPort {
    pub issuer: VerifiedNativeServiceIssuer,
    pub invocation: Digest,
    pub host: Rc<dyn RuntimeHostService>,
    pub graphics: Option<Rc<dyn RuntimeGraphicsPresentationService>>,
    pub callbacks: Option<WeakRuntimeGraphicsCallbackQueue>,
}

#[derive(Clone)]
pub(super) struct SelectedCallbackPort {
    pub issuer: VerifiedNativeServiceIssuer,
    pub invocation: Digest,
    pub call: Rc<dyn RuntimeCallService>,
}

#[derive(Clone)]
pub(super) struct SelectedDynamicPort {
    pub issuer: VerifiedNativeServiceIssuer,
    pub invocation: Digest,
    pub service: Rc<dyn RuntimeBuiltinService>,
}

fn mismatch(message: &'static str) -> crate::RuntimeError {
    crate::build_runtime_error(message)
        .with_identifier("RunMat:NativeSelected:ServiceMismatch")
        .build()
}

fn nonzero(invocation: Digest) -> bool {
    invocation.bytes().iter().any(|byte| *byte != 0)
}

impl RuntimeServicePorts {
    pub fn with_selected_linked_host(
        mut self,
        selected: &RuntimeContext,
        issuer: VerifiedNativeServiceIssuer,
        invocation: Digest,
        owner: &NativeLinkedHostPorts,
    ) -> Result<Self, crate::RuntimeError> {
        if !nonzero(invocation)
            || issuer.binding().kind
                != NativeDeploymentServiceKind::Effect(EffectKind::UserInterface)
            || !self.selected_local_matches(issuer.product(), invocation)
        {
            return Err(mismatch("linked host has the wrong issuer or invocation"));
        }
        let host = owner.host.clone();
        let graphics = owner.graphics.clone();
        let callbacks = owner.callbacks.as_ref();
        if callbacks.is_some_and(|queue| !queue.belongs_to(selected.local_identity()))
            || graphics.is_some() != callbacks.is_some()
        {
            return Err(mismatch(
                "linked host callback queue belongs to another invocation",
            ));
        }
        let selected_port = SelectedHostPort {
            issuer: issuer.clone(),
            invocation,
            host: host.clone(),
            graphics: graphics.clone(),
            callbacks: callbacks.map(RuntimeGraphicsCallbackQueue::downgrade),
        };
        Rc::make_mut(&mut self.shared).selected_ui = Some(selected_port);
        Rc::make_mut(&mut self.shared).host = Some(host);
        Rc::make_mut(&mut self.shared).graphics_presentation = graphics;
        Rc::make_mut(&mut self.shared).graphics_callbacks =
            callbacks.map(RuntimeGraphicsCallbackQueue::downgrade);
        Ok(self)
    }

    pub fn require_selected_linked_host(
        &self,
        issuer: &VerifiedNativeServiceIssuer,
        invocation: Digest,
    ) -> Result<(), crate::RuntimeError> {
        let selected = self
            .shared
            .selected_ui
            .as_ref()
            .ok_or_else(|| mismatch("selected linked host port is absent"))?;
        if selected.issuer != *issuer
            || selected.invocation != invocation
            || !self.selected_local_matches(issuer.product(), invocation)
            || !self
                .shared
                .host
                .as_ref()
                .is_some_and(|host| Rc::ptr_eq(host, &selected.host))
            || !same_optional_owner(
                self.shared.graphics_presentation.as_ref(),
                selected.graphics.as_ref(),
            )
            || !same_optional_queue(
                self.shared.graphics_callbacks.as_ref(),
                selected.callbacks.as_ref(),
            )
        {
            return Err(mismatch(
                "live linked host differs from the selected issuer",
            ));
        }
        Ok(())
    }

    pub fn with_selected_linked_callback(
        mut self,
        issuer: VerifiedNativeServiceIssuer,
        invocation: Digest,
        owner: NativeLinkedCallbackPort,
    ) -> Result<Self, crate::RuntimeError> {
        if !nonzero(invocation)
            || issuer.binding().kind
                != NativeDeploymentServiceKind::Effect(EffectKind::HostCallback)
            || !self.selected_local_matches(issuer.product(), invocation)
        {
            return Err(mismatch(
                "linked callable has the wrong issuer or invocation",
            ));
        }
        Rc::make_mut(&mut self.shared).call = Some(owner.call.clone());
        Rc::make_mut(&mut self.shared).selected_callback = Some(SelectedCallbackPort {
            issuer,
            invocation,
            call: owner.call,
        });
        Ok(self)
    }

    pub fn require_selected_linked_callback(
        &self,
        issuer: &VerifiedNativeServiceIssuer,
        invocation: Digest,
    ) -> Result<(), crate::RuntimeError> {
        let selected = self
            .shared
            .selected_callback
            .as_ref()
            .ok_or_else(|| mismatch("selected callable port is absent"))?;
        if selected.issuer != *issuer
            || selected.invocation != invocation
            || !self.selected_local_matches(issuer.product(), invocation)
            || !self
                .shared
                .call
                .as_ref()
                .is_some_and(|call| Rc::ptr_eq(call, &selected.call))
        {
            return Err(mismatch("live callable differs from the selected issuer"));
        }
        Ok(())
    }

    pub fn with_selected_linked_dynamic(
        mut self,
        issuer: VerifiedNativeServiceIssuer,
        invocation: Digest,
        owner: NativeLinkedDynamicPort,
    ) -> Result<Self, crate::RuntimeError> {
        if !nonzero(invocation)
            || issuer.binding().kind != NativeDeploymentServiceKind::HostedDynamicDispatch
            || !self.selected_local_matches(issuer.product(), invocation)
        {
            return Err(mismatch("linked dynamic bindings have the wrong issuer"));
        }
        let service: Rc<dyn RuntimeBuiltinService> =
            Rc::new(RevisionCheckedBindings(owner.into_service()));
        Rc::make_mut(&mut self.shared).builtin = Some(service.clone());
        Rc::make_mut(&mut self.shared).selected_dynamic = Some(SelectedDynamicPort {
            issuer,
            invocation,
            service,
        });
        Ok(self)
    }

    pub fn require_selected_linked_dynamic(
        &self,
        issuer: &VerifiedNativeServiceIssuer,
        invocation: Digest,
    ) -> Result<(), crate::RuntimeError> {
        let selected = self
            .shared
            .selected_dynamic
            .as_ref()
            .ok_or_else(|| mismatch("selected dynamic binding owner is absent"))?;
        if selected.issuer != *issuer
            || selected.invocation != invocation
            || !self.selected_local_matches(issuer.product(), invocation)
            || !self
                .shared
                .builtin
                .as_ref()
                .is_some_and(|builtin| Rc::ptr_eq(builtin, &selected.service))
        {
            return Err(mismatch(
                "live dynamic binding owner differs from the selected issuer",
            ));
        }
        Ok(())
    }
}

fn same_optional_owner<T: ?Sized>(left: Option<&Rc<T>>, right: Option<&Rc<T>>) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => Rc::ptr_eq(left, right),
        (None, None) => true,
        _ => false,
    }
}

fn same_optional_queue(
    left: Option<&WeakRuntimeGraphicsCallbackQueue>,
    right: Option<&WeakRuntimeGraphicsCallbackQueue>,
) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => left.upgrade().is_some_and(|owner| right.points_to(&owner)),
        (None, None) => true,
        _ => false,
    }
}

struct RevisionCheckedBindings(Rc<dyn RuntimeBuiltinService>);

impl RuntimeBuiltinService for RevisionCheckedBindings {
    fn bindings_by_name(&self, name: &str) -> Vec<crate::builtin::RuntimeBuiltinBinding> {
        self.0
            .bindings_by_name(name)
            .into_iter()
            .filter(|binding| self.0.binding_registration_revision(*binding).is_some())
            .collect()
    }

    fn binding_registration_revision(
        &self,
        binding: crate::builtin::RuntimeBuiltinBinding,
    ) -> Option<Digest> {
        self.0.binding_registration_revision(binding)
    }
}
