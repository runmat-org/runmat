use runmat_builtins::BuiltinBindingIdentity;
use runmat_value::Value;

use crate::context::host_allocation::HostAllocationError;
use crate::context::RuntimeContext;
use crate::RuntimeError;

pub type RuntimeBuiltinFuture =
    crate::execution::RuntimeContinuationFuture<'static, crate::execution::RootedValueSequence>;
/// Borrowed embedding ingress or an already-rooted language invocation.
/// The latter transfers its original segmented owners into the continuation.
pub enum BuiltinInvocationArguments<'a> {
    Borrowed(&'a [Value]),
    Retained(crate::call::BuiltinArguments),
}

impl BuiltinInvocationArguments<'_> {
    pub fn len(&self) -> usize {
        match self {
            Self::Borrowed(values) => values.len(),
            Self::Retained(values) => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, index: usize) -> Option<&Value> {
        match self {
            Self::Borrowed(values) => values.get(index),
            Self::Retained(values) => values.get(index),
        }
    }
}

impl std::ops::Index<usize> for BuiltinInvocationArguments<'_> {
    type Output = Value;

    fn index(&self, index: usize) -> &Value {
        self.get(index)
            .expect("builtin argument index is in bounds")
    }
}

/// Invocation controls and original argument ownership for one builtin call.
pub struct RuntimeBuiltinInvocation<'a> {
    pub args: BuiltinInvocationArguments<'a>,
    pub context: Option<RuntimeContext>,
    pub requested_outputs: Option<usize>,
    pub(crate) call_observation:
        Option<crate::dispatcher::direct_call_observation::CallAdmissionSink>,
}

impl<'a> RuntimeBuiltinInvocation<'a> {
    pub fn new(
        args: &'a [Value],
        context: Option<RuntimeContext>,
        requested_outputs: Option<usize>,
    ) -> Self {
        Self {
            args: BuiltinInvocationArguments::Borrowed(args),
            context,
            requested_outputs,
            call_observation: None,
        }
    }

    /// Retain a language invocation without materializing or retracing its
    /// captured, routing or supplied argument segments.
    pub fn retained(
        args: crate::call::BuiltinArguments,
        context: Option<RuntimeContext>,
        requested_outputs: Option<usize>,
    ) -> RuntimeBuiltinInvocation<'static> {
        RuntimeBuiltinInvocation {
            args: BuiltinInvocationArguments::Retained(args),
            context,
            requested_outputs,
            call_observation: None,
        }
    }

    #[doc(hidden)]
    pub fn into_continuation_input(self) -> Result<BuiltinContinuationInput, RuntimeError> {
        let args = match self.args {
            BuiltinInvocationArguments::Retained(args) => args,
            BuiltinInvocationArguments::Borrowed(values) => {
                crate::call::BuiltinArguments::from_borrowed(values, self.context.as_ref())?
            }
        };
        Ok(BuiltinContinuationInput {
            args,
            context: self
                .context
                .map(RuntimeContext::with_selected_builtin_output_projection),
            requested_outputs: self.requested_outputs,
            call_observation: self.call_observation,
        })
    }

    pub(crate) fn with_call_observation(
        mut self,
        observation: Option<crate::dispatcher::direct_call_observation::CallAdmissionSink>,
    ) -> Self {
        let host = observation
            .as_ref()
            .and_then(|sink| sink.prepared_host_execution());
        self.context = self
            .context
            .map(|context| context.with_prepared_host_execution(host));
        self.call_observation = observation;
        self
    }
}

pub type RuntimeBuiltinImplementation =
    for<'a> fn(RuntimeBuiltinInvocation<'a>) -> RuntimeBuiltinFuture;

/// The already-rooted inputs and typed invocation controls captured by one
/// generated canonical continuation. This is not a second value transport.
#[doc(hidden)]
pub struct BuiltinContinuationInput {
    pub args: crate::call::BuiltinArguments,
    pub context: Option<RuntimeContext>,
    pub requested_outputs: Option<usize>,
    pub(crate) call_observation:
        Option<crate::dispatcher::direct_call_observation::CallAdmissionSink>,
}

pub type RuntimeBuiltinContinuationLayout = fn() -> Result<u64, HostAllocationError>;

#[derive(Clone, Copy)]
#[repr(C)]
pub struct RuntimeBuiltinBinding {
    pub identity: BuiltinBindingIdentity,
    pub implementation: RuntimeBuiltinImplementation,
    pub continuation_layout: Option<RuntimeBuiltinContinuationLayout>,
    /// Declaration-owned physical association, carried by the selected
    /// implementation rather than resolved from process-wide evidence.
    pub provenance: Option<&'static runmat_builtins::BuiltinImplementationProvenance>,
}

impl RuntimeBuiltinBinding {
    pub const fn new(
        identity: BuiltinBindingIdentity,
        implementation: RuntimeBuiltinImplementation,
    ) -> Self {
        Self {
            identity,
            implementation,
            continuation_layout: None,
            provenance: None,
        }
    }

    pub const fn with_continuation_layout(
        mut self,
        layout: RuntimeBuiltinContinuationLayout,
    ) -> Self {
        self.continuation_layout = Some(layout);
        self
    }

    pub const fn with_provenance(
        mut self,
        provenance: &'static runmat_builtins::BuiltinImplementationProvenance,
    ) -> Self {
        self.provenance = Some(provenance);
        self
    }

    pub fn physical_provenance(
        self,
    ) -> Option<&'static runmat_builtins::BuiltinImplementationProvenance> {
        self.provenance.filter(|provenance| {
            provenance.authority
                == runmat_builtins::BuiltinImplementationAuthority::CanonicalBinding
                && provenance.name == self.identity.builtin.name
                && provenance.binding_variant == Some(self.identity.variant)
        })
    }
}

impl std::fmt::Debug for RuntimeBuiltinBinding {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RuntimeBuiltinBinding")
            .field("identity", &self.identity)
            .finish_non_exhaustive()
    }
}
