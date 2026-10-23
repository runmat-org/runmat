use crate::RuntimeError;
use runmat_thread_local::runmat_thread_local;
use runmat_types::{CallableFallbackPolicy, CallableIdentity, SourceId};
use runmat_value::Value;
use std::cell::RefCell;
use std::future::Future;
use std::ops::ControlFlow;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::Arc;

#[cfg(test)]
#[path = "user_functions/test_support.rs"]
pub mod test_support;

pub type UserFunctionFuture =
    crate::execution::RuntimeContinuationFuture<'static, crate::execution::RootedValueSequence>;
/// Legacy embedding callbacks own an opaque allocation. It is not a managed
/// concrete-future grant and may only be called through the unbounded adapter.
pub type UnscopedUserFunctionFuture =
    Pin<Box<dyn Future<Output = Result<crate::execution::RootedValueSequence, RuntimeError>>>>;
/// A missing pre-effect target returns the exact argument owner to the next
/// resolution tier. A selected target is terminal even for zero outputs.
/// Host selectors may return `Continue` only before invocation effects and
/// must return the original rooted argument owner, not a replacement or a clone.
/// After selection, retirement, cancellation and invocation errors are `Break`.
pub type DynamicFunctionLoadAttempt = ControlFlow<
    Result<crate::execution::RootedValueSequence, RuntimeError>,
    crate::call::RootedFunctionArguments,
>;
pub(crate) type CallableAttempt = ControlFlow<
    Result<crate::execution::RootedValueSequence, RuntimeError>,
    crate::call::RootedFunctionArguments,
>;
pub type DynamicFunctionLoadCompletion =
    ControlFlow<crate::execution::RootedValueSequence, crate::call::RootedFunctionArguments>;
pub type DynamicFunctionLoadFuture =
    crate::execution::RuntimeContinuationFuture<'static, DynamicFunctionLoadCompletion>;
pub type DynamicFunctionClearFuture = crate::execution::RuntimeContinuationFuture<'static, ()>;

#[path = "user_functions/dynamic_load.rs"]
mod dynamic_load;
pub use dynamic_load::await_dynamic_function_load;
pub type FunctionInvoker = dyn Fn(
    Option<crate::context::RuntimeContext>,
    usize,
    crate::call::RootedFunctionArguments,
    usize,
) -> Result<UserFunctionFuture, RuntimeError>;
/// Compatibility entrypoint for an embedding callback without a selected
/// RuntimeContext. It cannot mint foreign reentry authority.
pub type UnscopedFunctionInvoker = dyn Fn(usize, &[Value], usize) -> UnscopedUserFunctionFuture;
/// Selected exact-definition execution retains its original admitted allocation.
/// Admission failure is terminal and does not fall through to another invoker.
pub type BoundFunctionFuture = UserFunctionFuture;
pub type BoundFunctionInvoker = dyn Fn(
    crate::context::RuntimeContext,
    runmat_value::BoundFunctionBinding,
    crate::call::RootedFunctionArguments,
    usize,
    Option<runmat_value::LexicalEnvironment>,
) -> Result<BoundFunctionFuture, RuntimeError>;
#[derive(Debug)]
pub struct ExternalFunctionCall {
    pub function: usize,
    pub display_name: String,
    pub arguments: crate::call::RootedFunctionArguments,
    pub requested_outputs: usize,
}
pub type ExternalFunctionInvoker = dyn Fn(
    Option<crate::context::RuntimeContext>,
    ExternalFunctionCall,
) -> Result<UserFunctionFuture, RuntimeError>;
pub type UnscopedExternalFunctionInvoker =
    dyn Fn(ExternalFunctionCall) -> UnscopedUserFunctionFuture;
pub type LexicalFunctionFuture = crate::execution::RuntimeContinuationFuture<
    'static,
    crate::call::lexical::RootedLexicalCallResult,
>;
pub type UnscopedLexicalFunctionFuture = Pin<
    Box<dyn Future<Output = Result<crate::call::lexical::RootedLexicalCallResult, RuntimeError>>>,
>;
pub type LexicalFunctionInvoker = dyn Fn(
    Option<crate::context::RuntimeContext>,
    crate::call::lexical::LexicalCall,
) -> Result<LexicalFunctionFuture, RuntimeError>;
pub type UnscopedLexicalFunctionInvoker =
    dyn Fn(crate::call::lexical::LexicalCall) -> UnscopedLexicalFunctionFuture;

#[path = "user_functions/future_owner.rs"]
mod future_owner;
pub use future_owner::{admit_lexical_function_with, admit_user_function_with};
pub type FunctionResolver = dyn Fn(&str) -> Option<usize> + Send + Sync;
/// The active source owner's private-name target. A current publication needs
/// no lease; a retained selection carries its exact definition authority.
/// Numeric calls and public names do not use this route.
#[derive(Clone, Debug)]
pub enum ScopedSemanticNameTarget {
    Published(runmat_types::FunctionId),
    Retained(runmat_value::BoundFunctionBinding),
}

impl ScopedSemanticNameTarget {
    pub fn function(&self) -> runmat_types::FunctionId {
        match self {
            Self::Published(function) => *function,
            Self::Retained(binding) => runmat_types::FunctionId(binding.function),
        }
    }
}

/// Invocation-local private-name selection. Its retained definition leases
/// belong to one RuntimeContext and are never sent to another thread.
pub type ScopedNameTargetResolver = dyn Fn(&str) -> Option<ScopedSemanticNameTarget>;
/// Ordered definition facts from the selected semantic resolver owner. A
/// retained private body and a current publication may reuse a numeric ID;
/// observation must not collapse those distinct definition owners.
#[derive(Clone, Debug)]
pub struct SemanticNameCandidate {
    pub function: usize,
    pub source: Option<SourceFunctionInfo>,
    pub definition_owner: Option<Arc<runmat_value::FunctionDefinitionOwner>>,
}

impl SemanticNameCandidate {
    pub fn same_definition(&self, other: &Self) -> bool {
        match (&self.definition_owner, &other.definition_owner) {
            (Some(left), Some(right)) => Arc::ptr_eq(left, right),
            (None, None) => match (
                self.source
                    .as_ref()
                    .and_then(|source| source.binding_origin),
                other
                    .source
                    .as_ref()
                    .and_then(|source| source.binding_origin),
            ) {
                (Some(left), Some(right)) => left == right,
                _ => self.function == other.function,
            },
            _ => false,
        }
    }
}

pub type SemanticNameCandidates = dyn Fn(&str) -> Vec<SemanticNameCandidate> + Send + Sync;
/// A read-only view of the native loader's selected name. The executable
/// loader remains the owner of installed-module and platform-suffix policy.
pub type NativeNameObserverFuture = crate::execution::RuntimeContinuationFuture<
    'static,
    crate::execution::RootedResult<Option<NativeNameFact>>,
>;
pub type NativeNameObserver = dyn Fn(
    crate::context::RuntimeContext,
    String,
) -> Result<NativeNameObserverFuture, RuntimeError>;

#[path = "user_functions/native_observation.rs"]
mod native_observation;
pub use native_observation::{admit_native_name_observation_with, await_native_name_observation};

/// Trusted selected-host resolution and invocation. An absent target follows
/// the same original-owner, pre-effect contract as [`DynamicFunctionLoadAttempt`].
pub type SelectedNativeInvoker = dyn Fn(
    crate::context::RuntimeContext,
    String,
    crate::call::RootedFunctionArguments,
    usize,
) -> Result<DynamicFunctionLoadFuture, RuntimeError>;

/// Native resolution is either session-owned or confined to a selected
/// product. A selected product without a grant cannot use the session owner.
#[derive(Clone)]
pub enum NativeResolutionScope {
    Session,
    SelectedClosed,
    Selected(Rc<SelectedNativeRoute>),
    /// Issued only for an ordinary source invocation. A dynamic `.m` child
    /// must bind its own source owner before this authority can be reissued.
    #[cfg(not(target_arch = "wasm32"))]
    SelectedOrdinary {
        route: Rc<SelectedNativeRoute>,
        mex: Rc<crate::foreign::MexRuntimeSession>,
        source_owner: crate::source_context::SelectedProgramSourceOwner,
    },
}

/// Invocation-local native name owner. Both callbacks are supplied by the
/// same selected host and retain its private byte lease through each future.
#[derive(Clone)]
pub struct SelectedNativeRoute {
    observer: Rc<NativeNameObserver>,
    invoker: Rc<SelectedNativeInvoker>,
}

impl SelectedNativeRoute {
    pub fn new(observer: Rc<NativeNameObserver>, invoker: Rc<SelectedNativeInvoker>) -> Self {
        Self { observer, invoker }
    }

    pub async fn observe(
        &self,
        runtime: crate::context::RuntimeContext,
        name: String,
    ) -> Result<Option<NativeNameFact>, RuntimeError> {
        await_native_name_observation((self.observer)(runtime, name)).await
    }

    pub fn invoke(
        &self,
        runtime: crate::context::RuntimeContext,
        name: String,
        arguments: crate::call::RootedFunctionArguments,
        requested_outputs: usize,
    ) -> Result<DynamicFunctionLoadFuture, RuntimeError> {
        (self.invoker)(runtime, name, arguments, requested_outputs)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeNameFact {
    pub path: std::path::PathBuf,
    pub origin: Option<crate::builtins::common::path_search::FileCandidateOrigin>,
    pub installed: bool,
    /// Search-path discovery cannot claim an installed artifact identity.
    pub installed_artifact: Option<InstalledNativeArtifactFact>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstalledNativeArtifactFact {
    pub identity: runmat_types::ForeignArtifactIdentity,
    pub interop: runmat_types::InteropManifest,
}
#[path = "user_functions/source_facts.rs"]
mod source_facts;
pub use source_facts::*;

/// The point in callable resolution at which a session-backed loader runs.
///
/// Native callable artifacts such as MEX modules have language-defined
/// precedence over source functions. Ordinary dynamically discovered source
/// files are consulted only after the analyzed semantic registry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DynamicFunctionLoadPhase {
    BeforeSemantic,
    AfterSemantic,
    /// Select an exact source definition for `@name` without invoking it.
    /// A source observation carries its selected path and origin through
    /// admission; class-folder discovery must not be re-resolved as a plain
    /// source file after the precedence decision.
    Handle {
        class: Option<runmat_types::ClassIdentity>,
        selected_source: Option<crate::builtins::common::path_search::FileCandidate>,
    },
    /// A registered class has an unresolved external constructor declaration.
    /// Only a source entrypoint proven to own this exact constructor may run.
    RegisteredConstructorSource(runmat_types::ClassIdentity),
}

pub type DynamicFunctionLoader = dyn Fn(
    crate::context::RuntimeContext,
    String,
    crate::call::RootedFunctionArguments,
    usize,
    DynamicFunctionLoadPhase,
    Option<crate::callsite::CallsiteInfo>,
) -> Result<DynamicFunctionLoadFuture, RuntimeError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DynamicFunctionClearRequest {
    All,
    NativeExtensions,
    Named(String),
}

/// Refresh the source lookup inventory without clearing live native modules or workspace values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DynamicFunctionRefreshRequest {
    Default,
    Path,
    Toolbox,
    ToolboxCache,
}

pub type DynamicFunctionRefresher = dyn Fn(
    crate::context::RuntimeContext,
    DynamicFunctionRefreshRequest,
) -> Result<DynamicFunctionClearFuture, RuntimeError>;

pub type DynamicFunctionClearer = dyn Fn(
    crate::context::RuntimeContext,
    DynamicFunctionClearRequest,
) -> Result<DynamicFunctionClearFuture, RuntimeError>;

#[path = "user_functions/call_dispatch.rs"]
mod call_dispatch;
pub use call_dispatch::*;
#[path = "user_functions/name_resolution.rs"]
mod name_resolution;
pub use name_resolution::*;

runmat_thread_local! {
    static SEMANTIC_FUNCTION_INVOKER: RefCell<Option<Rc<FunctionInvoker>>> =
        const { RefCell::new(None) };
    static BOUND_FUNCTION_INVOKER: RefCell<Option<Rc<BoundFunctionInvoker>>> =
        const { RefCell::new(None) };
    static EXTERNAL_FUNCTION_INVOKER: RefCell<Option<Rc<ExternalFunctionInvoker>>> =
        const { RefCell::new(None) };
    static LEXICAL_FUNCTION_INVOKER: RefCell<Option<Rc<LexicalFunctionInvoker>>> =
        const { RefCell::new(None) };
    static SEMANTIC_FUNCTION_RESOLVER: RefCell<Option<Arc<FunctionResolver>>> =
        const { RefCell::new(None) };
    static SCOPED_NAME_TARGET_RESOLVER: RefCell<Option<Rc<ScopedNameTargetResolver>>> =
        const { RefCell::new(None) };
    static SEMANTIC_NAME_CANDIDATES: RefCell<Option<Arc<SemanticNameCandidates>>> =
        const { RefCell::new(None) };
    static SOURCE_FUNCTION_CATALOG: RefCell<Option<Arc<Vec<SourceFunctionInfo>>>> =
        const { RefCell::new(None) };
    static EXECUTION_SOURCE_RESOLVER: RefCell<Option<Rc<ExecutionSourceResolver>>> =
        const { RefCell::new(None) };
    static ACTIVE_SEMANTIC_FUNCTION_STACK: RefCell<Vec<usize>> =
        const { RefCell::new(Vec::new()) };
}

#[path = "user_functions/ports.rs"]
mod ports;
pub use ports::*;

fn active_state() -> Option<std::rc::Rc<crate::context::RuntimeContextState>> {
    crate::context::legacy::active().map(|context| std::rc::Rc::clone(context.state()))
}

pub(crate) fn current_session_lease() -> std::sync::Weak<()> {
    active_state().map_or_else(std::sync::Weak::new, |state| {
        Arc::downgrade(&state.session_lease)
    })
}

#[cfg(test)]
#[path = "user_functions/input_callsite_tests.rs"]
mod input_callsite_tests;

#[cfg(test)]
#[path = "user_functions/binding_origin_tests.rs"]
mod binding_origin_tests;

#[cfg(test)]
mod lexical_tests {
    use super::*;
    use crate::call::lexical::{LexicalCall, LexicalCallResult, LexicalCapture};

    #[test]
    fn lexical_invoker_preserves_binding_identity_and_restores_scope() {
        assert!(current_lexical_function_invoker().is_none());
        let guard = install_lexical_function_invoker(Some(Arc::new(|mut call| {
            Box::pin(async move {
                assert_eq!(call.function, 7);
                assert_eq!(call.arguments, vec![Value::Num(2.0)]);
                call.captures[0].value = Value::Num(5.0);
                crate::call::lexical::RootedLexicalCallResult::new(LexicalCallResult {
                    outputs: runmat_value::ValueSequence::single(Value::Num(7.0)).unwrap(),
                    captures: call.captures,
                })
                .map_err(|error| error.into_runtime_error())
            })
        })));
        let result = futures::executor::block_on(try_call_lexical_function(LexicalCall {
            function: 7,
            captures: vec![LexicalCapture {
                binding: runmat_types::BindingId(3),
                value: Value::Num(1.0),
            }],
            environment: None,
            arguments: vec![Value::Num(2.0)],
            requested_outputs: 1,
        }))
        .expect("lexical invoker is installed")
        .expect("lexical call succeeds");
        assert_eq!(result.captures[0].value, Value::Num(5.0));
        result.with_owned(|result| {
            assert_eq!(result.outputs.as_slice(), &[Value::Num(7.0)]);
        });
        drop(guard);
        assert!(current_lexical_function_invoker().is_none());
    }

    #[test]
    fn external_invoker_preserves_identity_kind_and_restores_scope() {
        assert!(current_external_function_invoker().is_none());
        let guard = install_external_function_invoker(Some(Arc::new(|call| {
            Box::pin(async move {
                assert_eq!(call.function, 0);
                assert_eq!(call.display_name, "published");
                assert!(call.arguments.iter().eq([Value::Num(2.0)].iter()));
                assert_eq!(call.requested_outputs, 1);
                crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(12.0))
            })
        })));
        let result =
            futures::executor::block_on(try_call_external_function(ExternalFunctionCall {
                function: 0,
                display_name: "published".into(),
                arguments: crate::call::RootedFunctionArguments::ordinary(
                    crate::execution::RootedValues::new(vec![Value::Num(2.0)])
                        .expect("root external invocation fixture"),
                ),
                requested_outputs: 1,
            }))
            .expect("external invoker is installed")
            .expect("external call succeeds");
        assert_eq!(result.as_slice(), &[Value::Num(12.0)]);
        drop(guard);
        assert!(current_external_function_invoker().is_none());
    }
}
