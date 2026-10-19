use super::{RuntimeContext, RuntimeServiceFuture};
use crate::execution::RootedValues;
use crate::{build_runtime_error, RuntimeError};
use runmat_execution::value::{ValueLimits, ValuePayload};
use runmat_execution::ExecutableUnitEnvelope;
use runmat_value::Value;

/// Finite admission limits for one callable-bearing value transfer.
///
/// The artifact owner applies these limits before decoding executable units or
/// publishing definitions. A caller may choose a smaller limit for a bounded
/// container such as a figure archive.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuntimeCallableArtifactLimits {
    pub max_values: usize,
    pub max_units: usize,
    pub max_unit_bytes: usize,
    pub max_functions: usize,
    pub max_instructions: usize,
    pub value_limits: ValueLimits,
}

impl Default for RuntimeCallableArtifactLimits {
    fn default() -> Self {
        Self {
            max_values: 4_096,
            max_units: 1_024,
            max_unit_bytes: runmat_execution::EXECUTABLE_UNIT_ENVELOPE_MAX_BYTES,
            max_functions: 65_536,
            max_instructions: 10_000_000,
            value_limits: ValueLimits::default(),
        }
    }
}

/// Canonical transport payload for values and every executable unit required
/// by their exact bound callables.
///
/// Unit bytes use the existing executable-envelope codec. This carrier does
/// not assign destination-local function IDs or keep a second callable
/// registry; the selected executor admits the units into its canonical
/// definition owner before decoding `values`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeCallableArtifactBundle {
    values: Vec<ValuePayload>,
    unit_bytes: Vec<Vec<u8>>,
}

impl RuntimeCallableArtifactBundle {
    pub fn new(
        values: Vec<ValuePayload>,
        unit_bytes: Vec<Vec<u8>>,
        limits: RuntimeCallableArtifactLimits,
    ) -> Result<Self, RuntimeCallableArtifactError> {
        if values.len() > limits.max_values {
            return Err(RuntimeCallableArtifactError::Limit {
                field: "callable_artifact.values",
                limit: limits.max_values,
            });
        }
        if unit_bytes.len() > limits.max_units {
            return Err(RuntimeCallableArtifactError::Limit {
                field: "callable_artifact.units",
                limit: limits.max_units,
            });
        }
        let total_unit_bytes = unit_bytes.iter().try_fold(0usize, |total, bytes| {
            total
                .checked_add(bytes.len())
                .ok_or(RuntimeCallableArtifactError::Limit {
                    field: "callable_artifact.unit_bytes",
                    limit: limits.max_unit_bytes,
                })
        })?;
        if total_unit_bytes > limits.max_unit_bytes {
            return Err(RuntimeCallableArtifactError::Limit {
                field: "callable_artifact.unit_bytes",
                limit: limits.max_unit_bytes,
            });
        }
        ValuePayload::validate_all(&values, limits.value_limits)
            .map_err(|source| RuntimeCallableArtifactError::InvalidValues { source })?;
        for (index, bytes) in unit_bytes.iter().enumerate() {
            ExecutableUnitEnvelope::admission(bytes)
                .map_err(|source| RuntimeCallableArtifactError::InvalidUnit { index, source })?;
        }
        Ok(Self { values, unit_bytes })
    }

    pub fn values(&self) -> &[ValuePayload] {
        &self.values
    }

    pub fn unit_bytes(&self) -> &[Vec<u8>] {
        &self.unit_bytes
    }

    pub fn into_parts(self) -> (Vec<ValuePayload>, Vec<Vec<u8>>) {
        (self.values, self.unit_bytes)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RuntimeCallableArtifactError {
    #[error("{field} exceeds its limit of {limit}")]
    Limit { field: &'static str, limit: usize },
    #[error("callable artifact values are invalid: {source}")]
    InvalidValues {
        #[source]
        source: runmat_execution::ContractError,
    },
    #[error("callable artifact unit {index} is invalid: {source}")]
    InvalidUnit {
        index: usize,
        #[source]
        source: runmat_execution::ContractError,
    },
}

impl RuntimeCallableArtifactError {
    pub const IDENTIFIER: &'static str = "RunMat:CallableArtifact:Invalid";

    pub fn into_runtime_error(self) -> RuntimeError {
        build_runtime_error(self.to_string())
            .with_identifier(Self::IDENTIFIER)
            .with_source(self)
            .build()
    }
}

/// Selected session authority for exporting and transactionally admitting
/// callable-bearing values. Invocation remains on `RuntimeCallService`; an
/// implementation must publish through the same definition owner used by the
/// selected bound-function invoker.
pub trait RuntimeCallableArtifactService {
    fn export(
        &self,
        runtime: RuntimeContext,
        values: Vec<Value>,
        limits: RuntimeCallableArtifactLimits,
    ) -> RuntimeServiceFuture<Result<RuntimeCallableArtifactBundle, RuntimeError>>;

    fn prepare_admit(
        &self,
        runtime: RuntimeContext,
        bundle: RuntimeCallableArtifactBundle,
        limits: RuntimeCallableArtifactLimits,
    ) -> RuntimeServiceFuture<Result<Box<dyn RuntimePreparedCallableAdmission>, RuntimeError>>;

    fn admit(
        &self,
        runtime: RuntimeContext,
        bundle: RuntimeCallableArtifactBundle,
        limits: RuntimeCallableArtifactLimits,
    ) -> RuntimeServiceFuture<Result<RootedValues, RuntimeError>> {
        let prepared = self.prepare_admit(runtime, bundle, limits);
        Box::pin(async move {
            let committed = prepared.await?.commit()?;
            Ok(committed.into_rooted_values())
        })
    }
}

/// A fully decoded callable import that has not published any executable
/// authority. Dropping it is a rollback with no destination mutation.
pub trait RuntimePreparedCallableAdmission {
    fn values(&self) -> &[Value];

    /// Publish into the exact generation captured during preparation. This is
    /// synchronous: it performs no callbacks, polling, or host effects.
    fn commit(self: Box<Self>) -> Result<RuntimeCallableArtifactCommit, RuntimeError>;
}

/// Type-erased owner retained after publication so replaced generations and
/// their finalizers can retire outside a downstream domain lock.
pub trait RuntimeCallableArtifactRetirement: 'static {}

impl<T: 'static> RuntimeCallableArtifactRetirement for T {}

/// Successful callable publication and the decoded values it authorized.
/// Keep this receipt alive until a downstream owner has adopted the values.
pub struct RuntimeCallableArtifactCommit {
    values: RootedValues,
    retirement: Box<dyn RuntimeCallableArtifactRetirement>,
}

impl RuntimeCallableArtifactCommit {
    #[doc(hidden)]
    pub fn new(
        values: RootedValues,
        retirement: Box<dyn RuntimeCallableArtifactRetirement>,
    ) -> Self {
        Self { values, retirement }
    }

    pub fn values(&self) -> &[Value] {
        self.values.values()
    }

    pub fn into_rooted_values(self) -> RootedValues {
        let Self { values, retirement } = self;
        drop(retirement);
        values
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::value::InlineValue;
    use std::cell::Cell;
    use std::rc::Rc;

    struct CollectingRetirement {
        handle: runmat_gc::GcHandle,
        observed_live: Rc<Cell<bool>>,
    }

    impl Drop for CollectingRetirement {
        fn drop(&mut self) {
            runmat_gc::gc_collect_major().expect("collect while retiring replaced generation");
            self.observed_live
                .set(runmat_gc::gc_clone_value(&self.handle).is_ok());
        }
    }

    fn scalar() -> ValuePayload {
        ValuePayload::Inline(Box::new(InlineValue::Logical(true)))
    }

    #[test]
    fn bundle_checks_value_count_before_value_validation() {
        let error = RuntimeCallableArtifactBundle::new(
            vec![scalar(), scalar()],
            Vec::new(),
            RuntimeCallableArtifactLimits {
                max_values: 1,
                ..RuntimeCallableArtifactLimits::default()
            },
        )
        .expect_err("value count must be bounded");
        assert!(matches!(
            error,
            RuntimeCallableArtifactError::Limit {
                field: "callable_artifact.values",
                limit: 1
            }
        ));
    }

    #[test]
    fn bundle_checks_aggregate_unit_bytes_before_envelope_decode() {
        let error = RuntimeCallableArtifactBundle::new(
            vec![scalar()],
            vec![vec![0; 5], vec![0; 5]],
            RuntimeCallableArtifactLimits {
                max_unit_bytes: 9,
                ..RuntimeCallableArtifactLimits::default()
            },
        )
        .expect_err("aggregate unit bytes must be bounded");
        assert!(matches!(
            error,
            RuntimeCallableArtifactError::Limit {
                field: "callable_artifact.unit_bytes",
                limit: 9
            }
        ));
    }

    #[test]
    fn bundle_checks_one_shared_value_budget() {
        let error = RuntimeCallableArtifactBundle::new(
            vec![scalar(), scalar()],
            Vec::new(),
            RuntimeCallableArtifactLimits {
                value_limits: ValueLimits {
                    max_nodes: 1,
                    ..ValueLimits::default()
                },
                ..RuntimeCallableArtifactLimits::default()
            },
        )
        .expect_err("members must share one aggregate value budget");
        assert!(matches!(
            error,
            RuntimeCallableArtifactError::InvalidValues { .. }
        ));
    }

    #[test]
    fn commit_retires_replaced_owner_before_releasing_returned_value_roots() {
        runmat_gc::gc_test_context(|| {
            let target = runmat_gc::gc_allocate_rooted(Value::String("retained".into()))
                .expect("allocate managed target");
            let handle = target.handle();
            let values = RootedValues::new(vec![Value::HandleObject(runmat_value::HandleRef {
                class_name: "RetainedCallableValue".into(),
                target: handle,
                valid: true,
            })])
            .expect("root returned callable value");
            target.unroot().expect("transfer root to commit receipt");
            let observed_live = Rc::new(Cell::new(false));
            let committed = RuntimeCallableArtifactCommit::new(
                values,
                Box::new(CollectingRetirement {
                    handle,
                    observed_live: Rc::clone(&observed_live),
                }),
            );

            let returned = committed.into_rooted_values();
            assert!(
                observed_live.get(),
                "retirement must run while returned values remain explicitly rooted"
            );
            assert_eq!(
                runmat_gc::gc_clone_value(&handle).expect("returned target remains readable"),
                Value::String("retained".into())
            );
            assert!(matches!(returned.values(), [Value::HandleObject(_)]));
        });
    }

    struct StubService;

    impl RuntimeCallableArtifactService for StubService {
        fn export(
            &self,
            _runtime: RuntimeContext,
            _values: Vec<Value>,
            _limits: RuntimeCallableArtifactLimits,
        ) -> RuntimeServiceFuture<Result<RuntimeCallableArtifactBundle, RuntimeError>> {
            unreachable!("port ownership test never invokes the service")
        }

        fn prepare_admit(
            &self,
            _runtime: RuntimeContext,
            _bundle: RuntimeCallableArtifactBundle,
            _limits: RuntimeCallableArtifactLimits,
        ) -> RuntimeServiceFuture<Result<Box<dyn RuntimePreparedCallableAdmission>, RuntimeError>>
        {
            unreachable!("port ownership test never invokes the service")
        }
    }

    #[test]
    fn parallel_port_projection_strips_parent_publication_authority() {
        let ports = super::super::RuntimeServicePorts::default()
            .with_callable_artifacts(Rc::new(StubService));
        assert!(ports.callable_artifacts().is_some());
        let isolated = ports.without_callable_artifacts();
        let error = match isolated.require_callable_artifacts("admit a worker callable") {
            Ok(_) => panic!("isolated worker must install its own artifact authority"),
            Err(error) => error,
        };
        assert_eq!(
            error.capability,
            super::super::RuntimeCapability::CallableArtifacts
        );
    }
}
