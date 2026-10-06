use super::{RuntimeContext, RuntimeServiceFuture};
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
    pub value_limits: ValueLimits,
}

impl Default for RuntimeCallableArtifactLimits {
    fn default() -> Self {
        Self {
            max_values: 4_096,
            max_units: 1_024,
            max_unit_bytes: runmat_execution::EXECUTABLE_UNIT_ENVELOPE_MAX_BYTES,
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

    fn admit(
        &self,
        runtime: RuntimeContext,
        bundle: RuntimeCallableArtifactBundle,
        limits: RuntimeCallableArtifactLimits,
    ) -> RuntimeServiceFuture<Result<Vec<Value>, RuntimeError>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::value::InlineValue;
    use std::rc::Rc;

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

        fn admit(
            &self,
            _runtime: RuntimeContext,
            _bundle: RuntimeCallableArtifactBundle,
            _limits: RuntimeCallableArtifactLimits,
        ) -> RuntimeServiceFuture<Result<Vec<Value>, RuntimeError>> {
            unreachable!("port ownership test never invokes the service")
        }
    }

    #[test]
    fn parallel_port_projection_strips_parent_publication_authority() {
        let ports = super::super::RuntimeServicePorts::default()
            .with_callable_artifacts(Rc::new(StubService));
        assert!(ports.callable_artifacts().is_some());
        let isolated = ports.without_callable_artifacts();
        let error = isolated
            .require_callable_artifacts("admit a worker callable")
            .expect_err("isolated worker must install its own artifact authority");
        assert_eq!(
            error.capability,
            super::super::RuntimeCapability::CallableArtifacts
        );
    }
}
