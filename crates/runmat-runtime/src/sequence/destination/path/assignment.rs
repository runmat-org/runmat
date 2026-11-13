use crate::execution::{RootedValue, RootedValueLease};
use runmat_value::Value;

use super::{PreparedDestinationKind, PreparedSequenceDestination, SequenceDestinationBuilder};
use crate::runtime_error::semantic_error;
use crate::sequence::destination::{AssignmentStepSpec, SequenceEndpointSpec};
use crate::RuntimeError;

impl PreparedSequenceDestination {
    pub async fn prepare(
        root: &Value,
        steps: Vec<AssignmentStepSpec>,
        endpoint: SequenceEndpointSpec,
    ) -> Result<Self, RuntimeError> {
        let mut builder = SequenceDestinationBuilder::new(root)?;
        for step in steps {
            builder.push_step(step, None).await?;
        }
        builder.finish(endpoint).await
    }

    /// Prepare a destination from an already detached root without cloning it.
    pub async fn prepare_owned(
        root: Value,
        steps: Vec<AssignmentStepSpec>,
        endpoint: SequenceEndpointSpec,
    ) -> Result<Self, RuntimeError> {
        let mut builder = SequenceDestinationBuilder::from_owned(root)?;
        for step in steps {
            builder.push_step(step, None).await?;
        }
        builder.finish(endpoint).await
    }

    pub fn cardinality(&self) -> usize {
        match &self.kind {
            PreparedDestinationKind::Structural { endpoint, .. } => endpoint.cardinality(),
            PreparedDestinationKind::ObjectProtocol(destination) => destination.cardinality,
        }
    }

    pub async fn assign(
        self,
        root: Value,
        values: Vec<Value>,
        caller: Option<&str>,
    ) -> Result<RootedValue, RuntimeError> {
        let values = runmat_value::ValueSequence::comma_separated(values)
            .map_err(crate::sequence::sequence_error_to_runtime)?;
        self.assign_with_context(None, root, values, caller).await
    }

    pub async fn assign_in_context(
        self,
        context: &crate::context::RuntimeContext,
        root: Value,
        values: Vec<Value>,
        caller: Option<&str>,
    ) -> Result<RootedValue, RuntimeError> {
        let values = runmat_value::ValueSequence::comma_separated(values)
            .map_err(crate::sequence::sequence_error_to_runtime)?;
        self.assign_with_context(Some(context), root, values, caller)
            .await
    }

    /// Ordinary one-output assignment keeps its RHS inline in the canonical
    /// sequence carrier; it does not manufacture a one-element heap vector.
    pub async fn assign_fixed_in_context(
        self,
        context: &crate::context::RuntimeContext,
        root: Value,
        rhs: Value,
        caller: Option<&str>,
    ) -> Result<RootedValue, RuntimeError> {
        let values = crate::sequence::single_value_sequence(rhs)?;
        self.assign_with_context(Some(context), root, values, caller)
            .await
    }

    async fn assign_with_context(
        self,
        context: Option<&crate::context::RuntimeContext>,
        root: Value,
        values: runmat_value::ValueSequence,
        caller: Option<&str>,
    ) -> Result<RootedValue, RuntimeError> {
        let Self {
            kind,
            _retained_roots,
            _source_roots,
            method_frame,
            frame_control,
        } = self;
        if method_frame.is_some() {
            frame_control.check_owner(context)?;
        }
        let _inputs = match context {
            Some(runtime) => RootedValueLease::for_values_iterator_in_context(
                std::iter::once(&root).chain(values.iter()),
                runtime,
            )?,
            None => RootedValueLease::for_values(std::iter::once(&root).chain(values.iter()))
                .map_err(|error| error.into_runtime_error())?,
        };
        match kind {
            PreparedDestinationKind::Structural { steps, endpoint } => {
                super::structural::assign_path(
                    context,
                    root,
                    steps.into_iter(),
                    endpoint,
                    values,
                    caller,
                    method_frame.as_deref(),
                    _inputs,
                )
                .await
            }
            PreparedDestinationKind::ObjectProtocol(destination) => {
                let super::PreparedObjectProtocolDestination {
                    prefix,
                    expected_class,
                    cardinality_method,
                    suffix_steps,
                    endpoint,
                    path,
                    cardinality,
                } = allocator_api2::boxed::Box::into_inner(destination);
                if values.len() != cardinality {
                    return Err(semantic_error(
                        "CommaSeparatedListAssignmentArity",
                        format!(
                            "sequence assignment requires {cardinality} values, but received {}",
                            values.len()
                        ),
                    ));
                }
                if cardinality_method.as_ref().is_some_and(|method| {
                    !crate::object::protocol::resolved_method_is_current(method)
                }) {
                    return Err(semantic_error(
                        "StaleObjectProtocolResolution",
                        "object protocol binding changed after the destination was prepared",
                    ));
                }
                super::object_assignment::assign_object_path(
                    context,
                    root,
                    prefix.into_iter(),
                    expected_class,
                    suffix_steps,
                    endpoint,
                    path,
                    crate::builtins::introspection::object_indexing::PreparedObjectAssignmentCardinality {
                        count: cardinality, method: cardinality_method,
                    },
                    values,
                    caller,
                    method_frame.as_deref(),
                    _inputs,
                )
                .await
            }
        }
    }
}
