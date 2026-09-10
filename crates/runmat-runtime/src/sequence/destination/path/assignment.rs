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
        let mut builder = SequenceDestinationBuilder::new(root);
        for step in steps {
            builder.push_step(step, None).await?;
        }
        builder.finish(endpoint).await
    }

    pub const fn cardinality(&self) -> usize {
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
    ) -> Result<Value, RuntimeError> {
        match self.kind {
            PreparedDestinationKind::Structural { steps, endpoint } => {
                super::structural::assign_path(root, steps.into(), endpoint, values, caller).await
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
                } = *destination;
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
                    root,
                    prefix.into(),
                    expected_class,
                    suffix_steps,
                    endpoint,
                    path,
                    values,
                    caller,
                )
                .await
            }
        }
    }
}
