use super::{PreparedSequenceEndpoint, SequenceEndpointSpec};
use crate::indexing::plan::IndexPlan;
use crate::object::indexing::{ObjectSubscript, ObjectSubscriptPath};
use crate::runtime_error::semantic_error;
use crate::RuntimeError;
use runmat_value::Value;

mod structural;
use structural::{prepare_step, read_step};
mod assignment;
mod conversion;
mod object_assignment;
use conversion::{object_subscript_from_endpoint, object_subscript_from_step};

#[cfg(test)]
mod tests;

#[derive(Debug, Clone)]
pub enum AssignmentStepSpec {
    Member(runmat_types::MemberName),
    Parentheses {
        selectors: Vec<crate::object::indexing::ObjectIndexComponent>,
    },
    Braces(Vec<crate::object::indexing::ObjectIndexComponent>),
}

#[derive(Debug, Clone)]
pub(super) enum PreparedAssignmentStep {
    Member(runmat_types::MemberName),
    Parentheses {
        plan: IndexPlan,
        selector_values: Vec<crate::object::indexing::ObjectIndexComponent>,
    },
    Braces(Vec<crate::object::indexing::ObjectIndexComponent>),
}

#[derive(Debug, Clone)]
pub struct PreparedSequenceDestination {
    kind: PreparedDestinationKind,
}

#[derive(Debug, Clone)]
enum PreparedDestinationKind {
    Structural {
        steps: Vec<PreparedAssignmentStep>,
        endpoint: PreparedSequenceEndpoint,
    },
    ObjectProtocol(Box<PreparedObjectProtocolDestination>),
}

#[derive(Debug, Clone)]
struct PreparedObjectProtocolDestination {
    prefix: Vec<PreparedAssignmentStep>,
    expected_class: runmat_types::ClassIdentity,
    cardinality_method: Option<crate::object::protocol::ResolvedObjectMethod>,
    suffix_steps: Vec<AssignmentStepSpec>,
    endpoint: SequenceEndpointSpec,
    path: ObjectSubscriptPath,
    cardinality: usize,
}

#[derive(Debug, Clone)]
struct ObjectSuffixBuilder {
    target: Value,
    expected_class: runmat_types::ClassIdentity,
    structural_steps: Vec<AssignmentStepSpec>,
    steps: Vec<ObjectSubscript>,
}

#[derive(Debug, Clone)]
pub struct SequenceDestinationBuilder {
    current: Value,
    steps: Vec<PreparedAssignmentStep>,
    object_suffix: Option<ObjectSuffixBuilder>,
}

impl SequenceDestinationBuilder {
    pub fn new(root: &Value) -> Self {
        Self {
            current: root.clone(),
            steps: Vec::new(),
            object_suffix: None,
        }
    }

    pub fn current(&self) -> &Value {
        &self.current
    }

    pub fn selector_extent(
        &self,
        component_count: usize,
        component: usize,
    ) -> Result<usize, RuntimeError> {
        structural::selector_extent(&self.current, component_count, component)
    }

    pub async fn push_step(
        &mut self,
        step: AssignmentStepSpec,
        caller_function_name: Option<&str>,
    ) -> Result<(), RuntimeError> {
        if let Some(suffix) = &mut self.object_suffix {
            suffix.structural_steps.push(step.clone());
            suffix.steps.push(object_subscript_from_step(step));
            return Ok(());
        }
        if let Some(expected_class) =
            crate::object::indexing::class_name_from_base(&self.current).cloned()
        {
            self.object_suffix = Some(ObjectSuffixBuilder {
                target: self.current.clone(),
                expected_class,
                structural_steps: vec![step.clone()],
                steps: vec![object_subscript_from_step(step)],
            });
            return Ok(());
        }
        let step = prepare_step(&self.current, step).await?;
        self.current = read_step(self.current.clone(), &step, caller_function_name).await?;
        self.steps.push(step);
        Ok(())
    }

    pub async fn finish(
        mut self,
        endpoint: SequenceEndpointSpec,
    ) -> Result<PreparedSequenceDestination, RuntimeError> {
        if matches!(&endpoint, SequenceEndpointSpec::Member(_))
            && crate::object::indexing::class_name_from_base(&self.current).is_none()
            && !matches!(&self.current, Value::Struct(_) | Value::StructArray(_))
            && self.steps.iter().any(|step| {
                matches!(
                    step,
                    PreparedAssignmentStep::Parentheses { selector_values, .. }
                        if selector_values.iter().any(|selector| matches!(
                            selector,
                            crate::object::indexing::ObjectIndexComponent::Colon
                        ))
                )
            })
        {
            return Err(semantic_error(
                "UndefinedColonSequenceDestination",
                "colon cannot define the shape of an undefined structure-array destination",
            ));
        }
        if self.object_suffix.is_none() {
            if let Some(expected_class) =
                crate::object::indexing::class_name_from_base(&self.current).cloned()
            {
                self.object_suffix = Some(ObjectSuffixBuilder {
                    target: self.current.clone(),
                    expected_class,
                    structural_steps: Vec::new(),
                    steps: Vec::new(),
                });
            }
        }
        if let Some(mut suffix) = self.object_suffix {
            suffix
                .steps
                .push(object_subscript_from_endpoint(endpoint.clone()));
            let path = ObjectSubscriptPath::new(suffix.steps)?;
            let cardinality = crate::builtins::introspection::object_indexing::object_path_assignment_cardinality(
                &suffix.target,
                &path,
            )
            .await?;
            return Ok(PreparedSequenceDestination {
                kind: PreparedDestinationKind::ObjectProtocol(Box::new(
                    PreparedObjectProtocolDestination {
                        prefix: self.steps,
                        expected_class: suffix.expected_class,
                        cardinality_method: cardinality.method,
                        suffix_steps: suffix.structural_steps,
                        endpoint,
                        path,
                        cardinality: cardinality.count,
                    },
                )),
            });
        }
        let endpoint = PreparedSequenceEndpoint::prepare(&self.current, endpoint).await?;
        Ok(PreparedSequenceDestination {
            kind: PreparedDestinationKind::Structural {
                steps: self.steps,
                endpoint,
            },
        })
    }
}
