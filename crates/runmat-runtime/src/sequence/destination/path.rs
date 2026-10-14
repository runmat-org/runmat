use super::{PreparedSequenceEndpoint, SequenceEndpointSpec};
use crate::execution::{RootedValue, RootedValueLease};
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
mod retained;
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
    _retained_roots: RootedValueLease,
    _source_roots: RootedValueLease,
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
    current: RootedValue,
    source_roots: RootedValueLease,
    steps: Vec<PreparedAssignmentStep>,
    object_suffix: Option<ObjectSuffixBuilder>,
    selector_roots: Vec<RootedValueLease>,
}

impl SequenceDestinationBuilder {
    pub fn new(root: &Value) -> Result<Self, RuntimeError> {
        let source_roots = RootedValueLease::for_values(std::iter::once(root))
            .map_err(|error| error.into_runtime_error())?;
        Ok(Self {
            current: RootedValue::new(root.clone()).map_err(|error| error.into_runtime_error())?,
            source_roots,
            steps: Vec::new(),
            object_suffix: None,
            selector_roots: Vec::new(),
        })
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
        self.push_step_with_context(None, step, caller_function_name)
            .await
    }

    pub async fn push_step_in_context(
        &mut self,
        context: &crate::context::RuntimeContext,
        step: AssignmentStepSpec,
        caller_function_name: Option<&str>,
    ) -> Result<(), RuntimeError> {
        self.push_step_with_context(Some(context), step, caller_function_name)
            .await
    }

    async fn push_step_with_context(
        &mut self,
        context: Option<&crate::context::RuntimeContext>,
        step: AssignmentStepSpec,
        caller_function_name: Option<&str>,
    ) -> Result<(), RuntimeError> {
        let step_roots =
            RootedValueLease::for_traced(&step).map_err(|error| error.into_runtime_error())?;
        if let Some(suffix) = &mut self.object_suffix {
            suffix.structural_steps.push(step.clone());
            suffix.steps.push(object_subscript_from_step(step));
            self.selector_roots.push(step_roots);
            return Ok(());
        }
        if let Some(expected_class) =
            crate::object::indexing::class_name_from_base(&self.current).cloned()
        {
            self.object_suffix = Some(ObjectSuffixBuilder {
                target: self.current.value().clone(),
                expected_class,
                structural_steps: vec![step.clone()],
                steps: vec![object_subscript_from_step(step)],
            });
            self.selector_roots.push(step_roots);
            return Ok(());
        }
        let step = prepare_step(&self.current, step).await?;
        self.current = read_step(
            context,
            self.current.value().clone(),
            &step,
            caller_function_name,
        )
        .await?;
        self.steps.push(step);
        self.selector_roots.push(step_roots);
        Ok(())
    }

    pub async fn finish(
        self,
        endpoint: SequenceEndpointSpec,
    ) -> Result<PreparedSequenceDestination, RuntimeError> {
        self.finish_with_context(None, endpoint).await
    }

    pub async fn finish_in_context(
        self,
        context: &crate::context::RuntimeContext,
        endpoint: SequenceEndpointSpec,
    ) -> Result<PreparedSequenceDestination, RuntimeError> {
        self.finish_with_context(Some(context), endpoint).await
    }

    async fn finish_with_context(
        mut self,
        context: Option<&crate::context::RuntimeContext>,
        endpoint: SequenceEndpointSpec,
    ) -> Result<PreparedSequenceDestination, RuntimeError> {
        let _endpoint_roots =
            RootedValueLease::for_traced(&endpoint).map_err(|error| error.into_runtime_error())?;
        if matches!(&endpoint, SequenceEndpointSpec::Member(_))
            && crate::object::indexing::class_name_from_base(&self.current).is_none()
            && !matches!(
                self.current.value(),
                Value::Struct(_) | Value::StructArray(_)
            )
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
                    target: self.current.value().clone(),
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
            let cardinality = match context {
                Some(context) => {
                    crate::builtins::introspection::object_indexing::object_path_cardinality_in_context(
                        context,
                        &suffix.target,
                        &path,
                        runmat_types::ObjectIndexingContext::Assignment,
                        &crate::object::protocol::ObjectAccessContext::default(),
                    )
                    .await?
                }
                None => {
                    crate::builtins::introspection::object_indexing::object_path_assignment_cardinality(
                        &suffix.target,
                        &path,
                    )
                    .await?
                }
            };
            return PreparedSequenceDestination::rooted(
                PreparedDestinationKind::ObjectProtocol(Box::new(
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
                self.source_roots,
            );
        }
        let endpoint = PreparedSequenceEndpoint::prepare(&self.current, endpoint).await?;
        PreparedSequenceDestination::rooted(
            PreparedDestinationKind::Structural {
                steps: self.steps,
                endpoint,
            },
            self.source_roots,
        )
    }
}

impl PreparedSequenceDestination {
    fn rooted(
        kind: PreparedDestinationKind,
        source_roots: RootedValueLease,
    ) -> Result<Self, RuntimeError> {
        let retained_roots =
            RootedValueLease::for_traced(&kind).map_err(|error| error.into_runtime_error())?;
        Ok(Self {
            kind,
            _retained_roots: retained_roots,
            _source_roots: source_roots,
        })
    }
}
