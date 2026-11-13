use super::{PreparedSequenceEndpoint, SequenceEndpointSpec};
use crate::execution::{RootedValue, RootedValueLease};
use crate::indexing::plan::IndexPlan;
use crate::object::indexing::{ObjectSubscript, ObjectSubscriptPath};
use crate::runtime_error::semantic_error;
use crate::RuntimeError;
use runmat_value::Value;

mod creation;
mod frame;
mod structural;
pub use creation::assignment_container_in_context;
use structural::read_step;
mod assignment;
mod conversion;
mod object_assignment;
mod retained;
use super::control::{ControlBox, ControlVec, DestinationControl};
use conversion::{object_subscript_from_endpoint, object_subscript_from_step};

#[cfg(test)]
mod tests;

#[derive(Debug, Clone)]
pub enum AssignmentStepSpec {
    Member(runmat_value::Utf8Text),
    Parentheses {
        selectors: runmat_value::ExactArrayStorage<crate::object::indexing::ObjectIndexComponent>,
    },
    Braces(runmat_value::ExactArrayStorage<crate::object::indexing::ObjectIndexComponent>),
}

#[derive(Debug, Clone)]
pub(super) enum PreparedAssignmentStep {
    Member(runmat_value::Utf8Text),
    InitializingMember {
        member: runmat_value::Utf8Text,
        intent: runmat_types::AssignmentChildCreation,
        origin: ReceiverCreationOrigin,
    },
    Parentheses {
        plan: IndexPlan,
        selector_values:
            runmat_value::ExactArrayStorage<crate::object::indexing::ObjectIndexComponent>,
    },
    InitializingParentheses {
        plan: IndexPlan,
        selector_values:
            runmat_value::ExactArrayStorage<crate::object::indexing::ObjectIndexComponent>,
    },
    Braces(runmat_value::ExactArrayStorage<crate::object::indexing::ObjectIndexComponent>),
    InitializingCellContents {
        positions: runmat_value::ExactArrayStorage<usize>,
        intent: runmat_types::AssignmentChildCreation,
    },
}

/// Private provenance of a selected structural receiver, never inferred from
/// an empty numeric field's payload. Only canonical projection geometry mints
/// the new-row variant; it does not authorize conversion of an existing row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ReceiverCreationOrigin {
    Existing,
    NewStructureElement,
}

#[derive(Debug)]
pub struct PreparedSequenceDestination {
    kind: PreparedDestinationKind,
    _retained_roots: ControlVec<RootedValueLease>,
    _source_roots: RootedValueLease,
    method_frame: Option<ControlBox<crate::object::protocol::ObjectAccessContext>>,
    frame_control: DestinationControl,
}

#[derive(Debug)]
enum PreparedDestinationKind {
    Structural {
        steps: ControlVec<PreparedAssignmentStep>,
        endpoint: PreparedSequenceEndpoint,
    },
    ObjectProtocol(ControlBox<PreparedObjectProtocolDestination>),
}

#[derive(Debug)]
struct PreparedObjectProtocolDestination {
    prefix: ControlVec<PreparedAssignmentStep>,
    expected_class: runmat_value::SharedClassIdentity,
    cardinality_method: Option<crate::object::protocol::ResolvedObjectMethod>,
    suffix_steps: ControlVec<AssignmentStepSpec>,
    endpoint: SequenceEndpointSpec,
    path: ObjectSubscriptPath,
    cardinality: usize,
}

#[derive(Debug)]
struct ObjectSuffixBuilder {
    target: Value,
    expected_class: runmat_value::SharedClassIdentity,
    structural_steps: ControlVec<AssignmentStepSpec>,
    steps: ControlVec<ObjectSubscript>,
}

#[derive(Debug)]
pub struct SequenceDestinationBuilder {
    current: RootedValue,
    source_roots: RootedValueLease,
    steps: ControlVec<PreparedAssignmentStep>,
    object_suffix: Option<ObjectSuffixBuilder>,
    selector_roots: ControlVec<RootedValueLease>,
    control: DestinationControl,
    assignment_creation: runmat_types::AssignmentCreationPolicy,
    current_origin: ReceiverCreationOrigin,
    method_frame: Option<ControlBox<crate::object::protocol::ObjectAccessContext>>,
}

impl SequenceDestinationBuilder {
    pub fn new(root: &Value) -> Result<Self, RuntimeError> {
        Self::from_owned_with_context(None, root.clone())
    }
    pub fn new_in_context(
        context: &crate::context::RuntimeContext,
        root: &Value,
    ) -> Result<Self, RuntimeError> {
        Self::from_owned_with_context(Some(context), root.clone())
    }
    pub fn from_owned(root: Value) -> Result<Self, RuntimeError> {
        Self::from_owned_with_context(None, root)
    }
    pub fn from_owned_in_context(
        context: &crate::context::RuntimeContext,
        root: Value,
    ) -> Result<Self, RuntimeError> {
        Self::from_owned_with_context(Some(context), root)
    }
    fn from_owned_with_context(
        context: Option<&crate::context::RuntimeContext>,
        root: Value,
    ) -> Result<Self, RuntimeError> {
        let control = DestinationControl::new(context)?;
        let source_roots = match context {
            Some(runtime) => {
                RootedValueLease::for_values_iterator_in_context(std::iter::once(&root), runtime)?
            }
            None => RootedValueLease::for_values(std::iter::once(&root))
                .map_err(|e| e.into_runtime_error())?,
        };
        let current = RootedValue::in_optional_context(root, context)?;
        Ok(Self {
            current,
            source_roots,
            steps: control.vector(),
            object_suffix: None,
            selector_roots: control.vector(),
            control,
            assignment_creation: runmat_types::AssignmentCreationPolicy::ExistingOnly,
            current_origin: ReceiverCreationOrigin::Existing,
            method_frame: None,
        })
    }

    pub fn current(&self) -> &Value {
        &self.current
    }

    /// Resolve only the prefix preceding the current indexed step, and only
    /// when the canonical selector actually evaluates its outer `end`.
    /// The eventual assignment still owns its complete unsplit suffix.
    pub async fn prepare_current_receiver_for_end_in_context(
        &self,
        context: &crate::context::RuntimeContext,
        caller: Option<&str>,
    ) -> Result<crate::object::protocol::PreparedSubscriptReceiver, RuntimeError> {
        self.control.check_owner(Some(context))?;
        let (root, prefix) = if let Some(suffix) = &self.object_suffix {
            let prefix = if suffix.steps.is_empty() {
                None
            } else {
                let steps = crate::object::indexing::array_in_context(
                    Some(context),
                    suffix.steps.len(),
                    |index| Ok(suffix.steps[index].clone()),
                )?;
                Some(ObjectSubscriptPath::from_storage(steps)?)
            };
            (suffix.target.clone(), prefix)
        } else {
            (self.current.value().clone(), None)
        };
        if let Some(access) = self.method_frame.as_deref() {
            return crate::object::protocol::prepare_subscript_receiver_with_borrowed_access(
                context, root, prefix, access,
            )
            .await;
        }
        crate::object::protocol::prepare_subscript_receiver_in_context(
            context,
            root,
            prefix,
            crate::object::protocol::ObjectAccessContext::from_legacy_function_name(caller),
        )
        .await
    }

    /// Ordinary fixed assignments retain their original creation authority;
    /// legacy multi-output destinations keep their existing read-only prefix.
    pub fn with_assignment_creation(
        mut self,
        creation: runmat_types::AssignmentCreationPolicy,
    ) -> Self {
        self.assignment_creation = creation;
        self
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
        self.push_step_with_context(
            None,
            step,
            runmat_types::AssignmentChildCreation::ScalarStructure,
            caller_function_name,
        )
        .await
    }

    pub async fn push_step_in_context(
        &mut self,
        context: &crate::context::RuntimeContext,
        step: AssignmentStepSpec,
        caller_function_name: Option<&str>,
    ) -> Result<(), RuntimeError> {
        self.push_step_with_context(
            Some(context),
            step,
            runmat_types::AssignmentChildCreation::ScalarStructure,
            caller_function_name,
        )
        .await
    }

    /// A canonical fixed-path suffix supplies the missing child's exact
    /// container before the next selector (and its contextual end) executes.
    pub async fn push_assignment_step_in_context(
        &mut self,
        context: &crate::context::RuntimeContext,
        step: AssignmentStepSpec,
        child_creation: runmat_types::AssignmentChildCreation,
        caller_function_name: Option<&str>,
    ) -> Result<(), RuntimeError> {
        self.push_step_with_context(Some(context), step, child_creation, caller_function_name)
            .await
    }

    async fn push_step_with_context(
        &mut self,
        context: Option<&crate::context::RuntimeContext>,
        step: AssignmentStepSpec,
        child_creation: runmat_types::AssignmentChildCreation,
        caller_function_name: Option<&str>,
    ) -> Result<(), RuntimeError> {
        self.control.reserve(context, &mut self.selector_roots, 1)?;
        let step_roots = match context {
            Some(runtime) => {
                RootedValueLease::for_values_iterator_in_context(step.retained_values(), runtime)?
            }
            None => {
                RootedValueLease::for_traced(&step).map_err(|error| error.into_runtime_error())?
            }
        };
        if let Some(suffix) = &mut self.object_suffix {
            self.control
                .reserve(context, &mut suffix.structural_steps, 1)?;
            self.control.reserve(context, &mut suffix.steps, 1)?;
            suffix.structural_steps.push(step.clone());
            suffix.steps.push(object_subscript_from_step(step));
            self.selector_roots.push(step_roots);
            return Ok(());
        }
        if let Some(expected_class) =
            crate::object::indexing::class_name_owner_from_base(&self.current).cloned()
        {
            let mut structural_steps = self.control.vector();
            let mut steps = self.control.vector();
            self.control
                .push(context, &mut structural_steps, step.clone())?;
            self.control
                .push(context, &mut steps, object_subscript_from_step(step))?;
            self.object_suffix = Some(ObjectSuffixBuilder {
                target: self.current.value().clone(),
                expected_class,
                structural_steps,
                steps,
            });
            self.selector_roots.push(step_roots);
            return Ok(());
        }
        self.control.reserve(context, &mut self.steps, 1)?;
        let intent =
            if self.assignment_creation == runmat_types::AssignmentCreationPolicy::ExistingOnly {
                runmat_types::AssignmentChildCreation::ExistingOnly
            } else {
                child_creation
            };
        let step = structural::prepare_step_with_initialization(
            context,
            &self.current,
            step,
            intent,
            self.current_origin,
        )
        .await?;
        let next_origin = match &step {
            PreparedAssignmentStep::InitializingParentheses { plan, .. } => {
                use crate::aggregate::structure::operations::{
                    selected_structure_element, SelectedStructureElement,
                };
                match selected_structure_element(&self.current, plan)? {
                    SelectedStructureElement::New => ReceiverCreationOrigin::NewStructureElement,
                    SelectedStructureElement::Existing(0)
                        if self.current_origin == ReceiverCreationOrigin::NewStructureElement =>
                    {
                        self.current_origin
                    }
                    _ => ReceiverCreationOrigin::Existing,
                }
            }
            _ => ReceiverCreationOrigin::Existing,
        };
        self.current = read_step(
            context,
            self.current.value().clone(),
            &step,
            caller_function_name,
            self.method_frame.as_deref(),
        )
        .await?;
        self.current_origin = next_origin;
        self.control.reserve(context, &mut self.steps, 0)?;
        self.steps.push(step);
        self.selector_roots.push(step_roots);
        Ok(())
    }

    pub fn finish(
        self,
        endpoint: SequenceEndpointSpec,
    ) -> impl std::future::Future<Output = Result<PreparedSequenceDestination, RuntimeError>> {
        self.finish_with_context(None, endpoint)
    }

    pub fn finish_in_context<'a>(
        self,
        context: &'a crate::context::RuntimeContext,
        endpoint: SequenceEndpointSpec,
    ) -> impl std::future::Future<Output = Result<PreparedSequenceDestination, RuntimeError>> + 'a
    {
        self.finish_with_context(Some(context), endpoint)
    }

    async fn finish_with_context(
        mut self,
        context: Option<&crate::context::RuntimeContext>,
        endpoint: SequenceEndpointSpec,
    ) -> Result<PreparedSequenceDestination, RuntimeError> {
        endpoint.validate_fixed_mutation()?;
        let _endpoint_roots = match context {
            Some(runtime) => RootedValueLease::for_values_iterator_in_context(
                endpoint.retained_values(),
                runtime,
            )?,
            None => RootedValueLease::for_traced(&endpoint)
                .map_err(|error| error.into_runtime_error())?,
        };
        self.control
            .push(context, &mut self.selector_roots, _endpoint_roots)?;
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
                crate::object::indexing::class_name_owner_from_base(&self.current).cloned()
            {
                self.object_suffix = Some(ObjectSuffixBuilder {
                    target: self.current.value().clone(),
                    expected_class,
                    structural_steps: self.control.vector(),
                    steps: self.control.vector(),
                });
            }
        }
        if let Some(mut suffix) = self.object_suffix {
            self.control.push(
                context,
                &mut suffix.steps,
                object_subscript_from_endpoint(endpoint.clone()),
            )?;
            let path_steps =
                crate::object::indexing::array_in_context(context, suffix.steps.len(), |index| {
                    Ok(suffix.steps[index].clone())
                })?;
            let path = ObjectSubscriptPath::from_storage(path_steps)?;
            let mut destination = self.control.boxed(
                context,
                PreparedObjectProtocolDestination {
                    prefix: self.steps,
                    expected_class: suffix.expected_class,
                    cardinality_method: None,
                    suffix_steps: suffix.structural_steps,
                    endpoint: endpoint.clone(),
                    path,
                    cardinality: 0,
                },
            )?;
            let cardinality = if matches!(&endpoint, SequenceEndpointSpec::Fixed { .. }) {
                // Ordinary assignment has already requested exactly one RHS.
                // Do not turn its variable-width selectors into dynamic output
                // cardinality or invoke numArgumentsFromSubscript for nargout.
                crate::builtins::introspection::object_indexing::PreparedObjectAssignmentCardinality {
                    count: 1, method: None,
                }
            } else {
                crate::builtins::introspection::object_indexing::object_path_assignment_cardinality_with_access(
                    context, &suffix.target, &destination.path, self.method_frame.as_deref(),
                ).await?
            };
            destination.cardinality = cardinality.count;
            destination.cardinality_method = cardinality.method;
            self.control.reserve(context, &mut self.selector_roots, 0)?;
            return Ok(PreparedSequenceDestination {
                kind: PreparedDestinationKind::ObjectProtocol(destination),
                _retained_roots: self.selector_roots,
                _source_roots: self.source_roots,
                method_frame: self.method_frame,
                frame_control: self.control,
            });
        }
        let endpoint = PreparedSequenceEndpoint::prepare_with_access(
            context,
            &self.current,
            endpoint,
            self.method_frame.as_deref(),
        )
        .await?;
        self.control.reserve(context, &mut self.selector_roots, 0)?;
        Ok(PreparedSequenceDestination {
            kind: PreparedDestinationKind::Structural {
                steps: self.steps,
                endpoint,
            },
            _retained_roots: self.selector_roots,
            _source_roots: self.source_roots,
            method_frame: self.method_frame,
            frame_control: self.control,
        })
    }
}

impl AssignmentStepSpec {
    pub(super) fn components(&self) -> &[crate::object::indexing::ObjectIndexComponent] {
        match self {
            Self::Member(_) => &[],
            Self::Parentheses { selectors } | Self::Braces(selectors) => selectors,
        }
    }
    fn retained_values(&self) -> impl Iterator<Item = &Value> + Clone {
        self.components()
            .iter()
            .filter_map(|component| match component {
                crate::object::indexing::ObjectIndexComponent::Value(value) => Some(value),
                crate::object::indexing::ObjectIndexComponent::Colon
                | crate::object::indexing::ObjectIndexComponent::End => None,
            })
    }
    pub fn member_in_context(
        context: &crate::context::RuntimeContext,
        name: &str,
    ) -> Result<Self, RuntimeError> {
        crate::call::argument_text::copy_str(name, Some(context)).map(Self::Member)
    }

    pub fn dynamic_member_in_context(
        context: &crate::context::RuntimeContext,
        name: &Value,
    ) -> Result<Self, RuntimeError> {
        crate::call::argument_text::convert(name, Some(context)).map(Self::Member)
    }
}

impl PreparedSequenceDestination {
    #[cfg(test)]
    fn rooted(
        kind: PreparedDestinationKind,
        source_roots: RootedValueLease,
    ) -> Result<Self, RuntimeError> {
        let control = DestinationControl::new(None)?;
        let mut retained_roots = control.vector();
        control.push(
            None,
            &mut retained_roots,
            RootedValueLease::for_traced(&kind).map_err(|error| error.into_runtime_error())?,
        )?;
        Ok(Self {
            kind,
            _retained_roots: retained_roots,
            _source_roots: source_roots,
            method_frame: None,
            frame_control: control,
        })
    }
}
