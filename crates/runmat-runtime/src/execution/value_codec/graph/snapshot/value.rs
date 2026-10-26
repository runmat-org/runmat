use super::*;
use runmat_execution::value::{
    CallableForm, CallableOwner, CallableValue, GraphCallable, GraphCallableOwner, GraphField,
    GraphRegistered, GraphRegisteredKind,
};

impl<'a> Encoder<'a> {
    pub(super) fn process_value(
        &mut self,
        input: Input<'a>,
        index: GraphNodeRef,
        work: &mut Vec<Task<'a>>,
    ) -> Result<(), ValueCodecError> {
        let binding = match &input {
            Input::BoundTarget(binding) => Some(*binding),
            Input::SelectedAnchor(binding) => Some(binding),
            Input::Borrowed(_) | Input::Owned(_) | Input::SharedCellChild(_) => None,
        };
        if let Some(binding) = binding {
            let origin = binding
                .authority
                .export_origin(&self.issuer.session_lease())
                .ok_or_else(|| invalid("bound target has no selected portable origin"))?;
            if let Some(lease) = binding.authority.local_definition() {
                if lease.function() != binding.function {
                    return Err(invalid(
                        "bound target differs from its local definition lease",
                    ));
                }
                self.bound_owners.push((origin, lease.clone()));
            }
            self.bound_authorities
                .push((index, origin, binding.authority.clone()));
            self.pending_bound_targets.push((index, origin));
            return Ok(());
        }
        if matches!(input.value(), Value::ScopedFunctionHandle(_)) {
            return Err(ValueCodecError::unsupported(
                "$",
                "invocation-scoped callable is not portable",
            ));
        }
        if let Value::HandleObject(handle) = input.value() {
            if !handle.valid {
                return Err(invalid("selected object handle is invalid"));
            }
            if let Some(metadata) =
                crate::builtins::introspection::dynamicprops::checked_metadata_owner(handle)
                    .map_err(invalid)?
            {
                let receiver = self.push_child_values(
                    vec![Input::Owned(Value::HandleObject(metadata.receiver))],
                    work,
                )?[0];
                self.nodes[index.0 as usize] = Some(GraphNode::SelectedDynamicPropertyMetadata(
                    runmat_execution::value::GraphSelectedDynamicPropertyMetadata {
                        receiver,
                        property: metadata.property,
                    },
                ));
                return Ok(());
            }
            let target = runmat_gc::gc_with_value(&handle.target, |value| match value {
                Value::Object(object)
                    if object.class_name() == &*handle.class_name
                        && object.selected_class().is_some() =>
                {
                    Ok(object.clone())
                }
                _ => Err(invalid("selected object handle has no exact class target")),
            })
            .map_err(|_| invalid("selected object handle target is unavailable"))??;
            return self.process_value(Input::Owned(Value::Object(target)), index, work);
        }
        let node = if let Value::Cell(cell) = input.value() {
            let shape = cell
                .shape
                .iter()
                .map(|extent| u64::try_from(*extent).map_err(|_| invalid("cell shape overflow")))
                .collect::<Result<_, _>>()?;
            let children = input.cell_children();
            let values = self.push_child_values(children, work)?;
            GraphNode::Cell { shape, values }
        } else if matches!(input.value(), Value::Struct(_)) {
            let fields = input.named_children();
            let names = fields
                .iter()
                .map(|(name, _)| name.clone())
                .collect::<Vec<_>>();
            let values =
                self.push_child_values(fields.into_iter().map(|(_, value)| value).collect(), work)?;
            GraphNode::Struct {
                fields: names
                    .into_iter()
                    .zip(values)
                    .map(|(name, value)| GraphField { name, value })
                    .collect(),
            }
        } else if let Value::ObjectArray(array) = input.value() {
            let class = array.class_name().clone();
            let mut kind = None;
            let shape = array
                .shape()
                .iter()
                .map(|extent| {
                    u64::try_from(*extent).map_err(|_| invalid("object-array shape overflow"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let empty_provenance = if array.is_empty() {
                array
                    .selected_provenance()
                    .iter()
                    .map(|facts| {
                        if facts.class() != &class {
                            return Err(invalid("empty object-array provenance changes class"));
                        }
                        let selected = if facts.is_handle_class() {
                            runmat_execution::value::GraphSelectedObjectKind::Handle
                        } else {
                            runmat_execution::value::GraphSelectedObjectKind::Value
                        };
                        if kind.is_some_and(|known| known != selected) {
                            return Err(invalid(
                                "empty object-array provenance changes storage kind",
                            ));
                        }
                        kind = Some(selected);
                        Ok(Input::SelectedAnchor(facts.constructor().clone()))
                    })
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                for value in array.data() {
                    let facts = crate::object::protocol::selected_graph_receiver_facts(value)
                        .map_err(|error| invalid(error.to_string()))?
                        .ok_or_else(|| {
                            invalid("selected object array contains an unselected value")
                        })?;
                    if facts.class() != &class {
                        return Err(invalid("selected object array changes class"));
                    }
                    let selected = if facts.is_handle_class() {
                        runmat_execution::value::GraphSelectedObjectKind::Handle
                    } else {
                        runmat_execution::value::GraphSelectedObjectKind::Value
                    };
                    if kind.is_some_and(|known| known != selected) {
                        return Err(invalid("selected object array changes storage kind"));
                    }
                    kind = Some(selected);
                }
                Vec::new()
            };
            if array.is_empty() && empty_provenance.is_empty() {
                return Err(ValueCodecError::unsupported(
                    "$",
                    "empty object array has no selected product provenance",
                ));
            }
            let includes_unselected = array.has_unselected_provenance();
            let kind =
                kind.ok_or_else(|| invalid("selected object array has no selected class"))?;
            let values = self.push_child_values(input.object_array_children(), work)?;
            let empty_provenance = self.push_child_values(empty_provenance, work)?;
            GraphNode::SelectedObjectArray(runmat_execution::value::GraphSelectedObjectArray {
                class,
                kind,
                shape,
                values,
                empty_provenance,
                includes_unselected,
            })
        } else if let Value::Object(object) = input.value() {
            if let Some(facts) = object.selected_class() {
                if object.has_native_payload()
                    || object.class_name() != facts.class()
                    || facts.is_handle_class() != self.selected_handle_nodes.contains(&index)
                {
                    return Err(invalid(
                        "selected object receiver differs from its class facts",
                    ));
                }
                let constructor = facts.constructor();
                let origin = constructor
                    .authority
                    .export_origin(&self.issuer.session_lease())
                    .ok_or_else(|| invalid("selected object constructor has no portable origin"))?;
                if let Some(lease) = constructor.authority.local_definition() {
                    if lease.function() != constructor.function {
                        return Err(invalid(
                            "selected object constructor lease differs from its body",
                        ));
                    }
                    self.bound_owners.push((origin, lease.clone()));
                }
                self.bound_authorities
                    .push((index, origin, constructor.authority.clone()));
                let mut sorted_dynamic = object
                    .dynamic_property_definitions()
                    .map(|(_, definition)| definition)
                    .collect::<Vec<_>>();
                sorted_dynamic.sort_unstable_by(|left, right| left.name.cmp(&right.name));
                let dynamic_properties = sorted_dynamic
                    .iter()
                    .map(|property| runmat_execution::value::DurableDynamicProperty {
                        name: property.name.to_string(),
                        definition_name: property.name.to_string(),
                        defining_class: property.defining_class.as_identity().clone(),
                        get_access: property.get_access,
                        set_access: property.set_access,
                        dependent: property.dependent,
                        hidden: property.hidden,
                        transient: property.transient,
                        non_copyable: property.non_copyable,
                        abort_set: property.abort_set,
                        set_observable: property.set_observable,
                        get_observable: property.get_observable,
                        description: property.description.to_string(),
                    })
                    .collect::<Vec<_>>();
                let metadata_inputs = sorted_dynamic
                    .iter()
                    .filter_map(|property| property.metadata_handle)
                    .map(|target| {
                        Input::Owned(Value::HandleObject(runmat_value::HandleRef {
                            class_name:
                                crate::builtins::introspection::dynamicprops::DYNAMIC_PROPERTY_CLASS
                                    .into(),
                            target,
                            valid: true,
                        }))
                    })
                    .collect::<Vec<_>>();
                let dynamic_metadata = self.push_child_values(metadata_inputs, work)?;
                let class = facts.class().clone();
                let fields = input.named_children();
                let names = fields
                    .iter()
                    .map(|(name, _)| name.clone())
                    .collect::<Vec<_>>();
                let values = self.push_child_values(
                    fields.into_iter().map(|(_, value)| value).collect(),
                    work,
                )?;
                self.pending_selected_objects.push(PreparedSelectedObject {
                    index,
                    origin,
                    class,
                    kind: if self.selected_handle_nodes.contains(&index) {
                        runmat_execution::value::GraphSelectedObjectKind::Handle
                    } else {
                        runmat_execution::value::GraphSelectedObjectKind::Value
                    },
                    fields: names
                        .into_iter()
                        .zip(values)
                        .map(|(name, value)| GraphField { name, value })
                        .collect(),
                    dynamic_properties,
                    dynamic_metadata,
                });
                return Ok(());
            }
            let codec = super::super::super::immutable_class::GraphValueClass::for_class(
                object.class_name(),
            )?
            .ok_or_else(|| {
                ValueCodecError::unsupported("$", "value has no portable graph codec")
            })?;
            codec.validate(object, self.workspace.as_ref())?;
            let type_identity = codec.type_identity().to_owned();
            let fields = input.named_children();
            let names = fields
                .iter()
                .map(|(name, _)| name.clone())
                .collect::<Vec<_>>();
            let values =
                self.push_child_values(fields.into_iter().map(|(_, value)| value).collect(), work)?;
            GraphNode::Registered(GraphRegistered {
                kind: GraphRegisteredKind::ImmutableValueClass,
                type_identity,
                schema_version: codec.schema_version(),
                fields: names
                    .into_iter()
                    .zip(values)
                    .map(|(name, value)| GraphField { name, value })
                    .collect(),
            })
        } else if let Value::Closure(closure) = input.value() {
            let owner = match &closure.binding {
                Some(binding) => {
                    let origin = binding
                        .authority
                        .export_origin(&self.issuer.session_lease())
                        .ok_or_else(|| invalid("closure has no selected portable origin"))?;
                    if let Some(lease) = binding.authority.local_definition() {
                        self.bound_owners.push((origin, lease.clone()));
                    }
                    self.bound_authorities
                        .push((index, origin, binding.authority.clone()));
                    CallableOwner::Bound(origin)
                }
                None => CallableOwner::Workspace,
            };
            let name = closure.function_name.to_string();
            let role = closure.source_role;
            let source_file = closure.source_file.as_deref().map(str::to_owned);
            let environment = closure.lexical_environment.clone();
            let captures = self.push_child_values(input.capture_children(), work)?;
            let activation = match (owner, environment) {
                (CallableOwner::Bound(_), Some(environment)) => {
                    Some(self.push_activation(environment, work)?)
                }
                (CallableOwner::Workspace, None) => None,
                (_, Some(_)) => {
                    return Err(invalid("unbound closure carries a lexical activation"));
                }
                (_, None) => None,
            };
            let callable_digest = CallableValue::identity_digest(
                owner,
                &name,
                CallableForm::Closure,
                role,
                source_file.as_deref(),
            );
            if let CallableOwner::Bound(origin) = owner {
                self.pending_bound_callables.push(PreparedBoundCallable {
                    index,
                    origin,
                    qualified_name: name,
                    callable_digest,
                    form: CallableForm::Closure,
                    source_role: role,
                    source_file,
                    captures,
                    activation,
                });
                return Ok(());
            }
            GraphNode::Callable(GraphCallable {
                owner: GraphCallableOwner::Workspace,
                qualified_name: name,
                callable_digest,
                form: CallableForm::Closure,
                source_role: role,
                source_file,
                captures,
                activation,
            })
        } else {
            match input.value() {
                Value::ScopedFunctionHandle(_) => {
                    return Err(ValueCodecError::unsupported(
                        "$",
                        "invocation-scoped callables cannot leave their original execution lane",
                    ));
                }
                Value::FunctionHandle(name) => self.handle(
                    GraphCallableOwner::Workspace,
                    name,
                    None,
                    runmat_types::FunctionSourceRole::Unknown,
                ),
                Value::ExternalFunctionHandle(name) => self.handle(
                    GraphCallableOwner::External,
                    name,
                    None,
                    runmat_types::FunctionSourceRole::Unknown,
                ),
                Value::MethodFunctionHandle(name) => self.handle(
                    GraphCallableOwner::Method,
                    name,
                    None,
                    runmat_types::FunctionSourceRole::Method,
                ),
                Value::BoundFunctionHandle(handle) => {
                    let origin = handle
                        .authority
                        .export_origin(&self.issuer.session_lease())
                        .ok_or_else(|| invalid("bound callable has no selected portable origin"))?;
                    if let Some(lease) = handle.authority.local_definition() {
                        self.bound_owners.push((origin, lease.clone()));
                    }
                    self.bound_authorities
                        .push((index, origin, handle.authority.clone()));
                    self.pending_bound_callables.push(PreparedBoundCallable {
                        index,
                        origin,
                        qualified_name: handle.name.to_string(),
                        callable_digest: CallableValue::identity_digest(
                            CallableOwner::Bound(origin),
                            &handle.name,
                            CallableForm::Handle,
                            handle.source_role,
                            handle.source_file.as_deref(),
                        ),
                        form: CallableForm::Handle,
                        source_role: handle.source_role,
                        source_file: handle.source_file.as_deref().map(str::to_owned),
                        captures: Vec::new(),
                        activation: None,
                    });
                    return Ok(());
                }
                Value::Int(_)
                | Value::Num(_)
                | Value::Complex(_, _)
                | Value::Bool(_)
                | Value::LogicalArray(_)
                | Value::String(_)
                | Value::StringArray(_)
                | Value::CharArray(_)
                | Value::Tensor(_)
                | Value::SparseTensor(_)
                | Value::ComplexTensor(_)
                | Value::MException(_)
                | Value::Distributed(_)
                | Value::Composite(_) => {
                    self.pending_leaves.push((index, input));
                    return Ok(());
                }
                Value::OutputList(_) => {
                    return Err(ValueCodecError::transient_not_portable("$"));
                }
                Value::Symbolic(_)
                | Value::SymbolicArray(_)
                | Value::StructArray(_)
                | Value::GpuTensor(_)
                | Value::ObjectArray(_)
                | Value::HandleObject(_)
                | Value::Listener(_)
                | Value::ClassRef(_)
                | Value::Future(_)
                | Value::Task(_)
                | Value::Pool(_)
                | Value::Job(_)
                | Value::Foreign(_) => {
                    return Err(ValueCodecError::unsupported(
                        "$",
                        "value has no portable graph codec",
                    ));
                }
                Value::Cell(_) | Value::Struct(_) | Value::Object(_) | Value::Closure(_) => {
                    unreachable!("compound value handled above")
                }
            }
        };
        self.nodes[index.0 as usize] = Some(node);
        Ok(())
    }

    fn handle(
        &self,
        owner: GraphCallableOwner,
        name: &str,
        source_file: Option<String>,
        source_role: runmat_types::FunctionSourceRole,
    ) -> GraphNode {
        GraphNode::Callable(GraphCallable {
            owner,
            qualified_name: name.to_owned(),
            callable_digest: CallableValue::identity_digest(
                owner.portable_owner(),
                name,
                CallableForm::Handle,
                source_role,
                source_file.as_deref(),
            ),
            form: CallableForm::Handle,
            source_role,
            source_file,
            captures: Vec::new(),
            activation: None,
        })
    }
}
