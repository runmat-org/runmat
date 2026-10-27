use std::collections::HashMap;

use runmat_execution::value::{
    CallableForm, GraphBoundOwner, GraphCallableOwner, GraphLexicalLayout, GraphNode, GraphNodeRef,
    GraphRegisteredKind, GraphSelectedObjectKind, PayloadGraph, ValueLimits,
};
use runmat_execution::ExecutableSourceMapV2;
use runmat_gc::ExplicitRoot;
use runmat_value::{BoundFunctionBinding, CellArray, Closure, LexicalEnvironment, Value};

use super::{invalid, validate_bound_callable_proofs, GraphCallableProof, ValueCodecError};

pub(super) fn decode(
    graph: &PayloadGraph,
    runtime: &crate::context::RuntimeContext,
    source_map: &ExecutableSourceMapV2,
    source_issuer: Option<&super::GraphSourceIssuer>,
    layouts: &HashMap<GraphBoundOwner, GraphLexicalLayout>,
    proofs: &HashMap<GraphBoundOwner, GraphCallableProof>,
    bindings: &HashMap<GraphBoundOwner, BoundFunctionBinding>,
    selected_classes: &HashMap<
        GraphBoundOwner,
        std::rc::Rc<runmat_value::SelectedObjectClassFacts>,
    >,
    limits: ValueLimits,
) -> Result<crate::execution::RootedValues, ValueCodecError> {
    if let Some(issuer) = source_issuer {
        issuer.ensure_live()?;
    }
    let value_order = graph
        .validate_for_program_and_value_order(
            limits,
            Some(source_map),
            layouts,
            source_issuer.map_or(1, super::GraphSourceIssuer::selected_unit_count),
        )
        .map_err(|error| invalid(error.to_string()))?;
    validate_bound_callable_proofs(graph, proofs)?;
    for node in &graph.nodes {
        if let GraphNode::SelectedObject(object) = node {
            let binding = bindings
                .get(&object.constructor)
                .ok_or_else(|| invalid("selected object constructor has no admitted definition"))?;
            let facts = selected_classes.get(&object.constructor).ok_or_else(|| {
                invalid("selected object constructor has no admitted class product")
            })?;
            if facts.class() != &object.class
                || facts.constructor() != binding
                || facts.is_handle_class() != matches!(object.kind, GraphSelectedObjectKind::Handle)
                || binding.authority.export_origin(&runtime.session_lease())
                    != Some(object.constructor.origin)
            {
                return Err(invalid(
                    "selected object differs from its admitted constructor/class",
                ));
            }
        }
        let GraphNode::Callable(callable) = node else {
            continue;
        };
        let GraphCallableOwner::Bound(owner) = callable.owner else {
            continue;
        };
        let binding = bindings
            .get(&owner)
            .ok_or_else(|| invalid("bound unit owner is absent from selected definitions"))?;
        if binding.authority.export_origin(&runtime.session_lease()) != Some(owner.origin)
            || binding
                .authority
                .local_definition()
                .is_some_and(|lease| lease.function() != binding.function)
        {
            return Err(invalid(
                "bound binding differs from its selected function origin",
            ));
        }
    }
    let remaining_uses = value_use_counts(graph)?;
    let mut decoder = Decoder {
        runtime,
        graph,
        bindings,
        proofs,
        selected_classes,
        limits,
        source_issuer,
        environments: vec![None; graph.nodes.len()],
        environment_roots: Vec::new(),
        values: (0..graph.nodes.len()).map(|_| None).collect(),
        selected_handles: vec![None; graph.nodes.len()],
        selected_handle_roots: Vec::new(),
        selected_metadata: vec![None; graph.nodes.len()],
        selected_metadata_roots: Vec::new(),
        remaining_uses,
        visiting: vec![false; graph.nodes.len()],
    };
    decoder.allocate_selected_handle_shells()?;
    decoder.allocate_selected_metadata_shells()?;
    for (index, node) in graph.nodes.iter().enumerate() {
        if matches!(node, GraphNode::Activation(_)) {
            decoder.activation(index)?;
        }
    }
    // Allocate values bottom-up. `value` below only sees already-rooted
    // children, so even the portable maximum depth never becomes Rust stack
    // depth. Activation edges are separate: their shells were rooted above.
    for reference in value_order {
        decoder.build_value(reference, 0)?;
    }
    decoder.fill_selected_handle_shells()?;
    decoder.fill_selected_metadata_shells()?;
    for root in &graph.roots {
        decoder.build_value(*root, 0)?;
    }
    for (index, node) in graph.nodes.iter().enumerate() {
        let GraphNode::Activation(activation) = node else {
            continue;
        };
        let environment = decoder.environments[index]
            .as_ref()
            .ok_or_else(|| invalid("allocated activation is missing"))?
            .clone();
        for binding in &activation.bindings {
            if let Some(reference) = binding.value {
                let mut leases = Vec::new();
                let value = decoder.value(reference, 0, &mut leases)?;
                crate::call::lexical::write_owned_binding(&environment, binding.binding, value)
                    .map_err(|error| invalid(error.to_string()))?;
                drop(leases);
            }
        }
    }
    let mut roots = crate::execution::RootedValuesBuilder::with_capacity(graph.roots.len());
    for root in &graph.roots {
        let mut leases = Vec::new();
        let value = decoder.value(*root, 0, &mut leases)?;
        roots
            .push(value)
            .map_err(|error| invalid(error.to_string()))?;
        drop(leases);
    }
    let rooted = roots.finish().map_err(|error| invalid(error.to_string()))?;
    drop(decoder);
    Ok(rooted)
}

struct Decoder<'a> {
    runtime: &'a crate::context::RuntimeContext,
    graph: &'a PayloadGraph,
    bindings: &'a HashMap<GraphBoundOwner, BoundFunctionBinding>,
    proofs: &'a HashMap<GraphBoundOwner, GraphCallableProof>,
    selected_classes:
        &'a HashMap<GraphBoundOwner, std::rc::Rc<runmat_value::SelectedObjectClassFacts>>,
    limits: ValueLimits,
    source_issuer: Option<&'a super::GraphSourceIssuer>,
    environments: Vec<Option<LexicalEnvironment>>,
    environment_roots: Vec<ExplicitRoot>,
    values: Vec<Option<crate::execution::RootedValue>>,
    selected_handles: Vec<Option<runmat_gc::GcHandle>>,
    selected_handle_roots: Vec<ExplicitRoot>,
    selected_metadata: Vec<Option<runmat_gc::GcHandle>>,
    selected_metadata_roots: Vec<ExplicitRoot>,
    remaining_uses: Vec<usize>,
    visiting: Vec<bool>,
}

fn value_use_counts(graph: &PayloadGraph) -> Result<Vec<usize>, ValueCodecError> {
    let mut counts = vec![0usize; graph.nodes.len()];
    let mut count = |reference: GraphNodeRef| -> Result<(), ValueCodecError> {
        let use_count = &mut counts[reference.0 as usize];
        *use_count = use_count
            .checked_add(1)
            .ok_or_else(|| invalid("value reference count overflows"))?;
        Ok(())
    };
    for root in &graph.roots {
        count(*root)?;
    }
    for node in &graph.nodes {
        match node {
            GraphNode::Leaf { .. } => {}
            GraphNode::Cell { values, .. } => {
                for value in values {
                    count(*value)?;
                }
            }
            GraphNode::Struct { fields } => {
                for field in fields {
                    count(field.value)?;
                }
            }
            GraphNode::Registered(value) => {
                for field in &value.fields {
                    count(field.value)?;
                }
            }
            GraphNode::SelectedObject(value) => {
                for field in &value.fields {
                    count(field.value)?;
                }
            }
            GraphNode::SelectedObjectArray(value) => {
                for reference in value.values.iter().chain(&value.empty_provenance) {
                    count(*reference)?;
                }
            }
            GraphNode::SelectedDynamicPropertyMetadata(_) => {}
            GraphNode::Callable(value) => {
                for capture in &value.captures {
                    count(*capture)?;
                }
            }
            GraphNode::Activation(value) => {
                for binding in &value.bindings {
                    if let Some(reference) = binding.value {
                        count(reference)?;
                    }
                }
            }
        }
    }
    Ok(counts)
}

fn selected_dynamic_properties(
    output: &mut super::super::object_output::ObjectOutput<'_>,
    selected: &runmat_execution::value::GraphSelectedObject,
) -> Result<(), ValueCodecError> {
    for property in &selected.dynamic_properties {
        if property.name != property.definition_name {
            return Err(invalid("selected dynamic property names disagree"));
        }
        if output.insert_dynamic_source(property, None, "$")? {
            return Err(invalid("selected dynamic property repeats"));
        }
    }
    Ok(())
}

impl<'a> Decoder<'a> {
    fn object_output(
        &self,
        class: &str,
        fields: &[runmat_execution::value::GraphField],
        dynamic: &[runmat_execution::value::DurableDynamicProperty],
    ) -> Result<super::super::object_output::ObjectOutput<'a>, ValueCodecError> {
        super::super::object_output::ObjectOutput::new(
            class,
            fields.iter().map(|f| f.name.as_str()),
            dynamic,
            super::super::object_output::DynamicMetadataCopy::Synchronous,
            Some(self.runtime),
            "$",
        )
    }
    fn allocate_selected_metadata_shells(&mut self) -> Result<(), ValueCodecError> {
        for (index, node) in self.graph.nodes.iter().enumerate() {
            if !matches!(node, GraphNode::SelectedDynamicPropertyMetadata(_)) {
                continue;
            }
            let class = crate::builtins::introspection::dynamicprops::DYNAMIC_PROPERTY_CLASS;
            let object = self
                .object_output(class.display_name(), &[], &[])?
                .finish("$")?;
            let class_name = object.class_name_owner().clone();
            let root = runmat_gc::gc_allocate_rooted(Value::Object(object))
                .map_err(|error| invalid(error.to_string()))?;
            let value = Value::HandleObject(runmat_value::HandleRef {
                class_name,
                target: root.handle(),
                valid: true,
            });
            self.selected_metadata[index] = Some(root.handle());
            self.values[index] = Some(
                crate::execution::RootedValue::in_context(value, self.runtime)
                    .map_err(|error| ValueCodecError::runtime("$", error))?,
            );
            self.selected_metadata_roots.push(root);
        }
        Ok(())
    }

    fn allocate_selected_handle_shells(&mut self) -> Result<(), ValueCodecError> {
        for (index, node) in self.graph.nodes.iter().enumerate() {
            let GraphNode::SelectedObject(selected) = node else {
                continue;
            };
            if selected.kind != GraphSelectedObjectKind::Handle {
                continue;
            }
            let facts = std::rc::Rc::clone(
                self.selected_classes
                    .get(&selected.constructor)
                    .ok_or_else(|| invalid("selected handle class product is absent"))?,
            );
            let mut output = self.object_output(selected.class.display_name(), &[], &[])?;
            output.selected_class(facts, "$")?;
            let object = output.finish("$")?;
            let class_name = object.class_name_owner().clone();
            let root = runmat_gc::gc_allocate_rooted(Value::Object(object))
                .map_err(|error| invalid(error.to_string()))?;
            let value = Value::HandleObject(runmat_value::HandleRef {
                class_name,
                target: root.handle(),
                valid: true,
            });
            self.selected_handles[index] = Some(root.handle());
            self.values[index] = Some(
                crate::execution::RootedValue::in_context(value, self.runtime)
                    .map_err(|error| ValueCodecError::runtime("$", error))?,
            );
            self.selected_handle_roots.push(root);
        }
        Ok(())
    }

    fn fill_selected_handle_shells(&mut self) -> Result<(), ValueCodecError> {
        for index in 0..self.graph.nodes.len() {
            let GraphNode::SelectedObject(selected) = &self.graph.nodes[index] else {
                continue;
            };
            if selected.kind != GraphSelectedObjectKind::Handle {
                continue;
            }
            let handle = self.selected_handles[index]
                .ok_or_else(|| invalid("selected handle shell is absent"))?;
            let mut leases = Vec::new();
            let mut output = self.object_output(
                selected.class.display_name(),
                &selected.fields,
                &selected.dynamic_properties,
            )?;
            output.selected_class(
                std::rc::Rc::clone(
                    self.selected_classes
                        .get(&selected.constructor)
                        .ok_or_else(|| invalid("selected handle class product is absent"))?,
                ),
                "$",
            )?;
            for field in &selected.fields {
                let key = output.copy_text(&field.name, "$")?;
                if output.insert(key, self.value(field.value, 1, &mut leases)?, "$")? {
                    return Err(invalid("selected handle field repeats"));
                }
            }
            selected_dynamic_properties(&mut output, selected)?;
            for reference in &selected.dynamic_metadata {
                let GraphNode::SelectedDynamicPropertyMetadata(metadata) =
                    &self.graph.nodes[reference.0 as usize]
                else {
                    return Err(invalid("selected metadata changed after validation"));
                };
                let metadata_handle = self.selected_metadata[reference.0 as usize]
                    .ok_or_else(|| invalid("selected metadata shell is absent"))?;
                output.metadata_handle(&metadata.property, metadata_handle, "$")?;
                runmat_gc::gc_record_handle_write(
                    &handle,
                    &Value::HandleObject(runmat_value::HandleRef {
                        class_name:
                            crate::builtins::introspection::dynamicprops::DYNAMIC_PROPERTY_CLASS
                                .into(),
                        target: metadata_handle,
                        valid: true,
                    }),
                );
            }
            let destination = output.finish("$")?;
            runmat_gc::gc_with_value_mut(&handle, |stored| match stored {
                Value::Object(object) if object.class_name() == &selected.class => {
                    *object = destination;
                    Ok(())
                }
                _ => Err(invalid("selected handle shell changed before field fill")),
            })
            .map_err(|error| invalid(error.to_string()))??;
            drop(leases);
        }
        Ok(())
    }

    fn fill_selected_metadata_shells(&mut self) -> Result<(), ValueCodecError> {
        for (index, node) in self.graph.nodes.iter().enumerate() {
            let GraphNode::SelectedDynamicPropertyMetadata(metadata) = node else {
                continue;
            };
            let metadata_handle = self.selected_metadata[index]
                .ok_or_else(|| invalid("selected metadata shell is absent"))?;
            let receiver = self.selected_handles[metadata.receiver.0 as usize]
                .ok_or_else(|| invalid("selected metadata receiver shell is absent"))?;
            let receiver_class = runmat_gc::gc_with_value(&receiver, |stored| match stored {
                Value::Object(object) => Some(object.class_name_owner().clone()),
                _ => None,
            })
            .map_err(|error| invalid(error.to_string()))?
            .ok_or_else(|| invalid("selected metadata receiver changed after validation"))?;
            let definition = runmat_gc::gc_with_value(&receiver, |stored| {
                let Value::Object(object) = stored else {
                    return None;
                };
                object.dynamic_property(&metadata.property).cloned()
            })
            .map_err(|error| invalid(error.to_string()))?
            .ok_or_else(|| invalid("selected metadata definition is absent"))?;
            if definition.metadata_handle != Some(metadata_handle) {
                return Err(invalid("selected metadata differs from its receiver"));
            }
            let object = crate::builtins::introspection::dynamicprops::selected_metadata_object(
                &definition,
                runmat_value::HandleRef {
                    class_name: receiver_class,
                    target: receiver,
                    valid: true,
                },
            );
            runmat_gc::gc_record_handle_write(
                &metadata_handle,
                &Value::HandleObject(runmat_value::HandleRef {
                    class_name: object.class_name_owner().clone(),
                    target: receiver,
                    valid: true,
                }),
            );
            runmat_gc::gc_with_value_mut(&metadata_handle, |stored| {
                match stored {
                Value::Object(existing)
                    if existing.class_name()
                        == &crate::builtins::introspection::dynamicprops::DYNAMIC_PROPERTY_CLASS
                            .owned() =>
                {
                    *existing = object;
                    Ok(())
                }
                _ => Err(invalid("selected metadata shell changed before fill")),
            }
            })
            .map_err(|error| invalid(error.to_string()))??;
        }
        Ok(())
    }

    fn child_depth(&self, depth: u16) -> Result<u16, ValueCodecError> {
        depth
            .checked_add(1)
            .filter(|next| *next <= self.limits.max_depth)
            .ok_or_else(|| invalid("value construction exceeds its depth limit"))
    }

    fn activation(&mut self, index: usize) -> Result<LexicalEnvironment, ValueCodecError> {
        if let Some(environment) = &self.environments[index] {
            return Ok(environment.clone());
        }
        let mut path = Vec::new();
        let mut cursor = Some(index);
        while let Some(current) = cursor {
            if self.environments[current].is_some() {
                break;
            }
            let GraphNode::Activation(activation) = &self.graph.nodes[current] else {
                return Err(invalid("activation reference has the wrong node kind"));
            };
            path.push(current);
            cursor = activation.parent.map(|parent| parent.0 as usize);
        }
        for current in path.into_iter().rev() {
            let GraphNode::Activation(activation) = &self.graph.nodes[current] else {
                unreachable!("validated activation path")
            };
            let parent = activation
                .parent
                .map(|reference| {
                    self.environments[reference.0 as usize]
                        .as_ref()
                        .cloned()
                        .ok_or_else(|| invalid("activation parent was not allocated"))
                })
                .transpose()?;
            let (environment, roots) = crate::call::lexical::allocate_environment(parent)
                .map_err(|error| ValueCodecError::runtime("$", error))?;
            self.environment_roots.extend(roots);
            for binding in &activation.bindings {
                if !crate::call::lexical::reserve_owned_binding(&environment, binding.binding)
                    .map_err(|error| invalid(error.to_string()))?
                {
                    return Err(invalid("duplicate reconstructed lexical binding"));
                }
            }
            self.environments[current] = Some(environment);
        }
        self.environments[index]
            .as_ref()
            .cloned()
            .ok_or_else(|| invalid("activation was not allocated"))
    }

    fn value(
        &mut self,
        reference: GraphNodeRef,
        depth: u16,
        leases: &mut Vec<crate::execution::RootedValueLease>,
    ) -> Result<Value, ValueCodecError> {
        let index = reference.0 as usize;
        if self.remaining_uses[index] == 0 {
            return Err(invalid("reconstructed value was consumed too many times"));
        }
        self.build_value(reference, depth)?;
        let use_count = &mut self.remaining_uses[index];
        *use_count -= 1;
        if *use_count == 0 {
            let rooted = self.values[index]
                .take()
                .ok_or_else(|| invalid("reconstructed value is missing"))?;
            let (value, lease) = rooted.into_value_and_lease();
            leases.push(lease);
            Ok(value)
        } else {
            self.values[index]
                .as_ref()
                .map(|rooted| rooted.value().clone())
                .ok_or_else(|| invalid("reconstructed value is missing"))
        }
    }

    fn build_value(&mut self, reference: GraphNodeRef, depth: u16) -> Result<(), ValueCodecError> {
        if depth > self.limits.max_depth {
            return Err(invalid("value construction exceeds its depth limit"));
        }
        let index = reference.0 as usize;
        // A selected handle or metadata shell may already have been consumed
        // by its final incoming value edge before the topological construction
        // walk reaches this node. Its separately rooted GC shell remains the
        // one exact object to fill; never try to construct a second value.
        if self.values[index].is_some()
            || self.selected_handles[index].is_some()
            || self.selected_metadata[index].is_some()
        {
            return Ok(());
        }
        if std::mem::replace(&mut self.visiting[index], true) {
            return Err(invalid("value-only cycle cannot be reconstructed"));
        }
        if let GraphNode::Leaf { value } = &self.graph.nodes[index] {
            let rooted = match self.source_issuer {
                Some(issuer) => super::super::decode::decode_inline_value_owned_with_frame_sources(
                    value,
                    self.limits,
                    issuer,
                ),
                None => {
                    super::super::decode::decode_inline_value_owned_with_limits(value, self.limits)
                }
            }
            .map_err(|error| invalid(error.to_string()))?;
            self.values[index] = Some(rooted);
            self.visiting[index] = false;
            return Ok(());
        }
        let graph = self.graph;
        let node = &graph.nodes[index];
        let child_depth = self.child_depth(depth).ok();
        let mut child_leases = Vec::new();
        let value = match node {
            GraphNode::Leaf { .. } => unreachable!("leaf returned with its root"),
            GraphNode::Cell { shape, values } => {
                let data = values
                    .iter()
                    .map(|value| {
                        self.value(
                            *value,
                            child_depth.ok_or_else(|| {
                                invalid("value construction exceeds its depth limit")
                            })?,
                            &mut child_leases,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let shape = shape
                    .iter()
                    .map(|extent| {
                        usize::try_from(*extent)
                            .map_err(|_| invalid("cell shape is not host-addressable"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Value::Cell(
                    CellArray::new_with_shape(data, shape)
                        .map_err(|error| invalid(error.to_string()))?,
                )
            }
            GraphNode::Struct { fields } => {
                for field in fields {
                    self.build_value(
                        field.value,
                        child_depth
                            .ok_or_else(|| invalid("value construction exceeds its depth limit"))?,
                    )?;
                    if self.remaining_uses[field.value.0 as usize] == 0
                        || self.values[field.value.0 as usize].is_none()
                    {
                        return Err(invalid("reconstructed structure field is unavailable"));
                    }
                }
                let output = crate::execution::construct_struct_from_fields_in_context(
                    self.runtime,
                    fields.iter().map(|field| {
                        (
                            field.name.as_str(),
                            self.values[field.value.0 as usize]
                                .as_ref()
                                .expect("validated field owner")
                                .value(),
                        )
                    }),
                )
                .map_err(|error| ValueCodecError::runtime("$", error))?;
                for field in fields {
                    // Retire original cache uses only after final map/root
                    // publication. Child clones retain original payload owners.
                    drop(self.value(
                        field.value,
                        child_depth.expect("validated field depth"),
                        &mut child_leases,
                    )?);
                }
                let (value, lease) = output.into_value_and_lease();
                child_leases.push(lease);
                value
            }
            GraphNode::Registered(registered) => {
                if registered.kind != GraphRegisteredKind::ImmutableValueClass {
                    return Err(invalid("registered graph class has no runtime codec"));
                }
                let codec = super::super::immutable_class::GraphValueClass::for_wire(
                    &registered.type_identity,
                    registered.schema_version,
                )?;
                let mut output = self.object_output(codec.class_name(), &registered.fields, &[])?;
                for field in &registered.fields {
                    output.insert(
                        output.copy_text(&field.name, "$")?,
                        self.value(
                            field.value,
                            child_depth.ok_or_else(|| {
                                invalid("value construction exceeds its depth limit")
                            })?,
                            &mut child_leases,
                        )?,
                        "$",
                    )?;
                }
                let object = output.finish("$")?;
                codec.validate(
                    &object,
                    Some(&super::super::ValidationWorkspaceOwner::capture(
                        self.runtime,
                    )),
                )?;
                Value::Object(object)
            }
            GraphNode::SelectedObject(selected) => {
                if selected.kind != GraphSelectedObjectKind::Value {
                    return Err(invalid("selected handle was not preallocated"));
                }
                let facts = std::rc::Rc::clone(
                    self.selected_classes
                        .get(&selected.constructor)
                        .ok_or_else(|| invalid("selected object class product is absent"))?,
                );
                let mut output = self.object_output(
                    selected.class.display_name(),
                    &selected.fields,
                    &selected.dynamic_properties,
                )?;
                output.selected_class(facts, "$")?;
                for field in &selected.fields {
                    output.insert(
                        output.copy_text(&field.name, "$")?,
                        self.value(
                            field.value,
                            child_depth.ok_or_else(|| {
                                invalid("value construction exceeds its depth limit")
                            })?,
                            &mut child_leases,
                        )?,
                        "$",
                    )?;
                }
                selected_dynamic_properties(&mut output, selected)?;
                let object = output.finish("$")?;
                Value::Object(object)
            }
            GraphNode::SelectedObjectArray(selected) => {
                let mut shape = super::super::shape_copy::ShapeCopy::new(
                    &selected.shape,
                    Some(self.runtime),
                    "$",
                )?;
                shape.advance(usize::MAX, "$")?;
                let shape = shape.finish("$")?;
                let mut output=crate::execution::object_array::ObjectArrayConstruction::with_capacity_retaining_shape(
                    Some(self.runtime),crate::execution::object_array::Class::Name(selected.class.display_name()),
                    selected.values.len(),&shape,
                    if selected.values.is_empty(){selected.empty_provenance.len()}else{0},
                    selected.includes_unselected,
                ).map_err(|error|ValueCodecError::runtime("$",error))?;
                for reference in &selected.values {
                    output
                        .push(self.value(
                            *reference,
                            child_depth.ok_or_else(|| {
                                invalid("value construction exceeds its depth limit")
                            })?,
                            &mut child_leases,
                        )?)
                        .map_err(|error| ValueCodecError::runtime("$", error))?;
                }
                if selected.values.is_empty() {
                    for reference in &selected.empty_provenance {
                        let GraphNode::Callable(anchor) = &self.graph.nodes[reference.0 as usize]
                        else {
                            return Err(invalid("selected array anchor changed after validation"));
                        };
                        let GraphCallableOwner::Bound(owner) = anchor.owner else {
                            return Err(invalid("selected array anchor has no selected owner"));
                        };
                        let facts = std::rc::Rc::clone(
                            self.selected_classes.get(&owner).ok_or_else(|| {
                                invalid("selected array provenance lacks an admitted class product")
                            })?,
                        );
                        if facts.class() != &selected.class
                            || facts.is_handle_class()
                                != (selected.kind == GraphSelectedObjectKind::Handle)
                        {
                            return Err(invalid("selected array provenance changes class or kind"));
                        }
                        let value = self.value(
                            *reference,
                            child_depth.ok_or_else(|| {
                                invalid("value construction exceeds its depth limit")
                            })?,
                            &mut child_leases,
                        )?;
                        let Value::BoundFunctionHandle(handle) = value else {
                            return Err(invalid("selected array anchor is not a bound handle"));
                        };
                        if handle.function != facts.constructor().function
                            || handle.authority != facts.constructor().authority
                        {
                            return Err(invalid("selected array anchor differs from its class"));
                        }
                        output
                            .push_provenance(facts)
                            .map_err(|error| ValueCodecError::runtime("$", error))?;
                    }
                }
                output
                    .finish_value()
                    .map_err(|error| ValueCodecError::runtime("$", error))?
            }
            GraphNode::SelectedDynamicPropertyMetadata(_) => {
                return Err(invalid("selected metadata was not preallocated"));
            }
            GraphNode::Callable(callable) => {
                let source_file = match callable.owner {
                    GraphCallableOwner::Bound(origin) => self
                        .proofs
                        .get(&origin)
                        .ok_or_else(|| invalid("bound callable has no selected source proof"))?
                        .issued_source_file
                        .clone(),
                    _ => callable.source_file.clone(),
                };
                let captures = callable
                    .captures
                    .iter()
                    .map(|child| {
                        self.value(
                            *child,
                            child_depth.ok_or_else(|| {
                                invalid("value construction exceeds its depth limit")
                            })?,
                            &mut child_leases,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let binding = match callable.owner {
                    GraphCallableOwner::Bound(origin) => {
                        Some(self.bindings.get(&origin).cloned().ok_or_else(|| {
                            invalid("bound unit owner is absent from selected definitions")
                        })?)
                    }
                    _ => None,
                };
                let environment = callable
                    .activation
                    .map(|reference| {
                        self.environments[reference.0 as usize]
                            .as_ref()
                            .cloned()
                            .ok_or_else(|| invalid("callable activation was not allocated"))
                    })
                    .transpose()?;
                match callable.form {
                    CallableForm::Closure => {
                        Value::Closure(Closure::from_data(runmat_value::ClosureData {
                            function_name: callable.qualified_name.clone().into(),
                            binding,
                            source_role: callable.source_role,
                            source_file: source_file.clone().map(runmat_value::Utf8Text::new),
                            captures: captures.into(),
                            lexical_environment: environment,
                        }))
                    }
                    CallableForm::Handle => match callable.owner {
                        GraphCallableOwner::Workspace => {
                            Value::function_handle(callable.qualified_name.clone())
                        }
                        GraphCallableOwner::External => {
                            Value::external_function_handle(callable.qualified_name.clone())
                        }
                        GraphCallableOwner::Method => {
                            Value::method_function_handle(callable.qualified_name.clone())
                        }
                        GraphCallableOwner::Bound(_) => {
                            let binding =
                                binding.ok_or_else(|| invalid("bound handle lacks admission"))?;
                            Value::bound_function_handle(
                                callable.qualified_name.clone(),
                                binding.function,
                                binding.authority,
                                callable.source_role,
                                source_file,
                            )
                        }
                    },
                }
            }
            GraphNode::Activation(_) => return Err(invalid("activation is not a value")),
        };
        let rooted = crate::execution::RootedValue::in_context(value, self.runtime)
            .map_err(|error| ValueCodecError::runtime("$", error))?;
        self.values[index] = Some(rooted);
        drop(child_leases);
        self.visiting[index] = false;
        Ok(())
    }
}
