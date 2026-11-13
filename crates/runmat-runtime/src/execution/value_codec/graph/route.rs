use runmat_value::Value;

use crate::execution::DeferredInvocation;

/// Whether Direct inline transport would discard reachable selected identity,
/// immutable class provenance, metadata links, or a shared lexical activation.
/// This only chooses the route; Graph snapshot still validates the complete
/// invocation at spawn under its shared limits.
pub fn deferred_invocation_requires_graph(invocation: &DeferredInvocation) -> bool {
    let mut requires_graph = false;
    let _ = invocation.visit_retained_values(&mut |value| {
        if value_requires_graph(value) {
            requires_graph = true;
            None
        } else {
            Some(())
        }
    });
    requires_graph
}

fn value_requires_graph(root: &Value) -> bool {
    let mut work = vec![root];
    while let Some(value) = work.pop() {
        match value {
            Value::Closure(closure) => {
                if closure.lexical_environment.is_some() {
                    return true;
                }
                work.extend(&closure.captures);
            }
            Value::Cell(cell) => work.extend(cell.data()),
            Value::Struct(value) => work.extend(value.fields.values()),
            Value::StructArray(array) => {
                for element in array.elements() {
                    work.extend(element.fields.values());
                }
            }
            Value::Object(object) => {
                if object.selected_class().is_some() {
                    return true;
                }
                work.extend(object.properties().values());
            }
            Value::ObjectArray(array) => {
                if !array.selected_provenance().is_empty() {
                    return true;
                }
                for index in 0..array.len() {
                    if let Some(value) = array.get_linear(index) {
                        work.push(value);
                    }
                }
            }
            // Routing only observes reachable activations. Both value codecs
            // reject this transient builtin return carrier at admission.
            Value::OutputList(transient_values) => work.extend(transient_values),
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
            | Value::Symbolic(_)
            | Value::SymbolicArray(_)
            | Value::GpuTensor(_)
            | Value::Listener(_)
            | Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle(_)
            | Value::ScopedFunctionHandle(_)
            | Value::ClassRef(_)
            | Value::MException(_)
            | Value::Future(_)
            | Value::Task(_)
            | Value::Pool(_)
            | Value::Job(_)
            | Value::Distributed(_)
            | Value::Composite(_)
            | Value::Foreign(_) => {}
            Value::HandleObject(handle) => {
                // Ordinary/native handles have never been Direct-transferable;
                // leave their established unsupported path intact. A selected
                // receiver or its reciprocal metadata needs the Graph route.
                if handle.valid
                    && runmat_gc::gc_with_value(&handle.target, |target| {
                        matches!(target, Value::Object(object) if object.selected_class().is_some())
                    })
                    .unwrap_or(false)
                {
                    return true;
                }
                if handle
                    .class_name
                    .is(crate::builtins::introspection::dynamicprops::DYNAMIC_PROPERTY_CLASS)
                {
                    // The exact reciprocal-owner check, not the metadata
                    // class spelling, authorizes snapshot encoding. Route a
                    // malformed metadata handle there as well so Graph
                    // admission reports its typed invalidity before effects.
                    return match crate::builtins::introspection::dynamicprops::checked_metadata_owner(
                        handle,
                    ) {
                        Ok(Some(_)) | Err(_) => true,
                        Ok(None) => false,
                    };
                }
            }
        }
    }
    false
}
