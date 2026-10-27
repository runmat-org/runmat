//! Values retained while a sequence destination is prepared across calls.
use super::*;
use crate::object::indexing::{ObjectIndexComponent, ObjectIndexSelector};
use runmat_gc::{Trace, Tracer};

impl Trace for AssignmentStepSpec {
    fn trace(&self, tracer: &mut dyn Tracer) {
        visit_specs(std::slice::from_ref(self), &mut |value| value.trace(tracer));
    }
}

impl Trace for SequenceDestinationBuilder {
    fn trace(&self, tracer: &mut dyn Tracer) {
        self.visit_retained_values(&mut |value| value.trace(tracer));
    }
}

impl Trace for PreparedDestinationKind {
    fn trace(&self, tracer: &mut dyn Tracer) {
        visit_destination_kind(self, &mut |value| value.trace(tracer));
    }
}

impl Trace for PreparedSequenceDestination {
    fn trace(&self, tracer: &mut dyn Tracer) {
        self.visit_retained_values(&mut |value| value.trace(tracer));
    }
}

impl SequenceDestinationBuilder {
    pub fn visit_retained_values(&self, visitor: &mut impl FnMut(&Value)) {
        visitor(&self.current);
        visit_prepared_steps(&self.steps, visitor);
        if let Some(suffix) = &self.object_suffix {
            visitor(&suffix.target);
            visit_specs(&suffix.structural_steps, visitor);
            visit_subscripts(&suffix.steps, visitor);
        }
    }
}

impl PreparedSequenceDestination {
    pub fn visit_retained_values(&self, visitor: &mut impl FnMut(&Value)) {
        visit_destination_kind(&self.kind, visitor);
    }
}

fn visit_destination_kind(kind: &PreparedDestinationKind, visitor: &mut impl FnMut(&Value)) {
    match kind {
        PreparedDestinationKind::Structural { steps, endpoint } => {
            visit_prepared_steps(steps, visitor);
            endpoint.visit_retained_values(visitor);
        }
        PreparedDestinationKind::ObjectProtocol(object) => {
            visit_prepared_steps(&object.prefix, visitor);
            visit_specs(&object.suffix_steps, visitor);
            object.endpoint.visit_retained_values(visitor);
            visit_subscripts(object.path.steps(), visitor);
        }
    }
}

fn visit_components(components: &[ObjectIndexComponent], visitor: &mut impl FnMut(&Value)) {
    for component in components {
        match component {
            ObjectIndexComponent::Value(value) => visitor(value),
            ObjectIndexComponent::Colon | ObjectIndexComponent::End => {}
        }
    }
}

fn visit_prepared_steps(steps: &[PreparedAssignmentStep], visitor: &mut impl FnMut(&Value)) {
    for step in steps {
        match step {
            PreparedAssignmentStep::Member(_)
            | PreparedAssignmentStep::InitializingMember { .. } => {}
            PreparedAssignmentStep::Parentheses {
                selector_values, ..
            }
            | PreparedAssignmentStep::InitializingParentheses {
                selector_values, ..
            } => visit_components(selector_values, visitor),
            PreparedAssignmentStep::Braces(components) => visit_components(components, visitor),
            PreparedAssignmentStep::InitializingCellContents { .. } => {}
        }
    }
}

fn visit_specs(steps: &[AssignmentStepSpec], visitor: &mut impl FnMut(&Value)) {
    for step in steps {
        match step {
            AssignmentStepSpec::Member(_) => {}
            AssignmentStepSpec::Parentheses { selectors } => visit_components(selectors, visitor),
            AssignmentStepSpec::Braces(components) => visit_components(components, visitor),
        }
    }
}

fn visit_subscripts(steps: &[ObjectSubscript], visitor: &mut impl FnMut(&Value)) {
    for step in steps {
        match step.selector() {
            ObjectIndexSelector::IndexValues { components } => {
                visit_components(components, visitor)
            }
            ObjectIndexSelector::ScalarIndices { .. } | ObjectIndexSelector::Member(_) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn prepared_destination_retains_original_source_root() {
        runmat_gc::gc_test_context(|| {
            let initial = runmat_gc::gc_allocate_rooted(Value::String("source".into()))
                .expect("allocate source payload");
            let handle = initial.handle();
            let source = Value::Cell(
                runmat_value::CellArray::new_with_shape(
                    vec![Value::HandleObject(runmat_value::HandleRef {
                        class_name: "SourcePayload".into(),
                        target: handle,
                        valid: true,
                    })],
                    vec![1, 1],
                )
                .expect("source cell"),
            );
            let builder = SequenceDestinationBuilder::new(&source).expect("root source");
            initial.unroot().expect("transfer source to builder");
            let destination =
                futures::executor::block_on(builder.finish(SequenceEndpointSpec::CellContents(
                    runmat_value::ExactArrayStorage::shared_unbounded(vec![
                        ObjectIndexComponent::Colon,
                    ]),
                )))
                .expect("prepare cell destination");
            runmat_gc::gc_collect_major().expect("collect with prepared destination");
            assert_eq!(
                runmat_gc::gc_clone_value(&handle).expect("source remains rooted"),
                Value::String("source".into())
            );
            drop(destination);
            runmat_gc::gc_collect_major().expect("collect after destination retirement");
            assert!(runmat_gc::gc_clone_value(&handle).is_err());
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn prepared_selector_survives_collecting_builder_retirement() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        use runmat_value::{HandleRef, ObjectInstance};

        runmat_gc::gc_test_context(|| {
            let initial = runmat_gc::gc_allocate_rooted(Value::String("selector".into()))
                .expect("allocate selector payload");
            let handle = initial.handle();
            let selector = ObjectIndexComponent::Value(Value::HandleObject(HandleRef {
                class_name: "SelectorPayload".into(),
                target: handle,
                valid: true,
            }));
            let mut builder = SequenceDestinationBuilder::new(&Value::Object(ObjectInstance::new(
                "SelectorReceiver",
            )))
            .expect("root destination");
            futures::executor::block_on(builder.push_step(
                AssignmentStepSpec::Braces(runmat_value::ExactArrayStorage::shared_unbounded(
                    vec![selector.clone()],
                )),
                None,
            ))
            .expect("retain selector before preparing destination");
            initial.unroot().expect("transfer selector to builder");

            let path = ObjectSubscriptPath::single(ObjectSubscript::braces(
                ObjectIndexSelector::IndexValues {
                    components: runmat_value::ExactArrayStorage::shared_unbounded(vec![
                        selector.clone()
                    ]),
                },
            ));
            let prepared = PreparedSequenceDestination::rooted(
                PreparedDestinationKind::ObjectProtocol(
                    builder
                        .control
                        .boxed(
                            None,
                            PreparedObjectProtocolDestination {
                                prefix: builder.control.vector(),
                                expected_class: "SelectorReceiver".into(),
                                cardinality_method: None,
                                suffix_steps: builder.control.vector(),
                                endpoint: SequenceEndpointSpec::CellContents(
                                    runmat_value::ExactArrayStorage::shared_unbounded(vec![
                                        selector,
                                    ]),
                                ),
                                path,
                                cardinality: 1,
                            },
                        )
                        .expect("ordinary protocol control"),
                ),
                builder.source_roots.clone(),
            )
            .expect("root prepared selector before builder retires");

            let observed = Arc::new(AtomicBool::new(false));
            let observed_in_callback = Arc::clone(&observed);
            let identity = (handle.addr(), handle.epoch());
            let _observer = runmat_gc::RootRetirementObserver::register(Arc::new(move |event| {
                let runmat_gc::RootRetirement::Explicit(handle) = event else {
                    return;
                };
                if (handle.addr(), handle.epoch()) != identity
                    || observed_in_callback.swap(true, Ordering::SeqCst)
                {
                    return;
                }
                runmat_gc::gc_collect_major().expect("collect during builder root retirement");
                assert_eq!(
                    runmat_gc::gc_clone_value(&handle).expect("prepared selector stays live"),
                    Value::String("selector".into())
                );
            }))
            .expect("register collecting retirement observer");
            drop(builder);
            assert!(observed.load(Ordering::SeqCst));
            drop(prepared);
            runmat_gc::gc_collect_major().expect("collect after prepared destination retires");
            assert!(runmat_gc::gc_clone_value(&handle).is_err());
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn builder_owns_nested_handles_until_retirement() {
        runmat_gc::gc_test_context(|| {
            let initial = runmat_gc::gc_allocate_rooted(Value::String("destination".into()))
                .expect("allocate destination payload");
            let handle = initial.handle();
            let source = Value::Cell(
                runmat_value::CellArray::new_with_shape(
                    vec![Value::HandleObject(runmat_value::HandleRef {
                        class_name: "DestinationPayload".into(),
                        target: handle,
                        valid: true,
                    })],
                    vec![1, 1],
                )
                .expect("destination container"),
            );
            let builder = SequenceDestinationBuilder::new(&source).expect("root destination");
            initial.unroot().expect("transfer to builder");
            runmat_gc::gc_collect_major().expect("collect with live builder");
            assert_eq!(
                runmat_gc::gc_clone_value(&handle).expect("builder retains payload"),
                Value::String("destination".into())
            );
            drop(builder);
            runmat_gc::gc_collect_major().expect("collect after builder retirement");
            assert!(runmat_gc::gc_clone_value(&handle).is_err());
        });
    }

    #[test]
    fn builder_retains_current_and_prepared_selector_values() {
        let mut builder = SequenceDestinationBuilder::new(&Value::String("root".into()))
            .expect("root destination");
        builder
            .control
            .push(
                None,
                &mut builder.steps,
                PreparedAssignmentStep::Braces(runmat_value::ExactArrayStorage::shared_unbounded(
                    vec![
                        ObjectIndexComponent::Colon,
                        ObjectIndexComponent::Value(Value::String("selector".into())),
                    ],
                )),
            )
            .expect("admit ordinary prepared-step backing before push");
        let mut values = Vec::new();
        builder.visit_retained_values(&mut |value| values.push(value.clone()));
        assert_eq!(
            values,
            vec![
                Value::String("root".into()),
                Value::String("selector".into())
            ]
        );
    }

    #[test]
    fn endpoint_retains_value_selectors_but_not_colons_or_names() {
        let endpoint = SequenceEndpointSpec::CellContents(
            runmat_value::ExactArrayStorage::shared_unbounded(vec![
                ObjectIndexComponent::Value(Value::Num(2.0)),
                ObjectIndexComponent::Colon,
            ]),
        );
        let mut values = Vec::new();
        endpoint.visit_retained_values(&mut |value| values.push(value.clone()));
        SequenceEndpointSpec::Member("field".into())
            .visit_retained_values(&mut |value| values.push(value.clone()));
        assert_eq!(values, vec![Value::Num(2.0)]);
    }
}
