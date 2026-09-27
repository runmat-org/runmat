//! Values retained while a sequence destination is prepared across calls.
use super::*;
use crate::object::indexing::{ObjectIndexComponent, ObjectIndexSelector};

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
        match &self.kind {
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
}

fn visit_components(components: &[ObjectIndexComponent], visitor: &mut impl FnMut(&Value)) {
    for component in components {
        match component {
            ObjectIndexComponent::Value(value) => visitor(value),
            ObjectIndexComponent::Colon => {}
        }
    }
}

fn visit_prepared_steps(steps: &[PreparedAssignmentStep], visitor: &mut impl FnMut(&Value)) {
    for step in steps {
        match step {
            PreparedAssignmentStep::Member(_) => {}
            PreparedAssignmentStep::Parentheses {
                selector_values, ..
            } => visit_components(selector_values, visitor),
            PreparedAssignmentStep::Braces(components) => visit_components(components, visitor),
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

    #[test]
    fn builder_retains_current_and_prepared_selector_values() {
        let mut builder = SequenceDestinationBuilder::new(&Value::String("root".into()));
        builder.steps.push(PreparedAssignmentStep::Braces(vec![
            ObjectIndexComponent::Colon,
            ObjectIndexComponent::Value(Value::String("selector".into())),
        ]));
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
        let endpoint = SequenceEndpointSpec::CellContents(vec![
            ObjectIndexComponent::Value(Value::Num(2.0)),
            ObjectIndexComponent::Colon,
        ]);
        let mut values = Vec::new();
        endpoint.visit_retained_values(&mut |value| values.push(value.clone()));
        SequenceEndpointSpec::Member("field".into())
            .visit_retained_values(&mut |value| values.push(value.clone()));
        assert_eq!(values, vec![Value::Num(2.0)]);
    }
}
