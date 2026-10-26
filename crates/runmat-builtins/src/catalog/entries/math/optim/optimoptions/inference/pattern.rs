//! Selected Pattern constructor facts derive from the canonical field schema.
use super::*;
use crate::{
    OptimPatternAlgorithm, OptimPatternDefault as DefaultValue, OptimPatternOrder,
    OptimPatternPoll, OptimPatternScalarRule as Rule, OptimPatternSearch,
};

fn switch(field: Field) -> bool {
    matches!(
        field,
        Field::Cache
            | Field::ScaleMesh
            | Field::MeshRotate
            | Field::AccelerateMesh
            | Field::UseCompletePoll
            | Field::UseCompleteSearch
            | Field::UseVectorized
            | Field::UseParallel
    )
}
fn selector(field: Field) -> bool {
    matches!(
        field,
        Field::Algorithm
            | Field::Display
            | Field::PollMethod
            | Field::PollOrderAlgorithm
            | Field::SearchFcn
    )
}

pub(super) fn metadata(field: Field, value: &ValueFact) -> bool {
    if field == Field::SearchFcn {
        let target = |value: &ValueFact| {
            text(value)
                || matches!(
                    value.kind,
                    ValueKindFact::Callable(_) | ValueKindFact::Unknown
                )
        };
        if let ValueKindFact::Cell(cell) = &value.kind {
            if value.shape.element_count().is_some_and(|n| n == 0 || n > 3)
                || cell.elements.len() > 3
            {
                return false;
            }
            return cell.elements.first().is_none_or(target) && cell.elements.iter().enumerate().skip(1).all(|(index,value)| {
                matches!(value.kind,ValueKindFact::Unknown) ||
                index==2 && (matches!(value.kind,ValueKindFact::Struct(_)) || value.shape.element_count()==Some(0)) ||
                value.shape.element_count().is_none_or(|n|n==1) && value.storage!=StorageFact::Sparse && matches!(value.residency,ResidencyFact::Host|ResidencyFact::Unknown) && matches!(value.kind,ValueKindFact::Numeric(n) if n.domain==NumericDomain::Real)
            });
        }
        return target(value) || value.shape.element_count() == Some(0);
    }
    if matches!(value.kind, ValueKindFact::Unknown) || value.shape.element_count() == Some(0) {
        return true;
    }
    if matches!(field, Field::OutputFcn | Field::PlotFcn) {
        let callable = |value: &ValueFact| {
            matches!(
                value.kind,
                ValueKindFact::Callable(_) | ValueKindFact::Unknown
            ) || field == Field::PlotFcn && text(value)
        };
        return callable(value)
            || matches!(&value.kind, ValueKindFact::Cell(cell) if cell.elements.iter().all(callable));
    }
    if selector(field) {
        return text(value);
    }
    if DefaultValue::dimension_expression(field).is_some() && text(value) {
        return true;
    }
    if switch(field) && text(value) {
        return true;
    }
    if field == Field::MaxTime {
        if let ValueKindFact::Object(object) = &value.kind {
            return object
                .runtime_class
                .as_ref()
                .is_none_or(|class| class.is(runmat_types::standard::DURATION))
                && value.shape.element_count().is_none_or(|n| n == 1);
        }
    }
    value.shape.element_count().is_none_or(|n| n == 1)
        && value.storage != StorageFact::Sparse
        && (matches!(value.kind, ValueKindFact::Numeric(n) if n.domain == NumericDomain::Real)
            || switch(field) && matches!(value.kind, ValueKindFact::Logical))
}
pub(super) fn normalized(field: Field, source: Option<&ValueFact>) -> ValueFact {
    if let Some(source) = source {
        let mut fact = source.clone();
        if matches!(
            fact.kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Logical
        ) {
            fact.residency = ResidencyFact::Host;
        }
        return fact;
    }
    match DefaultValue::for_field(field) {
        Some(DefaultValue::Text(_) | DefaultValue::DimensionCount(_)) => {
            ValueFact::scalar(ValueKindFact::String)
        }
        Some(DefaultValue::Logical(_)) => ValueFact::scalar(ValueKindFact::Logical),
        Some(DefaultValue::Number(_)) => facts::double(ShapeFact::Scalar, StorageFact::Scalar),
        Some(DefaultValue::Empty) => {
            facts::double(ShapeFact::from(vec![Some(0), Some(0)]), StorageFact::Dense)
        }
        None => ValueFact::unknown(DynamicReason::RuntimeValue),
    }
}
pub(super) fn inspect(
    solver: Option<OptimSolver>,
    request: &CallRequest,
    field: Field,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> bool {
    if solver != Some(OptimSolver::Patternsearch) {
        return false;
    }
    let mut invalid = false;
    if let Some(text) = request.literals.literal_string_at(index) {
        invalid = match field {
            Field::Algorithm => OptimPatternAlgorithm::parse(&text).is_none(),
            Field::Display => OptimDisplay::parse(&text)
                .is_none_or(|display| !OptimSolver::Patternsearch.accepts_display(display)),
            Field::PollMethod => OptimPatternPoll::parse(&text).is_none(),
            Field::PollOrderAlgorithm => OptimPatternOrder::parse(&text).is_none(),
            Field::SearchFcn => OptimPatternSearch::parse(&text).is_none(),
            Field::UseParallel => false, // Effective Off/Auto/On grammar is checked by the runtime owner.
            _ if switch(field) => crate::OptimSwitch::parse(&text).is_none(),
            _ => Some(text.as_str()) != DefaultValue::dimension_expression(field),
        };
    }
    if let Some(number) = request.literals.numeric_at(index) {
        invalid |= if switch(field) {
            number != 0.0 && number != 1.0
        } else {
            Rule::for_field(field).is_some_and(|rule| !rule.accepts(number))
        };
    }
    if invalid {
        diagnostics.push(argument_error(
            "RM-CATALOG-PATTERN-OPTION",
            "option is outside its patternsearch grammar or scalar range",
            index,
        ));
    }
    true
}

#[cfg(test)]
mod search_tests {
    use super::*;
    #[test]
    fn parameterized_search_facts_preserve_payload_and_reject_nonhost_or_nonscalar_counts() {
        let count = facts::double(ShapeFact::Scalar, StorageFact::Scalar);
        let cell = |count: ValueFact| {
            let mut value = ValueFact::scalar(ValueKindFact::Cell(runmat_types::CellFact {
                element: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
                elements: vec![
                    ValueFact::scalar(ValueKindFact::String),
                    count.clone(),
                    count,
                ],
                elements_complete: true,
            }));
            value.shape = ShapeFact::from(vec![Some(1), Some(3)]);
            value
        };
        let original = cell(count.clone());
        assert!(metadata(Field::SearchFcn, &original));
        assert_eq!(normalized(Field::SearchFcn, Some(&original)), original);
        let mut nested = original.clone();
        if let ValueKindFact::Cell(cell) = &mut nested.kind {
            cell.elements[2] = ValueFact::scalar(ValueKindFact::Struct(
                runmat_types::StructFact::scalar(std::collections::BTreeMap::new(), true),
            ));
        }
        assert!(metadata(Field::SearchFcn, &nested));
        assert_eq!(normalized(Field::SearchFcn, Some(&nested)), nested);
        let mut resident = count.clone();
        resident.residency = ResidencyFact::Device { provider: None };
        assert!(!metadata(Field::SearchFcn, &cell(resident)));
        let mut vector = count;
        vector.shape = ShapeFact::from(vec![Some(2), Some(1)]);
        assert!(!metadata(Field::SearchFcn, &cell(vector)));
    }
}
