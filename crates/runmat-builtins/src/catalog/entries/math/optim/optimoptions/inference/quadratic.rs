//! Selected QP defaults observed by the existing constructor grammar.
use super::*;
pub(super) struct Selection {
    algorithm: Option<OptimAlgorithm>,
    default_limit: Option<bool>,
    provided: [bool; 3],
}
impl Default for Selection {
    fn default() -> Self {
        Self {
            algorithm: Some(OptimAlgorithm::InteriorPointConvex),
            default_limit: Some(true),
            provided: [false; 3],
        }
    }
}
impl Selection {
    pub(super) fn unknown(&mut self) {
        self.algorithm = None;
        self.default_limit = None;
        self.provided = [false; 3];
    }
    pub(super) fn merge(&mut self, source: &StructFact) {
        if !source.fields_complete {
            self.unknown();
            return;
        }
        for name in source.fields.keys() {
            if name.eq_ignore_ascii_case("Solver") {
                self.unknown();
            }
            match Field::parse_for_constructor(name, OptimSolver::Quadprog) {
                Some(Field::Algorithm) => self.algorithm = None,
                Some(Field::MaxIter) => self.default_limit = None,
                _ => {}
            }
        }
    }
    pub(super) fn record(
        &mut self,
        request: &CallRequest,
        field: Field,
        value: &ValueFact,
        index: usize,
    ) {
        for (slot, selected) in [Field::TolX, Field::ObjectiveLimit, Field::MaxTime]
            .into_iter()
            .enumerate()
        {
            if field == selected {
                self.provided[slot] = true;
            }
        }
        match field {
            Field::Algorithm => {
                self.algorithm = if value.shape.element_count() == Some(0) {
                    Some(OptimAlgorithm::InteriorPointConvex)
                } else {
                    request
                        .literals
                        .literal_string_at(index)
                        .and_then(|v| OptimAlgorithm::parse(&v))
                };
            }
            Field::MaxIter => self.default_limit = Some(value.shape.element_count() == Some(0)),
            _ => {}
        }
    }
    pub(super) fn finish(&self, fields: &mut BTreeMap<String, ValueFact>, complete: &mut bool) {
        match self.algorithm {
            Some(OptimAlgorithm::ActiveSet) => {
                if self.default_limit == Some(true) {
                    fields.insert(
                        Field::MaxIter.name().into(),
                        facts::double(ShapeFact::from(vec![Some(0), Some(0)]), StorageFact::Dense),
                    );
                } else if self.default_limit.is_none() {
                    fields.insert(
                        Field::MaxIter.name().into(),
                        ValueFact::unknown(DynamicReason::RuntimeValue),
                    );
                }
            }
            Some(_) => {
                if self.default_limit.is_none() {
                    fields.insert(
                        Field::MaxIter.name().into(),
                        ValueFact::unknown(DynamicReason::RuntimeValue),
                    );
                }
                for (slot, field) in [Field::TolX, Field::ObjectiveLimit, Field::MaxTime]
                    .into_iter()
                    .enumerate()
                {
                    if !self.provided[slot] {
                        fields.remove(field.name());
                    }
                }
            }
            None => {
                // Algorithm values in an imported structure are not known
                // merely from field names. Omit conditional default fields.
                for (slot, field) in [Field::TolX, Field::ObjectiveLimit, Field::MaxTime]
                    .into_iter()
                    .enumerate()
                {
                    if !self.provided[slot] {
                        fields.remove(field.name());
                    }
                }
                if self.default_limit != Some(false) {
                    fields.insert(
                        Field::MaxIter.name().into(),
                        ValueFact::unknown(DynamicReason::RuntimeValue),
                    );
                }
                *complete = false;
            }
        }
    }
}
