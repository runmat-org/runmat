use super::Projection;
use runmat_types::*;

impl Projection<'_> {
    pub(super) fn callable(&self, callable: &mut CallableFact) -> Result<(), String> {
        if let Some(identity) = &mut callable.identity {
            match identity {
                CallableIdentity::BoundFunction(function)
                | CallableIdentity::AnonymousFunction(function) => self.local(function),
                CallableIdentity::ExternalFunction { function, .. }
                    if self.locals.is_some_and(|(_, policy)| {
                        matches!(policy, super::References::ClosedUnit)
                    }) =>
                {
                    self.local(function)
                }
                _ => {}
            }
        }
        for fact in callable
            .parameters
            .iter_mut()
            .chain(&mut callable.outputs)
            .chain(&mut callable.captures)
        {
            self.fact(fact)?;
        }
        Ok(())
    }

    pub(super) fn fact(&self, fact: &mut ValueFact) -> Result<(), String> {
        self.kind(&mut fact.kind)
    }

    fn kind(&self, kind: &mut ValueKindFact) -> Result<(), String> {
        match kind {
            ValueKindFact::Cell(cell) => {
                self.fact(&mut cell.element)?;
                for fact in &mut cell.elements {
                    self.fact(fact)?;
                }
            }
            ValueKindFact::Struct(structure) => {
                for fact in structure.fields.values_mut() {
                    self.fact(fact)?;
                }
                for element in &mut structure.elements {
                    for fact in element.values_mut() {
                        self.fact(fact)?;
                    }
                }
            }
            ValueKindFact::Object(object) => {
                for fact in object.properties.values_mut() {
                    self.fact(fact)?;
                }
            }
            ValueKindFact::Callable(callable) => self.callable(callable)?,
            ValueKindFact::OutputList(sequence) => self.sequence(sequence)?,
            ValueKindFact::Execution(execution) => match execution {
                ExecutionFact::Future { output, .. }
                | ExecutionFact::Task { output, .. }
                | ExecutionFact::Job { output } => self.sequence(output)?,
                ExecutionFact::Pool => {}
            },
            ValueKindFact::Distributed(distributed) => {
                self.function(&mut distributed.id.function)?;
                self.owner(&mut distributed.owner)?;
                self.fact(&mut distributed.value)?;
            }
            ValueKindFact::Composite(composite) => {
                self.function(&mut composite.owner.0.function)?;
                self.fact(&mut composite.value)?;
            }
            ValueKindFact::Never
            | ValueKindFact::Unknown
            | ValueKindFact::Void
            | ValueKindFact::Numeric(_)
            | ValueKindFact::Logical
            | ValueKindFact::Character
            | ValueKindFact::String
            | ValueKindFact::Symbolic
            | ValueKindFact::ClassReference(_)
            | ValueKindFact::Exception(_)
            | ValueKindFact::Foreign(_) => {}
        }
        Ok(())
    }

    fn sequence(&self, sequence: &mut ValueSequenceFact) -> Result<(), String> {
        for fact in &mut sequence.outputs {
            self.fact(fact)?;
        }
        Ok(())
    }

    pub(super) fn owner(&self, owner: &mut DistributedOwner) -> Result<(), String> {
        self.function(match owner {
            DistributedOwner::Client(function) => function,
            DistributedOwner::Region(region) => &mut region.0.function,
        })
    }

    pub(super) fn region(
        &self,
        region: &mut crate::analysis::RegionAnalysis,
    ) -> Result<(), String> {
        let contract = &mut region.contract;
        self.function(&mut contract.id.function)?;
        self.point(&mut contract.entry)?;
        for point in &mut contract.exits {
            self.point(point)?;
        }
        for value in contract.live_in.iter_mut().chain(&mut contract.live_out) {
            self.value(value)?;
        }
        for value in &mut contract.value_facts {
            self.value(&mut value.value)?;
            self.fact(&mut value.fact)?;
        }
        for guard in &mut contract.guards {
            self.function(&mut guard.id.region.function)?;
            self.function(&mut guard.deopt.function)?;
            match &mut guard.condition {
                RegionGuardCondition::ValueFact { value, expected } => {
                    self.value(value)?;
                    self.fact(expected)?;
                }
                RegionGuardCondition::Class { value, expected } => {
                    self.value(value)?;
                    self.kind(expected)?;
                }
                RegionGuardCondition::Shape { value, .. }
                | RegionGuardCondition::Residency { value, .. }
                | RegionGuardCondition::Alias { value, .. } => self.value(value)?,
                RegionGuardCondition::RuntimeState { .. }
                | RegionGuardCondition::Capability { .. } => {}
            }
        }
        for point in &mut region.operations {
            self.point(point)?;
        }
        for consumer in &mut region.future_consumers {
            self.value(&mut consumer.value)?;
            self.point(&mut consumer.point)?;
        }
        contract.exits.sort_unstable();
        contract.live_in.sort_unstable();
        contract.live_out.sort_unstable();
        contract.value_facts.sort_by_key(|value| value.value);
        contract.guards.sort_by_key(|guard| guard.id);
        region.operations.sort_unstable();
        region.future_consumers.sort_unstable();
        contract
            .validate()
            .map_err(|error| format!("projected region: {}: {}", error.path, error.message))
    }
}
