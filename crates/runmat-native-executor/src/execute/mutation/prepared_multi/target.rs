use runmat_mir::{MirPlace, MirSequenceTarget};
use runmat_runtime::native::NativeValueRef;
use runmat_runtime::object::indexing::ObjectIndexComponent;
use runmat_runtime::sequence::{
    AssignmentStepSpec, PreparedSequenceDestination, SequenceDestinationBuilder,
    SequenceEndpointSpec,
};
use runmat_value::Value;

use crate::{NativeExecutorError, NativeExecutorResult};

use crate::execute::state::HostState;

use crate::execute::call_suspension::{
    begin_sequence_stage, take_completed_sequence_stage, SequenceStage,
};

#[derive(Clone)]
enum Phase {
    MaterializePath,
    PushStep(AssignmentStepSpec),
    MaterializeEndpoint,
    Finish(SequenceEndpointSpec),
}

pub(super) struct PendingSequenceTarget {
    pub(super) target: MirSequenceTarget,
    root: runmat_mir::MirLocalId,
    path: Vec<MirPlace>,
    builder: SequenceDestinationBuilder,
    index: usize,
    phase: Phase,
    region: Option<usize>,
    step: usize,
    retained: Vec<NativeValueRef>,
    absent_root: Option<NativeValueRef>,
}

pub(super) struct PreparedSequenceTarget {
    pub root: runmat_mir::MirLocalId,
    pub destination: PreparedSequenceDestination,
    pub absent_root: Option<NativeValueRef>,
    pub root_creation: runmat_mir::SequenceRootCreation,
}

impl PendingSequenceTarget {
    pub(super) fn new(
        state: &mut HostState,
        target: &MirSequenceTarget,
    ) -> NativeExecutorResult<Self> {
        let base = target.base();
        let mut path = Vec::new();
        let root = flatten_root(base, &mut path)?;
        let (root_value, absent_root) = state.sequence_root(root.0, target.root_creation())?;
        let path = path.into_iter().cloned().collect();
        let builder = SequenceDestinationBuilder::new(&root_value)?;
        let mut pending = Self {
            target: target.clone(),
            root,
            path,
            builder,
            index: 0,
            phase: Phase::MaterializePath,
            region: None,
            step: 0,
            retained: Vec::new(),
            absent_root,
        };
        pending.refresh_retained(state);
        Ok(pending)
    }

    pub(super) fn root_references(&self) -> impl Iterator<Item = NativeValueRef> + '_ {
        self.retained.iter().copied().chain(self.absent_root)
    }

    fn refresh_retained(&mut self, state: &mut HostState) {
        self.retained.clear();
        self.builder.visit_retained_values(&mut |value| {
            self.retained.push(state.arena.insert(value.clone()));
        });
        let components = match &self.phase {
            Phase::PushStep(AssignmentStepSpec::Parentheses { selectors }) => {
                Some(selectors.as_slice())
            }
            Phase::PushStep(AssignmentStepSpec::Braces(selectors)) => Some(selectors.as_slice()),
            Phase::Finish(SequenceEndpointSpec::CellContents(selectors)) => {
                Some(selectors.as_slice())
            }
            _ => None,
        };
        if let Some(components) = components {
            for component in components {
                if let ObjectIndexComponent::Value(value) = component {
                    self.retained.push(state.arena.insert(value.clone()));
                }
            }
        }
    }

    pub(super) fn advance(
        &mut self,
        state: &mut HostState,
    ) -> NativeExecutorResult<Option<PreparedSequenceTarget>> {
        let (region, next_step, _) = state.enter_contextual_region();
        match self.region {
            Some(expected) if expected != region || self.step != next_step => {
                return Err(NativeExecutorError::Host(
                    "pending sequence target does not match its exact preparation region".into(),
                ));
            }
            None => self.region = Some(region),
            Some(_) => {}
        }
        loop {
            let step = self.step;
            state.enter_contextual_step(region, step);
            let result = self.advance_step(state);
            let next_step = step + usize::from(result.is_ok());
            state.finish_contextual_step(region, step, next_step)?;
            match result {
                Ok(Some(destination)) => {
                    let root = match self.absent_root {
                        Some(seed) => seed,
                        None => state.local_reference(self.root.0)?,
                    };
                    state.finish_contextual_region(region, vec![root]);
                    return Ok(Some(PreparedSequenceTarget {
                        root: self.root,
                        destination,
                        absent_root: self.absent_root,
                        root_creation: self.target.root_creation(),
                    }));
                }
                Ok(None) => {
                    self.step += 1;
                    self.refresh_retained(state);
                }
                Err(error) => return Err(error),
            }
        }
    }

    fn advance_step(
        &mut self,
        state: &mut HostState,
    ) -> NativeExecutorResult<Option<PreparedSequenceDestination>> {
        match self.phase.clone() {
            Phase::MaterializePath if self.index < self.path.len() => {
                let step = assignment_step(state, self.builder.current(), &self.path[self.index])?;
                self.phase = Phase::PushStep(step);
                Ok(None)
            }
            Phase::MaterializePath => {
                self.phase = Phase::MaterializeEndpoint;
                Ok(None)
            }
            Phase::PushStep(spec) => {
                let runtime = state.runtime.clone();
                let caller = state.function.name.clone();
                let mut builder = self.builder.clone();
                let stage = stage_call(
                    state,
                    Box::pin(async move {
                        runtime
                            .scope(builder.push_step_in_context(&runtime, spec, Some(&caller)))
                            .await?;
                        Ok(SequenceStage::Builder(Box::new(builder)))
                    }),
                )?;
                let SequenceStage::Builder(builder) = stage else {
                    return Err(NativeExecutorError::Host(
                        "sequence step returned destination payload".into(),
                    ));
                };
                self.builder = *builder;
                self.index += 1;
                self.phase = Phase::MaterializePath;
                Ok(None)
            }
            Phase::MaterializeEndpoint => {
                self.phase = Phase::Finish(match &self.target {
                    MirSequenceTarget::Member { member, .. } => {
                        SequenceEndpointSpec::Member(member.clone())
                    }
                    MirSequenceTarget::DynamicMember { member, .. } => {
                        let member =
                            super::super::super::operand::materialize_operand(state, member)?;
                        SequenceEndpointSpec::Member(runmat_types::MemberName(
                            String::try_from(&member).map_err(NativeExecutorError::Host)?,
                        ))
                    }
                    MirSequenceTarget::CellContents { indexing, .. } => {
                        SequenceEndpointSpec::CellContents(
                            super::super::super::indexing::materialize_index_components(
                                state,
                                self.builder.current(),
                                indexing,
                            )?,
                        )
                    }
                });
                Ok(None)
            }
            Phase::Finish(endpoint) => {
                let runtime = state.runtime.clone();
                let builder = self.builder.clone();
                let stage = stage_call(
                    state,
                    Box::pin(async move {
                        let destination = runtime
                            .scope(builder.finish_in_context(&runtime, endpoint))
                            .await?;
                        Ok(SequenceStage::Destination(destination))
                    }),
                )?;
                let SequenceStage::Destination(destination) = stage else {
                    return Err(NativeExecutorError::Host(
                        "sequence finalization returned builder payload".into(),
                    ));
                };
                Ok(Some(destination))
            }
        }
    }
}

fn stage_call(
    state: &mut HostState,
    future: std::pin::Pin<
        Box<dyn std::future::Future<Output = NativeExecutorResult<SequenceStage>> + 'static>,
    >,
) -> NativeExecutorResult<SequenceStage> {
    let embedded = state.enter_embedded_call();
    let result = (|| {
        if let Some(stage) = take_completed_sequence_stage(state, embedded.as_ref())? {
            return Ok(stage);
        }
        begin_sequence_stage(state, embedded.clone(), future)
    })();
    let finished = state.finish_embedded_call(embedded.as_ref());
    match (result, finished) {
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error),
        (Ok(stage), Ok(())) => Ok(stage),
    }
}

fn assignment_step(
    state: &mut HostState,
    current: &Value,
    place: &MirPlace,
) -> NativeExecutorResult<AssignmentStepSpec> {
    match place {
        MirPlace::Member(_, member) => Ok(AssignmentStepSpec::Member(member.clone())),
        MirPlace::DynamicMember(_, member) => {
            let member = super::super::super::operand::materialize_operand(state, member)?;
            Ok(AssignmentStepSpec::Member(runmat_types::MemberName(
                String::try_from(&member).map_err(NativeExecutorError::Host)?,
            )))
        }
        MirPlace::Index(_, indexing) => {
            let selectors = super::super::super::indexing::materialize_index_components(
                state, current, indexing,
            )?;
            Ok(match indexing.kind {
                runmat_types::IndexKind::Paren => AssignmentStepSpec::Parentheses { selectors },
                runmat_types::IndexKind::Brace => AssignmentStepSpec::Braces(selectors),
            })
        }
        MirPlace::Local(_) | MirPlace::Binding(_) => Err(NativeExecutorError::Host(
            "sequence destination path contains a root as an interior step".into(),
        )),
    }
}

pub(super) fn root_local(place: &MirPlace) -> NativeExecutorResult<runmat_mir::MirLocalId> {
    let mut path = Vec::new();
    flatten_root(place, &mut path)
}

fn flatten_root<'a>(
    place: &'a MirPlace,
    path: &mut Vec<&'a MirPlace>,
) -> NativeExecutorResult<runmat_mir::MirLocalId> {
    match place {
        MirPlace::Local(local) => Ok(*local),
        MirPlace::Binding(binding) => Err(NativeExecutorError::Host(format!(
            "verified Native IR retains legacy MIR binding place {binding:?}"
        ))),
        MirPlace::Member(base, _) | MirPlace::DynamicMember(base, _) | MirPlace::Index(base, _) => {
            let root = flatten_root(base, path)?;
            path.push(place);
            Ok(root)
        }
    }
}
