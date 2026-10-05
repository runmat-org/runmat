use runmat_mir::{MirLocalId, MirPlace};
use runmat_runtime::native::{NativeSiteRequest, NativeValueRef};
use runmat_value::Value;

use crate::{NativeExecutorError, NativeExecutorResult};

use super::{flatten_place, HostState, PlaceSegment, PreparedPlaceWrite};

#[derive(Clone, Copy)]
enum Phase {
    ReadPrefix(usize),
    Writeback,
}

mod sequence;
pub(super) use sequence::{assign_prepared_sequence, assign_sequence, PendingSequenceAssign};

/// The invocation-owned cursor for one statement site. Values are arena
/// references, not copies of a caller's mutable receiver. A completed step is
/// never evaluated again after a suspended child protocol resumes.
pub(super) struct PendingAssign {
    request: NativeSiteRequest,
    place: MirPlace,
    root: MirLocalId,
    segments: Vec<PlaceSegment>,
    parents: Vec<(NativeValueRef, PlaceSegment)>,
    current: NativeValueRef,
    rhs: NativeValueRef,
    phase: Phase,
    step: usize,
    region: usize,
    delete: bool,
    allow_init: bool,
}

pub(super) struct PendingPlacePrepare {
    request: NativeSiteRequest,
    place: MirPlace,
    root: MirLocalId,
    segments: Vec<PlaceSegment>,
    parents: Vec<(NativeValueRef, PlaceSegment)>,
    current: NativeValueRef,
    region: usize,
    step: usize,
}

impl PendingPlacePrepare {
    pub(super) fn root_references(&self) -> impl Iterator<Item = NativeValueRef> + '_ {
        std::iter::once(self.current).chain(self.parents.iter().map(|(value, _)| *value))
    }
}

pub(super) fn prepare_place_write(
    state: &mut HostState,
    place: &MirPlace,
) -> NativeExecutorResult<(PreparedPlaceWrite, Value)> {
    let request = state.current_request.ok_or_else(|| {
        NativeExecutorError::Host("native place preparation has no active rvalue site".into())
    })?;
    let (region, next_step, _) = state.enter_contextual_region();
    let mut cursor = match state.pending_place_prepare.take() {
        Some(cursor)
            if cursor.request == request
                && cursor.place == *place
                && cursor.region == region
                && cursor.step == next_step =>
        {
            cursor
        }
        Some(_) => {
            return Err(NativeExecutorError::Host(
                "pending place preparation does not match the exact resumed site".into(),
            ));
        }
        None => {
            let mut segments = Vec::new();
            let root = flatten_place(place, &mut segments)?;
            let current = state.local_reference(root.0)?;
            PendingPlacePrepare {
                request,
                place: place.clone(),
                root,
                segments,
                parents: Vec::new(),
                current,
                region,
                step: 0,
            }
        }
    };
    while cursor.step < cursor.segments.len() {
        let step = cursor.step;
        state.enter_contextual_step(region, step);
        let parent = state.arena.get(cursor.current)?.clone();
        let segment = cursor.segments[step].clone();
        let result = read_step(state, parent, &segment, false);
        state.finish_contextual_step(region, step, step + usize::from(result.is_ok()))?;
        match result {
            Ok(value) => {
                cursor.parents.push((cursor.current, segment));
                cursor.current = state.arena.insert(value);
                cursor.step += 1;
            }
            Err(NativeExecutorError::CallSuspended) => {
                state.pending_place_prepare = Some(cursor);
                return Err(NativeExecutorError::CallSuspended);
            }
            Err(error) => return Err(error),
        }
    }
    let current = state.arena.get(cursor.current)?.clone();
    state.finish_contextual_region(region, vec![cursor.current]);
    Ok((
        PreparedPlaceWrite {
            root: cursor.root,
            parents: cursor.parents,
        },
        current,
    ))
}

pub(super) struct PendingPlainMulti {
    pub(super) request: NativeSiteRequest,
    pub(super) next_target: usize,
}

pub(super) fn skip_completed(state: &mut HostState, place: &MirPlace) -> NativeExecutorResult<()> {
    if matches!(place, MirPlace::Local(_)) {
        return Ok(());
    }
    skip_completed_region(state)
}

pub(super) fn skip_completed_region(state: &mut HostState) -> NativeExecutorResult<()> {
    let (_, _, completed) = state.enter_contextual_region();
    if completed.is_none() {
        return Err(NativeExecutorError::Host(
            "completed multi-assignment target has no retained result".into(),
        ));
    }
    Ok(())
}

impl PendingAssign {
    pub(super) fn root_references(&self) -> impl Iterator<Item = NativeValueRef> + '_ {
        std::iter::once(self.current)
            .chain(std::iter::once(self.rhs))
            .chain(self.parents.iter().map(|(parent, _)| *parent))
    }

    fn new(
        state: &mut HostState,
        request: NativeSiteRequest,
        region: usize,
        place: &MirPlace,
        explicit_root: Option<Value>,
        rhs: Value,
        delete: bool,
        allow_init: bool,
    ) -> NativeExecutorResult<Self> {
        let mut segments = Vec::new();
        let root = flatten_place(place, &mut segments)?;
        let create_struct = allow_init
            && matches!(
                segments.first(),
                Some(PlaceSegment::Member(_) | PlaceSegment::DynamicMember(_))
            );
        let current = match explicit_root {
            Some(root_value) => root_value,
            None => state.assignment_root(root.0, create_struct)?,
        };
        let current = state.arena.insert(current);
        let rhs = state.arena.insert(rhs);
        Ok(Self {
            request,
            place: place.clone(),
            root,
            segments,
            parents: Vec::new(),
            current,
            rhs,
            phase: Phase::ReadPrefix(0),
            step: 0,
            region,
            delete,
            allow_init,
        })
    }

    fn advance(&mut self, state: &mut HostState) -> NativeExecutorResult<bool> {
        let current = state.arena.get(self.current)?.clone();
        let (updated, next) = match self.phase {
            Phase::ReadPrefix(index) if index + 1 < self.segments.len() => {
                let segment = self.segments[index].clone();
                let child = read_step(state, current, &segment, self.allow_init)?;
                let parent = self.current;
                let child = state.arena.insert(child);
                self.parents.push((parent, segment));
                (child, Phase::ReadPrefix(index + 1))
            }
            Phase::ReadPrefix(_) => {
                let rhs = state.arena.get(self.rhs)?.clone();
                let segment = self.segments.last().ok_or_else(|| {
                    NativeExecutorError::Host("nonlocal assignment has no place segment".into())
                })?;
                let value = write_step(state, current, segment, rhs, self.delete, self.allow_init)?;
                (state.arena.insert(value), Phase::Writeback)
            }
            Phase::Writeback => {
                let Some((parent, segment)) = self.parents.last() else {
                    return Ok(true);
                };
                let parent_value = state.arena.get(*parent)?.clone();
                let updated = write_step(state, parent_value, segment, current, false, true)?;
                self.parents.pop();
                (state.arena.insert(updated), Phase::Writeback)
            }
        };
        self.current = updated;
        self.phase = next;
        self.step = self.step.checked_add(1).ok_or_else(|| {
            NativeExecutorError::Host("native mutation step ordinal overflowed".into())
        })?;
        Ok(matches!(self.phase, Phase::Writeback) && self.parents.is_empty())
    }
}

fn field_name(state: &mut HostState, segment: &PlaceSegment) -> NativeExecutorResult<String> {
    match segment {
        PlaceSegment::Member(name) => Ok(name.clone()),
        PlaceSegment::DynamicMember(operand) => {
            let value = super::super::operand::materialize_operand(state, operand)?;
            String::try_from(&value).map_err(|error| {
                NativeExecutorError::from(runmat_runtime::runtime_error::semantic_error(
                    "DynamicFieldName",
                    error,
                ))
            })
        }
        PlaceSegment::Index(_) => Err(NativeExecutorError::Host(
            "indexed mutation step has no member name".into(),
        )),
    }
}

fn member_call(
    state: &mut HostState,
    future: super::super::call_suspension::CallFuture,
) -> NativeExecutorResult<Value> {
    let embedded = state.enter_embedded_call();
    let result = (|| {
        if let Some(mut values) =
            super::super::call_suspension::take_completed(state, embedded.as_ref())?
        {
            if values.len() != 1 {
                return Err(NativeExecutorError::Host(
                    "completed member operation did not produce one value".into(),
                ));
            }
            return Ok(values.remove(0));
        }
        let mut values = super::super::call_suspension::begin(state, embedded.clone(), future)?;
        if values.len() != 1 {
            return Err(NativeExecutorError::Host(
                "member operation did not produce one value".into(),
            ));
        }
        Ok(values.remove(0))
    })();
    if let Ok(value) = &result {
        state.cache_embedded_call(embedded.as_ref(), std::slice::from_ref(value));
    }
    let finished = state.finish_embedded_call(embedded.as_ref());
    match (result, finished) {
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error),
        (Ok(value), Ok(())) => Ok(value),
    }
}

fn read_step(
    state: &mut HostState,
    base: Value,
    segment: &PlaceSegment,
    allow_init: bool,
) -> NativeExecutorResult<Value> {
    if let PlaceSegment::Index(indexing) = segment {
        let mut values = super::super::indexing::read_value(state, base, indexing, 1)?;
        return values.pop().ok_or_else(|| {
            NativeExecutorError::Host("nested indexing did not produce one value".into())
        });
    }
    let field = field_name(state, segment)?;
    let runtime = state.runtime.clone();
    let caller = state.function.name.clone();
    member_call(
        state,
        Box::pin(async move {
            let value = runtime
                .scope(runmat_runtime::object::resolve::load_member_with_context(
                    Some(&runtime),
                    base,
                    field,
                    allow_init,
                    Some(&caller),
                ))
                .await?;
            Ok(vec![value])
        }),
    )
}

fn write_step(
    state: &mut HostState,
    base: Value,
    segment: &PlaceSegment,
    rhs: Value,
    delete: bool,
    allow_init: bool,
) -> NativeExecutorResult<Value> {
    if let PlaceSegment::Index(indexing) = segment {
        return super::super::indexing::assign(state, base, indexing, rhs, delete);
    }
    let field = field_name(state, segment)?;
    let runtime = state.runtime.clone();
    let caller = state.function.name.clone();
    member_call(
        state,
        Box::pin(async move {
            let value = runtime
                .scope(
                    runmat_runtime::object::resolve::store_member_traced_in_context(
                        &runtime,
                        base,
                        field,
                        rhs,
                        allow_init,
                        Some(&caller),
                    ),
                )
                .await?;
            Ok(vec![value])
        }),
    )
}

pub(super) fn assign(
    state: &mut HostState,
    place: &MirPlace,
    rhs: Value,
    delete: bool,
    allow_init: bool,
) -> NativeExecutorResult<()> {
    if let MirPlace::Local(local) = place {
        let reference = state.arena.insert(rhs);
        return state.set_local(local.0, reference);
    }
    let (root, reference) = drive(state, place, None, rhs, delete, allow_init)?;
    state.set_local(root.0, reference)
}

pub(super) fn assign_to_root(
    state: &mut HostState,
    place: &MirPlace,
    root_value: Value,
    rhs: Value,
    delete: bool,
    allow_init: bool,
) -> NativeExecutorResult<(MirLocalId, Value)> {
    let (root, reference) = drive(state, place, Some(root_value), rhs, delete, allow_init)?;
    Ok((root, state.arena.get(reference)?.clone()))
}

fn drive(
    state: &mut HostState,
    place: &MirPlace,
    explicit_root: Option<Value>,
    rhs: Value,
    delete: bool,
    allow_init: bool,
) -> NativeExecutorResult<(MirLocalId, NativeValueRef)> {
    if let MirPlace::Local(local) = place {
        return Ok((*local, state.arena.insert(rhs)));
    }
    let request = state.current_request.ok_or_else(|| {
        NativeExecutorError::Host("native mutation has no active statement site".into())
    })?;
    let (region, next_step, _) = state.enter_contextual_region();
    let mut cursor = match state.pending_assign.take() {
        Some(cursor)
            if cursor.request == request
                && cursor.place == *place
                && cursor.region == region
                && cursor.step == next_step
                && cursor.delete == delete
                && cursor.allow_init == allow_init =>
        {
            cursor
        }
        Some(_) => {
            return Err(NativeExecutorError::Host(
                "pending mutation does not match the exact resumed place site".into(),
            ));
        }
        None => PendingAssign::new(
            state,
            request,
            region,
            place,
            explicit_root,
            rhs,
            delete,
            allow_init,
        )?,
    };
    loop {
        let step = cursor.step;
        state.enter_contextual_step(region, step);
        let result = cursor.advance(state);
        let next_step = if result.is_ok() { cursor.step } else { step };
        state.finish_contextual_step(region, step, next_step)?;
        match result {
            Ok(true) => {
                let reference = cursor.current;
                state.finish_contextual_region(region, vec![reference]);
                return Ok((cursor.root, reference));
            }
            Ok(false) => {}
            Err(NativeExecutorError::CallSuspended) => {
                state.pending_assign = Some(cursor);
                return Err(NativeExecutorError::CallSuspended);
            }
            Err(error) => return Err(error),
        }
    }
}
