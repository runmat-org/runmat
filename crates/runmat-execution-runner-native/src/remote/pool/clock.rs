//! Wave-clock checkpoint barrier and original observer ownership.

use std::sync::Arc;

use runmat_execution::security::{
    NodeClockEndV1, NodeClockStartV1, MAX_CLOCK_OBSERVATIONS_PER_NODE,
};
use runmat_execution::TaskId;
use tokio::sync::{mpsc, oneshot};

use super::RemotePoolDriver;
use crate::remote::clock_observer::{CertifiedWave, ClockCommand, RemoteWaveIdentity};
use crate::{NativeExecutionError, NativeExecutionResult};

impl RemotePoolDriver {
    pub(in crate::remote) fn install_clock_commands(
        &self,
        sender: mpsc::Sender<ClockCommand>,
    ) -> NativeExecutionResult<()> {
        let mut installed = self.clock_commands.lock().expect("clock bridge poisoned");
        if installed.replace(sender).is_some() {
            return Err(protocol("remote clock bridge was installed twice"));
        }
        Ok(())
    }

    fn clock_sender(&self) -> Option<mpsc::Sender<ClockCommand>> {
        self.clock_commands
            .lock()
            .expect("clock bridge poisoned")
            .clone()
    }

    pub(in crate::remote) async fn begin_clock_wave(
        &self,
        wave: RemoteWaveIdentity,
        planned_tasks: &[TaskId],
    ) -> NativeExecutionResult<(Option<NodeClockStartV1>, Option<u64>, Vec<TaskId>, bool)> {
        let Some(sender) = self.clock_sender() else {
            return Ok((None, None, Vec::new(), false));
        };
        // Serialize only original-start registration, not the waves themselves.
        let _admission = self.wave_admission.lock().await;
        // Snapshot scheduler identity before taking any timing lock. A task
        // already known to the driver is never eligible for a replacement
        // clock start merely because its original interval is missing.
        let already_admitted = {
            let driver = self.driver.lock().expect("remote driver poisoned");
            planned_tasks
                .iter()
                .map(|task| driver.task_record(*task).is_ok())
                .collect::<Vec<_>>()
        };
        let (new_tasks, observed_wave) = {
            let _clock_state = self.clock_state.lock().expect("clock state poisoned");
            let starts = self.clock_starts.lock().expect("clock starts poisoned");
            let ends = self.clock_ends.lock().expect("clock ends poisoned");
            let intervals = self
                .task_wall_intervals
                .lock()
                .expect("task wall intervals poisoned");
            let assigned = self
                .task_clocks
                .lock()
                .expect("task clock bindings poisoned");
            let waves = self.wave_clocks.lock().expect("wave clocks poisoned");
            let certified = self
                .certified_waves
                .lock()
                .expect("certified waves poisoned");
            if planned_tasks
                .iter()
                .zip(&already_admitted)
                .any(|(task, known)| {
                    intervals.get(task).is_none() && (*known || assigned.contains_key(task))
                })
            {
                return Err(protocol("admitted task lacks its original wall authority"));
            }
            if let Some(id) = waves.get(&wave) {
                let start = starts
                    .get(id)
                    .ok_or_else(|| protocol("restored wave lacks its pinned original observer"))?;
                if let Some(end) = ends.get(id) {
                    let wall = end.measured_wall_millis(start).map_err(protocol)?;
                    if planned_tasks.iter().any(|task| {
                        !matches!(intervals.get(task),
                        Some(super::timing::TaskWallInterval::Completed(saved)) if *saved == wall)
                    }) {
                        return Err(protocol(
                            "closed original wave differs from committed task timing",
                        ));
                    }
                    return Ok((None, Some(wall), Vec::new(), true));
                }
                return Ok((Some(start.clone()), None, planned_tasks.to_vec(), false));
            }
            if let Some(summary) = certified.get(&wave) {
                let mut tasks = planned_tasks.to_vec();
                tasks.sort_unstable();
                if tasks != summary.tasks || tasks.iter().any(|task| {
                    !matches!(intervals.get(task), Some(super::timing::TaskWallInterval::Completed(wall)) if *wall == summary.measured_wall_millis)
                }) {
                    return Err(protocol("certified replay differs from original completed wave"));
                }
                return Ok((None, Some(summary.measured_wall_millis), Vec::new(), false));
            }
            let mut new_tasks = Vec::new();
            let mut pending_tasks = Vec::new();
            let mut pending_owner = None;
            let mut needs_release = false;
            for (task, already_admitted) in planned_tasks.iter().zip(already_admitted) {
                match intervals.get(task) {
                    None if already_admitted || assigned.contains_key(task) => {
                        return Err(protocol("admitted task lacks its original wall authority"))
                    }
                    None => new_tasks.push(*task),
                    Some(super::timing::TaskWallInterval::ObservedPending(owner))
                        if assigned.get(task) == Some(owner)
                            && pending_owner.is_none_or(|prior| prior == *owner) =>
                    {
                        pending_owner = Some(*owner);
                        pending_tasks.push(*task);
                    }
                    Some(super::timing::TaskWallInterval::Completed(wall)) => {
                        let is_certified = certified.values().any(|summary| {
                            summary.measured_wall_millis == *wall && summary.tasks.contains(task)
                        });
                        let signed_unreleased = assigned.get(task).is_some_and(|id| {
                            ends.get(id).is_some_and(|end| {
                                starts.get(id).is_some_and(|start| {
                                    end.measured_wall_millis(start).ok() == Some(*wall)
                                })
                            })
                        });
                        if !is_certified && !signed_unreleased {
                            return Err(protocol(
                                "regrouped committed task lacks original wall authority",
                            ));
                        }
                        needs_release |= signed_unreleased;
                    }
                    _ => {
                        return Err(protocol(
                            "regrouped wave has unresolved original task timing",
                        ))
                    }
                }
            }
            if let Some(owner) = pending_owner {
                if !new_tasks.is_empty() {
                    return Err(protocol(
                        "regrouped wave combines unresolved and new task owners",
                    ));
                }
                pending_tasks.sort_unstable();
                let key = RemoteWaveIdentity::new(self.scope_id, &pending_tasks)?;
                if waves.get(&key) != Some(&owner)
                    || assigned.iter().filter(|(_, id)| **id == owner).count()
                        != pending_tasks.len()
                {
                    return Err(protocol(
                        "restored pending subset differs from original wave",
                    ));
                }
                return starts
                    .get(&owner)
                    .cloned()
                    .map(|start| (Some(start), None, pending_tasks, false))
                    .ok_or_else(|| protocol("restored pending subset lost original start"));
            }
            if new_tasks.is_empty() {
                return Ok((None, None, Vec::new(), needs_release));
            }
            let observed_wave = RemoteWaveIdentity::new(self.scope_id, &new_tasks)?;
            if let Some(id) = waves.get(&observed_wave) {
                return starts
                    .get(id)
                    .cloned()
                    .map(|start| (Some(start), None, new_tasks, false))
                    .ok_or_else(|| protocol("new task subset lost its original observer"));
            }
            if starts.len() >= MAX_CLOCK_OBSERVATIONS_PER_NODE {
                return Err(protocol(
                    "remote clock observation slots are full before wave admission",
                ));
            }
            (new_tasks, observed_wave)
        };
        let (reply, received) = oneshot::channel();
        sender
            .send(ClockCommand::Begin(reply))
            .await
            .map_err(protocol)?;
        let start = received.await.map_err(protocol)??;
        let id = start.owner.observation_id;
        let _clock_state = self.clock_state.lock().expect("clock state poisoned");
        let mut starts = self.clock_starts.lock().expect("clock starts poisoned");
        let mut waves = self.wave_clocks.lock().expect("wave clocks poisoned");
        if starts.contains_key(&id) || waves.values().any(|existing| *existing == id) {
            return Err(protocol("original node reused an observation identifier"));
        }
        starts.insert(id, start.clone());
        waves.insert(observed_wave, id);
        Ok((Some(start), None, new_tasks, false))
    }

    pub(in crate::remote) fn completed_tasks_certified(&self, tasks: &[TaskId]) -> bool {
        let intervals = self
            .task_wall_intervals
            .lock()
            .expect("task wall intervals poisoned");
        let certified = self
            .certified_waves
            .lock()
            .expect("certified waves poisoned");
        tasks.iter().all(|task| {
            matches!(intervals.get(task),
            Some(super::timing::TaskWallInterval::Completed(wall)) if
            certified.values().any(|summary| summary.measured_wall_millis == *wall
                && summary.tasks.contains(task)))
        })
    }

    pub(in crate::remote) fn arm_clock_wave(
        &self,
        start: &NodeClockStartV1,
        tasks: &[TaskId],
    ) -> NativeExecutionResult<()> {
        let _clock_state = self.clock_state.lock().expect("clock state poisoned");
        if tasks.is_empty() {
            return Err(protocol("remote wave has no task inventory"));
        }
        let id = start.owner.observation_id;
        let starts = self.clock_starts.lock().expect("clock starts poisoned");
        if starts.get(&id) != Some(start)
            || !self
                .wave_clocks
                .lock()
                .expect("wave clocks poisoned")
                .values()
                .any(|owner| *owner == id)
        {
            return Err(protocol(
                "remote wave start differs from its original pinned owner",
            ));
        }
        // Completion and admission acquire these in the same order. The outer
        // clock-state guard serializes signed start/wave metadata separately.
        let mut intervals = self
            .task_wall_intervals
            .lock()
            .expect("remote task wall intervals poisoned");
        let mut assigned = self
            .task_clocks
            .lock()
            .expect("task clock bindings poisoned");
        let mut barrier = self
            .uncheckpointed_tasks
            .lock()
            .expect("clock launch barrier poisoned");
        if tasks.iter().all(|task| assigned.get(task) == Some(&id)) {
            return tasks
                .iter()
                .all(|task| {
                    matches!(
                        intervals.get(task),
                        Some(super::timing::TaskWallInterval::ObservedPending(owner))
                            if *owner == id
                    )
                })
                .then_some(())
                .ok_or_else(|| protocol("restored wave lost its original wall interval"));
        }
        if assigned.values().any(|prior| *prior == id) {
            return Err(protocol(
                "restored wave task binding differs from original observer",
            ));
        }
        if tasks
            .iter()
            .any(|task| assigned.contains_key(task) || barrier.contains(task))
        {
            return Err(protocol("remote wave repeats an original clock owner"));
        }
        if tasks.iter().any(|task| intervals.contains_key(task)) {
            return Err(protocol(
                "remote wave task already has original wall authority",
            ));
        }
        for task in tasks {
            assigned.insert(*task, id);
            barrier.insert(*task);
            intervals.insert(*task, super::timing::TaskWallInterval::ObservedPending(id));
        }
        Ok(())
    }

    pub(in crate::remote) async fn acknowledge_clock_wave(
        self: &Arc<Self>,
        tasks: &[TaskId],
    ) -> NativeExecutionResult<()> {
        let Some(sender) = self.clock_sender() else {
            return Ok(());
        };
        let (reply, received) = oneshot::channel();
        sender
            .send(ClockCommand::Checkpoint(reply))
            .await
            .map_err(protocol)?;
        received.await.map_err(protocol)??;
        let launch = {
            let mut barrier = self
                .uncheckpointed_tasks
                .lock()
                .expect("clock launch barrier poisoned");
            let mut held = self.held_launches.lock().expect("held launches poisoned");
            for task in tasks {
                barrier.remove(task);
            }
            let mut released = Vec::new();
            held.retain(|request| {
                if barrier.contains(&request.task_id) {
                    true
                } else {
                    released.push(request.clone());
                    false
                }
            });
            released
        };
        for request in launch {
            self.launch(request);
        }
        Ok(())
    }

    pub(in crate::remote) async fn end_clock_wave(
        &self,
        start: &NodeClockStartV1,
        tasks: &[TaskId],
    ) -> NativeExecutionResult<Option<u64>> {
        let Some(sender) = self.clock_sender() else {
            return Ok(None);
        };
        let cached = self
            .clock_ends
            .lock()
            .expect("clock ends poisoned")
            .get(&start.owner.observation_id)
            .cloned();
        let already_closed = cached.is_some();
        let end: NodeClockEndV1 = if let Some(end) = cached {
            end
        } else {
            let (reply, received) = oneshot::channel();
            sender
                .send(ClockCommand::End {
                    start: Box::new(start.clone()),
                    respond: reply,
                })
                .await
                .map_err(protocol)?;
            received.await.map_err(protocol)??
        };
        let wall = end.measured_wall_millis(start).map_err(protocol)?;
        let _clock_state = self.clock_state.lock().expect("clock state poisoned");
        let id = start.owner.observation_id;
        let assigned = self
            .task_clocks
            .lock()
            .expect("task clock bindings poisoned");
        if tasks.iter().any(|task| assigned.get(task) != Some(&id)) {
            return Err(protocol("remote wave clock does not own every task"));
        }
        drop(assigned);
        let mut intervals = self
            .task_wall_intervals
            .lock()
            .expect("task wall intervals poisoned");
        for task in tasks {
            if !(matches!(intervals.get(task), Some(super::timing::TaskWallInterval::ObservedPending(owner)) if *owner == id)
                || already_closed
                    && matches!(intervals.get(task),
                    Some(super::timing::TaskWallInterval::Completed(saved)) if *saved == wall))
            {
                return Err(protocol(
                    "task interval is not pending under its original observer",
                ));
            }
        }
        for task in tasks {
            intervals.insert(*task, super::timing::TaskWallInterval::Completed(wall));
        }
        self.clock_ends
            .lock()
            .expect("clock ends poisoned")
            .insert(id, end);
        Ok(Some(wall))
    }

    pub(in crate::remote) async fn checkpoint_closed_wave(&self) -> NativeExecutionResult<()> {
        let sender = self
            .clock_sender()
            .ok_or_else(|| protocol("remote wave has no clock owner"))?;
        let (reply, received) = oneshot::channel();
        sender
            .send(ClockCommand::Close(reply))
            .await
            .map_err(protocol)?;
        received.await.map_err(protocol)??;
        Ok(())
    }

    pub(in crate::remote) fn closed_clock_release_entries(
        &self,
    ) -> NativeExecutionResult<
        Vec<runmat_execution_transport_native::control::ClockObservationReleaseEntryV1>,
    > {
        let starts = self.clock_starts.lock().expect("clock starts poisoned");
        let ends = self.clock_ends.lock().expect("clock ends poisoned");
        ends.iter()
            .map(|(id, end)| {
                let start = starts
                    .get(id)
                    .ok_or_else(|| protocol("closed clock has no original start"))?;
                end.measured_wall_millis(start).map_err(protocol)?;
                Ok(
                    runmat_execution_transport_native::control::ClockObservationReleaseEntryV1 {
                        observation_id: *id,
                        start_receipt_digest: start.receipt_digest().map_err(protocol)?,
                        end_receipt_digest: end.receipt_digest().map_err(protocol)?,
                    },
                )
            })
            .collect()
    }

    pub(in crate::remote) fn prune_released_clock_entries(
        &self,
        entries: &[runmat_execution_transport_native::control::ClockObservationReleaseEntryV1],
    ) -> NativeExecutionResult<()> {
        let _clock_state = self.clock_state.lock().expect("clock state poisoned");
        let mut starts = self.clock_starts.lock().expect("clock starts poisoned");
        let mut ends = self.clock_ends.lock().expect("clock ends poisoned");
        let intervals = self
            .task_wall_intervals
            .lock()
            .expect("task wall intervals poisoned");
        let mut assigned = self
            .task_clocks
            .lock()
            .expect("task clock bindings poisoned");
        let mut certified = Vec::with_capacity(entries.len());
        for entry in entries {
            let start = starts
                .get(&entry.observation_id)
                .ok_or_else(|| protocol("released clock start was already removed"))?;
            let end = ends
                .get(&entry.observation_id)
                .ok_or_else(|| protocol("released clock end was already removed"))?;
            if start.receipt_digest().map_err(protocol)? != entry.start_receipt_digest
                || end.receipt_digest().map_err(protocol)? != entry.end_receipt_digest
                || assigned.iter().any(|(task, id)| {
                    *id == entry.observation_id
                        && !matches!(
                            intervals.get(task),
                            Some(super::timing::TaskWallInterval::Completed(_))
                        )
                })
            {
                return Err(protocol(
                    "released clock differs from finalized task accounting",
                ));
            }
            let wall = end.measured_wall_millis(start).map_err(protocol)?;
            let mut tasks = assigned
                .iter()
                .filter_map(|(task, id)| (*id == entry.observation_id).then_some(*task))
                .collect::<Vec<_>>();
            tasks.sort_unstable();
            let wave = RemoteWaveIdentity::new(self.scope_id, &tasks)?;
            if self
                .wave_clocks
                .lock()
                .expect("wave clocks poisoned")
                .get(&wave)
                != Some(&entry.observation_id)
                || tasks.iter().any(|task| {
                    !matches!(intervals.get(task),
                    Some(super::timing::TaskWallInterval::Completed(saved)) if *saved == wall)
                })
            {
                return Err(protocol(
                    "released wave differs from original certified task inventory",
                ));
            }
            if certified.iter().any(|(prior, _)| *prior == wave) {
                return Err(protocol("release repeats a certified wave"));
            }
            certified.push((
                wave,
                CertifiedWave {
                    tasks,
                    measured_wall_millis: wall,
                },
            ));
        }
        {
            let mut summaries = self
                .certified_waves
                .lock()
                .expect("certified waves poisoned");
            if certified
                .iter()
                .any(|(wave, _)| summaries.contains_key(wave))
            {
                return Err(protocol("certified wave was released twice"));
            }
            summaries.extend(certified);
        }
        for entry in entries {
            starts.remove(&entry.observation_id);
            ends.remove(&entry.observation_id);
            assigned.retain(|_, id| *id != entry.observation_id);
            self.wave_clocks
                .lock()
                .expect("wave clocks poisoned")
                .retain(|_, id| *id != entry.observation_id);
        }
        Ok(())
    }
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
