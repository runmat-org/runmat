//! One original invocation's finite console component. Delegated process
//! subtrees reserve their whole grant before dispatch; local records are
//! admitted before capture allocation. No transcript is copied by this owner.
use crate::{NativeExecutionError, NativeExecutionResult};
use runmat_execution::console::{
    ConsoleLimits, ConsoleReceipt, ConsoleRecord, ConsoleStream, ConsoleTranscript,
    ScopedConsoleTranscript,
};
use runmat_execution_artifact::ProgramExecutionConsoleBudget as Budget;
use runmat_runtime::context::{RuntimeConsoleAdmission, RuntimeConsoleScopeIdentity as Identity};
use sha2::{Digest as _, Sha256};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

const METADATA: u64 = runmat_execution::console::CONSOLE_SCOPE_BYTES;
// Existing canonical scheduler transitions can retain/copy original evidence.
const COPIES: u64 = runmat_execution::console::CONSOLE_RETAINED_COPY_FACTOR;
const TERMINAL_FRAME: u64 = 8 * 1024 * 1024;
const FRAME_OVERHEAD: u64 = 64 * 1024;

#[derive(Clone, Copy, Default)]
struct Usage {
    scopes: u64,
    records: u64,
    text: u64,
    retained: u64,
    encoded: u64,
}

impl Usage {
    fn fits(self, budget: Budget) -> bool {
        self.scopes <= u64::from(budget.max_scopes)
            && self.records <= budget.max_records
            && self.text <= budget.max_text_bytes
            && self.retained <= budget.max_retained_bytes
            && self.encoded <= budget.max_encoded_bytes
    }
    fn plus(self, other: Self) -> NativeExecutionResult<Self> {
        let add = |a: u64, b: u64| {
            a.checked_add(b)
                .ok_or_else(|| invalid("console component overflow"))
        };
        Ok(Self {
            scopes: add(self.scopes, other.scopes)?,
            records: add(self.records, other.records)?,
            text: add(self.text, other.text)?,
            retained: add(self.retained, other.retained)?,
            encoded: add(self.encoded, other.encoded)?,
        })
    }
    fn minus(self, other: Self) -> Self {
        Self {
            scopes: self.scopes - other.scopes,
            records: self.records - other.records,
            text: self.text - other.text,
            retained: self.retained - other.retained,
            encoded: self.encoded - other.encoded,
        }
    }
    fn grant(b: Budget) -> Self {
        Self {
            scopes: u64::from(b.max_scopes),
            records: b.max_records,
            text: b.max_text_bytes,
            retained: b.max_retained_bytes,
            encoded: b.max_encoded_bytes,
        }
    }
}

fn invalid(message: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(message.to_string())
}

/// Replace only the source-console component in an existing owned-input floor.
/// Environment, graph input buffers and parent-monitor storage remain additive.
pub(crate) fn allocation_floor(
    original_floor: u64,
    controls: Option<ConsoleLimits>,
    budget: Option<Budget>,
) -> NativeExecutionResult<u64> {
    let Some(budget) = budget else {
        return Ok(original_floor);
    };
    let source = controls
        .ok_or_else(|| invalid("console grant has no source controls"))?
        .storage_profile()
        .map_err(invalid)?
        .minimum_memory_bytes;
    let subtree = budget.minimum_memory_bytes().map_err(invalid)?;
    original_floor
        .checked_sub(source)
        .and_then(|other| other.checked_add(source.max(subtree)))
        .ok_or_else(|| invalid("console allocation floor overflow or missing source component"))
}

pub(crate) fn artifact_floor(
    original_floor: u64,
    controls: Option<ConsoleLimits>,
    budget: Option<Budget>,
) -> NativeExecutionResult<u64> {
    let source = controls
        .map(|limits| limits.storage_profile())
        .transpose()
        .map_err(invalid)?;
    Ok(original_floor
        .max(source.map_or(0, |source| source.max_encoded_bytes))
        .max(budget.map_or(0, |budget| budget.max_encoded_bytes)))
}

fn capture_usage(records: u64, text: u64, sealed: bool) -> Usage {
    // Lazy source capture grows geometrically. Canonical terminal sealing and
    // decoding move the original Vec to exact capacity, including empty sources.
    let slots = if sealed || records == 0 {
        records
    } else {
        records.next_power_of_two()
    };
    Usage {
        scopes: 1,
        records,
        text,
        retained: (slots * std::mem::size_of::<ConsoleRecord>() as u64 + text + METADATA) * COPIES,
        encoded: records * 256 + text * 6 + METADATA,
    }
}

fn quote_subgrant(
    budget: Budget,
    controls: ConsoleLimits,
    max_in_flight: u32,
) -> NativeExecutionResult<Budget> {
    quote_subgrant_within(budget, controls, max_in_flight, u64::MAX)
}

/// `encoded_cap` bounds the grant's encoded component by the artifact bytes the
/// receiving task reserved, so the grant never exceeds its own reservation.
fn quote_subgrant_within(
    budget: Budget,
    controls: ConsoleLimits,
    max_in_flight: u32,
    encoded_cap: u64,
) -> NativeExecutionResult<Budget> {
    controls.validate().map_err(invalid)?;
    if max_in_flight == 0 {
        return Err(invalid(
            "console delegation has no original concurrency bound",
        ));
    }
    let divisor = u64::from(max_in_flight) + 1;
    let retained = budget.max_retained_bytes / divisor;
    let scopes = (u64::from(budget.max_scopes) / divisor)
        .min((retained / COPIES) / (METADATA * 2))
        .max(1);
    let encoded = (budget.max_encoded_bytes / divisor)
        .min(TERMINAL_FRAME - FRAME_OVERHEAD)
        .min(encoded_cap);
    let metadata = scopes * METADATA;
    let room = (retained / COPIES)
        .checked_sub(metadata)
        .ok_or_else(|| invalid("console subtree lacks metadata storage"))?;
    let wire = encoded
        .checked_sub(metadata + 1024)
        .ok_or_else(|| invalid("console subtree lacks encoded metadata storage"))?;
    let records = u64::from(controls.max_records)
        .min(budget.max_records / divisor)
        .min(room / (std::mem::size_of::<ConsoleRecord>() as u64 * 4))
        .min(wire / 512);
    let text = controls
        .max_text_bytes
        .min(budget.max_text_bytes / divisor)
        .min(room - records * std::mem::size_of::<ConsoleRecord>() as u64 * 2)
        .min((wire - records * 256) / 6);
    let grant = Budget {
        max_scopes: scopes as u32,
        max_records: records,
        max_text_bytes: text,
        max_retained_bytes: retained,
        max_encoded_bytes: encoded,
    };
    grant.validate().map_err(invalid)?;
    Ok(grant)
}

struct Source {
    controls: ConsoleLimits,
    usage: Usage,
    delegated: Option<Budget>,
    sealed: bool,
    receipt: Option<ConsoleReceipt>,
    fingerprint: Option<runmat_execution::Digest>,
}

struct State {
    used: Usage,
    sources: HashMap<Identity, Source>,
}

pub(crate) struct ConsoleRetentionOwner {
    budget: Budget,
    state: Mutex<State>,
}

impl ConsoleRetentionOwner {
    pub(crate) fn new(memory_bytes: u64) -> Arc<Self> {
        Self::owner(Budget::for_driver_allocation(memory_bytes))
    }
    fn owner(budget: Budget) -> Arc<Self> {
        Arc::new(Self {
            budget,
            state: Mutex::new(State {
                used: Usage {
                    encoded: 1024,
                    ..Usage::default()
                },
                sources: HashMap::new(),
            }),
        })
    }
    pub(crate) fn from_subgrant(budget: Budget) -> NativeExecutionResult<Arc<Self>> {
        budget.validate().map_err(invalid)?;
        if budget.max_encoded_bytes > TERMINAL_FRAME - FRAME_OVERHEAD {
            return Err(invalid(
                "console subtree grant exceeds its terminal frame component",
            ));
        }
        Ok(Self::owner(budget))
    }
    pub(crate) fn source_subgrant(
        memory_bytes: u64,
        controls: ConsoleLimits,
        max_in_flight: u32,
    ) -> NativeExecutionResult<Budget> {
        quote_subgrant(
            Budget::for_driver_allocation(memory_bytes),
            controls,
            max_in_flight,
        )
    }
    /// The grant a placed task launches with: quoted from its reserved memory
    /// and bounded by its reserved artifact bytes, which hold the encoded console.
    pub(crate) fn reserved_source_subgrant(
        resources: &runmat_execution::resource::ResourceRequest,
        controls: ConsoleLimits,
    ) -> NativeExecutionResult<Budget> {
        quote_subgrant_within(
            Budget::for_driver_allocation(resources.memory_bytes),
            controls,
            1,
            resources.max_artifact_bytes,
        )
    }
    pub(crate) fn limits(&self) -> Budget {
        self.budget
    }

    pub(crate) fn reserve_scope(
        &self,
        id: Identity,
        controls: ConsoleLimits,
    ) -> NativeExecutionResult<()> {
        controls.validate().map_err(invalid)?;
        let mut state = self.state.lock().expect("console admission poisoned");
        if let Some(original) = state.sources.get(&id) {
            return if original.controls == controls {
                Ok(())
            } else {
                Err(invalid("console scope controls changed"))
            };
        }
        let charge = capture_usage(0, 0, false);
        let used = state.used.plus(charge)?;
        if !used.fits(self.budget) {
            return Err(invalid(
                "console source metadata exceeds its root component",
            ));
        }
        state.sources.try_reserve(1).map_err(invalid)?;
        state.sources.insert(
            id,
            Source {
                controls,
                usage: charge,
                delegated: None,
                sealed: false,
                receipt: None,
                fingerprint: None,
            },
        );
        state.used = used;
        Ok(())
    }

    fn admit(&self, id: Identity, stream: ConsoleStream, text: &str) -> NativeExecutionResult<()> {
        if stream == ConsoleStream::ClearScreen && !text.is_empty() {
            return Err(invalid("console clear-screen record carries text"));
        }
        let mut state = self.state.lock().expect("console admission poisoned");
        let original = state
            .sources
            .get(&id)
            .ok_or_else(|| invalid("console source was not admitted"))?;
        if original.sealed || original.delegated.is_some() {
            return Err(invalid("console source has no local write authority"));
        }
        let records = original
            .usage
            .records
            .checked_add(1)
            .ok_or_else(|| invalid("console records overflow"))?;
        let bytes = original
            .usage
            .text
            .checked_add(text.len() as u64)
            .ok_or_else(|| invalid("console text overflows"))?;
        if records > u64::from(original.controls.max_records)
            || bytes > original.controls.max_text_bytes
        {
            return Err(invalid(
                "console write exceeds its original source controls",
            ));
        }
        let charge = capture_usage(records, bytes, false);
        let used = state.used.minus(original.usage).plus(charge)?;
        if !used.fits(self.budget) {
            return Err(invalid(
                "console write exceeds its admitted subtree component",
            ));
        }
        state.sources.get_mut(&id).expect("original source").usage = charge;
        state.used = used;
        Ok(())
    }

    pub(crate) fn settle_scope(
        &self,
        id: Identity,
        transcript: &ConsoleTranscript,
    ) -> NativeExecutionResult<()> {
        transcript.validate().map_err(invalid)?;
        let receipt = transcript.receipt().map_err(invalid)?;
        let mut state = self.state.lock().expect("console admission poisoned");
        let original = state
            .sources
            .get(&id)
            .ok_or_else(|| invalid("console terminal source was not admitted"))?;
        if original.sealed {
            return if original.receipt.as_ref() == Some(&receipt) {
                Ok(())
            } else {
                Err(invalid("console sealed source changed"))
            };
        }
        if original.delegated.is_some()
            || original.controls != transcript.limits()
            || transcript.records().len() as u64 > original.usage.records
            || transcript.text_bytes() > original.usage.text
        {
            return Err(invalid(
                "console terminal exceeds its original admitted writes",
            ));
        }
        // A failed allocation after pre-admission may retain a shorter prefix.
        let charge = capture_usage(
            transcript.records().len() as u64,
            transcript.text_bytes(),
            true,
        );
        state.used = state.used.minus(original.usage).plus(charge)?;
        let source = state.sources.get_mut(&id).expect("original source");
        source.usage = charge;
        source.sealed = true;
        source.receipt = Some(receipt);
        Ok(())
    }

    /// Leave one partition for the owning source and its fixed envelope. Each
    /// child grant is a total-subtree bound, not a fresh per-descendant allowance.
    pub(crate) fn delegate_scope(
        &self,
        id: Identity,
        controls: ConsoleLimits,
        max_in_flight: u32,
    ) -> NativeExecutionResult<Budget> {
        self.reserve_scope(id, controls)?;
        let mut state = self.state.lock().expect("console admission poisoned");
        let original = state.sources.get(&id).expect("registered source");
        if let Some(grant) = original.delegated {
            return Ok(grant);
        }
        if original.sealed || original.usage.records != 0 {
            return Err(invalid("written console source cannot be delegated"));
        }
        let grant = quote_subgrant(self.budget, controls, max_in_flight)?;
        let used = state.used.minus(original.usage).plus(Usage::grant(grant))?;
        if !used.fits(self.budget) {
            return Err(invalid(
                "console subtree reservation exceeds its original root component before dispatch",
            ));
        }
        let source = state.sources.get_mut(&id).expect("registered source");
        source.usage = Usage::grant(grant);
        source.delegated = Some(grant);
        state.used = used;
        Ok(grant)
    }

    fn subtree_usage(
        id: Identity,
        root: Option<&ConsoleTranscript>,
        scopes: &[ScopedConsoleTranscript],
    ) -> NativeExecutionResult<Usage> {
        if scopes.len() + usize::from(root.is_some())
            > runmat_execution::console::MAX_CONSOLE_SCOPES
        {
            return Err(invalid(
                "console subtree scope inventory exceeds its canonical bound",
            ));
        }
        let mut ids = HashSet::new();
        ids.try_reserve(scopes.len() + usize::from(root.is_some()))
            .map_err(invalid)?;
        let mut usage = Usage {
            encoded: 1024,
            ..Usage::default()
        };
        if let Some(root) = root {
            root.validate().map_err(invalid)?;
            ids.insert(id);
            usage = usage.plus(capture_usage(
                root.records().len() as u64,
                root.text_bytes(),
                true,
            ))?;
        }
        for scope in scopes {
            let key = Identity {
                scope_id: scope.scope_id,
                pool_id: scope.pool_id,
                task_id: scope.task_id,
                attempt_id: scope.attempt_id,
            };
            if !ids.insert(key) {
                return Err(invalid("console subtree repeats an original source"));
            }
            scope.transcript.validate().map_err(invalid)?;
            usage = usage.plus(capture_usage(
                scope.transcript.records().len() as u64,
                scope.transcript.text_bytes(),
                true,
            ))?;
        }
        Ok(usage)
    }

    fn subtree_fingerprint(
        id: Identity,
        root: Option<&ConsoleTranscript>,
        scopes: &[ScopedConsoleTranscript],
    ) -> NativeExecutionResult<runmat_execution::Digest> {
        let mut hash = Sha256::new();
        hash.update(b"runmat-console-subtree-v1");
        let mut append =
            |identity: Identity, transcript: &ConsoleTranscript| -> NativeExecutionResult<()> {
                hash.update(identity.scope_id.bytes());
                hash.update(identity.pool_id.bytes());
                hash.update(identity.task_id.bytes());
                hash.update(identity.attempt_id.bytes());
                // A small original receipt binds order, streams, text, completion
                // and controls without retaining another transcript or workspace.
                hash.update(
                    serde_json::to_vec(&transcript.receipt().map_err(invalid)?).map_err(invalid)?,
                );
                Ok(())
            };
        if let Some(root) = root {
            append(id, root)?;
        }
        for scope in scopes {
            append(
                Identity {
                    scope_id: scope.scope_id,
                    pool_id: scope.pool_id,
                    task_id: scope.task_id,
                    attempt_id: scope.attempt_id,
                },
                &scope.transcript,
            )?;
        }
        Ok(runmat_execution::Digest::sha256(hash.finalize()))
    }

    pub(crate) fn validate_subtree(
        &self,
        id: Identity,
        root: Option<&ConsoleTranscript>,
        scopes: &[ScopedConsoleTranscript],
    ) -> NativeExecutionResult<()> {
        if root.is_none() && scopes.is_empty() {
            return Ok(());
        }
        let state = self.state.lock().expect("console admission poisoned");
        let source = state
            .sources
            .get(&id)
            .ok_or_else(|| invalid("console reply has no pre-dispatch subtree reservation"))?;
        let grant = source
            .delegated
            .ok_or_else(|| invalid("console reply has no delegated grant"))?;
        if root.is_some_and(|root| root.limits() != source.controls) {
            return Err(invalid("console reply controls changed"));
        }
        if !Self::subtree_usage(id, root, scopes)?.fits(grant) {
            return Err(invalid(
                "console reply exceeds its original delegated subtree grant",
            ));
        }
        if source.sealed && source.fingerprint != Some(Self::subtree_fingerprint(id, root, scopes)?)
        {
            return Err(invalid("console original sealed subtree changed"));
        }
        Ok(())
    }

    pub(crate) fn settle_subtree_scope(
        &self,
        id: Identity,
        root: Option<&ConsoleTranscript>,
        scopes: &[ScopedConsoleTranscript],
    ) -> NativeExecutionResult<()> {
        // Absent evidence cannot refund a lease for a potentially late prefix.
        if root.is_none() && scopes.is_empty() {
            return Ok(());
        }
        self.validate_subtree(id, root, scopes)?;
        let charge = Self::subtree_usage(id, root, scopes)?;
        let fingerprint = Self::subtree_fingerprint(id, root, scopes)?;
        let mut state = self.state.lock().expect("console admission poisoned");
        let original = state.sources.get(&id).expect("validated subtree");
        if original.sealed {
            return Ok(());
        }
        state.used = state.used.minus(original.usage).plus(charge)?;
        let source = state.sources.get_mut(&id).expect("validated subtree");
        source.usage = charge;
        source.sealed = true;
        source.fingerprint = Some(fingerprint);
        Ok(())
    }

    pub(crate) fn restore_subtree(
        &self,
        id: Identity,
        controls: ConsoleLimits,
        root: Option<&ConsoleTranscript>,
        scopes: &[ScopedConsoleTranscript],
        max_in_flight: u32,
    ) -> NativeExecutionResult<()> {
        if root.is_none() && scopes.is_empty() {
            self.delegate_scope(id, controls, max_in_flight)?;
            return Ok(());
        }
        let charge = Self::subtree_usage(id, root, scopes)?;
        let fingerprint = Self::subtree_fingerprint(id, root, scopes)?;
        let mut state = self.state.lock().expect("console admission poisoned");
        let used = state.used.plus(charge)?;
        if !used.fits(self.budget) {
            return Err(invalid(
                "restored console exceeds its original root component",
            ));
        }
        state.sources.try_reserve(1).map_err(invalid)?;
        state.sources.insert(
            id,
            Source {
                controls,
                usage: charge,
                delegated: Some(self.budget),
                sealed: true,
                receipt: None,
                fingerprint: Some(fingerprint),
            },
        );
        state.used = used;
        Ok(())
    }
}

impl RuntimeConsoleAdmission for ConsoleRetentionOwner {
    fn quote_delegate(&self, limits: ConsoleLimits, max_in_flight: u32) -> Result<Budget, String> {
        quote_subgrant(self.budget, limits, max_in_flight).map_err(|e| e.to_string())
    }
    fn reserve(&self, id: Identity, limits: ConsoleLimits) -> Result<(), String> {
        self.reserve_scope(id, limits).map_err(|e| e.to_string())
    }
    fn admit_record(&self, id: Identity, stream: ConsoleStream, text: &str) -> Result<(), String> {
        self.admit(id, stream, text).map_err(|e| e.to_string())
    }
    fn settle(&self, id: Identity, transcript: &ConsoleTranscript) -> Result<(), String> {
        self.settle_scope(id, transcript).map_err(|e| e.to_string())
    }
    fn delegate(
        &self,
        id: Identity,
        limits: ConsoleLimits,
        max_in_flight: u32,
    ) -> Result<Budget, String> {
        self.delegate_scope(id, limits, max_in_flight)
            .map_err(|e| e.to_string())
    }
    fn settle_subtree(
        &self,
        id: Identity,
        root: Option<&ConsoleTranscript>,
        scopes: &[ScopedConsoleTranscript],
    ) -> Result<(), String> {
        self.settle_subtree_scope(id, root, scopes)
            .map_err(|e| e.to_string())
    }
    fn max_scopes(&self) -> u32 {
        self.budget.max_scopes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn implicit_child_quote_preserves_inputs_and_matches_dispatch_without_clamping() {
        let owner = ConsoleRetentionOwner::new(1024 * 1024 * 1024);
        let controls = ConsoleLimits {
            max_records: 4096,
            max_text_bytes: 1024 * 1024,
        };
        let quote = owner.quote_delegate(controls, 4).unwrap();
        let source = controls.storage_profile().unwrap();
        let other_inputs_environment_monitor = 17 * 1024 * 1024
            + 137_000
            + runmat_process_host::ipc::WORKER_PARENT_MONITOR_STACK_BYTES;
        let old_floor = other_inputs_environment_monitor + source.minimum_memory_bytes;
        let corrected = allocation_floor(old_floor, Some(controls), Some(quote)).unwrap();
        assert_eq!(
            corrected,
            other_inputs_environment_monitor
                + source
                    .minimum_memory_bytes
                    .max(quote.minimum_memory_bytes().unwrap())
        );
        assert!(corrected > old_floor);
        assert_eq!(
            owner.delegate_scope(identity(91), controls, 4).unwrap(),
            quote
        );
        assert_eq!(
            artifact_floor(12_000, Some(controls), Some(quote)).unwrap(),
            source.max_encoded_bytes.max(quote.max_encoded_bytes)
        );
        assert_eq!(allocation_floor(old_floor, None, None).unwrap(), old_floor);
        assert_eq!(artifact_floor(12_000, None, None).unwrap(), 12_000);
    }
    fn identity(n: u64) -> Identity {
        let bytes = n.to_be_bytes();
        Identity {
            scope_id: runmat_execution::ExecutionScopeId::derive(&[b"s"]),
            pool_id: runmat_execution::PoolId::derive(&[b"p"]),
            task_id: runmat_execution::TaskId::derive(&[&bytes]),
            attempt_id: runmat_execution::identity::AttemptId::derive(&[&bytes]),
        }
    }
    #[test]
    fn quiet_sources_only_spend_metadata_and_actual_writes_are_admitted_before_capture() {
        let owner = ConsoleRetentionOwner::new(1024 * 1024 * 1024);
        let limits = ConsoleLimits {
            max_records: 4096,
            max_text_bytes: 1024 * 1024,
        };
        for n in 0..100 {
            owner.reserve_scope(identity(n), limits).unwrap();
            let empty = runmat_execution::console::ConsoleCapture::new(limits)
                .unwrap()
                .finish(true);
            owner.settle_scope(identity(n), &empty).unwrap();
        }
        let state = owner.state.lock().unwrap();
        assert_eq!(state.used.records, 0);
        assert_eq!(state.used.text, 0);
        assert_eq!(state.used.retained, 100 * METADATA * COPIES);
        drop(state);
        let id = identity(101);
        owner.reserve_scope(id, limits).unwrap();
        owner.admit(id, ConsoleStream::Stdout, "original").unwrap();
        assert!(owner
            .admit(
                id,
                ConsoleStream::Stdout,
                &"x".repeat(limits.max_text_bytes as usize)
            )
            .is_err());
    }
    #[test]
    fn delegated_quiet_siblings_fit_and_forged_unreserved_prefix_does_not() {
        let owner = ConsoleRetentionOwner::new(1024 * 1024 * 1024);
        let limits = ConsoleLimits {
            max_records: 4096,
            max_text_bytes: 1024 * 1024,
        };
        let first = owner.delegate_scope(identity(1), limits, 2).unwrap();
        let second = owner.delegate_scope(identity(2), limits, 2).unwrap();
        assert!(first.max_text_bytes <= limits.max_text_bytes);
        assert!(second.max_encoded_bytes < TERMINAL_FRAME);
        let child = ConsoleRetentionOwner::from_subgrant(first).unwrap();
        child.reserve_scope(identity(1), limits).unwrap();
        let empty = runmat_execution::console::ConsoleCapture::new(limits)
            .unwrap()
            .finish(true);
        child.settle_scope(identity(1), &empty).unwrap();
        owner
            .settle_subtree_scope(identity(1), Some(&empty), &[])
            .unwrap();
        assert!(owner
            .validate_subtree(identity(99), Some(&empty), &[])
            .is_err());
        let mut changed = runmat_execution::console::ConsoleCapture::new(limits).unwrap();
        changed
            .push(ConsoleStream::Stdout, "changed".into())
            .unwrap();
        assert!(owner
            .validate_subtree(identity(1), Some(&changed.finish(true)), &[])
            .is_err());
    }

    #[test]
    fn placed_task_grant_fits_its_reserved_artifact_bytes() {
        let limits = ConsoleLimits {
            max_records: 4,
            max_text_bytes: 100,
        };
        let resources = runmat_execution::resource::ResourceRequest {
            cpu_millicores: 1000,
            memory_bytes: 1024 * 1024 * 1024,
            scratch_bytes: 0,
            max_wall_millis: 30_000,
            max_artifact_bytes: 1024 * 1024,
            max_egress_bytes: 0,
            max_relay_bytes: 0,
            accelerators: vec![],
            required_capabilities: Default::default(),
        };
        let grant = ConsoleRetentionOwner::reserved_source_subgrant(&resources, limits).unwrap();
        assert!(grant.max_encoded_bytes <= resources.max_artifact_bytes);
        assert!(grant.minimum_memory_bytes().unwrap() <= resources.memory_bytes);
        assert_eq!(grant.max_records, 4);
        assert_eq!(grant.max_text_bytes, 100);
        let unbounded =
            ConsoleRetentionOwner::source_subgrant(resources.memory_bytes, limits, 1).unwrap();
        assert!(unbounded.max_encoded_bytes > resources.max_artifact_bytes);
    }

    #[test]
    fn small_original_tasks_get_a_finite_metadata_and_payload_subgrant() {
        let limits = ConsoleLimits {
            max_records: 4096,
            max_text_bytes: 1024 * 1024,
        };
        let grant = ConsoleRetentionOwner::source_subgrant(8 * 1024 * 1024, limits, 1).unwrap();
        assert!(grant.max_scopes > 0 && grant.max_records > 0 && grant.max_text_bytes > 0);
        let owner = ConsoleRetentionOwner::from_subgrant(grant).unwrap();
        owner.reserve_scope(identity(1), limits).unwrap();
        owner
            .admit(identity(1), ConsoleStream::Stdout, "ordinary output")
            .unwrap();
    }
}
