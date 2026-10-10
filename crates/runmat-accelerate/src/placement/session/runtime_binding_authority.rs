//! Nonpersistable identity for selected runtime bindings in one placement session.
//! It says nothing about an installed artifact or cross-process code equivalence.

use runmat_builtins::BuiltinBindingIdentity;
use runmat_execution::{
    Digest, PlacementExecutionAuthorityRevision, PlacementInvalidation, PlacementRevision,
    ProgramRevision,
};
use runmat_runtime::context::{
    RuntimeVmFusionGenerationPermit, VmFusionRegionRegistration, VmFusionSemanticIdentity,
    VmFusionSessionRegistration,
};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::{cell::RefCell, collections::BTreeMap};

use super::PlacementSessionConfig;

// The runtime's own context-local identity uses the same checked AtomicU64
// pattern on native and wasm32. No pointer or clock enters an authority key.
static NEXT_SESSION: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
enum ExecutionBindingKey {
    DirectBuiltin(BuiltinBindingIdentity),
}

struct BindingRegistration {
    generation: Digest,
    authority: Digest,
}

pub(super) struct SessionRuntimeBindingAuthority {
    id: u64,
    registrations: RefCell<BTreeMap<ExecutionBindingKey, BindingRegistration>>,
    vm: RefCell<BTreeMap<VmSemanticKey, VmSemanticBucket>>,
}

#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
struct VmSemanticKey {
    semantic: VmFusionSemanticIdentity,
    program: Option<Digest>,
}

struct VmSemanticBucket {
    revision: PlacementRevision,
    generations: BTreeMap<u64, VmGeneration>,
}

struct VmGeneration {
    liveness: Weak<()>,
    permit: Arc<VmGenerationPermit>,
}

struct VmGenerationPermit(AtomicBool);

impl VmGenerationPermit {
    fn current() -> Self {
        Self(AtomicBool::new(true))
    }

    fn revoke(&self) {
        self.0.store(false, Ordering::Release);
    }
}

impl RuntimeVmFusionGenerationPermit for VmGenerationPermit {
    fn is_current(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

impl SessionRuntimeBindingAuthority {
    pub(super) fn new() -> Self {
        let id = NEXT_SESSION
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .expect("placement runtime binding session identity space exhausted");
        Self {
            id,
            registrations: RefCell::new(BTreeMap::new()),
            vm: RefCell::new(BTreeMap::new()),
        }
    }

    pub(super) fn revision_and_replaced(
        &self,
        program: Option<ProgramRevision>,
        binding: BuiltinBindingIdentity,
        registration: Digest,
        config: PlacementSessionConfig,
    ) -> (PlacementRevision, Option<Digest>) {
        let authority = Digest::sha256_serialized(&(
            "runmat.runtime-binding-session.v1",
            self.id,
            binding,
            registration,
        ))
        .expect("typed local runtime binding authority serializes");
        let replaced = self.replace(
            ExecutionBindingKey::DirectBuiltin(binding),
            registration,
            authority,
        );
        (
            super::super::revision::whole_work_runtime_binding_session(program, authority, config),
            replaced,
        )
    }

    pub(super) fn vm_registration(
        &self,
        program: Option<ProgramRevision>,
        region: VmFusionRegionRegistration,
        liveness: Weak<()>,
        config: PlacementSessionConfig,
    ) -> Option<VmFusionSessionRegistration> {
        liveness.upgrade()?;
        let program_identity = program
            .as_ref()
            .map(ProgramRevision::identity_digest)
            .transpose()
            .ok()?;
        let key = VmSemanticKey {
            semantic: region.semantic_identity(),
            program: program_identity,
        };
        let authority = Digest::sha256_serialized(&(
            "runmat.vm-fusion-region-learning.v2",
            self.id,
            key.semantic.digest(),
            key.program,
        ))
        .expect("typed VM fusion session authority serializes");
        let mut vm = self.vm.borrow_mut();
        sweep_vm(&mut vm);
        let live_count = vm
            .values()
            .map(|bucket| bucket.generations.len())
            .sum::<usize>();
        let is_existing = vm
            .get(&key)
            .is_some_and(|bucket| bucket.generations.contains_key(&region.installation()));
        if !is_existing && live_count >= config.max_cached_decisions {
            return None;
        }
        let bucket = vm.entry(key).or_insert_with(|| VmSemanticBucket {
            revision: super::super::revision::whole_work_runtime_binding_session(
                program, authority, config,
            ),
            generations: BTreeMap::new(),
        });
        let generation = bucket
            .generations
            .entry(region.installation())
            .or_insert_with(|| VmGeneration {
                liveness: liveness.clone(),
                permit: Arc::new(VmGenerationPermit::current()),
            });
        if !Weak::ptr_eq(&generation.liveness, &liveness) {
            return None;
        }
        if !generation.permit.is_current() {
            generation.permit = Arc::new(VmGenerationPermit::current());
        }
        Some(VmFusionSessionRegistration::new(
            bucket.revision.clone(),
            generation.permit.clone(),
        ))
    }

    pub(super) fn invalidate_vm_generations(&self, invalidation: &PlacementInvalidation) {
        for bucket in self.vm.borrow().values() {
            if invalidation_matches_revision(invalidation, &bucket.revision) {
                for generation in bucket.generations.values() {
                    generation.permit.revoke();
                }
            }
        }
    }

    fn replace(
        &self,
        key: ExecutionBindingKey,
        registration: Digest,
        authority: Digest,
    ) -> Option<Digest> {
        self.registrations
            .borrow_mut()
            .insert(
                key,
                BindingRegistration {
                    generation: registration,
                    authority,
                },
            )
            .and_then(|previous| {
                (previous.generation != registration).then_some(previous.authority)
            })
    }
}

fn sweep_vm(vm: &mut BTreeMap<VmSemanticKey, VmSemanticBucket>) {
    vm.retain(|_, bucket| {
        bucket
            .generations
            .retain(|_, generation| generation.liveness.upgrade().is_some());
        !bucket.generations.is_empty()
    });
}

fn invalidation_matches_revision(
    invalidation: &PlacementInvalidation,
    revision: &PlacementRevision,
) -> bool {
    match invalidation {
        PlacementInvalidation::All => true,
        PlacementInvalidation::Program { revision: program } => {
            revision.program.as_ref() == Some(program)
        }
        PlacementInvalidation::RuntimeBindingSession { digest } => {
            revision.authority
                == PlacementExecutionAuthorityRevision::RuntimeBindingSession(*digest)
        }
        PlacementInvalidation::Policy { digest } => revision.policy == *digest,
        PlacementInvalidation::Provider { .. }
        | PlacementInvalidation::HostService { .. }
        | PlacementInvalidation::Signature { .. } => false,
    }
}
