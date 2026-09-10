use runmat_runtime::object::protocol::PreparedSubscriptReceiver;

use super::{super::HostState, EmbeddedOperationIdentity};

impl HostState {
    pub fn cached_prepared_subscript_receiver(
        &self,
        owner: Option<&EmbeddedOperationIdentity>,
        step: usize,
    ) -> Option<PreparedSubscriptReceiver> {
        self.contextual_regions
            .prepared_subscript_receivers
            .get(&(owner.cloned(), step))
            .map(|(prepared, _)| prepared.clone())
    }

    pub fn cache_prepared_subscript_receiver(
        &mut self,
        owner: Option<EmbeddedOperationIdentity>,
        step: usize,
        prepared: PreparedSubscriptReceiver,
    ) {
        let receiver = self.arena.insert(prepared.receiver().clone());
        self.contextual_regions
            .prepared_subscript_receivers
            .insert((owner, step), (prepared, receiver));
    }
}
