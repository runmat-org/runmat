use super::{
    references::CachedRefsBuilder, EmbeddedOperationIdentity, EmbeddedOperationSegment,
    OperationScope,
};
use crate::{NativeExecutorError, NativeExecutorResult};

use super::super::HostState;

impl HostState {
    /// A completed operation is cached before its scope is popped. A cache
    /// admission error is terminal for this completed prefix; the callable is
    /// never re-entered to make the metadata allocation succeed later.
    pub fn settle_embedded_result<T>(
        &mut self,
        identity: Option<&EmbeddedOperationIdentity>,
        result: NativeExecutorResult<T>,
        values: impl FnOnce(&T) -> &[runmat_value::Value],
    ) -> NativeExecutorResult<T> {
        let cached = match &result {
            Ok(output) => self.cache_embedded_call(identity, values(output)),
            Err(_) => Ok(()),
        };
        let finished = self.finish_embedded_call(identity);
        match (result, cached, finished) {
            (Err(error), _, _) => Err(error),
            (Ok(_), Err(error), _) => Err(error),
            (Ok(_), Ok(()), Err(error)) => Err(error),
            (Ok(output), Ok(()), Ok(())) => Ok(output),
        }
    }

    pub fn enter_embedded_call(
        &mut self,
    ) -> NativeExecutorResult<Option<EmbeddedOperationIdentity>> {
        let Some(metadata) = self.contextual_regions.metadata.as_mut() else {
            return Ok(None);
        };
        let Some(scope) = metadata.operation_path.last() else {
            return Ok(None);
        };
        let ordinal = scope.next_call;
        let next_call = ordinal.checked_add(1).ok_or_else(|| {
            NativeExecutorError::Host("native embedded-call ordinal overflow".into())
        })?;
        super::prepare_vector_push(&mut metadata.operation_path, &self.runtime, &metadata.gate)?;
        metadata
            .operation_path
            .last_mut()
            .expect("nonempty path")
            .next_call = next_call;
        metadata.operation_path.push(OperationScope {
            segment: EmbeddedOperationSegment::Call { ordinal },
            next_call: 0,
        });
        match self.current_embedded_operation() {
            Ok(identity) => Ok(identity),
            Err(error) => {
                let metadata = self.contextual_regions.metadata_mut()?;
                metadata.operation_path.pop();
                metadata
                    .operation_path
                    .last_mut()
                    .expect("parent path")
                    .next_call = ordinal;
                Err(error)
            }
        }
    }

    pub fn finish_embedded_call(
        &mut self,
        identity: Option<&EmbeddedOperationIdentity>,
    ) -> NativeExecutorResult<()> {
        let Some(identity) = identity else {
            return Ok(());
        };
        let Some(EmbeddedOperationSegment::Call { ordinal }) = identity.path().last() else {
            return Err(NativeExecutorError::Host(
                "embedded call identity has no call segment".into(),
            ));
        };
        let next_region = self.contextual_regions.next_region;
        let metadata = self.contextual_regions.metadata_mut()?;
        let recorded = metadata.operation_region_ends.insert(
            &self.runtime,
            &metadata.gate,
            identity.clone(),
            next_region,
        );
        let popped = self.pop_operation(EmbeddedOperationSegment::Call { ordinal: *ordinal });
        recorded.map(|_| ()).and(popped)
    }

    pub fn restore_embedded_call_regions(&mut self, identity: &EmbeddedOperationIdentity) {
        if let Some(metadata) = self.contextual_regions.metadata.as_ref() {
            if let Some(next_region) = metadata.operation_region_ends.get(identity) {
                self.contextual_regions.next_region = *next_region;
            }
        }
    }

    pub fn current_embedded_operation(
        &self,
    ) -> NativeExecutorResult<Option<EmbeddedOperationIdentity>> {
        let Some(metadata) = self.contextual_regions.metadata.as_ref() else {
            return Ok(None);
        };
        if metadata.operation_path.is_empty() {
            return Ok(None);
        }
        EmbeddedOperationIdentity::prepare(&self.runtime, &metadata.operation_path).map(Some)
    }

    pub fn cached_embedded_call(
        &mut self,
        identity: &EmbeddedOperationIdentity,
    ) -> NativeExecutorResult<Option<Vec<runmat_value::Value>>> {
        let cached = self
            .contextual_regions
            .metadata()?
            .completed_operations
            .get(identity)
            .map(|references| {
                references
                    .as_slice()
                    .iter()
                    .map(|reference| self.arena.get(*reference).cloned())
                    .collect::<NativeExecutorResult<Vec<_>>>()
            })
            .transpose()?;
        if cached.is_some() {
            self.restore_embedded_call_regions(identity);
        }
        Ok(cached)
    }

    pub fn cache_embedded_call(
        &mut self,
        identity: Option<&EmbeddedOperationIdentity>,
        values: &[runmat_value::Value],
    ) -> NativeExecutorResult<()> {
        let Some(identity) = identity else {
            return Ok(());
        };
        let metadata = self.contextual_regions.metadata_mut()?;
        if metadata.completed_operations.get(identity).is_none() {
            metadata
                .completed_operations
                .prepare_one_more(&self.runtime, &metadata.gate)?;
        }
        let mut references = CachedRefsBuilder::prepare(&self.runtime, values.len())?;
        for value in values {
            let reference = self.arena.insert(value.clone(), Some(&self.runtime))?;
            if let Err(error) = references.push_initial(reference) {
                let _ = self.arena.release(reference);
                for prior in references.as_slice() {
                    let _ = self.arena.release(*prior);
                }
                return Err(error);
            }
        }
        let references = match references.finish() {
            Ok(references) => references,
            Err(error) => {
                for reference in references.as_slice() {
                    let _ = self.arena.release(*reference);
                }
                return Err(error);
            }
        };
        let previous = self
            .contextual_regions
            .metadata_mut()?
            .completed_operations
            .insert_ready(identity.clone(), references);
        if let Some(previous) = previous {
            for reference in previous.as_slice() {
                self.arena.release(*reference)?;
            }
        }
        Ok(())
    }
}
