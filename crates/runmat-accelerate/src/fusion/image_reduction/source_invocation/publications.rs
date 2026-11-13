//! Rebind physical checkpoint yields to exact selected source stores. A final
//! `single` after POWER is executed before its store, not published early.

use serde::ser::SerializeSeq;
use serde::{Serialize, Serializer};
use std::ops::Range;

use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageSourcePublication {
    pub producer: ValueId,
    pub published: ValueId,
    pub statement: Range<usize>,
}

struct PublicationIdentity<'a>(&'a [ImageSourcePublication]);

impl Serialize for PublicationIdentity<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(Some(self.0.len()))?;
        for publication in self.0 {
            sequence.serialize_element(&(
                publication.producer,
                publication.published,
                publication.statement.start,
                publication.statement.end,
            ))?;
        }
        sequence.end()
    }
}

impl ImageSourceInvocation {
    pub fn bind_publications(
        mut self,
        graph: &AccelGraph,
        publications: &[ImageSourcePublication],
    ) -> Option<Self> {
        if self.publications_bound
            || publications.len() != self.source.checkpoints.len()
            || publications
                .iter()
                .zip(&self.source.checkpoints)
                .any(|(publication, producer)| {
                    publication.producer != *producer
                        || !super::super::publication_follows_stage(
                            graph,
                            publication.producer,
                            publication.published,
                            publication.statement.clone(),
                        )
                })
        {
            return None;
        }
        let moments = self.moment_program.as_ref()?;
        if publications[0].published != u32::try_from(moments.mu.last()?.public_output).ok()?
            || publications[1].published != u32::try_from(moments.sigma.public_output).ok()?
        {
            return None;
        }
        let program = self.output_program.as_mut()?;
        for call in &mut program.calls {
            call.checkpoint = false;
        }
        let mut previous = None;
        for publication in &publications[2..] {
            let position = program
                .calls
                .iter()
                .position(|call| call.source_output == u64::from(publication.published))?;
            if previous.is_some_and(|last| position <= last) {
                return None;
            }
            previous = Some(position);
            program.calls[position].checkpoint = true;
        }
        if previous != program.calls.len().checked_sub(1) || !program.validate() {
            return None;
        }
        self.workload.as_mut()?.output_program = Some(program.clone());
        self.semantic_identity = Digest::sha256_serialized(&(
            "selected-image-publications-v1",
            self.semantic_identity,
            PublicationIdentity(publications),
        ))
        .ok()?;
        self.publications_bound = true;
        Some(self)
    }
}
