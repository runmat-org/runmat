//! One native settings transaction owns its lock, private stage and receipt.

use super::super::document::{
    apply_mutation, DocumentMutation, DocumentMutationError, DocumentPrepared,
};
use super::store::{NativeStoreError, NativeWrite};
use std::sync::mpsc;
use tokio::sync::oneshot;

#[derive(Debug, thiserror::Error)]
pub(super) enum TransactionError {
    #[error(transparent)]
    Store(#[from] NativeStoreError),
    #[error(transparent)]
    Mutation(#[from] DocumentMutationError),
}

pub(super) fn worker(
    mut write: NativeWrite,
    mutation: DocumentMutation,
    prepared: oneshot::Sender<Result<DocumentPrepared, TransactionError>>,
    authorized: mpsc::Receiver<()>,
    finished: oneshot::Sender<Result<(), NativeStoreError>>,
) {
    let preparation = (|| {
        let document = std::mem::take(&mut write.document);
        let (updated, prepared) = apply_mutation(document, mutation)?;
        write.document = updated;
        let stage = if prepared.changed {
            Some(write.stage()?)
        } else {
            None
        };
        Ok((prepared, stage))
    })();
    match preparation {
        Err(error) => {
            let _ = prepared.send(Err(error));
        }
        Ok((receipt, stage)) => {
            let needs_commit = receipt.changed;
            if prepared.send(Ok(receipt)).is_err() {
                return;
            }
            if !needs_commit {
                return;
            }
            if authorized.recv().is_err() {
                return;
            }
            let result = stage.expect("changed document has a stage").commit();
            let _ = finished.send(result);
        }
    }
}
