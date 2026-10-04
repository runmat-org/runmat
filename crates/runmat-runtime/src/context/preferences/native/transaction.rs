//! One native settings transaction owns its lock, private stage and receipt.

use super::store::{NativeStoreError, NativeWrite};
use crate::context::{UserPathSetting, UserPathSnapshot};
use runmat_execution::value::ValuePayload;
use std::sync::mpsc;
use tokio::sync::oneshot;

pub(super) enum Mutation {
    Set {
        group: String,
        entries: Vec<(String, ValuePayload)>,
    },
    ReadOrCreate {
        group: String,
        names: Vec<String>,
        defaults: Option<Vec<ValuePayload>>,
    },
    UserPath(UserPathSetting),
}

pub(super) struct Prepared {
    pub values: Vec<ValuePayload>,
    pub user_path: Option<UserPathSnapshot>,
    pub commit_required: bool,
}

#[derive(Debug, thiserror::Error)]
pub(super) enum TransactionError {
    #[error(transparent)]
    Store(#[from] NativeStoreError),
    #[error("preference '{name}' does not exist in group '{group}'")]
    Missing { group: String, name: String },
}

pub(super) fn worker(
    mut write: NativeWrite,
    mutation: Mutation,
    prepared: oneshot::Sender<Result<Prepared, TransactionError>>,
    authorized: mpsc::Receiver<()>,
    finished: oneshot::Sender<Result<(), NativeStoreError>>,
) {
    let preparation = (|| {
        let mut values = Vec::new();
        let mut changed = false;
        let mut return_user_path = false;
        match mutation {
            Mutation::Set { group, entries } => {
                let target = write.document.groups.entry(group).or_default();
                for (name, payload) in entries {
                    target.insert(name, payload);
                }
                changed = true;
            }
            Mutation::ReadOrCreate {
                group,
                names,
                defaults,
            } => {
                for (index, name) in names.iter().enumerate() {
                    let target = write.document.groups.entry(group.clone()).or_default();
                    if !target.contains_key(name) {
                        let Some(default) = defaults.as_ref().and_then(|values| values.get(index))
                        else {
                            return Err(TransactionError::Missing {
                                group,
                                name: name.clone(),
                            });
                        };
                        target.insert(name.clone(), default.clone());
                        changed = true;
                    }
                    values.push(
                        target
                            .get(name)
                            .expect("existing or inserted preference")
                            .clone(),
                    );
                }
            }
            Mutation::UserPath(setting) => {
                write.document.user_path = setting;
                write.document.user_path_revision = write
                    .document
                    .user_path_revision
                    .checked_add(1)
                    .ok_or_else(|| TransactionError::Store(NativeStoreError::RevisionExhausted))?;
                changed = true;
                return_user_path = true;
            }
        }
        let user_path = return_user_path.then(|| UserPathSnapshot {
            setting: write.document.user_path.clone(),
            revision: write.document.user_path_revision,
        });
        let stage = if changed { Some(write.stage()?) } else { None };
        Ok((
            Prepared {
                values,
                user_path,
                commit_required: changed,
            },
            stage,
        ))
    })();
    match preparation {
        Err(error) => {
            let _ = prepared.send(Err(error));
        }
        Ok((receipt, stage)) => {
            let needs_commit = receipt.commit_required;
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
