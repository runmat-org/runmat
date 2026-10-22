//! Resolve defaults under the same locked document that owns publication.
//! Runtime Values stay on the caller thread; only encoded payloads cross to
//! the blocking file worker.

use super::*;

pub(super) async fn read_or_create(
    service: NativePreferenceService,
    runtime: &RuntimeContext,
    mut budget: CpuCooperationBudget,
    group: String,
    names: Vec<String>,
    defaults: Vec<Value>,
    profiling: Option<&CommandProfilingReceipt>,
) -> Result<Vec<Value>, PreferenceServiceError> {
    checkpoint(&mut budget, 0).await?;
    let permit = budget
        .await_cancellable(service.gate.clone().acquire_owned())
        .await
        .map_err(cancelled)?
        .map_err(|source| operational("preference write gate closed", source))?;
    let write = service.acquire_write(&mut budget, profiling).await?;
    let (needed_tx, needed_rx) = oneshot::channel();
    let (mutation_tx, mutation_rx) = mpsc::channel();
    let (prepared_tx, prepared_rx) = oneshot::channel();
    let (authorized_tx, authorized_rx) = mpsc::channel();
    let (finished_tx, finished_rx) = oneshot::channel();
    let work = SelectionWork {
        write,
        group: group.clone(),
        names: names.clone(),
        needed_tx,
        mutation_rx,
        prepared_tx,
        authorized_rx,
        finished_tx,
    };
    let work = BlockingWork::new(
        move || {
            let _permit = permit;
            work.run();
        },
        profiling.and_then(CommandProfilingReceipt::begin_deferred_work),
    );
    service.driver.spawn_blocking(move || work.run());
    let needed = budget
        .await_cancellable(needed_rx)
        .await
        .map_err(cancelled)?
        .map_err(|source| operational("preference default selection ended", source))?;
    let mut encoded = vec![None; names.len()];
    for index in needed {
        encoded[index] = Some(
            super::super::encode_selected_preference(&defaults[index], &mut budget, runtime)
                .await?,
        );
        checkpoint(&mut budget, 1).await?;
    }
    mutation_tx
        .send(Mutation::ReadOrCreate {
            group,
            names,
            defaults: Some(encoded),
        })
        .map_err(|_| {
            operational(
                "preference default preparation owner ended",
                std::io::Error::new(std::io::ErrorKind::BrokenPipe, "preference worker ended"),
            )
        })?;
    observe_mutation(
        runtime,
        &mut budget,
        prepared_rx,
        authorized_tx,
        finished_rx,
    )
    .await
    .map(|(values, _)| values)
}

struct SelectionWork {
    write: NativeWrite,
    group: String,
    names: Vec<String>,
    needed_tx: oneshot::Sender<Vec<usize>>,
    mutation_rx: mpsc::Receiver<Mutation>,
    prepared_tx:
        oneshot::Sender<Result<super::super::document::DocumentPrepared, TransactionError>>,
    authorized_rx: mpsc::Receiver<()>,
    finished_tx: oneshot::Sender<Result<(), NativeStoreError>>,
}

impl SelectionWork {
    fn run(self) {
        let needed = missing_default_indices(&self.write.document, &self.group, &self.names);
        if self.needed_tx.send(needed).is_err() {
            return;
        }
        let Ok(mutation) = self.mutation_rx.recv() else {
            return;
        };
        worker(
            self.write,
            mutation,
            self.prepared_tx,
            self.authorized_rx,
            self.finished_tx,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abandoned_default_handshake_releases_locked_document_without_stage() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("settings.json");
        let store = NativeStore::new(path.clone()).unwrap();
        let write = store.try_begin_write().unwrap().unwrap();
        let (needed_tx, needed_rx) = oneshot::channel();
        let (mutation_tx, mutation_rx) = mpsc::channel();
        let (prepared_tx, _prepared_rx) = oneshot::channel();
        let (_authorized_tx, authorized_rx) = mpsc::channel();
        let (finished_tx, _finished_rx) = oneshot::channel();
        let worker = std::thread::spawn(move || {
            SelectionWork {
                write,
                group: "app".into(),
                names: vec!["first".into()],
                needed_tx,
                mutation_rx,
                prepared_tx,
                authorized_rx,
                finished_tx,
            }
            .run();
        });
        assert_eq!(futures::executor::block_on(needed_rx).unwrap(), vec![0]);
        drop(mutation_tx); // Abandoned caller never sends a prepared mutation.
        worker.join().unwrap();
        assert!(!path.exists(), "no application document was published");
        assert!(store.try_begin_write().unwrap().is_some(), "lock released");
    }
}
