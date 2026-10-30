//! Keep an original startup refusal available for the bridge owner's lifetime.
use super::{mpsc, Command, ExecutionServiceError};

pub(super) fn reject_until_closed(commands: mpsc::Receiver<Command>, error: ExecutionServiceError) {
    // Dropping a caller's responder must not stop other queued/future callers.
    // Only closure of every command sender retires this failed bridge owner.
    for command in commands {
        reject(command, &error);
    }
}

fn reject(command: Command, error: &ExecutionServiceError) {
    match command {
        Command::Submit { response, .. } => {
            let _ = response.send(Err(error.clone()));
        }
        Command::SubmitGraph { response, .. } => {
            let _ = response.send(Err(error.clone()));
        }
        Command::Inspect { response, .. } => {
            let _ = response.send(Err(error.clone()));
        }
        Command::InspectKind { response, .. } => {
            let _ = response.send(Err(error.clone()));
        }
        Command::Cancel { response, .. } => {
            let _ = response.send(Err(error.clone()));
        }
        Command::ExpiredGraphReceipts { response, .. } => {
            let _ = response.send(Err(error.clone()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::durable::{service_error, DurableJobBridge};
    use runmat_execution::{CancellationReason, JobHandle, JobId, OutputContract, RunId};
    use runmat_value::SharedJobHandle;

    #[test]
    fn startup_refusal_reaches_queued_and_future_distinct_responders_until_owner_closes() {
        for original in [
            ExecutionServiceError::Infrastructure("original runtime initialization failure".into()),
            service_error(crate::NativeExecutionError::Configuration(
                "original supervisor profile executable mismatch".into(),
            )),
        ] {
            let (commands, receiver) = mpsc::channel();
            let (response, queued) = mpsc::channel();
            commands
                .send(Command::ExpiredGraphReceipts {
                    candidates: Vec::new(),
                    response,
                })
                .unwrap_or_else(|_| panic!("fixture command receiver must remain open"));
            let retained = original.clone();
            let worker = std::thread::spawn(move || reject_until_closed(receiver, retained));
            assert_eq!(queued.recv().unwrap(), Err(original.clone()));
            let bridge = DurableJobBridge { commands };
            let handle = SharedJobHandle::new(JobHandle {
                id: JobId::derive(&[b"startup-error-job"]),
                run_id: RunId::derive(&[b"startup-error-run"]),
                generation: 1,
                outputs: OutputContract {
                    requested_outputs: 1,
                    owned_work: None,
                },
            });
            assert_eq!(bridge.try_response(&handle), Err(original.clone()));
            assert_eq!(bridge.workload_kind(handle.clone()), Err(original.clone()));
            assert_eq!(
                bridge.cancel(handle, CancellationReason::User),
                Err(original.clone())
            );
            assert_eq!(bridge.expired_graph_receipts(Vec::new()), Err(original));
            drop(bridge);
            worker.join().unwrap();
        }
    }
}
