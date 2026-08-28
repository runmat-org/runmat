use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};

use crate::cpython::Interpreter;
use crate::object::{PythonCallbackCommand, PythonLaneEvent};
use crate::{
    PythonCallbackInvocation, PythonError, PythonInstallation, PythonObjectMetadata, PythonValue,
};

static NEXT_SESSION_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone)]
struct ActiveReentry {
    session: u64,
    commands: mpsc::SyncSender<PythonCallbackCommand>,
}

thread_local! {
    static ACTIVE_REENTRY: RefCell<Option<ActiveReentry>> = const { RefCell::new(None) };
}

#[derive(Debug, Clone)]
pub struct PythonSessionConfig {
    pub installation: PythonInstallation,
    pub module_paths: Vec<std::path::PathBuf>,
}

#[derive(Debug, Clone)]
pub enum PythonCall {
    InvokeQualified {
        name: String,
        arguments: Vec<PythonValue>,
    },
    GetMember {
        receiver: crate::PythonObjectHandle,
        name: String,
    },
    SetMember {
        receiver: crate::PythonObjectHandle,
        name: String,
        value: PythonValue,
    },
    InvokeMember {
        receiver: crate::PythonObjectHandle,
        name: String,
        arguments: Vec<PythonValue>,
    },
    GetItem {
        receiver: crate::PythonObjectHandle,
        index: PythonValue,
    },
    SetItem {
        receiver: crate::PythonObjectHandle,
        index: PythonValue,
        value: PythonValue,
    },
    Iterate {
        receiver: crate::PythonObjectHandle,
    },
    ExecutePersistent {
        code: String,
        inputs: Vec<(String, PythonValue)>,
        outputs: Vec<String>,
    },
    ExecuteFile {
        path: std::path::PathBuf,
        arguments: Vec<String>,
        inputs: Vec<(String, PythonValue)>,
        outputs: Vec<String>,
    },
    Release {
        handle: crate::PythonObjectHandle,
    },
}

enum LaneRequest {
    Invoke {
        call: PythonCall,
        events: mpsc::SyncSender<PythonLaneEvent>,
    },
    Metadata {
        handle: crate::PythonObjectHandle,
        reply: mpsc::SyncSender<Result<PythonObjectMetadata, PythonError>>,
    },
    Shutdown,
}

struct SessionInner {
    id: u64,
    requests: mpsc::Sender<LaneRequest>,
    lane: Mutex<Option<JoinHandle<()>>>,
    interrupt: crate::cpython::PythonInterrupt,
}

impl Drop for SessionInner {
    fn drop(&mut self) {
        let _ = self.requests.send(LaneRequest::Shutdown);
        if let Some(lane) = self.lane.get_mut().ok().and_then(Option::take) {
            let _ = lane.join();
        }
    }
}

#[derive(Clone)]
pub struct PythonSession {
    inner: Arc<SessionInner>,
}

impl PythonSession {
    pub fn start(config: PythonSessionConfig) -> Result<Self, PythonError> {
        let (requests, receiver) = mpsc::channel();
        let (started, startup) = mpsc::sync_channel(1);
        let lane = thread::Builder::new()
            .name("runmat-python".to_owned())
            .spawn(move || {
                let interpreter = match Interpreter::start(&config.installation) {
                    Ok(interpreter) => {
                        if let Err(error) = interpreter.prepend_module_paths(&config.module_paths) {
                            let _ = started.send(Err(error));
                            interpreter.clear();
                            return;
                        }
                        let _ = started.send(Ok(interpreter.interrupt_handle()));
                        interpreter
                    }
                    Err(error) => {
                        let _ = started.send(Err(error));
                        return;
                    }
                };
                while let Ok(request) = receiver.recv() {
                    match request {
                        LaneRequest::Invoke { call, events } => {
                            let result = interpreter
                                .with_callback_events(events.clone(), || interpreter.invoke(call));
                            let _ = events.send(PythonLaneEvent::Complete(result));
                        }
                        LaneRequest::Metadata { handle, reply } => {
                            let _ = reply.send(interpreter.metadata(handle));
                        }
                        LaneRequest::Shutdown => break,
                    }
                }
                interpreter.clear();
            })
            .map_err(|error| {
                PythonError::host(
                    "PythonHostError",
                    format!("could not start the Python execution lane: {error}"),
                )
            })?;
        match startup.recv() {
            Ok(Ok(interrupt)) => Ok(Self {
                inner: Arc::new(SessionInner {
                    id: NEXT_SESSION_ID.fetch_add(1, Ordering::Relaxed),
                    requests,
                    lane: Mutex::new(Some(lane)),
                    interrupt,
                }),
            }),
            Ok(Err(error)) => {
                let _ = lane.join();
                Err(error)
            }
            Err(error) => {
                let _ = lane.join();
                Err(PythonError::host(
                    "PythonHostError",
                    format!("Python execution lane ended during startup: {error}"),
                ))
            }
        }
    }

    pub fn invoke(&self, call: PythonCall) -> Result<Vec<PythonValue>, PythonError> {
        self.invoke_with_callback(call, |_| {
            Err(PythonError::host(
                "PythonCallbackError",
                "a Python call attempted to invoke a RunMat callback where no callback handler was installed",
            ))
        })
    }

    pub fn invoke_with_callback(
        &self,
        call: PythonCall,
        callback: impl FnMut(PythonCallbackInvocation) -> Result<PythonValue, PythonError>,
    ) -> Result<Vec<PythonValue>, PythonError> {
        self.invoke_with_callback_and_cancellation(call, None, callback)
    }

    pub fn invoke_with_callback_and_cancellation(
        &self,
        call: PythonCall,
        cancellation: Option<&std::sync::atomic::AtomicBool>,
        mut callback: impl FnMut(PythonCallbackInvocation) -> Result<PythonValue, PythonError>,
    ) -> Result<Vec<PythonValue>, PythonError> {
        if let Some(active) = ACTIVE_REENTRY.with(|active| active.borrow().clone()) {
            if active.session == self.inner.id {
                let (reply, response) = mpsc::sync_channel(1);
                active
                    .commands
                    .send(PythonCallbackCommand::Reenter { call, reply })
                    .map_err(lane_ended)?;
                return response.recv().map_err(lane_ended)?;
            }
        }
        let (events, response) = mpsc::sync_channel(1);
        self.inner
            .requests
            .send(LaneRequest::Invoke { call, events })
            .map_err(lane_ended)?;
        let mut interrupted = false;
        loop {
            let event = match response.recv_timeout(std::time::Duration::from_millis(10)) {
                Ok(event) => event,
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    if !interrupted && cancellation.is_some_and(|flag| flag.load(Ordering::SeqCst))
                    {
                        if !self.inner.interrupt.request() {
                            return Err(PythonError::host(
                                "PythonCancellationError",
                                "the Python execution lane could not be interrupted",
                            ));
                        }
                        interrupted = true;
                    }
                    continue;
                }
                Err(error) => return Err(lane_ended(error)),
            };
            match event {
                PythonLaneEvent::Complete(result) => return result,
                PythonLaneEvent::Callback {
                    invocation,
                    commands,
                } => {
                    let previous = ACTIVE_REENTRY.with(|active| {
                        active.replace(Some(ActiveReentry {
                            session: self.inner.id,
                            commands: commands.clone(),
                        }))
                    });
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        callback(invocation)
                    }))
                    .unwrap_or_else(|_| {
                        Err(PythonError::host(
                            "PythonCallbackError",
                            "RunMat callback panicked",
                        ))
                    });
                    ACTIVE_REENTRY.with(|active| {
                        active.replace(previous);
                    });
                    commands
                        .send(PythonCallbackCommand::Return(result))
                        .map_err(lane_ended)?;
                }
            }
        }
    }

    pub fn metadata(
        &self,
        handle: crate::PythonObjectHandle,
    ) -> Result<PythonObjectMetadata, PythonError> {
        let (reply, response) = mpsc::sync_channel(1);
        self.inner
            .requests
            .send(LaneRequest::Metadata { handle, reply })
            .map_err(lane_ended)?;
        response.recv().map_err(lane_ended)?
    }
}

fn lane_ended(error: impl std::fmt::Display) -> PythonError {
    PythonError::host(
        "PythonHostError",
        format!("Python execution lane is unavailable: {error}"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        discover_python, PythonArray, PythonBufferOwner, PythonDType, PythonDateTime,
        PythonDiscoveryRequest, PythonObjectHandle, PythonTimeDelta,
    };

    #[derive(Debug)]
    struct TestBuffer(Vec<f64>);

    impl PythonBufferOwner for TestBuffer {
        fn address(&self) -> usize {
            self.0.as_ptr() as usize
        }

        fn byte_length(&self) -> usize {
            self.0.len() * std::mem::size_of::<f64>()
        }

        fn copy_bytes(&self) -> Vec<u8> {
            let length = self.byte_length();
            // SAFETY: the f64 allocation contains exactly length initialized
            // bytes and remains live for this immutable copy.
            unsafe { std::slice::from_raw_parts(self.0.as_ptr().cast::<u8>(), length) }.to_vec()
        }
    }

    fn session() -> Option<PythonSession> {
        let installation = discover_python(&PythonDiscoveryRequest::default()).ok()?;
        Some(
            PythonSession::start(PythonSessionConfig {
                installation,
                module_paths: Vec::new(),
            })
            .expect("start selected CPython"),
        )
    }

    #[test]
    fn invokes_qualified_functions_with_positional_and_keyword_arguments() {
        let Some(session) = session() else {
            return;
        };
        let result = session
            .invoke(PythonCall::InvokeQualified {
                name: "py.math.isclose".to_owned(),
                arguments: vec![
                    PythonValue::Float(1.0),
                    PythonValue::Float(1.000_001),
                    PythonValue::Keywords(vec![(
                        "rel_tol".to_owned(),
                        PythonValue::Float(0.000_01),
                    )]),
                ],
            })
            .expect("invoke math.isclose");
        assert!(matches!(result.as_slice(), [PythonValue::Bool(true)]));
    }

    #[test]
    fn persistent_code_keeps_workspace_and_file_code_does_not_share_it() {
        let Some(session) = session() else {
            return;
        };
        let first = session
            .invoke(PythonCall::ExecutePersistent {
                code: "counter = 40".to_owned(),
                inputs: Vec::new(),
                outputs: vec!["counter".to_owned()],
            })
            .expect("initialize persistent workspace");
        assert!(matches!(first.as_slice(), [PythonValue::Signed(40)]));
        let second = session
            .invoke(PythonCall::ExecutePersistent {
                code: "counter += 2".to_owned(),
                inputs: Vec::new(),
                outputs: vec!["counter".to_owned()],
            })
            .expect("reuse persistent workspace");
        assert!(matches!(second.as_slice(), [PythonValue::Signed(42)]));
    }

    #[test]
    fn opaque_objects_keep_identity_until_explicit_release() {
        let Some(session) = session() else {
            return;
        };
        let object = session
            .invoke(PythonCall::InvokeQualified {
                name: "py.object".to_owned(),
                arguments: Vec::new(),
            })
            .expect("construct Python object");
        let [PythonValue::Object(handle)] = object.as_slice() else {
            panic!("expected opaque object handle");
        };
        let metadata = session.metadata(*handle).expect("inspect live object");
        assert_eq!(metadata.type_name, "object");
        session
            .invoke(PythonCall::Release { handle: *handle })
            .expect("release object");
        let error = session
            .metadata(PythonObjectHandle(handle.0))
            .expect_err("released object must be stale");
        assert_eq!(error.type_name, "PythonStaleObject");
    }

    #[test]
    fn numeric_arrays_alias_runmat_storage_and_return_with_dtype_and_shape() {
        let Some(session) = session() else {
            return;
        };
        let owner = Arc::new(TestBuffer(vec![1.0, 2.0, 3.0, 4.0]));
        let address = owner.address();
        let result = session
            .invoke(PythonCall::ExecutePersistent {
                code: "address = values.__array_interface__['data'][0]\nreturned = values".into(),
                inputs: vec![(
                    "values".into(),
                    PythonValue::Array(PythonArray {
                        dtype: PythonDType::Float64,
                        shape: vec![2, 2],
                        column_major: true,
                        read_only: true,
                        owner,
                    }),
                )],
                outputs: vec!["address".into(), "returned".into()],
            })
            .expect("round-trip shared array");
        assert!(matches!(
            result.first(),
            Some(PythonValue::Signed(value)) if *value == address as i64
        ));
        let Some(PythonValue::Array(array)) = result.get(1) else {
            panic!("expected typed Python array");
        };
        assert_eq!(array.dtype, PythonDType::Float64);
        assert_eq!(array.shape, vec![2, 2]);
        assert!(array.column_major);
        assert_eq!(
            array.owner.copy_bytes(),
            TestBuffer(vec![1.0, 2.0, 3.0, 4.0]).copy_bytes()
        );
    }

    #[test]
    fn temporal_scalars_and_arrays_use_explicit_python_types() {
        let Some(session) = session() else {
            return;
        };
        let result = session
            .invoke(PythonCall::ExecutePersistent {
                code: "import datetime\nimport numpy as np\nscalar_date_type = type(date).__name__\nscalar_delta_type = type(delta).__name__\ndates = np.array(['1969-12-31T23:59:59.123456', '2026-08-27T12:34:56.654321'], dtype='datetime64[us]')\ndeltas = np.array([-1001, 2002], dtype='timedelta64[us]')".into(),
                inputs: vec![
                    (
                        "date".into(),
                        PythonValue::DateTime(PythonDateTime {
                            year: 2026,
                            month: 8,
                            day: 27,
                            hour: 12,
                            minute: 34,
                            second: 56,
                            microsecond: 654_321,
                        }),
                    ),
                    (
                        "delta".into(),
                        PythonValue::TimeDelta(PythonTimeDelta {
                            days: -1,
                            seconds: 86_399,
                            microseconds: 998_999,
                        }),
                    ),
                ],
                outputs: vec![
                    "scalar_date_type".into(),
                    "scalar_delta_type".into(),
                    "dates".into(),
                    "deltas".into(),
                ],
            })
            .expect("round-trip Python temporal values");
        assert!(matches!(result.first(), Some(PythonValue::String(value)) if value == "datetime"));
        assert!(matches!(result.get(1), Some(PythonValue::String(value)) if value == "timedelta"));
        let Some(PythonValue::Array(dates)) = result.get(2) else {
            panic!("expected datetime64 array");
        };
        assert_eq!(dates.dtype, PythonDType::DateTime64Micros);
        assert_eq!(dates.shape, vec![2]);
        assert_eq!(
            dates.owner.copy_bytes(),
            [-876_544_i64, 1_787_834_096_654_321_i64]
                .into_iter()
                .flat_map(i64::to_ne_bytes)
                .collect::<Vec<_>>()
        );
        let Some(PythonValue::Array(deltas)) = result.get(3) else {
            panic!("expected timedelta64 array");
        };
        assert_eq!(deltas.dtype, PythonDType::TimeDelta64Micros);
        assert_eq!(
            deltas.owner.copy_bytes(),
            [-1_001_i64, 2_002_i64]
                .into_iter()
                .flat_map(i64::to_ne_bytes)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn naive_datetime_converts_and_aware_datetime_keeps_python_identity() {
        let Some(session) = session() else {
            return;
        };
        let result = session
            .invoke(PythonCall::ExecutePersistent {
                code: "import datetime\nnaive = datetime.datetime(2026, 8, 27, 1, 2, 3, 456789)\naware = datetime.datetime(2026, 8, 27, tzinfo=datetime.timezone.utc)".into(),
                inputs: Vec::new(),
                outputs: vec!["naive".into(), "aware".into()],
            })
            .expect("read Python datetime values");
        assert!(matches!(
            result.first(),
            Some(PythonValue::DateTime(PythonDateTime {
                year: 2026,
                month: 8,
                day: 27,
                hour: 1,
                minute: 2,
                second: 3,
                microsecond: 456_789,
            }))
        ));
        assert!(matches!(result.get(1), Some(PythonValue::Object(_))));
    }

    #[test]
    fn exceptions_retain_python_type_message_and_traceback_frames() {
        let Some(session) = session() else {
            return;
        };
        let error = session
            .invoke(PythonCall::ExecutePersistent {
                code: "def fail_here():\n    return 1 / 0\nfail_here()".into(),
                inputs: Vec::new(),
                outputs: Vec::new(),
            })
            .expect_err("Python failure must cross the host boundary");
        assert_eq!(error.type_name, "ZeroDivisionError");
        assert!(error.message.contains("division by zero"));
        assert!(error
            .traceback
            .iter()
            .any(|frame| frame.function.as_deref() == Some("fail_here") && frame.line == Some(2)));
        assert!(error.formatted_traceback.contains("ZeroDivisionError"));
    }

    #[test]
    fn exceptions_retain_explicit_python_causes() {
        let Some(session) = session() else {
            return;
        };
        let error = session
            .invoke(PythonCall::ExecutePersistent {
                code: "try:\n    int('not-an-integer')\nexcept ValueError as cause:\n    raise RuntimeError('outer failure') from cause".into(),
                inputs: Vec::new(),
                outputs: Vec::new(),
            })
            .expect_err("chained Python failure must cross the host boundary");
        assert_eq!(error.type_name, "RuntimeError");
        let cause = error.cause.expect("explicit Python cause");
        assert_eq!(cause.type_name, "ValueError");
        assert!(cause.message.contains("not-an-integer"));
    }

    #[test]
    fn callbacks_return_on_the_origin_thread_and_can_reenter_python() {
        let Some(session) = session() else {
            return;
        };
        let nested = session.clone();
        let result = session
            .invoke_with_callback(
                PythonCall::ExecutePersistent {
                    code: "result = callback(40)".into(),
                    inputs: vec![("callback".into(), PythonValue::Callback(7))],
                    outputs: vec!["result".into()],
                },
                move |invocation| {
                    assert_eq!(invocation.callback, 7);
                    assert!(matches!(
                        invocation.arguments.as_slice(),
                        [PythonValue::Signed(40)]
                    ));
                    let nested_result = nested.invoke(PythonCall::InvokeQualified {
                        name: "py.operator.add".into(),
                        arguments: vec![PythonValue::Signed(40), PythonValue::Signed(2)],
                    })?;
                    nested_result.into_iter().next().ok_or_else(|| {
                        PythonError::host("PythonCallbackError", "nested call returned no value")
                    })
                },
            )
            .expect("callback and nested Python call");
        assert!(matches!(result.as_slice(), [PythonValue::Signed(42)]));
    }

    #[test]
    fn cancellation_interrupts_python_at_a_bytecode_boundary() {
        let Some(session) = session() else {
            return;
        };
        let cancellation = std::sync::atomic::AtomicBool::new(true);
        let error = session
            .invoke_with_callback_and_cancellation(
                PythonCall::ExecutePersistent {
                    code: "while True:\n    pass".into(),
                    inputs: Vec::new(),
                    outputs: Vec::new(),
                },
                Some(&cancellation),
                |_| {
                    Err(PythonError::host(
                        "PythonCallbackError",
                        "unexpected callback",
                    ))
                },
            )
            .expect_err("cancelled Python execution must stop");
        assert_eq!(error.type_name, "KeyboardInterrupt");
    }
}
