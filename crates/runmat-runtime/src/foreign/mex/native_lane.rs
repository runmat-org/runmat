use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::mpsc;

use futures::channel::{mpsc as async_mpsc, oneshot};
use futures::{FutureExt, StreamExt};
use runmat_mex::{
    MexBoundaryHostServices, MexDiagnostic, MexLoadError, MexModule, MexNativeInvocation,
    MxApiMode, MxArray, MxBoundaryInterface,
};

#[derive(Debug, Clone, Copy)]
pub(super) struct NativeModuleMetadata {
    pub mode: MxApiMode,
    pub interface: MxBoundaryInterface,
}

pub(super) struct NativeMexLane {
    commands: mpsc::Sender<NativeCommand>,
}

enum NativeCommand {
    Load {
        path: PathBuf,
        response: oneshot::Sender<Result<NativeModuleMetadata, String>>,
    },
    Invoke {
        path: PathBuf,
        inputs: Vec<MxArray>,
        requested_outputs: usize,
        callbacks: async_mpsc::UnboundedSender<BoundaryRequest>,
        response: oneshot::Sender<Result<MexNativeInvocation, String>>,
    },
    Clear {
        path: PathBuf,
        callbacks: async_mpsc::UnboundedSender<BoundaryRequest>,
        response: oneshot::Sender<Result<bool, String>>,
    },
    ShutdownModule {
        path: PathBuf,
        callbacks: async_mpsc::UnboundedSender<BoundaryRequest>,
        response: oneshot::Sender<Result<(), String>>,
    },
    Shutdown,
}

pub(super) enum BoundaryRequest {
    Eval {
        command: String,
        response: mpsc::SyncSender<Result<(), MexDiagnostic>>,
    },
    Call {
        function: String,
        arguments: Vec<MxArray>,
        requested_outputs: usize,
        response: mpsc::SyncSender<Result<Vec<MxArray>, MexDiagnostic>>,
    },
    GetVariable {
        workspace: String,
        name: String,
        response: mpsc::SyncSender<Result<Option<MxArray>, MexDiagnostic>>,
    },
    PutVariable {
        workspace: String,
        name: String,
        value: MxArray,
        response: mpsc::SyncSender<Result<(), MexDiagnostic>>,
    },
    GetObjectProperty {
        object: MxArray,
        name: String,
        response: mpsc::SyncSender<Result<MxArray, MexDiagnostic>>,
    },
    SetObjectProperty {
        object: MxArray,
        name: String,
        value: MxArray,
        response: mpsc::SyncSender<Result<MxArray, MexDiagnostic>>,
    },
}

impl NativeMexLane {
    pub(super) fn spawn() -> Result<Self, String> {
        let (commands, receiver) = mpsc::channel();
        std::thread::Builder::new()
            .name("runmat-mex-native".into())
            .spawn(move || lane_main(receiver))
            .map_err(|error| format!("could not start the MEX native lane: {error}"))?;
        Ok(Self { commands })
    }

    pub(super) async fn load(&self, path: &Path) -> Result<NativeModuleMetadata, String> {
        let (response, receiver) = oneshot::channel();
        self.commands
            .send(NativeCommand::Load {
                path: path.to_path_buf(),
                response,
            })
            .map_err(|_| "the MEX native lane stopped".to_string())?;
        receiver
            .await
            .map_err(|_| "the MEX native lane stopped while loading a module".to_string())?
    }

    pub(super) async fn invoke(
        &self,
        path: &Path,
        inputs: Vec<MxArray>,
        requested_outputs: usize,
        services: &dyn MexBoundaryHostServices,
    ) -> Result<MexNativeInvocation, String> {
        let (callback_sender, mut callbacks) = async_mpsc::unbounded();
        let (response, receiver) = oneshot::channel();
        self.commands
            .send(NativeCommand::Invoke {
                path: path.to_path_buf(),
                inputs,
                requested_outputs,
                callbacks: callback_sender,
                response,
            })
            .map_err(|_| "the MEX native lane stopped".to_string())?;

        wait_with_callbacks(receiver, &mut callbacks, services, "invocation").await
    }

    pub(super) async fn clear(
        &self,
        path: &Path,
        services: &dyn MexBoundaryHostServices,
    ) -> Result<bool, String> {
        let (callback_sender, mut callbacks) = async_mpsc::unbounded();
        let (response, receiver) = oneshot::channel();
        self.commands
            .send(NativeCommand::Clear {
                path: path.to_path_buf(),
                callbacks: callback_sender,
                response,
            })
            .map_err(|_| "the MEX native lane stopped".to_string())?;
        wait_with_callbacks(receiver, &mut callbacks, services, "clear").await
    }

    pub(super) async fn shutdown_module(
        &self,
        path: &Path,
        services: &dyn MexBoundaryHostServices,
    ) -> Result<(), String> {
        let (callback_sender, mut callbacks) = async_mpsc::unbounded();
        let (response, receiver) = oneshot::channel();
        self.commands
            .send(NativeCommand::ShutdownModule {
                path: path.to_path_buf(),
                callbacks: callback_sender,
                response,
            })
            .map_err(|_| "the MEX native lane stopped".to_string())?;
        wait_with_callbacks(receiver, &mut callbacks, services, "shutdown").await
    }
}

impl Drop for NativeMexLane {
    fn drop(&mut self) {
        let _ = self.commands.send(NativeCommand::Shutdown);
    }
}

fn lane_main(receiver: mpsc::Receiver<NativeCommand>) {
    let mut modules = HashMap::<PathBuf, MexModule>::new();
    while let Ok(command) = receiver.recv() {
        match command {
            NativeCommand::Load { path, response } => {
                let result = load_module(&mut modules, &path).map(|module| NativeModuleMetadata {
                    mode: module.api_mode(),
                    interface: module.boundary_interface(),
                });
                let _ = response.send(result.map_err(|error| error.to_string()));
            }
            NativeCommand::Invoke {
                path,
                inputs,
                requested_outputs,
                callbacks,
                response,
            } => {
                let result = load_module(&mut modules, &path).and_then(|module| {
                    module.invoke_native(
                        inputs,
                        requested_outputs,
                        module.api_mode(),
                        Rc::new(NativeBoundaryProxy { callbacks }),
                    )
                });
                let _ = response.send(result.map_err(|error| error.to_string()));
            }
            NativeCommand::Clear {
                path,
                callbacks,
                response,
            } => {
                let result = if let Some(module) = modules.get(&path) {
                    let cleared = module
                        .clear_with_boundary_services(Rc::new(NativeBoundaryProxy { callbacks }));
                    if cleared.as_ref().is_ok_and(|cleared| *cleared) {
                        modules.remove(&path);
                    }
                    cleared
                } else {
                    Ok(true)
                };
                let _ = response.send(result.map_err(|error| error.to_string()));
            }
            NativeCommand::ShutdownModule {
                path,
                callbacks,
                response,
            } => {
                let services = Rc::new(NativeBoundaryProxy { callbacks });
                let result = modules.get(&path).map_or(Ok(()), |module| {
                    module.shutdown_with_boundary_services(services)
                });
                modules.remove(&path);
                let _ = response.send(result.map_err(|error| error.to_string()));
            }
            NativeCommand::Shutdown => break,
        }
    }
}

async fn wait_with_callbacks<T>(
    receiver: oneshot::Receiver<Result<T, String>>,
    callbacks: &mut async_mpsc::UnboundedReceiver<BoundaryRequest>,
    services: &dyn MexBoundaryHostServices,
    operation: &str,
) -> Result<T, String> {
    let mut result = receiver.fuse();
    loop {
        futures::select! {
            response = result => {
                return response.map_err(|_| {
                    format!("the MEX native lane stopped during {operation}")
                })?;
            }
            request = callbacks.next().fuse() => {
                let Some(request) = request else {
                    return result.await.map_err(|_| {
                        format!("the MEX native lane stopped during {operation}")
                    })?;
                };
                service_request(request, services);
            }
        }
    }
}

fn load_module<'a>(
    modules: &'a mut HashMap<PathBuf, MexModule>,
    path: &Path,
) -> Result<&'a MexModule, MexLoadError> {
    if !modules.contains_key(path) {
        modules.insert(path.to_path_buf(), MexModule::load(path)?);
    }
    Ok(modules
        .get(path)
        .expect("a module inserted into the native lane must remain present"))
}

struct NativeBoundaryProxy {
    callbacks: async_mpsc::UnboundedSender<BoundaryRequest>,
}

impl NativeBoundaryProxy {
    fn request<T>(
        &self,
        build: impl FnOnce(mpsc::SyncSender<Result<T, MexDiagnostic>>) -> BoundaryRequest,
    ) -> Result<T, MexDiagnostic> {
        let (response, receiver) = mpsc::sync_channel(1);
        self.callbacks
            .unbounded_send(build(response))
            .map_err(|_| lane_diagnostic("the originating runtime task stopped"))?;
        receiver
            .recv()
            .map_err(|_| lane_diagnostic("the originating runtime task stopped"))?
    }
}

impl MexBoundaryHostServices for NativeBoundaryProxy {
    fn eval(&self, command: &str) -> Result<(), MexDiagnostic> {
        self.request(|response| BoundaryRequest::Eval {
            command: command.into(),
            response,
        })
    }

    fn call(
        &self,
        function: &str,
        arguments: Vec<MxArray>,
        requested_outputs: usize,
    ) -> Result<Vec<MxArray>, MexDiagnostic> {
        self.request(|response| BoundaryRequest::Call {
            function: function.into(),
            arguments,
            requested_outputs,
            response,
        })
    }

    fn get_variable(&self, workspace: &str, name: &str) -> Result<Option<MxArray>, MexDiagnostic> {
        self.request(|response| BoundaryRequest::GetVariable {
            workspace: workspace.into(),
            name: name.into(),
            response,
        })
    }

    fn put_variable(
        &self,
        workspace: &str,
        name: &str,
        value: MxArray,
    ) -> Result<(), MexDiagnostic> {
        self.request(|response| BoundaryRequest::PutVariable {
            workspace: workspace.into(),
            name: name.into(),
            value,
            response,
        })
    }

    fn get_object_property(&self, object: MxArray, name: &str) -> Result<MxArray, MexDiagnostic> {
        self.request(|response| BoundaryRequest::GetObjectProperty {
            object,
            name: name.into(),
            response,
        })
    }

    fn set_object_property(
        &self,
        object: MxArray,
        name: &str,
        value: MxArray,
    ) -> Result<MxArray, MexDiagnostic> {
        self.request(|response| BoundaryRequest::SetObjectProperty {
            object,
            name: name.into(),
            value,
            response,
        })
    }
}

fn service_request(request: BoundaryRequest, services: &dyn MexBoundaryHostServices) {
    match request {
        BoundaryRequest::Eval { command, response } => {
            let _ = response.send(services.eval(&command));
        }
        BoundaryRequest::Call {
            function,
            arguments,
            requested_outputs,
            response,
        } => {
            let _ = response.send(services.call(&function, arguments, requested_outputs));
        }
        BoundaryRequest::GetVariable {
            workspace,
            name,
            response,
        } => {
            let _ = response.send(services.get_variable(&workspace, &name));
        }
        BoundaryRequest::PutVariable {
            workspace,
            name,
            value,
            response,
        } => {
            let _ = response.send(services.put_variable(&workspace, &name, value));
        }
        BoundaryRequest::GetObjectProperty {
            object,
            name,
            response,
        } => {
            let _ = response.send(services.get_object_property(object, &name));
        }
        BoundaryRequest::SetObjectProperty {
            object,
            name,
            value,
            response,
        } => {
            let _ = response.send(services.set_object_property(object, &name, value));
        }
    }
}

fn lane_diagnostic(message: &str) -> MexDiagnostic {
    MexDiagnostic {
        identifier: Some("RunMat:MEX:NativeLane".into()),
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    struct EchoServices {
        origin: std::thread::ThreadId,
        calls: Cell<usize>,
    }

    impl MexBoundaryHostServices for EchoServices {
        fn eval(&self, _: &str) -> Result<(), MexDiagnostic> {
            Err(lane_diagnostic("unexpected eval callback"))
        }

        fn call(
            &self,
            function: &str,
            arguments: Vec<MxArray>,
            requested_outputs: usize,
        ) -> Result<Vec<MxArray>, MexDiagnostic> {
            assert_eq!(std::thread::current().id(), self.origin);
            assert_eq!(function, "lane_echo");
            assert_eq!(requested_outputs, 1);
            self.calls.set(self.calls.get() + 1);
            Ok(arguments)
        }

        fn get_variable(&self, _: &str, _: &str) -> Result<Option<MxArray>, MexDiagnostic> {
            Err(lane_diagnostic("unexpected workspace read"))
        }

        fn put_variable(&self, _: &str, _: &str, _: MxArray) -> Result<(), MexDiagnostic> {
            Err(lane_diagnostic("unexpected workspace write"))
        }

        fn get_object_property(&self, _: MxArray, _: &str) -> Result<MxArray, MexDiagnostic> {
            Err(lane_diagnostic("unexpected property read"))
        }

        fn set_object_property(
            &self,
            _: MxArray,
            _: &str,
            _: MxArray,
        ) -> Result<MxArray, MexDiagnostic> {
            Err(lane_diagnostic("unexpected property write"))
        }
    }

    #[test]
    fn native_lane_runs_module_while_origin_services_callbacks() {
        let directory = tempfile::tempdir().expect("temporary C++ MEX directory");
        let source = directory.path().join("native_lane.cpp");
        std::fs::write(
            &source,
            r#"
#include "mex.hpp"
#include "mexAdapter.hpp"

class MexFunction : public matlab::mex::Function {
public:
    void operator()(matlab::mex::ArgumentList outputs,
                    matlab::mex::ArgumentList inputs) override {
        auto engine = getEngine();
        outputs[0] = engine->feval(u"lane_echo", inputs[0]);
    }
};
"#,
        )
        .expect("write native-lane fixture");
        let artifact = runmat_mex::MexBuild::new(&source, directory.path())
            .compile()
            .expect("compile native-lane fixture");
        let lane = NativeMexLane::spawn().expect("start native lane");
        let metadata = futures::executor::block_on(lane.load(&artifact.module))
            .expect("load native-lane fixture");
        let values = runmat_mex::MxValueContext::new();
        let input = values
            .encode(
                &runmat_value::Value::Tensor(
                    runmat_value::Tensor::new(vec![2.0, 4.0, 8.0], vec![3, 1])
                        .expect("valid input tensor"),
                ),
                metadata.mode,
                metadata.interface,
            )
            .expect("encode native input");
        let services = EchoServices {
            origin: std::thread::current().id(),
            calls: Cell::new(0),
        };
        let invocation =
            futures::executor::block_on(lane.invoke(&artifact.module, vec![input], 1, &services))
                .expect("invoke native-lane fixture");
        assert_eq!(services.calls.get(), 1);
        let output = values
            .decode(&invocation.outputs[0])
            .expect("decode native output");
        let runmat_value::Value::Tensor(output) = output else {
            panic!("native lane must preserve the tensor output");
        };
        assert_eq!(output.as_f64_slice(), Some(&[2.0, 4.0, 8.0][..]));
        assert_eq!(output.shape, vec![3, 1]);
    }
}
