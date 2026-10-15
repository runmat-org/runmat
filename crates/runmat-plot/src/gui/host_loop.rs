//! Main-thread owner of native windows and their post-present frame ports.
//!
//! Only metadata, figures, and Send frame tickets cross into this loop. A
//! RunMat session, RuntimeContext, callable, or GC root never does.

use super::frame_command::{FrameCommand, FrameError, FramePort};
use super::lifecycle::{self, frame_ports};
use super::{PlotWindow, WindowConfig};
use futures::channel::oneshot;
use runmat_time::Instant;
use std::collections::HashMap;
use std::future::Future;
use std::num::NonZeroUsize;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError};
use std::sync::{Arc, Weak};
use std::task::{Context, Poll};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, Event, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
use winit::window::{Window, WindowId};

const QUEUED: u8 = 0;
const RETIRED: u8 = 1;
const DISPATCHED: u8 = 2;
const OPENED: u8 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowOpenDisposition {
    NotOpened,
    Indeterminate,
    Opened,
}

#[derive(Debug, thiserror::Error)]
pub enum WindowOpenError {
    #[error("native window-open queue is full")]
    Full,
    #[error("native window host is closed")]
    Closed,
    #[error("a zero figure handle cannot own a native window")]
    InvalidHandle,
    #[error("native window capacity is exhausted")]
    Capacity,
    #[error("native figure-window route admission failed: {0}")]
    Routing(String),
    #[error("native window creation failed: {0}")]
    Create(String),
}

struct OpenRequest {
    figure_handle: u32,
    config: WindowConfig,
    response: oneshot::Sender<Result<WindowOpenReceipt, WindowOpenError>>,
    state: Arc<AtomicU8>,
}

/// Send-only opener; polling the returned ticket admits a window request.
#[derive(Clone)]
pub struct WindowOpenPort {
    sender: SyncSender<OpenRequest>,
    shutdown: Arc<AtomicBool>,
    host_lifetime: Weak<()>,
}

impl WindowOpenPort {
    /// Descriptive liveness for admission, not a reservation against later close.
    pub fn is_closed(&self) -> bool {
        self.shutdown.load(Ordering::Acquire) || self.host_lifetime.upgrade().is_none()
    }
    pub fn open(&self, figure_handle: u32, config: WindowConfig) -> WindowOpenTicket {
        let (response, receiver) = oneshot::channel();
        let state = Arc::new(AtomicU8::new(QUEUED));
        WindowOpenTicket {
            sender: self.sender.clone(),
            request: Some(OpenRequest {
                figure_handle,
                config,
                response,
                state: Arc::clone(&state),
            }),
            receiver,
            state,
        }
    }

    /// Signals the OS-main-thread loop to retire its windows. This is not a
    /// callback or session cancellation mechanism.
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
    }
}

/// Dropping a queued open ticket atomically excludes later window creation.
pub struct WindowOpenTicket {
    sender: SyncSender<OpenRequest>,
    request: Option<OpenRequest>,
    receiver: oneshot::Receiver<Result<WindowOpenReceipt, WindowOpenError>>,
    state: Arc<AtomicU8>,
}

impl WindowOpenTicket {
    pub fn disposition(&self) -> WindowOpenDisposition {
        match self.state.load(Ordering::Acquire) {
            QUEUED | RETIRED => WindowOpenDisposition::NotOpened,
            DISPATCHED => WindowOpenDisposition::Indeterminate,
            OPENED => WindowOpenDisposition::Opened,
            _ => unreachable!("native window-open state"),
        }
    }

    pub fn abandon(&self) -> WindowOpenDisposition {
        let state =
            match self
                .state
                .compare_exchange(QUEUED, RETIRED, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => RETIRED,
                Err(state) => state,
            };
        match state {
            QUEUED | RETIRED => WindowOpenDisposition::NotOpened,
            DISPATCHED => WindowOpenDisposition::Indeterminate,
            OPENED => WindowOpenDisposition::Opened,
            _ => unreachable!("native window-open state"),
        }
    }
}

impl Future for WindowOpenTicket {
    type Output = Result<WindowOpenReceipt, WindowOpenError>;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        if let Some(request) = self.request.take() {
            if self.state.load(Ordering::Acquire) == RETIRED {
                return Poll::Ready(Err(WindowOpenError::Closed));
            }
            if request.figure_handle == 0 {
                self.abandon();
                return Poll::Ready(Err(WindowOpenError::InvalidHandle));
            }
            match self.sender.try_send(request) {
                Ok(()) => {}
                Err(TrySendError::Full(_)) => {
                    self.abandon();
                    return Poll::Ready(Err(WindowOpenError::Full));
                }
                Err(TrySendError::Disconnected(_)) => {
                    self.abandon();
                    return Poll::Ready(Err(WindowOpenError::Closed));
                }
            }
        }
        match Pin::new(&mut self.receiver).poll(context) {
            Poll::Ready(Ok(result)) => Poll::Ready(result),
            Poll::Ready(Err(_)) => Poll::Ready(Err(WindowOpenError::Closed)),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl Drop for WindowOpenTicket {
    fn drop(&mut self) {
        self.abandon();
    }
}

/// Linear window lifetime authority. Routing-table clones do not retain it.
/// Even an acknowledged open dropped before its receiver polls closes only
/// this window, including the dispatched-open/cancellation race.
pub struct WindowOpenReceipt {
    port: FramePort,
}

impl WindowOpenReceipt {
    pub fn frame_port(&self) -> &FramePort {
        &self.port
    }
}

impl Drop for WindowOpenReceipt {
    fn drop(&mut self) {
        self.port.close();
    }
}

struct HostedWindow {
    window: PlotWindow<'static>,
    last_render_time: Instant,
    modifiers: ModifiersState,
    _frame_registration: frame_ports::FramePortRegistration,
    _lifecycle_registration: lifecycle::WindowRegistration,
}

/// Create and run this owner on the OS main thread. The execution session is
/// constructed separately on its own current-thread lane after `open_port`
/// has been transferred; no session-local Rc may be moved into this owner.
pub struct NativeGraphicsHost {
    event_loop: EventLoop<FrameCommand>,
    opens: Receiver<OpenRequest>,
    sender: SyncSender<OpenRequest>,
    shutdown: Arc<AtomicBool>,
    maximum_windows: NonZeroUsize,
    host_lifetime: Arc<()>,
}

impl NativeGraphicsHost {
    pub fn new(
        maximum_pending_opens: NonZeroUsize,
        maximum_windows: NonZeroUsize,
    ) -> Result<Self, WindowOpenError> {
        let event_loop = EventLoop::<FrameCommand>::with_user_event()
            .build()
            .map_err(|error| WindowOpenError::Create(error.to_string()))?;
        let (sender, opens) = mpsc::sync_channel(maximum_pending_opens.get());
        Ok(Self {
            event_loop,
            opens,
            sender,
            shutdown: Arc::new(AtomicBool::new(false)),
            maximum_windows,
            host_lifetime: Arc::new(()),
        })
    }

    pub fn open_port(&self) -> WindowOpenPort {
        WindowOpenPort {
            sender: self.sender.clone(),
            shutdown: Arc::clone(&self.shutdown),
            host_lifetime: Arc::downgrade(&self.host_lifetime),
        }
    }

    pub fn run(self) -> Result<(), WindowOpenError> {
        let Self {
            event_loop,
            opens,
            sender,
            shutdown,
            maximum_windows,
            host_lifetime,
        } = self;
        // The host is a receiver, not its own producer. Releasing this clone
        // lets worker-lane retirement terminate the loop if no opener remains.
        drop(sender);
        let mut owner = HostApplication {
            opens,
            proxy: event_loop.create_proxy(),
            shutdown,
            maximum_windows: maximum_windows.get(),
            windows: HashMap::new(),
            opens_disconnected: false,
            _host_lifetime: host_lifetime,
        };
        event_loop
            .run_app(&mut owner)
            .map_err(|error| WindowOpenError::Create(error.to_string()))
    }
}

struct HostApplication {
    opens: Receiver<OpenRequest>,
    proxy: EventLoopProxy<FrameCommand>,
    shutdown: Arc<AtomicBool>,
    maximum_windows: usize,
    windows: HashMap<WindowId, HostedWindow>,
    opens_disconnected: bool,
    _host_lifetime: Arc<()>,
}

impl ApplicationHandler<FrameCommand> for HostApplication {
    fn resumed(&mut self, _target: &ActiveEventLoop) {}

    fn user_event(&mut self, _target: &ActiveEventLoop, command: FrameCommand) {
        let Some(window_id) = command.window_id() else {
            return;
        };
        if self
            .windows
            .get(&window_id)
            .is_some_and(HostedWindow::close_requested)
        {
            self.windows.remove(&window_id);
            return;
        }
        let Some(hosted) = self.windows.get_mut(&window_id) else {
            return;
        };
        let Some(lifetime) = hosted._lifecycle_registration.signal() else {
            return;
        };
        if !command.matches_window_lifetime(&lifetime) {
            return;
        }
        let Some(mut frame) = command.begin() else {
            return;
        };
        let figure = frame.take_figure();
        let result = hosted
            .window
            .set_figure(figure)
            .map_err(|error| FrameError::Prepare(error.to_string()))
            .and_then(|()| {
                hosted
                    .window
                    .render(std::time::Duration::ZERO)
                    .map_err(|error| FrameError::Render(error.to_string()))
            });
        // render() returns only after output.present(); mere delivery is not a receipt.
        frame.complete(result);
    }

    fn window_event(&mut self, target: &ActiveEventLoop, window_id: WindowId, event: WindowEvent) {
        if self
            .windows
            .get(&window_id)
            .is_some_and(HostedWindow::close_requested)
        {
            self.windows.remove(&window_id);
            return;
        }
        let Some(hosted) = self.windows.get_mut(&window_id) else {
            return;
        };
        let close = match &event {
            WindowEvent::CloseRequested => true,
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                matches!(event.physical_key, PhysicalKey::Code(KeyCode::Escape))
                    || (matches!(event.physical_key, PhysicalKey::Code(KeyCode::KeyQ))
                        && (hosted.modifiers.super_key() || hosted.modifiers.control_key()))
            }
            _ => false,
        };
        if close {
            self.windows.remove(&window_id);
            return;
        }
        hosted.window.handle_event(
            Event::WindowEvent { window_id, event },
            target,
            &mut hosted.last_render_time,
            &None,
            &mut hosted.modifiers,
        );
    }

    fn about_to_wait(&mut self, target: &ActiveEventLoop) {
        if self.shutdown.load(Ordering::Acquire) {
            target.exit();
            return;
        }
        // Retire exact old window routes before admitting an explicitly
        // requested replacement at a reused figure coordinate.
        self.windows.retain(|_, hosted| !hosted.close_requested());
        loop {
            match self.opens.try_recv() {
                Ok(request) => open_window(
                    request,
                    target,
                    &self.proxy,
                    &mut self.windows,
                    self.maximum_windows,
                ),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.opens_disconnected = true;
                    break;
                }
            }
        }
        for hosted in self.windows.values_mut() {
            hosted.window.handle_event(
                Event::AboutToWait,
                target,
                &mut hosted.last_render_time,
                &None,
                &mut hosted.modifiers,
            );
        }
        if self.windows.is_empty() && self.opens_disconnected {
            target.exit();
        } else {
            // Open ingress is a bounded channel, not a winit event source.
            target.set_control_flow(ControlFlow::WaitUntil(
                std::time::Instant::now() + std::time::Duration::from_millis(16),
            ));
        }
    }
}

impl HostedWindow {
    fn close_requested(&self) -> bool {
        self._lifecycle_registration
            .signal()
            .is_some_and(|signal| signal.load(Ordering::Acquire))
    }
}

fn open_window(
    request: OpenRequest,
    target: &ActiveEventLoop,
    proxy: &EventLoopProxy<FrameCommand>,
    windows: &mut HashMap<WindowId, HostedWindow>,
    maximum_windows: usize,
) {
    if windows.len() >= maximum_windows {
        reject_unpublished_open(request, WindowOpenError::Capacity);
        return;
    }
    let mut frame_registration = match frame_ports::reserve(request.figure_handle) {
        Ok(registration) => registration,
        Err(error) => {
            reject_unpublished_open(request, WindowOpenError::Routing(error));
            return;
        }
    };
    let lifecycle_registration = match lifecycle::register_handle(request.figure_handle) {
        Ok(registration) => registration,
        Err(error) => {
            reject_unpublished_open(request, WindowOpenError::Routing(error));
            return;
        }
    };
    if request
        .state
        .compare_exchange(QUEUED, DISPATCHED, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return;
    }
    let result = (|| {
        let attributes = Window::default_attributes()
            .with_title(&request.config.title)
            .with_inner_size(PhysicalSize::new(
                request.config.width,
                request.config.height,
            ))
            .with_resizable(request.config.resizable)
            .with_maximized(request.config.maximized);
        let native_window = Arc::new(
            target
                .create_window(attributes)
                .map_err(|error| WindowOpenError::Create(error.to_string()))?,
        );
        let mut window =
            pollster::block_on(PlotWindow::new_with_window(request.config, native_window))
                .map_err(|error| WindowOpenError::Create(error.to_string()))?;
        let close_signal = lifecycle_registration
            .signal()
            .expect("nonzero registered window");
        let port = FramePort::new(proxy.clone(), window.window.id(), close_signal);
        frame_registration
            .activate(port.clone())
            .map_err(WindowOpenError::Routing)?;
        if let Some(signal) = lifecycle_registration.signal() {
            window.install_close_signal(signal);
        }
        Ok((
            port,
            HostedWindow {
                window,
                last_render_time: Instant::now(),
                modifiers: ModifiersState::default(),
                _frame_registration: frame_registration,
                _lifecycle_registration: lifecycle_registration,
            },
        ))
    })();
    match result {
        Ok((port, hosted)) => {
            request.state.store(OPENED, Ordering::Release);
            if request
                .response
                .send(Ok(WindowOpenReceipt { port }))
                .is_ok()
            {
                windows.insert(hosted.window.window.id(), hosted);
            }
        }
        Err(error) => {
            let _ = request.response.send(Err(error));
        }
    }
}

fn reject_unpublished_open(request: OpenRequest, error: WindowOpenError) {
    if request
        .state
        .compare_exchange(QUEUED, RETIRED, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
    {
        let _ = request.response.send(Err(error));
    }
}

#[cfg(test)]
#[path = "host_loop/tests.rs"]
mod tests;
