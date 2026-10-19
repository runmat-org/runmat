/// A Send admission transferred with one interpreter-lane command. The owner
/// supplies the accounting policy; this crate knows only its physical extent.
pub trait PythonLaneObserver: Send {
    fn enter(self: Box<Self>) -> Box<dyn PythonLaneObservation>;
}

/// Created and consumed on the interpreter thread. Dropping an uncompleted
/// observation leaves its owner to record an unverified terminal outcome.
pub trait PythonLaneObservation {
    fn complete(self: Box<Self>, succeeded: bool);
}

pub(crate) fn observe<T, E>(
    observer: Option<Box<dyn PythonLaneObserver>>,
    operation: impl FnOnce() -> Result<T, E>,
) -> Result<T, E> {
    let active = observer.map(|observer| observer.enter());
    let result = operation();
    if let Some(active) = active {
        active.complete(result.is_ok());
    }
    result
}
