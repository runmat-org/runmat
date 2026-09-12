use crate::RuntimeError;

pub(crate) fn is_nonfatal_plot_setup_error(error: &RuntimeError) -> bool {
    let message = error.to_string().to_ascii_lowercase();
    message.contains("plotting is unavailable")
        || message.contains("non-main thread")
        || message.contains("interactive plotting failed")
        || message.contains("eventloop can't be recreated")
}

#[cfg(test)]
mod tests {
    use super::is_nonfatal_plot_setup_error;

    #[test]
    fn recognizes_environment_setup_failures_only() {
        for message in [
            "plotting is unavailable in this build",
            "plotting requires the non-main thread to stop",
            "interactive plotting failed during setup",
            "eventloop can't be recreated",
        ] {
            let error = crate::build_runtime_error(message).build();
            assert!(is_nonfatal_plot_setup_error(&error), "{message}");
        }

        let error = crate::build_runtime_error("plot data is invalid").build();
        assert!(!is_nonfatal_plot_setup_error(&error));
    }
}
