use runmat_builtins::BuiltinErrorDescriptor;
use runmat_value::{CharArray, Value};

#[derive(Debug, Clone)]
pub(super) struct TransferOutcome {
    status: f64,
    message: String,
    message_id: String,
}

impl TransferOutcome {
    pub(super) fn success() -> Self {
        Self {
            status: 1.0,
            message: String::new(),
            message_id: String::new(),
        }
    }

    pub(super) fn failure(
        message: impl Into<String>,
        error: &'static BuiltinErrorDescriptor,
    ) -> Self {
        Self {
            status: 0.0,
            message: message.into(),
            message_id: super::input::message_identifier(error).to_string(),
        }
    }

    pub(super) fn first_output(&self) -> Value {
        Value::Num(self.status)
    }

    pub(super) fn outputs(&self) -> Vec<Value> {
        vec![
            Value::Num(self.status),
            Value::CharArray(CharArray::new_row(&self.message)),
            Value::CharArray(CharArray::new_row(&self.message_id)),
        ]
    }

    pub(super) fn without_outputs(self, builtin: &'static str) -> crate::BuiltinResult<Value> {
        if self.status == 0.0 {
            return Err(crate::build_runtime_error(self.message)
                .with_builtin(builtin)
                .with_identifier(self.message_id)
                .build());
        }
        Ok(Value::output_list(Vec::new()))
    }

    #[cfg(test)]
    pub(super) fn status(&self) -> f64 {
        self.status
    }

    #[cfg(test)]
    pub(super) fn message(&self) -> &str {
        &self.message
    }

    #[cfg(test)]
    pub(super) fn message_id(&self) -> &str {
        &self.message_id
    }
}
