use runmat_value::{CharArray, Value};

pub(super) struct Outcome {
    status: f64,
    message: String,
    message_id: String,
}

impl Outcome {
    pub(super) fn success() -> Self {
        Self {
            status: 0.0,
            message: String::new(),
            message_id: String::new(),
        }
    }

    pub(super) fn failure(failure: super::errors::Failure) -> Self {
        Self {
            status: 1.0,
            message: failure.message,
            message_id: failure.descriptor.identifier.unwrap_or_default().to_owned(),
        }
    }

    pub(super) fn value(self, requested_outputs: Option<usize>) -> Value {
        let Some(requested_outputs) = requested_outputs else {
            return Value::Num(self.status);
        };
        if requested_outputs == 0 {
            return Value::OutputList(Vec::new());
        }
        let outputs = vec![
            Value::Num(self.status),
            character_row(&self.message),
            character_row(&self.message_id),
        ];
        crate::output_count::output_list_with_padding(requested_outputs, outputs)
    }
}

fn character_row(text: &str) -> Value {
    Value::CharArray(CharArray::new_row(text))
}
