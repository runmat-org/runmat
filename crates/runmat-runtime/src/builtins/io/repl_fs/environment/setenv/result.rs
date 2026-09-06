use runmat_value::{CharArray, Value};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SetenvResult {
    status: u8,
    message: String,
}

impl SetenvResult {
    pub(super) fn success() -> Self {
        Self {
            status: 0,
            message: String::new(),
        }
    }

    pub(super) fn failure(message: impl Into<String>) -> Self {
        Self {
            status: 1,
            message: message.into(),
        }
    }

    pub(super) fn render(self, requested: Option<usize>) -> Value {
        let outputs = vec![
            Value::Num(f64::from(self.status)),
            Value::CharArray(CharArray::new_row(&self.message)),
        ];
        match requested {
            None => outputs[0].clone(),
            Some(0) => Value::OutputList(Vec::new()),
            Some(count) => crate::output_count::output_list_with_padding(count, outputs),
        }
    }

    #[cfg(test)]
    pub(super) fn status(&self) -> u8 {
        self.status
    }
}
