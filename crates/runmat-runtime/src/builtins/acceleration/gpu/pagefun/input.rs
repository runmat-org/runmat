use super::*;
#[derive(Clone)]
pub(super) struct PageInput {
    pub(super) page_dims: Vec<usize>,
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) data: PageData,
}

#[derive(Clone)]
pub(super) enum PageData {
    Real(NumericStorage),
    Complex(Vec<(f64, f64)>, NumericDType),
}

impl PageInput {
    pub(super) fn from_value(value: Value) -> BuiltinResult<Self> {
        match value {
            value @ (Value::Tensor(_) | Value::Num(_) | Value::Int(_)) => Self::from_tensor(
                tensor_utils::value_into_tensor_for(BUILTIN_NAME, value).map_err(|error| {
                    pagefun_error_with_detail(&PAGEFUN_ERROR_INVALID_INPUT, &error)
                })?,
            ),
            Value::Bool(flag) => Self::from_tensor(
                Tensor::new(vec![if flag { 1.0 } else { 0.0 }], vec![1, 1])
                    .map_err(|e| pagefun_error_with_detail(&PAGEFUN_ERROR_INTERNAL, &e))?,
            ),
            Value::LogicalArray(array) => {
                Self::from_tensor(tensor_utils::logical_to_tensor(&array).map_err(|error| {
                    pagefun_error_with_detail(&PAGEFUN_ERROR_INVALID_INPUT, error)
                })?)
            }
            Value::Complex(re, im) => {
                let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                    .map_err(|e| pagefun_error_with_detail(&PAGEFUN_ERROR_INTERNAL, &e))?;
                Self::from_complex_tensor(tensor)
            }
            Value::ComplexTensor(t) => Self::from_complex_tensor(t),
            other => Err(pagefun_error_with_detail(
                &PAGEFUN_ERROR_INVALID_INPUT,
                format!("unsupported input type {}", other.type_name()),
            )),
        }
    }

    pub(super) fn from_tensor(tensor: Tensor) -> BuiltinResult<Self> {
        let shape = canonical_matrix_shape(tensor.shape());
        let data = tensor
            .into_numeric_storage()
            .map_err(|error| pagefun_error_with_detail(&PAGEFUN_ERROR_INTERNAL, error))?;
        if data.len() != checked_volume(&shape)? {
            return Err(pagefun_error_with_detail(
                &PAGEFUN_ERROR_INTERNAL,
                "tensor data does not match its shape",
            ));
        }
        let rows = shape[0];
        let cols = shape[1];
        let page_dims = if shape.len() > 2 {
            shape[2..].to_vec()
        } else {
            Vec::new()
        };
        Ok(Self {
            page_dims,
            rows,
            cols,
            data: PageData::Real(data),
        })
    }

    pub(super) fn from_complex_tensor(tensor: ComplexTensor) -> BuiltinResult<Self> {
        let shape = canonical_matrix_shape(tensor.shape());
        if tensor.len() != checked_volume(&shape)? {
            return Err(pagefun_error_with_detail(
                &PAGEFUN_ERROR_INTERNAL,
                "tensor data does not match its shape",
            ));
        }
        let rows = shape[0];
        let cols = shape[1];
        let page_dims = if shape.len() > 2 {
            shape[2..].to_vec()
        } else {
            Vec::new()
        };
        Ok(Self {
            page_dims,
            rows,
            cols,
            data: PageData::Complex(tensor.materialize_f64(), tensor.numeric_dtype()),
        })
    }

    pub(super) fn page_size(&self) -> usize {
        self.rows * self.cols
    }

    pub(super) fn is_complex(&self) -> bool {
        matches!(self.data, PageData::Complex(..))
    }

    pub(super) fn numeric_dtype(&self) -> Option<NumericDType> {
        match &self.data {
            PageData::Real(storage) => Some(storage.numeric_dtype()),
            PageData::Complex(_, dtype) => Some(*dtype),
        }
    }
}

pub(super) struct PreparedInput {
    pub(super) data: PageInput,
    pub(super) padded_dims: Vec<usize>,
    pub(super) strides: Vec<usize>,
}

impl PreparedInput {
    pub(super) fn new(input: PageInput, rank: usize) -> Self {
        let mut padded = input.page_dims.clone();
        padded.resize(rank, 1);
        let strides = compute_strides(&padded);
        Self {
            data: input,
            padded_dims: padded,
            strides,
        }
    }

    pub(super) fn rows(&self) -> usize {
        self.data.rows
    }

    pub(super) fn cols(&self) -> usize {
        self.data.cols
    }

    pub(super) fn is_complex(&self) -> bool {
        self.data.is_complex()
    }

    pub(super) fn page_value(&self, multi_index: &[usize]) -> BuiltinResult<Value> {
        let mut linear_page = 0usize;
        for (dim, stride) in self.strides.iter().enumerate() {
            let source_extent = self.padded_dims.get(dim).copied().unwrap_or(1);
            let requested = multi_index.get(dim).copied().unwrap_or(0);
            if source_extent == 0 {
                return Err(pagefun_internal_error("source page extent is zero"));
            }
            if source_extent != 1 && requested >= source_extent {
                return Err(pagefun_internal_error("page index out of bounds"));
            }
            let actual = if source_extent == 1 { 0 } else { requested };
            linear_page += actual * stride;
        }

        let offset = linear_page * self.data.page_size();
        match &self.data.data {
            PageData::Real(buffer) => {
                let end = offset + self.data.page_size();
                if end > buffer.len() {
                    return Err(pagefun_internal_error("page slice out of bounds"));
                }
                let indices = (offset..end).collect::<Vec<_>>();
                let storage = buffer
                    .gather(&indices)
                    .map_err(|error| pagefun_error_with_detail(&PAGEFUN_ERROR_INTERNAL, error))?;
                let tensor =
                    Tensor::from_numeric_storage(storage, vec![self.data.rows, self.data.cols])
                        .map_err(|e| pagefun_error_with_detail(&PAGEFUN_ERROR_INTERNAL, &e))?;
                Ok(Value::Tensor(tensor))
            }
            PageData::Complex(buffer, dtype) => {
                let end = offset + self.data.page_size();
                let slice = buffer
                    .get(offset..end)
                    .ok_or_else(|| pagefun_internal_error("page slice out of bounds"))?;
                let tensor = ComplexTensor::from_f64_values_with_dtype(
                    slice.to_vec(),
                    vec![self.data.rows, self.data.cols],
                    *dtype,
                )
                .map_err(|e| pagefun_error_with_detail(&PAGEFUN_ERROR_INTERNAL, &e))?;
                Ok(Value::ComplexTensor(tensor))
            }
        }
    }
}

pub(super) fn compute_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(dims.len());
    let mut stride = 1usize;
    for &dim in dims {
        strides.push(stride);
        stride = stride.saturating_mul(dim.max(1));
    }
    strides
}
