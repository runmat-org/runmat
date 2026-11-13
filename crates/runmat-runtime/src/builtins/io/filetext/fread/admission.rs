use super::*;

enum PreparedValue<'a, T> {
    Parsed(T),
    Resident(&'a Value),
}

impl<'a, T> PreparedValue<'a, T> {
    fn from_control(
        value: Option<&'a Value>,
        parser: fn(Option<&Value>) -> Result<T, String>,
        error: &'static BuiltinErrorDescriptor,
    ) -> BuiltinResult<Self> {
        match value {
            Some(resident @ Value::GpuTensor(_)) => Ok(Self::Resident(resident)),
            other => Ok(Self::Parsed(map_string_result(parser(other), error)?)),
        }
    }

    async fn resolve(
        self,
        parser: fn(Option<&Value>) -> Result<T, String>,
        error: &'static BuiltinErrorDescriptor,
    ) -> BuiltinResult<T> {
        match self {
            Self::Parsed(value) => Ok(value),
            Self::Resident(value) => {
                let host = gather_value(value).await?;
                map_string_result(parser(Some(&host)), error)
            }
        }
    }
}

pub(super) struct PreparedFreadCall<'a> {
    fid: PreparedValue<'a, i32>,
    size: PreparedValue<'a, SizeSpec>,
    precision: PrecisionSpec,
    skip: PreparedValue<'a, usize>,
    machine: Option<MachineFormat>,
    pub(super) like: Option<&'a Value>,
}

pub(super) struct ResolvedFreadControls {
    pub(super) size: SizeSpec,
    pub(super) precision: PrecisionSpec,
    pub(super) skip: usize,
    pub(super) machine: MachineFormat,
}

impl<'a> PreparedFreadCall<'a> {
    pub(super) fn new(
        fid: &'a Value,
        rest: crate::call::FunctionArgumentView<'a>,
    ) -> BuiltinResult<Self> {
        let (size, precision, skip, machine, like) =
            map_string_result(classify_arguments(rest), &FREAD_ERROR_INVALID_INPUT)?;
        if like.is_some() {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FREAD_LIKE_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
        preflight_control(fid, ControlRole::FileId)?;
        if let Some(value) = size {
            preflight_control(value, ControlRole::Size)?;
        }
        if let Some(value) = skip {
            preflight_control(value, ControlRole::Skip)?;
        }
        let size = PreparedValue::from_control(size, parse_size, &FREAD_ERROR_INVALID_INPUT)?;
        let precision = map_string_result(parse_precision(precision), &FREAD_ERROR_INVALID_OPTION)?;
        let skip = PreparedValue::from_control(skip, parse_skip, &FREAD_ERROR_INVALID_OPTION)?;
        let machine = machine
            .map(|value| {
                map_string_result(
                    parse_machine_format(Some(value), "native"),
                    &FREAD_ERROR_INVALID_OPTION,
                )
            })
            .transpose()?;
        if let Some(prototype) = like {
            map_string_result(
                validate_like_prototype(prototype),
                &FREAD_ERROR_INVALID_OPTION,
            )?;
            map_string_result(
                validate_like_output_kind(prototype, precision.output),
                &FREAD_ERROR_INVALID_OPTION,
            )?;
        }
        let fid = match fid {
            resident @ Value::GpuTensor(_) => PreparedValue::Resident(resident),
            host => PreparedValue::Parsed(validate_fid(map_string_result(
                parse_fid(host),
                &FREAD_ERROR_INVALID_INPUT,
            )?)?),
        };
        Ok(Self {
            fid,
            size,
            precision,
            skip,
            machine,
            like,
        })
    }

    pub(super) async fn resolve_fid(&self) -> BuiltinResult<i32> {
        match &self.fid {
            PreparedValue::Parsed(fid) => Ok(*fid),
            PreparedValue::Resident(value) => {
                let host = gather_value(value).await?;
                validate_fid(map_string_result(
                    parse_fid(&host),
                    &FREAD_ERROR_INVALID_INPUT,
                )?)
            }
        }
    }

    pub(super) async fn resolve_controls(
        self,
        default_machine: &str,
    ) -> BuiltinResult<ResolvedFreadControls> {
        let size = self
            .size
            .resolve(parse_size, &FREAD_ERROR_INVALID_INPUT)
            .await?;
        let skip = self
            .skip
            .resolve(parse_skip, &FREAD_ERROR_INVALID_OPTION)
            .await?;
        let machine = match self.machine {
            Some(machine) => machine,
            None => map_string_result(
                parse_machine_format(None, default_machine),
                &FREAD_ERROR_INVALID_OPTION,
            )?,
        };
        Ok(ResolvedFreadControls {
            size,
            precision: self.precision,
            skip,
            machine,
        })
    }
}

fn validate_fid(fid: i32) -> BuiltinResult<i32> {
    if fid < 0 {
        return Err(fread_error_with_detail(
            &FREAD_ERROR_INVALID_INPUT,
            "file identifier must be non-negative",
        ));
    }
    if fid < 3 {
        return Err(fread_error_with_detail(
            &FREAD_ERROR_INVALID_INPUT,
            "standard input/output identifiers are not supported yet",
        ));
    }
    Ok(fid)
}
