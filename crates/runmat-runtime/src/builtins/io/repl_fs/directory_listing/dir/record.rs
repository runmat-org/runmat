use chrono::{DateTime, Duration, Local, NaiveDate};
use runmat_filesystem::FsMetadata;
use std::time::SystemTime;

#[derive(Clone)]
pub(super) struct Record {
    pub name: String,
    pub folder: String,
    pub date: String,
    pub bytes: f64,
    pub is_dir: bool,
    pub datenum: f64,
}

impl Record {
    pub(super) fn from_metadata(folder: String, name: String, metadata: &FsMetadata) -> Self {
        let is_dir = metadata.is_dir();
        let (date, datenum) = timestamp(metadata.modified());
        Self {
            name,
            folder,
            date,
            bytes: if is_dir { 0.0 } else { metadata.len() as f64 },
            is_dir,
            datenum,
        }
    }

    pub(super) fn special(name: &str, folder: &str, metadata: Option<&FsMetadata>) -> Self {
        let (date, datenum) = timestamp(metadata.and_then(FsMetadata::modified));
        Self {
            name: name.into(),
            folder: folder.into(),
            date,
            bytes: 0.0,
            is_dir: true,
            datenum,
        }
    }
}

fn timestamp(time: Option<SystemTime>) -> (String, f64) {
    const DEFAULT_DATE: &str = "01-Jan-1970 00:00:00";
    const DEFAULT_DATENUM: f64 = 719_529.0;
    let Some(time) = time else {
        return (DEFAULT_DATE.into(), DEFAULT_DATENUM);
    };
    let datetime: DateTime<Local> = DateTime::from(time);
    (
        datetime.format("%d-%b-%Y %H:%M:%S").to_string(),
        datenum(datetime),
    )
}

fn datenum(datetime: DateTime<Local>) -> f64 {
    const SECONDS_PER_DAY: f64 = 86_400.0;
    const UNIX_DATENUM: f64 = 719_529.0;
    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1)
        .and_then(|date| date.and_hms_opt(0, 0, 0))
        .expect("the Unix epoch is a valid date");
    let duration = datetime.naive_local() - epoch;
    let seconds = duration.num_seconds();
    let nanos = (duration - Duration::seconds(seconds))
        .num_nanoseconds()
        .unwrap_or(0);
    (seconds as f64 + nanos as f64 / 1_000_000_000.0) / SECONDS_PER_DAY + UNIX_DATENUM
}
