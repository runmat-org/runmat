//! Bounded non-WAVE error classification. The historical MPEG test accepts a
//! valid frame at any offset after an optional ID3v2 header, so scan with a
//! three-byte overlap instead of buffering the entire file.

use super::*;

pub(super) async fn non_wave<R: Read + Seek>(
    file: &mut R,
    path: &Path,
    file_size: u64,
    magic: [u8; 4],
    cooperation: &mut AudioCooperation,
) -> BuiltinResult<&'static str> {
    if magic == *b"fLaC" || magic == *b"FORM" || magic == *b"OggS" {
        return Ok(COMPRESSED_AUDIO_ERROR);
    }
    let mut start = 0u64;
    if &magic[..3] == b"ID3" && file_size >= 10 {
        file.seek(SeekFrom::Start(0))
            .map_err(|source| io_error(path, source))?;
        let mut header = [0u8; 10];
        file.read_exact(&mut header)
            .map_err(|source| io_error(path, source))?;
        start = u64::try_from(skip_id3v2(&header)).expect("ID3 size fits u64");
    }
    if start > file_size {
        return Ok(UNSUPPORTED_AUDIO_ERROR);
    }
    file.seek(SeekFrom::Start(start))
        .map_err(|source| io_error(path, source))?;
    let mut buffer = [0u8; AUDIO_WORK_CHUNK + 3];
    let mut carry = 0usize;
    loop {
        cooperation.completed(0).await?;
        let read = file
            .read(&mut buffer[carry..])
            .map_err(|source| io_error(path, source))?;
        if read == 0 {
            break;
        }
        cooperation.completed(read).await?;
        let total = carry + read;
        if total >= 4 {
            for at in 0..=total - 4 {
                if buffer[at] == 0xff
                    && buffer[at + 1] & 0xe0 == 0xe0
                    && parse_mpeg_audio_header(&buffer[at..at + 4]).is_some()
                {
                    return Ok(COMPRESSED_AUDIO_ERROR);
                }
            }
        }
        carry = total.min(3);
        buffer.copy_within(total - carry..total, 0);
    }
    Ok(UNSUPPORTED_AUDIO_ERROR)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mpeg_header_crossing_input_chunk_is_classified_without_whole_file_buffer() {
        let mut bytes = vec![0u8; AUDIO_WORK_CHUNK - 2];
        bytes.extend_from_slice(&[0xff, 0xfb, 0x90, 0x64]);
        let mut cursor = std::io::Cursor::new(bytes.as_slice());
        let mut cooperation = AudioCooperation::new(None, AUDIOREAD_BUILTIN_NAME);
        let result = block_on(non_wave(
            &mut cursor,
            Path::new("x.mp3"),
            bytes.len() as u64,
            [0, 0, 0, 0],
            &mut cooperation,
        ))
        .expect("classification");
        assert_eq!(result, COMPRESSED_AUDIO_ERROR);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn id3_skip_and_unknown_payload_match_legacy_classifier() {
        let mut tagged = b"ID3\x03\x00\x00\x00\x00\x00\x04skip".to_vec();
        tagged.extend_from_slice(&[0xff, 0xfb, 0x90, 0x64]);
        let mut cursor = std::io::Cursor::new(tagged.as_slice());
        let mut cooperation = AudioCooperation::new(None, AUDIOREAD_BUILTIN_NAME);
        let result = block_on(non_wave(
            &mut cursor,
            Path::new("x.mp3"),
            tagged.len() as u64,
            *b"ID3\x03",
            &mut cooperation,
        ))
        .expect("classification");
        assert_eq!(result, COMPRESSED_AUDIO_ERROR);
        let mut unknown = std::io::Cursor::new(&b"not audio"[..]);
        assert_eq!(
            block_on(non_wave(
                &mut unknown,
                Path::new("x.bin"),
                9,
                *b"not ",
                &mut cooperation
            ))
            .unwrap(),
            UNSUPPORTED_AUDIO_ERROR
        );
    }
}
