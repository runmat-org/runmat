pub(in crate::builtins::io::repl_fs::path_syntax) fn split(text: &str) -> (String, String, String) {
    if text.is_empty() {
        return (String::new(), String::new(), String::new());
    }
    if ends_with_separator(text) {
        return (trim_trailing_separators(text), String::new(), String::new());
    }
    let separator = text.char_indices().rev().find(|(_, ch)| is_separator(*ch));
    let (folder, filename) = match separator {
        Some((0, ch)) => (ch.to_string(), &text[ch.len_utf8()..]),
        Some((index, ch))
            if cfg!(windows) && index == 2 && text.as_bytes().get(1) == Some(&b':') =>
        {
            (
                text[..index + ch.len_utf8()].to_owned(),
                &text[index + ch.len_utf8()..],
            )
        }
        Some((index, ch)) => (text[..index].to_owned(), &text[index + ch.len_utf8()..]),
        None => (String::new(), text),
    };
    let (name, extension) = match filename.rfind('.') {
        Some(index) => (filename[..index].to_owned(), filename[index..].to_owned()),
        None => (filename.to_owned(), String::new()),
    };
    (folder, name, extension)
}

fn trim_trailing_separators(text: &str) -> String {
    let mut end = text.len();
    while end > 1 {
        let Some((index, ch)) = text[..end].char_indices().next_back() else {
            break;
        };
        if !is_separator(ch) {
            break;
        }
        if cfg!(windows) && index == 2 && text.as_bytes().get(1) == Some(&b':') {
            break;
        }
        end = index;
    }
    text[..end].to_owned()
}

fn ends_with_separator(text: &str) -> bool {
    text.ends_with(std::path::MAIN_SEPARATOR) || (cfg!(windows) && text.ends_with('/'))
}

fn is_separator(ch: char) -> bool {
    ch == std::path::MAIN_SEPARATOR || (cfg!(windows) && ch == '/')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dotfiles_and_trailing_folders_split_lexically() {
        let separator = std::path::MAIN_SEPARATOR;
        assert_eq!(
            split(&format!("home{separator}.profile")),
            ("home".into(), "".into(), ".profile".into())
        );
        assert_eq!(
            split(&format!("home{separator}work{separator}")),
            (format!("home{separator}work"), "".into(), "".into())
        );
    }

    #[test]
    fn filenames_split_at_the_last_extension_separator() {
        assert_eq!(
            split("archive.part.tar"),
            ("".into(), "archive.part".into(), ".tar".into())
        );
        assert_eq!(split("README"), ("".into(), "README".into(), "".into()));
        assert_eq!(split(""), ("".into(), "".into(), "".into()));
    }

    #[test]
    fn root_is_preserved() {
        let separator = std::path::MAIN_SEPARATOR;
        assert_eq!(
            split(&separator.to_string()),
            (separator.to_string(), "".into(), "".into())
        );
    }

    #[cfg(not(windows))]
    #[test]
    fn backslash_is_a_filename_character_on_unix() {
        assert_eq!(
            split(r"folder\name.txt"),
            ("".into(), r"folder\name".into(), ".txt".into())
        );
    }
}
