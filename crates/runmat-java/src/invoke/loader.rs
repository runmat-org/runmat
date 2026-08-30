use std::path::{Path, PathBuf};

use jni::objects::{GlobalRef, JClass, JObject, JValue, JValueOwned};

use super::{conversion::WellKnownJavaClass, error::jni_error, JavaInvocationError, JavaSession};
use crate::{ClasspathLayer, ClasspathSnapshot, SessionClasspath};

impl JavaSession {
    pub fn classpath(&self) -> ClasspathSnapshot {
        self.classpath.borrow().snapshot()
    }

    pub fn add_dynamic_classpath(
        &self,
        entry: impl Into<PathBuf>,
    ) -> Result<ClasspathSnapshot, JavaInvocationError> {
        self.add_dynamic_classpath_at(entry, false)
    }

    pub fn add_dynamic_classpath_at(
        &self,
        entry: impl Into<PathBuf>,
        at_end: bool,
    ) -> Result<ClasspathSnapshot, JavaInvocationError> {
        let entry = existing_classpath_entry(entry.into())?;
        let mut candidate = self.classpath.borrow().clone();
        let mutation = if at_end {
            candidate.add_dynamic(entry)
        } else {
            candidate.add_dynamic_first(entry)
        };
        mutation.map_err(|error| JavaInvocationError::UnsupportedValue(error.to_string()))?;
        self.install_classpath(candidate)
    }

    pub fn replace_dynamic_classpath(
        &self,
        entries: impl IntoIterator<Item = PathBuf>,
    ) -> Result<ClasspathSnapshot, JavaInvocationError> {
        let entries = entries
            .into_iter()
            .map(existing_classpath_entry)
            .collect::<Result<Vec<_>, _>>()?;
        let mut candidate = self.classpath.borrow().clone();
        candidate
            .replace_dynamic(entries)
            .map_err(|error| JavaInvocationError::UnsupportedValue(error.to_string()))?;
        self.install_classpath(candidate)
    }

    pub fn remove_dynamic_classpath(
        &self,
        entry: &Path,
    ) -> Result<ClasspathSnapshot, JavaInvocationError> {
        let entry = existing_classpath_entry(entry.to_path_buf())?;
        let mut candidate = self.classpath.borrow().clone();
        candidate
            .remove_dynamic(&entry)
            .map_err(|error| JavaInvocationError::UnsupportedValue(error.to_string()))?;
        self.install_classpath(candidate)
    }

    pub(super) fn load_class<'local>(
        &self,
        environment: &mut jni::JNIEnv<'local>,
        class_name: &str,
    ) -> Result<JClass<'local>, JavaInvocationError> {
        self.ensure_class_loader(environment)?;
        self.install_context_class_loader(environment)?;
        let loader = self.class_loader.borrow();
        let Some(loader) = loader.as_ref() else {
            return environment
                .find_class(super::conversion::binary_name(class_name))
                .map_err(|error| jni_error(environment, error));
        };
        let loader = environment
            .new_local_ref(loader.as_obj())
            .map_err(|error| jni_error(environment, error))?;
        let name = JObject::from(
            environment
                .new_string(class_name)
                .map_err(|error| jni_error(environment, error))?,
        );
        let class = environment
            .call_static_method(
                "java/lang/Class",
                "forName",
                "(Ljava/lang/String;ZLjava/lang/ClassLoader;)Ljava/lang/Class;",
                &[
                    JValue::Object(&name),
                    JValue::Bool(1),
                    JValue::Object(&loader),
                ],
            )
            .and_then(JValueOwned::l)
            .map_err(|error| jni_error(environment, error))?;
        Ok(JClass::from(class))
    }

    pub fn class_exists(&self, class_name: &str) -> Result<bool, JavaInvocationError> {
        self.process.with_attached(
            |environment| match self.load_class(environment, class_name) {
                Ok(_) => Ok(true),
                Err(JavaInvocationError::Exception(exception))
                    if WellKnownJavaClass::from_name(&exception.class_name)
                        .is_some_and(WellKnownJavaClass::is_missing_class_error) =>
                {
                    Ok(false)
                }
                Err(error) => Err(error),
            },
        )
    }

    fn install_classpath(
        &self,
        candidate: SessionClasspath,
    ) -> Result<ClasspathSnapshot, JavaInvocationError> {
        let entries = loader_entries(&candidate);
        self.process.with_attached(|environment| {
            let loader = build_class_loader(environment, &entries)?;
            *self.class_loader.borrow_mut() = loader;
            self.install_context_class_loader(environment)?;
            Ok::<(), JavaInvocationError>(())
        })?;
        *self.classpath.borrow_mut() = candidate;
        Ok(self.classpath())
    }

    fn ensure_class_loader(
        &self,
        environment: &mut jni::JNIEnv<'_>,
    ) -> Result<(), JavaInvocationError> {
        if self.class_loader.borrow().is_some() {
            return Ok(());
        }
        let entries = loader_entries(&self.classpath.borrow());
        if entries.is_empty() {
            return Ok(());
        }
        *self.class_loader.borrow_mut() = build_class_loader(environment, &entries)?;
        Ok(())
    }

    fn install_context_class_loader(
        &self,
        environment: &mut jni::JNIEnv<'_>,
    ) -> Result<(), JavaInvocationError> {
        let loader = self.class_loader.borrow();
        let Some(loader) = loader.as_ref() else {
            return Ok(());
        };
        let thread = environment
            .call_static_method(
                "java/lang/Thread",
                "currentThread",
                "()Ljava/lang/Thread;",
                &[],
            )
            .and_then(JValueOwned::l)
            .map_err(|error| jni_error(environment, error))?;
        environment
            .call_method(
                thread,
                "setContextClassLoader",
                "(Ljava/lang/ClassLoader;)V",
                &[JValue::Object(loader.as_obj())],
            )
            .map_err(|error| jni_error(environment, error))?;
        Ok(())
    }
}

fn loader_entries(classpath: &SessionClasspath) -> Vec<PathBuf> {
    classpath
        .entries(ClasspathLayer::Project)
        .iter()
        .chain(classpath.entries(ClasspathLayer::Dynamic))
        .cloned()
        .collect()
}

fn existing_classpath_entry(entry: PathBuf) -> Result<PathBuf, JavaInvocationError> {
    if entry.to_string_lossy().contains("://") {
        return Ok(entry);
    }
    if !entry.exists() {
        return Err(JavaInvocationError::UnsupportedValue(format!(
            "Java classpath entry {} does not exist",
            entry.display()
        )));
    }
    entry.canonicalize().map_err(|error| {
        JavaInvocationError::UnsupportedValue(format!(
            "Java classpath entry {} cannot be resolved: {error}",
            entry.display()
        ))
    })
}

fn build_class_loader(
    environment: &mut jni::JNIEnv<'_>,
    entries: &[PathBuf],
) -> Result<Option<GlobalRef>, JavaInvocationError> {
    if entries.is_empty() {
        return Ok(None);
    }
    let length = i32::try_from(entries.len()).map_err(|_| {
        JavaInvocationError::UnsupportedValue("Java classpath has too many entries".into())
    })?;
    let urls = environment
        .new_object_array(length, "java/net/URL", JObject::null())
        .map_err(|error| jni_error(environment, error))?;
    for (index, entry) in entries.iter().enumerate() {
        let entry_text = entry.to_string_lossy();
        let path = JObject::from(
            environment
                .new_string(entry_text.as_ref())
                .map_err(|error| jni_error(environment, error))?,
        );
        let url = if entry_text.contains("://") {
            environment
                .new_object(
                    "java/net/URL",
                    "(Ljava/lang/String;)V",
                    &[JValue::Object(&path)],
                )
                .map_err(|error| jni_error(environment, error))?
        } else {
            let file = environment
                .new_object(
                    "java/io/File",
                    "(Ljava/lang/String;)V",
                    &[JValue::Object(&path)],
                )
                .map_err(|error| jni_error(environment, error))?;
            let uri = environment
                .call_method(&file, "toURI", "()Ljava/net/URI;", &[])
                .and_then(JValueOwned::l)
                .map_err(|error| jni_error(environment, error))?;
            environment
                .call_method(uri, "toURL", "()Ljava/net/URL;", &[])
                .and_then(JValueOwned::l)
                .map_err(|error| jni_error(environment, error))?
        };
        environment
            .set_object_array_element(&urls, index as i32, url)
            .map_err(|error| jni_error(environment, error))?;
    }
    let parent = environment
        .call_static_method(
            "java/lang/ClassLoader",
            "getSystemClassLoader",
            "()Ljava/lang/ClassLoader;",
            &[],
        )
        .and_then(JValueOwned::l)
        .map_err(|error| jni_error(environment, error))?;
    let urls_object = JObject::from(urls);
    let loader = environment
        .new_object(
            "java/net/URLClassLoader",
            "([Ljava/net/URL;Ljava/lang/ClassLoader;)V",
            &[JValue::Object(&urls_object), JValue::Object(&parent)],
        )
        .map_err(|error| jni_error(environment, error))?;
    environment
        .new_global_ref(loader)
        .map(Some)
        .map_err(|error| jni_error(environment, error))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_dynamic_entry_fails_before_classpath_mutation() {
        let missing = PathBuf::from("fixture-does-not-exist.jar");
        assert!(existing_classpath_entry(missing).is_err());
    }
}
