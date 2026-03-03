use thiserror::Error;

/// Crate-wide error type for API skeleton operations.
///
/// # Examples
///
/// ```rust
/// use ad_tensors_rs::{Error, Result};
///
/// fn maybe_fail(flag: bool) -> Result<()> {
///     if flag {
///         Ok(())
///     } else {
///         Err(Error::NotImplemented { op: "demo" })
///     }
/// }
///
/// assert!(maybe_fail(true).is_ok());
/// assert!(maybe_fail(false).is_err());
/// ```
#[derive(Debug, Error)]
pub enum Error {
    /// A required thread-local global context is missing.
    #[error("missing global context for type `{type_name}`")]
    MissingGlobalContext { type_name: &'static str },

    /// Stored context could not be downcast to the expected type.
    #[error("global context type mismatch: expected `{expected}`")]
    ContextTypeMismatch { expected: &'static str },

    /// Placeholder for unimplemented POC kernel behavior.
    #[error("`{op}` is not implemented in this POC")]
    NotImplemented { op: &'static str },
}

/// Convenience result alias.
///
/// # Examples
///
/// ```rust
/// use ad_tensors_rs::{Error, Result};
///
/// let ok: Result<i32> = Ok(1);
/// let err: Result<i32> = Err(Error::NotImplemented { op: "sample" });
///
/// assert_eq!(ok.unwrap(), 1);
/// assert!(err.is_err());
/// ```
pub type Result<T> = std::result::Result<T, Error>;
