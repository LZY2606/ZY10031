//! Bounded checkpoints which let parsers backtrack over streams which cannot
//! be cloned.
//!
//! [`Stream`] wraps any [`StreamOnce`] + [`Positioned`] stream and records the
//! tokens it yields in a buffer. While at least one checkpoint is active the
//! buffer retains every token since the *oldest* active checkpoint so that the
//! stream can be reset to it. Checkpoints form a stack: a checkpoint created
//! while another one is active is nested inside it and committing the inner
//! checkpoint does **not** commit the outer one. Once the last checkpoint is
//! committed the buffered data is released.
//!
//! Each checkpoint is created with a *token budget*. Resetting to the
//! checkpoint succeeds only if no more than `budget` tokens were read from the
//! underlying stream since the checkpoint was created. If the budget was
//! exceeded the reset fails explicitly and the error at the furthest position
//! is preserved. Committing a checkpoint always succeeds, regardless of how
//! many tokens were read.
//!
//! Checkpoints are usually created through the
//! [`checkpoint`](../parser/combinator/fn.checkpoint.html) parser combinator
//! which commits the checkpoint when its parser succeeds and resets to it when
//! the parser fails without committing (a bounded version of
//! [`attempt`](../parser/combinator/fn.attempt.html) which does not require
//! the stream to be `Clone`). For imperative use,
//! [`Stream::checkpoint_guard`] returns a RAII guard: **dropping the guard
//! without committing it resets the stream to the checkpoint** (if the token
//! budget was exceeded the reset fails and an error is returned by the next
//! call to `uncons`).
//!
//! # Example
//!
//! A small framing parser where an optional new header is read first and, if
//! the input turns out not to be of that version, the stream is reset to the
//! checkpoint and the old format is tried instead. Both headers share the
//! payload parser and the whole input never needs to be cloned.
//!
//! ```
//! # extern crate combine;
//! # fn main() {
//! use combine::{any, token, Parser};
//! use combine::parser::combinator::{attempt, checkpoint};
//! use combine::parser::repeat::count_min_max;
//! use combine::stream::checkpoint::Stream as CheckpointedStream;
//!
//! // A frame starts with a versioned header (`N2` is the new version, `N1`
//! // the old one) followed by a length prefixed payload.
//! let new_header = checkpoint(3, attempt((token(b'N'), token(b'2'), any())))
//!     .map(|(_, _, len)| len);
//! let old_header = (token(b'N'), token(b'1'), any()).map(|(_, _, len)| len);
//! let mut frame = new_header
//!     .or(old_header)
//!     .then_partial(|len: &mut u8| count_min_max(*len as usize, *len as usize, any()));
//!
//! // New style frame
//! let result = frame.parse(CheckpointedStream::new(&b"N2\x03abc"[..]));
//! assert_eq!(result.map(|t| t.0), Ok(b"abc".to_vec()));
//!
//! // Old style frame, parsed after resetting to the checkpoint
//! let result = frame.parse(CheckpointedStream::new(&b"N1\x02hi"[..]));
//! assert_eq!(result.map(|t| t.0), Ok(b"hi".to_vec()));
//! # }
//! ```
//!
//! # Notes
//!
//! * The wrapper is token oriented and does not implement `RangeStreamOnce`,
//!   so parsers which require ranges (such as `string`) cannot be used
//!   directly on it.
//! * Outside of an active checkpoint the wrapper retains its read history so
//!   that combinators such as `attempt` and `or` keep working. This history is
//!   trimmed every time a checkpoint is committed or reset, so parsers which
//!   commit their checkpoints promptly only ever retain a bounded amount of
//!   input.
use alloc::collections::VecDeque;
use alloc::vec::Vec;

use crate::{
    error::{ParseError, StreamError},
    lib::{cmp, fmt, ops},
    stream::{Positioned, ResetStream, StreamErrorFor, StreamOnce},
};

/// A stream which supports bounded, stack ordered checkpoints.
///
/// This is implemented by [`Stream`] and may be implemented for custom
/// streams. The [`checkpoint`](../parser/combinator/fn.checkpoint.html)
/// combinator is generic over this trait.
///
/// Checkpoints form a stack: `commit_checkpoint` and `reset_checkpoint`
/// always apply to the most recently begun checkpoint which is still active.
/// Committing an inner checkpoint does not commit the checkpoints which
/// enclose it.
///
/// ```
/// # extern crate combine;
/// # fn main() {
/// use combine::StreamOnce;
/// use combine::stream::checkpoint::{CheckpointStream, Stream};
///
/// let mut stream = Stream::new(&b"ab"[..]);
/// stream.begin_checkpoint(2);
/// assert_eq!(stream.uncons(), Ok(b'a'));
/// // Still within the token budget, so the reset succeeds
/// stream.reset_checkpoint().unwrap();
/// assert_eq!(stream.uncons(), Ok(b'a'));
/// # }
/// ```
pub trait CheckpointStream: crate::stream::Stream {
    /// Begins a new checkpoint at the current position. At most `budget`
    /// tokens may be read from the underlying stream before a later
    /// `reset_checkpoint` to this checkpoint fails.
    fn begin_checkpoint(&mut self, budget: usize);

    /// Commits the innermost active checkpoint. Any buffered data which is
    /// only needed by checkpoints which are no longer active is released.
    fn commit_checkpoint(&mut self);

    /// Resets the stream to the innermost active checkpoint.
    ///
    /// Fails if more tokens than the checkpoint's budget were read since the
    /// checkpoint was begun. The checkpoint is removed from the stack in
    /// either case.
    fn reset_checkpoint(&mut self) -> Result<(), Self::Error>;
}

#[derive(Debug)]
struct ActiveCheckpoint {
    /// Absolute offset to reset to.
    offset: usize,
    /// Maximum number of tokens which may be read before resetting fails.
    budget: usize,
}

/// A `Stream` wrapper which adds bounded, stack ordered checkpoints to any
/// `StreamOnce + Positioned` stream without requiring the stream to be
/// `Clone`.
///
/// See the [module level documentation](index.html) for more information.
pub struct Stream<Input>
where
    Input: StreamOnce + Positioned,
{
    iter: Input,
    /// Absolute offset of the read head.
    offset: usize,
    /// Absolute offset one past the last token pulled from `iter`.
    buffer_offset: usize,
    /// Tokens pulled from `iter` but not yet past the retention floor.
    buffer: VecDeque<(Input::Token, Input::Position)>,
    /// Stack of active checkpoints, oldest first.
    checkpoints: Vec<ActiveCheckpoint>,
    /// Error produced by a failed reset of a dropped checkpoint guard,
    /// returned by the next `uncons` call.
    deferred_error: Option<StreamErrorFor<Input>>,
}

impl<Input> fmt::Debug for Stream<Input>
where
    Input: StreamOnce + Positioned + fmt::Debug,
    Input::Token: fmt::Debug,
    Input::Position: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Stream")
            .field("iter", &self.iter)
            .field("offset", &self.offset)
            .field("buffer_offset", &self.buffer_offset)
            .field("buffer", &self.buffer)
            .field("checkpoints", &self.checkpoints)
            .finish()
    }
}

impl<Input> Stream<Input>
where
    Input: StreamOnce + Positioned,
{
    /// Constructs a new checkpoint stream wrapping `iter`.
    pub fn new(iter: Input) -> Stream<Input> {
        Stream {
            iter,
            offset: 0,
            buffer_offset: 0,
            buffer: VecDeque::new(),
            checkpoints: Vec::new(),
            deferred_error: None,
        }
    }

    /// Returns a reference to the wrapped stream.
    pub fn get_ref(&self) -> &Input {
        &self.iter
    }

    /// Returns a mutable reference to the wrapped stream.
    ///
    /// It is inadvisable to directly read from the underlying stream.
    pub fn get_mut(&mut self) -> &mut Input {
        &mut self.iter
    }

    /// Consumes this wrapper, returning the wrapped stream.
    pub fn into_inner(self) -> Input {
        self.iter
    }

    /// Returns the number of tokens which are currently retained in the
    /// buffer.
    ///
    /// While checkpoints are active this is bounded by the tokens read since
    /// the oldest active checkpoint. Once all checkpoints are committed only
    /// tokens which have not been consumed yet are retained.
    pub fn buffered_len(&self) -> usize {
        self.buffer.len()
    }

    /// Returns the number of currently active (not yet committed or reset)
    /// checkpoints.
    pub fn active_checkpoints(&self) -> usize {
        self.checkpoints.len()
    }

    /// Begins a checkpoint with the given token budget and returns a guard
    /// for it.
    ///
    /// The guard dereferences to the stream so parsers can be run on it while
    /// the checkpoint is active. Guards nest according to stack order: a
    /// guard created from another guard must be committed, reset or dropped
    /// before the outer guard.
    ///
    /// **Dropping the guard without committing it resets the stream to the
    /// checkpoint.** If more tokens than `budget` were read the reset fails
    /// and an error is returned by the next call to `uncons` instead.
    ///
    /// ```
    /// # extern crate combine;
    /// # fn main() {
    /// use combine::{Parser, StreamOnce};
    /// use combine::parser::byte::byte;
    /// use combine::stream::checkpoint::Stream;
    ///
    /// let mut stream = Stream::new(&b"ab"[..]);
    /// {
    ///     let mut guard = stream.checkpoint_guard(2);
    ///     assert!(byte(b'a').parse_stream(&mut *guard).is_ok());
    ///     // Dropped without `commit`: the stream is reset to the checkpoint
    /// }
    /// assert_eq!(stream.uncons(), Ok(b'a'));
    /// # }
    /// ```
    #[must_use = "an uncommitted checkpoint guard resets the stream when dropped"]
    pub fn checkpoint_guard(&mut self, budget: usize) -> CheckpointGuard<'_, Input> {
        self.begin_checkpoint(budget);
        CheckpointGuard {
            stream: self,
            resolved: false,
        }
    }

    /// Evicts buffered tokens which can no longer be observed: tokens before
    /// the read head which no active checkpoint can reset to.
    fn evict(&mut self) {
        let floor = match self.checkpoints.first() {
            Some(oldest) => cmp::min(oldest.offset, self.offset),
            None => self.offset,
        };
        let buffer_start = self.buffer_offset - self.buffer.len();
        if floor > buffer_start {
            self.buffer
                .drain(..cmp::min(floor - buffer_start, self.buffer.len()));
        }
        if self.checkpoints.is_empty() && self.buffer.is_empty() {
            // Release the buffer capacity promptly once nothing can reference
            // it anymore
            self.buffer = VecDeque::new();
        }
    }
}

impl<Input> CheckpointStream for Stream<Input>
where
    Input: StreamOnce + Positioned,
{
    fn begin_checkpoint(&mut self, budget: usize) {
        self.checkpoints.push(ActiveCheckpoint {
            offset: self.offset,
            budget,
        });
    }

    fn commit_checkpoint(&mut self) {
        if self.checkpoints.pop().is_some() {
            self.evict();
        } else {
            debug_assert!(
                false,
                "commit_checkpoint called without an active checkpoint"
            );
        }
    }

    fn reset_checkpoint(&mut self) -> Result<(), Self::Error> {
        let checkpoint = match self.checkpoints.pop() {
            Some(checkpoint) => checkpoint,
            None => {
                debug_assert!(
                    false,
                    "reset_checkpoint called without an active checkpoint"
                );
                return Err(ParseError::from_error(
                    self.position(),
                    StreamErrorFor::<Self>::message_static_message(
                        "reset_checkpoint called without an active checkpoint",
                    ),
                ));
            }
        };
        if self.buffer_offset - checkpoint.offset > checkpoint.budget {
            self.evict();
            // Report the error at the furthest position reached so that
            // merging it with the parse error keeps the most useful
            // information
            return Err(ParseError::from_error(
                self.iter.position(),
                StreamErrorFor::<Self>::message_static_message("checkpoint token budget exceeded"),
            ));
        }
        self.offset = checkpoint.offset;
        self.evict();
        Ok(())
    }
}

impl<Input> StreamOnce for Stream<Input>
where
    Input: StreamOnce + Positioned,
{
    type Token = Input::Token;
    type Range = Input::Range;
    type Position = Input::Position;
    type Error = Input::Error;

    #[inline]
    fn uncons(&mut self) -> Result<Input::Token, StreamErrorFor<Self>> {
        if let Some(err) = self.deferred_error.take() {
            return Err(err);
        }
        if self.offset >= self.buffer_offset {
            let position = self.iter.position();
            let token = self.iter.uncons()?;
            self.buffer_offset += 1;
            self.offset += 1;
            self.buffer.push_back((token.clone(), position));
            Ok(token)
        } else if self.offset < self.buffer_offset - self.buffer.len() {
            // We have backtracked to far
            Err(StreamError::message_static_message("Backtracked to far"))
        } else {
            let value = self.buffer[self.buffer.len() - (self.buffer_offset - self.offset)]
                .0
                .clone();
            self.offset += 1;
            Ok(value)
        }
    }

    fn is_partial(&self) -> bool {
        self.iter.is_partial()
    }
}

impl<Input> ResetStream for Stream<Input>
where
    Input: StreamOnce + Positioned,
{
    type Checkpoint = usize;

    fn checkpoint(&self) -> Self::Checkpoint {
        self.offset
    }

    fn reset(&mut self, checkpoint: Self::Checkpoint) -> Result<(), Self::Error> {
        if checkpoint == self.offset {
            Ok(())
        } else if checkpoint < self.buffer_offset - self.buffer.len()
            || checkpoint > self.buffer_offset
        {
            // We have backtracked to far
            Err(ParseError::from_error(
                self.position(),
                StreamErrorFor::<Self>::message_static_message("Backtracked to far"),
            ))
        } else {
            self.offset = checkpoint;
            Ok(())
        }
    }
}

impl<Input> Positioned for Stream<Input>
where
    Input: StreamOnce + Positioned,
{
    #[inline]
    fn position(&self) -> Self::Position {
        if self.offset >= self.buffer_offset {
            self.iter.position()
        } else if self.offset < self.buffer_offset - self.buffer.len() {
            self.buffer
                .front()
                .expect("At least 1 element in the buffer")
                .1
                .clone()
        } else {
            self.buffer[self.buffer.len() - (self.buffer_offset - self.offset)]
                .1
                .clone()
        }
    }
}

/// RAII guard for a checkpoint created with [`Stream::checkpoint_guard`].
///
/// While the guard is alive it dereferences to the stream so that parsers can
/// be run on it. **Dropping the guard without calling
/// [`commit`](struct.CheckpointGuard.html#method.commit) resets the stream to
/// the checkpoint** (best effort: if the token budget was exceeded the reset
/// fails and an error is returned by the next call to `uncons`). Use
/// [`reset`](struct.CheckpointGuard.html#method.reset) to handle a failed
/// reset explicitly.
#[must_use = "an uncommitted checkpoint guard resets the stream when dropped"]
pub struct CheckpointGuard<'a, Input>
where
    Input: StreamOnce + Positioned,
{
    stream: &'a mut Stream<Input>,
    resolved: bool,
}

impl<'a, Input> CheckpointGuard<'a, Input>
where
    Input: StreamOnce + Positioned,
{
    /// Commits the checkpoint, consuming the guard. Buffered data which is no
    /// longer needed by any active checkpoint is released.
    pub fn commit(mut self) {
        self.stream.commit_checkpoint();
        self.resolved = true;
    }

    /// Resets the stream to the checkpoint, consuming the guard.
    ///
    /// Fails if more tokens than the checkpoint's budget were read since the
    /// checkpoint was begun.
    pub fn reset(mut self) -> Result<(), <Stream<Input> as StreamOnce>::Error> {
        self.resolved = true;
        self.stream.reset_checkpoint()
    }
}

impl<'a, Input> Drop for CheckpointGuard<'a, Input>
where
    Input: StreamOnce + Positioned,
{
    fn drop(&mut self) {
        if !self.resolved {
            // Dropping an uncommitted guard is equivalent to resetting to the
            // checkpoint. A failed reset (budget exceeded) cannot be reported
            // here so it is deferred to the next `uncons` call.
            if self.stream.reset_checkpoint().is_err() {
                self.stream.deferred_error =
                    Some(StreamErrorFor::<Stream<Input>>::message_static_message(
                        "checkpoint dropped after its token budget was exceeded",
                    ));
            }
        }
    }
}

impl<'a, Input> ops::Deref for CheckpointGuard<'a, Input>
where
    Input: StreamOnce + Positioned,
{
    type Target = Stream<Input>;

    fn deref(&self) -> &Self::Target {
        self.stream
    }
}

impl<'a, Input> ops::DerefMut for CheckpointGuard<'a, Input>
where
    Input: StreamOnce + Positioned,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.stream
    }
}
