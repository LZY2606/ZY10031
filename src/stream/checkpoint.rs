//! A stream wrapper which adds *bounded checkpoints* to any [`StreamOnce`] + [`Positioned`]
//! stream without requiring the stream to be `Clone`.
//!
//! Most of combine's resettable streams (such as `&[u8]` or `&str`) implement [`ResetStream`] by
//! cloning the remaining input, which is cheap for slices. Streams which cannot be cloned (for
//! example readers or asynchronous sources) instead have to buffer any input that a parser may
//! want to rewind to. [`buffered::Stream`] does this with a fixed size ring buffer, which forces
//! the buffering decision to be made up front for the whole stream.
//!
//! [`checkpoint::Stream`] instead buffers *on demand*: while at least one bounded checkpoint is
//! active it retains the tokens read since the oldest active checkpoint and as soon as every
//! checkpoint has been committed the buffer is released again. This makes it possible to backtrack over a small, bounded region — such as
//! the optional header of a frame in a framing protocol — without cloning or buffering the
//! remaining input.
//!
//! [`buffered::Stream`]: ../buffered/struct.Stream.html
//! [`checkpoint::Stream`]: struct.Stream.html
//!
//! # Example
//!
//! ```
//! # extern crate combine;
//! use combine::{Parser, StreamOnce};
//! use combine::parser::byte::byte;
//! use combine::stream::checkpoint;
//!
//! # fn main() {
//! let mut stream = checkpoint::Stream::new(&b"v2 payload"[..]);
//!
//! {
//!     // Create a checkpoint which allows rewinding at most 4 tokens.
//!     let mut checkpoint = stream.checkpoint(4);
//!     let mut header = (byte(b'v'), byte(b'9'));
//!     // Parsing happens *through* the checkpoint.
//!     let result = header.parse_stream(&mut *checkpoint);
//!     // `v` matched but `9` did not.
//!     assert!(result.is_err());
//!     // Dropping the checkpoint without committing resets the stream.
//! }
//!
//! assert_eq!(stream.uncons().ok(), Some(b'v'));
//! # }
//! ```

use alloc::{collections::VecDeque, vec::Vec};

use crate::{
    error::{ParseError, StreamError},
    lib::{
        fmt,
        ops::{Deref, DerefMut},
    },
    stream::{Positioned, ResetStream, StreamErrorFor, StreamOnce},
};

#[derive(Debug, Clone, PartialEq)]
struct Entry {
    /// Absolute token offset at which the checkpoint was created.
    offset: usize,
    /// Maximum number of tokens that may be read while still allowing a reset.
    budget: usize,
}

/// `Stream` wrapper which provides *bounded checkpoints* over any [`StreamOnce`] + [`Positioned`]
/// stream, without requiring the wrapped stream to be `Clone`.
///
/// See the [module level documentation](index.html) for an overview.
///
/// Note that tokens are only buffered while at least one checkpoint is active: everything
/// read since the oldest active checkpoint is retained and the buffer is released as soon as
/// every checkpoint has been committed. The
/// [`ResetStream`] implementation (used by combinators such as [`attempt`] and [`choice`]) can
/// therefore only reset to positions which are still retained by an active checkpoint — outside
/// of a checkpointed region a reset to an earlier position fails with a "Backtracked too far"
/// error, exactly like [`buffered::Stream`] does once its lookahead is exceeded. Parsers which
/// need to backtrack should do so inside a bounded checkpoint (or wrap the stream in
/// [`buffered::Stream`]).
///
/// [`attempt`]: ../../fn.attempt.html
/// [`choice`]: ../../parser/choice/fn.choice.html
/// [`buffered::Stream`]: ../buffered/struct.Stream.html
pub struct Stream<Input>
where
    Input: StreamOnce + Positioned,
{
    inner: Input,
    /// Tokens read from `inner` which a checkpoint (or a not-yet-replayed reset) may still need.
    buffer: VecDeque<(Input::Token, Input::Position)>,
    /// Absolute offset of the next token to yield.
    offset: usize,
    /// Absolute offset one past the last token read from `inner`.
    buffer_offset: usize,
    /// Stack of active checkpoints. Guards and parsers push/pop this in LIFO order.
    checkpoints: Vec<Entry>,
}

impl<Input> fmt::Debug for Stream<Input>
where
    Input: StreamOnce + Positioned + fmt::Debug,
    Input::Token: fmt::Debug,
    Input::Position: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("checkpoint::Stream")
            .field("inner", &self.inner)
            .field("buffer", &self.buffer)
            .field("offset", &self.offset)
            .field("buffer_offset", &self.buffer_offset)
            .field("checkpoints", &self.checkpoints)
            .finish()
    }
}

impl<Input> Stream<Input>
where
    Input: StreamOnce + Positioned,
{
    /// Constructs a new checkpointing stream wrapping `inner`.
    ///
    /// No buffering happens until the first checkpoint is created.
    pub fn new(inner: Input) -> Stream<Input> {
        Stream {
            inner,
            buffer: VecDeque::new(),
            offset: 0,
            buffer_offset: 0,
            checkpoints: Vec::new(),
        }
    }

    /// Returns a reference to the wrapped stream.
    pub fn get_ref(&self) -> &Input {
        &self.inner
    }

    /// Returns a mutable reference to the wrapped stream.
    ///
    /// It is inadvisable to directly read from the underlying stream while checkpoints are
    /// active.
    pub fn get_mut(&mut self) -> &mut Input {
        &mut self.inner
    }

    /// Consumes this wrapper, returning the underlying stream.
    pub fn into_inner(self) -> Input {
        self.inner
    }

    /// Returns the number of checkpoints which are currently active.
    pub fn active_checkpoints(&self) -> usize {
        self.checkpoints.len()
    }

    /// Returns the number of tokens currently retained in the internal buffer.
    ///
    /// The buffer only retains tokens which an active checkpoint (or a recent reset) may still
    /// need, so this is bounded by the number of tokens read since the oldest active
    /// checkpoint.
    #[doc(hidden)]
    pub fn buffered_token_count(&self) -> usize {
        self.buffer.len()
    }

    /// Creates a bounded checkpoint at the current position of the stream.
    ///
    /// While the returned guard is alive the stream retains the tokens read from this position
    /// onwards, up to `budget` tokens. The guard dereferences to the stream itself so parsing
    /// continues through it:
    ///
    /// * [`commit`][`Checkpoint::commit`] consumes the checkpoint, keeping the current position
    ///   and releasing the buffered tokens once no older checkpoint needs them.
    /// * [`reset`][`Checkpoint::reset`] rewinds the stream to the position of the checkpoint.
    ///   If more than `budget` tokens were read since the checkpoint was created the reset
    ///   **fails** and the stream is left at its current position.
    /// * Dropping the guard without committing **resets** the stream, exactly as if
    ///   [`reset`][`Checkpoint::reset`] had been called (a failed reset, due to an exceeded
    ///   budget, is silently ignored and the stream keeps its current position).
    ///
    /// Checkpoints nest and are resolved in stack (LIFO) order: committing an inner checkpoint
    /// never commits an outer one.
    pub fn checkpoint(&mut self, budget: usize) -> Checkpoint<'_, Input> {
        self.checkpoints.push(Entry {
            offset: self.offset,
            budget,
        });
        Checkpoint {
            stream: self,
            active: true,
        }
    }

    /// Removes the innermost checkpoint, keeping the current position.
    fn commit_checkpoint(&mut self) {
        self.checkpoints
            .pop()
            .expect("checkpoint stack corrupted: commit without an active checkpoint");
        self.trim_buffer();
    }

    /// Removes the innermost checkpoint, rewinding the stream to its position.
    ///
    /// Fails if more tokens than the checkpoint's budget were read since it was created.
    fn reset_checkpoint(&mut self) -> Result<(), Input::Error> {
        let entry = self
            .checkpoints
            .pop()
            .expect("checkpoint stack corrupted: reset without an active checkpoint");
        let result = if self.buffer_offset - entry.offset > entry.budget {
            Err(Input::Error::from_error(
                self.position(),
                StreamErrorFor::<Self>::message_static_message(
                    "checkpoint reset failed: token budget exceeded",
                ),
            ))
        } else if entry.offset < self.buffer_offset - self.buffer.len() {
            Err(Input::Error::from_error(
                self.position(),
                StreamErrorFor::<Self>::message_static_message("Backtracked too far"),
            ))
        } else {
            self.offset = entry.offset;
            Ok(())
        };
        self.trim_buffer();
        result
    }

    /// Drops buffered tokens which no active checkpoint and no pending replay can still need.
    ///
    /// Everything since the oldest active checkpoint is retained so that both the checkpoints
    /// themselves and any `ResetStream` checkpoints created by parsers inside the checkpointed
    /// region keep working; the budget only limits whether a checkpoint may *reset*, not what
    /// is buffered.
    fn trim_buffer(&mut self) {
        let keep_from = self
            .checkpoints
            .iter()
            .map(|entry| entry.offset)
            .min()
            // Tokens after `self.offset` may still be replayed after a reset.
            .map_or(self.offset, |offset| offset.min(self.offset));
        let buffer_start = self.buffer_offset - self.buffer.len();
        if keep_from > buffer_start {
            self.buffer.drain(..keep_from - buffer_start);
        }
    }
}

/// A bounded checkpoint created by [`Stream::checkpoint`].
///
/// The guard dereferences to the underlying [`checkpoint::Stream`](struct.Stream.html) so that
/// parsers can run while the checkpoint is active.
///
/// # Drop semantics
///
/// Dropping a `Checkpoint` without calling [`commit`](#method.commit) **resets** the stream to
/// the position of the checkpoint, exactly as if [`reset`](#method.reset) had been called. If
/// the token budget was exceeded the reset fails silently and the stream keeps its current
/// position. Use [`commit`](#method.commit) to keep the current position instead.
///
/// [`reset`](#method.reset) and [`commit`](#method.commit) consume the guard, so the drop
/// behavior only applies to guards which are dropped without either method being called.
pub struct Checkpoint<'a, Input>
where
    Input: StreamOnce + Positioned,
{
    stream: &'a mut Stream<Input>,
    active: bool,
}

impl<'a, Input> Checkpoint<'a, Input>
where
    Input: StreamOnce + Positioned,
{
    /// Returns the budget this checkpoint was created with.
    pub fn budget(&self) -> usize {
        self.stream
            .checkpoints
            .last()
            .map(|entry| entry.budget)
            .unwrap_or(0)
    }

    /// Returns the number of tokens read from the underlying stream since this checkpoint was
    /// created. Once this exceeds [`budget`](#method.budget) the checkpoint can no longer be
    /// reset.
    pub fn tokens_read(&self) -> usize {
        self.stream
            .checkpoints
            .last()
            .map(|entry| self.stream.buffer_offset - entry.offset)
            .unwrap_or(0)
    }

    /// Commits the checkpoint: the current position is kept and the tokens buffered for this
    /// checkpoint are released (unless an older checkpoint still needs them).
    ///
    /// Committing an inner checkpoint never commits an outer one.
    pub fn commit(mut self) {
        self.stream.commit_checkpoint();
        self.active = false;
    }

    /// Resets the stream to the position at which this checkpoint was created.
    ///
    /// # Errors
    ///
    /// Fails if more than [`budget`](#method.budget) tokens were read since the checkpoint was
    /// created. In that case the stream is left at its current position and an error is
    /// returned at that (furthest) position.
    pub fn reset(mut self) -> Result<(), Input::Error> {
        let result = self.stream.reset_checkpoint();
        self.active = false;
        result
    }
}

impl<'a, Input> Deref for Checkpoint<'a, Input>
where
    Input: StreamOnce + Positioned,
{
    type Target = Stream<Input>;

    fn deref(&self) -> &Self::Target {
        self.stream
    }
}

impl<'a, Input> DerefMut for Checkpoint<'a, Input>
where
    Input: StreamOnce + Positioned,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.stream
    }
}

impl<'a, Input> Drop for Checkpoint<'a, Input>
where
    Input: StreamOnce + Positioned,
{
    fn drop(&mut self) {
        if self.active {
            // Dropping an uncommitted checkpoint resets the stream. If the budget was exceeded
            // the reset fails and the stream is left at its current position.
            let _ = self.stream.reset_checkpoint();
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
            self.inner.position()
        } else if self.offset < self.buffer_offset - self.buffer.len() {
            self.buffer
                .front()
                .map(|&(_, ref position)| position.clone())
                .unwrap_or_else(|| self.inner.position())
        } else {
            self.buffer[self.buffer.len() - (self.buffer_offset - self.offset)]
                .1
                .clone()
        }
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
        if checkpoint < self.buffer_offset - self.buffer.len() || checkpoint > self.buffer_offset {
            // The tokens needed to reset to `checkpoint` are no longer (or not yet) buffered.
            Err(Self::Error::from_error(
                self.position(),
                StreamErrorFor::<Self>::message_static_message("Backtracked too far"),
            ))
        } else {
            self.offset = checkpoint;
            Ok(())
        }
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
        if self.offset < self.buffer_offset {
            let buffer_start = self.buffer_offset - self.buffer.len();
            if self.offset < buffer_start {
                // We have backtracked to far
                return Err(StreamError::message_static_message("Backtracked too far"));
            }
            let token = self.buffer[self.offset - buffer_start].0.clone();
            self.offset += 1;
            Ok(token)
        } else {
            if self.checkpoints.is_empty() {
                // No checkpoint can observe the buffered tokens anymore, release them.
                self.buffer.clear();
            }
            let position = self.inner.position();
            let token = self.inner.uncons()?;
            self.buffer_offset += 1;
            self.offset += 1;
            if !self.checkpoints.is_empty() {
                self.buffer.push_back((token.clone(), position));
                self.trim_buffer();
            }
            Ok(token)
        }
    }

    fn is_partial(&self) -> bool {
        self.inner.is_partial()
    }
}
