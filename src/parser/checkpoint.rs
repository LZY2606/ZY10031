//! Bounded checkpoint combinator: rewindable parsing over streams which cannot be cloned.
//!
//! [`checkpoint`] wraps a parser so that, while it runs, the tokens it reads are buffered by a
//! [`checkpoint::Stream`]. If the parser fails the stream is rewound to the position at which
//! the checkpoint was created — as long as no more than `budget` tokens were read — and the
//! failure is reported as uncommitted, exactly as if [`attempt`] had been used. This lets a
//! caller speculatively parse (for example) the optional header of a frame and fall back to an
//! older format without cloning the remaining input.
//!
//! Unlike [`attempt`] the rewind is *bounded*: if more than `budget` tokens were read before
//! the failure, the reset fails and the error is instead reported as **committed**, preserving
//! the furthest error (a "token budget exceeded" error is merged in at the furthest position).
//! This bounds the memory used for buffering to `budget` tokens per active checkpoint.
//!
//! [`checkpoint::Stream`]: ../../stream/checkpoint/struct.Stream.html
//! [`attempt`]: ../../fn.attempt.html
//!
//! # Example: a small framing parser
//!
//! A framing protocol with an optional versioned header. The parser first tries the new `V2`
//! header behind a checkpoint; if the tag does not match, the stream is rewound and the old
//! `V1` header is tried instead. Both formats share the same payload parser.
//!
//! ```
//! # extern crate combine;
//! use combine::{any, Parser, Stream};
//! use combine::parser::byte::byte;
//! use combine::parser::checkpoint::checkpoint;
//! use combine::parser::choice::choice;
//! use combine::parser::repeat::count;
//! use combine::stream::checkpoint::Stream as CheckpointStream;
//!
//! #[derive(Debug, PartialEq)]
//! enum Version { V1, V2 }
//!
//! #[derive(Debug, PartialEq)]
//! struct Frame { version: Version, payload: Vec<u8> }
//!
//! // The payload parser is shared between the old and the new frame format.
//! fn payload<Input>() -> impl Parser<Input, Output = Vec<u8>>
//! where
//!     Input: Stream<Token = u8>,
//! {
//!     any().then(|len: u8| count(len as usize, any()))
//! }
//!
//! # fn main() {
//! let mut frame = choice((
//!     // Try the new `V2` header, rewinding at most 4 tokens if it does not match.
//!     checkpoint(4, (byte(b'V'), byte(b'2')))
//!         .with(payload().map(|payload| Frame { version: Version::V2, payload })),
//!     // Fall back to the old `V1` header.
//!     (byte(b'V'), byte(b'1'))
//!         .with(payload().map(|payload| Frame { version: Version::V1, payload })),
//! ));
//!
//! // Old format input: the `V2` attempt fails and is rewound, `V1` succeeds.
//! let (frame_v1, _) = frame.parse(CheckpointStream::new(&b"V1\x03abc"[..])).unwrap();
//! assert_eq!(frame_v1, Frame { version: Version::V1, payload: b"abc".to_vec() });
//!
//! // New format input: the checkpoint commits after the `V2` header matches.
//! let (frame_v2, _) = frame.parse(CheckpointStream::new(&b"V2\x03abc"[..])).unwrap();
//! assert_eq!(frame_v2, Frame { version: Version::V2, payload: b"abc".to_vec() });
//! # }
//! ```

use crate::{
    error::{
        ParseError,
        ParseResult::{self, *},
        StreamError,
    },
    parser::ParseMode,
    stream::{checkpoint::Stream as CheckpointStream, StreamErrorFor, StreamOnce},
    Parser,
};

// Error added to the furthest parse error when a checkpoint cannot be reset because more
// than `budget` tokens were read.
const BUDGET_EXCEEDED_MESSAGE: &str = "checkpoint reset failed: token budget exceeded";

fn budget_exceeded<Input>(mut error: Input::Error) -> Input::Error
where
    Input: StreamOnce + crate::stream::Positioned,
{
    error.add(StreamErrorFor::<Input>::message_static_message(
        BUDGET_EXCEEDED_MESSAGE,
    ));
    error
}

/// Parser returned by [`checkpoint`].
#[derive(Copy, Clone)]
pub struct Checkpoint<P>(usize, P);

impl<Input, P> Parser<CheckpointStream<Input>> for Checkpoint<P>
where
    Input: StreamOnce + crate::stream::Positioned,
    P: Parser<CheckpointStream<Input>>,
{
    type Output = P::Output;
    type PartialState = P::PartialState;

    #[inline]
    fn parse_stream(
        &mut self,
        input: &mut CheckpointStream<Input>,
    ) -> ParseResult<P::Output, <CheckpointStream<Input> as StreamOnce>::Error> {
        self.parse_lazy(input)
    }

    parse_mode!(CheckpointStream<Input>);

    #[inline]
    fn parse_committed_mode<M>(
        &mut self,
        mode: M,
        input: &mut CheckpointStream<Input>,
        state: &mut Self::PartialState,
    ) -> ParseResult<Self::Output, <CheckpointStream<Input> as StreamOnce>::Error>
    where
        M: ParseMode,
    {
        self.parse_mode(mode, input, state)
    }

    #[inline]
    fn parse_mode_impl<M>(
        &mut self,
        mode: M,
        input: &mut CheckpointStream<Input>,
        state: &mut Self::PartialState,
    ) -> ParseResult<Self::Output, <CheckpointStream<Input> as StreamOnce>::Error>
    where
        M: ParseMode,
    {
        let is_partial = input.is_partial();
        let mut checkpoint = input.checkpoint(self.0);
        match self.1.parse_committed_mode(mode, &mut *checkpoint, state) {
            CommitOk(x) => {
                checkpoint.commit();
                CommitOk(x)
            }
            PeekOk(x) => {
                checkpoint.commit();
                PeekOk(x)
            }
            PeekErr(err) => match checkpoint.reset() {
                Ok(()) => PeekErr(err),
                // The budget was exceeded: the failure stays committed and the furthest
                // error is preserved, with the reset failure noted in it.
                Err(_) => CommitErr(budget_exceeded::<CheckpointStream<Input>>(err.error)),
            },
            CommitErr(err) => {
                if is_partial && err.is_unexpected_end_of_input() {
                    // The inner parser needs more input to decide. Rewind to the checkpoint so
                    // that none of the checkpointed input is committed; the next call (with
                    // more input available) re-parses from the checkpoint with a fresh state.
                    match checkpoint.reset() {
                        Ok(()) => {
                            *state = Default::default();
                            CommitErr(err)
                        }
                        Err(_) => CommitErr(budget_exceeded::<CheckpointStream<Input>>(err)),
                    }
                } else {
                    match checkpoint.reset() {
                        // Behave as `attempt`: the failure did not commit any input.
                        Ok(()) => PeekErr(err.into()),
                        // The budget was exceeded: the failure stays committed and the
                        // furthest error is preserved, with the reset failure noted in it.
                        Err(_) => CommitErr(budget_exceeded::<CheckpointStream<Input>>(err)),
                    }
                }
            }
        }
    }

    forward_parser!(CheckpointStream<Input>, add_error add_committed_expected_error parser_count, 1);
}

/// Creates a bounded checkpoint around `parser` on a [`checkpoint::Stream`].
///
/// If `parser` fails after reading at most `budget` tokens, the stream is rewound to the
/// position at which the checkpoint was created and the failure is reported as uncommitted (as
/// if [`attempt`] had been used), allowing [`choice`] to try another alternative. If `parser`
/// succeeds the checkpoint is committed and the buffered tokens are released.
///
/// If more than `budget` tokens were read before the failure the reset fails: the failure is
/// then reported as **committed** and the furthest error is preserved (merged with a "token
/// budget exceeded" error). This guarantees that at most `budget` tokens per active checkpoint
/// are ever buffered.
///
/// On partial streams (see [`PartialStream`]) a checkpoint which runs out of input rewinds to
/// the checkpoint position before signaling that more input is needed, so no checkpointed
/// input is committed across an asynchronous `Pending` and the token budget as well as error
/// positions stay accurate when parsing resumes.
///
/// [`checkpoint::Stream`]: ../../stream/checkpoint/struct.Stream.html
/// [`attempt`]: ../../fn.attempt.html
/// [`choice`]: ../../parser/choice/fn.choice.html
/// [`PartialStream`]: ../../stream/struct.PartialStream.html
///
/// # Examples
///
/// See the [module level documentation](index.html) for a full framing parser example.
pub fn checkpoint<Input, P>(budget: usize, parser: P) -> Checkpoint<P>
where
    Input: StreamOnce + crate::stream::Positioned,
    P: Parser<CheckpointStream<Input>>,
{
    Checkpoint(budget, parser)
}
