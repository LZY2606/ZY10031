//! Tests for bounded checkpoints (`stream::checkpoint` and
//! `parser::combinator::checkpoint`).
#![cfg(feature = "std")]

use std::collections::VecDeque;

use combine::{
    any,
    error::{ParseError, UnexpectedParse},
    parser::{
        byte::byte,
        combinator::{attempt, checkpoint},
        repeat::count_min_max,
    },
    stream::{
        buffered,
        checkpoint::{CheckpointStream, Stream as CheckpointedStream},
        easy, position, read, Positioned, StreamOnce,
    },
    token, Parser,
};

/// The framing parser used in the documentation: a frame starts with a
/// versioned header (`N2` new, `N1` old) followed by a length prefixed
/// payload. Both versions share the payload parser and the new header is
/// probed speculatively behind a bounded checkpoint.
fn frame_parser<Input>() -> impl Parser<Input, Output = Vec<u8>>
where
    Input: CheckpointStream<Token = u8>,
{
    let new_header =
        checkpoint(3, attempt((token(b'N'), token(b'2'), any()))).map(|(_, _, len)| len);
    let old_header = (token(b'N'), token(b'1'), any()).map(|(_, _, len)| len);
    new_header
        .or(old_header)
        .then_partial(|len: &mut u8| count_min_max(*len as usize, *len as usize, any()))
}

#[test]
fn checkpoint_framing_new_and_old_headers_share_payload() {
    let mut frame = frame_parser();
    let result = frame.parse(CheckpointedStream::new(&b"N2\x03abc"[..]));
    assert_eq!(result.map(|t| t.0), Ok(b"abc".to_vec()));

    let result = frame.parse(CheckpointedStream::new(&b"N1\x02hi"[..]));
    assert_eq!(result.map(|t| t.0), Ok(b"hi".to_vec()));
}

#[test]
fn slice_checkpoint_reset_and_commit() {
    let mut parser = checkpoint(2, attempt((byte(b'a'), byte(b'b')))).or((byte(b'a'), byte(b'c')));

    // Committed: the first alternative matches and the checkpoint is committed
    let result = parser.parse(CheckpointedStream::new(&b"ab"[..]));
    assert_eq!(result.map(|t| t.0), Ok((b'a', b'b')));

    // Reset within the token budget: the second alternative is tried from the
    // checkpoint
    let result = parser.parse(CheckpointedStream::new(&b"ac"[..]));
    assert_eq!(result.map(|t| t.0), Ok((b'a', b'c')));
}

#[test]
fn slice_checkpoint_budget_exceeded_is_committed() {
    // The attempted parser reads 2 tokens before failing which is more than
    // the budget of 1, so the reset fails and no alternative is tried
    let mut parser = checkpoint(1, attempt((byte(b'a'), byte(b'b')))).or((byte(b'a'), byte(b'c')));
    let result = parser.parse(CheckpointedStream::new(&b"ac"[..]));
    assert!(result.is_err());
}

#[test]
fn easy_checkpoint_budget_exceeded_keeps_furthest_error() {
    // The attempted parser reads 4 tokens before failing, exceeding the
    // budget of 2
    let mut parser = checkpoint(
        2,
        attempt((byte(b'A'), byte(b'B'), byte(b'C'), byte(b'D'))).map(|_| "new"),
    )
    .or((byte(b'A'), byte(b'B'), byte(b'C')).map(|_| "old"));

    let input = &b"ABCX"[..];
    let stream = CheckpointedStream::new(easy::Stream(position::Stream::new(input)));
    let err = parser.parse(stream).map(|t| t.0).unwrap_err();

    // The reset failed explicitly and the error is reported at the furthest
    // position which was reached (4 tokens were read)
    assert_eq!(err.position, 4);
    assert!(
        err.errors.iter().any(|e| matches!(
            e,
            easy::Error::Message(easy::Info::Static(msg))
                if *msg == "checkpoint token budget exceeded"
        )),
        "{:?}",
        err
    );
}

#[test]
fn buffered_reader_checkpoint() {
    // A `Read` based stream cannot be cloned, the checkpoint stream provides
    // backtracking for it regardless
    let input = b"N1\x02hi";
    let stream = CheckpointedStream::new(buffered::Stream::new(
        position::Stream::new(read::Stream::new(&input[..])),
        1,
    ));
    let mut frame = frame_parser();
    assert_eq!(frame.parse(stream).map(|t| t.0).ok(), Some(b"hi".to_vec()));
}

/// A growable stream which behaves like a asynchronous stream: it returns
/// `end_of_input` while it is partial (more input may arrive) which is how
/// combine models `Poll::Pending`.
struct PendingStream {
    buffer: VecDeque<u8>,
    position: usize,
    eof: bool,
}

impl PendingStream {
    fn new() -> Self {
        PendingStream {
            buffer: VecDeque::new(),
            position: 0,
            eof: false,
        }
    }

    fn feed(&mut self, bytes: &[u8]) {
        self.buffer.extend(bytes);
    }

    fn finish(&mut self) {
        self.eof = true;
    }
}

impl StreamOnce for PendingStream {
    type Token = u8;
    type Range = &'static [u8];
    type Position = usize;
    type Error = UnexpectedParse;

    fn uncons(&mut self) -> Result<u8, UnexpectedParse> {
        match self.buffer.pop_front() {
            Some(b) => {
                self.position += 1;
                Ok(b)
            }
            None => Err(UnexpectedParse::Eoi),
        }
    }

    fn is_partial(&self) -> bool {
        !self.eof
    }
}

impl Positioned for PendingStream {
    fn position(&self) -> usize {
        self.position
    }
}

#[test]
fn async_checkpoint_survives_pending() {
    let mut stream = CheckpointedStream::new(easy::Stream(PendingStream::new()));
    // Only the first byte of the header is available
    stream.get_mut().0.feed(b"N");

    let mut frame = frame_parser();
    let mut state = Default::default();

    // The parse cannot complete yet, it is `Pending` until more input arrives
    let err = frame.parse_with_state(&mut stream, &mut state).unwrap_err();
    assert!(err.is_unexpected_end_of_input());
    // The checkpoint is still active and retains the token read so far
    assert_eq!(stream.active_checkpoints(), 1);
    assert_eq!(stream.buffered_len(), 1);

    // The rest of the frame arrives
    stream.get_mut().0.feed(b"2\x03abc");
    stream.get_mut().0.finish();

    let payload = frame.parse_with_state(&mut stream, &mut state).unwrap();
    assert_eq!(payload, b"abc");
    assert_eq!(stream.active_checkpoints(), 0);
}

#[test]
fn async_checkpoint_reset_across_pending() {
    let mut stream = CheckpointedStream::new(easy::Stream(PendingStream::new()));
    // Only the first byte of the header is available, the parser suspends in
    // the middle of the speculative new header parse
    stream.get_mut().0.feed(b"N");

    // Speculatively parse the new frame format behind a checkpoint guard
    let mut new_frame = checkpoint(3, attempt((token(b'N'), token(b'2'), any())))
        .map(|(_, _, len)| len)
        .then_partial(|len: &mut u8| {
            count_min_max::<Vec<u8>, _, _>(*len as usize, *len as usize, any())
        });
    let mut state = Default::default();

    {
        let mut guard = stream.checkpoint_guard(3);
        // Suspends while waiting for more input: `Pending`
        let err = new_frame
            .parse_with_state(&mut *guard, &mut state)
            .unwrap_err();
        assert!(err.is_unexpected_end_of_input());

        // The frame turns out to use the old header, so the checkpoint
        // (created before the suspension) is reset across the `Pending`
        // boundary
        guard.get_mut().0.feed(b"1\x02hi");
        guard.get_mut().0.finish();

        let err = new_frame
            .parse_with_state(&mut *guard, &mut state)
            .unwrap_err();
        assert!(!err.is_unexpected_end_of_input());
        guard.reset().unwrap();
    }

    // The old format can now be parsed from the reset position
    let mut old_frame = (token(b'N'), token(b'1'), any())
        .map(|(_, _, len)| len)
        .then_partial(|len: &mut u8| {
            count_min_max::<Vec<u8>, _, _>(*len as usize, *len as usize, any())
        });
    let payload = old_frame
        .parse_with_state(&mut stream, &mut Default::default())
        .unwrap();
    assert_eq!(payload, b"hi");
}

#[test]
fn async_checkpoint_error_position_across_pending() {
    let mut stream = CheckpointedStream::new(easy::Stream(PendingStream::new()));
    stream.get_mut().0.feed(b"N2\x05ab");

    let mut frame = frame_parser();
    let mut state = Default::default();

    // The header promises 5 payload bytes but only 2 are available
    let err = frame.parse_with_state(&mut stream, &mut state).unwrap_err();
    assert!(err.is_unexpected_end_of_input());

    // One more byte arrives and then the stream ends unexpectedly
    stream.get_mut().0.feed(b"c");
    stream.get_mut().0.finish();

    let err = frame.parse_with_state(&mut stream, &mut state).unwrap_err();
    // All 6 bytes were consumed so the error is reported at position 6, the
    // suspension did not disturb the positions
    assert_eq!(err.position, 6);
}

#[test]
fn nested_checkpoints_follow_stack_order() {
    // The inner checkpoint commits `AB` successfully but the outer parser
    // fails afterwards, so the outer checkpoint must still be active and
    // reset the stream all the way to the start
    let inner = checkpoint(5, attempt((byte(b'A'), byte(b'B'))));
    let mut parser = checkpoint(10, attempt((inner, byte(b'X'))).map(|_| "inner")).or((
        byte(b'A'),
        byte(b'B'),
        byte(b'C'),
    )
        .map(|_| "fallback"));

    let result = parser.parse(CheckpointedStream::new(&b"ABC"[..]));
    assert_eq!(result.map(|t| t.0), Ok("fallback"));
}

#[test]
fn nested_checkpoint_guards_follow_stack_order() {
    let mut stream = CheckpointedStream::new(&b"abcd"[..]);
    let mut outer = stream.checkpoint_guard(10);
    assert_eq!(outer.uncons(), Ok(b'a'));
    {
        let mut inner = outer.checkpoint_guard(5);
        assert_eq!(inner.uncons(), Ok(b'b'));
        inner.commit();
    }
    // Committing the inner guard must not commit the outer one
    assert_eq!(outer.active_checkpoints(), 1);
    outer.reset().unwrap();
    assert_eq!(stream.uncons(), Ok(b'a'));
}

#[test]
fn checkpoint_guard_drop_resets_stream() {
    let mut stream = CheckpointedStream::new(&b"ab"[..]);
    {
        let mut guard = stream.checkpoint_guard(2);
        assert_eq!(guard.uncons(), Ok(b'a'));
        // Dropped without `commit`: equivalent to `reset`
    }
    assert_eq!(stream.active_checkpoints(), 0);
    assert_eq!(stream.uncons(), Ok(b'a'));
}

#[test]
fn checkpoint_guard_drop_after_budget_exceeded_defers_error() {
    let mut stream = CheckpointedStream::new(&b"abc"[..]);
    {
        let mut guard = stream.checkpoint_guard(1);
        assert_eq!(guard.uncons(), Ok(b'a'));
        assert_eq!(guard.uncons(), Ok(b'b'));
        // Dropped after reading more tokens than the budget allows: the reset
        // fails and the error is reported by the next `uncons` call
    }
    assert_eq!(stream.uncons(), Err(UnexpectedParse::Unexpected));
}

#[test]
fn checkpoint_guard_commit_keeps_position() {
    let mut stream = CheckpointedStream::new(&b"ab"[..]);
    {
        let mut guard = stream.checkpoint_guard(2);
        assert_eq!(guard.uncons(), Ok(b'a'));
        guard.commit();
    }
    assert_eq!(stream.active_checkpoints(), 0);
    // Committed, so the stream continues after the tokens read in the guard
    assert_eq!(stream.uncons(), Ok(b'b'));
}

#[test]
fn checkpoint_memory_bound_long_stream() {
    const FRAMES: usize = 25_000;
    let mut input = Vec::with_capacity(FRAMES * 4);
    for i in 0..FRAMES as u32 {
        input.push(0xFF);
        input.extend_from_slice(&i.to_le_bytes()[..3]);
    }

    let mut stream = CheckpointedStream::new(&input[..]);
    let mut frame = (byte(0xFF), count_min_max::<Vec<u8>, _, _>(3, 3, any()));
    let mut parsed = 0;
    loop {
        let mut guard = stream.checkpoint_guard(4);
        if frame.parse_stream(&mut *guard).is_ok() {
            guard.commit();
            parsed += 1;
            // Committing the checkpoint releases the buffered frame, the
            // whole input is never retained
            assert_eq!(stream.buffered_len(), 0);
        } else {
            break;
        }
    }
    assert_eq!(parsed, FRAMES);
    assert_eq!(stream.buffered_len(), 0);
    assert_eq!(stream.active_checkpoints(), 0);
}

#[test]
fn checkpoint_preserves_attempt_choice_and_commit_semantics() {
    // A committed failure inside a checkpoint stays committed: `or` does not
    // try the alternative, exactly as if the checkpoint were not present
    let mut parser = checkpoint(8, (byte(b'a'), byte(b'b'))).or((byte(b'a'), byte(b'c')));
    let result = parser.parse(CheckpointedStream::new(&b"ac"[..]));
    assert!(result.is_err());

    // `attempt` inside the checkpoint downgrades the committed failure so
    // the alternative is tried after the reset
    let mut parser = checkpoint(8, attempt((byte(b'a'), byte(b'b')))).or((byte(b'a'), byte(b'c')));
    let result = parser.parse(CheckpointedStream::new(&b"ac"[..]));
    assert_eq!(result.map(|t| t.0), Ok((b'a', b'c')));
}
