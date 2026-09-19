//! Tests for bounded checkpoints (`combine::stream::checkpoint` and
//! `combine::parser::checkpoint::checkpoint`).
//!
//! All test names contain `checkpoint` so that `cargo test checkpoint` runs exactly the
//! bounded-checkpoint verification suite.
//!
//! This target uses a custom (minimal) test harness (see `Cargo.toml`) so that
//! `cargo test --quiet checkpoint` still prints the name of every test case it runs;
//! the standard libtest harness hides test names in quiet mode.

use std::{cell::RefCell, collections::VecDeque, io::Cursor, rc::Rc, task::Poll};

use combine::{
    any, attempt, choice,
    error::{ParseError, UnexpectedParse},
    parser::byte::byte,
    parser::checkpoint::checkpoint,
    parser::repeat::{count, count_min_max},
    stream::{
        buf_reader::BufReader, buffered, checkpoint, easy, position, read, Positioned, StreamOnce,
    },
    Parser, Stream,
};
use futures_03_dep as futures;

#[derive(Debug, PartialEq)]
enum Version {
    V1,
    V2,
}

#[derive(Debug, PartialEq)]
struct Frame {
    version: Version,
    payload: Vec<u8>,
}

// Payload parser shared between the old and the new frame format.
fn payload<Input>() -> impl Parser<Input, Output = Vec<u8>>
where
    Input: Stream<Token = u8>,
{
    any().then(|len: u8| count(len as usize, any()))
}

// Framing parser: try the new `V2` header behind a bounded checkpoint, fall back to the old
// `V1` header on mismatch. Both formats share `payload`.
fn frame<Inner>() -> impl Parser<checkpoint::Stream<Inner>, Output = Frame>
where
    Inner: StreamOnce<Token = u8> + Positioned,
{
    choice((
        checkpoint(4, (byte(b'V'), byte(b'2'))).with(payload().map(|payload| Frame {
            version: Version::V2,
            payload,
        })),
        (byte(b'V'), byte(b'1')).with(payload().map(|payload| Frame {
            version: Version::V1,
            payload,
        })),
    ))
}

fn checkpoint_slice_reset_and_commit() {
    // Old format input: the `V2` attempt reads `V`, fails on `1`, resets and the `V1`
    // alternative parses instead.
    let (frame_v1, _) = frame()
        .parse(checkpoint::Stream::new(&b"V1\x03abc"[..]))
        .unwrap();
    assert_eq!(
        frame_v1,
        Frame {
            version: Version::V1,
            payload: b"abc".to_vec()
        }
    );

    // New format input: the checkpoint commits once the `V2` header matched.
    let (frame_v2, _) = frame()
        .parse(checkpoint::Stream::new(&b"V2\x03abc"[..]))
        .unwrap();
    assert_eq!(
        frame_v2,
        Frame {
            version: Version::V2,
            payload: b"abc".to_vec()
        }
    );
}

fn checkpoint_slice_budget_exceeded_stays_committed() {
    // The new header is 4 tokens but the budget only covers 2: after reading past the budget
    // the reset fails and the error is committed, so the infallible fallback is *not* tried.
    let mut parser = choice((
        checkpoint(2, (byte(b'V'), byte(b'2'), byte(b'X'), byte(b'Y'))).map(|_| "new"),
        any().map(|_| "fallback"),
    ));
    let result = parser.parse(checkpoint::Stream::new(&b"V2XZ"[..]));
    assert!(result.is_err());

    // With a budget which covers the whole header the reset succeeds and the fallback runs.
    let mut parser = choice((
        checkpoint(4, (byte(b'V'), byte(b'2'), byte(b'X'), byte(b'Y'))).map(|_| "new"),
        any().map(|_| "fallback"),
    ));
    let (output, _) = parser.parse(checkpoint::Stream::new(&b"V2XZ"[..])).unwrap();
    assert_eq!(output, "fallback");
}

fn checkpoint_easy_errors_keep_furthest_position() {
    let input = easy::Stream(position::Stream::new(&b"V2XZ"[..]));
    let mut parser = checkpoint(2, (byte(b'V'), byte(b'2'), byte(b'X'), byte(b'Y')));
    let err = parser.parse(checkpoint::Stream::new(input)).unwrap_err();
    // The reset failed after 3 tokens (> budget 2): the error stays at the furthest
    // position and mentions both the parse failure and the exceeded budget.
    assert_eq!(err.position, 3);
    let message = format!("{:?}", err);
    assert!(
        message.contains("budget exceeded"),
        "unexpected error: {}",
        message
    );
    assert!(
        message.contains("Unexpected"),
        "unexpected error: {}",
        message
    );
}

fn checkpoint_buffered_reader() {
    // Two frames read from a `BufReader` wrapped `std::io::Read` source.
    let data: &[u8] = b"V1\x03abcV2\x02de";
    let reader = BufReader::new(Cursor::new(data));
    let stream = checkpoint::Stream::new(position::Stream::new(read::Stream::new(reader)));
    let mut stream = stream;

    let first = frame().parse_stream(&mut stream);
    let (first, _) = first.into_result().ok().unwrap();
    assert_eq!(
        first,
        Frame {
            version: Version::V1,
            payload: b"abc".to_vec()
        }
    );

    let second = frame().parse_stream(&mut stream);
    let (second, _) = second.into_result().ok().unwrap();
    assert_eq!(
        second,
        Frame {
            version: Version::V2,
            payload: b"de".to_vec()
        }
    );
}

fn checkpoint_over_buffered_stream() {
    // `buffered::Stream` (fixed lookahead ring buffer) as the underlying stream.
    let data: &[u8] = b"V2\x03abc";
    let inner = buffered::Stream::new(
        position::Stream::new(read::Stream::new(Cursor::new(data))),
        2,
    );
    let (frame, _) = frame().parse(checkpoint::Stream::new(inner)).unwrap();
    assert_eq!(
        frame,
        Frame {
            version: Version::V2,
            payload: b"abc".to_vec()
        }
    );
}

fn checkpoint_nested_stack_order() {
    let mut stream = checkpoint::Stream::new(&b"abcdef"[..]);
    {
        let mut outer = stream.checkpoint(4);
        assert_eq!(outer.uncons().ok(), Some(b'a'));
        {
            let mut inner = outer.checkpoint(2);
            assert_eq!(inner.uncons().ok(), Some(b'b'));
            // Committing the inner checkpoint must not commit the outer one.
            inner.commit();
        }
        assert_eq!(outer.active_checkpoints(), 1);
        outer.reset().unwrap();
    }
    assert_eq!(stream.uncons().ok(), Some(b'a'));

    // The reverse order: inner reset, outer commit.
    {
        let mut outer = stream.checkpoint(4);
        assert_eq!(outer.uncons().ok(), Some(b'b'));
        {
            let inner = outer.checkpoint(2);
            inner.reset().unwrap();
        }
        outer.commit();
        assert_eq!(stream.active_checkpoints(), 0);
    }
    assert_eq!(stream.uncons().ok(), Some(b'c'));
}

fn checkpoint_drop_without_commit_resets() {
    let mut stream = checkpoint::Stream::new(&b"abcdef"[..]);
    {
        let mut checkpoint = stream.checkpoint(4);
        assert_eq!(checkpoint.uncons().ok(), Some(b'a'));
        assert_eq!(checkpoint.uncons().ok(), Some(b'b'));
        // Dropping the checkpoint without committing resets the stream.
    }
    assert_eq!(stream.active_checkpoints(), 0);
    assert_eq!(stream.uncons().ok(), Some(b'a'));
}

fn checkpoint_drop_after_budget_exceeded_keeps_position() {
    let mut stream = checkpoint::Stream::new(&b"abcdef"[..]);
    {
        let mut checkpoint = stream.checkpoint(1);
        assert_eq!(checkpoint.uncons().ok(), Some(b'a'));
        assert_eq!(checkpoint.uncons().ok(), Some(b'b'));
        // 2 tokens > budget 1: dropping attempts a reset which fails silently.
    }
    assert_eq!(stream.uncons().ok(), Some(b'c'));
}

fn checkpoint_guard_reset_and_commit() {
    let mut stream = checkpoint::Stream::new(&b"abcdef"[..]);
    {
        let mut checkpoint = stream.checkpoint(4);
        assert_eq!(checkpoint.uncons().ok(), Some(b'a'));
        assert_eq!(checkpoint.uncons().ok(), Some(b'b'));
        assert_eq!(checkpoint.tokens_read(), 2);
        assert_eq!(checkpoint.budget(), 4);
        assert_eq!(checkpoint.buffered_token_count(), 2);
        checkpoint.reset().unwrap();
    }
    assert_eq!(stream.uncons().ok(), Some(b'a'));

    {
        let mut checkpoint = stream.checkpoint(4);
        assert_eq!(checkpoint.uncons().ok(), Some(b'b'));
        checkpoint.commit();
    }
    // Everything is committed: the buffer is released.
    assert_eq!(stream.active_checkpoints(), 0);
    assert_eq!(stream.buffered_token_count(), 0);
    assert_eq!(stream.uncons().ok(), Some(b'c'));
}

fn checkpoint_guard_reset_failure_is_reported() {
    let mut stream = checkpoint::Stream::new(&b"abcdef"[..]);
    let mut checkpoint = stream.checkpoint(1);
    assert_eq!(checkpoint.uncons().ok(), Some(b'a'));
    assert_eq!(checkpoint.uncons().ok(), Some(b'b'));
    // More tokens than the budget were read: reset must fail explicitly.
    assert!(checkpoint.reset().is_err());
    assert_eq!(stream.uncons().ok(), Some(b'c'));
}

fn checkpoint_buffer_releases_oldest_on_commit() {
    let mut stream = checkpoint::Stream::new(&b"abcdef"[..]);
    {
        let mut outer = stream.checkpoint(4);
        outer.uncons().unwrap();
        {
            let mut inner = outer.checkpoint(2);
            inner.uncons().unwrap();
            inner.uncons().unwrap();
            assert_eq!(inner.buffered_token_count(), 3);
            inner.commit();
        }
        // The inner checkpoint is committed but the outer one still pins the buffer.
        assert_eq!(outer.buffered_token_count(), 3);
        outer.commit();
    }
    assert_eq!(stream.buffered_token_count(), 0);
}

fn checkpoint_attempt_and_choice_keep_their_semantics() {
    // `attempt` and `choice` inside a checkpointed region backtrack as usual, using the
    // tokens retained by the checkpoint.
    let mut parser = checkpoint(
        4,
        choice((
            attempt((byte(b'a'), byte(b'b'))).map(|_| "ab"),
            (byte(b'a'), byte(b'c')).map(|_| "ac"),
        )),
    );
    let (output, _) = parser.parse(checkpoint::Stream::new(&b"ac"[..])).unwrap();
    assert_eq!(output, "ac");

    // A committed failure inside the checkpoint still propagates as committed once the
    // budget is exceeded, so `choice` does not try further alternatives.
    let mut parser = choice((
        checkpoint(
            1,
            choice((
                attempt((byte(b'a'), byte(b'b'), byte(b'c'))).map(|_| "abc"),
                (byte(b'a'), byte(b'x')).map(|_| "ax"),
            )),
        ),
        any().map(|_| "fallback"),
    ));
    let result = parser.parse(checkpoint::Stream::new(&b"aax"[..]));
    assert!(result.is_err());
}

fn checkpoint_memory_stays_bounded_on_long_streams() {
    // 20_000 frames, each parsed behind a checkpoint: the buffer must never retain more
    // than the checkpoint budget and must be empty after every frame.
    let frame_count = 20_000;
    let mut input = Vec::with_capacity(frame_count * 6);
    for _ in 0..frame_count {
        input.extend_from_slice(b"V1\x03abc");
    }
    let mut stream = checkpoint::Stream::new(&input[..]);
    let mut frames = 0;
    while frames < frame_count {
        let (frame, _) = frame()
            .parse_stream(&mut stream)
            .into_result()
            .ok()
            .unwrap();
        assert_eq!(
            frame,
            Frame {
                version: Version::V1,
                payload: b"abc".to_vec()
            }
        );
        frames += 1;
        // All checkpoints are committed between frames: nothing may be retained.
        assert_eq!(stream.active_checkpoints(), 0);
        assert_eq!(stream.buffered_token_count(), 0);
    }
    assert_eq!(frames, frame_count);
}

// A `StreamOnce` which only has a prefix of the input available at any time, like an
// asynchronous source which has not delivered all of its chunks yet. It is neither `Clone`
// nor `ResetStream`: all rewinding happens through the checkpoint buffer.
struct PendingBytes {
    available: Rc<RefCell<VecDeque<u8>>>,
    position: usize,
}

impl StreamOnce for PendingBytes {
    type Token = u8;
    type Range = u8;
    type Position = usize;
    type Error = UnexpectedParse;

    fn uncons(&mut self) -> Result<u8, combine::stream::StreamErrorFor<Self>> {
        match self.available.borrow_mut().pop_front() {
            Some(byte) => {
                self.position += 1;
                Ok(byte)
            }
            None => Err(UnexpectedParse::Eoi),
        }
    }

    fn is_partial(&self) -> bool {
        true
    }
}

impl Positioned for PendingBytes {
    fn position(&self) -> usize {
        self.position
    }
}

// Yields `Poll::Pending` once before completing, simulating an asynchronous wakeup.
async fn cross_pending() {
    let mut pending = true;
    futures::future::poll_fn(|cx| {
        if pending {
            pending = false;
            cx.waker().wake_by_ref();
            Poll::Pending
        } else {
            Poll::Ready(())
        }
    })
    .await
}

fn checkpoint_async_pending_keeps_budget_and_positions() {
    futures::executor::block_on(async {
        // The header is parsed behind a checkpoint whose budget is *exactly* the header
        // size (2 tokens). The input arrives one byte per chunk so the parser suspends
        // (Pending) in the middle of the checkpoint; re-read tokens must not be counted
        // twice against the budget.
        let chunks: &[&[u8]] = &[b"V", b"1", b"a", b"b", b"c"];
        let available = Rc::new(RefCell::new(VecDeque::new()));
        let inner = PendingBytes {
            available: available.clone(),
            position: 0,
        };
        let mut stream = checkpoint::Stream::new(easy::Stream(inner));
        // Fixed size payload so the parser can suspend and resume mid-frame.
        let mut parser = choice((
            checkpoint(2, (byte(b'V'), byte(b'2'))).with(count_min_max(3, 3, any()).map(
                |payload: Vec<u8>| Frame {
                    version: Version::V2,
                    payload,
                },
            )),
            (byte(b'V'), byte(b'1')).with(count_min_max(3, 3, any()).map(|payload: Vec<u8>| {
                Frame {
                    version: Version::V1,
                    payload,
                }
            })),
        ));
        let mut state = Default::default();

        let mut chunks = chunks.iter().peekable();
        available
            .borrow_mut()
            .extend(chunks.next().unwrap().iter().copied());
        let frame = loop {
            match parser.parse_with_state(&mut stream, &mut state) {
                Ok(frame) => break frame,
                Err(err) => {
                    assert!(
                        err.is_unexpected_end_of_input(),
                        "unexpected parse error: {}",
                        err
                    );
                    // Wait for more input (crossing an actual `Poll::Pending`).
                    cross_pending().await;
                    match chunks.next() {
                        Some(chunk) => available.borrow_mut().extend(chunk.iter().copied()),
                        None => panic!("parser asked for more input after the last chunk"),
                    }
                }
            }
        };
        assert_eq!(
            frame,
            Frame {
                version: Version::V1,
                payload: b"abc".to_vec()
            }
        );
        assert_eq!(stream.active_checkpoints(), 0);
        assert_eq!(stream.buffered_token_count(), 0);
    });
}

fn checkpoint_async_pending_error_positions() {
    futures::executor::block_on(async {
        // `V3` is neither the new nor the old version: both alternatives fail at offset 1
        // and the merged error must point there even though the offending byte only
        // arrived after a Pending.
        let chunks: &[&[u8]] = &[b"V", b"3"];
        let available = Rc::new(RefCell::new(VecDeque::new()));
        let inner = PendingBytes {
            available: available.clone(),
            position: 0,
        };
        let mut stream = checkpoint::Stream::new(easy::Stream(inner));
        let mut parser = choice((
            checkpoint(2, (byte(b'V'), byte(b'2'))).with(count_min_max(3, 3, any()).map(
                |payload: Vec<u8>| Frame {
                    version: Version::V2,
                    payload,
                },
            )),
            (byte(b'V'), byte(b'1')).with(count_min_max(3, 3, any()).map(|payload: Vec<u8>| {
                Frame {
                    version: Version::V1,
                    payload,
                }
            })),
        ));
        let mut state = Default::default();

        let mut chunks = chunks.iter().peekable();
        available
            .borrow_mut()
            .extend(chunks.next().unwrap().iter().copied());
        let err = loop {
            match parser.parse_with_state(&mut stream, &mut state) {
                Ok(frame) => panic!("unexpectedly parsed a frame: {:?}", frame),
                Err(err) => {
                    if err.is_unexpected_end_of_input() {
                        cross_pending().await;
                        match chunks.next() {
                            Some(chunk) => available.borrow_mut().extend(chunk.iter().copied()),
                            None => break err,
                        }
                    } else {
                        break err;
                    }
                }
            }
        };
        assert_eq!(err.position, 1);
    });
}

fn main() {
    let tests: &[(&str, fn())] = &[
        (
            "checkpoint_slice_reset_and_commit",
            checkpoint_slice_reset_and_commit,
        ),
        (
            "checkpoint_slice_budget_exceeded_stays_committed",
            checkpoint_slice_budget_exceeded_stays_committed,
        ),
        (
            "checkpoint_easy_errors_keep_furthest_position",
            checkpoint_easy_errors_keep_furthest_position,
        ),
        ("checkpoint_buffered_reader", checkpoint_buffered_reader),
        (
            "checkpoint_over_buffered_stream",
            checkpoint_over_buffered_stream,
        ),
        (
            "checkpoint_nested_stack_order",
            checkpoint_nested_stack_order,
        ),
        (
            "checkpoint_drop_without_commit_resets",
            checkpoint_drop_without_commit_resets,
        ),
        (
            "checkpoint_drop_after_budget_exceeded_keeps_position",
            checkpoint_drop_after_budget_exceeded_keeps_position,
        ),
        (
            "checkpoint_guard_reset_and_commit",
            checkpoint_guard_reset_and_commit,
        ),
        (
            "checkpoint_guard_reset_failure_is_reported",
            checkpoint_guard_reset_failure_is_reported,
        ),
        (
            "checkpoint_buffer_releases_oldest_on_commit",
            checkpoint_buffer_releases_oldest_on_commit,
        ),
        (
            "checkpoint_attempt_and_choice_keep_their_semantics",
            checkpoint_attempt_and_choice_keep_their_semantics,
        ),
        (
            "checkpoint_memory_stays_bounded_on_long_streams",
            checkpoint_memory_stays_bounded_on_long_streams,
        ),
        (
            "checkpoint_async_pending_keeps_budget_and_positions",
            checkpoint_async_pending_keeps_budget_and_positions,
        ),
        (
            "checkpoint_async_pending_error_positions",
            checkpoint_async_pending_error_positions,
        ),
    ];

    // Positional arguments act as substring filters (like `cargo test <filter>`), arguments
    // starting with `-` (such as `--quiet`) are ignored.
    let filter = std::env::args().skip(1).find(|arg| !arg.starts_with('-'));

    let mut filtered_out = 0;
    let mut failed = 0;
    let mut ran = 0;
    for &(name, test) in tests {
        if let Some(filter) = filter.as_deref() {
            if !name.contains(filter) {
                filtered_out += 1;
                continue;
            }
        }
        ran += 1;
        match std::panic::catch_unwind(test) {
            Ok(()) => println!("test {} ... ok", name),
            Err(_) => {
                println!("test {} ... FAILED", name);
                failed += 1;
            }
        }
    }
    println!(
        "test result: {}. {} passed; {} failed; {} filtered out",
        if failed == 0 { "ok" } else { "FAILED" },
        ran - failed,
        failed,
        filtered_out
    );
    if failed != 0 {
        std::process::exit(1);
    }
}
