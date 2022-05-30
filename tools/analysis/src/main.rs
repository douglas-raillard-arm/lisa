// #![feature(future_join)]
#![feature(generators)]
#![feature(stmt_expr_attributes)]
#![feature(proc_macro_hygiene)]
#![feature(type_alias_impl_trait)]
#![feature(generic_associated_types)]

// The musl libc allocator is pretty slow, switching to mimalloc or jemalloc
// makes the resulting binary significantly faster, as we allocate pretty
// heavily when parsing JSON. mimalloc crate compiles much more quickly though.
#[cfg(target_arch = "x86_64")]
use mimalloc::MiMalloc;
#[global_allocator]
#[cfg(target_arch = "x86_64")]
static GLOBAL: MiMalloc = MiMalloc;

use ::futures::{
    future::{join_all, FutureExt},
    pin_mut,
};
use clap::Parser;
use core::{
    fmt::Debug,
    future::Future,
    iter::zip,
    ptr,
    task::{Context, Poll, RawWaker, RawWakerVTable, Waker},
};

use memmap::Mmap;
use serde_json::Deserializer;
use std::{
    collections::BTreeMap,
    fs::File,
    io::{stdin, BufReader, Cursor, Read},
    path::PathBuf,
};

use serde::{Deserialize, Serialize};
use serde_json::value::Value;

mod analysis;
mod event;
mod eventreq;
mod futures;
mod string;

use crate::{
    analysis::{get_analyses, EventWindow, HasKDim, TraceEventStream},
    event::{Event, EventData, EventID},
};

#[derive(clap::Parser, Debug)]
struct Cli {
    #[clap(subcommand)]
    cmd: Subcommands,
}

#[derive(clap::Subcommand, Debug)]
enum Subcommands {
    Run {
        /// The path to the file to read
        #[clap(parse(from_os_str))]
        path: PathBuf,
        analyses: String,
        #[clap(default_value = "\"none\"")]
        window: String,
    },
    List,
}

fn noop(_p: *const ()) {}
fn noop_clone(_p: *const ()) -> RawWaker {
    RawWaker::new(_p, &NOOP_WAKER_VTABLE)
}
const NOOP_WAKER_VTABLE: RawWakerVTable = RawWakerVTable::new(noop_clone, noop, noop, noop);
#[inline]
pub fn noop_waker(data: *const ()) -> Waker {
    let raw = RawWaker::new(data, &NOOP_WAKER_VTABLE);
    unsafe { Waker::from_raw(raw) }
}

pub fn do_run<SelectFn, R: Read, F: Future<Output = T>, T: Debug + Serialize>(
    mut stream: TraceEventStream<SelectFn>,
    fut: F,
    reader: R,
) -> Result<impl Debug + Serialize, String>
where
    SelectFn: FnMut(&Event) -> Option<<Event as HasKDim>::KDim>,
{
    let events = Deserializer::from_reader(reader).into_iter::<Event>();

    pin_mut!(fut);

    let mut errors = vec![];
    let mut res = None;

    let waker = noop_waker(ptr::null());
    let mut ctx = Context::from_waker(&waker);

    for (id, event) in zip(1.., events) {
        let event = match event {
            Ok(Event {
                data: EventData::UnknownEvent,
                ..
            }) => continue,
            Ok(mut event) => {
                event.id = EventID::new(id);
                event
            }
            Err(error) => {
                errors.push((id, error));
                continue;
            }
        };

        stream.set_curr_event(event);
        match fut.as_mut().poll(&mut ctx) {
            Poll::Pending => (),
            Poll::Ready(x) => {
                res = Some(x);
                break;
            }
        }
    }

    if !errors.is_empty() {
        Err(format!("Errors while parsing events: {:?}", errors))
    } else {
        match res {
            None => Err("Coroutine did not finish when asked to".to_string()),
            Some(x) => Ok(x),
        }
    }
}

fn open_path(path: PathBuf) -> Result<Box<dyn Read>, String> {
    if path == PathBuf::from("-") {
        Ok(Box::new(BufReader::new(stdin())))
    } else {
        // Open the file in read-only mode
        let file = File::open(path.clone())
            .map_err(|x| format!("Could not open {}: {}", path.display(), x))?;

        Ok(match unsafe { Mmap::map(&file) } {
            // Divide by 2 the JSON overhead compared to BufReader
            Ok(mmap) => Box::new(Cursor::new(mmap)),
            Err(_) => Box::new(BufReader::with_capacity(8192 * 2, file)),
        })
    }
}

#[derive(Deserialize)]
struct CliAnalysis {
    name: String,
    args: Value,
}

fn run(path: PathBuf, analyses: String, window: String) -> Result<(), String> {
    let reader = open_path(path)?;
    let map = get_analyses();

    let window: EventWindow = serde_json::from_str(&window)
        .map_err(|x| format!("Could not decode window JSON: {}", x))?;

    let anas: Vec<CliAnalysis> =
        serde_json::from_str(&analyses).map_err(|x| format!("Could not analyses JSON: {}", x))?;
    let mut futures = vec![];

    let mut prev_selected = false;
    let stream = TraceEventStream::new(move |event: &Event| match window {
        EventWindow::Time(start, stop) => match event {
            Event {
                data: EventData::EndOfStream,
                ..
            } if prev_selected => Some(event.kdim()),
            _ => {
                let selected = event.ts <= stop && event.ts >= start;
                let is_start = selected && !prev_selected;
                let is_stop = !selected && prev_selected;
                prev_selected = selected;

                if is_start {
                    Some(start)
                } else if is_stop {
                    Some(stop)
                } else {
                    None
                }
            }
        },
        EventWindow::None => match event {
            Event {
                data: EventData::StartOfStream | EventData::EndOfStream,
                ..
            } => Some(event.kdim()),
            _ => None,
        },
    });

    for ana in anas {
        let args = ana.args;
        let ana = map
            .get(&ana.name as &str)
            .ok_or(format!("Analysis does not exist: {}", ana.name))?;
        let fut = (ana.f)(stream.clone(), &args);
        futures.push(fut);
    }

    // TODO: join_all() seems to be faster
    // let fut = JoinedFuture::new(futures);
    let fut = join_all(futures);
    let fut = Box::pin(fut);
    let x = do_run(stream, fut, reader)?;
    let x = serde_json::ser::to_string(&x).map_err(|x| x.to_string())?;
    println!("{}", x);
    Ok(())
}

fn list() -> Result<(), String> {
    let map: BTreeMap<_, _> = get_analyses::<fn(&Event) -> Option<<Event as HasKDim>::KDim>>()
        .into_iter()
        .map(|(name, ana)| (name, BTreeMap::from([("eventreq", ana.eventreq)])))
        .collect();

    let map =
        serde_json::ser::to_string(&map).map_err(|x| format!("could not encode JSON: {:?}", x))?;
    println!("{}", map);
    Ok(())
}

fn main() -> Result<(), String> {
    let args = Cli::parse();
    match args.cmd {
        Subcommands::Run {
            path,
            analyses,
            window,
        } => run(path, analyses, window),
        Subcommands::List => list(),
    }
}

// fn main() {
//     // let events = vec![
//     //     Event {
//     //         id: 0,
//     //         data: EventData::EventFoo(EventFooFields { value: 42 }),
//     //     },
//     //     Event {
//     //         id: 0,
//     //         data: EventData::EventBar(EventBarFields { value: 101 }),
//     //     },
//     //     Event {
//     //         id: 0,
//     //         data: EventData::EventFoo(EventFooFields { value: 43 }),
//     //     },
//     //     Event {
//     //         id: 0,
//     //         data: EventData::EventBar(EventBarFields { value: 102 }),
//     //     },
//     //     Event {
//     //         id: 0,
//     //         data: EventData::EndOfStream { ts: 0 },
//     //     },
//     //     Event {
//     //         id: 0,
//     //         data: EventData::CloseStream,
//     //     },
//     // ];
//     // let events: Vec<EventData> = events.iter().map(|event| event.data).collect();

//     let path = "./trace.json";
//     // Open the file in read-only mode with buffer.
//     let file = File::open(path).unwrap();
//     let reader = BufReader::new(file);

//     // let events: Vec<EventData> = serde_json::from_reader(reader).unwrap();
//     let events = Deserializer::from_reader(reader).into_iter::<EventData>();

//     // for _ in 0..19 {
//     //     events.extend(events.clone());
//     // }

//     for event in events {
//         let event = event.unwrap();
//         println!("{:?}", event);
//     }

//     // println!("size={}", events.len());

//     // println!("{:?}", events);

//     // let s = serde_json::to_string(&events).unwrap();
//     // let events: Vec<EventData> = serde_json::from_str(&s).unwrap();
//     // println!("success {}", events.len());
// }
