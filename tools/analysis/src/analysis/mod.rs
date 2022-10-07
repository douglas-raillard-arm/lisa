use core::{fmt::Debug, future::Future, pin::Pin};
use std::{
    cell::RefCell,
    collections::{btree_map::Entry, BTreeMap},
    rc::Rc,
};

use futures::{
    future::FutureExt,
    stream::Stream,
    task::{Context, Poll},
    StreamExt,
};

use erased_serde::serialize_trait_object;
use futures_async_stream::stream;
use pin_project::pin_project;
use schemars::{schema::RootSchema, schema_for, JsonSchema};
use serde::{Deserialize, Serialize, Serializer};
use serde_json::value::Value;

use crate::{
    event::{Event, EventData, EventID, Timestamp},
    eventreq::EventReq,
};

macro_rules! make_row_struct {
    ($(#[$attr:meta])* struct $name:ident { $($field_name:ident : $field_type:ty),+ $(,)?} ) => {
        $(#[$attr])* struct $name { $($field_name: $field_type),+ }

        impl crate::analysis::Row for $name {
            type AsTuple = ($($field_type),+);

            fn columns() -> ::std::vec::Vec<&'static str> {
                vec![$(stringify!($field_name)),+]
            }
        }

        impl From<<$name as crate::analysis::Row>::AsTuple> for $name {
            fn from(x: <$name as crate::analysis::Row>::AsTuple) -> $name {
                let ($($field_name),+) = x;
                $name{ $($field_name),+ }
            }
        }

        impl From<$name> for <$name as crate::analysis::Row>::AsTuple {
            fn from(x: $name) -> <$name as crate::analysis::Row>::AsTuple {
                ($(x.$field_name),+)
            }
        }
    }
}

mod test;

#[derive(Clone)]
pub struct RawEventStream {
    last_seen: EventID,
    closed: bool,
    pub curr_event: Rc<RefCell<Option<Event>>>,
}

impl RawEventStream {
    pub fn new() -> RawEventStream {
        RawEventStream {
            last_seen: EventID::new(0),
            closed: false,
            // Encapsulate the current event into an Rc<Refcell<>> so that all
            // the RawEventStream derived from this one will share the same
            // "slot".
            curr_event: Rc::new(RefCell::new(None)),
        }
    }
}

impl Stream for RawEventStream {
    type Item = Event;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.closed {
            Poll::Ready(None)
        } else {
            let mut closed = false;
            let mut last_seen = self.last_seen;
            let res = match &*self.curr_event.borrow() {
                None => Poll::Pending,
                Some(event) => {
                    last_seen = event.id;
                    if self.last_seen == event.id {
                        Poll::Pending
                    } else {
                        closed = match event.data {
                            EventData::EndOfStream => true,
                            _ => false,
                        };
                        Poll::Ready(self.curr_event.borrow().clone())
                    }
                }
            };
            self.last_seen = last_seen;
            self.closed = closed;
            // Make sure to call the waker once, so that the future will be
            // rescheduled for polling next time an event is available. Otherwise,
            // combinators like futures::future::join_all() will simply not attempt
            // to poll, thinking that the Future will not make progress.
            cx.waker().wake_by_ref();
            res
        }
    }
}

#[derive(Debug)]
pub enum SignalValue<K, V, KDim> {
    Initial(KDim, K, V),
    Current(KDim, K, V),
    Final(KDim, K, V),
}

pub enum SignalUpdate<K, V, UpdateFn = fn(Option<&V>) -> Option<V>> {
    Update(K, V),
    UpdateWith(K, UpdateFn),
    Finished(K),
}

pub trait Splitter<'a, I, K, V, UpdateFn> {
    type SplitStream: Stream<Item = SignalUpdate<K, V, UpdateFn>> + 'a;
    fn split(self: &Self, x: &'a I) -> Self::SplitStream;
}

impl<'a, F, I, R, K, V, UpdateFn> Splitter<'a, I, K, V, UpdateFn> for F
where
    (F, I, R, K, V, UpdateFn): 'a,
    F: Fn(&'a I) -> R,
    R: Stream<Item = SignalUpdate<K, V, UpdateFn>> + 'a,
{
    type SplitStream = impl Stream<Item = SignalUpdate<K, V, UpdateFn>>;
    fn split(&self, x: &'a I) -> Self::SplitStream {
        self(x)
    }
}

pub trait MultiplexedStream: Stream
where
    Self::Item: HasKDim,
{
    // TODO: Using GAT makes the trait not object-safe. Are we cool with that ?
    // Otherwise we will need to revert back to a Box::pin()

    // SplitFn needs to be a parameter since the actual DemuxStream type will
    // embed a value of type SplitFn in implementations.
    type DemuxStream<K, V, SplitFn, UpdateFn>: Stream<
        Item = SignalValue<K, V, <Self::Item as HasKDim>::KDim>,
    >;
    fn demux<K, V, SplitFn, UpdateFn>(
        self,
        split: SplitFn,
    ) -> Self::DemuxStream<K, V, SplitFn, UpdateFn>
    where
        K: Ord + Eq + Clone,
        V: Clone,
        UpdateFn: FnOnce(Option<&V>) -> Option<V>,
        SplitFn: for<'a> Splitter<'a, Self::Item, K, V, UpdateFn>;
}

pub trait EventStream: MultiplexedStream<Item = Event> {
    fn fork(&self) -> Self;
}

#[pin_project]
#[derive(Clone)]
pub struct TraceEventStream<SelectFn> {
    #[pin]
    stream: RawEventStream,
    last_selected: bool,
    select: SelectFn,
}

impl<SelectFn> TraceEventStream<SelectFn> {
    pub fn new(select: SelectFn) -> Self {
        TraceEventStream {
            stream: RawEventStream::new(),
            last_selected: false,
            select,
        }
    }

    pub fn set_curr_event(&mut self, event: Event) {
        self.stream.curr_event.replace(Some(event));
    }
}

impl<SelectFn> Stream for TraceEventStream<SelectFn>
where
    SelectFn: FnMut(&Event) -> Option<<Event as HasKDim>::KDim>,
{
    type Item = Event;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();
        loop {
            let polled = this.stream.as_mut().poll_next(cx);
            return match &polled {
                Poll::Ready(Some(x)) => {
                    let last_selected = *this.last_selected;
                    let selected = match (this.select)(x) {
                        Some(_) => !last_selected,
                        None => last_selected,
                    };
                    *this.last_selected = selected;

                    if selected {
                        polled
                    } else {
                        continue;
                    }
                }
                _ => polled,
            };
        }
    }
}

impl<SelectFn> MultiplexedStream for TraceEventStream<SelectFn>
where
    SelectFn: FnMut(&Event) -> Option<<Event as HasKDim>::KDim>,
{
    type DemuxStream<K, V, SplitFn, UpdateFn> =
        impl Stream<Item = SignalValue<K, V, <Self::Item as HasKDim>::KDim>>;

    fn demux<K, V, SplitFn, UpdateFn>(
        self,
        split: SplitFn,
    ) -> Self::DemuxStream<K, V, SplitFn, UpdateFn>
    where
        K: Ord + Eq + Clone,
        V: Clone,
        UpdateFn: FnOnce(Option<&V>) -> Option<V>,
        SplitFn: for<'a> Splitter<'a, Self::Item, K, V, UpdateFn>,
    {
        demux(self.stream, self.select, split)
    }
}

impl<SelectFn> EventStream for TraceEventStream<SelectFn>
where
    SelectFn: Clone + FnMut(&Event) -> Option<<Event as HasKDim>::KDim>,
{
    fn fork(&self) -> Self {
        self.clone()
    }
}

pub trait HasKDim {
    type KDim: Clone;
    fn kdim(&self) -> Self::KDim;
}

impl HasKDim for Event {
    type KDim = Timestamp;
    fn kdim(&self) -> Self::KDim {
        self.ts
    }
}

#[stream(item = SignalValue<K, V, <S::Item as HasKDim>::KDim>)]
async fn demux<S, K, V, SplitFn, SelectFn, UpdateFn>(
    stream: S,
    mut select: SelectFn,
    splitter: SplitFn,
) where
    K: Ord + Eq + Clone,
    V: Clone,
    S: Stream,
    S::Item: HasKDim,
    SelectFn: FnMut(&S::Item) -> Option<<S::Item as HasKDim>::KDim>,
    SplitFn: for<'a> Splitter<'a, S::Item, K, V, UpdateFn>,
    UpdateFn: FnOnce(Option<&V>) -> Option<V>,
{
    let mut map: BTreeMap<K, V> = BTreeMap::new();
    let mut prev_selected = false;

    #[for_await]
    for x in stream {
        let selected = match select(&x) {
            Some(kdim) => {
                // stop
                if prev_selected {
                    // We don't care about the order, as signals are assumed to be
                    // independent
                    for (k, v) in map.iter() {
                        yield SignalValue::Final(kdim.clone(), k.clone(), v.clone());
                    }
                // start
                } else {
                    for (k, v) in map.iter() {
                        yield SignalValue::Initial(kdim.clone(), k.clone(), v.clone());
                    }
                }
                !prev_selected
            }
            None => prev_selected,
        };

        #[for_await]
        for update in splitter.split(&x) {
            match update {
                SignalUpdate::Update(k, v) => {
                    let initial = map.insert(k.clone(), v.clone()).is_none();
                    if selected {
                        let kdim = x.kdim();
                        if initial {
                            yield SignalValue::Initial(kdim.clone(), k.clone(), v.clone())
                        }
                        // Emit a Current value no matter what, as this allows easy
                        // filtering on the client side for "real" data: Initial
                        // values are not "real" and can be ignored e.g. to count
                        // the number of actual frequency updates within the window.
                        yield SignalValue::Current(kdim, k, v)
                    }
                }
                SignalUpdate::UpdateWith(k, f) => {
                    let entry = map.entry(k.clone());
                    match entry {
                        Entry::Occupied(mut entry) => {
                            let v = f(Some(entry.get()));
                            let kdim = x.kdim();
                            match v {
                                Some(v) => {
                                    entry.insert(v.clone());
                                    if selected {
                                        yield SignalValue::Current(kdim, k, v);
                                    }
                                }
                                None => {
                                    let v = entry.remove();
                                    if selected {
                                        yield SignalValue::Final(kdim, k, v);
                                    }
                                }
                            }
                        }
                        Entry::Vacant(entry) => {
                            let v = f(None);
                            match v {
                                Some(v) => {
                                    entry.insert(v.clone());
                                    if selected {
                                        let kdim = x.kdim();
                                        yield SignalValue::Initial(
                                            kdim.clone(),
                                            k.clone(),
                                            v.clone(),
                                        );
                                        yield SignalValue::Current(kdim, k, v);
                                    }
                                }
                                None => {}
                            }
                        }
                    };
                }
                SignalUpdate::Finished(k) => {
                    // We expect a Finished update to come after an Update update,
                    // so since it's the only place where keys are removed, the key
                    // should be present. If it's not present, we simply ignore it
                    // as it indicates a double-Finished.
                    match map.remove(&k) {
                        Some(removed) if selected => yield SignalValue::Final(x.kdim(), k, removed),
                        _ => (),
                    }
                }
            }
        }

        prev_selected = selected;
    }
}

// BothUpdateFn and BothStream are small abomination that are only there to
// work-around the unavailibility of impl Trait in return type of closures:
// https://github.com/rust-lang/rust/issues/62223.
// The only other alternative is to use a Box, with the dynamic allocation cost
// paid every time we yield to generate an UpdateFn closure.
// Sadly, this also forces to copy-paste the entire set of constraints on types
type BothUpdateFn<I, K, Splitter1, V1, UpdateFn1, Splitter2, V2, UpdateFn2>
where
    K: Clone + 'static,
    V1: Clone + 'static,
    V2: Clone + 'static,
    Splitter1: for<'a> Splitter<'a, I, K, V1, UpdateFn1>,
    Splitter2: for<'a> Splitter<'a, I, K, V2, UpdateFn2>,
    UpdateFn1: FnOnce(Option<&V1>) -> Option<V1> + 'static,
    UpdateFn2: FnOnce(Option<&V2>) -> Option<V2> + 'static,
= impl FnOnce(Option<&(Option<V1>, Option<V2>)>) -> Option<(Option<V1>, Option<V2>)>;

type BothStream<'a, I, K, Splitter1, V1, UpdateFn1, Splitter2, V2, UpdateFn2>
where
    I: 'a,
    K: Clone + 'static,
    V1: Clone + 'static,
    V2: Clone + 'static,
    Splitter1: for<'b> Splitter<'b, I, K, V1, UpdateFn1> + 'a,
    Splitter2: for<'b> Splitter<'b, I, K, V2, UpdateFn2> + 'a,
    UpdateFn1: FnOnce(Option<&V1>) -> Option<V1> + 'static,
    UpdateFn2: FnOnce(Option<&V2>) -> Option<V2> + 'static,
= impl Stream<
        Item = SignalUpdate<
            K,
            (Option<V1>, Option<V2>),
            BothUpdateFn<I, K, Splitter1, V1, UpdateFn1, Splitter2, V2, UpdateFn2>,
        >,
    > + 'a;

pub fn both<I, K, Splitter1, V1, UpdateFn1, Splitter2, V2, UpdateFn2>(
    // splitters need to be 'static since they will be held onto by the Stream,
    // which itself can live for an arbitrarily decided 'c. That means it can be
    // asked to live up to 'static duration.
    splitter1: &'static Splitter1,
    splitter2: &'static Splitter2,
) -> impl for<'a> Fn(&'a I) -> BothStream<'a, I, K, Splitter1, V1, UpdateFn1, Splitter2, V2, UpdateFn2>
where
    K: Clone + 'static,
    V1: Clone + 'static,
    V2: Clone + 'static,
    Splitter1: for<'a> Splitter<'a, I, K, V1, UpdateFn1>,
    Splitter2: for<'a> Splitter<'a, I, K, V2, UpdateFn2>,
    UpdateFn1: FnOnce(Option<&V1>) -> Option<V1> + 'static,
    UpdateFn2: FnOnce(Option<&V2>) -> Option<V2> + 'static,
{
    move |x: &I| {
        #[stream]
        async move {
            enum StreamID<T1, T2> {
                First(T1),
                Second(T2),
            }
            enum Update<V, F> {
                Value(V),
                Func(F),
                Finished,
            }
            #[for_await]
            for update in splitter1
                .split(x)
                .map(StreamID::First)
                .chain(splitter2.split(x).map(StreamID::Second))
            {
                let (k, update) = match update {
                    StreamID::First(update) => {
                        let (k, v) = match update {
                            SignalUpdate::Update(k, v) => (k, Update::Value(v)),
                            SignalUpdate::UpdateWith(k, f) => (k, Update::Func(f)),
                            SignalUpdate::Finished(k) => (k, Update::Finished),
                        };
                        (k, StreamID::First(v))
                    }
                    StreamID::Second(update) => {
                        let (k, v) = match update {
                            SignalUpdate::Update(k, v) => (k, Update::Value(v)),
                            SignalUpdate::UpdateWith(k, f) => (k, Update::Func(f)),
                            SignalUpdate::Finished(k) => (k, Update::Finished),
                        };
                        (k, StreamID::Second(v))
                    }
                };

                yield SignalUpdate::UpdateWith(k, |v1v2: Option<&(Option<V1>, Option<V2>)>| {
                    let (v1, v2) = match v1v2 {
                        None => (&None, &None),
                        Some((v1, v2)) => (v1, v2),
                    };
                    let (v1, v2) = match update {
                        StreamID::First(update) => {
                            let v2 = v2.clone();
                            match update {
                                Update::Value(v) => (Some(v), v2),
                                Update::Func(f) => (f(v1.as_ref()), v2),
                                Update::Finished => (None, v2),
                            }
                        }
                        StreamID::Second(update) => {
                            let v1 = v1.clone();
                            match update {
                                Update::Value(v) => (v1, Some(v)),
                                Update::Func(f) => (v1, f(v2.as_ref())),
                                Update::Finished => (v1, None),
                            }
                        }
                    };
                    match (v1, v2) {
                        (None, None) => None,
                        x => Some(x),
                    }
                })
            }
        }
    }
}

type ErrorMsg = std::string::String;

#[derive(Clone, Debug, Serialize, PartialEq, Eq, JsonSchema)]
pub enum AnalysisError {
    #[serde(rename = "input")]
    Input(ErrorMsg),
    #[serde(rename = "serialization")]
    Serialization(ErrorMsg),
}

// Allow serializing any Box<dyn AnalysisValue>, which is not possible with
// serde alone since Serialize is not object-safe.
pub trait AnalysisValue: Debug + erased_serde::Serialize {
    fn get_schema(&self) -> RootSchema;
}
serialize_trait_object!(AnalysisValue);
impl<T: Serialize + Debug + JsonSchema> AnalysisValue for T {
    fn get_schema(&self) -> RootSchema {
        schema_for!(T)
    }
}

fn serialize_ok<S: Serializer>(x: &Box<dyn AnalysisValue>, s: S) -> Result<S::Ok, S::Error> {
    #[derive(Serialize)]
    struct WithSchema<T> {
        schema: RootSchema,
        value: T,
    }
    WithSchema {
        schema: x.get_schema(),
        value: x,
    }
    .serialize(s)
}

#[derive(Debug, Serialize)]
pub enum AnalysisResult {
    #[serde(rename = "err")]
    Err(AnalysisError),
    #[serde(rename = "ok")]
    #[serde(serialize_with = "serialize_ok")]
    Ok(Box<dyn AnalysisValue>),
}

#[derive(Debug, Serialize, JsonSchema)]
struct Table<Item> {
    columns: Vec<&'static str>,
    data: Vec<Item>,
}

pub trait Row {
    type AsTuple;
    fn columns() -> Vec<&'static str>;
}

impl AnalysisResult {
    pub fn new<T: AnalysisValue + JsonSchema + 'static>(x: T) -> Self {
        AnalysisResult::Ok(Box::new(x))
    }

    pub async fn from_row_stream<S, Item>(stream: S) -> Self
    where
        S: Stream<Item = Item>,
        Item: Row + Into<<Item as Row>::AsTuple>,
        <Item as Row>::AsTuple: Serialize + JsonSchema + Debug + 'static,
    {
        // Convert the records to tuples for efficient JSON encoding.
        let data: Vec<Item::AsTuple> = stream.map(|x| x.into()).collect().await;
        let columns = Item::columns();
        let data = Table { data, columns };
        Self::new(data)
    }
}

pub struct Analysis<S: EventStream> {
    pub name: &'static str,
    pub eventreq: EventReq,
    pub f: Box<
        dyn Fn(S, &Value) -> Pin<Box<dyn Future<Output = AnalysisResult> + 'static>> + Send + Sync,
    >,
}

impl<S> Analysis<S>
where
    S: EventStream + 'static,
{
    pub fn new<Fut, P>(name: &'static str, f: fn(S, P) -> Fut, eventreq: EventReq) -> Analysis<S>
    where
        Fut: Future<Output = AnalysisResult> + 'static,
        P: for<'de> Deserialize<'de> + 'static,
        S: EventStream,
    {
        Analysis {
            name,
            f: Box::new(move |stream, x| {
                Box::pin(match serde_json::value::from_value(x.clone()) {
                    Ok(x) => f(stream, x).right_future(),
                    Err(error) => {
                        async move { AnalysisResult::Err(AnalysisError::Input(error.to_string())) }
                            .left_future()
                    }
                })
            }),
            eventreq,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum EventWindow {
    #[serde(rename = "none")]
    None,
    #[serde(rename = "time")]
    Time(Timestamp, Timestamp),
}

pub trait HasEventReq {
    fn eventreq() -> EventReq;
}

#[macro_export]
macro_rules! analysis {
    (name: $name:ident, events: $events:tt, ($stream:ident: EventStream, $param:ident: $param_ty:ty) $body:block) => {
        pub async fn $name<S: EventStream>($stream: S, $param: $param_ty) -> AnalysisResult {
            $body
        }

        #[allow(non_camel_case_types)]
        pub struct $name {}

        impl $crate::analysis::HasEventReq for $name {
            fn eventreq() -> $crate::eventreq::EventReq {
                $crate::event_req!($events)
            }
        }
    };
}

macro_rules! build_analyses_descriptors {
    ($($path:path),* $(,)?) => {
        [
            $(
                Analysis::new(
                    stringify!($path),
                    $path,
                    <$path as HasEventReq>::eventreq(),
                ),
            )*
        ]
    };
}

pub fn get_analyses<SelectFn>() -> BTreeMap<&'static str, Analysis<TraceEventStream<SelectFn>>>
where
    SelectFn: Clone + FnMut(&Event) -> Option<<Event as HasKDim>::KDim> + 'static,
{
    // The content of this file is a single call to
    // build_analyses_descriptors!() containing the reference to all the
    // analyses in the crate.
    let analyses = include!(concat!(env!("OUT_DIR"), "/analyses_list.rs"));

    let mut map = BTreeMap::new();
    analyses.map(|ana| map.insert(ana.name, ana));
    map
}
