use core::fmt::Debug;

use futures::{stream::Stream, StreamExt};
use futures_async_stream::{for_await, stream};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    analysis::{both, AnalysisResult, EventStream, SignalUpdate, SignalValue},
    event::{Comm, Event, EventData, Freq, Timestamp, CPU, PID},
};

use crate::analysis;
pub(crate) use make_row_struct;

#[stream(item = SignalUpdate<CPU, Freq>)]
async fn cpufreq(x: &Event) {
    match &x.data {
        EventData::EventCPUFrequency(fields) => yield SignalUpdate::Update(fields.cpu, fields.freq),
        _ => (), //yield SignalUpdate::Ignore,
    }
}

async fn sum<S: EventStream>(stream: S) -> u64 {
    let mut acc: Freq = Freq::new(0);

    #[for_await]
    // for x in stream.demux(cpufreq) {
    //     match x {
    //         SignalValue::Initial(_, cpu, (ts, freq)) => {
    //             eprintln!("initial value: CPU {cpu:?} @ {ts:?} = {freq:?}")
    //         }
    //         SignalValue::Current(_cpu, (_ts, freq)) => acc += freq as u64,
    //         SignalValue::Final(_, cpu, (ts, freq)) => {
    //             eprintln!("last value: CPU {cpu:?} @ {ts:?} = {freq:?}")
    //         }
    //     }
    // }
    for x in stream.demux(both(&cpufreq, &cpufreq)) {
        match x {
            SignalValue::Initial(ts, cpu, (Some(freq), _)) => {
                eprintln!("initial value: CPU {cpu:?} @ {ts:?} = {freq:?}")
            }
            SignalValue::Current(_ts, _cpu, (Some(freq), _)) => acc = acc + freq,
            SignalValue::Final(ts, cpu, (Some(freq), _)) => {
                eprintln!("last value: CPU {cpu:?} @ {ts:?} = {freq:?}")
            }
            _ => eprintln!("other"),
        }
    }
    let acc: u32 = acc.into();
    acc.into()
}

#[stream(item = T::Item)]
async fn replicate<T>(n: u32, stream: T)
where
    T: Stream,
    T::Item: Clone,
{
    #[for_await]
    for x in stream {
        for _ in 0..n {
            yield x.clone();
        }
    }
}

async fn count<T: Stream>(stream: T) -> u64
where
    T::Item: Debug,
{
    let mut acc = 0;
    #[for_await]
    for _event in stream {
        // println!("{:?}", event);
        acc += 1;
    }
    acc
}

#[stream(item = SignalUpdate<PID, Comm>)]
async fn pid_comm(x: &Event) {
    match &x.data {
        EventData::EventTaskRename(fields) => {
            yield SignalUpdate::Update(fields.pid, fields.newcomm.clone())
        }
        _ => {}
    }
}

trait TaskID: Clone {
    fn new(pid: &PID, comm: &Comm) -> Self;
    fn pid(self: &Self) -> Option<PID>;
    fn comm(self: &Self) -> Option<Comm>;

    const STORES_PID: bool;
    const STORES_COMM: bool;
}

impl TaskID for (PID, Comm) {
    fn new(pid: &PID, comm: &Comm) -> Self {
        (pid.clone(), comm.clone())
    }
    fn pid(self: &Self) -> Option<PID> {
        Some(self.0.clone())
    }
    fn comm(self: &Self) -> Option<Comm> {
        Some(self.1.clone())
    }
    const STORES_PID: bool = true;
    const STORES_COMM: bool = true;
}
impl TaskID for Comm {
    fn new(_pid: &PID, comm: &Comm) -> Self {
        comm.clone()
    }
    fn pid(self: &Self) -> Option<PID> {
        None
    }
    fn comm(self: &Self) -> Option<Comm> {
        Some(self.clone())
    }
    const STORES_PID: bool = false;
    const STORES_COMM: bool = true;
}
impl TaskID for PID {
    fn new(pid: &PID, _comm: &Comm) -> Self {
        pid.clone()
    }
    fn pid(self: &Self) -> Option<PID> {
        Some(self.clone())
    }
    fn comm(self: &Self) -> Option<Comm> {
        None
    }
    const STORES_PID: bool = true;
    const STORES_COMM: bool = false;
}

// This is a limited set of state compared to what can be described in struct
// task_struct, but sched_switch will only dump states with only one bit set and
// a state listed in the TASK_REPORT mask
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize, JsonSchema)]
enum KernelTaskState {
    Running,
    Interruptible,
    Uninterruptible,
    Stopped,
    Traced,
    Dead,
    Zombie,
    Parked,
    // If the conversion failed and unknown flags are passed
    Unknown(u32),
}

impl From<u32> for KernelTaskState {
    fn from(flags: u32) -> Self {
        if flags == 0 {
            KernelTaskState::Running
        } else if flags & 0x1 != 0 {
            KernelTaskState::Interruptible
        } else if flags & 0x2 != 0 {
            KernelTaskState::Uninterruptible
        } else if flags & 0x4 != 0 {
            KernelTaskState::Stopped
        } else if flags & 0x8 != 0 {
            KernelTaskState::Traced
        } else if flags & 0x10 != 0 {
            KernelTaskState::Dead
        } else if flags & 0x20 != 0 {
            KernelTaskState::Zombie
        } else if flags & 0x40 != 0 {
            KernelTaskState::Parked
        } else if flags & 0x80 != 0 {
            // This one is really fishy: the code is the one of TASK_DEAD, but
            // TASK_DEAD is not part of TASK_REPORT and therefore should not
            // appear in the bitflags logged by sched_switch events. However,
            // there is another flag called TASK_IDLE_REPORT that happens to
            // have the exact same value (coincidence ?) and is not part of
            // TASK_REPORT either, but is set after the processing of the state
            // with TASK_REPORT. At the same time, the format string of
            // sched_switch explicitly maps TASK_DEAD to "I" letter. The stink
            // is real.
            //
            // SInce TASK_IDLE "state" is actually defined as
            // "TASK_UNINTERRUPTIBLE | TASK_NOLOAD", consider it as an
            // Uninterruptible sleep.
            KernelTaskState::Uninterruptible
            // Another fishy one: TASK_REPORT_MAX==0x100 value is used to report
            // running preempted tasks. Since we can already see that a task is
            // preempted with TaskState::Inactive(KernelTaskState::Running),
            // there is no need to map it to something else than Running.
        } else if flags & 0x100 != 0 {
            KernelTaskState::Running
        } else {
            KernelTaskState::Unknown(flags)
        }
    }
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize, JsonSchema)]
enum TaskFinishedReason {
    Dead,
    Renamed(Comm),
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize, JsonSchema)]
enum TaskState {
    Waking(CPU),
    Active(CPU),
    // Preempted is TaskState::Inactive(KernelState::Running)
    Inactive(KernelTaskState),
    Finished(TaskFinishedReason),
}

// impl Display for TaskState {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         write!(f, "{:?}", self)
//     }
// }

#[stream(item = SignalUpdate<T, TaskState>)]
async fn tasks_state<T: TaskID + 'static>(x: &Event) {
    match &x.data {
        EventData::EventSchedWakeup(fields) => {
            yield SignalUpdate::Update(
                T::new(&fields.pid, &fields.comm),
                TaskState::Waking(fields.target_cpu),
            )
        }
        EventData::EventSchedSwitch(fields) => {
            let prev_kernel = KernelTaskState::from(fields.prev_state);
            let prev_id = T::new(&fields.prev_pid, &fields.prev_comm);
            let cpu = fields.__cpu;
            // The task is switched out in 2 cases only:
            // 1. It is dead
            // 2. It is preempted, or otherwise stopped (e.g. SIGSTOP)
            match prev_kernel {
                // We detect the end task using prev_state=Z. This avoids
                // requiring a sched_process_exit event that may not have been
                // collected in usual cases, and also makes manual trace
                // analysis somewhat easier and is less likely to break
                // developer assumptions (i.e. the task finishes the last time
                // it's switched out and not before). If someone wants something
                // based on sched_process_exit, they are likely having special
                // needs and they should probably make their own signal.
                //
                // Using sched_switch also conveniently allows detecting the end
                // of a task at the exact same timestamp another task takes
                // over. This can facilitate aligning things in a GUI without
                // having weird gaps.
                KernelTaskState::Dead | KernelTaskState::Zombie => {
                    yield SignalUpdate::Update(
                        prev_id.clone(),
                        TaskState::Finished(TaskFinishedReason::Dead),
                    );
                    yield SignalUpdate::Finished(prev_id);
                }
                // We could still have KernelTaskState::Running, but that means
                // the task has been preempted.
                _ => yield SignalUpdate::Update(prev_id, TaskState::Inactive(prev_kernel)),
            }
            yield SignalUpdate::Update(
                T::new(&fields.next_pid, &fields.next_comm),
                TaskState::Active(cpu),
            )
        }
        // If we don't store the comm, changes to the comm are irrelevant. If we
        // do store the comm and the task gets renamed, we consider this as the
        // end of the current task. Any subsequent reuse (by that PID or another
        // one) of the same comm will appear as a new task freshly created.
        EventData::EventTaskRename(fields) if T::STORES_COMM => {
            let id = T::new(&fields.pid, &fields.oldcomm);
            yield SignalUpdate::Update(
                id.clone(),
                TaskState::Finished(TaskFinishedReason::Renamed(fields.newcomm.clone())),
            );
            yield SignalUpdate::Finished(id);
        }
        _ => {}
    }
}

#[derive(Debug, Serialize, JsonSchema)]
struct DynamicTaskID(PID, Option<Comm>);

impl From<PID> for DynamicTaskID {
    fn from(pid: PID) -> Self {
        DynamicTaskID(pid, None)
    }
}

impl From<(PID, Comm)> for DynamicTaskID {
    fn from(pidcomm: (PID, Comm)) -> Self {
        let (pid, comm) = pidcomm;
        DynamicTaskID(pid, Some(comm))
    }
}

make_row_struct! {
    struct TaskstateRow {
        ts: Timestamp,
        task: DynamicTaskID,
        state: TaskState,
    }
}

fn _tasks_states<S: EventStream, T: TaskID + Ord + Into<DynamicTaskID> + 'static>(stream: S) -> impl Stream<Item = TaskstateRow> {
    stream.demux(&tasks_state::<T>).filter_map(|x| async {
        match x {
            SignalValue::Current(ts, task, state) => Some(TaskstateRow { ts, task: task.into(), state }),
            _ => None,
        }
    })
}

crate::const_event_req!(EVENTS1, ("sched_switch" and "cpu_frequency"));

analysis! {
    name:
    hello,
    events: ({EVENTS1}),
    (stream: EventStream, _x: Option<u32>) {
        AnalysisResult::new(sum(stream.fork()).await)
    }
}

analysis! {
    name: hello2,
    events: ("sched_wakeup" and "sched_switch" and "task_rename"),
    (stream: EventStream, _x: Option<u32>) {
        AnalysisResult::from_row_stream(_tasks_states::<_, PID>(stream)).await
    }
}

analysis! {
    name: hello3,
    events: ("sched_wakeup" and "sched_switch" and "task_rename"),
    (stream: EventStream, _x: ()) {
        AnalysisResult::from_row_stream(_tasks_states::<_, PID>(stream)).await
    }
}

#[derive(Deserialize, JsonSchema, Debug)]
pub struct TasksStatesParams {
    track_comm: bool,
}

analysis! {
    name: tasks_states,
    events: ("sched_wakeup" and "sched_switch" and "task_rename"),
    (stream: EventStream, args: TasksStatesParams) {
        if args.track_comm {
            AnalysisResult::from_row_stream(_tasks_states::<_, (PID, Comm)>(stream)).await
        } else {
            AnalysisResult::from_row_stream(_tasks_states::<_, PID>(stream)).await
        }
    }
}
