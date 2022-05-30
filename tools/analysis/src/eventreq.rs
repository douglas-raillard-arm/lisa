use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum EventReq {
    #[serde(rename = "single")]
    SingleEvent(&'static str),
    #[serde(rename = "or")]
    OrGroup(Vec<EventReq>),
    #[serde(rename = "and")]
    AndGroup(Vec<EventReq>),
    #[serde(rename = "optional")]
    OptionalGroup(Vec<EventReq>),
    #[serde(rename = "dynamic")]
    DynamicGroup(Vec<EventReq>),
}

fn fmt_group(
    f: &mut fmt::Formatter,
    reqs: &[EventReq],
    op: &'static str,
    prefix: &'static str,
) -> fmt::Result {
    write!(f, "({}", prefix)?;
    let body = reqs
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(op);
    write!(f, "{}", body)?;
    write!(f, ")")
}

impl fmt::Display for EventReq {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            EventReq::SingleEvent(name) => write!(f, "{}", name),
            EventReq::OrGroup(reqs) => fmt_group(f, reqs, " or ", ""),
            EventReq::AndGroup(reqs) => fmt_group(f, reqs, " and ", ""),
            EventReq::OptionalGroup(reqs) => fmt_group(f, reqs, " and ", "optional: "),
            EventReq::DynamicGroup(reqs) => fmt_group(f, reqs, " and ", "one group of: "),
        }
    }
}
