use core::{
    fmt,
    fmt::Display,
    ops::{Deref, DerefMut},
};
use serde::{Deserialize, Serialize};

use schemars::{gen::SchemaGenerator, schema::Schema, JsonSchema};

// type StringImplem = std::string::String;
type StringImplem = smartstring::alias::String;

#[derive(Clone, Debug, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct String(StringImplem);

impl PartialEq<String> for String {
    fn eq(&self, other: &String) -> bool {
        self.0.eq(&other.0)
    }
}

impl<T> PartialEq<T> for String
where
    StringImplem: PartialEq<T>,
{
    fn eq(&self, other: &T) -> bool {
        self.0.eq(&other)
    }
}

impl Deref for String {
    type Target = StringImplem;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for String {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl JsonSchema for String {
    fn schema_name() -> std::string::String {
        <std::string::String as JsonSchema>::schema_name()
    }
    fn json_schema(gen: &mut SchemaGenerator) -> Schema {
        <std::string::String as JsonSchema>::json_schema(gen)
    }

    fn is_referenceable() -> bool {
        <std::string::String as JsonSchema>::is_referenceable()
    }
}
impl Display for String {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}
