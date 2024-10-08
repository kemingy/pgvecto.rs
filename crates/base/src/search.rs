use crate::always_equal::AlwaysEqual;
use crate::distance::Distance;
use crate::vector::VectorOwned;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::fmt::Display;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Handle {
    tenant_id: u128,
    cluster_id: u64,
    database_id: u32,
    index_id: u32,
}

impl Handle {
    pub fn new(tenant_id: u128, cluster_id: u64, database_id: u32, index_id: u32) -> Self {
        Self {
            tenant_id,
            cluster_id,
            database_id,
            index_id,
        }
    }
}

impl Display for Handle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:032x}{:016x}{:08x}{:08x}",
            self.tenant_id, self.cluster_id, self.database_id, self.index_id
        )
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Pointer {
    value: u64,
}

impl Pointer {
    pub fn new(value: u64) -> Self {
        Self { value }
    }
    pub fn as_u64(self) -> u64 {
        self.value
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct Payload {
    pointer: Pointer,
    time: u64,
}

impl Payload {
    pub fn new(pointer: Pointer, time: u64) -> Self {
        Self { pointer, time }
    }
    pub fn pointer(&self) -> Pointer {
        self.pointer
    }
    pub fn time(&self) -> u64 {
        self.time
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Element {
    pub distance: Distance,
    pub payload: AlwaysEqual<Payload>,
}

pub trait Vectors<V: VectorOwned> {
    fn dims(&self) -> u32;
    fn len(&self) -> u32;
    fn vector(&self, i: u32) -> V::Borrowed<'_>;
}

pub trait Collection {
    fn payload(&self, i: u32) -> Payload;
}

pub trait Source {
    fn get_main<T: Any>(&self) -> Option<&T>;
    fn get_main_len(&self) -> u32;
    fn check_existing(&self, i: u32) -> bool;
}

pub trait RerankerPop<T> {
    fn pop(&mut self) -> Option<(Distance, u32, T)>;
}

pub trait RerankerPush {
    fn push(&mut self, u: u32);
}

pub trait FlatReranker<T>: RerankerPop<T> {}

impl<'a, T> RerankerPop<T> for Box<dyn FlatReranker<T> + 'a> {
    fn pop(&mut self) -> Option<(Distance, u32, T)> {
        self.as_mut().pop()
    }
}
