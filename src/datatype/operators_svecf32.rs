use crate::datatype::memory_svecf32::{SVecf32Input, SVecf32Output};
use crate::error::*;
use base::operator::*;
use base::scalar::*;
use base::vector::*;
use num_traits::Zero;
use std::num::NonZero;
use std::ops::Deref;

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vectors_svecf32_operator_add(lhs: SVecf32Input<'_>, rhs: SVecf32Input<'_>) -> SVecf32Output {
    check_matched_dims(lhs.dims() as _, rhs.dims() as _);

    let size1 = lhs.len();
    let size2 = rhs.len();
    let mut pos1 = 0;
    let mut pos2 = 0;
    let mut pos = 0;
    let mut indexes = vec![0; size1 + size2];
    let mut values = vec![F32::zero(); size1 + size2];
    let lhs = lhs.for_borrow();
    let rhs = rhs.for_borrow();
    while pos1 < size1 && pos2 < size2 {
        let lhs_index = lhs.indexes()[pos1];
        let rhs_index = rhs.indexes()[pos2];
        let lhs_value = lhs.values()[pos1];
        let rhs_value = rhs.values()[pos2];
        indexes[pos] = lhs_index.min(rhs_index);
        values[pos] = F32((lhs_index <= rhs_index) as u32 as f32) * lhs_value
            + F32((lhs_index >= rhs_index) as u32 as f32) * rhs_value;
        pos1 += (lhs_index <= rhs_index) as usize;
        pos2 += (lhs_index >= rhs_index) as usize;
        pos += (!values[pos].is_zero()) as usize;
    }
    for i in pos1..size1 {
        indexes[pos] = lhs.indexes()[i];
        values[pos] = lhs.values()[i];
        pos += 1;
    }
    for i in pos2..size2 {
        indexes[pos] = rhs.indexes()[i];
        values[pos] = rhs.values()[i];
        pos += 1;
    }
    indexes.truncate(pos);
    values.truncate(pos);

    SVecf32Output::new(SVecf32Borrowed::new(lhs.dims(), &indexes, &values))
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vectors_svecf32_operator_minus(lhs: SVecf32Input<'_>, rhs: SVecf32Input<'_>) -> SVecf32Output {
    check_matched_dims(lhs.dims() as _, rhs.dims() as _);

    let size1 = lhs.len();
    let size2 = rhs.len();
    let mut pos1 = 0;
    let mut pos2 = 0;
    let mut pos = 0;
    let mut indexes = vec![0; size1 + size2];
    let mut values = vec![F32::zero(); size1 + size2];
    let lhs = lhs.for_borrow();
    let rhs = rhs.for_borrow();
    while pos1 < size1 && pos2 < size2 {
        let lhs_index = lhs.indexes()[pos1];
        let rhs_index = rhs.indexes()[pos2];
        let lhs_value = lhs.values()[pos1];
        let rhs_value = rhs.values()[pos2];
        indexes[pos] = lhs_index.min(rhs_index);
        values[pos] = F32((lhs_index <= rhs_index) as u32 as f32) * lhs_value
            - F32((lhs_index >= rhs_index) as u32 as f32) * rhs_value;
        pos1 += (lhs_index <= rhs_index) as usize;
        pos2 += (lhs_index >= rhs_index) as usize;
        pos += (!values[pos].is_zero()) as usize;
    }
    for i in pos1..size1 {
        indexes[pos] = lhs.indexes()[i];
        values[pos] = lhs.values()[i];
        pos += 1;
    }
    for i in pos2..size2 {
        indexes[pos] = rhs.indexes()[i];
        values[pos] = -rhs.values()[i];
        pos += 1;
    }
    indexes.truncate(pos);
    values.truncate(pos);

    SVecf32Output::new(SVecf32Borrowed::new(lhs.dims(), &indexes, &values))
}

/// Calculate the element-wise multiplication of two sparse vectors.
#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vectors_svecf32_operator_mul(lhs: SVecf32Input<'_>, rhs: SVecf32Input<'_>) -> SVecf32Output {
    check_matched_dims(lhs.dims() as _, rhs.dims() as _);

    let size1 = lhs.len();
    let size2 = rhs.len();
    let mut pos1 = 0;
    let mut pos2 = 0;
    let mut pos = 0;
    let mut indexes = vec![0; std::cmp::min(size1, size2)];
    let mut values = vec![F32::zero(); std::cmp::min(size1, size2)];
    let lhs = lhs.for_borrow();
    let rhs = rhs.for_borrow();
    while pos1 < size1 && pos2 < size2 {
        let lhs_index = lhs.indexes()[pos1];
        let rhs_index = rhs.indexes()[pos2];
        match lhs_index.cmp(&rhs_index) {
            std::cmp::Ordering::Less => {
                pos1 += 1;
            }
            std::cmp::Ordering::Equal => {
                // only both indexes are not zero, values are multiplied
                let lhs_value = lhs.values()[pos1];
                let rhs_value = rhs.values()[pos2];
                indexes[pos] = lhs_index;
                values[pos] = lhs_value * rhs_value;
                pos1 += 1;
                pos2 += 1;
                // only increment pos if the value is not zero
                pos += (!values[pos].is_zero()) as usize;
            }
            std::cmp::Ordering::Greater => {
                pos2 += 1;
            }
        }
    }
    indexes.truncate(pos);
    values.truncate(pos);

    SVecf32Output::new(SVecf32Borrowed::new(lhs.dims(), &indexes, &values))
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vectors_svecf32_operator_lt(lhs: SVecf32Input<'_>, rhs: SVecf32Input<'_>) -> bool {
    check_matched_dims(lhs.dims() as _, rhs.dims() as _);
    compare(lhs, rhs).is_lt()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vectors_svecf32_operator_lte(lhs: SVecf32Input<'_>, rhs: SVecf32Input<'_>) -> bool {
    check_matched_dims(lhs.dims() as _, rhs.dims() as _);
    compare(lhs, rhs).is_le()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vectors_svecf32_operator_gt(lhs: SVecf32Input<'_>, rhs: SVecf32Input<'_>) -> bool {
    check_matched_dims(lhs.dims() as _, rhs.dims() as _);
    compare(lhs, rhs).is_gt()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vectors_svecf32_operator_gte(lhs: SVecf32Input<'_>, rhs: SVecf32Input<'_>) -> bool {
    check_matched_dims(lhs.dims() as _, rhs.dims() as _);
    compare(lhs, rhs).is_ge()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vectors_svecf32_operator_eq(lhs: SVecf32Input<'_>, rhs: SVecf32Input<'_>) -> bool {
    check_matched_dims(lhs.dims() as _, rhs.dims() as _);
    lhs.deref().for_borrow() == rhs.deref().for_borrow()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vectors_svecf32_operator_neq(lhs: SVecf32Input<'_>, rhs: SVecf32Input<'_>) -> bool {
    check_matched_dims(lhs.dims() as _, rhs.dims() as _);
    lhs.deref().for_borrow() != rhs.deref().for_borrow()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vectors_svecf32_operator_cosine(lhs: SVecf32Input<'_>, rhs: SVecf32Input<'_>) -> f32 {
    check_matched_dims(lhs.dims() as _, rhs.dims() as _);
    SVecf32Cos::distance(lhs.for_borrow(), rhs.for_borrow()).to_f32()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vectors_svecf32_operator_dot(lhs: SVecf32Input<'_>, rhs: SVecf32Input<'_>) -> f32 {
    check_matched_dims(lhs.dims() as _, rhs.dims() as _);
    SVecf32Dot::distance(lhs.for_borrow(), rhs.for_borrow()).to_f32()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vectors_svecf32_operator_l2(lhs: SVecf32Input<'_>, rhs: SVecf32Input<'_>) -> f32 {
    check_matched_dims(lhs.dims() as _, rhs.dims() as _);
    SVecf32L2::distance(lhs.for_borrow(), rhs.for_borrow()).to_f32()
}

fn compare(a: SVecf32Input<'_>, b: SVecf32Input<'_>) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    assert!(a.dims() == b.dims());
    let lhs = a.for_borrow();
    let rhs = b.for_borrow();
    let mut pos = 0;
    let size1 = lhs.len() as usize;
    let size2 = rhs.len() as usize;
    while pos < size1 && pos < size2 {
        let lhs_index = lhs.indexes()[pos];
        let rhs_index = rhs.indexes()[pos];
        let lhs_value = lhs.values()[pos];
        let rhs_value = rhs.values()[pos];
        match lhs_index.cmp(&rhs_index) {
            Ordering::Less => return lhs_value.cmp(&F32::zero()),
            Ordering::Greater => return F32::zero().cmp(&rhs_value),
            Ordering::Equal => match lhs_value.cmp(&rhs_value) {
                Ordering::Equal => {}
                x => return x,
            },
        }
        pos += 1;
    }
    match size1.cmp(&size2) {
        Ordering::Less => F32::zero().cmp(&rhs.values()[pos]),
        Ordering::Greater => lhs.values()[pos].cmp(&F32::zero()),
        Ordering::Equal => Ordering::Equal,
    }
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vectors_svecf32_sphere_l2_in(
    lhs: SVecf32Input<'_>,
    rhs: pgrx::composite_type!("sphere_svector"),
) -> bool {
    let center: SVecf32Output = match rhs.get_by_index(NonZero::new(1).unwrap()) {
        Ok(Some(s)) => s,
        Ok(None) => pgrx::error!("Bad input: empty center at sphere"),
        Err(_) => unreachable!(),
    };
    check_matched_dims(lhs.dims(), center.dims());
    let radius: f32 = match rhs.get_by_index(NonZero::new(2).unwrap()) {
        Ok(Some(s)) => s,
        Ok(None) => pgrx::error!("Bad input: empty radius at sphere"),
        Err(_) => unreachable!(),
    };
    SVecf32L2::distance(lhs.for_borrow(), center.for_borrow()) < F32(radius)
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vectors_svecf32_sphere_dot_in(
    lhs: SVecf32Input<'_>,
    rhs: pgrx::composite_type!("sphere_svector"),
) -> bool {
    let center: SVecf32Output = match rhs.get_by_index(NonZero::new(1).unwrap()) {
        Ok(Some(s)) => s,
        Ok(None) => pgrx::error!("Bad input: empty center at sphere"),
        Err(_) => unreachable!(),
    };
    check_matched_dims(lhs.dims(), center.dims());
    let radius: f32 = match rhs.get_by_index(NonZero::new(2).unwrap()) {
        Ok(Some(s)) => s,
        Ok(None) => pgrx::error!("Bad input: empty radius at sphere"),
        Err(_) => unreachable!(),
    };
    SVecf32Dot::distance(lhs.for_borrow(), center.for_borrow()) < F32(radius)
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vectors_svecf32_sphere_cos_in(
    lhs: SVecf32Input<'_>,
    rhs: pgrx::composite_type!("sphere_svector"),
) -> bool {
    let center: SVecf32Output = match rhs.get_by_index(NonZero::new(1).unwrap()) {
        Ok(Some(s)) => s,
        Ok(None) => pgrx::error!("Bad input: empty center at sphere"),
        Err(_) => unreachable!(),
    };
    check_matched_dims(lhs.dims(), center.dims());
    let radius: f32 = match rhs.get_by_index(NonZero::new(2).unwrap()) {
        Ok(Some(s)) => s,
        Ok(None) => pgrx::error!("Bad input: empty radius at sphere"),
        Err(_) => unreachable!(),
    };
    SVecf32Cos::distance(lhs.for_borrow(), center.for_borrow()) < F32(radius)
}
