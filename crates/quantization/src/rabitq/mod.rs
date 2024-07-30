use std::ops::Div;

use self::operator::OperatorRaBitQ;
use crate::reranker::error_based::ErrorBasedReranker;
use base::index::{RaBitQuantizationOptions, SearchOptions, VectorOptions};
use base::operator::{Borrowed, Owned, Scalar};
use base::scalar::{ScalarLike, F32};
use base::search::{Reranker, Vectors};
use base::vector::{VectorBorrowed, VectorOwned};
use base::pod::Pod;

use num_traits::{Float, One, Zero};
use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize};

pub mod operator;

const EPSILON: f32 = 1.9;
const THETA_LOG_DIM: u32 = 4;
const DEFAULT_X_DOT_PRODUCT: f32 = 0.8;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaBitFactor<O: OperatorRaBitQ> {
    pub distance_square: Scalar<O>,
    pub error_bound: Scalar<O>,
    pub factor_ip: Scalar<O>,
    pub factor_ppc: Scalar<O>,
    pub binary_vec: Vec<u64>,
}

unsafe impl Pod for RaBitFactor<OperatorRaBitQ> {}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct RaBitQuantizer<O: OperatorRaBitQ> {
    dim: u32,
    dim_pad_64: u32,
    dim_pad_sqrt: Scalar<O>,
    error_base: Scalar<O>,
    projection: Vec<Vec<Scalar<O>>>,
    rand_bias: Vec<Scalar<O>>,
    one_vec: Vec<Scalar<O>>,
    // binary_vec_x: MmapArray<Vec<u64>>,
    // distance_to_centroid_square: MmapArray<Scalar<O>>,
    // error_bound: MmapArray<Scalar<O>>,
    // factor_ip: MmapArray<Scalar<O>>,
    // factor_ppc: MmapArray<Scalar<O>>,
}

impl<O: OperatorRaBitQ> RaBitQuantizer<O> {
    pub fn train(
        vector_options: VectorOptions,
        _options: RaBitQuantizationOptions,
        _vectors: &impl Vectors<O>,
        _transform: impl Fn(Borrowed<'_, O>) -> Owned<O> + Copy,
    ) -> Self {
        let dim_pad = (vector_options.dims + 63) / 64 * 64;
        let mut rand_bias = Vec::with_capacity(dim_pad as usize);
        let mut rng = thread_rng();
        for _ in 0..dim_pad {
            rand_bias.push(Scalar::<O>::from_f32(rng.gen()));
        }
        let projection = O::gen_random_orthogonal(dim_pad as usize);
        // let n = vectors.len() as usize;
        // let mut distance_to_centroid = vec![Scalar::<O>::zero(); n];
        // let mut distance_to_centroid_square = vec![Scalar::<O>::zero(); n];
        // let mut quantized_x = vec![vec![Scalar::<O>::zero(); dim_pad as usize]; n];
        // for i in 0..n {
        //     let vector = transform(vectors.vector(i as u32)).as_borrowed().to_vec();
        //     distance_to_centroid_square[i] = O::vector_dot_product(&vector, &vector);
        //     distance_to_centroid[i] = distance_to_centroid_square[i].sqrt();
        //     for j in 0..vector_options.dims as usize {
        //         quantized_x[i][j] = O::vector_dot_product(&projection[j], &vector);
        //     }
        // }
        // let mut binary_vec_x = Vec::with_capacity(n);
        // let mut signed_x = Vec::with_capacity(n);
        // for i in 0..(n) {
        //     binary_vec_x.push(O::vector_binarize_u64(&quantized_x[i]));
        //     signed_x.push(O::vector_binarize_one(&quantized_x[i]));
        // }
        // let mut dot_product_x = vec![Scalar::<O>::zero(); n];
        // for i in 0..(n) {
        //     let norm = O::vector_dot_product(&quantized_x[i], &quantized_x[i]).sqrt()
        //         * Scalar::<O>::from_f32(dim_pad as f32).sqrt();
        //     dot_product_x[i] = if norm.is_normal() {
        //         O::vector_dot_product(&quantized_x[i], &signed_x[i]).div(norm)
        //     } else {
        //         Scalar::<O>::from_f32(DEFAULT_X_DOT_PRODUCT)
        //     }
        // }

        // let mut error_bound = Vec::with_capacity(n);
        // let mut factor_ip = Vec::with_capacity(n);
        // let mut factor_ppc = Vec::with_capacity(n);
        let error_base = Scalar::<O>::from_f32(2.0 * EPSILON / (dim_pad as f32 - 1.0).sqrt());
        let dim_pad_sqrt = Scalar::<O>::from_f32(dim_pad as f32).sqrt();
        let one_vec = vec![Scalar::<O>::one(); dim_pad as usize];
        // for i in 0..n {
        //     let xc_over_dot_product = distance_to_centroid[i] / dot_product_x[i];
        //     error_bound.push(
        //         error_base
        //             * (xc_over_dot_product * xc_over_dot_product - distance_to_centroid_square[i])
        //                 .sqrt(),
        //     );
        //     let ip = Scalar::<O>::from_f32(-2.0) / dim_pad_sqrt * xc_over_dot_product;
        //     factor_ip.push(ip);
        //     factor_ppc.push(ip * O::vector_dot_product(&one_vec, &signed_x[i]));
        // }

        Self {
            dim: vector_options.dims,
            dim_pad_64: dim_pad,
            projection,
            rand_bias,
            error_base,
            dim_pad_sqrt,
            one_vec,
            // binary_vec_x,
            // distance_to_centroid_square,
            // error_bound,
            // factor_ip,
            // factor_ppc,
        }
    }

    pub fn width(&self) -> usize {
        (self.dim / 64) as usize
    }

    pub fn encode(&self, vector: &[Scalar<O>]) -> RaBitFactor<O> {
        let distance_square = O::vector_dot_product(&vector, &vector);
        let distance = distance_square.sqrt();
        let mut quantized = vec![Scalar::<O>::zero(); self.dim_pad_64 as usize];
        for i in 0..self.dim as usize {
            quantized[i] = O::vector_dot_product(&self.projection[i], &vector);
        }
        let binary_vec = O::vector_binarize_u64(&quantized);
        let signed = O::vector_binarize_one(&quantized);
        let norm = O::vector_dot_product(&quantized, &quantized).sqrt()
            * Scalar::<O>::from_f32(self.dim_pad_64 as f32).sqrt();
        let dot_product = if norm.is_normal() {
            O::vector_dot_product(&quantized, &signed).div(norm)
        } else {
            Scalar::<O>::from_f32(DEFAULT_X_DOT_PRODUCT)
        };
        let xc_over_dot_product = distance / dot_product;
        let error_bound = self.error_base * (xc_over_dot_product * xc_over_dot_product - distance_square).sqrt();
        let factor_ip = Scalar::<O>::from_f32(-2.0) / self.dim_pad_sqrt * xc_over_dot_product;
        let factor_ppc = factor_ip * O::vector_dot_product(&self.one_vec, &signed);
        RaBitFactor {
            distance_square,
            error_bound,
            factor_ip,
            factor_ppc,
            binary_vec,
        }
    }

    pub fn flat_rerank<'a, T: 'a>(
        &'a self,
        _vector: Borrowed<'a, O>,
        _opts: &'a SearchOptions,
        _r: impl Fn(u32) -> (F32, T) + 'a,
    ) -> Box<dyn Reranker<T> + 'a> {
        unimplemented!()
    }

    pub fn ivf_naive_rerank<'a, T: 'a>(
        &'a self,
        _vector: Borrowed<'a, O>,
        _opts: &'a SearchOptions,
        _r: impl Fn(u32) -> (F32, T) + 'a,
    ) -> Box<dyn Reranker<T> + 'a> {
        unimplemented!()
    }

    pub fn preprocess(&self, lhs: Borrowed<'_, O>) -> O::RabitQuantizationPreprocessed {
        O::rabit_quantization_preprocess(
            self.dim_pad_64 as usize,
            lhs,
            &self.projection,
            &self.rand_bias,
        )
    }

    pub fn ivf_residual_rerank<'a, T: 'a>(
        &'a self,
        vectors: Vec<Owned<O>>,
        _: &'a SearchOptions,
        c: impl Fn(u32) -> &'a RaBitFactor<O> + 'a,
        r: impl Fn(u32) -> (F32, T) + 'a,
    ) -> Box<dyn Reranker<T, usize> + 'a> {
        let p = vectors
            .into_iter()
            .map(|vector| {
                O::rabit_quantization_preprocess(
                    self.dim_pad_64 as usize,
                    vector.as_borrowed(),
                    &self.projection,
                    &self.rand_bias,
                )
            })
            .collect::<Vec<_>>();

        Box::new(ErrorBasedReranker::new(
            move |xi, ci| {
                O::rabit_quantization_process(
                    c(xi),
                    &p[ci],
                )
            },
            r,
        ))
    }

    pub fn graph_rerank<'a, T: 'a>(
        &'a self,
        _vector: Borrowed<'a, O>,
        _opts: &'a SearchOptions,
        _r: impl Fn(u32) -> (F32, T) + 'a,
    ) -> Box<dyn Reranker<T> + 'a> {
        unimplemented!()
    }
}
