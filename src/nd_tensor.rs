use std::cmp::max;
use std::fmt::{Display, Formatter};
use std::iter::{Sum, zip};
use std::ops::{Add, Div, Mul, Sub};

use num::{Float, Integer};
use rand::distributions::{DistIter, Distribution};
use rand::Rng;

use crate::copy_vec;

fn shape_to_offsets(shape: &Vec<usize>) -> Vec<usize> {
    (0..shape.len())
        .map(|i| shape[i + 1..].iter().product::<usize>())
        .collect::<Vec<_>>()
}

fn dot_product(a: &Vec<usize>, b: &Vec<usize>) -> usize {
    a.iter().zip(b.iter())
        .map(|(o, i)| o * i)
        .sum::<usize>()
}

fn unflatten_index(i: usize, offsets: &Vec<usize>) -> Vec<usize> {
    let mut i = i;
    let mut out = vec![];
    offsets.iter().for_each(|o| {
        let (a, b) = i.div_rem(o);
        i = b;
        out.push(a);
    });

    out
}

fn walk_shape(shape: &Vec<usize>) -> impl Iterator<Item=Vec<usize>> {
    let offsets = shape_to_offsets(shape);
    let total = shape.iter().product();

    (0..total).map(move |i| unflatten_index(i, &offsets))
}

#[derive(Debug)]
pub struct NDTensor<T: Float> {
    data: Vec<T>,
    pub(crate) shape: Vec<usize>,
    offsets: Vec<usize>,
}


impl<T: Float> NDTensor<T> {
    pub fn from_vector(data: &Vec<T>) -> Self {
        let shape = vec![data.len()];
        let offsets = shape_to_offsets(&shape);
        Self {
            data: data.iter().copied().collect(),
            shape,
            offsets,
        }
    }

    pub fn from_rng<D: Distribution<T>, R: Rng>(rng: &mut DistIter<D, R, T>, shape: &Vec<usize>) -> Self {
        let data = rng.take(shape.iter().product()).collect();

        let offsets = shape_to_offsets(shape);
        Self {
            data,
            shape: copy_vec!(shape),
            offsets,
        }
    }
    pub fn reshape(&mut self, shape: &Vec<usize>) {
        assert_eq!(shape.iter().product::<usize>(), self.numel(),
                   "New shape must contain the same number of elements as old shape");

        self.shape = copy_vec!(shape);
        self.offsets = shape_to_offsets(&self.shape);
    }

    pub fn get(&self, indices: &Vec<usize>) -> &T {
        assert_eq!(indices.len(), self.shape.len());
        zip(indices, &self.shape).for_each(|(i, s)|
            assert!(i < s, "Index is out of bounds: {i} with size {s}"));

        let flat_idx = dot_product(&self.offsets, indices);

        &self.data[flat_idx]
    }

    pub fn pow(&self, e: T) -> Self {
        Self {
            data: self.data.iter().map(|x| x.powf(e)).collect(),
            shape: copy_vec!(self.shape),
            offsets: copy_vec!(self.offsets),

        }
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn transpose(&self) -> Self {
        assert_eq!(self.shape.len(), 2);

        let mut out = vec![];
        for i in 0..self.shape[1] {
            for j in 0..self.shape[0] {
                out.push(self.get(&vec![j, i]).clone())
            }
        }

        let new_shape = self.shape.iter().copied().rev().collect();

        Self {
            data: out,
            offsets: shape_to_offsets(&new_shape),
            shape: new_shape,
        }
    }
}

impl<T: Float> Clone for NDTensor<T> {
    fn clone(&self) -> Self {
        Self {
            data: copy_vec!(self.data),
            shape: copy_vec!(self.shape),
            offsets: copy_vec!(self.offsets),
        }
    }
}

impl<T: Float + Display> Display for NDTensor<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // todo: this is horrendous, but it works
        for (i, &x) in self.data.iter().enumerate() {
            let unravelled = unflatten_index(i, &self.offsets);

            let last_idx = *unravelled.last().unwrap();

            if last_idx == 0 {
                // Indenting
                let num_indents = unravelled.iter().rev().skip(1)
                    .take_while(|&&x| x == 0)
                    .count();
                let start = self.shape.len() - 1 - num_indents;

                for j in 0..num_indents {
                    for _ in 0..start + j {
                        write!(f, "\t")?;
                    }
                    write!(f, "[\n")?;
                }

                // Left tabbing
                let num_tabs = self.shape.len() - 1;
                for _ in 0..num_tabs { write!(f, "\t")?; }

                // Inner start
                write!(f, "[ ")?
            }

            // Value
            write!(f, "{}\t", x)?;


            if last_idx == *self.shape.last().unwrap() - 1 {
                // Inner end
                write!(f, " ]\n")?;

                // Dedenting
                let num_dedents = zip(
                    unravelled.iter().rev(),
                    self.shape.iter().rev(),
                )
                    .skip(1)
                    .take_while(|(&x, &s)| x == s - 1)
                    .count();

                let start = self.shape.len() - 1 - num_dedents;

                for j in (0..num_dedents).rev() {
                    for _ in 0..start + j {
                        write!(f, "\t")?;
                    }

                    write!(f, "]")?;

                    // Special case - last character we don't want \n
                    if start + j > 0 { write!(f, "\n")?; }
                }
            }
        }

        Ok(())
    }
}

impl<T: Float + Sum> NDTensor<T> {
    /// 2D matrix multiplication (A, B) * (B, C) -> (A, C)
    pub fn matmul(&self, other: &Self) -> Self {
        assert_eq!(self.shape.len(), 2);
        assert_eq!(other.shape.len(), 2);
        assert_eq!(self.shape[1], other.shape[0]);

        let new_shape = vec![self.shape[0], other.shape[1]];


        let out = (0..new_shape[0]).map(|i| {
            (0..new_shape[1]).map(move |j| {
                // Dot product
                (0..self.shape[1]).map(|k| {
                    *self.get(&vec![i, k]) * *other.get(&vec![k, j])
                }).sum()
            })
        })
            .flatten()
            .collect::<Vec<_>>();


        Self {
            data: out,
            offsets: shape_to_offsets(&new_shape),
            shape: new_shape,
        }
    }

    /// Sum over a given axis
    pub fn sum_axis(&self, axis: usize) -> Self {
        assert!(axis < self.shape.len());

        let mut new_shape = copy_vec!(self.shape);
        new_shape[axis] = 1;

        let out = walk_shape(&new_shape)
            .map(|mut idx|
                (0..self.shape[axis])
                    .map(|i| {
                        idx[axis] = i;
                        *self.get(&idx)
                    }).sum::<T>()
            )
            .collect::<Vec<T>>();


        Self {
            data: out,
            offsets: shape_to_offsets(&new_shape),
            shape: new_shape,
        }
    }
}


impl<T: Float + for<'a> Sum<&'a T>> NDTensor<T> {
    pub fn sum(&self) -> T {
        self.data.iter().sum()
    }
}

/// Broadcasted elementwise operation
macro_rules! impl_broadcasted {
    ($trait_name:ident, $func_name:ident, $op:tt) => {
impl<T: Float> $trait_name for &NDTensor<T> {
    type Output = NDTensor<T>;

    fn $func_name(self, rhs: Self) -> Self::Output {
        zip(&self.shape, &rhs.shape).for_each(|(s1, s2)|
            assert!(s1 == s2 || s1 == &1 || s2 == &1));

        // Calculate output shape
        let new_shape = zip(&self.shape, &rhs.shape)
            .map(|(&s1, &s2)| max(s1, s2))
            .collect::<Vec<_>>();
        let new_offsets = shape_to_offsets(&new_shape);

        let mut new_data = vec![];
        for i in 0..new_shape.iter().product() {
            // Turn dest flat into dest shaped
            let unflattened_dest = unflatten_index(i, &new_offsets);

            // Get corresponding lhs index
            let unflattened_lhs = zip(&unflattened_dest, &self.shape)
                .map(|(&i, &s)| if s == 1 { 0 } else { i })
                .collect::<Vec<_>>();

            // Get corresponding lhs index
            let unflattened_rhs = zip(&unflattened_dest, &rhs.shape)
                .map(|(&i, &s)| if s == 1 { 0 } else { i })
                .collect::<Vec<_>>();

            // Compute
            new_data.push(*self.get(&unflattened_lhs) $op *rhs.get(&unflattened_rhs));
        }

        Self::Output {
            data: new_data,
            shape: new_shape,
            offsets: new_offsets,
        }
    }
}
    };
}

impl_broadcasted!(Add, add, +);
impl_broadcasted!(Sub, sub, -);
impl_broadcasted!(Mul, mul, *);
impl_broadcasted!(Div, div, /);


/// Tensor + float operations
macro_rules! impl_scalar {
    ($trait_name:ident, $func_name:ident, $op:tt) => {
        
impl<T: Float> $trait_name<&T> for &NDTensor<T> {
    type Output = NDTensor<T>;

    fn $func_name(self, rhs: &T) -> Self::Output {
        Self::Output {
            data: self.data.iter().map(|&x| x $op *rhs).collect(),
            shape: copy_vec!(self.shape),
            offsets: copy_vec!(self.offsets),
        }
    }
}
        
    };
}

impl_scalar!(Add, add, +);
impl_scalar!(Sub, sub, -);
impl_scalar!(Mul, mul, *);
impl_scalar!(Div, div, /);


#[cfg(test)]
mod tests {
    use crate::nd_tensor::{NDTensor, walk_shape};

    #[test]
    fn nd_tensor() {
        let mut tensor = NDTensor::from_vector(&vec![0.1, 0.2, 0.3, 0.4]);
        println!("{}", tensor);

        // Reshapes
        tensor.reshape(&vec![2, 2]);
        println!("{}", tensor);
        tensor.reshape(&vec![1, 2, 2, 1]);
        println!("{}", tensor);

        // Indexing
        assert_eq!(tensor.get(&vec![0, 0, 0, 0]), &0.1);
        assert_eq!(tensor.get(&vec![0, 0, 1, 0]), &0.2);
        assert_eq!(tensor.get(&vec![0, 1, 0, 0]), &0.3);
        assert_eq!(tensor.get(&vec![0, 1, 1, 0]), &0.4);

        let mut tensor1 = NDTensor::from_vector(&vec![0.1, 0.2, 0.3, 0.4]);
        tensor1.reshape(&vec![4, 1]);
        let mut tensor2 = NDTensor::from_vector(&vec![0.1, 0.2, 0.3, 0.4]);
        tensor2.reshape(&vec![1, 4]);

        // Broadcasted ops
        println!("{:?}", &tensor1 + &tensor2);
        println!("{:?}", &tensor1 - &tensor2);
        println!("{:?}", &tensor1 * &tensor2);
        println!("{:?}", &tensor1 / &tensor2);

        let mut tensor1 = NDTensor::from_vector(&vec![0.1, 0.2]);
        tensor1.reshape(&vec![1, 2]);
        let mut tensor2 = NDTensor::from_vector(&vec![0.1, 0.2, 0.3, 0.4]);
        tensor2.reshape(&vec![2, 2]);

        println!("{:?}", &tensor1 + &tensor2);
        println!("{:?}", &tensor1 - &tensor2);
        println!("{:?}", &tensor1 * &tensor2);
        println!("{:?}", &tensor1 / &tensor2);

        // Scalar ops
        println!("{:?}", &tensor1 + &2.);
        println!("{:?}", &tensor1 - &2.);
        println!("{:?}", &tensor1 * &2.);
        println!("{:?}", &tensor1 / &2.);

        // Matmul
        let mut tensor1 = NDTensor::from_vector(&vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        tensor1.reshape(&vec![3, 2]);
        let mut tensor2 = NDTensor::from_vector(&vec![0.1, 0.2]);
        tensor2.reshape(&vec![2, 1]);

        println!("{:?}", &tensor1.matmul(&tensor2));

        // Transpose
        let mut tensor1 = NDTensor::from_vector(&vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        tensor1.reshape(&vec![3, 2]);
        let tensor2 = tensor1.transpose();
        assert_eq!(tensor2.shape, vec![2, 3]);
        assert_eq!(tensor2.get(&vec![1, 1]), &0.4);
        println!("{:?}", tensor1);
        println!("{:?}", tensor2);

        // Sum axis
        let mut tensor1 = NDTensor::from_vector(&vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        tensor1.reshape(&vec![3, 2]);
        println!("{}", tensor1.sum_axis(0));
        println!("{:?}", tensor1.sum_axis(0).shape);
        println!("{}", tensor1.sum_axis(1));
        println!("{:?}", tensor1.sum_axis(1).shape);
    }

    #[test]
    fn utils() {
        let shape = vec![1, 2, 2, 1];


        for i in walk_shape(&shape) {
            println!("{:?}", i);
        }
    }
}