#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
use alloc::string::{String,ToString};
use core::fmt;

use anyhow::ensure;
use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use ts_rs::TS;
use crate::field::goldilocks_field::GoldilocksField;
use crate::field::types::{Field, PrimeField64, Sample};
use crate::hash::poseidon::Poseidon;
use crate::iop::target::Target;
use crate::plonk::config::GenericHashOut;

/// A prime order field with the features we need to use it as a base field in our argument system.
pub trait RichField: PrimeField64 + Poseidon + ts_rs::TS {}

impl RichField for GoldilocksField {}

pub const NUM_HASH_OUT_ELTS: usize = 4;

/// Represents a ~256 bit hash output.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "serialize_rkyv", derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize))]
#[cfg_attr(feature = "serialize_speedy", derive(speedy::Readable, speedy::Writable))]
#[cfg_attr(feature = "serialize_bytemuck", derive(bytemuck::Pod, bytemuck::Zeroable))]
#[serde(bound = "")]
#[repr(transparent)]
pub struct HashOut<F: Field> {
    pub elements: [F; NUM_HASH_OUT_ELTS],
}

impl<F: Field> TS for HashOut<F> {
    type WithoutGenerics = HashOut<GoldilocksField>;

    fn name() -> String { "HashOut".to_string() }
    fn inline() -> String { "{ elements: [bigint, bigint, bigint, bigint] }".to_string() }
    fn inline_flattened() -> String { "elements: [bigint, bigint, bigint, bigint]".to_string() }
    fn decl() -> String { "export type HashOut = { elements: [bigint, bigint, bigint, bigint] };".to_string() }
    fn decl_concrete() -> String { Self::decl() }
}

impl<F: Field> HashOut<F> {
    pub const ZERO: Self = Self {
        elements: [F::ZERO; NUM_HASH_OUT_ELTS],
    };

    // TODO: Switch to a TryFrom impl.
    pub fn from_vec(elements: Vec<F>) -> Self {
        debug_assert!(elements.len() == NUM_HASH_OUT_ELTS);
        Self {
            elements: elements.try_into().unwrap(),
        }
    }

    pub fn from_partial(elements_in: &[F]) -> Self {
        let mut elements = [F::ZERO; NUM_HASH_OUT_ELTS];
        elements[0..elements_in.len()].copy_from_slice(elements_in);
        Self { elements }
    }
}

impl<F: Field> From<[F; NUM_HASH_OUT_ELTS]> for HashOut<F> {
    fn from(elements: [F; NUM_HASH_OUT_ELTS]) -> Self {
        Self { elements }
    }
}

impl<F: Field> TryFrom<&[F]> for HashOut<F> {
    type Error = anyhow::Error;

    fn try_from(elements: &[F]) -> Result<Self, Self::Error> {
        ensure!(elements.len() == NUM_HASH_OUT_ELTS);
        Ok(Self {
            elements: elements.try_into().unwrap(),
        })
    }
}

impl<F> Sample for HashOut<F>
where
    F: Field,
{
    #[inline]
    fn sample<R>(rng: &mut R) -> Self
    where
        R: rand::RngCore + ?Sized,
    {
        Self {
            elements: [
                F::sample(rng),
                F::sample(rng),
                F::sample(rng),
                F::sample(rng),
            ],
        }
    }
}

impl<F: RichField> GenericHashOut<F> for HashOut<F> {
    fn to_bytes(&self) -> Vec<u8> {
        self.elements
            .into_iter()
            .flat_map(|x| x.to_canonical_u64().to_le_bytes())
            .collect()
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        HashOut {
            elements: bytes
                .chunks(8)
                .take(NUM_HASH_OUT_ELTS)
                .map(|x| F::from_canonical_u64(u64::from_le_bytes(x.try_into().unwrap())))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        }
    }

    fn to_vec(&self) -> Vec<F> {
        self.elements.to_vec()
    }
}

impl<F: Field> Default for HashOut<F> {
    fn default() -> Self {
        Self::ZERO
    }
}

/// Represents a ~256 bit hash output.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct HashOutTarget {
    pub elements: [Target; NUM_HASH_OUT_ELTS],
}

impl HashOutTarget {
    // TODO: Switch to a TryFrom impl.
    pub fn from_vec(elements: Vec<Target>) -> Self {
        debug_assert!(elements.len() == NUM_HASH_OUT_ELTS);
        Self {
            elements: elements.try_into().unwrap(),
        }
    }

    pub fn from_partial(elements_in: &[Target], zero: Target) -> Self {
        let mut elements = [zero; NUM_HASH_OUT_ELTS];
        elements[0..elements_in.len()].copy_from_slice(elements_in);
        Self { elements }
    }
}

impl From<[Target; NUM_HASH_OUT_ELTS]> for HashOutTarget {
    fn from(elements: [Target; NUM_HASH_OUT_ELTS]) -> Self {
        Self { elements }
    }
}

impl TryFrom<&[Target]> for HashOutTarget {
    type Error = anyhow::Error;

    fn try_from(elements: &[Target]) -> Result<Self, Self::Error> {
        ensure!(elements.len() == NUM_HASH_OUT_ELTS);
        Ok(Self {
            elements: elements.try_into().unwrap(),
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MerkleCapTarget(pub Vec<HashOutTarget>);

/// Hash consisting of a byte array.
#[derive(Eq, PartialEq, Copy, Clone, Debug)]
pub struct BytesHash<const N: usize>(pub [u8; N]);

impl<const N: usize> Sample for BytesHash<N> {
    #[inline]
    fn sample<R>(rng: &mut R) -> Self
    where
        R: rand::RngCore + ?Sized,
    {
        let mut buf = [0; N];
        rng.fill_bytes(&mut buf);
        Self(buf)
    }
}

impl<F: RichField, const N: usize> GenericHashOut<F> for BytesHash<N> {
    fn to_bytes(&self) -> Vec<u8> {
        self.0.to_vec()
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        Self(bytes.try_into().unwrap())
    }

    fn to_vec(&self) -> Vec<F> {
        self.0
            // Chunks of 7 bytes since 8 bytes would allow collisions.
            .chunks(7)
            .map(|bytes| {
                let mut arr = [0; 8];
                arr[..bytes.len()].copy_from_slice(bytes);
                F::from_canonical_u64(u64::from_le_bytes(arr))
            })
            .collect()
    }
}

impl<const N: usize> Serialize for BytesHash<N> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_bytes(&self.0)
    }
}

struct ByteHashVisitor<const N: usize>;

impl<'de, const N: usize> Visitor<'de> for ByteHashVisitor<N> {
    type Value = BytesHash<N>;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "an array containing exactly {} bytes", N)
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: de::SeqAccess<'de>,
    {
        let mut bytes = [0u8; N];
        for i in 0..N {
            let next_element = seq.next_element()?;
            match next_element {
                Some(value) => bytes[i] = value,
                None => return Err(de::Error::invalid_length(i, &self)),
            }
        }
        Ok(BytesHash(bytes))
    }

    fn visit_bytes<E>(self, s: &[u8]) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        let bytes = s.try_into().unwrap();
        Ok(BytesHash(bytes))
    }
}

impl<'de, const N: usize> Deserialize<'de> for BytesHash<N> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_seq(ByteHashVisitor::<N>)
    }
}


#[cfg(test)]
mod tests {

    use super::*;


    #[cfg_attr(feature = "serialize_rkyv", derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize))]
    #[cfg_attr(feature = "serialize_speedy", derive(speedy::Readable, speedy::Writable))]
    #[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone, Copy)]
    #[serde(bound = "")]
    pub struct TestStructA<F: PrimeField64> {
        pub a: HashOut<F>,
        pub b: u64,
        pub c: [HashOut<F>; 32],
        pub d: F,
    }
    #[cfg(any(feature = "serialize_rkyv", feature = "serialize_speedy"))]
    impl<F: PrimeField64> TestStructA<F> {
        pub fn new_rand() -> Self {
            Self {
                a: HashOut::rand(),
                b: 1337,
                c: core::array::from_fn(|_| HashOut::rand()),
                d: F::rand(),
            }
        }
    }

    #[cfg_attr(feature = "serialize_rkyv", derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize))]
    #[cfg_attr(feature = "serialize_speedy", derive(speedy::Readable, speedy::Writable))]
    #[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone, Copy)]
    #[serde(bound = "")]
    pub struct TestStructB<F: PrimeField64> {
        pub w: TestStructA<F>,
        pub x: [u8; 32],
        pub y: HashOut<F>,
        pub z: [TestStructA<F>; 4],
    }
    #[cfg(any(feature = "serialize_rkyv", feature = "serialize_speedy"))]
    impl<F: PrimeField64> TestStructB<F> {
        pub fn new_rand() -> Self {
            Self {
                w: TestStructA::new_rand(),
                x: core::array::from_fn(|_| (F::rand().to_canonical_u64()&0xffu64) as u8),
                y: HashOut::rand(),
                z: core::array::from_fn(|_| TestStructA::new_rand()),
            }
        }
    }

    #[cfg_attr(feature = "serialize_rkyv", derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize))]
    #[cfg_attr(feature = "serialize_speedy", derive(speedy::Readable, speedy::Writable))]
    #[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone)]
    #[serde(bound = "")]
    pub struct MerkleProofStructC<F: PrimeField64> {
        pub siblings: Vec<HashOut<F>>,
        pub value: HashOut<F>,
        pub root: HashOut<F>,
        pub index: F,
    }
    
    #[cfg(any(feature = "serialize_rkyv", feature = "serialize_speedy"))]
    impl<F: PrimeField64> MerkleProofStructC<F> {
        pub fn new_rand(depth: usize) -> Self {
            Self {
                siblings: (0..depth).map(|_| HashOut::rand()).collect(),
                value: HashOut::rand(),
                root: HashOut::rand(),
                index: F::rand(),
            }
        }
    }
    

    #[test]
    #[cfg(feature = "serialize_rkyv")]
    fn test_rkyv_round_trip() {
        use crate::field::goldilocks_field::GoldilocksField as GF;
        // Test for TestStructA

        use rkyv::rancor;
        let original_a = TestStructA::<GF>::new_rand();
        let bytes_a = rkyv::to_bytes::<rancor::Error>(&original_a).expect("failed to serialize A");
        let deserialized_a: TestStructA<GF> = rkyv::from_bytes::<_, rancor::Error>(&bytes_a).expect("failed to deserialize A");
        assert_eq!(original_a, deserialized_a);

        // Test for TestStructB
        let original_b = TestStructB::<GF>::new_rand();
        let bytes_b = rkyv::to_bytes::<rancor::Error>(&original_b).expect("failed to serialize B");
        let deserialized_b: TestStructB<GF> = rkyv::from_bytes::<_, rancor::Error>(&bytes_b).expect("failed to deserialize B");
        assert_eq!(original_b, deserialized_b);

        // Test for MerkleProofStructC
        let original_c = MerkleProofStructC::<GF>::new_rand(16);
        let bytes_c = rkyv::to_bytes::<rancor::Error>(&original_c).expect("failed to serialize C");
        let deserialized_c: MerkleProofStructC<GF> = rkyv::from_bytes::<_, rancor::Error>(&bytes_c).expect("failed to deserialize C");
        assert_eq!(original_c, deserialized_c);
    }

    #[test]
    #[cfg(feature = "serialize_speedy")]
    fn test_speedy_round_trip() {
        use crate::field::goldilocks_field::GoldilocksField as GF;

        use speedy::{Readable, Writable};

        // Test for TestStructA
        let original_a = TestStructA::<GF>::new_rand();
        let bytes_a = original_a.write_to_vec().expect("failed to serialize A");
        let deserialized_a = TestStructA::<GF>::read_from_buffer(&bytes_a).expect("failed to deserialize A");
        assert_eq!(original_a, deserialized_a);

        // Test for TestStructB
        let original_b = TestStructB::<GF>::new_rand();
        let bytes_b = original_b.write_to_vec().expect("failed to serialize B");
        let deserialized_b = TestStructB::<GF>::read_from_buffer(&bytes_b).expect("failed to deserialize B");
        assert_eq!(original_b, deserialized_b);
        
        // Test for MerkleProofStructC
        let original_c = MerkleProofStructC::<GF>::new_rand(16);
        let bytes_c = original_c.write_to_vec().expect("failed to serialize C");
        let deserialized_c = MerkleProofStructC::<GF>::read_from_buffer(&bytes_c).expect("failed to deserialize C");
        assert_eq!(original_c, deserialized_c);
    }
    
}