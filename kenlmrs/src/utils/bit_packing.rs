/// Bit-level packing routines matching KenLM's C++ util/bit_packing.hh
///
/// WARNING: The write functions assume that memory is zero initially.
/// This makes them faster and is the appropriate case for mmapped language model construction.

/// Determine byte order at compile time
#[cfg(target_endian = "little")]
#[inline]
fn bit_pack_shift(bit: u8, _length: u8) -> u8 {
    bit
}

#[cfg(target_endian = "big")]
#[inline]
fn bit_pack_shift(bit: u8, length: u8) -> u8 {
    64 - length - bit
}

#[cfg(target_endian = "little")]
#[inline]
fn bit_pack_shift32(bit: u8, _length: u8) -> u8 {
    bit
}

#[cfg(target_endian = "big")]
#[inline]
fn bit_pack_shift32(bit: u8, length: u8) -> u8 {
    32 - length - bit
}

#[inline]
fn read_off(base: &[u8], bit_off: u64) -> u64 {
    let byte_offset = (bit_off >> 3) as usize;
    if byte_offset + 8 <= base.len() {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&base[byte_offset..byte_offset + 8]);
        u64::from_ne_bytes(bytes)
    } else {
        0
    }
}

/// Pack integers up to 57 bits using their least significant digits.
/// The length is specified using mask: Assumes mask == (1 << length) - 1 where length <= 57.
#[inline]
pub fn read_int57(base: &[u8], bit_off: u64, length: u8, mask: u64) -> u64 {
    (read_off(base, bit_off) >> bit_pack_shift(bit_off as u8 & 7, length)) & mask
}

/// Assumes value < (1 << length) and length <= 57.
/// Assumes the memory is zero initially.
#[inline]
pub fn write_int57(base: &mut [u8], bit_off: u64, length: u8, value: u64) {
    let byte_offset = (bit_off >> 3) as usize;
    if byte_offset + 8 <= base.len() {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&base[byte_offset..byte_offset + 8]);
        let mut val = u64::from_ne_bytes(bytes);
        val |= value << bit_pack_shift(bit_off as u8 & 7, length);
        base[byte_offset..byte_offset + 8].copy_from_slice(&val.to_ne_bytes());
    }
}

/// Same caveats as above, but for a 25 bit limit.
#[inline]
pub fn read_int25(base: &[u8], bit_off: u64, length: u8, mask: u32) -> u32 {
    let byte_offset = (bit_off >> 3) as usize;
    if byte_offset + 4 <= base.len() {
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(&base[byte_offset..byte_offset + 4]);
        let val = u32::from_ne_bytes(bytes);
        (val >> bit_pack_shift32(bit_off as u8 & 7, length)) & mask
    } else {
        0
    }
}

#[inline]
pub fn write_int25(base: &mut [u8], bit_off: u64, length: u8, value: u32) {
    let byte_offset = (bit_off >> 3) as usize;
    if byte_offset + 4 <= base.len() {
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(&base[byte_offset..byte_offset + 4]);
        let mut val = u32::from_ne_bytes(bytes);
        val |= value << bit_pack_shift32(bit_off as u8 & 7, length);
        base[byte_offset..byte_offset + 4].copy_from_slice(&val.to_ne_bytes());
    }
}

const K_SIGN_BIT: u32 = 0x80000000;

#[inline]
pub fn read_float32(base: &[u8], bit_off: u64) -> f32 {
    let encoded = (read_off(base, bit_off) >> bit_pack_shift(bit_off as u8 & 7, 32)) as u32;
    f32::from_bits(encoded)
}

#[inline]
pub fn write_float32(base: &mut [u8], bit_off: u64, value: f32) {
    write_int57(base, bit_off, 32, value.to_bits() as u64);
}

/// Read a non-positive float stored in 31 bits
#[inline]
pub fn read_non_positive_float31(base: &[u8], bit_off: u64) -> f32 {
    let mut encoded = (read_off(base, bit_off) >> bit_pack_shift(bit_off as u8 & 7, 31)) as u32;
    // Sign bit set means negative
    encoded |= K_SIGN_BIT;
    f32::from_bits(encoded)
}

/// Write a non-positive float in 31 bits
#[inline]
pub fn write_non_positive_float31(base: &mut [u8], bit_off: u64, value: f32) {
    let mut encoded = value.to_bits();
    encoded &= !K_SIGN_BIT;
    write_int57(base, bit_off, 31, encoded as u64);
}

/// Return bits required to store integers up to max_value
pub fn required_bits(max_value: u64) -> u8 {
    if max_value == 0 {
        return 0;
    }
    64 - max_value.leading_zeros() as u8
}

/// Bit mask helper
#[derive(Debug, Clone, Copy, Default)]
pub struct BitsMask {
    pub bits: u8,
    pub mask: u64,
}

impl BitsMask {
    pub fn by_max(max_value: u64) -> Self {
        let bits = required_bits(max_value);
        Self {
            bits,
            mask: if bits > 0 { (1u64 << bits) - 1 } else { 0 },
        }
    }

    pub fn by_bits(bits: u8) -> Self {
        Self {
            bits,
            mask: if bits > 0 { (1u64 << bits) - 1 } else { 0 },
        }
    }

    pub fn from_max(&mut self, max_value: u64) {
        self.bits = required_bits(max_value);
        self.mask = if self.bits > 0 {
            (1u64 << self.bits) - 1
        } else {
            0
        };
    }
}

/// Bit address for reading/writing bit-packed values
#[derive(Debug, Clone, Default)]
pub struct BitAddress {
    pub base: Vec<u8>,
    pub offset: u64,
}

impl BitAddress {
    pub fn new(base: Vec<u8>, offset: u64) -> Self {
        Self { base, offset }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_required_bits() {
        assert_eq!(required_bits(0), 0);
        assert_eq!(required_bits(1), 1);
        assert_eq!(required_bits(2), 2);
        assert_eq!(required_bits(3), 2);
        assert_eq!(required_bits(7), 3);
        assert_eq!(required_bits(255), 8);
    }

    #[test]
    fn test_bits_mask() {
        let mask = BitsMask::by_max(255);
        assert_eq!(mask.bits, 8);
        assert_eq!(mask.mask, 255);
    }

    #[test]
    fn test_read_write_int57() {
        let mut buffer = vec![0u8; 16];
        write_int57(&mut buffer, 0, 8, 42);
        assert_eq!(read_int57(&buffer, 0, 8, 0xFF), 42);
    }

    #[test]
    fn test_read_write_float32() {
        let mut buffer = vec![0u8; 16];
        let value = -0.5f32;
        write_float32(&mut buffer, 0, value);
        let read_value = read_float32(&buffer, 0);
        assert!((read_value - value).abs() < 1e-6);
    }
}
