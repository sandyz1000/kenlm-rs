use super::count::WarningAction;

// Add missing types
#[derive(Debug, Clone, Default)]
pub struct ChainConfig {
    pub memory: u64,
    pub temp_prefix: String,
}

#[derive(Clone, Debug)]
pub struct Discount {
    pub amount: [f32; 4],
}

#[derive(Debug, Clone, Default)]
pub struct InitialProbabilitiesConfig {
    // These should be small buffers to keep the adder from getting too far ahead
    adder_in: ChainConfig,
    adder_out: ChainConfig,
    // SRILM doesn't normally interpolate unigrams.
    interpolate_unigrams: bool,
}

#[derive(Debug, Clone)]
pub struct DiscountConfig {
    // Overrides discounts for orders [1,discount_override.size()].
    overwrite: Vec<Discount>,
    // If discounting fails for an order, copy them from here.
    fallback: Discount,
    //  What to do when discounts are out of range or would trigger divison by
    //  zero. It does something other than THROW_UP, use fallback_discount.
    bad_action: WarningAction,
}
