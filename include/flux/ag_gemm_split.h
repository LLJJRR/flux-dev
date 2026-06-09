#pragma once

namespace bytedance::flux {

// 第一版建议先用 2 或 4，不要一上来太大
static constexpr int kAGGemmSplit = 1;

// 根据你实际最大 world_size 调整。
// 如果 world_size=8, split=4，需要 32 个 signal。
// 如果 world_size=16, split=4，需要 64 个 signal。
static constexpr int kAGGemmMaxWorldSize = 16;
static constexpr int kAGGemmNumSignals = kAGGemmMaxWorldSize * kAGGemmSplit;

}  // namespace bytedance::flux
