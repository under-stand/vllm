# 最小修复边界（只设计，未改生产代码）

对照的 main 仍是今早的 `418af607`。11:17 UTC 时上游 main 已到 `179ebd9896`，中间 9 个提交都不碰 `scheduler.py`、`kv_cache_manager.py`、`kv_cache_utils.py`，所以今早的矩阵没有重跑。ABA 探针跑在工作区这份 `418af607` 上。

## 历史关闭原因（原文，不是推断）

#61769 没有被 review 打回实现错误。2026-11-12 orozery 在 [评论](https://github.com/vllm-project/vllm/pull/61769#issuecomment-6818514908) 写的是：

> I think the invalid block paths is silently deprecated. There was an initiate to actually remove/simplify it at the time. AFAIK this flow is currently not supported for hybrid models. I don't think we currently want to further invest in it.

GitHub 上他的 association 是 CONTRIBUTOR。这是明确的「不要再往这条路径投」的意见，不是「bug 不存在」或「补丁测错了」。作者 ArijitDeb 随即关闭 PR，并在 [收尾评论](https://github.com/vllm-project/vllm/pull/61769#issuecomment-6830191057) 里写明路径在 main 上仍活着，#61768 继续开着。

#62287（preempt，叠在 #61769 上）由 Lewis-panda 在 [2026-11-13](https://github.com/vllm-project/vllm/pull/62287#issuecomment-6845088071) 关闭，理由是引用 orozery 的那句，并留下 #61912。没有新的技术否决。

#62438 不是这条驱逐缺口。作者自己关的：[先说清理自动化贡献量](https://github.com/vllm-project/vllm/pull/62438#issuecomment-6916443218)，[再说明让给已合入的 #65060](https://github.com/vllm-project/vllm/pull/62438#issuecomment-6917018558)。#65060 在 2026-11-03 合并，做的是 MultiConnector 按子 connector 汇总 consumer/producer，不是按物理 block id 驱逐。xsxszab 在 #62438 上建议过「不要对所有 connector 无条件 defer」，那是建议，不是关闭理由。

推断（他没有点名 PR）：orozery 说的 remove/simplify 对得上仍开着的 [#42461](https://github.com/vllm-project/vllm/pull/42461)（`Remove per-block KV transfer error handling`，带 `stale`，更新停在 2026-10-31，未合并）。不能把这当成项目已经决定删掉 `invalid_block_ids`。

## 现在谁在做

| 项 | 状态 | 和 abort/preempt 缺口的关系 |
| --- | --- | --- |
| #69299 `ecbebf9f` | 仍 open，head 未变，评论只有 bot | 只修「报告时 owner 还在」的 `recompute`。今早探针已核对 |
| #61768 / #61912 / #61905 | 仍 open | 分别是 abort、preempt、id 复用后误伤别人。没有新的实现 PR |
| #42461 | 仍 open、stale | 想删掉 per-block 协议，改成 request id |
| #65923 | 仍 open | 多 KV group 的 `failed_sync_recving`，不是这组单 group block id |
| #61769 / #62287 | closed，未合并 | 原始 block id 账本。没有人在 main 上接着写 |

main 上没有 `_finished_inflight` 这类账本。没有第二套正在合入的 abort/preempt 修复。

## 探针否定了什么

`probe_block_reuse_aba.py`，producer、async scheduler class、`defer_block_free=false`、池子只有 9 个 block。先 abort 同步加载，再 `schedule()` 一个不同前缀，8 个物理 id 全部被它拿走（顺序颠倒）。新请求 `schedule()` 返回时，8 个 id 的 hash 已经没有一个等于失败加载时的快照。

| 做法 | 结果 |
| --- | --- |
| 现有 `_handle_invalid_blocks` | `fail`：无关请求 `FINISHED_ERROR`，computed 被收到 64，它自己的前缀命中剩 64。`recompute`：请求仍 `RUNNING`，computed 收到 64。旧前缀命中是 0（hash 已被盖掉） |
| 探针里按原始 id 全部 `evict_blocks` | 无关请求还在、computed 仍是 128，但它的前缀命中从 112 变成 0 |
| 探针里只驱逐「当前 hash 仍等于快照」的 id | 一个都不驱逐。无关请求命中保持 112，状态不动。旧前缀命中已经是 0 |
| 没有复用时只驱逐被报告的那一个 block | hash 掩码 `HHH-HHHH`，相同前缀命中 112→48 |

最后一行对得上 `FullAttentionManager.find_longest_cache_hit`：链式 hash，缺一块后面都不会命中。所以 full attention 上不必为了正确性把后缀 id 全删掉；删掉第一个仍匹配的无效块就停住命中。

## 建议的最小边界

不要合回 #61769/#62287 那套「记下原始 block id，报告来了就按 id 驱逐并走现有扫描」。复用之后这会删掉新请求的合法 hash，现有扫描还会按 id 把 `fail`/`recompute` 安到新请求上。#61905 描述的就是这件事，这次在 `418af607` 上又量到了。

也不要为了这个缺口把 `defer_block_free` 扩到所有 producer。#62438 那样做已被更窄的 #65060 取代，而且会让纯 producer 多占 block。

如果还要修当前这条 block 级协议，最小改动是：

1. owner 离开 `running`、同步加载的输出还没回来时，记下 `(block_id, block_hash)`。报告里只有 id，事后已经无法知道旧 hash。
2. 迟到报告只驱逐「当前 `block_hash` 仍等于记下的 hash」的块。hash 已经换成别人的，就不动。full attention 上驱逐被报告的那一块就够把命中从 112 打回 48。
3. 不要因为物理 id 现在落在另一个请求上，就对那个请求做 `fail` 或 `recompute`。这和第 2 步必须一起做，否则现有扫描仍会误伤。

这比 generation 计数更贴切：相同前缀被重新挂上 hash 时，KV 仍可能是没加载到的内容，hash 相等就该驱逐。generation 变了就跳过，反而会把这个坏命中留住。generation 账本这次不推荐。

`ref_cnt==0` 在这次「`schedule()` 已经返回」的窗口里和 hash 检查结论一致（复用并完成 `cache_blocks` 后 ref>0 且 hash 已变）。它覆盖不了「拿着 block 但还没改 hash」的路径，所以不能拿它代替 hash。同步 `allocate_slots` 会在 `schedule()` 返回前 `cache_blocks`，这个窗口在 `update_from_output` 里看不见。异步加载本来就不在调度时写 hash。

明确不做进这一步的：preempt 后把 `fail` 收成 `FINISHED_ERROR`、丢掉 in-flight sample、readmission drain。那是 #62287 的产品行为，不是缓存身份问题。

## 回归

已有：`probe_orphan_invalid_blocks.py`（报告先于 abort / abort 先于报告 / preempt / 压力 / reset）。

新增：`probe_block_reuse_aba.py`。以后的补丁至少要保住：

- 没有复用时，只清仍匹配的被报告块，相同前缀命中 48，更早的块仍是 `H`。
- 复用并改写 hash 之后，迟到报告不把新请求打成 `FINISHED_ERROR` 或把 computed 收到 64，新前缀命中仍是 112。
- 按原始 id 全量 `evict_blocks` 不能当正确实现（它把新前缀命中打成 0）。
- 成功加载再 abort 的命中仍是 112（今早的对照）。

## 还要人决定的

1. orozery 说不要再投资这条路径，#42461 还开着要删掉它。现在做 hash 检查，是逆着这句意见的小补丁。
2. 误伤新请求（#61905）要不要和缓存驱逐绑在同一次改动里。只驱逐、不改扫描，复用场景仍然会 `fail`/rewind 新请求。
3. preempt 的 `fail` 是否必须变成 `FINISHED_ERROR`，以及 producer 以外是否丢 sample。这次不替你选。

## 建议

**先不要写生产修复。** 缺口还在，但关闭原因是「这条协议不想继续加功能」，不是补丁被证明错了；而「只留原始 id」本身也不安全。若明确要逆着 #42461 / orozery 的意见补当前协议，范围就停在上面的 hash 相等驱逐，并且不要按 id 改无关请求。在那之前不动代码。
