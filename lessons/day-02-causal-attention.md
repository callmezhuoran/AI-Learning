# D2：手写单头 causal attention

状态：已布置，待学习者实现、实验和反馈。尚无 D2 完成或实测证据。

最后整理：2026-09-10。

## 今日目标与时间预算

解释每一步的张量形状，并用实验证明“当前 token 看不到未来”。这是后续理解 KV cache 和 FlashAttention 的基础。

| 时间 | 任务 |
|---|---|
| 20 分钟 | 理解计算过程，手写各步形状 |
| 45 分钟 | 用 PyTorch 实现 attention |
| 35 分钟 | 检查数值正确性与因果性 |
| 20 分钟 | 整理代码、结果和解释 |

本次使用 CPU、FP32，模型部分只保留三个投影和 attention，不需要先做性能优化。

## 信息在哪一步跨 token 流动

设输入 X 为 `[2, 4, 8]`：两个样本，每个样本四个 token，每个 token 八个特征。三个不同的线性投影得到：

```text
Q = X @ Wq     [2, 4, 8]
K = X @ Wk     [2, 4, 8]
V = X @ Wv     [2, 4, 8]
```

Q 表示当前位置用什么特征匹配其他位置，K 提供各位置可供匹配的特征，V 是匹配后实际汇总的信息。这是帮助理解的解释，实际含义由模型训练形成。

```text
scores = Q @ K.transpose(-2, -1) / sqrt(8)   [2, 4, 4]
probs  = softmax(scores + causal_mask)      [2, 4, 4]
output = probs @ V                         [2, 4, 8]
```

`scores[b, i, j]` 表示第 b 个样本中，位置 i 的 query 与位置 j 的 key 的匹配分数。`probs @ V` 让每个位置按权重汇总其他位置的信息。

causal mask 加在分数上，允许关注自己与过去：

```text
        被关注的位置 j
          0     1     2     3
位置 0    0    -∞    -∞    -∞
位置 1    0     0    -∞    -∞
位置 2    0     0     0    -∞
位置 3    0     0     0     0
```

0 表示保留原分数，负无穷使对应 softmax 概率为零。

## 实现题：补全四步

权重直接按“输入维 × 输出维”存储，因此这里写作 `X @ W`，区别于 D1 中展示的 `nn.Linear` 权重布局。

以下代码保留 TODO，需由学习者自己实现。不要将脚手架视为已完成的实验。

```python
import math
import torch
import torch.nn.functional as F

torch.manual_seed(42)

B, T, D = 2, 4, 8
x = torch.randn(B, T, D)

wq = torch.randn(D, D) / math.sqrt(D)
wk = torch.randn(D, D) / math.sqrt(D)
wv = torch.randn(D, D) / math.sqrt(D)

q, k, v = x @ wq, x @ wk, x @ wv


def attention(q, k, v):
    # TODO 1：点积并缩放
    scores = ...

    # TODO 2：将严格上三角，也就是未来位置，设为 -inf
    # 可使用 torch.triu 和 masked_fill
    masked_scores = ...

    # TODO 3：在正确的维度上做 softmax
    probs = ...

    # TODO 4：按概率汇总 V
    output = ...

    return output, probs


output, probs = attention(q, k, v)
print("output shape:", output.shape)
print("第一个样本的注意力概率：\n", probs[0])
```

## 实验验收

逐项验证，不只检查输出形状：

1. 输出形状为 `[2, 4, 8]`。
2. probs 每行之和约为 1，严格上三角为零。
3. 第一个 token 的输出应等于它自己的 V，因为它只能关注自己。
4. 保持权重不变，只修改 `x[:, 3, :]`，重新计算 Q/K/V 和输出；前三个位置的输出应保持不变。
5. 与下面的 PyTorch 官方实现对照，保存最大绝对误差及断言结果。

参考验证代码中的 `unsqueeze(1)` 添加大小为 1 的 head 维度；关闭 dropout，启用 causal mask。[官方接口说明](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

```python
reference = F.scaled_dot_product_attention(
    q.unsqueeze(1),
    k.unsqueeze(1),
    v.unsqueeze(1),
    dropout_p=0.0,
    is_causal=True,
).squeeze(1)

print("最大绝对误差：", (output - reference).abs().max().item())
torch.testing.assert_close(output, reference, rtol=1e-5, atol=1e-6)
```

如果断言失败，保留结果并定位是形状、mask、softmax 维度还是数值问题，不直接修改阈值让它通过。

## 需要自己的解释

1. softmax 为什么沿最后一维计算？它在给哪些对象分配权重？
2. 如果先 softmax，再把未来位置的概率置零，为什么不等价？可以用“一行两个分数都是 0，但第二个位置不可见”手算。

## 提交反馈

```text
实际投入时间：
attention 函数／代码位置：
probs[0]：
各项验收结果：
最大绝对误差及参考断言结果：
两个问题的解释：
卡住的地方：
```

下一步候选是将 attention 改成逐 token 执行，研究哪些计算可以缓存。是否进入 D3 取决于本次反馈，不按日期自动推进。
