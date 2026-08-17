# 钩子公式库（动态版）
> 由 douyin_trend_analyzer 于 2026-08-16 18:36 自动更新
> Agent 在 `script_generation` 阶段从这里选取钩子公式。

## 当前热门钩子公式

### Tier 1 — 高完播率（>60%）

```yaml
- hook_type: shock
  template: "[无文字，纯画面冲击3秒]"
  example: "[富士山日出金色时刻航拍]"
  effectiveness: 0.72
  tier: Tier 1

- hook_type: suspense
  template: "去了{N}次{destination}才知道的{M}件事"
  example: "去了5次富士山才知道的3个隐藏机位"
  effectiveness: 0.68
  tier: Tier 1

- hook_type: contrast
  template: "以为{destination}很{negative}，结果..."
  example: "以为京都很无聊，结果不想走了"
  effectiveness: 0.63
  tier: Tier 1

- hook_type: number
  template: "{N}天{M}元玩转{destination}"
  example: "3天800元玩转东京"
  effectiveness: 0.55
  tier: Tier 2

- hook_type: resonance
  template: "如果只剩一天在{destination}..."
  example: "如果只剩一天在京都..."
  effectiveness: 0.52
  tier: Tier 2

```

## 更新日志

| 日期 | 更新内容 | 数据来源 |
|---|---|---|
| 2026-08-16 | 自动更新（5个钩子公式） | 抖音趋势分析 |
